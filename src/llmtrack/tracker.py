"""Core Tracker implementation, decorator, context manager, and span tracking."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Generator,
    Iterator,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from .backoff import (
    BackoffConfig,
    compute_backoff_delay,
    get_retry_after,
    is_retryable,
)
from .extractors import ExtractorCallable, extract_usage
from .pricing import PricingTable
from .storage import CallRecord, Storage

logger = logging.getLogger("llmtrack.tracker")

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# Context var for tracking active span
_current_span: ContextVar[Optional["Span"]] = ContextVar("_current_span", default=None)


class BudgetExceededError(Exception):
    """Raised when the daily budget limit is exceeded."""


class Span:
    """Context manager for tracking an LLM call block."""

    def __init__(
        self,
        tracker: "Tracker",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        request_id: Optional[str] = None,
        extractor: Optional[ExtractorCallable] = None,
    ) -> None:
        self.tracker = tracker
        self.provider = provider
        self.model = model
        self.tag = tag
        self.request_id = request_id
        self.extractor = extractor

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self.cost_usd: Optional[float] = None
        self.status: str = "ok"
        self.error_type: Optional[str] = None
        self.error_message: Optional[str] = None
        self.attempts: int = 1
        self.retries: int = 0
        self._recorded: bool = False
        self._token: Any = None

    def __enter__(self) -> "Span":
        self.tracker._check_budget()
        self.start_time = time.perf_counter()
        self._token = _current_span.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        self.end_time = time.perf_counter()
        if exc_val is not None:
            self.status = "error"
            self.error_type = exc_type.__name__ if exc_type else "Error"
            self.error_message = str(exc_val)

        self._flush()
        if self._token is not None:
            _current_span.reset(self._token)
        return False  # Do not suppress exceptions

    async def __aenter__(self) -> "Span":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    def record(
        self,
        response: Any = None,
        model: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        tag: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Record usage information from a response object or explicit parameters."""
        if tag is not None:
            self.tag = tag
        if request_id is not None:
            self.request_id = request_id

        if response is not None:
            extracted_model, in_tok, out_tok = extract_usage(
                response, provider=self.provider, custom_extractor=self.extractor
            )
            if self.model is None:
                self.model = extracted_model or model
            if input_tokens is None:
                input_tokens = in_tok
            if output_tokens is None:
                output_tokens = out_tok

        if model is not None:
            self.model = model
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        if total_tokens is not None:
            self.total_tokens = total_tokens
        elif self.input_tokens is not None or self.output_tokens is not None:
            self.total_tokens = (self.input_tokens or 0) + (self.output_tokens or 0)

        if cost_usd is not None:
            self.cost_usd = cost_usd
        elif self.model and (self.input_tokens is not None or self.output_tokens is not None):
            self.cost_usd = self.tracker.pricing.calculate_cost(
                self.model, self.input_tokens, self.output_tokens
            )

        self._recorded = True

    def _flush(self) -> None:
        latency_ms = (self.end_time - self.start_time) * 1000.0 if self.start_time else 0.0
        # If cost has not been computed yet and we have model/tokens
        if (
            self.cost_usd is None
            and self.model
            and (self.input_tokens is not None or self.output_tokens is not None)
        ):
            self.cost_usd = self.tracker.pricing.calculate_cost(
                self.model, self.input_tokens, self.output_tokens
            )

        record = CallRecord(
            provider=self.provider,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cost_usd=self.cost_usd,
            latency_ms=round(latency_ms, 2),
            attempts=self.attempts,
            retries=self.retries,
            status=self.status,
            error_type=self.error_type,
            error_message=self.error_message,
            tag=self.tag,
            request_id=self.request_id,
        )
        self.tracker.storage.insert(record)


class Tracker:
    """Central tracker coordinating SQLite persistence, pricing calculation, and retry backoff."""

    def __init__(
        self,
        db_path: str | Path = "llm_calls.db",
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        max_daily_usd: Optional[float] = None,
    ) -> None:
        self.db_path = db_path
        self.storage = Storage(db_path=db_path)
        self.pricing = PricingTable(overrides=pricing)
        self.max_daily_usd = max_daily_usd

    def close(self) -> None:
        """Close database connections."""
        self.storage.close()

    def _check_budget(self) -> None:
        """Verify daily spend if max_daily_usd is configured."""
        if self.max_daily_usd is None:
            return
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sql = (
            "SELECT SUM(cost_usd) AS daily_cost FROM llm_calls "
            "WHERE date(ts_utc) = ? AND status = 'ok'"
        )
        rows = self.storage.query(sql, (today_utc,))
        current = rows[0]["daily_cost"] if rows and rows[0]["daily_cost"] is not None else 0.0
        if current >= self.max_daily_usd:
            raise BudgetExceededError(
                f"Daily budget limit of ${self.max_daily_usd:.2f} exceeded. "
                f"Current spend: ${current:.4f}"
            )

    def span(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        request_id: Optional[str] = None,
        extractor: Optional[ExtractorCallable] = None,
    ) -> Span:
        """Create a context manager Span for manually tracking a block of code."""
        return Span(
            tracker=self,
            provider=provider,
            model=model,
            tag=tag,
            request_id=request_id,
            extractor=extractor,
        )

    def track(
        self,
        _fn: Optional[F] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tag: Optional[str] = None,
        request_id: Optional[str] = None,
        max_retries: int = 5,
        base: float = 1.0,
        cap: float = 60.0,
        retry_on: Optional[Tuple[Type[BaseException], ...]] = None,
        extractor: Optional[ExtractorCallable] = None,
        stream: bool = False,
    ) -> Any:
        """Decorator to wrap sync or async LLM API calls with tracking and retry backoff."""

        def decorator(fn: F) -> F:
            backoff_cfg = BackoffConfig(
                max_retries=max_retries,
                base=base,
                cap=cap,
                retry_on=retry_on,
            )

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    self._check_budget()
                    start_time = time.perf_counter()
                    attempts = 0
                    retries = 0

                    for attempt in range(backoff_cfg.max_retries + 1):
                        attempts += 1
                        try:
                            result = await fn(*args, **kwargs)
                            if stream or inspect.isasyncgen(result):
                                return self._wrap_async_stream(
                                    result,
                                    start_time=start_time,
                                    attempts=attempts,
                                    retries=retries,
                                    provider=provider,
                                    model=model,
                                    tag=tag,
                                    request_id=request_id,
                                    extractor=extractor,
                                )

                            latency_ms = (time.perf_counter() - start_time) * 1000.0
                            ext_model, in_tok, out_tok = extract_usage(
                                result, provider=provider, custom_extractor=extractor
                            )
                            resolved_model = model or ext_model
                            cost = self.pricing.calculate_cost(resolved_model, in_tok, out_tok)
                            tot_tok = (
                                (in_tok or 0) + (out_tok or 0)
                                if (in_tok is not None or out_tok is not None)
                                else None
                            )

                            self.storage.insert(
                                CallRecord(
                                    provider=provider,
                                    model=resolved_model,
                                    input_tokens=in_tok,
                                    output_tokens=out_tok,
                                    total_tokens=tot_tok,
                                    cost_usd=cost,
                                    latency_ms=round(latency_ms, 2),
                                    attempts=attempts,
                                    retries=retries,
                                    status="ok",
                                    tag=tag,
                                    request_id=request_id,
                                )
                            )
                            return result

                        except Exception as exc:
                            if attempt < backoff_cfg.max_retries and is_retryable(
                                exc, backoff_cfg.retry_on
                            ):
                                retries += 1
                                delay = compute_backoff_delay(
                                    attempt,
                                    base=backoff_cfg.base,
                                    cap=backoff_cfg.cap,
                                    retry_after=get_retry_after(exc),
                                )
                                logger.info(
                                    "llmtrack: Retryable %s. Retrying in %.2fs (attempt %d/%d)",
                                    type(exc).__name__,
                                    delay,
                                    attempts,
                                    backoff_cfg.max_retries,
                                )
                                await asyncio.sleep(delay)
                                continue

                            # Final failure
                            latency_ms = (time.perf_counter() - start_time) * 1000.0
                            self.storage.insert(
                                CallRecord(
                                    provider=provider,
                                    model=model,
                                    latency_ms=round(latency_ms, 2),
                                    attempts=attempts,
                                    retries=retries,
                                    status="error",
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                    tag=tag,
                                    request_id=request_id,
                                )
                            )
                            raise

                return cast(F, async_wrapper)

            else:

                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    self._check_budget()
                    start_time = time.perf_counter()
                    attempts = 0
                    retries = 0

                    for attempt in range(backoff_cfg.max_retries + 1):
                        attempts += 1
                        try:
                            result = fn(*args, **kwargs)
                            if stream or inspect.isgenerator(result):
                                return self._wrap_sync_stream(
                                    result,
                                    start_time=start_time,
                                    attempts=attempts,
                                    retries=retries,
                                    provider=provider,
                                    model=model,
                                    tag=tag,
                                    request_id=request_id,
                                    extractor=extractor,
                                )

                            latency_ms = (time.perf_counter() - start_time) * 1000.0
                            ext_model, in_tok, out_tok = extract_usage(
                                result, provider=provider, custom_extractor=extractor
                            )
                            resolved_model = model or ext_model
                            cost = self.pricing.calculate_cost(resolved_model, in_tok, out_tok)
                            tot_tok = (
                                (in_tok or 0) + (out_tok or 0)
                                if (in_tok is not None or out_tok is not None)
                                else None
                            )

                            self.storage.insert(
                                CallRecord(
                                    provider=provider,
                                    model=resolved_model,
                                    input_tokens=in_tok,
                                    output_tokens=out_tok,
                                    total_tokens=tot_tok,
                                    cost_usd=cost,
                                    latency_ms=round(latency_ms, 2),
                                    attempts=attempts,
                                    retries=retries,
                                    status="ok",
                                    tag=tag,
                                    request_id=request_id,
                                )
                            )
                            return result

                        except Exception as exc:
                            if attempt < backoff_cfg.max_retries and is_retryable(
                                exc, backoff_cfg.retry_on
                            ):
                                retries += 1
                                delay = compute_backoff_delay(
                                    attempt,
                                    base=backoff_cfg.base,
                                    cap=backoff_cfg.cap,
                                    retry_after=get_retry_after(exc),
                                )
                                logger.info(
                                    "llmtrack: Retryable %s. Retrying in %.2fs (attempt %d/%d)",
                                    type(exc).__name__,
                                    delay,
                                    attempts,
                                    backoff_cfg.max_retries,
                                )
                                time.sleep(delay)
                                continue

                            # Final failure
                            latency_ms = (time.perf_counter() - start_time) * 1000.0
                            self.storage.insert(
                                CallRecord(
                                    provider=provider,
                                    model=model,
                                    latency_ms=round(latency_ms, 2),
                                    attempts=attempts,
                                    retries=retries,
                                    status="error",
                                    error_type=type(exc).__name__,
                                    error_message=str(exc),
                                    tag=tag,
                                    request_id=request_id,
                                )
                            )
                            raise

                return cast(F, sync_wrapper)

        if _fn is not None:
            return decorator(_fn)
        return decorator

    def _wrap_sync_stream(
        self,
        generator_or_iter: Iterator[Any],
        start_time: float,
        attempts: int,
        retries: int,
        provider: Optional[str],
        model: Optional[str],
        tag: Optional[str],
        request_id: Optional[str],
        extractor: Optional[ExtractorCallable],
    ) -> Generator[Any, None, None]:
        """Wrap a synchronous stream iterator to log metrics upon completion."""
        last_item: Any = None
        error_occurred = False
        try:
            for item in generator_or_iter:
                last_item = item
                yield item
        except Exception as exc:
            error_occurred = True
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.storage.insert(
                CallRecord(
                    provider=provider,
                    model=model,
                    latency_ms=round(latency_ms, 2),
                    attempts=attempts,
                    retries=retries,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    tag=tag,
                    request_id=request_id,
                )
            )
            raise
        finally:
            if not error_occurred:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                ext_model, in_tok, out_tok = extract_usage(
                    last_item, provider=provider, custom_extractor=extractor
                )
                resolved_model = model or ext_model
                cost = self.pricing.calculate_cost(resolved_model, in_tok, out_tok)
                tot_tok = (
                    (in_tok or 0) + (out_tok or 0)
                    if (in_tok is not None or out_tok is not None)
                    else None
                )
                self.storage.insert(
                    CallRecord(
                        provider=provider,
                        model=resolved_model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        total_tokens=tot_tok,
                        cost_usd=cost,
                        latency_ms=round(latency_ms, 2),
                        attempts=attempts,
                        retries=retries,
                        status="ok",
                        tag=tag,
                        request_id=request_id,
                    )
                )

    async def _wrap_async_stream(
        self,
        generator_or_iter: AsyncIterator[Any],
        start_time: float,
        attempts: int,
        retries: int,
        provider: Optional[str],
        model: Optional[str],
        tag: Optional[str],
        request_id: Optional[str],
        extractor: Optional[ExtractorCallable],
    ) -> AsyncGenerator[Any, None]:
        """Wrap an asynchronous stream iterator to log metrics upon completion."""
        last_item: Any = None
        error_occurred = False
        try:
            async for item in generator_or_iter:
                last_item = item
                yield item
        except Exception as exc:
            error_occurred = True
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.storage.insert(
                CallRecord(
                    provider=provider,
                    model=model,
                    latency_ms=round(latency_ms, 2),
                    attempts=attempts,
                    retries=retries,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    tag=tag,
                    request_id=request_id,
                )
            )
            raise
        finally:
            if not error_occurred:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                ext_model, in_tok, out_tok = extract_usage(
                    last_item, provider=provider, custom_extractor=extractor
                )
                resolved_model = model or ext_model
                cost = self.pricing.calculate_cost(resolved_model, in_tok, out_tok)
                tot_tok = (
                    (in_tok or 0) + (out_tok or 0)
                    if (in_tok is not None or out_tok is not None)
                    else None
                )
                self.storage.insert(
                    CallRecord(
                        provider=provider,
                        model=resolved_model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        total_tokens=tot_tok,
                        cost_usd=cost,
                        latency_ms=round(latency_ms, 2),
                        attempts=attempts,
                        retries=retries,
                        status="ok",
                        tag=tag,
                        request_id=request_id,
                    )
                )


# Global default tracker instance (lazy initialized)
_default_tracker: Optional[Tracker] = None


def get_default_tracker() -> Tracker:
    """Get or lazily initialize the default global Tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = Tracker()
    return _default_tracker


def track(
    _fn: Optional[F] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tag: Optional[str] = None,
    request_id: Optional[str] = None,
    max_retries: int = 5,
    base: float = 1.0,
    cap: float = 60.0,
    retry_on: Optional[Tuple[Type[BaseException], ...]] = None,
    extractor: Optional[ExtractorCallable] = None,
    stream: bool = False,
) -> Any:
    """Module-level decorator using default Tracker instance."""
    return get_default_tracker().track(
        _fn,
        provider=provider,
        model=model,
        tag=tag,
        request_id=request_id,
        max_retries=max_retries,
        base=base,
        cap=cap,
        retry_on=retry_on,
        extractor=extractor,
        stream=stream,
    )
