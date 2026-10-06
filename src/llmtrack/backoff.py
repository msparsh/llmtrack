"""Retry and exponential backoff logic for handling rate limits and transient errors."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type

logger = logging.getLogger("llmtrack.backoff")

# Standard exception types that are always retryable
RETRYABLE_STD_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
)

# Common names for transient / rate-limit exceptions across SDKs (OpenAI, Anthropic, HTTPX, urllib3)
RETRYABLE_NAME_KEYWORDS = (
    "ratelimit",
    "rate_limit",
    "timeout",
    "timedout",
    "connectionerror",
    "apiconnection",
    "servererror",
    "internalserver",
    "serviceunavailable",
    "overloaded",
    "badgateway",
    "gatewaytimeout",
)

NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 405, 422}


@dataclass
class BackoffConfig:
    """Configuration for retry backoff behavior."""

    max_retries: int = 5
    base: float = 1.0
    cap: float = 60.0
    retry_on: Optional[Tuple[Type[BaseException], ...]] = None


def get_status_code(exc: BaseException) -> Optional[int]:
    """Attempt to extract HTTP status code from an exception or its nested response."""
    for attr in ("status_code", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    # Check exc.response
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if isinstance(val, int):
                return val

    return None


def get_retry_after(exc: BaseException) -> Optional[float]:
    """Extract 'Retry-After' duration in seconds if present in headers or attributes."""
    # Attribute directly on exception
    val = getattr(exc, "retry_after", None)
    if isinstance(val, (int, float)) and val >= 0:
        return float(val)

    # Headers on response
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp else None
    if headers and hasattr(headers, "get"):
        header_val = headers.get("retry-after") or headers.get("Retry-After")
        if header_val is not None:
            try:
                # Could be integer seconds
                return max(0.0, float(header_val))
            except (ValueError, TypeError):
                pass

    # Check error message for patterns like "retry after 3.5 seconds"
    msg = str(exc)
    match = re.search(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            pass

    return None


def is_retryable(
    exc: BaseException,
    retry_on: Optional[Tuple[Type[BaseException], ...]] = None,
) -> bool:
    """Determine whether an exception should be retried.

    Retries on:
    - explicit matches in retry_on tuple
    - HTTP 429
    - HTTP 5xx (500, 502, 503, 504)
    - standard TimeoutError / ConnectionError
    - exception class names matching rate limits or transient errors

    Does NOT retry 4xx errors other than 429 (e.g. 400, 401, 403, 404).
    """
    if retry_on is not None and isinstance(exc, retry_on):
        return True

    status = get_status_code(exc)
    if status is not None:
        if status == 429:
            return True
        if 500 <= status < 600:
            return True
        if status in NON_RETRYABLE_STATUS_CODES:
            return False

    if isinstance(exc, RETRYABLE_STD_EXCEPTIONS):
        return True

    exc_name = type(exc).__name__.lower()
    for kw in RETRYABLE_NAME_KEYWORDS:
        if kw in exc_name:
            return True

    return False


def compute_backoff_delay(
    attempt: int,
    base: float = 1.0,
    cap: float = 60.0,
    retry_after: Optional[float] = None,
    rng: Optional[Any] = None,
) -> float:
    """Compute backoff delay with full jitter.

    Formula: sleep = random(0, min(cap, base * 2**attempt)).
    If retry_after is provided, it is honored up to `cap`.
    """
    if retry_after is not None and retry_after > 0:
        return min(cap, retry_after)

    upper = min(cap, base * (2**attempt))
    random_fn = rng.uniform if rng is not None else random.uniform
    return random_fn(0.0, upper)
