"""Tests for Tracker decorator, span context manager, retry loops, and async support."""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llmtrack.tracker import BudgetExceededError, Tracker


class RateLimit429(Exception):
    status_code = 429


class BadRequest400(Exception):
    status_code = 400


@pytest.fixture
def temp_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tracker_test.db"
        tracker = Tracker(db_path=db_path)
        yield tracker
        tracker.close()


def test_successful_call_logging(temp_tracker):
    @temp_tracker.track(provider="anthropic", tag="test-success")
    def fake_anthropic_call(prompt: str):
        return SimpleNamespace(
            model="claude-3-5-sonnet",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
        )

    res = fake_anthropic_call("hello")
    assert res.model == "claude-3-5-sonnet"

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "anthropic"
    assert row["model"] == "claude-3-5-sonnet"
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 500
    assert row["total_tokens"] == 1500
    assert row["attempts"] == 1
    assert row["retries"] == 0
    assert row["status"] == "ok"
    assert row["cost_usd"] > 0
    assert row["tag"] == "test-success"


@patch("time.sleep", return_value=None)
def test_retry_then_success(mock_sleep, temp_tracker):
    call_count = 0

    @temp_tracker.track(provider="openai", tag="retry-test", max_retries=3, base=0.01)
    def fake_failing_then_succeeding():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimit429("Rate limit hit")
        return SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )

    res = fake_failing_then_succeeding()
    assert res.model == "gpt-4o"
    assert call_count == 3
    assert mock_sleep.call_count == 2

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["attempts"] == 3
    assert row["retries"] == 2
    assert row["status"] == "ok"


@patch("time.sleep", return_value=None)
def test_non_retryable_failure_raises_immediately(mock_sleep, temp_tracker):
    call_count = 0

    @temp_tracker.track(provider="openai", tag="bad-request", max_retries=3)
    def fake_bad_request():
        nonlocal call_count
        call_count += 1
        raise BadRequest400("Invalid parameter")

    with pytest.raises(BadRequest400):
        fake_bad_request()

    assert call_count == 1
    assert mock_sleep.call_count == 0

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["attempts"] == 1
    assert row["retries"] == 0
    assert row["status"] == "error"
    assert row["error_type"] == "BadRequest400"
    assert "Invalid parameter" in row["error_message"]


@patch("time.sleep", return_value=None)
def test_final_retry_failure(mock_sleep, temp_tracker):
    call_count = 0

    @temp_tracker.track(provider="openai", tag="persistent-fail", max_retries=2, base=0.01)
    def persistent_failure():
        nonlocal call_count
        call_count += 1
        raise RateLimit429("Always 429")

    with pytest.raises(RateLimit429):
        persistent_failure()

    # 1 initial + 2 retries = 3 attempts total
    assert call_count == 3
    assert mock_sleep.call_count == 2

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["attempts"] == 3
    assert row["retries"] == 2
    assert row["status"] == "error"
    assert row["error_type"] == "RateLimit429"


def test_unknown_model_pricing(temp_tracker):
    @temp_tracker.track(provider="custom")
    def call_unknown():
        return SimpleNamespace(
            model="unknown-future-model",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )

    call_unknown()
    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    assert rows[0]["model"] == "unknown-future-model"
    assert rows[0]["cost_usd"] is None


def test_async_tracking_and_retry(temp_tracker):
    async def run_test():
        call_count = 0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_async_sleep:

            @temp_tracker.track(provider="openai", max_retries=2, base=0.01)
            async def fake_async_call():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimit429("Async rate limit")
                return SimpleNamespace(
                    model="gpt-4o-mini",
                    usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
                )

            res = await fake_async_call()
            assert res.model == "gpt-4o-mini"
            assert call_count == 2
            assert mock_async_sleep.call_count == 1

        rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
        assert len(rows) == 1
        row = rows[0]
        assert row["model"] == "gpt-4o-mini"
        assert row["attempts"] == 2
        assert row["retries"] == 1
        assert row["status"] == "ok"

    asyncio.run(run_test())


def test_broken_db_does_not_crash_call():
    broken_tracker = Tracker(db_path="Z:/invalid_path/cant_write.db")

    @broken_tracker.track(provider="anthropic")
    def call():
        return {"model": "claude-3-5-sonnet", "usage": {"input_tokens": 10, "output_tokens": 10}}

    # The function call must still return successfully!
    res = call()
    assert res["model"] == "claude-3-5-sonnet"


def test_span_context_manager(temp_tracker):
    with temp_tracker.span(provider="openai", tag="span-test") as span:
        resp = SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )
        span.record(resp)

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["total_tokens"] == 150
    assert rows[0]["status"] == "ok"
    assert rows[0]["tag"] == "span-test"


def test_span_exception_tracking(temp_tracker):
    with pytest.raises(RuntimeError):
        with temp_tracker.span(provider="anthropic", model="claude-3-5-haiku"):
            raise RuntimeError("Something failed inside span")

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_type"] == "RuntimeError"
    assert "Something failed inside span" in rows[0]["error_message"]


def test_sync_streaming_wrapper(temp_tracker):
    @temp_tracker.track(provider="openai", model="gpt-4o", stream=True)
    def fake_stream():
        yield SimpleNamespace(
            model="gpt-4o",
            usage=None,
        )
        yield SimpleNamespace(
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=60),
        )

    chunks = list(fake_stream())
    assert len(chunks) == 2

    rows = temp_tracker.storage.query("SELECT * FROM llm_calls")
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["input_tokens"] == 40
    assert rows[0]["output_tokens"] == 60
    assert rows[0]["status"] == "ok"


def test_budget_guard(temp_tracker):
    temp_tracker.max_daily_usd = 0.001

    # Insert an existing call that costs $0.005
    @temp_tracker.track(provider="anthropic")
    def call1():
        return SimpleNamespace(
            model="claude-3-5-sonnet",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=1000),
        )

    # First call will run and exceed 0.001
    call1()

    # Second call should be blocked by budget check
    with pytest.raises(BudgetExceededError):
        call1()
