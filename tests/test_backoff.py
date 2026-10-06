"""Tests for retry classification, backoff calculation, and retry-after header extraction."""

from llmtrack.backoff import (
    compute_backoff_delay,
    get_retry_after,
    is_retryable,
)


class CustomRateLimitError(Exception):
    pass


class CustomBadRequestError(Exception):
    status_code = 400


class CustomHttp429Error(Exception):
    status_code = 429


class CustomHttp503Error(Exception):
    status_code = 503


def test_is_retryable():
    # Rate limit and server errors
    assert is_retryable(CustomRateLimitError("Rate limit exceeded")) is True
    assert is_retryable(CustomHttp429Error()) is True
    assert is_retryable(CustomHttp503Error()) is True
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(ConnectionError()) is True

    # Non-retryable client errors
    assert is_retryable(CustomBadRequestError()) is False
    assert is_retryable(ValueError("Invalid argument")) is False


def test_retry_after_extraction():
    # Attribute
    exc1 = Exception("Rate limited")
    setattr(exc1, "retry_after", 3.5)
    assert get_retry_after(exc1) == 3.5

    # Header in response
    class MockResponse:
        headers = {"retry-after": "5"}

    exc2 = Exception("Rate limited")
    setattr(exc2, "response", MockResponse())
    assert get_retry_after(exc2) == 5.0

    # From string pattern
    exc3 = Exception("Rate limit exceeded. Please retry after 2.5s.")
    assert get_retry_after(exc3) == 2.5


def test_backoff_delay_calculation():
    # Honor retry_after
    delay = compute_backoff_delay(attempt=1, base=1.0, cap=60.0, retry_after=4.2)
    assert delay == 4.2

    # Respect cap
    delay_capped = compute_backoff_delay(attempt=10, base=1.0, cap=15.0, retry_after=30.0)
    assert delay_capped == 15.0

    # Jitter range check: 0 <= delay <= min(cap, base * 2**attempt)
    for attempt in range(5):
        d = compute_backoff_delay(attempt=attempt, base=1.0, cap=60.0)
        assert 0.0 <= d <= min(60.0, 1.0 * (2**attempt))
