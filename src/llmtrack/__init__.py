"""llmtrack: Lightweight, zero-dependency LLM call tracking and rate-limit backoff."""

from .storage import CallRecord, Storage
from .tracker import BudgetExceededError, Span, Tracker, track

__all__ = [
    "track",
    "Tracker",
    "Span",
    "CallRecord",
    "Storage",
    "BudgetExceededError",
]

__version__ = "0.1.0"
