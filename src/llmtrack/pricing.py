"""Model token pricing registry and cost calculation for llmtrack."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("llmtrack.pricing")

# Prices in USD per 1M tokens: (input_per_1M, output_per_1M)
# Verified reference pricing for major models (2025/2026 rates).
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1-mini": (1.10, 4.40),
    "o1-preview": (15.00, 60.00),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Google Gemini
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    # DeepSeek
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
}


class PricingTable:
    """Lookup table for model pricing with longest-prefix matching and override support."""

    def __init__(self, overrides: Optional[dict[str, tuple[float, float]]] = None) -> None:
        self._pricing: dict[str, tuple[float, float]] = dict(DEFAULT_PRICING)
        if overrides:
            self._pricing.update(overrides)

        # Sort prefixes by length descending so longest prefix matches first
        self._sorted_prefixes = sorted(self._pricing.keys(), key=len, reverse=True)
        self._warned_models: set[str] = set()

    def get_pricing(self, model: str) -> Optional[tuple[float, float]]:
        """Get (input_per_1M, output_per_1M) for model via prefix matching."""
        if not model:
            return None

        # Exact match check first
        if model in self._pricing:
            return self._pricing[model]

        # Case-insensitive / prefix matching
        model_lower = model.lower()
        for prefix in self._sorted_prefixes:
            if model_lower.startswith(prefix.lower()):
                return self._pricing[prefix]

        return None

    def calculate_cost(
        self,
        model: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> Optional[float]:
        """Compute USD cost based on tokens.

        Returns None if model is unknown or tokens are None.
        Issues a one-time warning if model is unknown.
        """
        if not model:
            return None

        rates = self.get_pricing(model)
        if rates is None:
            if model not in self._warned_models:
                self._warned_models.add(model)
                logger.warning(
                    "llmtrack: Unknown model '%s'. Cost will be logged as NULL. "
                    "Configure pricing via Tracker(pricing={'%s': (in_usd, out_usd)}).",
                    model,
                    model,
                )
            return None

        in_rate, out_rate = rates
        inp = input_tokens or 0
        out = output_tokens or 0

        # Cost = (tokens / 1,000,000) * rate_per_1M
        cost = (inp / 1_000_000.0 * in_rate) + (out / 1_000_000.0 * out_rate)
        return round(cost, 6)
