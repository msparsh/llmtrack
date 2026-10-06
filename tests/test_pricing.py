"""Tests for pricing lookup, prefix matching, and cost computation."""

import logging

from llmtrack.pricing import PricingTable


def test_pricing_exact_and_prefix_match():
    pricing = PricingTable()
    # Exact or prefix match
    rates = pricing.get_pricing("claude-3-5-sonnet")
    assert rates == (3.00, 15.00)

    # Date-versioned suffix matches prefix
    rates_versioned = pricing.get_pricing("claude-3-5-sonnet-20241022")
    assert rates_versioned == (3.00, 15.00)

    # Distinguish gpt-4o vs gpt-4o-mini
    rates_mini = pricing.get_pricing("gpt-4o-mini-2024-07-18")
    assert rates_mini == (0.15, 0.60)

    rates_4o = pricing.get_pricing("gpt-4o-2024-08-06")
    assert rates_4o == (2.50, 10.00)


def test_calculate_cost():
    pricing = PricingTable()
    # gpt-4o: 2.50 / 1M input, 10.00 / 1M output
    # 1,000 in, 500 out -> (1000/1e6 * 2.50) + (500/1e6 * 10.00) = 0.0025 + 0.0050 = 0.0075
    cost = pricing.calculate_cost("gpt-4o", 1000, 500)
    assert cost == 0.0075


def test_unknown_model_returns_none_and_warns(caplog):
    pricing = PricingTable()
    with caplog.at_level(logging.WARNING):
        cost1 = pricing.calculate_cost("unknown-model-xyz", 100, 100)
        assert cost1 is None
        # Call again: should only warn once
        cost2 = pricing.calculate_cost("unknown-model-xyz", 200, 200)
        assert cost2 is None

    warning_records = [r for r in caplog.records if "unknown-model-xyz" in r.message]
    assert len(warning_records) == 1


def test_custom_pricing_overrides():
    pricing = PricingTable(overrides={"my-custom-model": (1.0, 2.0)})
    cost = pricing.calculate_cost("my-custom-model", 1_000_000, 1_000_000)
    assert cost == 3.0
