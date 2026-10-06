"""Demo simulation script for llmtrack.

Generates realistic simulated LLM traffic (~500 calls) with:
- Multiple providers & models (OpenAI, Anthropic, DeepSeek)
- Realistic token usages and latency distributions
- Simulated rate limits (429) with exponential backoff & retries
- Occasional failures and errors
- Feature tags for cost-attribution
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from llmtrack import Tracker
from llmtrack.storage import CallRecord

DB_FILE = "llm_calls.db"

MODELS = [
    ("openai", "gpt-4o", (200, 2000), (50, 800), (300, 1800)),
    ("openai", "gpt-4o-mini", (100, 1500), (50, 400), (120, 600)),
    ("anthropic", "claude-3-5-sonnet", (300, 3000), (100, 1200), (450, 2200)),
    ("anthropic", "claude-3-5-haiku", (100, 1200), (30, 300), (150, 700)),
    ("openai", "o3-mini", (500, 2500), (200, 1500), (600, 3500)),
    ("deepseek", "deepseek-chat", (150, 1800), (40, 500), (200, 1000)),
]

TAGS = [
    "summarizer",
    "customer-support",
    "code-eval",
    "rag-retrieval",
    "agent-planner",
    "translation",
]


def run_simulation(num_calls: int = 500) -> None:
    print(f"[*] Initializing llmtrack demo with {num_calls} simulated calls...", flush=True)
    tracker = Tracker(db_path=DB_FILE)

    now = datetime.now(timezone.utc)
    base_time = now - timedelta(days=6)

    records: list[CallRecord] = []
    total_retried = 0
    total_errors = 0

    print("[*] Generating simulated LLM calls with realistic patterns...", flush=True)

    for i in range(num_calls):
        provider, model, (min_in, max_in), (min_out, max_out), (min_lat, max_lat) = random.choice(
            MODELS
        )
        tag = random.choice(TAGS)

        # Spread timestamps across past 7 days
        day_offset = random.uniform(0, 6)
        call_time = (base_time + timedelta(days=day_offset)).isoformat()

        base_latency = random.uniform(min_lat, max_lat)
        if random.random() < 0.05:
            base_latency *= random.uniform(2.0, 4.0)

        in_tok = random.randint(min_in, max_in)
        out_tok = random.randint(min_out, max_out)

        roll = random.random()

        if roll < 0.88:
            # Clean success
            cost = tracker.pricing.calculate_cost(model, in_tok, out_tok)
            records.append(
                CallRecord(
                    provider=provider,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=in_tok + out_tok,
                    cost_usd=cost,
                    latency_ms=round(base_latency, 1),
                    attempts=1,
                    retries=0,
                    status="ok",
                    tag=tag,
                    request_id=f"req_{i:05d}",
                    ts_utc=call_time,
                )
            )

        elif roll < 0.96:
            # Retried and recovered
            retries = random.choice([1, 2, 3])
            total_retried += 1
            cost = tracker.pricing.calculate_cost(model, in_tok, out_tok)
            records.append(
                CallRecord(
                    provider=provider,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=in_tok + out_tok,
                    cost_usd=cost,
                    latency_ms=round(base_latency + (retries * 800), 1),
                    attempts=retries + 1,
                    retries=retries,
                    status="ok",
                    tag=tag,
                    request_id=f"req_{i:05d}",
                    ts_utc=call_time,
                )
            )

        elif roll < 0.99:
            # Rate limit failure
            total_errors += 1
            records.append(
                CallRecord(
                    provider=provider,
                    model=model,
                    latency_ms=round(base_latency + 2500, 1),
                    attempts=4,
                    retries=3,
                    status="error",
                    error_type="RateLimitError",
                    error_message="HTTP 429: Rate limit exceeded. Please back off.",
                    tag=tag,
                    request_id=f"req_{i:05d}",
                    ts_utc=call_time,
                )
            )

        else:
            # Client bad request
            total_errors += 1
            records.append(
                CallRecord(
                    provider=provider,
                    model=model,
                    latency_ms=round(random.uniform(50, 120), 1),
                    attempts=1,
                    retries=0,
                    status="error",
                    error_type="BadRequestError",
                    error_message="HTTP 400: Invalid context window or max_tokens exceeded.",
                    tag=tag,
                    request_id=f"req_{i:05d}",
                    ts_utc=call_time,
                )
            )

    tracker.storage.insert_many(records)
    tracker.close()

    print(f"[+] Successfully simulated {len(records)} calls into '{DB_FILE}'!", flush=True)
    print(f"    - Clean calls: {len(records) - total_retried - total_errors}", flush=True)
    print(f"    - Retried & recovered calls: {total_retried}", flush=True)
    print(f"    - Logged errors: {total_errors}", flush=True)
    print("\nNext step: Run the analytics report CLI:", flush=True)
    print("    python -m llmtrack.cli report", flush=True)
    print("    python -m llmtrack.cli report --since 7d", flush=True)
    print("    python -m llmtrack.cli report --format json", flush=True)


if __name__ == "__main__":
    run_simulation(500)
