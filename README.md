<div align="center">

<img src="assets/banner.svg" alt="llmtrack banner" width="100%" />

<p align="center">
  <strong>Zero-dependency LLM call tracking, token cost attribution & rate-limit backoff.</strong>
</p>

<p align="center">
  <a href="https://github.com/msparsh/llmtrack/actions"><img src="https://img.shields.io/badge/CI-passing-34d399?style=flat-square&logo=githubactions&logoColor=white" alt="CI" /></a>
  <a href="https://pypi.org/project/llmtrack"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.14-blue?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://sqlite.org/"><img src="https://img.shields.io/badge/Storage-SQLite_WAL-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License" /></a>
</p>

</div>

---

A small, lightweight library that wraps any LLM API call with a decorator or context manager. It logs **tokens**, **USD cost**, **latency**, **retries**, and **errors** to a local SQLite database with automatic rate-limit backoff, analytics SQL queries, and a CLI report.

- **Zero dependencies**: Pure Python standard library (`sqlite3`, `time`, `functools`, `random`).
- **Resilient**: Automatic exponential backoff with full jitter for 429s, 5xx, and timeouts.
- **Fail-safe**: Logging failures never break your LLM calls.
- **Sync & Async**: Works with `def` and `async def`.

---

## Quickstart

```bash
pip install llmtrack
```

```python
import openai
from llmtrack import track

client = openai.OpenAI()


@track(provider="openai", tag="summarizer")
def summarize(text: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )


summarize("Autonomous agents require clean, local telemetry...")
```

View your metrics:
```bash
llmtrack report
```

<div align="center">
  <img src="assets/terminal_preview.svg" alt="llmtrack CLI Report Preview" width="100%" />
</div>

---

## How It Works

<div align="center">
  <img src="assets/architecture.svg" alt="llmtrack Architecture" width="100%" />
</div>

---

## Usage

### Custom Tracker & Async Support
```python
from llmtrack import Tracker

tracker = Tracker(db_path="calls.db")


@tracker.track(provider="anthropic", tag="agent", max_retries=3)
async def ask_claude(prompt: str):
    return await client.messages.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": prompt}],
    )
```

### Context Manager (`span`)
```python
with tracker.span(provider="openai", model="gpt-4o", tag="eval") as span:
    response = client.chat.completions.create(...)
    span.record(response)
```

### Daily Budget Guard
```python
tracker = Tracker(max_daily_usd=10.00)  # Raises BudgetExceededError before call if limit reached
```

### Pricing Overrides
```python
tracker = Tracker(pricing={"my-model": (1.50, 4.00)})  # $/1M input and output tokens
```

---

## CLI Report

```bash
llmtrack report                    # Formatted terminal tables
llmtrack report --since 7d         # Filter by timeframe (e.g. 24h, 7d, 30d)
llmtrack report --format json      # Export JSON
llmtrack report --format html      # Export standalone HTML dashboard
```

---

## Design Decisions

- **One row per logical call**: Retries and attempts are tracked on the logical call rather than bloating the DB with partial attempt rows.
- **Full jitter backoff**: Jittered exponential delay (`random(0, min(cap, base * 2^attempt))`) prevents thundering herd rate-limit retries.
- **NULL cost for unknown models**: `llmtrack` never guesses pricing. Unknown models log `cost_usd = NULL` and warn once so you can provide custom pricing.
- **SQLite WAL mode**: Provides high-concurrency readers and writers without external database infrastructure.

---

## Demo & Testing

Run the included simulation to generate 500 realistic calls (with latency spikes, 429s, and retries) and test the CLI:

```bash
python examples/demo.py
llmtrack report
pytest -v
```

---

## License

Apache 2.0 © [Sparsh Mishra](https://github.com/msparsh)
