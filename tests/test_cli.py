"""Tests for CLI analytics reports and duration parsing."""

import json
import tempfile
from pathlib import Path

from llmtrack.cli import load_queries, main, parse_since, run_report
from llmtrack.storage import CallRecord, Storage


def test_load_queries():
    queries = load_queries()
    assert "cost_per_day" in queries
    assert "p95_latency" in queries
    assert "error_and_retry_rates" in queries
    assert "top_tags" in queries
    assert "slowest_calls" in queries
    assert "rate_limit_incidents" in queries


def test_parse_since():
    iso_7d = parse_since("7d")
    assert iso_7d is not None
    iso_24h = parse_since("24h")
    assert iso_24h is not None
    assert parse_since(None) is None
    assert parse_since("invalid") is None


def test_cli_report_json(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "cli_test.db"
        storage = Storage(db_path=db_path)
        try:
            storage.insert(
                CallRecord(
                    provider="openai",
                    model="gpt-4o",
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=0.00075,
                    latency_ms=120.0,
                    status="ok",
                )
            )
        finally:
            storage.close()

        code = run_report(db_path=str(db_path), output_format="json")
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "Cost & Volume Per Day" in data
        assert len(data["Cost & Volume Per Day"]) == 1
        assert data["Cost & Volume Per Day"][0]["model"] == "gpt-4o"


def test_cli_report_csv(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "cli_test.db"
        storage = Storage(db_path=db_path)
        try:
            storage.insert(
                CallRecord(
                    provider="anthropic",
                    model="claude-3-5-sonnet",
                    latency_ms=150.0,
                    status="ok",
                )
            )
        finally:
            storage.close()

        code = run_report(db_path=str(db_path), output_format="csv")
        assert code == 0
        captured = capsys.readouterr()
        assert "Cost & Volume Per Day" in captured.out
        assert "claude-3-5-sonnet" in captured.out


def test_cli_report_table(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "cli_test.db"
        storage = Storage(db_path=db_path)
        try:
            storage.insert(
                CallRecord(
                    provider="anthropic",
                    model="claude-3-5-sonnet",
                    latency_ms=150.0,
                    status="ok",
                )
            )
        finally:
            storage.close()

        code = run_report(db_path=str(db_path), output_format="table")
        assert code == 0
        captured = capsys.readouterr()
        assert "Cost & Volume Per Day" in captured.out or "p95" in captured.out


def test_cli_main_entrypoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "cli_test.db"
        storage = Storage(db_path=db_path)
        try:
            storage.insert(CallRecord(provider="openai", model="gpt-4o-mini", latency_ms=50.0))
        finally:
            storage.close()

        code = main(["report", "--db", str(db_path), "--format", "json"])
        assert code == 0


def test_cli_nonexistent_db(capsys):
    code = run_report(db_path="non_existent_file_path.db")
    assert code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
