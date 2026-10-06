"""Tests for SQLite storage and p95 query verification against Python statistics."""

import math
import tempfile
from pathlib import Path

from llmtrack.cli import load_queries
from llmtrack.storage import CallRecord, Storage


def test_storage_creation_and_insertion():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = Storage(db_path=db_path)
        try:
            record = CallRecord(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.00075,
                latency_ms=123.4,
                attempts=1,
                retries=0,
                status="ok",
                tag="unit-test",
            )
            assert storage.insert(record) is True

            rows = storage.query("SELECT * FROM llm_calls")
            assert len(rows) == 1
            row = rows[0]
            assert row["provider"] == "openai"
            assert row["model"] == "gpt-4o"
            assert row["input_tokens"] == 100
            assert row["output_tokens"] == 50
            assert row["total_tokens"] == 150
            assert row["cost_usd"] == 0.00075
            assert row["status"] == "ok"
            assert row["tag"] == "unit-test"
        finally:
            storage.close()


def test_broken_db_never_crashes():
    # Provide an invalid path that cannot be written to
    invalid_path = Path("Z:/non_existent_drive_12345/impossible.db")
    storage = Storage(db_path=invalid_path)
    try:
        record = CallRecord(provider="anthropic", latency_ms=10.0, status="ok")
        # insert should gracefully catch and return False without raising
        result = storage.insert(record)
        assert result is False
    finally:
        storage.close()


def test_p95_sql_vs_python_statistics():
    """Verify that the SQLite nearest-rank p95 matches mathematical definition."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "p95_test.db"
        storage = Storage(db_path=db_path)
        try:
            # Insert 100 known latency values: 1.0 to 100.0 ms
            latencies = [float(i) for i in range(1, 101)]
            for lat in latencies:
                storage.insert(
                    CallRecord(
                        provider="test",
                        model="model-a",
                        latency_ms=lat,
                        status="ok",
                        ts_utc="2026-10-06T10:00:00Z",
                    )
                )

            # Nearest rank method for 100 items:
            # rank = ceil(0.95 * 100) = 95
            # The 95th sorted item (1-indexed) is 95.0
            queries = load_queries()
            p95_sql = queries["p95_latency"]
            rows = storage.query(p95_sql, (None, None))
            assert len(rows) == 1
            assert rows[0]["model"] == "model-a"
            assert rows[0]["calls"] == 100
            # Check nearest-rank p95 value
            assert math.isclose(rows[0]["p95_ms"], 95.0, abs_tol=0.1)
        finally:
            storage.close()


def test_error_truncation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "err_test.db"
        storage = Storage(db_path=db_path)
        try:
            long_msg = "x" * 1000
            record = CallRecord(
                provider="openai",
                latency_ms=50.0,
                status="error",
                error_message=long_msg,
            )
            storage.insert(record)
            rows = storage.query("SELECT error_message FROM llm_calls")
            assert len(rows[0]["error_message"]) == 500
            assert rows[0]["error_message"].endswith("...")
        finally:
            storage.close()
