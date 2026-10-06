"""SQLite storage and schema management for llmtrack."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

logger = logging.getLogger("llmtrack.storage")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    retries INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    tag TEXT,
    request_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls (ts_utc);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls (model);
CREATE INDEX IF NOT EXISTS idx_llm_calls_status ON llm_calls (status);
"""


@dataclass
class CallRecord:
    """Record representing a single logical LLM call."""

    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    attempts: int = 1
    retries: int = 0
    status: str = "ok"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    tag: Optional[str] = None
    request_id: Optional[str] = None
    ts_utc: Optional[str] = None

    def to_row(self) -> tuple[Any, ...]:
        ts = self.ts_utc or datetime.now(timezone.utc).isoformat()
        total_tokens = self.total_tokens
        if total_tokens is None and (
            self.input_tokens is not None or self.output_tokens is not None
        ):
            total_tokens = (self.input_tokens or 0) + (self.output_tokens or 0)

        err_msg = self.error_message
        if err_msg and len(err_msg) > 500:
            err_msg = err_msg[:497] + "..."

        return (
            ts,
            self.provider,
            self.model,
            self.input_tokens,
            self.output_tokens,
            total_tokens,
            self.cost_usd,
            self.latency_ms,
            self.attempts,
            self.retries,
            self.status,
            self.error_type,
            err_msg,
            self.tag,
            self.request_id,
        )


class Storage:
    """Thread-safe SQLite storage for LLM tracking records."""

    def __init__(self, db_path: str | Path = "llm_calls.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # Ensure parent directories exist
            if self.db_path != ":memory:":
                parent = Path(self.db_path).parent
                if parent and not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for high concurrency
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception as e:
                logger.warning("Could not set PRAGMA journal_mode=WAL: %s", e)
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables and indexes."""
        try:
            with self._lock:
                conn = self._get_connection()
                conn.executescript(SCHEMA_SQL)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize database schema at %s: %s", self.db_path, e)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for thread-safe database operations."""
        with self._lock:
            conn = self._get_connection()
            yield conn
            conn.commit()

    def insert(self, record: CallRecord) -> bool:
        """Safely insert a call record into SQLite. Never raises exceptions."""
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO llm_calls (
                        ts_utc, provider, model,
                        input_tokens, output_tokens, total_tokens,
                        cost_usd, latency_ms, attempts, retries,
                        status, error_type, error_message, tag, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    record.to_row(),
                )
            return True
        except Exception as e:
            logger.warning("llmtrack: Failed to log call to SQLite: %s", e)
            return False

    def insert_many(self, records: Sequence[CallRecord]) -> bool:
        """Safely bulk-insert multiple call records in a single transaction."""
        try:
            with self.transaction() as conn:
                conn.executemany(
                    """
                    INSERT INTO llm_calls (
                        ts_utc, provider, model,
                        input_tokens, output_tokens, total_tokens,
                        cost_usd, latency_ms, attempts, retries,
                        status, error_type, error_message, tag, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [r.to_row() for r in records],
                )
            return True
        except Exception as e:
            logger.warning("llmtrack: Failed to bulk log calls to SQLite: %s", e)
            return False

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Execute a parameterized query and return list of dicts."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def close(self) -> None:
        """Close all connections opened by this Storage instance."""
        with self._lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        self._local.conn = None
