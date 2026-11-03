"""Append-only SQLite audit ledger with a verifiable SHA-256 hash chain."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

DB_PATH = Path(os.getenv(
    "FINGUARD_DB_PATH",
    str(Path(__file__).resolve().parents[1] / "finguard_audit.db"),
)).expanduser().resolve()
_LOCK = RLock()
GENESIS_HASH = "0" * 64

class AuditLedger:
    def __init__(self, db_path: str | Path = DB_PATH, *, initialize: bool = True):
        self.db_path = str(db_path)
        self._initialized = False
        self._initializing = False
        self._initialize_lock = RLock()
        if initialize:
            self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self):
        """Commit or roll back a transaction, and always release its connection."""
        if not self._initialized and not self._initializing:
            self._initialize()
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._initialize_lock:
            if self._initialized:
                return
            self._initializing = True
            try:
                with self._connection() as conn:
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = FULL")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS audit_records (
                            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            txn_id TEXT NOT NULL,
                            timestamp_utc TEXT NOT NULL,
                            fraud_score REAL NOT NULL,
                            verdict TEXT NOT NULL,
                            trust_score REAL NOT NULL,
                            manipulation_risk TEXT NOT NULL,
                            payload_json TEXT NOT NULL,
                            previous_hash TEXT NOT NULL,
                            record_hash TEXT NOT NULL UNIQUE
                        )
                    """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_audit_records_txn_id "
                        "ON audit_records (txn_id, audit_id DESC)"
                    )
                self._initialized = True
            finally:
                self._initializing = False

    @staticmethod
    def _hash(previous_hash: str, txn_id: str, timestamp_utc: str, payload_json: str) -> str:
        message = f"{previous_hash}|{txn_id}|{timestamp_utc}|{payload_json}"
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    def append(self, payload: dict) -> str:
        payload_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        with _LOCK, self._connection() as conn:
            # Serialize competing appenders across processes before choosing the
            # previous hash; a process-local lock alone is insufficient.
            conn.execute("BEGIN IMMEDIATE")
            last = conn.execute(
                "SELECT record_hash FROM audit_records ORDER BY audit_id DESC LIMIT 1"
            ).fetchone()
            previous = last["record_hash"] if last else GENESIS_HASH
            txn_id = str(payload["txn_id"])
            digest = self._hash(previous, txn_id, timestamp, payload_json)
            conn.execute("""
                INSERT INTO audit_records
                (txn_id,timestamp_utc,fraud_score,verdict,trust_score,manipulation_risk,
                 payload_json,previous_hash,record_hash)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (txn_id, timestamp, float(payload["fraud_score"]), payload["verdict"],
                  float(payload["trust_score"]), payload["manipulation_risk"],
                  payload_json, previous, digest))
            conn.commit()
        return digest

    def verify_chain(self) -> tuple[bool, int]:
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM audit_records ORDER BY audit_id ASC").fetchall()
        previous = GENESIS_HASH
        for index, row in enumerate(rows, start=1):
            if row["previous_hash"] != previous:
                return False, index
            expected = self._hash(previous, row["txn_id"], row["timestamp_utc"], row["payload_json"])
            if not hmac.compare_digest(expected, row["record_hash"]):
                return False, index
            try:
                payload = json.loads(row["payload_json"])
                denormalized_matches = (
                    str(payload["txn_id"]) == row["txn_id"]
                    and float(payload["fraud_score"]) == float(row["fraud_score"])
                    and payload["verdict"] == row["verdict"]
                    and float(payload["trust_score"]) == float(row["trust_score"])
                    and payload["manipulation_risk"] == row["manipulation_risk"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return False, index
            if not denormalized_matches:
                return False, index
            previous = row["record_hash"]
        return True, len(rows)

    def list_records(self, limit: int = 100) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_records ORDER BY audit_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_record(self, txn_id: str) -> dict | None:
        """Fetch the newest audit row for a transaction using the indexed ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM audit_records WHERE txn_id = ? ORDER BY audit_id DESC LIMIT 1",
                (txn_id,),
            ).fetchone()
        return dict(row) if row else None

ledger = AuditLedger(initialize=False)
