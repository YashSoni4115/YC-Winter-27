"""SQLite write-ahead audit and durable, exactly-once request identifiers."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import sqlite3
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import IO, Any


def json_default(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, Enum)):
        return value.isoformat() if isinstance(value, datetime) else value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def encode(value: object) -> str:
    return json.dumps(value, default=json_default, allow_nan=False, sort_keys=True)


class DecisionStore:
    def __init__(self, path: str | Path) -> None:
        self._resources = ExitStack()
        self._lock_file: IO[str] | None = None
        if str(path) != ":memory:":
            path = Path(path).expanduser().resolve()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # The ExitStack owns this file for the entire runtime lifetime.
            self._lock_file = self._resources.enter_context(
                Path(str(path) + ".lock").open("a+")  # noqa: SIM115
            )
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._resources.close()
                raise RuntimeError("database already owned by another simulation runtime") from exc
        try:
            self.connection = sqlite3.connect(path, check_same_thread=False)
            self._initialize()
        except Exception:
            if hasattr(self, "connection"):
                self.connection.close()
            self._resources.close()
            raise

    def _initialize(self) -> None:
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > 2:
            raise ValueError("database schema is newer than this service")
        if version < 1:
            migration = Path(__file__).with_name("migrations") / "001_initial.sql"
            self.connection.executescript(migration.read_text())
        if version < 2:
            migration = Path(__file__).with_name("migrations") / "002_repeated_proposal_audits.sql"
            self.connection.executescript(migration.read_text())
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
        if self._lock_file is not None:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._resources.close()
            self._lock_file = None

    def metadata(self, key: str) -> Any:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_metadata(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key, encode(value))
        )

    def reserve(self, key: str, request_hash: str) -> dict[str, Any] | None:
        with self.connection:
            row = self.connection.execute(
                "SELECT * FROM commands WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise ValueError("idempotency key already belongs to a different request")
                if row["status"] == "pending":
                    raise RuntimeError("request has unresolved execution; automatic replay blocked")
                return json.loads(row["result_json"])  # type: ignore[no-any-return]
            self.connection.execute(
                "INSERT INTO commands VALUES(?,?, 'pending', NULL, ?)",
                (key, request_hash, datetime.now(timezone.utc).isoformat()),
            )
        return None

    def unresolved(self) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM commands WHERE status='pending' LIMIT 1"
            ).fetchone()
            is not None
        )

    def finish(self, key: str, result: Mapping[str, Any]) -> None:
        self.connection.execute(
            "UPDATE commands SET status='complete',result_json=? WHERE idempotency_key=?",
            (encode(result), key),
        )

    def add_telemetry(self, observation: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO telemetry VALUES(?,?,?)",
            (observation.step_index, observation.timestamp_utc.isoformat(), encode(observation)),
        )

    def history(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        if table not in {"telemetry", "decisions"}:
            raise ValueError("invalid history table")
        column = "sequence" if table == "telemetry" else "id"
        rows = self.connection.execute(
            f"SELECT * FROM {table} ORDER BY {column} DESC LIMIT ?", (limit,)
        ).fetchall()
        decoded = []
        for row in rows:
            item = dict(row)
            for key in list(item):
                if key.endswith("_json"):
                    value = item.pop(key)
                    item[key.removesuffix("_json")] = json.loads(value) if value else None
            decoded.append(item)
        return decoded
