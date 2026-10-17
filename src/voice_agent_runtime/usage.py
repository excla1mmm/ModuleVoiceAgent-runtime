"""Usage metering for billing: aggregate counters per day, tenant (PBX) and agent.

Only counters are kept - no phone numbers, call ids or texts. They are added up in RAM when a call ends and
flushed to a small SQLite file (``VA_USAGE_DB``, a data volume) every ``VA_USAGE_FLUSH_SECONDS`` and at
shutdown, so a crash loses at most that interval. Days are UTC dates of the call start. Without a database
path the counters live in RAM for the life of the process.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from contextlib import closing
from datetime import date
from pathlib import Path

COUNTERS = ("calls", "connected_seconds", "agent_speech_seconds", "caller_turns", "llm_requests",
            "llm_input_tokens", "llm_output_tokens", "llm_estimated_requests", "tts_characters", "fallbacks")
_REAL = {"connected_seconds", "agent_speech_seconds"}
_SCHEMA = ("CREATE TABLE IF NOT EXISTS usage (day TEXT NOT NULL, tenant TEXT NOT NULL, agent TEXT NOT NULL, "
           + ", ".join(f"{c} {'REAL' if c in _REAL else 'INTEGER'} NOT NULL DEFAULT 0" for c in COUNTERS)
           + ", PRIMARY KEY (day, tenant, agent)) WITHOUT ROWID")
_UPSERT = (f"INSERT INTO usage (day, tenant, agent, {', '.join(COUNTERS)}) VALUES (?, ?, ?, "
           + ", ".join("?" for _ in COUNTERS) + ") ON CONFLICT (day, tenant, agent) DO UPDATE SET "
           + ", ".join(f"{c} = {c} + excluded.{c}" for c in COUNTERS))

Key = tuple[str, str, str]  # (day, tenant, agent)


def _zero() -> dict[str, float]:
    return dict.fromkeys(COUNTERS, 0)


def _rounded(counters: dict[str, float]) -> dict[str, float]:
    return {c: round(v, 1) if c in _REAL else int(v) for c, v in counters.items()}


class UsageMeter:
    def __init__(self, path: Path | None):
        self.path = path
        self._pending: dict[Key, dict[str, float]] = defaultdict(_zero)
        self._lock = threading.Lock()  # guards _pending
        self._io = threading.Lock()  # serializes flushes and reads, so a read never misses a row in flight
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as db, db:
                db.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, day: str, tenant: str, agent: str, counters: dict[str, float]) -> None:
        """Adds one call's counters (thread-safe, no I/O)."""
        unknown = set(counters) - set(COUNTERS)
        if unknown:
            raise ValueError(f"unknown usage counters: {', '.join(sorted(unknown))}")
        with self._lock:
            row = self._pending[(day, tenant, agent)]
            for name, value in counters.items():
                row[name] += value

    def flush(self) -> None:
        """Writes the pending counters to the database; on failure they are kept for the next flush."""
        if self.path is None:
            return
        with self._io:
            with self._lock:
                pending, self._pending = self._pending, defaultdict(_zero)
            if not pending:
                return
            try:
                with closing(self._connect()) as db, db:
                    db.executemany(_UPSERT, [(*key, *(row[c] for c in COUNTERS)) for key, row in pending.items()])
            except sqlite3.Error:
                with self._lock:
                    for key, row in pending.items():
                        for name, value in row.items():
                            self._pending[key][name] += value
                raise

    def _rows(self, where: str, args: tuple) -> dict[Key, dict[str, float]]:
        """Stored plus pending counters matching a filter (applied to pending rows by the caller)."""
        rows: dict[Key, dict[str, float]] = defaultdict(_zero)
        if self.path is not None:
            with closing(self._connect()) as db:
                for record in db.execute(f"SELECT day, tenant, agent, {', '.join(COUNTERS)} FROM usage {where}", args):
                    row = rows[tuple(record[:3])]
                    for name, value in zip(COUNTERS, record[3:], strict=True):
                        row[name] += value
        with self._lock:
            pending = {key: dict(row) for key, row in self._pending.items()}
        for key, row in pending.items():
            for name, value in row.items():
                rows[key][name] += value
        return rows

    def query(self, tenant: str, start: date, end: date) -> dict:
        """Counters of one tenant per day and agent, per agent and in total, for days start..end inclusive."""
        first, last = start.isoformat(), end.isoformat()
        with self._io:
            rows = self._rows("WHERE tenant = ? AND day BETWEEN ? AND ?", (tenant, first, last))
        days, agents, total = [], defaultdict(_zero), _zero()
        for (day, row_tenant, agent), row in sorted(rows.items()):
            if row_tenant != tenant or not first <= day <= last:
                continue
            days.append({"date": day, "agent_id": agent, **_rounded(row)})
            for name, value in row.items():
                agents[agent][name] += value
                total[name] += value
        return {"from": first, "to": last, "days": days,
                "agents": {agent: _rounded(row) for agent, row in sorted(agents.items())}, "total": _rounded(total)}

    def totals(self, tenant: str | None = None) -> dict[str, dict[str, float]]:
        """All-time counters per tenant (one tenant, or all of them)."""
        where, args = ("WHERE tenant = ?", (tenant,)) if tenant is not None else ("", ())
        with self._io:
            rows = self._rows(where, args)
        result: dict[str, dict[str, float]] = defaultdict(_zero)
        for (_, row_tenant, _), row in rows.items():
            if tenant is not None and row_tenant != tenant:
                continue
            for name, value in row.items():
                result[row_tenant][name] += value
        return {t: _rounded(row) for t, row in sorted(result.items())}
