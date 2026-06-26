"""Persistent state (Phase 7) — SQLite tables for signals and closed trades.

Two jobs: (1) a *durable* one-trade-per-day guard so a crash-and-relaunch can't
re-enter a symbol already traded today, and (2) an audit log of every signal and
closed trade. Plain rows — no LLM, no journal prose (Phase 6 dropped from v1).
"""

from __future__ import annotations

import datetime as dt
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT NOT NULL,
    et_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry REAL, stop REAL, target REAL, qty INTEGER,
    order_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_day ON signals (symbol, et_date);

CREATE TABLE IF NOT EXISTS trades (
    symbol TEXT NOT NULL,
    entry_ts TEXT, exit_ts TEXT,
    direction TEXT,
    entry_price REAL, exit_price REAL, qty INTEGER, pnl REAL
);
"""


def _iso_date(value: dt.date | str) -> str:
    return value.isoformat() if isinstance(value, dt.date) else str(value)


class StateStore:
    """SQLite-backed signal/trade store. ``path=":memory:"`` for ephemeral use."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_signal(
        self, *, ts: str, et_date: dt.date | str, symbol: str, direction: str,
        entry: float, stop: float, target: float | None, qty: int, order_id: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO signals (ts, et_date, symbol, direction, entry, stop, "
            "target, qty, order_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, _iso_date(et_date), symbol, direction, entry, stop, target, qty, order_id),
        )
        self._conn.commit()

    def has_traded_today(self, symbol: str, et_date: dt.date | str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM signals WHERE symbol=? AND et_date=? LIMIT 1",
            (symbol, _iso_date(et_date)),
        ).fetchone()
        return row is not None

    def last_signal(self, symbol: str) -> dict | None:
        """The most recent recorded signal for ``symbol`` (to rebuild tracking on restart)."""
        row = self._conn.execute(
            "SELECT * FROM signals WHERE symbol=? ORDER BY rowid DESC LIMIT 1", (symbol,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_trade(
        self, *, symbol: str, entry_ts: str, exit_ts: str, direction: str,
        entry_price: float, exit_price: float, qty: int, pnl: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO trades (symbol, entry_ts, exit_ts, direction, entry_price, "
            "exit_price, qty, pnl) VALUES (?,?,?,?,?,?,?,?)",
            (symbol, entry_ts, exit_ts, direction, entry_price, exit_price, qty, pnl),
        )
        self._conn.commit()

    def trades(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM trades ORDER BY rowid").fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
