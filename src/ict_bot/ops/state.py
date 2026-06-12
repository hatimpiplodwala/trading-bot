"""SQLite state store (Phase 7).

Tables: positions, daily_pnl, signals, journal. (veto_log is added only if the
v2 LLM veto is built.) Tracks open positions and realized daily P&L.
"""

from __future__ import annotations
