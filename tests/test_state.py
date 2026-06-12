"""StateStore — durable one-trade-per-day guard and trade log (SQLite).

The in-memory ORBSession enforces one-trade-per-day within a run; StateStore makes
it survive a restart (a crash-and-relaunch mid-session must not re-enter) and keeps
a persistent record of signals and closed trades.
"""

from __future__ import annotations

import datetime as dt

from ict_bot.ops.state import StateStore


def test_has_traded_today_flips_after_signal():
    s = StateStore(":memory:")
    day = dt.date(2026, 6, 12)
    assert s.has_traded_today("SPY", day) is False

    s.record_signal(
        ts="2026-06-12T14:00:00Z", et_date=day, symbol="SPY", direction="long",
        entry=500.0, stop=499.5, target=None, qty=199, order_id="abc",
    )
    assert s.has_traded_today("SPY", day) is True
    assert s.has_traded_today("SPY", dt.date(2026, 6, 13)) is False
    # Per-symbol: a SPY signal does not mark QQQ as traded.
    assert s.has_traded_today("QQQ", day) is False


def test_signals_and_trades_persist_across_reopen(tmp_path):
    path = str(tmp_path / "state.db")
    s = StateStore(path)
    day = dt.date(2026, 6, 12)
    s.record_signal(
        ts="2026-06-12T14:00:00Z", et_date=day, symbol="SPY", direction="long",
        entry=500.0, stop=499.5, target=None, qty=199, order_id="abc",
    )
    s.record_trade(
        symbol="SPY", entry_ts="2026-06-12T14:00:00Z", exit_ts="2026-06-12T19:45:00Z",
        direction="long", entry_price=500.0, exit_price=502.0, qty=199, pnl=398.0,
    )
    s.close()

    reopened = StateStore(path)
    assert reopened.has_traded_today("SPY", day) is True  # survives restart
    trades = reopened.trades()
    assert len(trades) == 1
    assert abs(trades[0]["pnl"] - 398.0) < 1e-9
