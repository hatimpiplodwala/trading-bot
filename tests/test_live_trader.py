"""LiveTrader — the decision+execution step and the poll-schedule math.

The real sleep-loop and live REST fetch are validated by a manual dry-run during
market hours. Here we drive ``handle_bar`` with a DryRunBroker + in-memory state
to pin the live behaviour: enter on breakout, honour the durable one-per-day guard
across a restart, flatten by the close, and reconcile a broker-side stop-out.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ict_bot.broker.dryrun import DryRunBroker
from ict_bot.ops.alerts import LogNotifier
from ict_bot.ops.live_trader import LiveTrader, next_boundary
from ict_bot.ops.state import StateStore
from ict_bot.strategy.orb_session import Enter, ORBSession

_ET = "America/New_York"


def _times(start="09:30", end="15:45"):
    t, end_t = pd.Timestamp(f"2000-01-01 {start}"), pd.Timestamp(f"2000-01-01 {end}")
    out = []
    while t <= end_t:
        out.append(t.strftime("%H:%M"))
        t += pd.Timedelta(minutes=15)
    return out


def _day(date, overrides=None, base=500.0, rng=0.4):
    overrides = overrides or {}
    rows = []
    for hhmm in _times():
        ts = pd.Timestamp(f"{date} {hhmm}", tz=_ET).tz_convert("UTC")
        o, h, low, c = overrides.get(hhmm, (base, base + rng, base - rng, base))
        rows.append((ts, o, h, low, c))
    return rows


def _frame(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [1000] * len(rows)}, index=idx)


def _idx(frame, hhmm):
    et = frame.index.tz_convert(_ET)
    for k, t in enumerate(et):
        if t.strftime("%H:%M") == hhmm:
            return k
    raise KeyError(hhmm)


def _session():
    return ORBSession(
        range_end="10:00", no_entry_after="15:30", flatten_time="15:45",
        risk_pct=0.005, atr_length=14, stop_floor_mult=0.5, target_rr=None,
        daily_loss_limit=0.02)


def _trader(broker, state, session):
    return LiveTrader("SPY", None, broker, state, LogNotifier(), session)


def _drive(trader, frame, upto=None):
    upto = upto or len(frame)
    return [trader.handle_bar(frame.iloc[:k]) for k in range(1, upto + 1)]


_BREAKOUT_LONG = {
    "09:30": (500.0, 501.0, 500.0, 500.5),
    "09:45": (500.5, 500.8, 499.5, 500.2),
    "10:00": (500.2, 502.2, 500.2, 502.0),
}
_DAY = "2026-06-11"


def test_next_boundary_rounds_up_15m_with_buffer():
    now = dt.datetime(2026, 6, 12, 10, 7, 30, tzinfo=dt.timezone.utc)
    assert next_boundary(now, 15, 20) == dt.datetime(2026, 6, 12, 10, 15, 20, tzinfo=dt.timezone.utc)
    now2 = dt.datetime(2026, 6, 12, 10, 45, 1, tzinfo=dt.timezone.utc)
    assert next_boundary(now2, 15, 0) == dt.datetime(2026, 6, 12, 11, 0, 0, tzinfo=dt.timezone.utc)


def test_handle_bar_enters_on_breakout_and_records():
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = _trader(broker, state, session)
    frame = _frame(_day(_DAY, _BREAKOUT_LONG))
    actions = _drive(trader, frame, upto=_idx(frame, "10:00") + 1)

    assert any(isinstance(a, Enter) for a in actions)
    pos = broker.get_position("SPY")
    assert pos is not None and pos.side == "long"
    assert state.has_traded_today("SPY", dt.date(2026, 6, 11))
    assert [o["kind"] for o in broker.orders] == ["entry"]


def test_handle_bar_respects_durable_one_per_day_after_restart():
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    # A previous run already traded SPY today; this run starts with a fresh session.
    state.record_signal(ts="2026-06-11T13:45:00Z", et_date=dt.date(2026, 6, 11),
                        symbol="SPY", direction="long", entry=500.0, stop=499.5,
                        target=None, qty=10, order_id="prev")
    trader = _trader(broker, state, session)
    frame = _frame(_day(_DAY, _BREAKOUT_LONG))
    _drive(trader, frame, upto=_idx(frame, "10:00") + 1)

    assert broker.get_position("SPY") is None       # durable guard blocks re-entry
    assert all(o["kind"] != "entry" for o in broker.orders)


def test_full_day_enters_then_flattens_by_close():
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = _trader(broker, state, session)
    _drive(trader, _frame(_day(_DAY, _BREAKOUT_LONG)))

    assert broker.get_position("SPY") is None       # flat by EOD
    kinds = [o["kind"] for o in broker.orders]
    assert kinds.count("entry") == 1 and kinds.count("close") == 1
    assert len(state.trades()) == 1


def test_reconcile_records_stop_out_when_position_vanishes():
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = _trader(broker, state, session)
    frame = _frame(_day(_DAY, _BREAKOUT_LONG))
    i10 = _idx(frame, "10:00")
    _drive(trader, frame, upto=i10 + 1)
    assert broker.get_position("SPY") is not None

    broker._positions.pop("SPY")                     # broker-side stop fires between polls
    trader.handle_bar(frame.iloc[: i10 + 2])
    assert len(state.trades()) == 1
    assert trader._entry is None
