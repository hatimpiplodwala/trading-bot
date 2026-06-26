"""LiveTrader — the decision+execution step and the poll-schedule math.

The real sleep-loop and live REST fetch are validated by a manual dry-run during
market hours. Here we drive ``handle_bar`` with a DryRunBroker + in-memory state
to pin the live behaviour: enter on breakout, honour the durable one-per-day guard
across a restart, flatten by the close, and reconcile a broker-side stop-out.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ict_bot.broker.dryrun import DryRunBroker
from ict_bot.ops.alerts import LogNotifier
from ict_bot.ops.live_trader import LiveTrader, next_boundary, timeframe_minutes
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


class _FakeFeed:
    def __init__(self, frame):
        self._frame = frame

    def get_historical_bars(self, symbol, timeframe, start, end):
        return self._frame


class _Rec:
    def __init__(self):
        self.events = []

    def notify(self, event, message):
        self.events.append((event, message))


@pytest.mark.parametrize("tf,minutes", [("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240)])
def test_timeframe_minutes(tf, minutes):
    assert timeframe_minutes(tf) == minutes


def test_timeframe_minutes_rejects_garbage():
    with pytest.raises(ValueError):
        timeframe_minutes("frog")


def test_next_boundary_rounds_up_15m_with_buffer():
    now = dt.datetime(2026, 6, 12, 10, 7, 30, tzinfo=dt.timezone.utc)
    assert next_boundary(now, 15, 20) == dt.datetime(2026, 6, 12, 10, 15, 20, tzinfo=dt.timezone.utc)
    now2 = dt.datetime(2026, 6, 12, 10, 45, 1, tzinfo=dt.timezone.utc)
    assert next_boundary(now2, 15, 0) == dt.datetime(2026, 6, 12, 11, 0, 0, tzinfo=dt.timezone.utc)


def test_heartbeat_due_after_interval():
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = LiveTrader("SPY", None, broker, state, LogNotifier(), session, heartbeat_hours=1)
    t0 = dt.datetime(2026, 6, 15, 14, 0, tzinfo=dt.timezone.utc)
    trader._last_heartbeat = t0
    assert trader._heartbeat_due(t0 + dt.timedelta(minutes=30)) is False
    assert trader._heartbeat_due(t0 + dt.timedelta(hours=1, minutes=1)) is True
    # firing resets the timer
    assert trader._heartbeat_due(t0 + dt.timedelta(hours=1, minutes=2)) is False


def test_frame_trim_caps_length():
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = LiveTrader("SPY", None, broker, state, LogNotifier(), session, max_frame_bars=10)
    trader._frame = _frame(_day(_DAY))           # 26 bars
    trader._trim_frame()
    assert len(trader._frame) == 10              # keeps the most recent only


def test_request_stop_exits_run_loop_cleanly():
    feed = _FakeFeed(_frame(_day(_DAY)))
    notifier = _Rec()
    trader = LiveTrader("SPY", feed, DryRunBroker(100_000.0, True),
                        StateStore(":memory:"), notifier, _session())
    trader.request_stop()
    trader.run(max_iterations=3, sleep_fn=lambda s: None,
               now_fn=lambda: dt.datetime(2026, 6, 15, 14, 0, tzinfo=dt.timezone.utc))
    events = [e for e, _ in notifier.events]
    assert "stop" in events and "error" not in events


def test_interruptible_sleep_returns_early_when_stopped():
    # A stop requested mid-sleep (as the signal handler does) must be observed
    # within a slice or two, not swallowed by the full duration.
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = LiveTrader("SPY", None, broker, state, LogNotifier(), session)
    calls = {"n": 0}

    def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 2:
            trader.request_stop()

    trader._interruptible_sleep(100.0, fake_sleep)
    assert calls["n"] <= 3


def test_run_polls_on_entry_timeframe_grid_not_hardcoded_15m():
    # Poll cadence must follow entry_timeframe: a 5m bot wakes on the 5-minute
    # grid (14:07:30 -> 14:10:00 = 150s), not the old hardcoded 15m grid (450s).
    slept = []
    trader = LiveTrader(
        "SPY", _FakeFeed(_frame(_day(_DAY))), DryRunBroker(100_000.0, market_open=False),
        StateStore(":memory:"), _Rec(), _session(), timeframe="5m", poll_buffer_s=0,
    )
    now = dt.datetime(2026, 6, 15, 14, 7, 30, tzinfo=dt.timezone.utc)
    trader.run(max_iterations=1, sleep_fn=lambda s: slept.append(s), now_fn=lambda: now)
    assert round(sum(slept)) == 150


def test_run_honors_max_iterations_when_market_closed():
    feed = _FakeFeed(_frame(_day(_DAY)))
    notifier = _Rec()
    trader = LiveTrader("SPY", feed, DryRunBroker(100_000.0, market_open=False),
                        StateStore(":memory:"), notifier, _session())
    trader.run(max_iterations=2, sleep_fn=lambda s: None,
               now_fn=lambda: dt.datetime(2026, 6, 15, 14, 0, tzinfo=dt.timezone.utc))
    assert "stop" in [e for e, _ in notifier.events]


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


def test_flatten_cancels_resting_stop_before_closing():
    # The OTO entry leaves a resting stop at the broker. The timed flatten must
    # cancel it BEFORE liquidating: otherwise Alpaca can reject the close (the
    # stop is holding the shares) leaving an unprotected overnight position, or
    # the orphaned stop can fire after the close and re-open a position before
    # its DAY expiry.
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    trader = _trader(broker, state, session)
    _drive(trader, _frame(_day(_DAY, _BREAKOUT_LONG)))

    kinds = [o["kind"] for o in broker.orders]
    assert "cancel" in kinds
    assert kinds.index("cancel") < kinds.index("close")  # cancel before liquidation
    cancel = next(o for o in broker.orders if o["kind"] == "cancel")
    assert cancel["symbol"] == "SPY"


def test_reconcile_force_flattens_stale_overnight_position():
    # A position entered yesterday that the bot never flattened (it slept through
    # the 15:45 exit) must be force-closed on the next startup - not held for days.
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    state.record_signal(ts="2026-06-22T14:30:00+00:00", et_date=dt.date(2026, 6, 22),
                        symbol="SPY", direction="short", entry=745.96, stop=749.84,
                        target=None, qty=6, order_id="x")
    broker.submit_entry_with_stop("SPY", "short", 6, 749.84)  # broker still holds it
    trader = LiveTrader("SPY", None, broker, state, _Rec(), session, flatten_time="15:45")
    trader._frame = _frame(_day("2026-06-22"))                # gives a close for the exit proxy

    now = dt.datetime(2026, 6, 23, 13, 35, tzinfo=dt.timezone.utc)  # 09:35 ET, NEXT day
    trader.reconcile_open_position(now)

    assert broker.get_position("SPY") is None                 # force-flattened
    kinds = [o["kind"] for o in broker.orders]
    assert "cancel" in kinds and kinds[-1] == "close"         # cancelled stale stop, then closed
    assert len(state.trades()) == 1                           # recorded the close
    assert trader._entry is None


def test_reconcile_keeps_same_day_position_before_flatten():
    # A legit mid-session restart must NOT close the live position - just rebuild
    # tracking so reconciliation/exit still work.
    broker, state, session = DryRunBroker(100_000.0, True), StateStore(":memory:"), _session()
    state.record_signal(ts="2026-06-23T14:30:00+00:00", et_date=dt.date(2026, 6, 23),
                        symbol="SPY", direction="short", entry=745.96, stop=749.84,
                        target=None, qty=6, order_id="x")
    broker.submit_entry_with_stop("SPY", "short", 6, 749.84)
    trader = LiveTrader("SPY", None, broker, state, _Rec(), session, flatten_time="15:45")
    trader._frame = _frame(_day("2026-06-23"))

    now = dt.datetime(2026, 6, 23, 15, 0, tzinfo=dt.timezone.utc)  # 11:00 ET, before 15:45
    trader.reconcile_open_position(now)

    assert broker.get_position("SPY") is not None             # kept
    assert trader._entry is not None and trader._entry["direction"] == "short"  # tracking restored
    assert all(o["kind"] != "close" for o in broker.orders)


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
