"""ORBSession — the live decision core, tested for parity with the backtest.

ORBSession runs the same opening-range-breakout logic as the backtest's
ORBStrategy.next(), but broker-agnostic: fed the session's bars so far it returns
an action (NONE / FLATTEN / Enter). These tests pin the entry, one-per-day,
late-cutoff and flatten rules without any network or broker.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.strategy.orb_session import Action, Enter, ORBSession

_ET = "America/New_York"


def _times(start: str = "09:30", end: str = "15:45") -> list[str]:
    t = pd.Timestamp(f"2000-01-01 {start}")
    end_t = pd.Timestamp(f"2000-01-01 {end}")
    out: list[str] = []
    while t <= end_t:
        out.append(t.strftime("%H:%M"))
        t += pd.Timedelta(minutes=15)
    return out


def _day(date: str, overrides: dict[str, tuple] | None = None,
         base: float = 500.0, rng: float = 0.4) -> list[tuple]:
    overrides = overrides or {}
    rows = []
    for hhmm in _times():
        ts = pd.Timestamp(f"{date} {hhmm}", tz=_ET).tz_convert("UTC")
        if hhmm in overrides:
            o, h, low, c = overrides[hhmm]
        else:
            o = c = base
            h, low = base + rng, base - rng
        rows.append((ts, o, h, low, c))
    return rows


def _frame(rows: list[tuple]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([r[0] for r in rows], name="timestamp")
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [1000] * len(rows)},
        index=idx,
    )


def _session() -> ORBSession:
    return ORBSession(
        range_end="10:00", no_entry_after="15:30", flatten_time="15:45",
        risk_pct=0.005, atr_length=14, stop_floor_mult=0.5, target_rr=None,
        daily_loss_limit=0.02,
    )


def _replay(session, frame, equity=100_000.0, has_position=False):
    """Feed every bar prefix to on_bar; return the list of actions."""
    actions = []
    for k in range(1, len(frame) + 1):
        actions.append(session.on_bar(frame.iloc[:k], equity, has_position))
    return actions


# A clean long-breakout day: OR is [499.5, 501.0]; 10:00 closes at 502.0.
_BREAKOUT_LONG = {
    "09:30": (500.0, 501.0, 500.0, 500.5),
    "09:45": (500.5, 500.8, 499.5, 500.2),
    "10:00": (500.2, 502.2, 500.2, 502.0),
}


def test_or_window_bars_do_nothing():
    rows = _day("2026-06-11", _BREAKOUT_LONG)
    frame = _frame(rows)
    # First two bars are inside the opening range -> no action.
    s = _session()
    assert s.on_bar(frame.iloc[:1], 100_000.0, False) is Action.NONE
    assert s.on_bar(frame.iloc[:2], 100_000.0, False) is Action.NONE


def test_breakout_enters_long_with_range_stop():
    frame = _frame(_day("2026-06-11", _BREAKOUT_LONG))
    actions = _replay(_session(), frame)
    enters = [a for a in actions if isinstance(a, Enter)]
    assert len(enters) == 1
    e = enters[0]
    assert e.direction == "long"
    assert e.stop == 499.5          # opposite side of the range binds over the ATR floor
    assert e.qty > 0
    assert e.target is None          # target_rr null -> exit at the close, no TP


def test_only_one_trade_per_day():
    # A second close beyond the range later the same day must not re-enter.
    overrides = dict(_BREAKOUT_LONG)
    overrides["11:00"] = (502.0, 503.5, 502.0, 503.2)  # another breakout-y bar
    frame = _frame(_day("2026-06-11", overrides))
    enters = [a for a in _replay(_session(), frame) if isinstance(a, Enter)]
    assert len(enters) == 1


def test_no_entry_after_cutoff():
    # Breakout only appears at 15:30 (== no_entry_after) -> too late, skip.
    overrides = {"15:30": (500.2, 503.0, 500.2, 502.5)}
    frame = _frame(_day("2026-06-11", overrides))
    enters = [a for a in _replay(_session(), frame) if isinstance(a, Enter)]
    assert enters == []


def test_flatten_when_holding_at_flatten_time():
    frame = _frame(_day("2026-06-11"))
    actions = _replay(_session(), frame, has_position=True)
    # The 15:45 bar (== flatten_time) must request a flatten; earlier bars do not.
    et = frame.index.tz_convert(_ET)
    flatten_idx = [k for k, t in enumerate(et) if t.strftime("%H:%M") == "15:45"][0]
    assert actions[flatten_idx] is Action.FLATTEN
    assert all(a is Action.NONE for a in actions[:flatten_idx])


def test_new_day_resets_one_per_day():
    s = _session()
    _replay(s, _frame(_day("2026-06-11", _BREAKOUT_LONG)))           # day 1: enters
    actions2 = _replay(s, _frame(_day("2026-06-12", _BREAKOUT_LONG)))  # day 2: fresh
    assert any(isinstance(a, Enter) for a in actions2)


def test_close_inside_range_does_not_enter():
    frame = _frame(_day("2026-06-11"))  # flat day, never breaks out
    enters = [a for a in _replay(_session(), frame) if isinstance(a, Enter)]
    assert enters == []
