"""Opening Range Breakout pure-logic tests (momentum strategy).

The opening range is the high/low of the first bars of the session; a close
beyond it is a breakout (long above the high, short below the low). The stop sits
at the opposite side of the range, floored so a tiny range can't give a
noise-tight stop. The strategy wiring (per-day state, flat-by-close) is covered
by the integration test.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.strategy.orb import (
    opening_range,
    orb_breakout,
    orb_stop,
    orb_target,
)


def _bars(highs, lows):
    idx = pd.date_range("2024-03-01 14:30", periods=len(highs), freq="15min", tz="UTC")
    return pd.DataFrame({"high": highs, "low": lows}, index=idx)


def test_opening_range_is_high_low_of_window():
    assert opening_range(_bars([101, 102], [99, 100])) == (102.0, 99.0)


def test_opening_range_none_when_empty():
    assert opening_range(_bars([], [])) is None


def test_breakout_long_above_high():
    assert orb_breakout(close=102.5, or_high=102.0, or_low=99.0) == "long"


def test_breakout_short_below_low():
    assert orb_breakout(close=98.5, or_high=102.0, or_low=99.0) == "short"


def test_no_breakout_inside_range():
    assert orb_breakout(close=100.5, or_high=102.0, or_low=99.0) is None
    assert orb_breakout(close=102.0, or_high=102.0, or_low=99.0) is None  # touch != close beyond


def test_stop_long_at_range_low_when_wider_than_floor():
    # entry 102.5, or_low 99 -> 3.5 risk > 0.5*ATR(2)=1 floor -> stop at the OR low.
    assert orb_stop("long", entry=102.5, or_high=102.0, or_low=99.0, atr=2.0, atr_floor_mult=0.5) == 99.0


def test_stop_long_uses_floor_when_range_too_tight():
    # entry 102.1, or_low 102.0 -> 0.1 risk < floor 1.0 -> push stop to entry - 1.0.
    assert orb_stop("long", entry=102.1, or_high=102.0, or_low=102.0, atr=2.0, atr_floor_mult=0.5) == 101.1


def test_stop_short_mirrors_long():
    assert orb_stop("short", entry=98.5, or_high=102.0, or_low=99.0, atr=2.0, atr_floor_mult=0.5) == 102.0
    assert orb_stop("short", entry=98.9, or_high=99.0, or_low=99.0, atr=2.0, atr_floor_mult=0.5) == 99.9


def test_target_none_when_rr_none():
    assert orb_target("long", entry=100.0, stop=98.0, rr=None) is None


def test_target_is_r_multiple_when_set():
    assert orb_target("long", entry=100.0, stop=98.0, rr=2.0) == 104.0   # 2R of 2-pt risk
    assert orb_target("short", entry=100.0, stop=102.0, rr=2.0) == 96.0
