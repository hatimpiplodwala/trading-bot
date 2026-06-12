"""Opening Range Breakout (intraday momentum, both directions).

A separate signal path from the ICT engine. The opening range is the high/low of
the session's first bars; the first bar to *close* beyond it is a momentum
breakout (long above the high, short below the low). The stop sits at the
opposite side of the range, floored by a fraction of ATR so a narrow range can't
produce a noise-tight stop. Pure functions here; per-day state, one-trade-a-day,
and flat-by-close live in the backtest strategy.
"""

from __future__ import annotations

import pandas as pd


def opening_range(or_bars: pd.DataFrame) -> tuple[float, float] | None:
    """``(high, low)`` of the opening-range bars, or ``None`` if there are none."""
    if len(or_bars) == 0:
        return None
    return float(or_bars["high"].max()), float(or_bars["low"].min())


def orb_breakout(close: float, or_high: float, or_low: float) -> str | None:
    """Breakout direction from a close beyond the range (touch alone is not enough)."""
    if close > or_high:
        return "long"
    if close < or_low:
        return "short"
    return None


def orb_stop(
    direction: str,
    entry: float,
    or_high: float,
    or_low: float,
    atr: float,
    atr_floor_mult: float = 0.5,
) -> float:
    """Stop at the opposite side of the range, but never tighter than the ATR floor."""
    floor_dist = atr_floor_mult * atr
    if direction == "long":
        return min(or_low, entry - floor_dist)
    return max(or_high, entry + floor_dist)


def orb_target(direction: str, entry: float, stop: float, rr: float | None) -> float | None:
    """Optional R-multiple target; ``None`` when no fixed target is configured."""
    if rr is None:
        return None
    risk = abs(entry - stop)
    return entry + rr * risk if direction == "long" else entry - rr * risk
