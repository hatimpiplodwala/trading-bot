"""Structural stops & liquidity targets (structural rework, Approach A).

Replaces the old fixed-2R target + range-extreme stop with ICT-native trade
management:

* **Structural stop** — the nearest swing *against* the trade (swing low below a
  long, swing high above a short): the price that invalidates the setup.
* **Liquidity target** — the nearest opposing resting-liquidity pool *beyond*
  entry (swing high above a long, swing low below a short), searched over a
  bounded recent window so it is reachable within the session.

The nearest-on-side selection is pure and unit-tested; swing extraction leans on
smc and is covered by the real-data integration test.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ict_bot.ict._smc import smc


def _nearest_on_side(levels: Iterable[float], entry: float, side: str) -> float | None:
    """Nearest level strictly on one side of entry (``"below"`` -> highest below,
    ``"above"`` -> lowest above). ``None`` if no level is on that side."""
    if side == "below":
        cands = [lvl for lvl in levels if lvl < entry]
        return max(cands) if cands else None
    cands = [lvl for lvl in levels if lvl > entry]
    return min(cands) if cands else None


def _swing_levels(window: pd.DataFrame, swing_length: int, high_low: int) -> list[float]:
    """smc swing levels of one kind (``1`` = highs, ``-1`` = lows) in ``window``."""
    positional = window.reset_index(drop=True)
    swings = smc.swing_highs_lows(positional, swing_length=swing_length)
    levels = swings.loc[swings["HighLow"] == high_low, "Level"].dropna()
    return [float(x) for x in levels]


def structural_stop_level(
    window: pd.DataFrame, direction: str, entry: float, swing_length: int = 20
) -> float | None:
    """Nearest swing that invalidates the setup (below a long / above a short)."""
    if direction == "long":
        return _nearest_on_side(_swing_levels(window, swing_length, -1), entry, "below")
    return _nearest_on_side(_swing_levels(window, swing_length, 1), entry, "above")


def liquidity_target(
    window: pd.DataFrame,
    direction: str,
    entry: float,
    swing_length: int = 5,
    lookback: int = 40,
) -> float | None:
    """Nearest opposing liquidity pool beyond entry (above a long / below a short).

    Searches the last ``lookback`` bars with a small ``swing_length`` so the pool
    is a *nearby* intraday level the price can plausibly reach before the close.
    """
    recent = window.tail(lookback)
    if direction == "long":
        return _nearest_on_side(_swing_levels(recent, swing_length, 1), entry, "above")
    return _nearest_on_side(_swing_levels(recent, swing_length, -1), entry, "below")
