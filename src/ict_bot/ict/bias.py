"""HTF bias engine (Phase 3).

Combines Daily + 1H market structure into a single bias. Daily leads; the 1H
only vetoes on a direct contradiction (opposite bias). Longs are allowed only in
bullish bias, shorts only in bearish.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ict_bot.ict.structure import market_structure


def combine_bias(daily: str, h1: str) -> str:
    """Combine Daily and 1H bias. Daily leads; 1H vetoes on direct conflict."""
    if daily == "neutral":
        return "neutral"
    if h1 != "neutral" and h1 != daily:
        return "neutral"
    return daily


@dataclass(frozen=True)
class HTFBias:
    """Combined higher-timeframe bias and its components."""

    bias: str  # "bullish" | "bearish" | "neutral"
    daily: str
    h1: str

    def allows_long(self) -> bool:
        return self.bias == "bullish"

    def allows_short(self) -> bool:
        return self.bias == "bearish"


def htf_bias(
    daily_ohlc: pd.DataFrame,
    h1_ohlc: pd.DataFrame,
    swing_length: int = 50,
) -> HTFBias:
    """Compute combined HTF bias from Daily and 1H OHLC windows."""
    daily = market_structure(daily_ohlc, swing_length=swing_length).bias
    h1 = market_structure(h1_ohlc, swing_length=swing_length).bias
    return HTFBias(bias=combine_bias(daily, h1), daily=daily, h1=h1)
