"""Market structure (Phase 2).

Wraps ``smc.swing_highs_lows()`` and ``smc.bos_choch()``. Current bias is taken
from the most recently *broken* structure event (BOS or CHOCH) — unbroken levels
are not yet confirmed and never set the bias (no lookahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ict_bot.ict._smc import smc


@dataclass(frozen=True)
class MarketStructure:
    """Confirmed structure state at the end of a window."""

    bias: str  # "bullish" | "bearish" | "neutral"
    last_event: str | None  # "BOS" | "CHOCH" | None
    last_level: float | None
    broken_index: int | None


def interpret_structure(bos_choch: pd.DataFrame) -> MarketStructure:
    """Derive :class:`MarketStructure` from an smc ``bos_choch`` frame."""
    broken = bos_choch[bos_choch["BrokenIndex"].notna()]
    if broken.empty:
        return MarketStructure("neutral", None, None, None)

    row = broken.loc[broken["BrokenIndex"].idxmax()]
    if pd.notna(row["BOS"]):
        event, direction = "BOS", row["BOS"]
    else:
        event, direction = "CHOCH", row["CHOCH"]

    bias = "bullish" if direction > 0 else "bearish"
    return MarketStructure(
        bias=bias,
        last_event=event,
        last_level=float(row["Level"]),
        broken_index=int(row["BrokenIndex"]),
    )


def market_structure(ohlc: pd.DataFrame, swing_length: int = 50) -> MarketStructure:
    """Compute confirmed market structure for an OHLC window.

    ``ohlc`` is open-time-indexed; smc operates positionally so the index is not
    required to be reset. Callers pass only closed bars up to the decision bar.
    """
    swings = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    return interpret_structure(smc.bos_choch(ohlc, swings))
