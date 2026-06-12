"""Order blocks (Phase 2).

Wraps ``smc.ob()`` to locate order blocks, then tracks each block's lifecycle
ourselves: active -> mitigated -> breaker. A block is *mitigated* when price
overlaps its zone, and becomes a *breaker* when a bar CLOSES through the far side
(below bottom for a bullish OB, above top for a bearish OB). OBs are the most
discretionary ICT concept — we pin the smc definition and don't second-guess
against guru charts (Gotcha #2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from ict_bot.ict._smc import smc


@dataclass(frozen=True)
class OrderBlock:
    """An order block and its lifecycle state."""

    type: str  # "bullish" (OB=1) | "bearish" (OB=-1)
    top: float
    bottom: float
    volume: float
    formed_at: pd.Timestamp
    state: str  # "active" | "mitigated" | "breaker"
    mitigated_at: pd.Timestamp | None
    broken_at: pd.Timestamp | None


def ob_state(
    ob_type: str,
    top: float,
    bottom: float,
    formed_pos: int,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> tuple[str, int | None, int | None]:
    """Classify an OB by scanning bars after ``formed_pos``.

    Returns ``(state, mitigated_pos, broken_pos)`` with absolute positions.
    """
    mitigated_pos: int | None = None
    broken_pos: int | None = None

    for i in range(formed_pos + 1, len(closes)):
        tapped = lows[i] <= top and highs[i] >= bottom
        if tapped and mitigated_pos is None:
            mitigated_pos = i

        broke = closes[i] < bottom if ob_type == "bullish" else closes[i] > top
        if broke:
            broken_pos = i
            break

    if broken_pos is not None:
        state = "breaker"
    elif mitigated_pos is not None:
        state = "mitigated"
    else:
        state = "active"
    return state, mitigated_pos, broken_pos


def detect_order_blocks(ohlc: pd.DataFrame, swing_length: int = 50) -> list[OrderBlock]:
    """Detect order blocks in an open-time-indexed OHLC window."""
    timestamps = ohlc.index
    positional = ohlc.reset_index(drop=True)
    swings = smc.swing_highs_lows(positional, swing_length=swing_length)
    obs = smc.ob(positional, swings)

    highs = positional["high"].tolist()
    lows = positional["low"].tolist()
    closes = positional["close"].tolist()

    out: list[OrderBlock] = []
    present = obs[obs["OB"].notna()]
    for pos, row in present.iterrows():
        ob_type = "bullish" if row["OB"] > 0 else "bearish"
        state, mit, broke = ob_state(
            ob_type, float(row["Top"]), float(row["Bottom"]), pos, highs, lows, closes
        )
        out.append(
            OrderBlock(
                type=ob_type,
                top=float(row["Top"]),
                bottom=float(row["Bottom"]),
                volume=float(row["OBVolume"]),
                formed_at=timestamps[pos],
                state=state,
                mitigated_at=timestamps[mit] if mit is not None else None,
                broken_at=timestamps[broke] if broke is not None else None,
            )
        )
    return out
