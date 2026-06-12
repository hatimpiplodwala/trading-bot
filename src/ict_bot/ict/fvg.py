"""Fair Value Gaps (Phase 2).

Wraps ``smc.fvg(join_consecutive=True)`` and maps its positional output to
timestamped :class:`FVG` dataclasses with mitigation state. smc encodes an
unmitigated gap as ``MitigatedIndex`` 0 (or NaN); a positive integer is the bar
position at which the gap was filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ict_bot.ict._smc import smc


@dataclass(frozen=True)
class FVG:
    """A fair value gap and its mitigation state."""

    type: str  # "bullish" (FVG=1) | "bearish" (FVG=-1)
    top: float
    bottom: float
    formed_at: pd.Timestamp
    mitigated: bool
    mitigated_at: pd.Timestamp | None


def fvgs_from_frame(fvg_df: pd.DataFrame, timestamps: pd.DatetimeIndex) -> list[FVG]:
    """Convert an smc ``fvg`` frame (positional) into timestamped FVGs."""
    out: list[FVG] = []
    present = fvg_df[fvg_df["FVG"].notna()]
    for pos, row in present.iterrows():
        mit = row["MitigatedIndex"]
        mitigated = pd.notna(mit) and mit != 0
        out.append(
            FVG(
                type="bullish" if row["FVG"] > 0 else "bearish",
                top=float(row["Top"]),
                bottom=float(row["Bottom"]),
                formed_at=timestamps[pos],
                mitigated=bool(mitigated),
                mitigated_at=timestamps[int(mit)] if mitigated else None,
            )
        )
    return out


def detect_fvgs(ohlc: pd.DataFrame) -> list[FVG]:
    """Detect FVGs in an open-time-indexed OHLC window."""
    timestamps = ohlc.index
    positional = ohlc.reset_index(drop=True)
    frame = smc.fvg(positional, join_consecutive=True)
    return fvgs_from_frame(frame, timestamps)
