"""Liquidity (Phase 2).

Wraps ``smc.liquidity()`` (BSL/SSL pools, sweeps) and adds equal-highs/lows
detection by clustering swing points whose prices sit within ``factor x ATR``
(default 0.1 x ATR) of one another. Buy-side liquidity rests at the top of equal
highs; sell-side at the bottom of equal lows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from ict_bot.ict._smc import smc

Point = tuple[Any, float]


@dataclass(frozen=True)
class EqualLevels:
    """A pool of equal highs or equal lows (relatively-equal swing prices)."""

    type: str  # "equal_highs" | "equal_lows"
    level: float
    count: int
    members: tuple[Any, ...]


def cluster_equal_levels(
    points: Sequence[Point], tolerance: float, min_count: int = 2
) -> list[list[Point]]:
    """Group points whose prices fall within ``tolerance`` (cluster span bound).

    Returns clusters of at least ``min_count`` members, ordered by price.
    """
    if not points:
        return []

    ordered = sorted(points, key=lambda p: p[1])
    clusters: list[list[Point]] = []
    current: list[Point] = [ordered[0]]
    for point in ordered[1:]:
        if point[1] - current[0][1] <= tolerance:
            current.append(point)
        else:
            clusters.append(current)
            current = [point]
    clusters.append(current)
    return [c for c in clusters if len(c) >= min_count]


def equal_highs_lows(
    swing_points: Sequence[tuple[Any, float, str]],
    atr: float,
    factor: float = 0.1,
) -> list[EqualLevels]:
    """Cluster swing highs and lows into equal-level liquidity pools.

    ``swing_points`` are ``(id, price, kind)`` with ``kind`` in {"high", "low"}.
    """
    tolerance = factor * atr
    highs = [(i, p) for i, p, kind in swing_points if kind == "high"]
    lows = [(i, p) for i, p, kind in swing_points if kind == "low"]

    out: list[EqualLevels] = []
    for cluster in cluster_equal_levels(highs, tolerance):
        out.append(
            EqualLevels(
                "equal_highs",
                level=max(p for _, p in cluster),
                count=len(cluster),
                members=tuple(i for i, _ in cluster),
            )
        )
    for cluster in cluster_equal_levels(lows, tolerance):
        out.append(
            EqualLevels(
                "equal_lows",
                level=min(p for _, p in cluster),
                count=len(cluster),
                members=tuple(i for i, _ in cluster),
            )
        )
    return out


def _atr(ohlc: pd.DataFrame, length: int) -> float:
    """Wilder-style ATR (SMA of true range) over ``length`` bars; last value."""
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(true_range.rolling(length).mean().iloc[-1])


def detect_equal_levels(
    ohlc: pd.DataFrame,
    swing_length: int = 50,
    atr_length: int = 14,
    factor: float = 0.1,
) -> list[EqualLevels]:
    """Detect equal-highs/lows liquidity pools in an OHLC window."""
    positional = ohlc.reset_index(drop=True)
    swings = smc.swing_highs_lows(positional, swing_length=swing_length)
    present = swings[swings["HighLow"].notna()]
    swing_points = [
        (ohlc.index[pos], float(row["Level"]), "high" if row["HighLow"] > 0 else "low")
        for pos, row in present.iterrows()
    ]
    return equal_highs_lows(swing_points, _atr(ohlc, atr_length), factor)


def detect_liquidity(ohlc: pd.DataFrame, swing_length: int = 50) -> pd.DataFrame:
    """Raw smc liquidity (BSL/SSL pools + sweeps) for the window."""
    positional = ohlc.reset_index(drop=True)
    swings = smc.swing_highs_lows(positional, swing_length=swing_length)
    return smc.liquidity(positional, swings)
