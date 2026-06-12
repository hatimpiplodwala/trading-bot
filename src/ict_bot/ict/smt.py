"""SMT divergence (Phase 3, v1 core).

Detects correlation breaks between SPY (primary) and QQQ/IWM (references):

- **Bearish SMT**: SPY makes a higher high while a reference fails to confirm
  (equal or lower high).
- **Bullish SMT**: SPY makes a lower low while a reference fails to confirm
  (equal or higher low).

Requires time-aligned bars across symbols and inherits the IEX data-quality
caveat threefold (Gotcha #5). References are divergence input only — never
tradeable in v1 (Gotcha #8).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ict_bot.ict._smc import smc

Pair = tuple[float, float]  # (older pivot, newer pivot)


@dataclass(frozen=True)
class SMTSignal:
    """An SMT divergence (or absence of one)."""

    type: str | None  # "bullish" | "bearish" | None
    reference: str | None


def is_bearish_smt(primary_highs: Pair, ref_highs: Pair) -> bool:
    """Primary makes a higher high; reference fails to (equal/lower high)."""
    p_old, p_new = primary_highs
    r_old, r_new = ref_highs
    return p_new > p_old and r_new <= r_old


def is_bullish_smt(primary_lows: Pair, ref_lows: Pair) -> bool:
    """Primary makes a lower low; reference fails to (equal/higher low)."""
    p_old, p_new = primary_lows
    r_old, r_new = ref_lows
    return p_new < p_old and r_new >= r_old


def evaluate_smt(
    primary_highs: Pair | None,
    ref_highs: dict[str, Pair],
    primary_lows: Pair | None,
    ref_lows: dict[str, Pair],
) -> SMTSignal:
    """Return the first divergence found (highs checked before lows)."""
    if primary_highs is not None:
        for ref, rh in ref_highs.items():
            if is_bearish_smt(primary_highs, rh):
                return SMTSignal("bearish", ref)
    if primary_lows is not None:
        for ref, rl in ref_lows.items():
            if is_bullish_smt(primary_lows, rl):
                return SMTSignal("bullish", ref)
    return SMTSignal(None, None)


def _last_two_swings(
    ohlc: pd.DataFrame, swing_length: int, kind: str
) -> list[tuple[pd.Timestamp, float]] | None:
    positional = ohlc.reset_index(drop=True)
    swings = smc.swing_highs_lows(positional, swing_length=swing_length)
    target = 1 if kind == "high" else -1
    pts = swings[swings["HighLow"] == target]
    if len(pts) < 2:
        return None
    last2 = pts.tail(2)
    return [(ohlc.index[pos], float(row["Level"])) for pos, row in last2.iterrows()]


def detect_smt(
    primary_ohlc: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    swing_length: int = 50,
) -> SMTSignal:
    """Detect SMT divergence of ``primary_ohlc`` against time-aligned references.

    Reference highs/lows are read at the primary's swing timestamps; a reference
    missing those bars is skipped.
    """
    highs = _last_two_swings(primary_ohlc, swing_length, "high")
    lows = _last_two_swings(primary_ohlc, swing_length, "low")

    primary_highs = (highs[0][1], highs[1][1]) if highs else None
    primary_lows = (lows[0][1], lows[1][1]) if lows else None

    ref_highs: dict[str, Pair] = {}
    ref_lows: dict[str, Pair] = {}
    for name, ref in references.items():
        if highs:
            r = ref["high"].reindex([highs[0][0], highs[1][0]])
            if r.notna().all():
                ref_highs[name] = (float(r.iloc[0]), float(r.iloc[1]))
        if lows:
            r = ref["low"].reindex([lows[0][0], lows[1][0]])
            if r.notna().all():
                ref_lows[name] = (float(r.iloc[0]), float(r.iloc[1]))

    return evaluate_smt(primary_highs, ref_highs, primary_lows, ref_lows)
