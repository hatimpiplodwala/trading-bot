"""Risk & position sizing (Phase 4).

``position_size(equity, entry, stop, risk_pct)`` -> WHOLE shares (Alpaca rejects
bracket/OCO orders on fractional shares — Gotcha #9); round down and sanity-cap.
ATR14 via ``pandas_ta_classic.atr(length=14)`` with a manual Wilder fallback;
stop = entry +/- max(1.5 x ATR14, structural invalidation). Daily loss limit
halts new entries at -2% realized P&L per ET day. Max concurrent positions: 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


def position_size(
    equity: float,
    entry: float,
    stop: float,
    risk_pct: float,
    max_notional_pct: float = 1.0,
) -> int:
    """Whole-share size for a fixed-fractional risk budget.

    Shares = (equity * risk_pct) / per-share-risk, floored to an integer, then
    capped so notional never exceeds ``max_notional_pct`` of equity (this cap
    binds frequently on a ~$570 instrument with a 0.5% budget — without it the
    risk math implies heavy leverage). Returns 0 on invalid inputs or when the
    budget buys less than one share.
    """
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0 or entry <= 0 or equity <= 0:
        return 0
    risk_qty = math.floor((equity * risk_pct) / per_share_risk)
    notional_cap = math.floor((equity * max_notional_pct) / entry)
    return max(min(risk_qty, notional_cap), 0)


def _wilder_atr(ohlc: pd.DataFrame, length: int = 14) -> float:
    """Latest Wilder-smoothed ATR (manual fallback / reference implementation)."""
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder's smoothing == EMA with alpha = 1/length.
    atr = true_range.ewm(alpha=1 / length, adjust=False).mean()
    return float(atr.iloc[-1])


def compute_atr(ohlc: pd.DataFrame, length: int = 14) -> float:
    """Latest ATR via ``pandas_ta_classic``, falling back to the manual Wilder ATR.

    The dependency is unverified on pandas 3.0 across all paths, so any failure
    or all-NaN result degrades to ``_wilder_atr`` rather than killing the engine.
    """
    try:
        import pandas_ta_classic as ta

        series = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], length=length)
        if series is not None and series.notna().any():
            return float(series.dropna().iloc[-1])
    except Exception:
        pass
    return _wilder_atr(ohlc, length=length)


def stop_loss(
    entry: float,
    direction: str,
    atr: float,
    structural_level: float | None = None,
    atr_mult: float = 1.5,
) -> float:
    """Stop = entry +/- max(atr_mult * ATR, distance to structural invalidation)."""
    atr_dist = atr_mult * atr
    if direction == "long":
        struct_dist = (entry - structural_level) if structural_level is not None else 0.0
        return entry - max(atr_dist, struct_dist)
    struct_dist = (structural_level - entry) if structural_level is not None else 0.0
    return entry + max(atr_dist, struct_dist)


def take_profit(entry: float, stop: float, direction: str, rr: float = 2.0) -> float:
    """Target at ``rr`` times the entry-to-stop risk, in the trade's direction."""
    risk = abs(entry - stop)
    return entry + rr * risk if direction == "long" else entry - rr * risk


@dataclass
class DailyLossLimit:
    """Realized-P&L circuit breaker for a single ET trading day.

    Call :meth:`start_day` at each ET day boundary (resets the running tally and
    re-anchors the percentage to that day's opening equity), :meth:`register`
    on each realized close, and gate new entries on :meth:`allows_entry`.
    """

    limit_pct: float
    starting_equity: float = 0.0
    realized_pnl: float = 0.0

    def start_day(self, starting_equity: float) -> None:
        self.starting_equity = starting_equity
        self.realized_pnl = 0.0

    def register(self, pnl: float) -> None:
        self.realized_pnl += pnl

    def breached(self) -> bool:
        if self.starting_equity <= 0:
            return False
        return self.realized_pnl <= -self.limit_pct * self.starting_equity

    def allows_entry(self) -> bool:
        return not self.breached()
