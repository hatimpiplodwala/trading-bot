"""Signal generation (Phase 4).

On each closed bar: run detectors -> check confluences -> if score >= 60 and risk
checks pass, emit a ``CandidateSignal`` dataclass. The exact same code path is
reused by the backtest harness (Phase 5) — no separate live/backtest logic.

This module is split into two seams:

* ``SignalEngine.evaluate`` — the pure decision: score a ``SetupContext``, apply
  the daily loss gate, size the position, and emit or suppress. Fully covered by
  unit tests.
* the detector -> ``SetupContext`` builder (``build_setup_context``) — wires the
  ICT detectors into a context for the current bar. Covered by the real-data
  integration smoke, since it depends on smc detector output.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ict_bot.ict.bias import htf_bias
from ict_bot.ict.fvg import detect_fvgs
from ict_bot.ict.order_blocks import detect_order_blocks
from ict_bot.ict.sessions import Sessions
from ict_bot.ict.smt import detect_smt
from ict_bot.ict.zones import DealingRange
from ict_bot.strategy.confluence import (
    Confluences,
    meets_threshold,
    score_confluences,
)
from ict_bot.strategy.risk import (
    DailyLossLimit,
    compute_atr,
    position_size,
    stop_loss,
    take_profit,
)

# Map the directional bias to a trade side and to the smc convention used by
# the FVG/OB/zone detectors ("bullish"/"bearish").
_SIDE_TO_SMC = {"long": "bullish", "short": "bearish"}


@dataclass(frozen=True)
class SetupContext:
    """Everything the engine needs to judge one candidate bar.

    ``direction`` is the HTF-bias-proposed side ("long" | "short" | "neutral");
    ``confluences`` records which scored factors are present at this bar.
    """

    timestamp: pd.Timestamp
    direction: str
    entry: float
    atr: float
    structural_level: float | None
    confluences: Confluences


@dataclass(frozen=True)
class CandidateSignal:
    """A risk-sized trade candidate ready for the broker layer (Phase 7)."""

    timestamp: pd.Timestamp
    direction: str
    entry: float
    stop: float
    target: float
    qty: int
    score: int
    confluences: Confluences


class SignalEngine:
    """Scores setups and emits sized candidates, sharing one code path live and
    in backtest."""

    def __init__(
        self,
        weights: dict[str, int],
        min_score: int,
        risk_pct: float = 0.005,
        atr_mult: float = 1.5,
        rr: float = 2.0,
        max_notional_pct: float = 1.0,
        daily_gate: DailyLossLimit | None = None,
    ) -> None:
        self.weights = weights
        self.min_score = min_score
        self.risk_pct = risk_pct
        self.atr_mult = atr_mult
        self.rr = rr
        self.max_notional_pct = max_notional_pct
        # A default un-started gate allows entries (starting_equity 0 -> never breached).
        self.daily_gate = daily_gate if daily_gate is not None else DailyLossLimit(limit_pct=1.0)

    @classmethod
    def from_settings(cls, settings: dict, daily_gate: DailyLossLimit | None = None) -> "SignalEngine":
        signal = settings["signal"]
        risk = settings["risk"]
        return cls(
            weights=signal["weights"],
            min_score=signal["min_confluence_score"],
            risk_pct=risk["risk_per_trade"],
            atr_mult=risk["atr_stop_mult"],
            daily_gate=daily_gate,
        )

    def evaluate(self, ctx: SetupContext, equity: float) -> CandidateSignal | None:
        """Emit a sized candidate, or ``None`` if any gate fails."""
        if ctx.direction not in ("long", "short"):
            return None

        score = score_confluences(ctx.confluences, self.weights)
        if not meets_threshold(score, self.min_score):
            return None

        if not self.daily_gate.allows_entry():
            return None

        stop = stop_loss(
            ctx.entry, ctx.direction, ctx.atr, ctx.structural_level, self.atr_mult
        )
        qty = position_size(
            equity, ctx.entry, stop, self.risk_pct, self.max_notional_pct
        )
        if qty <= 0:
            return None

        target = take_profit(ctx.entry, stop, ctx.direction, self.rr)
        return CandidateSignal(
            timestamp=ctx.timestamp,
            direction=ctx.direction,
            entry=ctx.entry,
            stop=stop,
            target=target,
            qty=qty,
            score=score,
            confluences=ctx.confluences,
        )


# --------------------------------------------------------------------------- #
# Pure builder helpers                                                         #
# --------------------------------------------------------------------------- #

def direction_from_bias(bias: str) -> str:
    """Trade side proposed by the HTF bias."""
    return {"bullish": "long", "bearish": "short"}.get(bias, "neutral")


def killzone_confluences(kill_zone: str | None) -> tuple[bool, bool]:
    """``(silver_bullet, ny_kill_zone)`` flags for the current kill zone.

    Silver Bullet (10:00-11:00 ET) is nested inside NY AM, so it lights both.
    NY AM/PM light the kill-zone flag only. Pre-market is observe-only (Gotcha
    #9) and outside any window lights neither.
    """
    if kill_zone == "silver_bullet":
        return (True, True)
    if kill_zone in ("ny_am", "ny_pm"):
        return (False, True)
    return (False, False)


def ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two ``(low, high)`` price ranges intersect (touching counts)."""
    return a[0] <= b[1] and b[0] <= a[1]


# --------------------------------------------------------------------------- #
# Detector -> SetupContext builder (integration-smoke tested on real data)     #
# --------------------------------------------------------------------------- #

def build_setup_context(
    *,
    timestamp: pd.Timestamp,
    entry_tf: pd.DataFrame,
    daily: pd.DataFrame,
    h1: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    sessions: Sessions,
    swing_length: int = 20,
    range_lookback: int = 60,
    entry_lookback: int = 300,
    h1_lookback: int = 1500,
    daily_lookback: int = 250,
) -> SetupContext | None:
    """Assemble a :class:`SetupContext` for the current bar from the detectors.

    Returns ``None`` when there is no directional HTF bias (nothing to trade) or
    the entry window is too short to form a dealing range. All frames must end at
    the bar being decided — only closed bars, no lookahead.

    Detectors run on bounded *rolling* tails of each frame (``entry_lookback`` /
    ``h1_lookback`` / ``daily_lookback``): this mirrors how the live engine keeps
    a recent window rather than rescanning years of history, keeps stale FVGs/OBs
    out of scope, and makes a bar-by-bar replay tractable. The per-confluence
    checks are deliberately modest v1 heuristics; their exact ICT fidelity is
    validated by the Phase 5 backtest, not asserted here.
    """
    bias = htf_bias(
        daily.tail(daily_lookback), h1.tail(h1_lookback), swing_length=swing_length
    ).bias
    direction = direction_from_bias(bias)
    if direction == "neutral":
        return None

    entry_window = entry_tf.tail(entry_lookback)
    range_window = entry_window.tail(range_lookback)
    if len(range_window) < 6:
        return None

    entry = float(entry_window["close"].iloc[-1])
    smc_dir = _SIDE_TO_SMC[direction]

    dr = DealingRange(
        low=float(range_window["low"].min()), high=float(range_window["high"].max())
    )
    if not dr.low < dr.high:
        return None

    # Kill zone.
    kz = sessions.current_kill_zone(timestamp)
    silver_bullet, ny_kill_zone = killzone_confluences(kz)

    # Premium/discount + OTE, relative to trade direction.
    discount_premium = dr.in_discount(entry) if direction == "long" else dr.in_premium(entry)
    ote = dr.in_ote(entry, smc_dir)

    # Active FVG overlapping an active OB, both on the trade side.
    fvgs = [f for f in detect_fvgs(entry_window) if not f.mitigated and f.type == smc_dir]
    obs = [
        o
        for o in detect_order_blocks(entry_window, swing_length=swing_length)
        if o.state == "active" and o.type == smc_dir
    ]
    fvg_ob_overlap = any(
        ranges_overlap((f.bottom, f.top), (o.bottom, o.top)) for f in fvgs for o in obs
    )

    # SMT divergence supporting the trade direction.
    smt = detect_smt(entry_window, references, swing_length=swing_length)
    smt_divergence = smt.type == smc_dir

    # Liquidity sweep: a recent bar wicked beyond the prior extreme and closed
    # back inside (a stop raid on the side we're fading).
    liquidity_sweep = _swept_liquidity(range_window, direction)

    confluences = Confluences(
        htf_aligned=True,  # direction is non-neutral, i.e. HTF agrees by construction
        silver_bullet=silver_bullet,
        ny_kill_zone=ny_kill_zone,
        discount_premium=discount_premium,
        ote=ote,
        fvg_ob_overlap=fvg_ob_overlap,
        smt_divergence=smt_divergence,
        liquidity_sweep=liquidity_sweep,
    )

    structural_level = dr.low if direction == "long" else dr.high
    atr = compute_atr(entry_window)

    return SetupContext(
        timestamp=timestamp,
        direction=direction,
        entry=entry,
        atr=atr,
        structural_level=structural_level,
        confluences=confluences,
    )


def _swept_liquidity(window: pd.DataFrame, direction: str) -> bool:
    """Recent close-back-inside after raiding the *prior* extreme of the window.

    The prior extreme (everything but the last 3 bars) is the resting liquidity;
    a wick beyond it that closes back inside is the sweep we want to fade.
    """
    if len(window) < 6:
        return False
    prior, recent = window.iloc[:-3], window.tail(3)
    if direction == "long":
        prior_low = float(prior["low"].min())
        return bool((recent["low"] < prior_low).any() and float(recent["close"].iloc[-1]) > prior_low)
    prior_high = float(prior["high"].max())
    return bool((recent["high"] > prior_high).any() and float(recent["close"].iloc[-1]) < prior_high)
