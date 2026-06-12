"""Signal engine decision-logic tests (Phase 4).

The engine turns a SetupContext (direction + price geometry + which confluences
are present) into a CandidateSignal, or suppresses it. A signal is emitted only
when the confluence score clears the threshold, the daily loss limit allows a new
entry, and the risk-sized quantity is at least one whole share. Stop/target/qty
geometry is delegated to the risk module (tested there); here we test the gating
and the wiring. Detector -> context assembly is covered by the integration smoke.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.strategy.confluence import Confluences
from ict_bot.strategy.risk import DailyLossLimit
from ict_bot.strategy.signal import (
    CandidateSignal,
    SetupContext,
    SignalEngine,
    build_setup_context,
    direction_from_bias,
    killzone_confluences,
    ranges_overlap,
    resolve_direction,
)

WEIGHTS = {
    "htf_aligned": 25,
    "silver_bullet": 20,
    "ny_kill_zone": 10,
    "discount_premium": 15,
    "ote": 10,
    "fvg_ob_overlap": 15,
    "smt_divergence": 10,
    "liquidity_sweep": 15,
}

TS = pd.Timestamp("2024-03-01 15:00", tz="UTC")


def _engine(daily_gate=None, min_target_rr=1.0, fallback_rr=None) -> SignalEngine:
    return SignalEngine(
        weights=WEIGHTS,
        min_score=60,
        risk_pct=0.005,
        atr_floor_mult=0.5,
        min_target_rr=min_target_rr,
        fallback_rr=fallback_rr,
        daily_gate=daily_gate,
    )


def _ctx(
    confluences: Confluences, direction: str = "long", target: float | None = 106.0
) -> SetupContext:
    # entry 100, structural 95 -> stop 95 (5-pt risk); target 106 -> 1.2R.
    return SetupContext(
        timestamp=TS,
        direction=direction,
        entry=100.0,
        atr=2.0,
        structural_level=95.0,
        target=target,
        confluences=confluences,
    )


# htf_aligned(25) + silver_bullet(20) + ny_kill_zone(10) + fvg_ob_overlap(15) = 70
STRONG = Confluences(htf_aligned=True, silver_bullet=True, ny_kill_zone=True, fvg_ob_overlap=True)
# htf_aligned(25) + ny_kill_zone(10) = 35
WEAK = Confluences(htf_aligned=True, ny_kill_zone=True)


def test_emits_candidate_using_liquidity_target():
    sig = _engine().evaluate(_ctx(STRONG), equity=100_000)
    assert isinstance(sig, CandidateSignal)
    assert sig.direction == "long"
    assert sig.score == 70
    assert sig.entry == 100.0
    assert sig.stop == 95.0          # structural 5 below beats 0.5*ATR floor (1)
    assert sig.target == 106.0       # the liquidity target carried in the context
    assert sig.qty == 100            # floor(500 / 5)
    assert sig.timestamp == TS


def test_skips_when_liquidity_target_too_close():
    # target 102 -> 0.4R < min_target_rr (1.0), no fallback -> no trade.
    assert _engine().evaluate(_ctx(STRONG, target=102.0), equity=100_000) is None


def test_uses_fallback_rr_when_no_liquidity_target():
    sig = _engine(fallback_rr=2.0).evaluate(_ctx(STRONG, target=None), equity=100_000)
    assert sig is not None
    assert sig.target == 110.0       # 2R of the 5-pt risk via fallback


def test_skips_when_no_target_and_no_fallback():
    assert _engine().evaluate(_ctx(STRONG, target=None), equity=100_000) is None


def test_suppresses_below_threshold():
    assert _engine().evaluate(_ctx(WEAK), equity=100_000) is None


def test_suppresses_when_daily_limit_breached():
    gate = DailyLossLimit(limit_pct=0.02)
    gate.start_day(starting_equity=100_000)
    gate.register(-3_000.0)  # -3% > 2% limit
    assert _engine(daily_gate=gate).evaluate(_ctx(STRONG), equity=100_000) is None


def test_suppresses_when_size_rounds_to_zero():
    # Tiny equity -> sub-one-share -> no trade even with a strong score.
    assert _engine().evaluate(_ctx(STRONG), equity=500) is None


def test_suppresses_when_direction_neutral():
    assert _engine().evaluate(_ctx(STRONG, direction="neutral"), equity=100_000) is None


def test_losing_streak_halts_engine_entries():
    # Acceptance: the daily loss limit halts new entries after a losing streak.
    gate = DailyLossLimit(limit_pct=0.02)
    gate.start_day(starting_equity=100_000)
    engine = _engine(daily_gate=gate)
    assert engine.evaluate(_ctx(STRONG), equity=100_000) is not None  # first entry ok
    gate.register(-1_200.0)
    gate.register(-1_200.0)  # cumulative -2.4% > 2% limit
    assert engine.evaluate(_ctx(STRONG), equity=100_000) is None      # halted


def test_from_settings_builds_consistent_engine():
    from ict_bot.config import load_settings

    settings = load_settings()
    engine = SignalEngine.from_settings(settings)
    # Wiring check (tune-proof): threshold and weights come straight from settings.
    assert engine.min_score == settings["signal"]["min_confluence_score"]
    assert engine.weights["htf_aligned"] == 25
    # A full-confluence context (score 100) clears any threshold <= 100.
    everything = Confluences(
        htf_aligned=True, silver_bullet=True, ny_kill_zone=True, discount_premium=True,
        ote=True, fvg_ob_overlap=True, smt_divergence=True, liquidity_sweep=True,
    )
    sig = engine.evaluate(_ctx(everything), equity=100_000)
    assert sig is not None and sig.score == 100


# --- pure builder helpers ---

def test_direction_from_bias():
    assert direction_from_bias("bullish") == "long"
    assert direction_from_bias("bearish") == "short"
    assert direction_from_bias("neutral") == "neutral"


def test_killzone_confluences():
    # Silver bullet sits inside NY AM -> both flags; AM/PM -> kill zone only;
    # premarket is observe-only (Gotcha #9) -> neither; outside -> neither.
    assert killzone_confluences("silver_bullet") == (True, True)
    assert killzone_confluences("ny_am") == (False, True)
    assert killzone_confluences("ny_pm") == (False, True)
    assert killzone_confluences("ny_premarket") == (False, False)
    assert killzone_confluences(None) == (False, False)


def test_ranges_overlap():
    assert ranges_overlap((100.0, 102.0), (101.0, 103.0))     # partial
    assert ranges_overlap((100.0, 105.0), (101.0, 102.0))     # contained
    assert ranges_overlap((100.0, 101.0), (101.0, 102.0))     # touching edge
    assert not ranges_overlap((100.0, 101.0), (102.0, 103.0)) # disjoint


def test_injected_neutral_direction_short_circuits():
    # An injected "neutral" direction returns None before any detector runs, so
    # the HTF frames are never consulted (caching path for the backtest).
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ctx = build_setup_context(
        timestamp=TS,
        entry_tf=empty,
        daily=empty,
        h1=empty,
        references={},
        sessions=None,
        direction="neutral",
    )
    assert ctx is None
