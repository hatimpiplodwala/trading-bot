"""Order block tests (Phase 2): active -> mitigated -> breaker state machine.

smc.ob locates order blocks; the lifecycle is ours and fully tested here. Per
bar after formation: the OB is *tapped* (mitigated) when price overlaps its zone,
and becomes a *breaker* when a bar CLOSES through the far side (below bottom for a
bullish OB, above top for a bearish OB).
"""

from __future__ import annotations

from ict_bot.ict.order_blocks import ob_state

# Bullish OB zone: bottom=100, top=105, formed at position 2.
TOP, BOTTOM, FORMED = 105.0, 100.0, 2


def test_active_when_untouched():
    highs = [0, 0, 110, 111, 112, 113]
    lows = [0, 0, 108, 109, 110, 111]
    closes = [0, 0, 109, 110, 111, 112]
    state, mit, broke = ob_state("bullish", TOP, BOTTOM, FORMED, highs, lows, closes)
    assert state == "active"
    assert mit is None and broke is None


def test_mitigated_when_tapped_but_not_closed_through():
    # Bar 4 dips into the zone (low 102) but closes back above bottom.
    highs = [0, 0, 110, 111, 106, 112]
    lows = [0, 0, 108, 109, 102, 110]
    closes = [0, 0, 109, 110, 106, 111]
    state, mit, broke = ob_state("bullish", TOP, BOTTOM, FORMED, highs, lows, closes)
    assert state == "mitigated"
    assert mit == 4 and broke is None


def test_breaker_when_bullish_ob_closes_below_bottom():
    # Bar 4 taps; bar 5 closes at 99 (< bottom) -> breaker.
    highs = [0, 0, 110, 111, 106, 104]
    lows = [0, 0, 108, 109, 102, 98]
    closes = [0, 0, 109, 110, 106, 99]
    state, mit, broke = ob_state("bullish", TOP, BOTTOM, FORMED, highs, lows, closes)
    assert state == "breaker"
    assert mit == 4 and broke == 5


def test_breaker_when_bearish_ob_closes_above_top():
    # Bearish OB: becomes breaker when a bar closes above top.
    highs = [0, 0, 104, 106, 0, 0]
    lows = [0, 0, 101, 103, 0, 0]
    closes = [0, 0, 103, 106, 0, 0]  # bar 3 closes 106 > top
    state, mit, broke = ob_state("bearish", TOP, BOTTOM, FORMED, highs, lows, closes)
    assert state == "breaker"
    assert broke == 3


def test_clean_break_without_prior_tap():
    # Price gaps entirely below the zone and closes there: broken, never tapped.
    highs = [0, 0, 110, 98, 97, 96]
    lows = [0, 0, 108, 95, 94, 93]
    closes = [0, 0, 109, 96, 95, 94]
    state, mit, broke = ob_state("bullish", TOP, BOTTOM, FORMED, highs, lows, closes)
    assert state == "breaker"
    assert mit is None and broke == 3
