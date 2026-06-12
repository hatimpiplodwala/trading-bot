"""SMT divergence tests (Phase 3).

Bearish SMT: the primary (SPY) makes a higher high while a correlated index
fails to confirm (equal/lower high). Bullish SMT: primary makes a lower low
while the reference fails to confirm (equal/higher low). Swing extraction +
cross-symbol lookup is covered by the integration smoke; the rule is tested here.
"""

from __future__ import annotations

from ict_bot.ict.smt import SMTSignal, evaluate_smt, is_bearish_smt, is_bullish_smt


def test_bearish_smt_when_reference_fails_higher_high():
    assert is_bearish_smt((100.0, 105.0), (100.0, 99.0))   # ref lower high
    assert is_bearish_smt((100.0, 105.0), (100.0, 100.0))  # ref equal high (fails to confirm)


def test_no_bearish_smt_when_both_make_higher_high():
    assert not is_bearish_smt((100.0, 105.0), (100.0, 106.0))


def test_no_bearish_smt_when_primary_not_higher_high():
    assert not is_bearish_smt((105.0, 104.0), (100.0, 99.0))


def test_bullish_smt_when_reference_fails_lower_low():
    assert is_bullish_smt((100.0, 95.0), (100.0, 101.0))   # ref higher low
    assert is_bullish_smt((100.0, 95.0), (100.0, 100.0))   # ref equal low


def test_no_bullish_smt_when_both_make_lower_low():
    assert not is_bullish_smt((100.0, 95.0), (100.0, 94.0))


def test_evaluate_picks_diverging_reference():
    # SPY higher high; QQQ confirms (HH) but IWM diverges (LH) -> bearish via IWM.
    signal = evaluate_smt(
        primary_highs=(100.0, 105.0),
        ref_highs={"QQQ": (100.0, 106.0), "IWM": (100.0, 99.0)},
        primary_lows=(90.0, 88.0),
        ref_lows={"QQQ": (90.0, 87.0), "IWM": (90.0, 86.0)},
    )
    assert signal == SMTSignal(type="bearish", reference="IWM")


def test_evaluate_returns_none_when_all_confirm():
    signal = evaluate_smt(
        primary_highs=(100.0, 105.0),
        ref_highs={"QQQ": (100.0, 106.0)},
        primary_lows=(90.0, 88.0),
        ref_lows={"QQQ": (90.0, 87.0)},
    )
    assert signal.type is None
    assert signal.reference is None


def test_evaluate_detects_bullish_low_divergence():
    signal = evaluate_smt(
        primary_highs=(100.0, 101.0),
        ref_highs={"QQQ": (100.0, 102.0)},
        primary_lows=(90.0, 85.0),          # SPY lower low
        ref_lows={"QQQ": (90.0, 92.0)},     # QQQ higher low -> bullish SMT
    )
    assert signal == SMTSignal(type="bullish", reference="QQQ")
