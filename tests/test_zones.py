"""Premium/discount zone tests (Phase 2) — pure dealing-range math."""

from __future__ import annotations

import pytest

from ict_bot.ict.zones import DealingRange


@pytest.fixture
def dr() -> DealingRange:
    # Range low=100, high=200 -> size 100, equilibrium 150.
    return DealingRange(low=100.0, high=200.0)


def test_equilibrium_and_size(dr):
    assert dr.equilibrium == 150.0
    assert dr.size == 100.0


def test_level_at(dr):
    assert dr.level_at(0.0) == 100.0
    assert dr.level_at(1.0) == 200.0
    assert dr.level_at(0.62) == pytest.approx(162.0)


def test_position(dr):
    assert dr.position(150.0) == 0.5
    assert dr.position(125.0) == 0.25


def test_zone_classification(dr):
    assert dr.zone(160.0) == "premium"
    assert dr.zone(140.0) == "discount"
    assert dr.zone(150.0) == "equilibrium"


def test_in_premium_discount(dr):
    assert dr.in_discount(140.0) and not dr.in_premium(140.0)
    assert dr.in_premium(160.0) and not dr.in_discount(160.0)


def test_ote_bullish_band_is_deep_discount(dr):
    # 62-79% retracement of an up-move -> position 0.38 down to 0.21.
    lo, hi = dr.ote("bullish")
    assert (lo, hi) == pytest.approx((121.0, 138.0))
    assert dr.in_ote(130.0, "bullish")
    assert not dr.in_ote(165.0, "bullish")


def test_ote_bearish_band_is_premium(dr):
    lo, hi = dr.ote("bearish")
    assert (lo, hi) == pytest.approx((162.0, 179.0))
    assert dr.in_ote(170.0, "bearish")
    assert not dr.in_ote(130.0, "bearish")


def test_rejects_invalid_range():
    with pytest.raises(ValueError):
        DealingRange(low=200.0, high=100.0)  # low must be < high
