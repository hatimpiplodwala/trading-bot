"""Liquidity tests (Phase 2): equal-highs/lows clustering within 0.1x ATR.

smc.liquidity supplies BSL/SSL pools and sweeps; our addition is clustering
swing points whose prices sit within a tolerance (0.1 x ATR) into equal-high /
equal-low liquidity pools. A cluster's span is bounded by the tolerance.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.ict.liquidity import EqualLevels, cluster_equal_levels, equal_highs_lows


def _pts(prices: list[float]) -> list[tuple[int, float]]:
    return [(i, p) for i, p in enumerate(prices)]


def test_clusters_three_close_levels():
    clusters = cluster_equal_levels(_pts([100.0, 100.05, 100.08]), tolerance=0.1)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_splits_when_span_exceeds_tolerance():
    # 100.0/100.05 are within 0.1; 100.5 is far -> one pair, singleton dropped.
    clusters = cluster_equal_levels(_pts([100.0, 100.05, 100.5]), tolerance=0.1)
    assert len(clusters) == 1
    assert {round(p, 2) for _, p in clusters[0]} == {100.0, 100.05}


def test_no_cluster_when_all_separated():
    clusters = cluster_equal_levels(_pts([100.0, 100.2, 100.4]), tolerance=0.1)
    assert clusters == []


def test_span_bounded_not_chained():
    # Drift 100.0->100.08->100.16: 100.16 - 100.0 = 0.16 > 0.1, so it must NOT
    # all chain into one cluster.
    clusters = cluster_equal_levels(_pts([100.0, 100.08, 100.16]), tolerance=0.1)
    assert all((max(p for _, p in c) - min(p for _, p in c)) <= 0.1 for c in clusters)


def test_equal_highs_lows_typed_and_leveled():
    ts = pd.date_range("2026-06-12", periods=4, freq="1D", tz="UTC")
    swing_points = [
        (ts[0], 100.00, "high"),
        (ts[1], 100.05, "high"),
        (ts[2], 90.00, "low"),
        (ts[3], 90.04, "low"),
    ]
    levels = equal_highs_lows(swing_points, atr=1.0, factor=0.1)  # tolerance 0.1

    by_type = {l.type: l for l in levels}
    assert set(by_type) == {"equal_highs", "equal_lows"}
    # Buy-side liquidity rests at the top of equal highs; sell-side at the bottom.
    assert by_type["equal_highs"].level == 100.05
    assert by_type["equal_lows"].level == 90.00
    assert by_type["equal_highs"].count == 2


def test_equal_levels_requires_two_points():
    levels = equal_highs_lows([(0, 100.0, "high")], atr=1.0, factor=0.1)
    assert levels == []
