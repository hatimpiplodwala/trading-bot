"""Structural-stop and liquidity-target tests (structural rework, Approach A).

The stop sits at the nearest swing *against* the trade (swing low below a long /
swing high above a short); the target is the nearest opposing liquidity pool
*beyond* entry (swing high above a long / swing low below a short). The pure
nearest-on-side selection is unit-tested here; the smc swing extraction is
covered end-to-end by the real-data integration test.
"""

from __future__ import annotations

import pytest

from ict_bot.config import REPO_ROOT
from ict_bot.data.store import BarStore
from ict_bot.strategy.exits import (
    _nearest_on_side,
    liquidity_target,
    structural_stop_level,
)


def test_nearest_below_picks_highest_below_entry():
    assert _nearest_on_side([90.0, 95.0, 98.0, 101.0], entry=100.0, side="below") == 98.0


def test_nearest_above_picks_lowest_above_entry():
    assert _nearest_on_side([90.0, 101.0, 103.0, 110.0], entry=100.0, side="above") == 101.0


def test_nearest_on_side_none_when_empty_side():
    assert _nearest_on_side([101.0, 102.0], entry=100.0, side="below") is None
    assert _nearest_on_side([90.0, 99.0], entry=100.0, side="above") is None
    assert _nearest_on_side([], entry=100.0, side="below") is None


def test_nearest_on_side_excludes_entry_level():
    # A level exactly at entry is neither above nor below -> excluded.
    assert _nearest_on_side([100.0, 97.0], entry=100.0, side="below") == 97.0


@pytest.fixture(scope="module")
def spy15():
    df = BarStore(REPO_ROOT / "data" / "parquet").read_bars("SPY", "15m")
    if len(df) < 500:
        pytest.skip("no local SPY 15m data")
    return df


def test_structural_stop_below_entry_for_long(spy15):
    window = spy15.tail(300)
    entry = float(window["close"].iloc[-1])
    stop = structural_stop_level(window, "long", entry, swing_length=20)
    if stop is not None:
        assert stop < entry  # a long's structural invalidation sits below entry


def test_structural_stop_above_entry_for_short(spy15):
    window = spy15.tail(300)
    entry = float(window["close"].iloc[-1])
    stop = structural_stop_level(window, "short", entry, swing_length=20)
    if stop is not None:
        assert stop > entry


def test_liquidity_target_beyond_entry(spy15):
    window = spy15.tail(300)
    entry = float(window["close"].iloc[-1])
    long_t = liquidity_target(window, "long", entry, swing_length=5, lookback=40)
    short_t = liquidity_target(window, "short", entry, swing_length=5, lookback=40)
    if long_t is not None:
        assert long_t > entry   # buy-side liquidity is above
    if short_t is not None:
        assert short_t < entry  # sell-side liquidity is below
