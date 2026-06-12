"""Risk & position-sizing tests (Phase 4).

Position size is risk-budget / per-share-risk, floored to whole shares (Alpaca
rejects fractional bracket orders, Gotcha #9) and capped by a notional fraction
of equity — the latter binds often on a ~$570 instrument with a 0.5% budget.
Stops are entry +/- max(1.5*ATR14, structural distance). The daily loss limit
halts new entries once realized P&L crosses -2% of the day's starting equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ict_bot.strategy.risk import (
    DailyLossLimit,
    compute_atr,
    position_size,
    stop_loss,
    take_profit,
    _wilder_atr,
)


def _flat_frame(n: int = 40, span: float = 2.0) -> pd.DataFrame:
    """Bars with constant true range == span and no gaps -> ATR converges to span."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    mid = 100.0
    return pd.DataFrame(
        {
            "open": np.full(n, mid),
            "high": np.full(n, mid + span / 2),
            "low": np.full(n, mid - span / 2),
            "close": np.full(n, mid),
            "volume": np.full(n, 1000),
        },
        index=idx,
    )


# --- position sizing ---

def test_position_size_floors_to_whole_shares():
    # risk budget 500, per-share risk 3 -> 166.67 -> 166 whole shares.
    qty = position_size(equity=100_000, entry=100.0, stop=97.0, risk_pct=0.005)
    assert qty == 166


def test_position_size_notional_cap_binds():
    # Risk-based qty would be 333 (500 / 1.5), but 333 * 570 = ~$190k > equity.
    # Capped to floor(100000 / 570) = 175 shares.
    qty = position_size(equity=100_000, entry=570.0, stop=568.5, risk_pct=0.005)
    assert qty == 175


def test_position_size_zero_when_rounds_below_one_share():
    qty = position_size(equity=1_000, entry=100.0, stop=90.0, risk_pct=0.005)
    assert qty == 0


def test_position_size_zero_on_invalid_inputs():
    assert position_size(equity=100_000, entry=100.0, stop=100.0, risk_pct=0.005) == 0
    assert position_size(equity=0, entry=100.0, stop=95.0, risk_pct=0.005) == 0


# --- ATR ---

def test_wilder_atr_converges_to_constant_true_range():
    atr = _wilder_atr(_flat_frame(), length=14)
    assert atr == pytest.approx(2.0, abs=1e-9)


def test_compute_atr_matches_manual_on_flat_frame():
    df = _flat_frame()
    assert compute_atr(df, length=14) == pytest.approx(_wilder_atr(df, length=14), rel=1e-6)


# --- stops ---

def test_stop_long_atr_dominated():
    # 1.5 * ATR(2) = 3 wider than the 1-pt structural distance.
    assert stop_loss(entry=100.0, direction="long", atr=2.0, structural_level=99.0) == 97.0


def test_stop_long_structural_dominated():
    # Structural invalidation 5 pts below beats 1.5 * ATR(2) = 3.
    assert stop_loss(entry=100.0, direction="long", atr=2.0, structural_level=95.0) == 95.0


def test_stop_short_mirrors_long():
    assert stop_loss(entry=100.0, direction="short", atr=2.0, structural_level=101.0) == 103.0
    assert stop_loss(entry=100.0, direction="short", atr=2.0, structural_level=106.0) == 106.0


# --- targets ---

def test_take_profit_is_r_multiple_of_risk():
    # Long: risk = 100 - 97 = 3; 2R target = 100 + 6 = 106.
    assert take_profit(entry=100.0, stop=97.0, direction="long", rr=2.0) == 106.0
    # Short: risk = 103 - 100 = 3; 2R target = 100 - 6 = 94.
    assert take_profit(entry=100.0, stop=103.0, direction="short", rr=2.0) == 94.0


# --- daily loss limit ---

def test_daily_loss_limit_halts_after_losing_streak():
    gate = DailyLossLimit(limit_pct=0.02)
    gate.start_day(starting_equity=100_000)
    assert gate.allows_entry()
    for _ in range(3):
        gate.register(-700.0)  # cumulative -2100 = -2.1% > 2% limit after 3
    assert gate.breached()
    assert not gate.allows_entry()


def test_daily_loss_limit_allows_within_budget():
    gate = DailyLossLimit(limit_pct=0.02)
    gate.start_day(starting_equity=100_000)
    gate.register(-1_500.0)  # -1.5% < 2% limit
    assert not gate.breached()
    assert gate.allows_entry()


def test_start_day_resets_realized_pnl():
    gate = DailyLossLimit(limit_pct=0.02)
    gate.start_day(starting_equity=100_000)
    gate.register(-5_000.0)
    assert gate.breached()
    gate.start_day(starting_equity=95_000)  # new ET day
    assert not gate.breached()
    assert gate.allows_entry()
