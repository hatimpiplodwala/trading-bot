"""Portfolio aggregation: summing sub-account equity curves + portfolio metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from ict_bot.backtest.portfolio import combine_equity_curves, portfolio_metrics


def _idx(n):
    return pd.date_range("2022-01-03", periods=n, freq="D", tz="UTC")


def test_combine_equity_curves_sums_subaccounts():
    idx = _idx(2)
    a = pd.Series([50_000.0, 51_000.0], index=idx)  # +1000
    b = pd.Series([50_000.0, 49_500.0], index=idx)  # -500
    combined = combine_equity_curves([a, b])
    assert combined.iloc[0] == 100_000.0
    assert combined.iloc[-1] == 100_500.0


def test_combine_aligns_and_ffills_mismatched_indexes():
    a = pd.Series([50_000.0, 51_000.0], index=_idx(2))
    b = pd.Series([50_000.0, 52_000.0, 53_000.0], index=_idx(3))
    combined = combine_equity_curves([a, b])
    # `a` is forward-filled to the third timestamp (51_000 + 53_000).
    assert combined.iloc[-1] == 104_000.0


def test_portfolio_metrics_return_drawdown_and_trades():
    eq = pd.Series(
        [100_000, 101_000, 100_000, 102_000, 103_200],
        index=_idx(5), dtype=float,
    )
    trades = pd.DataFrame({
        "PnL": [500.0, -200.0, 700.0],
        "EntryPrice": [100.0, 100.0, 100.0],
        "SL": [99.0, 99.0, 99.0],
        "Size": [10, 10, 10],
    })
    m = portfolio_metrics(eq, trades)
    assert m["total_return_pct"] == pytest.approx(3.2)
    assert m["max_drawdown_pct"] < 0                      # dipped to 100k from 101k
    assert m["num_trades"] == 3
    assert m["profit_factor"] == pytest.approx(1200 / 200)  # gains 1200, losses 200
    assert m["win_rate_pct"] == pytest.approx(2 / 3 * 100)
