"""Capital-split ensemble runner (real SPY 15m data; skips if absent)."""

from __future__ import annotations

import pandas as pd
import pytest

from ict_bot.backtest.ensemble import DEFAULT_OR_WINDOWS, ensemble_backtest
from ict_bot.backtest.metrics import extract_metrics
from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore


@pytest.fixture(scope="module")
def spy():
    df = BarStore(REPO_ROOT / "data" / "parquet").read_bars("SPY", "15m")
    if len(df) < 2000:
        pytest.skip("insufficient local SPY 15m data")
    return df[df.index >= df.index[-1] - pd.Timedelta(days=150)]


def test_single_window_ensemble_matches_direct_run(spy):
    settings = load_settings()
    res = ensemble_backtest(spy, settings, or_windows=["10:00"], cash=100_000.0, fill_mode="next_open")
    direct = extract_metrics(run_orb_backtest(spy, settings, cash=100_000.0, fill_mode="next_open"))
    # One variant at full capital is exactly the plain ORB run.
    assert res["metrics"]["num_trades"] == direct["num_trades"]


def test_three_window_ensemble_aggregates(spy):
    settings = load_settings()
    res = ensemble_backtest(spy, settings, or_windows=DEFAULT_OR_WINDOWS, cash=100_000.0)
    m = res["metrics"]
    assert {"total_return_pct", "profit_factor", "sharpe", "max_drawdown_pct", "num_trades"} <= set(m)
    assert res["equity"].iloc[0] == pytest.approx(100_000.0, rel=1e-6)  # sub-accounts sum to cash
    # The ensemble takes trades from each window, so it sees more than any single one.
    one = ensemble_backtest(spy, settings, or_windows=["10:00"], cash=100_000.0)
    assert m["num_trades"] >= one["metrics"]["num_trades"]
