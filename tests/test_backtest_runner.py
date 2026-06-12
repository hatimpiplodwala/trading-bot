"""Backtest runner tests (Phase 5).

Includes the explicit lookahead-bias check the PRD mandates (task 2): at every
step the strategy sees only bars up to the current one — the visible last close
always equals the canonical close at that visible length, never a future bar.
The end-to-end smoke runs the real ICT engine through backtesting.py on stored
SPY data and confirms a clean stats object comes back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from backtesting import Backtest, Strategy

from ict_bot.backtest.metrics import extract_metrics, to_backtesting_frame
from ict_bot.backtest.runner import run_backtest
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore


def _synthetic(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="5min", tz="UTC")
    idx.name = "timestamp"
    close = pd.Series(100.0 + np.arange(n), index=idx)  # distinct, increasing closes
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )


def test_next_sees_only_data_up_to_current_bar():
    frame = _synthetic(50)

    class Recorder(Strategy):
        def init(self):
            self.records = []

        def next(self):
            self.records.append((len(self.data), float(self.data.Close[-1])))

    stats = Backtest(to_backtesting_frame(frame), Recorder, cash=100_000).run()
    records = stats._strategy.records

    canonical_close = frame["close"].to_numpy()
    # At every visible length L, the last visible close is canonical[L-1] — i.e.
    # no future bar has leaked into the window.
    for length, seen_close in records:
        assert seen_close == pytest.approx(canonical_close[length - 1])
    # Lengths advance by exactly one bar and reach the end of the frame.
    lengths = [length for length, _ in records]
    assert lengths == list(range(lengths[0], len(frame) + 1))
    assert lengths[-1] == len(frame)


@pytest.fixture(scope="module")
def store():
    return BarStore(REPO_ROOT / "data" / "parquet")


def test_run_backtest_smoke(store):
    spy15 = store.read_bars("SPY", "15m")
    qqq15 = store.read_bars("QQQ", "15m")
    iwm15 = store.read_bars("IWM", "15m")
    daily = store.read_bars("SPY", "1d")
    h1 = store.read_bars("SPY", "1h")
    if min(len(spy15), len(qqq15), len(iwm15)) < 1000 or len(daily) < 60 or len(h1) < 500:
        pytest.skip("insufficient local data for backtest smoke")

    # A short recent window keeps the smoke fast; HTF/reference frames stay full.
    start = spy15.index[-1] - pd.Timedelta(days=25)
    segment = spy15[spy15.index >= start]
    settings = load_settings()

    stats = run_backtest(
        entry_segment=segment,
        daily=daily,
        h1=h1,
        references={"QQQ": qqq15, "IWM": iwm15},
        settings=settings,
        cash=100_000.0,
    )

    assert "# Trades" in stats.index
    assert np.isfinite(stats["Equity Final [$]"])
    metrics = extract_metrics(stats)
    assert metrics["num_trades"] >= 0
    assert "profit_factor" in metrics
