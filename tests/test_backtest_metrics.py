"""Backtest metrics + report tests (Phase 5).

``extract_metrics`` maps a backtesting.py stats Series into our metric set
(including avg R-multiple, computed from per-trade PnL vs. entry-to-stop risk,
and trades/month from the run span). ``format_report`` renders in-sample and OOS
side by side and states the OOS verdict (PF > 1.2 and max DD < 15%).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ict_bot.backtest.metrics import extract_metrics, format_report, to_backtesting_frame


def _canonical():
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [10, 20, 30],
        },
        index=idx,
    )


def _fake_stats():
    trades = pd.DataFrame(
        {
            "PnL": [100.0, -50.0, 200.0],
            "EntryPrice": [100.0, 100.0, 100.0],
            "SL": [97.0, 97.0, 98.0],   # risk/share 3, 3, 2
            "Size": [10, 10, 10],
        }
    )
    return pd.Series(
        {
            "Return [%]": 5.0,
            "Win Rate [%]": 66.6667,
            "Profit Factor": 6.0,
            "Max. Drawdown [%]": -8.0,
            "Sharpe Ratio": 1.2,
            "# Trades": 3,
            "Start": pd.Timestamp("2023-01-01"),
            "End": pd.Timestamp("2023-07-01"),  # 181 days ~ 5.95 months
            "_trades": trades,
        }
    )


def test_to_backtesting_frame_capitalizes_columns():
    bt = to_backtesting_frame(_canonical())
    assert list(bt.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert bt["Close"].tolist() == [1.2, 2.2, 3.2]
    assert bt.index.equals(_canonical().index)


def test_extract_metrics_maps_core_fields():
    m = extract_metrics(_fake_stats())
    assert m["total_return_pct"] == pytest.approx(5.0)
    assert m["win_rate_pct"] == pytest.approx(66.6667)
    assert m["profit_factor"] == pytest.approx(6.0)
    assert m["max_drawdown_pct"] == pytest.approx(-8.0)
    assert m["sharpe"] == pytest.approx(1.2)
    assert m["num_trades"] == 3


def test_extract_metrics_avg_r_multiple():
    # R = PnL / (|Entry - SL| * Size): 100/30, -50/30, 200/20 -> mean ~ 3.889.
    m = extract_metrics(_fake_stats())
    assert m["avg_r_multiple"] == pytest.approx((100 / 30 - 50 / 30 + 200 / 20) / 3, rel=1e-6)


def test_extract_metrics_trades_per_month():
    m = extract_metrics(_fake_stats())
    assert m["trades_per_month"] == pytest.approx(3 / (181 / 30.44), rel=1e-3)


def test_extract_metrics_handles_no_trades():
    stats = _fake_stats()
    stats["_trades"] = pd.DataFrame(columns=["PnL", "EntryPrice", "SL", "Size"])
    stats["# Trades"] = 0
    m = extract_metrics(stats)
    assert m["num_trades"] == 0
    assert m["avg_r_multiple"] == 0.0
    assert m["trades_per_month"] == 0.0


def test_format_report_shows_both_windows_and_verdict():
    m = extract_metrics(_fake_stats())
    report = format_report(
        in_sample=m,
        oos=m,
        meta={"symbol": "SPY", "generated": "2026-06-12", "slippage_cents": 1},
    )
    assert "In-Sample" in report and "Out-of-Sample" in report
    assert "Profit Factor" in report and "Max" in report
    assert "SPY" in report
    # PF 6.0 > 1.2 and DD -8% (|8| < 15) -> OOS passes.
    assert "PASS" in report


def test_format_report_marks_oos_failure():
    good = extract_metrics(_fake_stats())
    bad = dict(good)
    bad["profit_factor"] = 0.9  # below 1.2 threshold
    report = format_report(
        in_sample=good,
        oos=bad,
        meta={"symbol": "SPY", "generated": "2026-06-12", "slippage_cents": 1},
    )
    assert "FAIL" in report
