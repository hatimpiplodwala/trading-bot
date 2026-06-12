"""Backtest metrics extraction & report rendering (Phase 5).

Pure functions over a ``backtesting.py`` stats Series — no backtesting import, so
they are unit-testable with a synthetic stats object. The OOS verdict gate is
PF > 1.2 and max drawdown < 15% (in-sample alone proves nothing).
"""

from __future__ import annotations

import pandas as pd

_DAYS_PER_MONTH = 30.44

# Out-of-sample acceptance thresholds (PRD Phase 5).
PF_MIN = 1.2
MAX_DD_PCT = 15.0


def to_backtesting_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical OHLCV -> backtesting.py's capitalized column names."""
    return df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )[["Open", "High", "Low", "Close", "Volume"]]


def _avg_r_multiple(trades: pd.DataFrame) -> float:
    """Mean R-multiple = PnL / (entry-to-stop risk * size), over closed trades."""
    if trades is None or len(trades) == 0:
        return 0.0
    risk_per_share = (trades["EntryPrice"] - trades["SL"]).abs()
    risk = risk_per_share * trades["Size"].abs()
    valid = risk > 0
    if not valid.any():
        return 0.0
    r = trades.loc[valid, "PnL"] / risk[valid]
    return float(r.mean())


def extract_metrics(stats: pd.Series) -> dict:
    """Map a backtesting.py stats Series into our reporting metric set."""
    trades = stats._trades
    num_trades = int(stats["# Trades"])

    span_days = (stats["End"] - stats["Start"]).days
    months = span_days / _DAYS_PER_MONTH if span_days > 0 else 0.0
    trades_per_month = (num_trades / months) if months > 0 else 0.0

    return {
        "total_return_pct": float(stats["Return [%]"]),
        "win_rate_pct": float(stats["Win Rate [%]"]) if num_trades else 0.0,
        "profit_factor": float(stats["Profit Factor"]),
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "avg_r_multiple": _avg_r_multiple(trades),
        "num_trades": num_trades,
        "trades_per_month": trades_per_month,
    }


def oos_passes(metrics: dict) -> bool:
    """OOS verdict: PF > 1.2 and max drawdown shallower than 15%."""
    pf = metrics["profit_factor"]
    dd = abs(metrics["max_drawdown_pct"])
    return pd.notna(pf) and pf > PF_MIN and dd < MAX_DD_PCT


_ROWS = [
    ("Total return [%]", "total_return_pct", "{:.2f}"),
    ("Win rate [%]", "win_rate_pct", "{:.2f}"),
    ("Profit Factor", "profit_factor", "{:.2f}"),
    ("Max. Drawdown [%]", "max_drawdown_pct", "{:.2f}"),
    ("Sharpe", "sharpe", "{:.2f}"),
    ("Avg R-multiple", "avg_r_multiple", "{:.2f}"),
    ("# Trades", "num_trades", "{:.0f}"),
    ("Trades/month", "trades_per_month", "{:.2f}"),
]


def _cell(metrics: dict, key: str, fmt: str) -> str:
    value = metrics[key]
    if isinstance(value, float) and pd.isna(value):
        return "n/a"
    return fmt.format(value)


def format_report(in_sample: dict, oos: dict, meta: dict) -> str:
    """Render an in-sample vs. out-of-sample metrics report as markdown."""
    verdict = "PASS" if oos_passes(oos) else "FAIL"
    lines = [
        f"# Backtest report — {meta['symbol']}",
        "",
        f"Generated: {meta['generated']}  ·  Slippage: {meta.get('slippage_cents', 0)}c/share",
        "",
        "| Metric | In-Sample | Out-of-Sample |",
        "| --- | ---: | ---: |",
    ]
    for label, key, fmt in _ROWS:
        lines.append(f"| {label} | {_cell(in_sample, key, fmt)} | {_cell(oos, key, fmt)} |")
    lines += [
        "",
        f"**OOS verdict: {verdict}** "
        f"(gate: Profit Factor > {PF_MIN} and max drawdown < {MAX_DD_PCT:.0f}%).",
        "",
        "_In-sample results are the tuning surface and prove nothing on their own; "
        "the out-of-sample window is the verdict (Gotcha #6)._",
    ]
    return "\n".join(lines)
