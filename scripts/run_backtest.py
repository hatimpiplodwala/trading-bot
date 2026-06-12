"""Run the ICT backtest with a train / out-of-sample split and write a report.

Splits the stored SPY entry-timeframe history into an in-sample tuning window
and an untouched out-of-sample window (the verdict — Gotcha #6), runs the SAME
signal/risk engine the live loop uses over each, and writes an in-sample vs. OOS
metrics report to ``backtest/results/YYYY-MM-DD.md``.

Tuning rule: iterate confluence weights / ATR multiples against the in-sample
numbers only. The OOS run is the verdict, never the tuning target. PF > 1.2 and
max drawdown < 15% on OOS is the bar to clear.

Run: ``uv run python scripts/run_backtest.py``
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ict_bot.backtest.metrics import extract_metrics, format_report
from ict_bot.backtest.runner import run_backtest
from ict_bot.backtest.split import train_oos_split
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore

CASH = 100_000.0


def main() -> None:
    settings = load_settings()
    symbol = settings["symbol"]
    entry_tf = settings["entry_timeframe"]
    refs = settings["smt_reference_symbols"]
    oos_months = settings["backtest"]["oos_months"]

    store = BarStore(REPO_ROOT / "data" / "parquet")
    entry = store.read_bars(symbol, entry_tf)
    daily = store.read_bars(symbol, "1d")
    h1 = store.read_bars(symbol, "1h")
    references = {r: store.read_bars(r, entry_tf) for r in refs}

    if len(entry) < 1000 or len(daily) < 60 or len(h1) < 500:
        raise SystemExit(
            f"insufficient stored data for {symbol} "
            f"({len(entry)} {entry_tf}, {len(daily)} 1d, {len(h1)} 1h). "
            "Run scripts/download_history.py first."
        )

    train, oos = train_oos_split(entry, oos_months=oos_months)
    print(
        f"{symbol} {entry_tf}: {len(entry)} bars "
        f"({entry.index[0].date()} -> {entry.index[-1].date()})\n"
        f"  in-sample: {len(train)} bars ({train.index[0].date()} -> {train.index[-1].date()})\n"
        f"  out-of-sample: {len(oos)} bars ({oos.index[0].date()} -> {oos.index[-1].date()})"
    )

    print("Running in-sample backtest...")
    is_stats = run_backtest(train, daily, h1, references, settings, cash=CASH)
    print("Running out-of-sample backtest...")
    oos_stats = run_backtest(oos, daily, h1, references, settings, cash=CASH)

    is_metrics = extract_metrics(is_stats)
    oos_metrics = extract_metrics(oos_stats)

    generated = dt.date.today().isoformat()
    report = format_report(
        in_sample=is_metrics,
        oos=oos_metrics,
        meta={
            "symbol": symbol,
            "generated": generated,
            "slippage_cents": settings["backtest"]["slippage_cents"],
        },
    )

    out_dir = REPO_ROOT / "backtest" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{generated}.md"
    out_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
