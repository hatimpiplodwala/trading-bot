"""Run the Opening Range Breakout backtest with a train/OOS split, SPY + QQQ.

Same honest pipeline as the ICT backtest: tune nothing here, just measure the
strategy on the earlier ~18 months (in-sample) and the untouched ~6 months
(out-of-sample). Writes a report per symbol to ``backtest/results/`` and prints a
side-by-side OOS comparison.

Run: ``uv run python scripts/run_orb_backtest.py``
"""

from __future__ import annotations

import datetime as dt

from ict_bot.backtest.metrics import extract_metrics, format_report, oos_passes
from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.backtest.split import train_oos_split
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore

CASH = 100_000.0
SYMBOLS = ["SPY", "QQQ"]


def main() -> None:
    settings = load_settings()
    tf = settings["entry_timeframe"]
    oos_months = settings["backtest"]["oos_months"]
    store = BarStore(REPO_ROOT / "data" / "parquet")
    out_dir = REPO_ROOT / "backtest" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = dt.date.today().isoformat()

    summary = []
    for symbol in SYMBOLS:
        entry = store.read_bars(symbol, tf)
        if len(entry) < 1000:
            print(f"{symbol}: insufficient data ({len(entry)} {tf} bars) — skipping")
            continue

        train, oos = train_oos_split(entry, oos_months=oos_months)
        print(f"{symbol} {tf}: in-sample {len(train)} bars, OOS {len(oos)} bars — running...")
        is_metrics = extract_metrics(run_orb_backtest(train, settings, cash=CASH))
        oos_metrics = extract_metrics(run_orb_backtest(oos, settings, cash=CASH))

        report = format_report(
            is_metrics,
            oos_metrics,
            meta={"symbol": f"{symbol} (ORB)", "generated": generated,
                  "slippage_cents": settings["backtest"]["slippage_cents"]},
        )
        (out_dir / f"orb-{symbol}-{generated}.md").write_text(report, encoding="utf-8")
        print("\n" + report + "\n")
        summary.append((symbol, is_metrics, oos_metrics))

    if summary:
        print("=" * 60)
        print(f"{'OOS summary':<14}{'PF':>8}{'Return%':>10}{'Sharpe':>9}{'Trades':>8}  Verdict")
        for symbol, _, oos in summary:
            print(f"{symbol:<14}{oos['profit_factor']:>8.2f}{oos['total_return_pct']:>10.2f}"
                  f"{oos['sharpe']:>9.2f}{oos['num_trades']:>8}  "
                  f"{'PASS' if oos_passes(oos) else 'FAIL'}")


if __name__ == "__main__":
    main()
