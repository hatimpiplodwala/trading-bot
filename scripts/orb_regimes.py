"""ORB across market regimes — per-calendar-year, realistic next_open fills.

Our headline OOS window (late-2025..2026) is all bull market. This slices the
full 2022-2026 history by calendar year and runs the SAME untuned ORB on each, to
see whether the edge holds in the 2022 bear/chop. Regime attribution, not tuning:
no parameter is chosen from these results.

Run: ``uv run python scripts/orb_regimes.py``
"""

from __future__ import annotations

from ict_bot.backtest.metrics import extract_metrics, oos_passes
from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore

SYMBOLS = ["SPY", "QQQ"]
CASH = 100_000.0


def _row(label: str, m: dict) -> str:
    return (
        f"{label:<10}{m['profit_factor']:>7.2f}{m['total_return_pct']:>9.2f}"
        f"{m['sharpe']:>8.2f}{m['avg_r_multiple']:>8.2f}{m['win_rate_pct']:>7.1f}"
        f"{m['num_trades']:>6}  {'PASS' if oos_passes(m) else 'FAIL'}"
    )


def main() -> None:
    settings = load_settings()
    tf = settings["entry_timeframe"]
    store = BarStore(REPO_ROOT / "data" / "parquet")

    for s in SYMBOLS:
        df = store.read_bars(s, tf)
        print("\n" + "=" * 72)
        print(f"[{s}] ORB BY CALENDAR YEAR  (next_open fills, untuned)")
        print(
            f"{'year':<10}{'PF':>7}{'Ret%':>9}{'Sharpe':>8}{'avgR':>8}"
            f"{'Win%':>7}{'N':>6}  Verdict"
        )
        for y in sorted(set(df.index.year)):
            seg = df[df.index.year == y]
            if len(seg) < 200:
                print(f"{y:<10}  (only {len(seg)} bars — skipped)")
                continue
            m = extract_metrics(run_orb_backtest(seg, settings, cash=CASH, fill_mode="next_open"))
            print(_row(str(y), m))


if __name__ == "__main__":
    main()
