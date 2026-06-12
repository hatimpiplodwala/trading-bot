"""ORB robustness checks: fill realism + opening-range window sweep.

Pressure-tests on the validated ORB edge, run on the SAME honest train/OOS split.
None of these *tune* anything — they measure how fragile the edge is to two
assumptions we should not trust blindly:

1. **Fill realism.** Headline ORB fills entries (and the flatten) at the signal
   bar's *close* — optimistic. ``next_open`` re-fills at the following bar's open
   (confirm the close beyond the range, then act). If the edge survives that, the
   edge is not a fill-timing artifact.
2. **Opening-range window.** We use 30 min. Sweep 15/30/60 min. If the edge only
   exists at exactly 30 min, it is a knife-edge curve-fit, not a real effect.

Run: ``uv run python scripts/harden_orb.py``
"""

from __future__ import annotations

import copy

from ict_bot.backtest.metrics import extract_metrics, oos_passes
from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.backtest.split import train_oos_split
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore

SYMBOLS = ["SPY", "QQQ"]
CASH = 100_000.0


def _row(label: str, m: dict) -> str:
    return (
        f"{label:<22}{m['profit_factor']:>7.2f}{m['total_return_pct']:>10.2f}"
        f"{m['sharpe']:>8.2f}{m['avg_r_multiple']:>8.2f}{m['win_rate_pct']:>8.1f}"
        f"{m['num_trades']:>7}  {'PASS' if oos_passes(m) else 'FAIL'}"
    )


def _header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print(
        f"{'window / mode':<22}{'PF':>7}{'Ret%':>10}{'Sharpe':>8}"
        f"{'avgR':>8}{'Win%':>8}{'N':>7}  Verdict"
    )


def main() -> None:
    settings = load_settings()
    tf = settings["entry_timeframe"]
    oos_months = settings["backtest"]["oos_months"]
    store = BarStore(REPO_ROOT / "data" / "parquet")

    splits = {}
    for s in SYMBOLS:
        entry = store.read_bars(s, tf)
        if len(entry) < 1000:
            print(f"{s}: insufficient data ({len(entry)} {tf} bars) — skipping")
            continue
        splits[s] = train_oos_split(entry, oos_months=oos_months)

    # Experiment 1 — fill realism (close vs next_open), IS then OOS per mode.
    for s, (train, oos) in splits.items():
        _header(f"[{s}] FILL REALISM")
        for mode in ("close", "next_open"):
            is_m = extract_metrics(run_orb_backtest(train, settings, cash=CASH, fill_mode=mode))
            oos_m = extract_metrics(run_orb_backtest(oos, settings, cash=CASH, fill_mode=mode))
            print(_row(f"{mode}  IS", is_m))
            print(_row(f"{mode}  OOS", oos_m))

    # Experiment 2 — opening-range window sweep (OOS, realistic next_open fills).
    windows = {"15m": "09:45", "30m": "10:00", "60m": "10:30"}
    for s, (_, oos) in splits.items():
        _header(f"[{s}] OR-WINDOW SWEEP  (OOS, next_open fills)")
        for name, end in windows.items():
            cfg = copy.deepcopy(settings)
            cfg["opening_range"]["end"] = end
            oos_m = extract_metrics(run_orb_backtest(oos, cfg, cash=CASH, fill_mode="next_open"))
            print(_row(f"OR={name} (->{end})", oos_m))


if __name__ == "__main__":
    main()
