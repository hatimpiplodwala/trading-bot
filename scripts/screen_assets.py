"""Cross-asset ORB screen: capital-split ensemble, walk-forward, diversified portfolio.

Runs the FIXED opening-range-window ensemble (no parameter selection) on each
candidate asset over the full 2022-2026 history (already out-of-sample, since
nothing is fitted), and:

  1. reports each asset's full-period metrics vs the OOS gate (PF>1.2, DD<15%),
  2. shows walk-forward stability (how many rolling windows each asset passes),
  3. prints the cross-asset correlation of daily returns,
  4. builds the equal-weight diversified portfolio of the PASSING assets and
     compares it to the current production bot (plain SPY single-window ORB).

We ship the live multi-asset executor only if the diversified portfolio beats
plain SPY-ORB on a risk-adjusted basis. Run: uv run python scripts/screen_assets.py
"""

from __future__ import annotations

import pandas as pd

from ict_bot.backtest.ensemble import DEFAULT_OR_WINDOWS, ensemble_backtest
from ict_bot.backtest.metrics import extract_metrics, oos_passes
from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.backtest.portfolio import combine_equity_curves, portfolio_metrics
from ict_bot.backtest.walkforward import walk_forward_ensemble
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore

UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
CASH = 100_000.0


def _fmt(label: str, m: dict) -> str:
    return (
        f"{label:<16}{m['profit_factor']:>7.2f}{m['total_return_pct']:>9.2f}"
        f"{m['sharpe']:>8.2f}{m['max_drawdown_pct']:>9.2f}{m['avg_r_multiple']:>8.2f}"
        f"{m['num_trades']:>7}  {'PASS' if oos_passes(m) else 'FAIL'}"
    )


def _hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print(f"{'asset':<16}{'PF':>7}{'Ret%':>9}{'Sharpe':>8}{'MaxDD%':>9}"
          f"{'avgR':>8}{'N':>7}  Verdict")


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("1D").last().dropna().pct_change().dropna()


def main() -> None:
    settings = load_settings()
    tf = settings["entry_timeframe"]
    store = BarStore(REPO_ROOT / "data" / "parquet")

    results: dict[str, dict] = {}
    for sym in UNIVERSE:
        df = store.read_bars(sym, tf)
        if len(df) < 2000:
            print(f"{sym}: insufficient data ({len(df)} bars) — skipping")
            continue
        results[sym] = ensemble_backtest(df, settings, DEFAULT_OR_WINDOWS, CASH)
        results[sym]["df"] = df

    # 1 — per-asset full-period ensemble
    _hdr(f"PER-ASSET ENSEMBLE (OR windows {DEFAULT_OR_WINDOWS}, next_open, 2022-2026)")
    for sym, r in results.items():
        print(_fmt(sym, r["metrics"]))

    # 2 — walk-forward stability (rolling 12-month windows, 6-month step)
    print("\n" + "=" * 78)
    print("WALK-FORWARD STABILITY (rolling 12mo / 6mo step — windows passing the gate)")
    for sym, r in results.items():
        wf = walk_forward_ensemble(r["df"], settings, DEFAULT_OR_WINDOWS, CASH)
        n_pass = sum(oos_passes(w["metrics"]) for w in wf)
        pfs = [w["metrics"]["profit_factor"] for w in wf]
        rng = f"PF {min(pfs):.2f}–{max(pfs):.2f}" if pfs else "n/a"
        print(f"  {sym:<6} {n_pass}/{len(wf)} windows pass   ({rng})")

    # 3 — cross-asset correlation of the ensemble's daily returns
    rets = pd.DataFrame({sym: _daily_returns(r["equity"]) for sym, r in results.items()})
    print("\n" + "=" * 78)
    print("CROSS-ASSET CORRELATION (ensemble daily returns)")
    print(rets.corr().round(2).to_string())

    # 4 — diversified portfolio of the PASSING assets vs plain SPY-ORB
    passers = [s for s, r in results.items() if oos_passes(r["metrics"])]
    print("\n" + "=" * 78)
    print(f"DIVERSIFIED PORTFOLIO  (equal-weight passers: {passers or 'NONE'})")
    _hdr_printed = False

    # baseline: the current production bot — SPY, single 30-min window
    spy_single = extract_metrics(
        run_orb_backtest(results["SPY"]["df"], settings, cash=CASH, fill_mode="next_open")
    )
    print(_fmt("SPY single (now)", spy_single))
    print(_fmt("SPY ensemble", results["SPY"]["metrics"]))

    if passers:
        m = len(passers)
        scaled = [results[s]["equity"] * (1.0 / m) for s in passers]  # equal-weight, normalized to CASH
        port_equity = combine_equity_curves(scaled)
        port_trades = pd.concat([results[s]["trades"] for s in passers], ignore_index=True)
        port = portfolio_metrics(port_equity, port_trades)
        print(_fmt(f"DIVERSIFIED ({m})", port))


if __name__ == "__main__":
    main()
