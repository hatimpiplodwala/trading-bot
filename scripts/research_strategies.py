"""Strategy research harness — hunt for a higher-Sharpe edge beyond ORB.

Tests genuinely *different* strategy families (not ORB re-tuning, which is
exhausted) under the same OOS discipline that killed ICT: tune params on the
in-sample window only, judge on the untouched out-of-sample window, gate on
PF>1.2 & DD<15% (``oos_passes``), rank survivors by OOS Sharpe.

Families:
  Benchmarks  B1 buy-and-hold SPY · B2 200-DMA-timed SPY · B3 current ORB
  Multi-day   M1 trend/momentum · M2 RSI-2 mean-reversion · M3 overnight drift
              · M4 cross-asset dual-momentum rotation
  Intraday    I1 VWAP reversion · I2 opening-gap fade  (added separately)

Every daily family reduces to a daily strategy-return series; metrics come from
the SAME pipeline as ORB (``portfolio_metrics`` + ``oos_passes``), so the verdict
is apples-to-apples. Run:
    uv run python scripts/research_strategies.py
    uv run python scripts/research_strategies.py --oos-months 24
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from ict_bot.backtest.metrics import extract_metrics, oos_passes
from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.backtest.portfolio import portfolio_metrics
from ict_bot.config import load_settings
from ict_bot.data.resample import resample_ohlcv
from ict_bot.data.session import regular_hours
from ict_bot.data.store import BarStore
from ict_bot.strategy.rsi2 import rsi

UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
CAPITAL = 10_000.0


# --- data -----------------------------------------------------------------
def load_daily(store: BarStore, symbol: str) -> pd.DataFrame:
    """RTH daily OHLC from on-disk 15m bars (ponytail: '1440min' dodges the
    pandas-3 ``Timedelta(Day)`` break in resample_ohlcv for the '1D' alias;
    RTH never crosses UTC midnight so a 1440-min bucket == one trading day)."""
    b15 = store.read_bars(symbol, "15m")
    if b15.empty:
        return b15
    return resample_ohlcv(regular_hours(b15, "09:30", "16:00"), "1440min")


# --- evaluation core ------------------------------------------------------
def _metrics(strat_r: pd.Series, active: pd.Series, price: pd.Series, capital: float) -> dict:
    """Equity + per-active-day trades -> the SAME metric dict ORB reports.

    Daily families have no per-trade stop, so each in-market day is one 'bet':
    PF/win-rate come from daily P&L, Sharpe/DD from the equity curve. avg_R is
    meaningless without a stop (SL==Entry -> 0)."""
    strat_r = strat_r.fillna(0.0)
    equity = capital * (1 + strat_r).cumprod()
    pnl = equity.diff()
    in_mkt = active.reindex(pnl.index).fillna(0.0).astype(bool)
    bet = pnl[in_mkt].dropna()
    px = price.reindex(bet.index)
    trades = pd.DataFrame({"PnL": bet.values, "EntryPrice": px.values,
                           "SL": px.values, "Size": 1.0})
    return portfolio_metrics(equity, trades)


def strat_returns(price: pd.Series, position: pd.Series,
                  ret: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Daily strat-return + in-market flag. ``position`` is the weight decided at
    each bar's close — shifted so the return it earns is the NEXT bar's (no
    lookahead)."""
    r = (price.pct_change() if ret is None else ret)
    pos_prev = position.shift(1).reindex(r.index).fillna(0.0)
    return pos_prev * r, (pos_prev != 0).astype(float)


def evaluate(price: pd.Series, position: pd.Series, oos_months: int,
             capital: float = CAPITAL, ret: pd.Series | None = None) -> tuple[dict, dict]:
    """Split by calendar months; (in-sample, OOS) metric dicts."""
    strat_r, active = strat_returns(price, position, ret)
    return evaluate_series(strat_r, active, price, oos_months, capital)


# --- strategy position series ---------------------------------------------
def pos_buy_hold(px: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=px.index)


def pos_dma(px: pd.Series, n: int = 200) -> pd.Series:
    return (px > px.rolling(n).mean()).astype(float)


def pos_momentum(px: pd.Series, n: int) -> pd.Series:
    """Long while above the N-day SMA (time-series momentum / trend filter)."""
    return (px > px.rolling(n).mean()).astype(float)


def pos_rsi2(px: pd.Series, entry_t: float, exit_t: float = 60.0) -> pd.Series:
    """Connors-style: buy oversold pullbacks in an uptrend, exit on recovery."""
    r2 = rsi(px, 2)
    sma200 = px.rolling(200).mean()
    sma5 = px.rolling(5).mean()
    enter = (r2 < entry_t) & (px > sma200)
    leave = (r2 > exit_t) | (px > sma5)
    pos = np.zeros(len(px))
    holding = False
    e, lv = enter.to_numpy(), leave.to_numpy()
    for i in range(len(px)):
        holding = (holding and not lv[i]) or ((not holding) and e[i])
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=px.index)


def overnight_return(daily: pd.DataFrame) -> pd.Series:
    """Gap return: today's open / yesterday's close - 1 (the overnight hold)."""
    return (daily["open"] / daily["close"].shift(1) - 1.0)


def rsi2_returns(d: pd.DataFrame, entry_t: float, fill: str = "close",
                 slip_bps: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """Event-driven RSI-2 daily returns under a realistic fill model.

    ``close``    — act at the signal bar's close (≈ market-on-close order).
    ``next_open`` — act at the OPEN of the day after the signal (pessimistic; the
                    overnight bounce after an oversold close is given up on entry).
    ``slip_bps`` charges a per-side cost on the entry and exit days.
    Returns (daily strat return, in-market flag), both aligned to the realized day.
    """
    close, open_ = d["close"], d["open"]
    r2, sma200, sma5 = rsi(close, 2), close.rolling(200).mean(), close.rolling(5).mean()
    enter = ((r2 < entry_t) & (close > sma200)).to_numpy()
    leave = ((r2 > 60) | (close > sma5)).to_numpy()
    c, o, n = close.to_numpy(), open_.to_numpy(), len(close)
    slip = slip_bps / 1e4
    ret, active = np.zeros(n), np.zeros(n)
    holding = False
    for t in range(1, n):
        if fill == "close":
            if holding:
                ret[t] = c[t] / c[t - 1] - 1.0
                active[t] = 1.0
            if holding and leave[t]:
                holding = False
                ret[t] -= slip
            elif (not holding) and enter[t]:
                holding = True
                ret[t] -= slip
        else:  # next_open
            if holding and leave[t - 1]:        # signalled at close[t-1] -> exit open[t]
                ret[t] = o[t] / c[t - 1] - 1.0 - slip
                active[t], holding = 1.0, False
            elif (not holding) and enter[t - 1]:  # signalled at close[t-1] -> enter open[t]
                ret[t] = c[t] / o[t] - 1.0 - slip
                active[t], holding = 1.0, True
            elif holding:
                ret[t] = c[t] / c[t - 1] - 1.0
                active[t] = 1.0
    return pd.Series(ret, index=close.index), pd.Series(active, index=close.index)


# --- param selection on TRAIN only ----------------------------------------
def _train_sharpe(price: pd.Series, position: pd.Series, oos_months: int,
                  ret: pd.Series | None = None) -> float:
    is_m, _ = evaluate(price, position, oos_months, ret=ret)
    return is_m["sharpe"]


def best_param(price: pd.Series, make_pos, grid, oos_months: int,
               ret: pd.Series | None = None):
    """Pick the param maximizing IN-SAMPLE Sharpe (never touches OOS)."""
    scored = [(p, _train_sharpe(price, make_pos(p), oos_months, ret)) for p in grid]
    return max(scored, key=lambda kv: kv[1])[0], scored


# --- cross-asset dual momentum (M4) ---------------------------------------
def rotation_return(closes: pd.DataFrame, lookback: int, top_n: int) -> tuple[pd.Series, pd.Series]:
    """Monthly hold the top-N assets by trailing ``lookback``-day return, only
    those with positive absolute momentum (else that slot sits in cash).
    Returns (daily strat return, daily 'in market' flag)."""
    rets = closes.pct_change()
    mom = closes.pct_change(lookback)
    month_id = closes.index.tz_localize(None).to_period("M")
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    last_month = None
    cur = pd.Series(0.0, index=closes.columns)
    for i, ts in enumerate(closes.index):
        if month_id[i] != last_month:  # rebalance on the first day of each month
            last_month = month_id[i]
            m = mom.iloc[i]
            ranked = m[m > 0].sort_values(ascending=False)
            picks = list(ranked.index[:top_n])
            cur = pd.Series(0.0, index=closes.columns)
            if picks:
                cur[picks] = 1.0 / top_n  # equal weight; empty slots -> cash
        weights.iloc[i] = cur.values
    strat_r = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    active = weights.shift(1).fillna(0.0).sum(axis=1) > 0  # aligned with strat_r
    return strat_r, active.astype(float)


def evaluate_series(strat_r: pd.Series, active: pd.Series, price: pd.Series,
                    oos_months: int, capital: float = CAPITAL) -> tuple[dict, dict]:
    """(in-sample, OOS) metrics for a prebuilt daily strat-return series.
    ``active`` is the in-market flag already aligned with ``strat_r`` (the day the
    return is realized)."""
    cutoff = strat_r.index.max() - pd.DateOffset(months=oos_months)
    is_mask, oos_mask = strat_r.index <= cutoff, strat_r.index > cutoff
    act = active.reindex(strat_r.index).fillna(0.0)
    return (_metrics(strat_r[is_mask], act[is_mask], price, capital),
            _metrics(strat_r[oos_mask], act[oos_mask], price, capital))


def per_year(strat_r: pd.Series, active: pd.Series, price: pd.Series,
             capital: float = CAPITAL) -> dict[int, dict]:
    """Per-calendar-year metrics — the regime-stability check (bear/chop/bull)."""
    act = active.reindex(strat_r.index).fillna(0.0)
    out = {}
    for y in sorted(set(strat_r.index.year)):
        m = strat_r.index.year == y
        if m.sum() >= 20:
            out[y] = _metrics(strat_r[m], act[m], price, capital)
    return out


# --- ORB benchmark (B3) ---------------------------------------------------
def orb_oos_metrics(store: BarStore, settings: dict, oos_months: int) -> dict:
    b15 = store.read_bars("SPY", "15m")
    cutoff = b15.index.max() - pd.DateOffset(months=oos_months)
    oos = b15[b15.index > cutoff]
    stats = run_orb_backtest(oos, settings, cash=CAPITAL, fill_mode="next_open")
    return extract_metrics(stats)


# --- reporting ------------------------------------------------------------
def _row(name: str, horizon: str, is_m: dict, oos_m: dict, note: str = "") -> dict:
    gate = "PASS" if oos_passes(oos_m) else "fail"
    return {"strategy": name, "horizon": horizon, "is_sharpe": is_m["sharpe"],
            "oos_sharpe": oos_m["sharpe"], "oos_ret%": oos_m["total_return_pct"],
            "oos_pf": oos_m["profit_factor"], "oos_dd%": oos_m["max_drawdown_pct"],
            "oos_win%": oos_m["win_rate_pct"], "gate": gate, "note": note}


def validate_fills(daily: dict, entry_t: int, oos_months: int) -> None:
    """Re-test the M2 winner under realistic execution: close (~market-on-close)
    vs next-open (pessimistic), with/without per-side slippage. The #1 live risk —
    RSI-2 acts on the oversold close and part of the bounce is the overnight gap."""
    spy = daily["SPY"]["close"]
    legs = ["SPY", "QQQ", "IWM"]
    print(f"\n=== M2 RSI-2 fill validation (entry<{entry_t}, OOS last {oos_months}mo) ===")
    print("close ~ market-on-close;  next_open = enter/exit the morning AFTER the "
          "signal (gives up the overnight bounce).")
    print("Sanity: 'SPY close slip0' should reproduce the vectorized OOS Sharpe ~1.43.\n")
    head = f"{'config':<30}{'IS Sh':>7}{'OOS Sh':>8}{'PF':>7}{'DD%':>7}{'ret%':>8}  gate"
    print(head)
    print("-" * len(head))

    def show(name: str, sr: pd.Series, ac: pd.Series) -> None:
        is_m, oos_m = evaluate_series(sr, ac, spy, oos_months)
        g = "PASS" if oos_passes(oos_m) else "fail"
        print(f"{name:<30}{is_m['sharpe']:>7.2f}{oos_m['sharpe']:>8.2f}"
              f"{oos_m['profit_factor']:>7.2f}{oos_m['max_drawdown_pct']:>7.1f}"
              f"{oos_m['total_return_pct']:>8.1f}  {g}")

    def basket(fill: str, slip: float) -> tuple[pd.Series, pd.Series]:
        srs, acs = zip(*(rsi2_returns(daily[s], entry_t, fill, slip) for s in legs))
        return (pd.concat(srs, axis=1).mean(axis=1),
                (pd.concat(acs, axis=1).sum(axis=1) > 0).astype(float))

    for fill in ("close", "next_open"):
        for slip in (0.0, 5.0):
            sr, ac = rsi2_returns(daily["SPY"], entry_t, fill, slip)
            show(f"SPY    {fill:<9} slip{slip:.0f}bps", sr, ac)
            br, ba = basket(fill, slip)
            show(f"basket {fill:<9} slip{slip:.0f}bps", br, ba)
        print()

    br, ba = basket("next_open", 5.0)
    py = per_year(br, ba, spy)
    cells = "  ".join(f"{y}:Sh{m['sharpe']:+.2f}/PF{m['profit_factor']:.2f}"
                      for y, m in py.items())
    print(f"Worst-case (next_open + 5bps) basket per-year:\n  {cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strategy research vs. ORB (OOS-honest).")
    ap.add_argument("--oos-months", type=int, default=18)
    ap.add_argument("--fills", action="store_true",
                    help="run only the M2 fill-realism validation and exit")
    args = ap.parse_args()
    om = args.oos_months

    settings = load_settings()
    store = BarStore("data/parquet")
    daily = {s: load_daily(store, s) for s in UNIVERSE}
    spy = daily["SPY"]["close"]
    closes = pd.DataFrame({s: daily[s]["close"] for s in UNIVERSE}).dropna()

    if args.fills:
        validate_fills(daily, entry_t=10, oos_months=om)
        return

    rows = []

    # Benchmarks
    rows.append(_row("B1 buy-hold SPY", "hold", *evaluate(spy, pos_buy_hold(spy), om)))
    rows.append(_row("B2 200DMA SPY", "swing", *evaluate(spy, pos_dma(spy, 200), om)))
    try:
        orb = orb_oos_metrics(store, settings, om)
        rows.append({"strategy": "B3 ORB SPY", "horizon": "intraday",
                     "is_sharpe": float("nan"), "oos_sharpe": orb["sharpe"],
                     "oos_ret%": orb["total_return_pct"], "oos_pf": orb["profit_factor"],
                     "oos_dd%": orb["max_drawdown_pct"], "oos_win%": orb["win_rate_pct"],
                     "gate": "PASS" if oos_passes(orb) else "fail", "note": "incumbent"})
    except Exception as e:  # noqa: BLE001 — research script, surface and continue
        print(f"  ORB benchmark failed: {e}")

    # M1 momentum — tune SMA length on train
    n, _ = best_param(spy, lambda p: pos_momentum(spy, p), [50, 100, 150, 200], om)
    rows.append(_row(f"M1 momentum SMA{n} SPY", "swing", *evaluate(spy, pos_momentum(spy, n), om),
                     note=f"SMA={n} (train-picked)"))

    # M2 RSI-2 mean-reversion — tune entry threshold on train
    t, _ = best_param(spy, lambda p: pos_rsi2(spy, p), [5, 10, 15, 20], om)
    rows.append(_row(f"M2 RSI2<{t} SPY", "swing", *evaluate(spy, pos_rsi2(spy, t), om),
                     note=f"entry={t} (train-picked)"))

    # M3 overnight drift — parameter-free; whole period OOS-clean
    onr = overnight_return(daily["SPY"])
    rows.append(_row("M3 overnight SPY", "overnight",
                     *evaluate(spy, pos_buy_hold(spy), om, ret=onr)))
    onr_pos = pos_dma(spy, 200)  # variant: only overnight-hold in uptrends
    rows.append(_row("M3 overnight>200DMA", "overnight",
                     *evaluate(spy, onr_pos, om, ret=onr), note="trend-filtered"))

    # M4 cross-asset dual-momentum rotation — tune lookback & top-N on train
    grid = [(lb, n) for lb in (63, 126, 252) for n in (1, 2)]
    best, best_sh = None, -1e9
    for lb, topn in grid:
        sr, act = rotation_return(closes, lb, topn)
        is_m, _ = evaluate_series(sr, act, spy, om)
        if is_m["sharpe"] > best_sh:
            best_sh, best = is_m["sharpe"], (lb, topn, sr, act)
    lb, topn, sr, act = best
    rows.append(_row(f"M4 rotation lb{lb}/top{topn}", "swing",
                     *evaluate_series(sr, act, spy, om),
                     note=f"lookback={lb}d top{topn} of {len(UNIVERSE)} (train-picked)"))

    # I2 opening-gap fade — intraday, flat by close (vectorized on daily O/C).
    # gap known at open; fade it; exit at close. Causal, one trade/day.
    o, c = daily["SPY"]["open"], daily["SPY"]["close"]
    gap = o / c.shift(1) - 1.0
    ret_i2 = -np.sign(gap) * (c / o - 1.0)  # short up-gaps, long down-gaps
    rows.append(_row("I2 gap-fade SPY", "intraday",
                     *evaluate(spy, pos_buy_hold(spy), om, ret=ret_i2)))

    # M2 basket — equal-weight RSI-2 over the 3 equity ETFs (corr ~0.18 -> diversifies)
    legs = ["SPY", "QQQ", "IWM"]
    leg_sr, leg_ac = [], []
    for s in legs:
        px = daily[s]["close"]
        srx, acx = strat_returns(px, pos_rsi2(px, t))
        leg_sr.append(srx)
        leg_ac.append(acx)
    basket_r = pd.concat(leg_sr, axis=1).mean(axis=1)
    basket_active = (pd.concat(leg_ac, axis=1).sum(axis=1) > 0).astype(float)
    rows.append(_row(f"M2 basket RSI2<{t} (SPY/QQQ/IWM)", "swing",
                     *evaluate_series(basket_r, basket_active, spy, om),
                     note="equal-weight 3 equity ETFs"))

    # report
    df = pd.DataFrame(rows)
    df = df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    fmt = {"is_sharpe": "{:.2f}", "oos_sharpe": "{:.2f}", "oos_ret%": "{:.1f}",
           "oos_pf": "{:.2f}", "oos_dd%": "{:.1f}", "oos_win%": "{:.0f}"}
    show = df.copy()
    for k, f in fmt.items():
        show[k] = show[k].map(lambda v: f.format(v) if pd.notna(v) else "n/a")
    print(f"\n=== Strategy research - OOS = last {om} months "
          f"({spy.index.max().date()} back {om}mo) ===")
    print(f"Universe: {UNIVERSE} | daily {spy.index.min().date()}..{spy.index.max().date()} "
          f"({len(spy)} bars) | capital ${CAPITAL:,.0f}\n")
    print(show.to_string(index=False))
    print("\nGate = PF>1.2 & DD<15% (oos_passes). Ranked by OOS Sharpe. "
          "Daily families: 'win%' is per-in-market-day.")

    # ---- stress tests on the gate-passers (the part that killed prior winners) ----
    survivors = [r for r in rows if r["gate"] == "PASS" and r["strategy"].startswith(("M", "B"))]
    print("\n" + "=" * 70)
    print("STRESS TESTS (survivors must hold up here, not just one 18mo window)")
    print("=" * 70)

    def _yr_line(label, py):
        cells = "  ".join(f"{y}:Sh{m['sharpe']:+.2f}/PF{m['profit_factor']:.2f}"
                          for y, m in py.items())
        print(f"  {label:<22} {cells}")

    # M2 RSI-2 — per-year, param robustness, multi-asset generalization
    print(f"\n[M2 RSI-2 mean-reversion]  param picked on train: entry<{t}")
    sr2, ac2 = strat_returns(spy, pos_rsi2(spy, t))
    _yr_line("SPY per-year", per_year(sr2, ac2, spy))
    print("  param robustness (entry threshold -> OOS Sharpe / PF / gate):")
    for p in [5, 10, 15, 20]:
        _, om2 = evaluate(spy, pos_rsi2(spy, p), om)
        print(f"    entry<{p:<3} OOS Sharpe {om2['sharpe']:+.2f}  PF {om2['profit_factor']:.2f}  "
              f"DD {om2['max_drawdown_pct']:.1f}%  {'PASS' if oos_passes(om2) else 'fail'}")
    print(f"  multi-asset generalization (same rule entry<{t}, OOS):")
    m2_oos_curves = {}
    for s in UNIVERSE:
        px = daily[s]["close"]
        sr, ac = strat_returns(px, pos_rsi2(px, t))
        _, om2 = evaluate(px, pos_rsi2(px, t), om)
        cutoff = px.index.max() - pd.DateOffset(months=om)
        m2_oos_curves[s] = sr[sr.index > cutoff]
        print(f"    {s}: OOS Sharpe {om2['sharpe']:+.2f}  PF {om2['profit_factor']:.2f}  "
              f"ret {om2['total_return_pct']:+.1f}%  DD {om2['max_drawdown_pct']:.1f}%  "
              f"{'PASS' if oos_passes(om2) else 'fail'}")
    corr = pd.DataFrame(m2_oos_curves).corr()
    print(f"  OOS daily-return correlation across assets (mean off-diag "
          f"{corr.values[~np.eye(len(corr), dtype=bool)].mean():.2f}): "
          "low = diversifiable edge")

    # M2 basket — per-year (the diversified winner)
    print("\n[M2 basket RSI-2 (SPY/QQQ/IWM)]")
    _yr_line("basket per-year", per_year(basket_r, basket_active, spy))

    # I2 gap-fade — per-year (IS<<OOS divergence demands a regime check)
    print("\n[I2 opening-gap fade SPY]  IS Sharpe 0.04 vs OOS 1.52 -> regime check:")
    _yr_line("gap-fade per-year", per_year(ret_i2.fillna(0.0), pos_buy_hold(spy), spy))

    # M4 rotation — per-year
    print(f"\n[M4 dual-momentum rotation]  lookback={lb}d top{topn}")
    _yr_line("portfolio per-year", per_year(sr, act, spy))

    print("\n(Survivors:", ", ".join(r["strategy"] for r in survivors) or "none", ")")


if __name__ == "__main__":
    main()
