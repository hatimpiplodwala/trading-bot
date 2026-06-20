"""ORB edge-improvement research — honest train/OOS, no production changes.

The 4.5-year history shows the ORB edge decayed (2022-23 PF ~1.46 -> 2024-25 PF
~1.05). This tests a few *economically-motivated* variants to see if any restores
it WITHOUT overfitting: each variant's parameter is chosen on the in-sample
(train) window only, then judged ONCE on the untouched out-of-sample window
(Gotcha #6). Variants live here only; the validated live core is untouched until
something earns promotion.

Hypotheses (realistic next_open fills throughout):
  1. Trend filter   - take longs only when close > SMA(L), shorts only when
                      close < SMA(L). Counter-trend breakouts fake out in chop.
  2. Breakout buffer- require the close to clear the range by b*ATR (filter
                      marginal pokes that immediately reverse).
  3. Profit target  - exit at an R-multiple instead of only the timed flatten.

Run: ``uv run python scripts/improve_orb.py``
"""

from __future__ import annotations

import copy

import pandas as pd
from backtesting import Backtest

from ict_bot.backtest.metrics import extract_metrics, oos_passes, to_backtesting_frame
from ict_bot.backtest.orb_runner import ORBStrategy, run_orb_backtest
from ict_bot.backtest.split import train_oos_split
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.session import regular_hours
from ict_bot.data.store import BarStore
from ict_bot.strategy.orb import orb_breakout, orb_stop, orb_target
from ict_bot.strategy.risk import compute_atr, position_size

_ET = "America/New_York"
CASH = 100_000.0
SYMBOL = "SPY"


class VariantORB(ORBStrategy):
    """ORBStrategy + optional trend filter and breakout buffer (research only).

    Both gates default off, so with ``trend_lookback=None`` and ``buffer_atr=0``
    this reproduces the baseline ORB exactly (regression-checked in ``main``).
    """

    trend_lookback: int | None = None
    buffer_atr: float = 0.0
    min_atr_pct: float = 0.0  # skip entries when ATR/price is below this (low-vol stand-aside)

    def init(self) -> None:
        super().init()
        self._sma = (
            self.entry_canonical["close"].rolling(self.trend_lookback).mean().to_numpy()
            if self.trend_lookback else None
        )

    def next(self) -> None:  # mirrors ORBStrategy.next, with two extra gates
        i = len(self.data)
        c = i - 1
        ts = self.entry_canonical.index[c]
        et = ts.tz_convert(_ET)
        et_min = et.hour * 60 + et.minute

        if et.date() != self._cur_day:
            self._cur_day = et.date()
            self.daily_gate.start_day(self.equity)
            self._or_high = self._or_low = None
            self._traded = False

        for trade in self.closed_trades[self._seen_closed:]:
            self.daily_gate.register(trade.pl)
        self._seen_closed = len(self.closed_trades)

        high, low, close = float(self.data.High[-1]), float(self.data.Low[-1]), float(self.data.Close[-1])

        if et_min < self._range_end_min:
            self._or_high = high if self._or_high is None else max(self._or_high, high)
            self._or_low = low if self._or_low is None else min(self._or_low, low)
            return

        if self.position:
            if self._should_flatten(c, et_min):
                self.position.close()
            return

        if self._traded or et_min >= self._no_entry_min:
            return
        if self._or_high is None or not self.daily_gate.allows_entry():
            return

        direction = orb_breakout(close, self._or_high, self._or_low)
        if direction is None:
            return

        atr = compute_atr(self.entry_canonical.iloc[:i], length=self._atr_length)

        # --- gate 0: volatility regime (stand aside in low-vol grinds) ---
        if self.min_atr_pct and atr / close < self.min_atr_pct:
            return
        # --- gate 1: breakout buffer ---
        if self.buffer_atr:
            if direction == "long" and close <= self._or_high + self.buffer_atr * atr:
                return
            if direction == "short" and close >= self._or_low - self.buffer_atr * atr:
                return
        # --- gate 2: trend filter ---
        if self._sma is not None:
            sma = self._sma[c]
            if sma == sma:  # not NaN
                if direction == "long" and close <= sma:
                    return
                if direction == "short" and close >= sma:
                    return

        stop = orb_stop(direction, close, self._or_high, self._or_low, atr, self._stop_floor_mult)
        qty = position_size(self.equity, close, stop, self._risk_pct)
        if qty <= 0:
            return
        target = orb_target(direction, close, stop, self._target_rr)
        self._traded = True
        if direction == "long":
            self.buy(size=qty, sl=stop, tp=target)
        else:
            self.sell(size=qty, sl=stop, tp=target)


def run_variant(segment, settings, *, trend_lookback=None, buffer_atr=0.0, min_atr_pct=0.0,
                cash=CASH, fill_mode="next_open"):
    rh = settings.get("regular_hours", {})
    segment = regular_hours(segment, rh.get("start", "09:30"), rh.get("end", "16:00"))

    class _V(VariantORB):
        pass

    _V.entry_canonical = segment
    _V.settings = settings
    _V.fill_mode = fill_mode
    _V.trend_lookback = trend_lookback
    _V.buffer_atr = buffer_atr
    _V.min_atr_pct = min_atr_pct
    spread = (settings["backtest"]["slippage_cents"] / 100.0) / float(segment["close"].mean())
    bt = Backtest(to_backtesting_frame(segment), _V, cash=cash, spread=spread,
                  exclusive_orders=False, finalize_trades=True, trade_on_close=(fill_mode == "close"))
    return bt.run()


def _row(label, m):
    return (f"{label:<26}{m['profit_factor']:>7.2f}{m['total_return_pct']:>9.2f}"
            f"{m['sharpe']:>8.2f}{m['avg_r_multiple']:>8.2f}{m['win_rate_pct']:>7.1f}"
            f"{m['num_trades']:>6}  {'PASS' if oos_passes(m) else 'FAIL'}")


def _hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print(f"{'variant':<26}{'PF':>7}{'Ret%':>9}{'Sharpe':>8}{'avgR':>8}{'Win%':>7}{'N':>6}  OOS-gate")


def _select_on_train(train, settings, candidates, make_kwargs):
    """Pick the candidate with the best TRAIN Sharpe (selection never sees OOS)."""
    best, best_sharpe = None, -1e9
    rows = []
    for cand in candidates:
        m = extract_metrics(run_variant(train, settings, **make_kwargs(cand)))
        rows.append((cand, m))
        if m["sharpe"] > best_sharpe:
            best, best_sharpe = cand, m["sharpe"]
    return best, rows


def main() -> None:
    settings = load_settings()
    tf = settings["entry_timeframe"]
    store = BarStore(REPO_ROOT / "data" / "parquet")
    entry = store.read_bars(SYMBOL, tf)
    train, oos = train_oos_split(entry, oos_months=settings["backtest"]["oos_months"])
    print(f"{SYMBOL} {tf}: train {len(train)} bars, OOS {len(oos)} bars (next_open fills)")

    # --- baseline (regression check: VariantORB with gates off == run_orb_backtest) ---
    base_tr = extract_metrics(run_variant(train, settings))
    base_oos = extract_metrics(run_variant(oos, settings))
    ref_oos = extract_metrics(run_orb_backtest(oos, settings, fill_mode="next_open"))
    assert abs(base_oos["profit_factor"] - ref_oos["profit_factor"]) < 1e-6, "variant baseline drifted!"
    _hdr("BASELINE (no filter)")
    print(_row("baseline  TRAIN", base_tr))
    print(_row("baseline  OOS", base_oos))

    # --- H1: trend filter (select lookback on TRAIN, judge on OOS) ---
    best_L, rows = _select_on_train(
        train, settings, [26, 52, 104], lambda L: {"trend_lookback": L})
    _hdr("H1  TREND FILTER  (long>SMA, short<SMA; lookback chosen on TRAIN)")
    for L, m in rows:
        print(_row(f"trend L={L}  TRAIN{'  *' if L == best_L else ''}", m))
    h1_oos = extract_metrics(run_variant(oos, settings, trend_lookback=best_L))
    print(_row(f"trend L={best_L}  OOS", h1_oos))

    # --- H2: breakout buffer (select on TRAIN, judge on OOS) ---
    best_b, rows = _select_on_train(
        train, settings, [0.1, 0.25, 0.5], lambda b: {"buffer_atr": b})
    _hdr("H2  BREAKOUT BUFFER  (clear range by b*ATR; b chosen on TRAIN)")
    for b, m in rows:
        print(_row(f"buffer={b}  TRAIN{'  *' if b == best_b else ''}", m))
    h2_oos = extract_metrics(run_variant(oos, settings, buffer_atr=best_b))
    print(_row(f"buffer={best_b}  OOS", h2_oos))

    # --- H3: profit target (settings-driven; select on TRAIN, judge on OOS) ---
    def with_rr(rr):
        cfg = copy.deepcopy(settings)
        cfg["orb"]["target_rr"] = rr
        return cfg
    best_rr, best_sh = None, -1e9
    _hdr("H3  PROFIT TARGET  (R-multiple; rr chosen on TRAIN)")
    rr_rows = []
    for rr in [1.5, 2.0, 3.0]:
        m = extract_metrics(run_orb_backtest(train, with_rr(rr), fill_mode="next_open"))
        rr_rows.append((rr, m))
        if m["sharpe"] > best_sh:
            best_rr, best_sh = rr, m["sharpe"]
    for rr, m in rr_rows:
        print(_row(f"target {rr}R  TRAIN{'  *' if rr == best_rr else ''}", m))
    h3_oos = extract_metrics(run_orb_backtest(oos, with_rr(best_rr), fill_mode="next_open"))
    print(_row(f"target {best_rr}R  OOS", h3_oos))

    # --- H4: stop width (wider ATR floor = more noise tolerance) ---
    def with_floor(mult):
        cfg = copy.deepcopy(settings)
        cfg["orb"]["stop_atr_floor_mult"] = mult
        return cfg
    best_fl, best_sh, fl_rows = None, -1e9, []
    _hdr("H4  STOP WIDTH  (ATR floor mult; chosen on TRAIN; 0.5 = baseline)")
    for fl in [0.5, 1.0, 1.5, 2.0]:
        m = extract_metrics(run_orb_backtest(train, with_floor(fl), fill_mode="next_open"))
        fl_rows.append((fl, m))
        if m["sharpe"] > best_sh:
            best_fl, best_sh = fl, m["sharpe"]
    for fl, m in fl_rows:
        print(_row(f"floor={fl}xATR  TRAIN{'  *' if fl == best_fl else ''}", m))
    h4_oos = extract_metrics(run_orb_backtest(oos, with_floor(best_fl), fill_mode="next_open"))
    print(_row(f"floor={best_fl}xATR  OOS", h4_oos))

    # --- H5: entry cutoff (drop late, lower-quality breakouts) ---
    def with_cutoff(hhmm):
        cfg = copy.deepcopy(settings)
        cfg["exits"]["no_entry_after"] = hhmm
        return cfg
    best_co, best_sh, co_rows = None, -1e9, []
    _hdr("H5  ENTRY CUTOFF  (no_entry_after; chosen on TRAIN; 15:30 = baseline)")
    for co in ["11:00", "12:00", "13:00", "15:30"]:
        m = extract_metrics(run_orb_backtest(train, with_cutoff(co), fill_mode="next_open"))
        co_rows.append((co, m))
        if m["sharpe"] > best_sh:
            best_co, best_sh = co, m["sharpe"]
    for co, m in co_rows:
        print(_row(f"cutoff={co}  TRAIN{'  *' if co == best_co else ''}", m))
    h5_oos = extract_metrics(run_orb_backtest(oos, with_cutoff(best_co), fill_mode="next_open"))
    print(_row(f"cutoff={best_co}  OOS", h5_oos))

    # --- H6: volatility-regime filter (stand aside in low-vol; targets the decay) ---
    best_v, rows = _select_on_train(
        train, settings, [0.002, 0.003, 0.004, 0.005], lambda v: {"min_atr_pct": v})
    _hdr("H6  VOL-REGIME FILTER  (skip if ATR/price < t; t chosen on TRAIN)")
    for v, m in rows:
        print(_row(f"min_atr={v:.3f}  TRAIN{'  *' if v == best_v else ''}", m))
    h6_oos = extract_metrics(run_variant(oos, settings, min_atr_pct=best_v))
    print(_row(f"min_atr={best_v:.3f}  OOS", h6_oos))

    # --- best single filter -> combine the two gates if both beat baseline OOS ---
    _hdr("COMBO  trend + buffer (only if both improved OOS)")
    if h1_oos["sharpe"] > base_oos["sharpe"] and h2_oos["sharpe"] > base_oos["sharpe"]:
        combo = extract_metrics(run_variant(oos, settings, trend_lookback=best_L, buffer_atr=best_b))
        print(_row(f"trend L={best_L} + buffer {best_b}  OOS", combo))
    else:
        print("skipped - at least one gate did not beat baseline OOS Sharpe.")

    print(f"\nBaseline OOS: PF {base_oos['profit_factor']:.2f}, Sharpe {base_oos['sharpe']:.2f}. "
          "A variant only 'improves the edge' if it clears this OOS, not just TRAIN.")


if __name__ == "__main__":
    main()
