"""Replay ET trading days through the live ORB decision core.

Answers "what would the bot have done?" without a running service. It drives the
*real* :class:`ORBSession` (the same core the live loop and backtest use) over
freshly-fetched Alpaca bars, then models execution the way the broker would have:
a market OTO entry at the signal bar's close, a resting stop that fills intrabar
at its level, and a forced flatten at ``exits.flatten_time``.

    # one day, full bar-by-bar timeline
    uv run python scripts/simulate_day.py 2026-06-18
    uv run python scripts/simulate_day.py 2026-06-18 --equity 5000

    # a date range / the last N weeks: compact table + aggregate stats
    uv run python scripts/simulate_day.py --from 2026-05-18 --to 2026-06-19 --equity 5000
    uv run python scripts/simulate_day.py --weeks 5 --equity 5000

    # A/B the unproven 3R profit target vs the baseline on the SAME bars
    uv run python scripts/simulate_day.py --weeks 5 --equity 5000                 # baseline
    uv run python scripts/simulate_day.py --weeks 5 --equity 5000 --target-rr 3   # 3R variant

Equity is sized against the live paper account (what the bot would have used);
``--equity`` overrides it. P&L uses the bot's own proxy convention (entry =
signal-bar close), so it matches what state.db logs. Data is paper/IEX; fills are
modelled, not broker truth. ``--target-rr`` is a SHADOW A/B only - the live bot
keeps trading the validated baseline (target_rr: null) in config/settings.yaml;
adopt 3R only if it wins over a meaningful FORWARD sample.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from dataclasses import dataclass

import pandas as pd

from ict_bot.broker.alpaca import AlpacaBroker
from ict_bot.config import REPO_ROOT, load_env, load_settings, require_env
from ict_bot.data.feed import AlpacaFeed
from ict_bot.data.session import regular_hours
from ict_bot.data.store import BarStore
from ict_bot.strategy.orb import orb_breakout
from ict_bot.strategy.orb_session import Action, Enter, ORBSession

_ET = "America/New_York"


@dataclass
class DayResult:
    """One ET day's outcome. ``status`` is trade | skipped | no-breakout | no-or."""

    date: dt.date
    status: str = "no-breakout"
    direction: str | None = None
    qty: int = 0
    entry: float = 0.0
    exit: float = 0.0
    reason: str = ""            # STOP | FLATTEN | EOD
    pnl: float = 0.0
    r: float = 0.0


def _fetch(feed: AlpacaFeed, symbol: str, tf: str, start: dt.date, end: dt.date, seed_days: int):
    """RTH bars from ``seed_days`` (+pad) before ``start`` through end of ``end``."""
    lo = dt.datetime.combine(start, dt.time(0), tzinfo=dt.timezone.utc) - dt.timedelta(days=seed_days + 4)
    hi = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0), tzinfo=dt.timezone.utc)
    return regular_hours(feed.get_historical_bars(symbol, tf, lo, hi))


def _load_range(settings: dict, start: dt.date, end: dt.date, use_parquet: bool) -> pd.DataFrame:
    """RTH bars for ``[start - warmup, end]`` from the local store or the API."""
    symbol, tf = settings["symbol"], settings["entry_timeframe"]
    seed_days = settings.get("live", {}).get("seed_days", 4)
    lo = dt.datetime.combine(start, dt.time(0), tzinfo=dt.timezone.utc) - dt.timedelta(days=seed_days + 4)
    hi = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0), tzinfo=dt.timezone.utc)
    if use_parquet:
        store = BarStore(REPO_ROOT / "data" / "parquet")
        raw = store.read_bars(symbol, tf, pd.Timestamp(lo), pd.Timestamp(hi))
    else:
        feed = AlpacaFeed(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET"))
        raw = feed.get_historical_bars(symbol, tf, lo, hi)
    return regular_hours(raw)


def _with_target(settings: dict, target_rr: float | None) -> dict:
    """Settings copy with the ORB profit target overridden (for the 3R A/B)."""
    if target_rr is None:
        return settings
    settings = copy.deepcopy(settings)
    settings["orb"]["target_rr"] = target_rr
    return settings


def _equity(settings: dict, override: float | None) -> tuple[float, str]:
    if override is not None:
        return override, "override"
    try:
        broker = AlpacaBroker(
            require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET"),
            paper=settings.get("broker", {}).get("paper", True),
        )
        return broker.get_equity(), "live paper account"
    except Exception as exc:  # offline / bad creds - still useful with a default
        print(f"  (could not read account equity: {exc}; using $100,000)")
        return 100_000.0, "fallback default"


def _simulate_day(bars: pd.DataFrame, target: dt.date, equity: float,
                  settings: dict, verbose: bool) -> DayResult | None:
    """Replay ``target`` through a fresh ORBSession; return its outcome (or None).

    ``bars`` must end at/after ``target`` and carry enough prior history to warm
    ATR. Only ``target``'s bars are acted on; earlier bars just warm state.
    """
    et_dates = bars.index.tz_convert(_ET).date
    day = bars[et_dates == target]
    if day.empty:
        return None
    bars = bars[bars.index <= day.index[-1]]  # never peek past the target day
    day_start = bars.index.get_loc(day.index[0])

    session = ORBSession.from_settings(settings)
    res = DayResult(date=target)
    entry: Enter | None = None
    entry_price = 0.0
    done = False  # already entered+closed today; ignore later breakout closes

    for k in range(1, len(bars) + 1):
        ts = bars.index[k - 1]
        row = bars.iloc[k - 1]
        action = session.on_bar(bars.iloc[:k], equity, has_position=entry is not None)
        if k - 1 < day_start:
            continue

        et = ts.tz_convert(_ET)
        label = f"{et:%H:%M}  O={row.open:7.2f} H={row.high:7.2f} L={row.low:7.2f} C={row.close:7.2f}"

        if entry is None and not done:
            if isinstance(action, Enter):
                entry, entry_price = action, float(row.close)
                res.status = "trade"
                res.direction, res.qty, res.entry = action.direction, action.qty, entry_price
                if verbose:
                    tgt = f" target {action.target:.2f}" if action.target is not None else " target none (exit at close)"
                    print(f"{label}  >> ENTER {action.direction.upper()} x{action.qty} @ {entry_price:.2f} "
                          f"| stop {action.stop:.2f}{tgt}")
            elif session._or_high is None:
                if verbose:
                    print(f"{label}  .. building opening range")
            elif orb_breakout(float(row.close), session._or_high, session._or_low) is not None:
                if res.status != "trade":
                    res.status = "skipped"
                if verbose:
                    bo = orb_breakout(float(row.close), session._or_high, session._or_low)
                    print(f"{label}  >> breakout {bo} but SKIPPED (size 0 at ${equity:,.0f} / gate / cutoff)")
            elif verbose:
                print(f"{label}  .. watching (OR {session._or_low:.2f}-{session._or_high:.2f}, no breakout close)")
        elif entry is not None:
            hit = (row.low <= entry.stop) if entry.direction == "long" else (row.high >= entry.stop)
            tgt_hit = entry.target is not None and (
                (row.high >= entry.target) if entry.direction == "long" else (row.low <= entry.target))
            if hit:  # stop checked first: pessimistic when one bar spans both levels
                _finalize(res, entry, entry_price, entry.stop, "STOP")
                entry, done = None, True
                if verbose:
                    print(f"{label}  >> STOP @ {res.exit:.2f}")
            elif tgt_hit:
                _finalize(res, entry, entry_price, entry.target, "TARGET")
                entry, done = None, True
                if verbose:
                    print(f"{label}  >> TARGET @ {res.exit:.2f}")
            elif action is Action.FLATTEN:
                _finalize(res, entry, entry_price, float(row.close), "FLATTEN")
                entry, done = None, True
                if verbose:
                    print(f"{label}  >> FLATTEN @ close @ {res.exit:.2f}")
            elif verbose:
                print(f"{label}  .. holding {entry.direction} (stop {entry.stop:.2f})")

    if entry is not None:  # dataset ended mid-position (shouldn't on a full RTH day)
        _finalize(res, entry, entry_price, float(bars.iloc[-1].close), "EOD")
    if session._or_high is None:
        res.status = "no-or"
    return res


def _finalize(res: DayResult, entry: Enter, entry_price: float, exit_price: float, reason: str) -> None:
    sign = 1 if entry.direction == "long" else -1
    res.exit = exit_price
    res.reason = reason
    res.pnl = (exit_price - entry_price) * entry.qty * sign
    risk = abs(entry_price - entry.stop) * entry.qty
    res.r = res.pnl / risk if risk else 0.0


# --- single-day (verbose) -------------------------------------------------
def simulate(target: dt.date, equity_override: float | None = None,
             target_rr: float | None = None) -> int:
    load_env()
    settings = _with_target(load_settings(), target_rr)
    symbol, tf = settings["symbol"], settings["entry_timeframe"]
    seed_days = settings.get("live", {}).get("seed_days", 4)
    feed = AlpacaFeed(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET"))
    bars = _fetch(feed, symbol, tf, target, target, seed_days)

    if bars[bars.index.tz_convert(_ET).date == target].empty:
        print(f"No {symbol} bars for {target} ({target:%A}) - market closed (holiday/weekend).")
        return 1

    equity, src = _equity(settings, equity_override)
    tag = f"  [A/B target {target_rr}R]" if target_rr is not None else ""
    print(f"\n=== {symbol} {target:%A} {target} - ORB simulation{tag} ===")
    print(f"Equity for sizing: ${equity:,.2f} ({src})\n")
    res = _simulate_day(bars, target, equity, settings, verbose=True)

    if res.status == "trade":
        print(f"\nResult: {res.direction.upper()} x{res.qty}  entry {res.entry:.2f} -> "
              f"exit {res.exit:.2f}  ({res.reason})  P&L ${res.pnl:,.2f}  ({res.r:+.2f}R)")
    elif res.status == "skipped":
        print("\nBreakout occurred but the bot took no position (0 shares at this equity / gate / cutoff).")
    elif res.status == "no-or":
        print("\nNo opening range formed.")
    else:
        print("\nNo breakout - price never closed beyond the opening range. Bot stayed flat.")
    return 0


# --- range (summary) ------------------------------------------------------
def simulate_range(start: dt.date, end: dt.date, equity_override: float | None = None,
                   use_parquet: bool = False, target_rr: float | None = None) -> int:
    load_env()
    settings = _with_target(load_settings(), target_rr)
    symbol = settings["symbol"]
    seed_pad = settings.get("live", {}).get("seed_days", 4) + 4
    bars = _load_range(settings, start, end, use_parquet)

    equity, src = _equity(settings, equity_override)
    trading_days = sorted({d for d in bars.index.tz_convert(_ET).date if start <= d <= end})
    if not trading_days:
        print(f"No {symbol} trading days with data in {start} .. {end}.")
        return 1

    # Bound each day's warmup to ~the live seed window: faithful to how the bot
    # actually starts (it never sees years of warmup), and keeps 1000+ days fast.
    results = []
    for d in trading_days:
        lo_ts = pd.Timestamp(d, tz="UTC") - pd.Timedelta(days=seed_pad)
        res = _simulate_day(bars[bars.index >= lo_ts], d, equity, settings, verbose=False)
        if res is not None:
            results.append(res)
    src += ", local parquet" if use_parquet else ""
    if target_rr is not None:
        src += f", A/B target {target_rr}R"
    _print_summary(symbol, trading_days[0], trading_days[-1], equity, src, results)
    return 0


def _stats(results: list[DayResult], equity: float) -> dict:
    trades = [r for r in results if r.status == "trade"]
    wins = [r for r in trades if r.pnl > 0]
    gross_win = sum(r.pnl for r in wins)
    gross_loss = -sum(r.pnl for r in trades if r.pnl <= 0)
    net = sum(r.pnl for r in trades)
    # Max drawdown on the (non-compounding, fixed-base) cumulative-$ curve.
    cum = peak = max_dd = 0.0
    for r in trades:
        cum += r.pnl
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {
        "days": len(results), "trades": len(trades), "wins": len(wins),
        "losses": len(trades) - len(wins), "net": net, "ret_pct": 100 * net / equity,
        "pf": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "gross_win": gross_win, "gross_loss": gross_loss, "sum_r": sum(r.r for r in trades),
        "max_dd": max_dd, "max_dd_pct": 100 * max_dd / equity,
        "best": max((r.pnl for r in trades), default=0.0),
        "worst": min((r.pnl for r in trades), default=0.0),
    }


def _print_summary(symbol, start, end, equity, src, results: list[DayResult]) -> None:
    print(f"\n=== {symbol} ORB simulation  {start} .. {end} ===")
    print(f"Equity for sizing: ${equity:,.2f} ({src})  |  {len(results)} trading days\n")

    if len(results) <= 35:  # short range: full per-day table
        print(f"{'Date':<12}{'Day':<10}{'Action':<22}{'P&L':>10}{'R':>8}")
        print("-" * 62)
        for r in results:
            wd = r.date.strftime("%a")
            if r.status == "trade":
                act = f"{r.direction.upper()} x{r.qty} {r.reason}"
                print(f"{str(r.date):<12}{wd:<10}{act:<22}{r.pnl:>10.2f}{r.r:>8.2f}")
            else:
                act = {"skipped": "breakout skipped", "no-or": "no opening range"}.get(r.status, "no breakout")
                print(f"{str(r.date):<12}{wd:<10}{act:<22}{'-':>10}{'-':>8}")
        print("-" * 62)
    else:  # long range: per-year breakdown
        print(f"{'Year':<6}{'Days':>6}{'Trades':>8}{'W/L':>10}{'Net $':>12}{'Ret %':>9}{'PF':>7}{'MaxDD%':>9}")
        print("-" * 67)
        for yr in sorted({r.date.year for r in results}):
            yr_res = [r for r in results if r.date.year == yr]
            s = _stats(yr_res, equity)
            wl = f"{s['wins']}/{s['losses']}"
            print(f"{yr:<6}{s['days']:>6}{s['trades']:>8}{wl:>10}{s['net']:>12,.2f}"
                  f"{s['ret_pct']:>9.2f}{s['pf']:>7.2f}{s['max_dd_pct']:>9.2f}")
        print("-" * 67)

    s = _stats(results, equity)
    win = f", win {100*s['wins']/s['trades']:.0f}%" if s["trades"] else ""
    print(f"\nTrades: {s['trades']}  (W {s['wins']} / L {s['losses']}{win})  |  "
          f"flat days: {s['days'] - s['trades']}")
    if s["trades"]:
        print(f"Net P&L: ${s['net']:,.2f}  ({s['ret_pct']:+.2f}% on ${equity:,.0f}, non-compounding)  |  "
              f"sum {s['sum_r']:+.2f}R")
        print(f"Profit factor: {s['pf']:.2f}  |  gross win ${s['gross_win']:,.2f} / gross loss ${s['gross_loss']:,.2f}")
        print(f"Max drawdown: ${s['max_dd']:,.2f} ({s['max_dd_pct']:.2f}%)  |  "
              f"best day ${s['best']:,.2f} / worst ${s['worst']:,.2f}")
    print("\n(entry=signal-bar close, stop fills at its level, flat at the close - an OPTIMISTIC "
          "fill model.\n The rigorous next-open+slippage verdict is run_orb_backtest.py.)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate the ORB bot over ET trading day(s).")
    ap.add_argument("date", nargs="?", help="single ET date YYYY-MM-DD (verbose timeline)")
    ap.add_argument("--from", dest="start", help="range start ET date YYYY-MM-DD")
    ap.add_argument("--to", dest="end", help="range end ET date YYYY-MM-DD")
    ap.add_argument("--weeks", type=int, help="simulate the last N calendar weeks up to today")
    ap.add_argument("--equity", type=float, default=None,
                    help="size against this equity instead of the live account (e.g. 5000)")
    ap.add_argument("--parquet", action="store_true",
                    help="read bars from the local data/parquet store (for multi-year runs)")
    ap.add_argument("--target-rr", type=float, default=None, dest="target_rr",
                    help="A/B: exit at this R-multiple target (e.g. 3) instead of only the close")
    args = ap.parse_args()

    today = dt.datetime.now(dt.timezone.utc).astimezone().date()
    if args.weeks:
        return simulate_range(today - dt.timedelta(weeks=args.weeks), today, args.equity,
                              args.parquet, args.target_rr)
    if args.start or args.end:
        start = dt.date.fromisoformat(args.start) if args.start else today - dt.timedelta(days=7)
        end = dt.date.fromisoformat(args.end) if args.end else today
        return simulate_range(start, end, args.equity, args.parquet, args.target_rr)
    target = dt.date.fromisoformat(args.date) if args.date else today
    return simulate(target, equity_override=args.equity, target_rr=args.target_rr)


if __name__ == "__main__":
    sys.exit(main())
