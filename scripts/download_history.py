"""Bootstrap historical bars into the DuckDB/Parquet store (Phase 1).

Pulls ~2 years of bars for SPY and the SMT reference symbols (QQQ, IWM) at the
fetchable timeframes (5m, 15m, 1h, 1d). The 4h timeframe is derived on demand via
``data.resample`` — the single tested no-lookahead path — so it is not fetched.

Run with:
    uv run python scripts/download_history.py
    uv run python scripts/download_history.py --years 2 --symbols SPY QQQ IWM
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from ict_bot.config import REPO_ROOT, load_env, load_settings, require_env
from ict_bot.data.feed import AlpacaFeed
from ict_bot.data.store import BarStore

UTC = dt.timezone.utc
DERIVED_TIMEFRAMES = {"4h"}  # built by resampling, never fetched


def parse_args(settings: dict) -> argparse.Namespace:
    default_symbols = [settings["symbol"], *settings.get("smt_reference_symbols", [])]
    parser = argparse.ArgumentParser(description="Bootstrap historical bar data.")
    parser.add_argument("--years", type=float, default=settings["data"]["history_years"])
    parser.add_argument("--symbols", nargs="+", default=default_symbols)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "parquet")
    return parser.parse_args()


def main() -> None:
    load_env()
    settings = load_settings()
    args = parse_args(settings)

    feed = AlpacaFeed(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET"))
    store = BarStore(args.data_dir)

    timeframes = [tf for tf in settings["timeframes"] if tf not in DERIVED_TIMEFRAMES]
    end = dt.datetime.now(UTC)
    start = end - dt.timedelta(days=round(args.years * 365))

    print(f"Downloading {args.years}y of {args.symbols} at {timeframes} -> {args.data_dir}")
    for symbol in args.symbols:
        for timeframe in timeframes:
            bars = feed.get_historical_bars(symbol, timeframe, start, end)
            store.append_bars(symbol, timeframe, bars)
            print(f"  {symbol} {timeframe}: {len(bars):>7} bars")

    print("Done. Next: uv run python scripts/validate_iex.py  (Gotcha #5 gate)")


if __name__ == "__main__":
    main()
