"""IEX data-validity check — GATING for Phase 1 (Gotcha #5).

Compares Alpaca IEX daily high/low extremes against a consolidated reference
(Yahoo Finance) for SPY, QQQ, and IWM over a recent window. ICT is built on
precise swing extremes and liquidity sweeps — exactly the wick tips a
single-exchange feed distorts. If extremes diverge materially (verdict REVIEW),
record a decision to move to paid SIP (Alpaca Algo Trader Plus) before trusting
any detector built on this data.

Run with: ``uv run python scripts/download_history.py`` first (optional), then
``uv run python scripts/validate_iex.py``. Exits non-zero if any symbol needs
review, so it can gate a pipeline.
"""

from __future__ import annotations

import datetime as dt
import sys

import httpx
import pandas as pd

from ict_bot.config import load_env, load_settings, require_env
from ict_bot.data.feed import AlpacaFeed
from ict_bot.data.quality import compare_daily_extremes, iex_verdict

LOOKBACK_DAYS = 60
THRESHOLD_PCT = 0.1
UTC = dt.timezone.utc


def fetch_yahoo_daily(symbol: str, days: int) -> pd.DataFrame:
    """Consolidated daily high/low from Yahoo Finance (free reference tape)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "3mo", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (ict-bot iex-validation)"}
    resp = httpx.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    df = pd.DataFrame({"high": quote["high"], "low": quote["low"]}, index=idx)
    return df.dropna()


def main() -> None:
    load_env()
    settings = load_settings()
    feed = AlpacaFeed(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET"))

    symbols = [settings["symbol"], *settings.get("smt_reference_symbols", [])]
    end = dt.datetime.now(UTC)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)

    overall_ok = True
    for symbol in symbols:
        iex = feed.get_historical_bars(symbol, "1d", start, end)
        ref = fetch_yahoo_daily(symbol, LOOKBACK_DAYS)
        stats = compare_daily_extremes(iex, ref)
        verdict = iex_verdict(stats, threshold_pct=THRESHOLD_PCT)
        overall_ok &= verdict == "PASS"

        print(
            f"{symbol}: {verdict}  days={stats['days']}  "
            f"high dev median/max = {stats['high_dev_pct']['median']:.3f}%/"
            f"{stats['high_dev_pct']['max']:.3f}%  "
            f"low dev median/max = {stats['low_dev_pct']['median']:.3f}%/"
            f"{stats['low_dev_pct']['max']:.3f}%"
        )

    if overall_ok:
        print(f"\nPASS — IEX daily extremes track the reference within {THRESHOLD_PCT}%.")
    else:
        print(
            "\nREVIEW — IEX diverges from the consolidated tape. ICT sweeps/extremes "
            "may be unreliable on IEX. Decide: accept and document, or move to paid "
            "SIP (Alpaca Algo Trader Plus) before Phase 2."
        )
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
