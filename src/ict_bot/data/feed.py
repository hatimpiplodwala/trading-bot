"""Alpaca market data feed (Phase 1).

``AlpacaFeed``: historical bars via ``StockHistoricalDataClient`` and live bars
via ``StockDataStream`` (WebSocket), using the free IEX feed by default
(``DataFeed.IEX``). See Gotcha #5 in prd.md on IEX-vs-SIP data quality.

Bars are normalized to the canonical schema used across the project: indexed by
OPEN (start) time in UTC, columns ``[open, high, low, close, volume]``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Awaitable, Callable, Iterable

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

CANONICAL = ["open", "high", "low", "close", "volume"]

_UNIT = {
    "m": TimeFrameUnit.Minute,
    "h": TimeFrameUnit.Hour,
    "d": TimeFrameUnit.Day,
}


def parse_timeframe(text: str) -> TimeFrame:
    """Parse ``"5m"``/``"15m"``/``"1h"``/``"4h"``/``"1d"`` into a ``TimeFrame``."""
    text = text.strip().lower()
    if len(text) < 2 or text[-1] not in _UNIT or not text[:-1].isdigit():
        raise ValueError(f"Unrecognized timeframe: {text!r}")
    return TimeFrame(int(text[:-1]), _UNIT[text[-1]])


def normalize_bars(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize an Alpaca bars frame to the canonical OHLCV schema.

    Accepts the MultiIndex ``(symbol, timestamp)`` frame returned by
    ``get_stock_bars(...).df`` (or an empty frame) and returns a single
    timestamp-indexed (UTC) frame with ``CANONICAL`` columns, sorted ascending.
    """
    if df.empty:
        return pd.DataFrame(
            columns=CANONICAL, index=pd.DatetimeIndex([], name="timestamp")
        )

    if isinstance(df.index, pd.MultiIndex):
        if symbol is not None and "symbol" in df.index.names:
            df = df.xs(symbol, level="symbol")
        else:
            df = df.droplevel("symbol") if "symbol" in df.index.names else df.droplevel(0)

    out = df[CANONICAL].copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    return out.sort_index()


def live_bar_to_dict(bar: Any) -> dict[str, Any]:
    """Normalize a streamed Alpaca ``Bar`` into a canonical row dict."""
    return {
        "symbol": bar.symbol,
        "timestamp": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


class AlpacaFeed:
    """Historical + live SPY/QQQ/IWM bars from Alpaca (paper account, IEX feed)."""

    def __init__(self, api_key: str, secret: str, feed: DataFeed = DataFeed.IEX) -> None:
        self._api_key = api_key
        self._secret = secret
        self._feed = feed
        self._hist = StockHistoricalDataClient(api_key, secret)

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Fetch historical bars for ``symbol`` in ``[start, end]`` (IEX feed)."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=parse_timeframe(timeframe),
            start=start,
            end=end,
            feed=self._feed,
        )
        barset = self._hist.get_stock_bars(request)
        return normalize_bars(barset.df, symbol=symbol)

    def stream_bars(
        self,
        symbols: Iterable[str],
        on_bar: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Subscribe to live bars and run the WebSocket loop (blocking).

        ``on_bar`` is an async callback receiving canonical row dicts. Reconnect
        / backoff handling is layered on in Phase 7's live loop.
        """
        stream = StockDataStream(self._api_key, self._secret, feed=self._feed)

        async def _handler(bar: Any) -> None:
            await on_bar(live_bar_to_dict(bar))

        stream.subscribe_bars(_handler, *symbols)
        stream.run()
