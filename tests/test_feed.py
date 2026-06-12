"""AlpacaFeed pure-logic tests (Phase 1).

The network calls (historical fetch, WebSocket stream) need live keys and are
exercised by scripts/download_history.py. Here we test the deterministic
transforms: timeframe parsing and bar normalization to our canonical schema.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ict_bot.data.feed import live_bar_to_dict, normalize_bars, parse_timeframe

UTC = dt.timezone.utc


@pytest.mark.parametrize(
    "text,amount,unit",
    [
        ("5m", 5, TimeFrameUnit.Minute),
        ("15m", 15, TimeFrameUnit.Minute),
        ("1h", 1, TimeFrameUnit.Hour),
        ("4h", 4, TimeFrameUnit.Hour),
        ("1d", 1, TimeFrameUnit.Day),
    ],
)
def test_parse_timeframe(text, amount, unit):
    # TimeFrame has no __eq__, so compare the meaningful fields.
    tf = parse_timeframe(text)
    assert tf.amount_value == amount
    assert tf.unit_value == unit


def test_parse_timeframe_rejects_unknown():
    with pytest.raises(ValueError):
        parse_timeframe("7x")


def _alpaca_df() -> pd.DataFrame:
    """Mimic ``StockBarsRequest`` output: MultiIndex (symbol, timestamp)."""
    ts = [dt.datetime(2026, 6, 12, 9, 30, tzinfo=UTC), dt.datetime(2026, 6, 12, 9, 35, tzinfo=UTC)]
    idx = pd.MultiIndex.from_tuples(
        [("SPY", ts[0]), ("SPY", ts[1])], names=["symbol", "timestamp"]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.5],
            "high": [101.0, 102.0],
            "low": [99.5, 100.0],
            "close": [100.5, 101.5],
            "volume": [10, 12],
            "trade_count": [3, 4],
            "vwap": [100.2, 101.0],
        },
        index=idx,
    )


def test_normalize_bars_canonical_schema():
    out = normalize_bars(_alpaca_df(), symbol="SPY")

    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "timestamp"
    assert out.index.tz is not None
    assert len(out) == 2
    assert out.iloc[0]["open"] == 100.0
    assert out.iloc[-1]["close"] == 101.5


def test_normalize_bars_empty():
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    )
    out = normalize_bars(empty, symbol="SPY")
    assert out.empty
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_live_bar_to_dict():
    bar = SimpleNamespace(
        symbol="SPY",
        timestamp=dt.datetime(2026, 6, 12, 9, 30, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=42,
        trade_count=5,
        vwap=100.3,
    )
    row = live_bar_to_dict(bar)
    assert row == {
        "symbol": "SPY",
        "timestamp": dt.datetime(2026, 6, 12, 9, 30, tzinfo=UTC),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 42,
    }
