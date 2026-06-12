"""Resampling tests (Phase 1) — CRITICAL: prove no lookahead bias.

Convention: bars are indexed by OPEN (start) time in UTC (Alpaca's native
convention). A higher-timeframe bar covering [open, open+timeframe) is only
emitted once fully closed — i.e. its close time <= the last base bar's close.
The forming (incomplete) final bucket must never appear.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ict_bot.data.resample import resample_ohlcv

UTC = dt.timezone.utc


def _bars(rows: list[tuple[str, float, float, float, float, int]]) -> pd.DataFrame:
    """Build an open-time-indexed OHLCV frame from (HH:MM, o, h, l, c, v) rows."""
    index = [dt.datetime(2026, 6, 12, int(t[:2]), int(t[3:]), tzinfo=UTC) for t, *_ in rows]
    data = [r[1:] for r in rows]
    return pd.DataFrame(
        data, index=pd.DatetimeIndex(index, name="timestamp"),
        columns=["open", "high", "low", "close", "volume"],
    )


# Six 5-minute bars = exactly two complete 15-minute buckets.
_FIVE_MIN = _bars([
    ("09:30", 100.0, 101.0, 99.5, 100.5, 10),
    ("09:35", 100.5, 102.0, 100.0, 101.5, 12),
    ("09:40", 101.5, 101.8, 100.8, 101.0, 8),
    ("09:45", 101.0, 103.0, 100.5, 102.5, 20),
    ("09:50", 102.5, 102.7, 101.0, 101.2, 5),
    ("09:55", 101.2, 101.9, 100.2, 100.9, 7),
])


def test_aggregates_ohlcv_correctly():
    out = resample_ohlcv(_FIVE_MIN, "15min")

    first = out.iloc[0]
    assert first["open"] == 100.0          # first open of the bucket
    assert first["high"] == 102.0          # max high across the 3 bars
    assert first["low"] == 99.5            # min low across the 3 bars
    assert first["close"] == 101.0         # last close of the bucket
    assert first["volume"] == 30           # 10 + 12 + 8


def test_bar_is_labeled_by_open_time():
    out = resample_ohlcv(_FIVE_MIN, "15min")
    assert out.index[0] == dt.datetime(2026, 6, 12, 9, 30, tzinfo=UTC)
    assert out.index[1] == dt.datetime(2026, 6, 12, 9, 45, tzinfo=UTC)


def test_two_complete_buckets_emitted():
    out = resample_ohlcv(_FIVE_MIN, "15min")
    assert len(out) == 2


def test_forming_bar_is_dropped_no_lookahead():
    # Add a 7th 5m bar opening 10:00 -> starts a 3rd 15m bucket that is NOT closed.
    forming = pd.concat([
        _FIVE_MIN,
        _bars([("10:00", 100.9, 105.0, 100.0, 104.0, 99)]),
    ])
    out = resample_ohlcv(forming, "15min")

    # The forming 10:00 bucket must not appear; still exactly two closed bars.
    assert len(out) == 2
    assert dt.datetime(2026, 6, 12, 10, 0, tzinfo=UTC) not in out.index
    # And its inflated high (105) must not leak into any emitted bar.
    assert out["high"].max() == 103.0


def test_15m_bar_contains_only_5m_bars_closed_by_its_close():
    out = resample_ohlcv(_FIVE_MIN, "15min")
    timeframe = pd.Timedelta(minutes=15)
    last_base_close = _FIVE_MIN.index[-1] + pd.Timedelta(minutes=5)

    for open_time in out.index:
        close_time = open_time + timeframe
        # Every emitted bar is fully closed by the last observed base bar.
        assert close_time <= last_base_close
        # Constituent base bars all open within [open, close).
        members = _FIVE_MIN[(_FIVE_MIN.index >= open_time) & (_FIVE_MIN.index < close_time)]
        assert out.loc[open_time, "close"] == members.iloc[-1]["close"]


def test_empty_input_returns_empty():
    empty = _FIVE_MIN.iloc[0:0]
    out = resample_ohlcv(empty, "15min")
    assert out.empty


def test_requires_tz_aware_index():
    naive = _FIVE_MIN.tz_localize(None)
    with pytest.raises(ValueError):
        resample_ohlcv(naive, "15min")
