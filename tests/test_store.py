"""BarStore tests (Phase 1): DuckDB/Parquet round-trip, partitioning, upsert."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ict_bot.data.store import BarStore

UTC = dt.timezone.utc
COLS = ["open", "high", "low", "close", "volume"]


def _bars(times: list[dt.datetime], base: float = 100.0) -> pd.DataFrame:
    rows = [(base + i, base + i + 1, base + i - 1, base + i + 0.5, 10 + i) for i in range(len(times))]
    return pd.DataFrame(
        rows, index=pd.DatetimeIndex(times, name="timestamp"), columns=COLS
    )


@pytest.fixture
def store(tmp_path):
    return BarStore(tmp_path / "parquet")


def test_round_trip(store):
    times = [dt.datetime(2026, 6, 12, 9, 30 + 5 * i, tzinfo=UTC) for i in range(3)]
    store.append_bars("SPY", "5m", _bars(times))

    out = store.read_bars("SPY", "5m")
    assert list(out.columns) == COLS
    assert len(out) == 3
    assert out.index[0] == times[0]
    assert out.index.tz is not None  # tz-aware (UTC) preserved


def test_read_missing_returns_empty_with_columns(store):
    out = store.read_bars("SPY", "5m")
    assert out.empty
    assert list(out.columns) == COLS


def test_range_is_half_open(store):
    times = [dt.datetime(2026, 6, 12, 9, 0, tzinfo=UTC) + dt.timedelta(minutes=5 * i) for i in range(4)]
    store.append_bars("SPY", "5m", _bars(times))

    out = store.read_bars("SPY", "5m", start=times[1], end=times[3])
    # start inclusive, end exclusive -> times[1], times[2] only.
    assert list(out.index) == [times[1], times[2]]


def test_append_is_idempotent(store):
    times = [dt.datetime(2026, 6, 12, 9, 30 + 5 * i, tzinfo=UTC) for i in range(3)]
    store.append_bars("SPY", "5m", _bars(times))
    store.append_bars("SPY", "5m", _bars(times))

    out = store.read_bars("SPY", "5m")
    assert len(out) == 3  # no duplicates


def test_append_upserts_overlapping_timestamps(store):
    t = dt.datetime(2026, 6, 12, 9, 30, tzinfo=UTC)
    store.append_bars("SPY", "5m", _bars([t], base=100.0))
    # Re-append same timestamp with a different close -> newer value wins.
    updated = _bars([t], base=200.0)
    store.append_bars("SPY", "5m", updated)

    out = store.read_bars("SPY", "5m")
    assert len(out) == 1
    assert out.iloc[0]["close"] == updated.iloc[0]["close"]


def test_partitions_by_year_month(tmp_path):
    store = BarStore(tmp_path / "parquet")
    times = [dt.datetime(2026, 1, 31, 23, 55, tzinfo=UTC), dt.datetime(2026, 2, 1, 0, 0, tzinfo=UTC)]
    store.append_bars("SPY", "5m", _bars(times))

    part_dir = tmp_path / "parquet" / "SPY" / "5m"
    files = sorted(p.name for p in part_dir.glob("*.parquet"))
    assert files == ["2026-01.parquet", "2026-02.parquet"]
    assert len(store.read_bars("SPY", "5m")) == 2


def test_symbols_and_timeframes_are_isolated(store):
    t = dt.datetime(2026, 6, 12, 9, 30, tzinfo=UTC)
    store.append_bars("SPY", "5m", _bars([t]))
    store.append_bars("QQQ", "5m", _bars([t]))
    store.append_bars("SPY", "15m", _bars([t]))

    assert len(store.read_bars("SPY", "5m")) == 1
    assert len(store.read_bars("QQQ", "5m")) == 1
    assert len(store.read_bars("SPY", "15m")) == 1
