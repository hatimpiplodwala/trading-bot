"""Train / out-of-sample split tests (Phase 5, Gotcha #6).

The most recent ``oos_months`` are reserved untouched as the verdict set; the
earlier bars are the in-sample tuning set. The two must be disjoint, cover the
whole input, and the train set must end strictly before the OOS set begins.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.backtest.split import train_oos_split


def _daily(months: int) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=months * 30, freq="D", tz="UTC")
    return pd.DataFrame({"close": range(len(idx))}, index=idx)


def test_oos_is_the_most_recent_window():
    df = _daily(24)
    train, oos = train_oos_split(df, oos_months=6)
    cutoff = df.index.max() - pd.DateOffset(months=6)
    assert (oos.index > cutoff).all()
    assert (train.index <= cutoff).all()


def test_split_is_disjoint_and_complete():
    df = _daily(24)
    train, oos = train_oos_split(df, oos_months=6)
    assert len(train) + len(oos) == len(df)
    assert train.index.max() < oos.index.min()
    assert train.index.intersection(oos.index).empty


def test_oos_spans_roughly_requested_months():
    df = _daily(24)
    _, oos = train_oos_split(df, oos_months=6)
    span_days = (oos.index.max() - oos.index.min()).days
    assert 170 <= span_days <= 190  # ~6 months of daily bars
