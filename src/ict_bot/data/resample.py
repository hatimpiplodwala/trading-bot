"""Multi-timeframe resampling (Phase 1).

Aggregate base bars to higher timeframes. CRITICAL: never include a forming bar
(Gotcha #1). Convention: bars are indexed by OPEN (start) time in UTC; a
higher-timeframe bar covering ``[open, open + timeframe)`` is emitted only once
fully closed — its close time must be <= the last base bar's close. The forming
(incomplete) final bucket is dropped. This is the single source of resampling
for both the live loop and the backtest, so the no-lookahead guarantee holds
identically in each.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample open-time-indexed UTC OHLCV bars to a higher ``timeframe``.

    ``timeframe`` is a pandas offset alias (e.g. ``"15min"``, ``"1h"``, ``"4h"``,
    ``"1D"``). Only fully-closed bars are returned.
    """
    if df.index.tz is None:
        raise ValueError("resample_ohlcv requires a tz-aware (UTC) index")
    if df.empty:
        return df.copy()

    df = df.sort_index()

    # Base bar period = smallest gap between consecutive opens (robust to data
    # gaps, where a missing bar would inflate the median).
    base_period = df.index.to_series().diff().min()
    last_base_close = df.index[-1] + base_period

    resampled = (
        df.resample(timeframe, label="left", closed="left", origin="start_day")
        .agg(_AGG)
        .dropna(subset=["open"])
    )

    tf_delta = pd.Timedelta(to_offset(timeframe))
    closed = resampled.index + tf_delta <= last_base_close
    resampled = resampled[closed]

    resampled["volume"] = resampled["volume"].astype(df["volume"].dtype)
    return resampled
