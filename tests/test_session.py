"""Regular-hours filter — drops the pre/post-market bars IEX history carries."""

from __future__ import annotations

import pandas as pd

from ict_bot.data.session import regular_hours

_ET = "America/New_York"


def _frame(et_times):
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"2026-06-15 {t}", tz=_ET).tz_convert("UTC") for t in et_times],
        name="timestamp",
    )
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx
    )


def test_keeps_only_regular_session_bars():
    df = _frame(["08:00", "09:15", "09:30", "09:45", "12:00", "15:45", "16:00", "16:30"])
    kept = regular_hours(df).index.tz_convert(_ET).strftime("%H:%M").tolist()
    # 09:30 inclusive, 16:00 exclusive; pre-market and post-market dropped.
    assert kept == ["09:30", "09:45", "12:00", "15:45"]


def test_empty_frame_is_returned_unchanged():
    empty = _frame([]).iloc[0:0]
    assert regular_hours(empty).empty
