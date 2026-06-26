"""Restrict intraday bars to the regular trading session (default 09:30-16:00 ET).

Alpaca's free IEX history carries pre- and post-market bars. Left in, they corrupt
the strategy two ways: pre-market bars fold into the opening range (the range start
isn't a time the breakout logic checks), and post-market bars make the backtest's
next-open flatten hold past the close — which the live loop never does, breaking
backtest/live parity. Both the backtest runner and the live seed restrict to
regular hours through this one helper, so they always see the same bars.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.config import hhmm_to_min

_ET = "America/New_York"


def regular_hours(df: pd.DataFrame, start: str = "09:30", end: str = "16:00") -> pd.DataFrame:
    """Keep only bars whose ET open time is in ``[start, end)`` (drops extended hours)."""
    if df.empty:
        return df
    et = df.index.tz_convert(_ET)
    et_min = et.hour * 60 + et.minute
    return df[(et_min >= hhmm_to_min(start)) & (et_min < hhmm_to_min(end))]
