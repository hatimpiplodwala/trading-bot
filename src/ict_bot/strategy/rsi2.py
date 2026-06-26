"""Connors-style RSI-2 mean-reversion signals (shared by backtest + live).

Pure functions over a daily close series — no state, no broker, no clock — so the
live RSI2Session and the research backtest decide from identical code, exactly as
``strategy/orb.py`` serves both paths for ORB. Entry: oversold (RSI(2) < entry_t)
while in an uptrend (close > SMA(trend_len)). Exit: recovered (RSI(2) > exit_t) or
close back above the short SMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder-smoothed RSI of ``close`` over ``length`` (NaN seed filled to 50)."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def is_entry(close: pd.Series, entry_t: float, trend_len: int = 200) -> bool:
    """True if the latest bar is oversold within an uptrend."""
    if len(close) < trend_len:
        return False
    sma = close.rolling(trend_len).mean().iloc[-1]
    return bool(rsi(close, 2).iloc[-1] < entry_t and close.iloc[-1] > sma)


def is_exit(close: pd.Series, exit_t: float = 60.0, exit_sma_len: int = 5) -> bool:
    """True if the latest bar has recovered (RSI back up, or close above short SMA)."""
    sma = close.rolling(exit_sma_len).mean().iloc[-1]
    return bool(rsi(close, 2).iloc[-1] > exit_t or close.iloc[-1] > sma)
