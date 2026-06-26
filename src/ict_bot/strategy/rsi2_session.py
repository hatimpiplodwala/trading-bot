"""RSI-2 mean-reversion decision core (live/backtest parity, like ORBSession).

Pure: given a symbol's daily close history through the latest bar and whether a
position is open, return the action. No broker, clock, or network — fully
unit-testable, and the live M2Trader decides exactly as the backtest does.
"""

from __future__ import annotations

from enum import Enum, auto

import pandas as pd

from ict_bot.strategy.rsi2 import is_entry, is_exit


class M2Action(Enum):
    ENTER = auto()
    EXIT = auto()
    HOLD = auto()
    NONE = auto()


class RSI2Session:
    """Per-symbol RSI-2 state machine driven one daily bar at a time."""

    def __init__(self, *, entry_t: float, exit_t: float,
                 trend_len: int, exit_sma_len: int) -> None:
        self._entry_t = entry_t
        self._exit_t = exit_t
        self._trend_len = trend_len
        self._exit_sma_len = exit_sma_len

    @classmethod
    def from_settings(cls, settings: dict) -> "RSI2Session":
        m2 = settings["m2"]
        return cls(entry_t=m2["entry_t"], exit_t=m2["exit_t"],
                   trend_len=m2["trend_len"], exit_sma_len=m2["exit_sma_len"])

    def decide(self, daily_close: pd.Series, has_position: bool) -> M2Action:
        if has_position:
            return M2Action.EXIT if is_exit(
                daily_close, self._exit_t, self._exit_sma_len) else M2Action.HOLD
        if is_entry(daily_close, self._entry_t, self._trend_len):
            return M2Action.ENTER
        return M2Action.NONE
