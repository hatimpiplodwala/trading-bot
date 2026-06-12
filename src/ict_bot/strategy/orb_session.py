"""Live ORB decision core — backtest parity, broker-agnostic.

This is the same Opening Range Breakout logic the backtest drives through
``ORBStrategy.next()`` (:mod:`ict_bot.backtest.orb_runner`), extracted so the live
loop and the backtest share one source of truth. Fed the session's bars so far,
the current account equity and whether a position is open, :meth:`ORBSession.on_bar`
returns the action to take — never touching a broker, a clock, or the network, so
it is fully unit-testable and guarantees the live bot decides exactly as the
backtest did.

Live execution differs from the backtest only in *fills*: the backtest can model
the stop/target intrabar, whereas live we submit a market entry plus a resting
stop at the broker and force a flat at the close. The *decision* (when to enter,
which direction, the stop, the size, when to flatten) is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd

from ict_bot.strategy.orb import orb_breakout, orb_stop, orb_target
from ict_bot.strategy.risk import DailyLossLimit, compute_atr, position_size

_ET = "America/New_York"


def _to_min(hhmm: str) -> int:
    """``"HH:MM"`` (ET wall-clock) -> minutes since midnight."""
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


class Action(Enum):
    """A non-entry decision from :meth:`ORBSession.on_bar`."""

    NONE = auto()
    FLATTEN = auto()


@dataclass(frozen=True)
class Enter:
    """An entry decision: open ``qty`` shares ``direction`` with a stop (and TP)."""

    direction: str  # "long" | "short"
    stop: float
    target: float | None
    qty: int


class ORBSession:
    """Per-session ORB state machine: one trade/day, flat by the close."""

    def __init__(
        self,
        *,
        range_end: str,
        no_entry_after: str,
        flatten_time: str,
        risk_pct: float,
        atr_length: int,
        stop_floor_mult: float,
        target_rr: float | None,
        daily_loss_limit: float,
    ) -> None:
        self._range_end_min = _to_min(range_end)
        self._no_entry_min = _to_min(no_entry_after)
        self._flatten_min = _to_min(flatten_time)
        self._risk_pct = risk_pct
        self._atr_length = atr_length
        self._stop_floor_mult = stop_floor_mult
        self._target_rr = target_rr
        self._gate = DailyLossLimit(limit_pct=daily_loss_limit)
        self._cur_day = None
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._traded = False

    @classmethod
    def from_settings(cls, settings: dict) -> "ORBSession":
        return cls(
            range_end=settings["opening_range"]["end"],
            no_entry_after=settings["exits"]["no_entry_after"],
            flatten_time=settings["exits"]["flatten_time"],
            risk_pct=settings["risk"]["risk_per_trade"],
            atr_length=settings["risk"]["atr_length"],
            stop_floor_mult=settings["orb"]["stop_atr_floor_mult"],
            target_rr=settings["orb"]["target_rr"],
            daily_loss_limit=settings["risk"]["daily_loss_limit"],
        )

    def register_realized(self, pnl: float) -> None:
        """Feed a realized close P&L into the daily-loss circuit breaker."""
        self._gate.register(pnl)

    def on_bar(self, bars: pd.DataFrame, equity: float, has_position: bool) -> Action | Enter:
        """Decide on the just-closed bar (``bars`` ends with the current bar)."""
        ts = bars.index[-1]
        et = ts.tz_convert(_ET)
        et_min = et.hour * 60 + et.minute

        if et.date() != self._cur_day:  # new session: reset OR + per-day state
            self._cur_day = et.date()
            self._gate.start_day(equity)
            self._or_high = self._or_low = None
            self._traded = False

        high = float(bars["high"].iloc[-1])
        low = float(bars["low"].iloc[-1])
        close = float(bars["close"].iloc[-1])

        # Inside the opening range: extend it, take no trade.
        if et_min < self._range_end_min:
            self._or_high = high if self._or_high is None else max(self._or_high, high)
            self._or_low = low if self._or_low is None else min(self._or_low, low)
            return Action.NONE

        # Flat by the close once we hold a position.
        if has_position:
            return Action.FLATTEN if et_min >= self._flatten_min else Action.NONE

        if self._traded or et_min >= self._no_entry_min:
            return Action.NONE
        if self._or_high is None or not self._gate.allows_entry():
            return Action.NONE

        direction = orb_breakout(close, self._or_high, self._or_low)
        if direction is None:
            return Action.NONE

        atr = compute_atr(bars, length=self._atr_length)
        stop = orb_stop(direction, close, self._or_high, self._or_low, atr, self._stop_floor_mult)
        qty = position_size(equity, close, stop, self._risk_pct)
        if qty <= 0:
            return Action.NONE

        target = orb_target(direction, close, stop, self._target_rr)
        self._traded = True
        return Enter(direction=direction, stop=stop, target=target, qty=qty)
