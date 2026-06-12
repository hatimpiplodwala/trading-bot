"""Opening Range Breakout backtest harness (momentum strategy).

A parallel runner to :mod:`ict_bot.backtest.runner`, driving the ORB signal
(:mod:`ict_bot.strategy.orb`) bar-by-bar through backtesting.py. Single series,
one trade per day, flat by the close — no HTF/reference frames needed. Reuses the
shared pieces: canonical->backtesting frame, slippage spread, risk sizing, and
the daily loss limit.
"""

from __future__ import annotations

import pandas as pd
from backtesting import Backtest, Strategy

from ict_bot.backtest.metrics import to_backtesting_frame
from ict_bot.backtest.runner import _hhmm_to_minutes
from ict_bot.strategy.orb import opening_range, orb_breakout, orb_stop, orb_target
from ict_bot.strategy.risk import DailyLossLimit, compute_atr, position_size

_ET = "America/New_York"


class ORBStrategy(Strategy):
    """Opening Range Breakout, both directions, one trade per day, flat by close.

    ``fill_mode`` controls execution realism:

    - ``"close"`` (default): entries and the flatten fill at the *signal bar's
      close* (``trade_on_close=True``). Optimistic — assumes you transact at the
      price you only know once the bar has closed.
    - ``"next_open"``: entries and the flatten fill at the *next bar's open*
      (``trade_on_close=False``). The pessimistic, realistic model — confirm the
      close beyond the range, then act on the following bar. To keep the
      flat-by-close invariant the flatten is brought forward one bar so its
      next-open fill still lands inside the same session.
    """

    entry_canonical: pd.DataFrame = None
    settings: dict = None
    fill_mode: str = "close"

    def init(self) -> None:
        s = self.settings
        self._range_end_min = _hhmm_to_minutes(s["opening_range"]["end"])
        self._no_entry_min = _hhmm_to_minutes(s["exits"]["no_entry_after"])
        self._flatten_min = _hhmm_to_minutes(s["exits"]["flatten_time"])
        self._risk_pct = s["risk"]["risk_per_trade"]
        self._atr_length = s["risk"]["atr_length"]
        self._stop_floor_mult = s["orb"]["stop_atr_floor_mult"]
        self._target_rr = s["orb"]["target_rr"]
        self.daily_gate = DailyLossLimit(limit_pct=s["risk"]["daily_loss_limit"])
        self._seen_closed = 0
        self._cur_day = None
        self._or_high = None
        self._or_low = None
        self._traded = False
        # Per-bar flag: is this the last bar of its ET session? Used to keep the
        # next-open flatten inside the session (close one bar early so the fill,
        # at the following bar's open, is still same-day). Robust to half-days.
        et_dates = self.entry_canonical.index.tz_convert(_ET).date
        n = len(et_dates)
        self._last_of_session = [
            k == n - 1 or et_dates[k + 1] != et_dates[k] for k in range(n)
        ]

    def _should_flatten(self, c: int, et_min: int) -> bool:
        """Whether to close the open position on the current bar ``c``."""
        n = len(self.entry_canonical)
        if self.fill_mode == "next_open":
            # Close now so the next-open fill lands on (at latest) the session's
            # last bar — never the next session's open.
            return c + 1 >= n or self._last_of_session[c + 1]
        last_of_session = c + 1 >= n or self._last_of_session[c]
        return et_min >= self._flatten_min or last_of_session

    def next(self) -> None:
        i = len(self.data)
        c = i - 1
        ts = self.entry_canonical.index[c]
        et = ts.tz_convert(_ET)
        et_min = et.hour * 60 + et.minute

        if et.date() != self._cur_day:  # new session: reset OR + per-day state
            self._cur_day = et.date()
            self.daily_gate.start_day(self.equity)
            self._or_high = self._or_low = None
            self._traded = False

        for trade in self.closed_trades[self._seen_closed:]:
            self.daily_gate.register(trade.pl)
        self._seen_closed = len(self.closed_trades)

        high, low, close = float(self.data.High[-1]), float(self.data.Low[-1]), float(self.data.Close[-1])

        # Build the opening range while inside the OR window — no trading yet.
        if et_min < self._range_end_min:
            self._or_high = high if self._or_high is None else max(self._or_high, high)
            self._or_low = low if self._or_low is None else min(self._or_low, low)
            return

        # Flat by close (or the session's last bar — half-days, dataset end).
        if self.position:
            if self._should_flatten(c, et_min):
                self.position.close()
            return

        if self._traded or et_min >= self._no_entry_min:
            return
        if self._or_high is None or not self.daily_gate.allows_entry():
            return

        direction = orb_breakout(close, self._or_high, self._or_low)
        if direction is None:
            return

        atr = compute_atr(self.entry_canonical.iloc[:i], length=self._atr_length)
        stop = orb_stop(direction, close, self._or_high, self._or_low, atr, self._stop_floor_mult)
        qty = position_size(self.equity, close, stop, self._risk_pct)
        if qty <= 0:
            return

        target = orb_target(direction, close, stop, self._target_rr)
        self._traded = True
        if direction == "long":
            self.buy(size=qty, sl=stop, tp=target)
        else:
            self.sell(size=qty, sl=stop, tp=target)


def _configured_strategy(
    entry_canonical: pd.DataFrame, settings: dict, fill_mode: str
) -> type[ORBStrategy]:
    class _Configured(ORBStrategy):
        pass

    _Configured.entry_canonical = entry_canonical
    _Configured.settings = settings
    _Configured.fill_mode = fill_mode
    return _Configured


def run_orb_backtest(
    entry_segment: pd.DataFrame,
    settings: dict,
    cash: float = 100_000.0,
    fill_mode: str = "close",
) -> pd.Series:
    """Run ORB over one entry-timeframe segment; return backtesting stats.

    ``fill_mode`` is ``"close"`` (fill at the signal bar's close — optimistic) or
    ``"next_open"`` (fill at the next bar's open — realistic). See
    :class:`ORBStrategy`.
    """
    if fill_mode not in ("close", "next_open"):
        raise ValueError(f"fill_mode must be 'close' or 'next_open', got {fill_mode!r}")
    strategy = _configured_strategy(entry_segment, settings, fill_mode)
    mean_price = float(entry_segment["close"].mean())
    spread = (settings["backtest"]["slippage_cents"] / 100.0) / mean_price
    bt = Backtest(
        to_backtesting_frame(entry_segment),
        strategy,
        cash=cash,
        spread=spread,
        exclusive_orders=False,
        finalize_trades=True,
        trade_on_close=(fill_mode == "close"),
    )
    return bt.run()
