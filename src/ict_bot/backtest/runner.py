"""Backtest harness (Phase 5).

Wraps the live signal + risk engine in a ``backtesting.py`` Strategy, fed
bar-by-bar. The SAME code paths as live (no separate logic): each bar slices the
HTF/reference frames to the current timestamp and calls the very same
``build_setup_context`` + ``SignalEngine.evaluate`` the live loop uses. Enforces a
train/test split (see :mod:`ict_bot.backtest.split`): tune on the earlier ~18
months, judge on an untouched ~6-month out-of-sample window. A 1-cent/share
slippage assumption is modelled as a bid-ask spread (Gotcha #7).

``backtesting.py`` feeds a single series, so SPY entry-timeframe bars are the
primary data; Daily, 1H, and the QQQ/IWM references travel as configured class
attributes and are sliced to ``<= ts`` each bar — never beyond (no lookahead).
"""

from __future__ import annotations

import pandas as pd
from backtesting import Backtest, Strategy

from ict_bot.backtest.metrics import to_backtesting_frame
from ict_bot.ict.sessions import Sessions
from ict_bot.strategy.risk import DailyLossLimit
from ict_bot.strategy.signal import (
    SignalEngine,
    build_setup_context,
    resolve_direction,
)

_ET = "America/New_York"


def _hhmm_to_minutes(hhmm: str) -> int:
    """\"HH:MM\" -> minutes since midnight."""
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


class ICTStrategy(Strategy):
    """ICT signal/risk engine driven bar-by-bar by backtesting.py.

    Configuration is injected as class attributes by :func:`run_backtest` (the
    frames can't ride through ``bt.run(**params)``, which is for optimizable
    scalars only).
    """

    entry_canonical: pd.DataFrame = None  # SPY entry-TF bars (canonical, lowercase)
    daily: pd.DataFrame = None
    h1: pd.DataFrame = None
    references: dict[str, pd.DataFrame] = None
    sessions: Sessions = None
    settings: dict = None
    swing_length: int = 20

    def init(self) -> None:
        gate = DailyLossLimit(limit_pct=self.settings["risk"]["daily_loss_limit"])
        self.engine = SignalEngine.from_settings(self.settings, daily_gate=gate)
        exits = self.settings["exits"]
        self._flatten_min = _hhmm_to_minutes(exits["flatten_time"])
        self._no_entry_min = _hhmm_to_minutes(exits["no_entry_after"])
        self._seen_closed = 0
        self._cur_day = None
        self._dir_cache_key = None
        self._dir_cache_val = None

    def next(self) -> None:
        i = len(self.data)  # number of visible bars == current position + 1
        ts = self.entry_canonical.index[i - 1]
        et = ts.tz_convert(_ET)
        et_min = et.hour * 60 + et.minute

        # Daily loss-limit bookkeeping (reset per ET day, register realized P&L).
        if et.date() != self._cur_day:
            self._cur_day = et.date()
            self.engine.daily_gate.start_day(self.equity)
        for trade in self.closed_trades[self._seen_closed:]:
            self.engine.daily_gate.register(trade.pl)
        self._seen_closed = len(self.closed_trades)

        if self.position:  # max one concurrent position (v1)
            # Flat by the close: force-exit at/after the flatten bar, or on the
            # session's last bar (covers half-days and the dataset end). With
            # trade_on_close the exit settles same-session — nothing overnight.
            last_of_session = (
                i >= len(self.entry_canonical)
                or self.entry_canonical.index[i].tz_convert(_ET).date() != et.date()
            )
            if et_min >= self._flatten_min or last_of_session:
                self.position.close()
            return

        if et_min >= self._no_entry_min:  # too late to develop intraday
            return

        daily_slice = self.daily.loc[:ts]
        h1_slice = self.h1.loc[:ts]
        # HTF direction is constant until a new Daily/1H bar closes; cache it by
        # the latest (daily, h1) timestamps so the heavy market-structure call
        # runs once per HTF bar instead of once per entry bar (exact, not lossy).
        key = (
            daily_slice.index[-1] if len(daily_slice) else None,
            h1_slice.index[-1] if len(h1_slice) else None,
        )
        if key != self._dir_cache_key:
            self._dir_cache_key = key
            self._dir_cache_val = resolve_direction(
                daily_slice, h1_slice, swing_length=self.swing_length
            )
        direction = self._dir_cache_val

        ctx = build_setup_context(
            timestamp=ts,
            entry_tf=self.entry_canonical.iloc[:i],
            daily=daily_slice,
            h1=h1_slice,
            references={k: v.loc[:ts] for k, v in self.references.items()},
            sessions=self.sessions,
            direction=direction,
            swing_length=self.swing_length,
        )
        if ctx is None:
            return
        signal = self.engine.evaluate(ctx, equity=self.equity)
        if signal is None:
            return

        # Tag each trade with the confluences present (score is derivable) so a
        # backtest can attribute realized edge to each confluence, or bucket by
        # score — in-sample analysis only, never an OOS target.
        if signal.direction == "long":
            self.buy(size=signal.qty, sl=signal.stop, tp=signal.target, tag=signal.confluences)
        else:
            self.sell(size=signal.qty, sl=signal.stop, tp=signal.target, tag=signal.confluences)


def _configured_strategy(
    entry_canonical: pd.DataFrame,
    daily: pd.DataFrame,
    h1: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    sessions: Sessions,
    settings: dict,
    swing_length: int,
) -> type[ICTStrategy]:
    class _Configured(ICTStrategy):
        pass

    _Configured.entry_canonical = entry_canonical
    _Configured.daily = daily
    _Configured.h1 = h1
    _Configured.references = references
    _Configured.sessions = sessions
    _Configured.settings = settings
    _Configured.swing_length = swing_length
    return _Configured


def run_backtest(
    entry_segment: pd.DataFrame,
    daily: pd.DataFrame,
    h1: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    settings: dict,
    cash: float = 100_000.0,
    swing_length: int = 20,
) -> pd.Series:
    """Run the ICT strategy over one entry-TF segment; return backtesting stats.

    ``entry_segment`` is the canonical SPY entry-TF frame for the window (the
    detectors warm up within its head). Slippage of ``slippage_cents`` per share
    is modelled as a spread fraction off the segment's mean price.
    """
    bt_frame = to_backtesting_frame(entry_segment)
    strategy = _configured_strategy(
        entry_segment,
        daily,
        h1,
        references,
        sessions=Sessions.from_settings(settings),
        settings=settings,
        swing_length=swing_length,
    )
    mean_price = float(entry_segment["close"].mean())
    spread = (settings["backtest"]["slippage_cents"] / 100.0) / mean_price
    bt = Backtest(
        bt_frame,
        strategy,
        cash=cash,
        spread=spread,
        exclusive_orders=False,
        finalize_trades=True,
        # Fill on the signal bar's close (not next open): makes the intraday
        # flat-by-close exit settle same-session. Decisions still use only data
        # up to that close, so no lookahead is introduced.
        trade_on_close=True,
    )
    return bt.run()
