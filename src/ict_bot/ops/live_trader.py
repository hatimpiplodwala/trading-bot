"""Live paper-trading driver (Phase 7) — the ORB loop around the decision core.

Glues the data feed -> :class:`~ict_bot.strategy.orb_session.ORBSession` -> broker,
with durable state and notifications. The decision (when/what/how-much to trade)
lives entirely in ORBSession, which the backtest also drives — so the bot trades
exactly as it was validated. This module only handles the *real-world* concerns:
seeding history, waking on each bar close, executing actions, and reconciling what
the broker actually did (e.g. a stop that fired between polls).

``handle_bar`` (one decision+execution step) and ``next_boundary`` (poll schedule)
are pure enough to unit-test; the live ``run`` loop and REST fetch are validated by
a manual ``--dry-run`` during market hours.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from ict_bot.strategy.orb_session import Action, Enter

_ET = "America/New_York"
_UTC = dt.timezone.utc
log = logging.getLogger(__name__)


def next_boundary(now: dt.datetime, interval_min: int = 15, buffer_s: int = 0) -> dt.datetime:
    """The next ``interval_min`` grid time strictly after ``now``, plus a buffer.

    The buffer lets the just-closed bar settle at the data provider before we poll.
    """
    floor = now.replace(
        minute=(now.minute // interval_min) * interval_min, second=0, microsecond=0
    )
    return floor + dt.timedelta(minutes=interval_min, seconds=buffer_s)


class LiveTrader:
    """Drives ORB live on a single symbol: seed -> poll on bar close -> execute."""

    def __init__(
        self, symbol, feed, broker, state, notifier, session,
        timeframe: str = "15m", seed_days: int = 4, poll_buffer_s: int = 20,
        heartbeat_hours: float = 1.0, max_frame_bars: int = 500,
    ) -> None:
        self._symbol = symbol
        self._feed = feed
        self._broker = broker
        self._state = state
        self._notifier = notifier
        self._session = session
        self._timeframe = timeframe
        self._seed_days = seed_days
        self._poll_buffer_s = poll_buffer_s
        self._heartbeat_hours = heartbeat_hours
        self._max_frame_bars = max_frame_bars
        self._frame: pd.DataFrame | None = None
        self._last_ts: pd.Timestamp | None = None
        # The trade we believe is open, for reconciliation + the local trade log.
        self._entry: dict | None = None
        self._stopped = False
        self._last_heartbeat: dt.datetime | None = None

    # --- lifecycle / health ----------------------------------------------
    def request_stop(self) -> None:
        """Ask the run loop to finish the current poll and exit (signal-safe)."""
        self._stopped = True

    def _heartbeat_due(self, now: dt.datetime) -> bool:
        if self._heartbeat_hours <= 0 or self._last_heartbeat is None:
            return False
        if (now - self._last_heartbeat).total_seconds() >= self._heartbeat_hours * 3600:
            self._last_heartbeat = now
            return True
        return False

    def _emit_heartbeat(self) -> None:
        pos = self._broker.get_position(self._symbol)
        held = f"{pos.side} x{pos.qty}" if pos is not None else "flat"
        self._notifier.notify("heartbeat", f"{self._symbol} alive — {held}")

    def _trim_frame(self) -> None:
        """Cap the rolling frame so a multi-day run keeps stable memory."""
        if self._max_frame_bars and self._frame is not None and len(self._frame) > self._max_frame_bars:
            self._frame = self._frame.iloc[-self._max_frame_bars:]

    def _interruptible_sleep(self, seconds: float, sleep_fn) -> None:
        """Sleep in ~1s slices so a stop requested mid-sleep is honored promptly.

        One long sleep would swallow the Ctrl-C our signal handler turns into
        :meth:`request_stop` — the handler may run without interrupting the OS
        sleep, so the loop wouldn't notice until the (up to 15-min) sleep ended
        and NSSM would force-kill instead. Slicing lets the loop observe the flag
        within a second and exit cleanly (logging the stop, leaving the position
        and its broker stop alone).
        """
        remaining = seconds
        while remaining > 0 and not self._stopped:
            sleep_fn(min(1.0, remaining))
            remaining -= 1.0

    # --- one decision + execution step -----------------------------------
    def handle_bar(self, frame: pd.DataFrame):
        """Decide and act on the latest bar of ``frame``; return the action."""
        ts = frame.index[-1]
        et = ts.tz_convert(_ET)
        et_date = et.date()
        close = float(frame["close"].iloc[-1])
        self._last_ts = ts

        pos = self._broker.get_position(self._symbol)

        # A position we were tracking has vanished -> the broker stop fired.
        if self._entry is not None and pos is None:
            self._record_close(self._entry["stop"], ts, reason="stop")

        equity = self._broker.get_equity()
        action = self._session.on_bar(frame, equity, pos is not None)

        if isinstance(action, Enter):
            return self._enter(action, ts, et_date, close)
        if action is Action.FLATTEN:
            self._broker.close_position(self._symbol)
            self._record_close(close, ts, reason="flat-by-close")
            self._notifier.notify("flatten", f"{self._symbol} flat @ {close:.2f}")
        return action

    def _enter(self, action: Enter, ts, et_date, close: float):
        # Durable one-per-day guard — survives a mid-session restart.
        if self._state.has_traded_today(self._symbol, et_date):
            self._notifier.notify("skip", f"{self._symbol} already traded {et_date}")
            return action
        order_id = self._broker.submit_entry_with_stop(
            self._symbol, action.direction, action.qty, action.stop, action.target
        )
        self._state.record_signal(
            ts=ts.isoformat(), et_date=et_date, symbol=self._symbol,
            direction=action.direction, entry=close, stop=action.stop,
            target=action.target, qty=action.qty, order_id=order_id,
        )
        self._entry = {
            "direction": action.direction, "qty": action.qty, "entry_price": close,
            "entry_ts": ts, "stop": action.stop, "target": action.target,
        }
        self._notifier.notify(
            "signal",
            f"{self._symbol} {action.direction} x{action.qty} @ {close:.2f} "
            f"stop {action.stop:.2f}",
        )
        return action

    def _record_close(self, exit_price: float, exit_ts, reason: str) -> None:
        """Record an (approximate) closed trade and feed the daily-loss gate.

        Exit price is a proxy (stop level for a stop-out, bar close for a timed
        flatten); Alpaca's paper account holds the authoritative fills.
        """
        if self._entry is None:
            return
        e = self._entry
        sign = 1 if e["direction"] == "long" else -1
        pnl = (exit_price - e["entry_price"]) * e["qty"] * sign
        self._session.register_realized(pnl)
        self._state.record_trade(
            symbol=self._symbol, entry_ts=e["entry_ts"].isoformat(),
            exit_ts=exit_ts.isoformat(), direction=e["direction"],
            entry_price=e["entry_price"], exit_price=exit_price, qty=e["qty"], pnl=pnl,
        )
        self._notifier.notify("close", f"{self._symbol} {reason} pnl={pnl:.2f}")
        self._entry = None

    # --- live I/O (validated by manual dry-run, not unit tests) -----------
    def seed(self) -> None:
        """Fetch recent history and warm the session's opening-range state."""
        end = dt.datetime.now(_UTC)
        start = end - dt.timedelta(days=self._seed_days + 4)  # pad for weekends/ATR
        self._frame = self._feed.get_historical_bars(self._symbol, self._timeframe, start, end)
        if self._frame is None or self._frame.empty:
            log.warning("seed: no history for %s", self._symbol)
            return
        equity = self._broker.get_equity()
        has_pos = self._broker.get_position(self._symbol) is not None
        for k in range(1, len(self._frame) + 1):  # replay to rebuild today's OR / traded flag
            self._session.on_bar(self._frame.iloc[:k], equity, has_pos)
        self._last_ts = self._frame.index[-1]
        log.info("seeded %d %s bars through %s", len(self._frame), self._symbol, self._last_ts)

    def poll_once(self) -> None:
        """Fetch any newly-closed bars and process each in order."""
        end = dt.datetime.now(_UTC)
        start = (self._last_ts.to_pydatetime() - dt.timedelta(hours=1)) if self._last_ts is not None \
            else end - dt.timedelta(days=self._seed_days + 4)
        recent = self._feed.get_historical_bars(self._symbol, self._timeframe, start, end)
        if recent is None or recent.empty:
            return
        if self._frame is None:
            self._frame = recent
        new = recent if self._last_ts is None else recent[recent.index > self._last_ts]
        for ts in new.index:
            self._frame.loc[ts] = recent.loc[ts]
            self._frame = self._frame.sort_index()
            self.handle_bar(self._frame.loc[:ts])
        self._trim_frame()

    def run(self, *, max_iterations: int | None = None,
            sleep_fn=time.sleep, now_fn=None) -> None:
        """Service loop: seed, then poll on each bar boundary during market hours.

        Polls only while the market is open and emits an hourly heartbeat.
        :meth:`request_stop` (wired to SIGINT/SIGTERM in ``main``) makes it finish
        the current cycle and exit cleanly, leaving any open position and its
        resting broker stop untouched. ``max_iterations``/``sleep_fn``/``now_fn``
        exist so the loop is testable without real time.
        """
        now_fn = now_fn or (lambda: dt.datetime.now(_UTC))
        self.seed()
        self._notifier.notify("start", f"ORB live loop up for {self._symbol}")
        self._last_heartbeat = now_fn()
        iterations = 0
        while not self._stopped:
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1
            now = now_fn()
            target = next_boundary(now, 15, self._poll_buffer_s)
            self._interruptible_sleep(max((target - now).total_seconds(), 0), sleep_fn)
            if self._stopped:
                break
            try:
                tick = now_fn()
                if self._heartbeat_due(tick):
                    self._emit_heartbeat()
                if self._broker.is_market_open():
                    self.poll_once()
            except Exception as exc:  # never let one bad poll kill the loop
                log.exception("poll failed")
                self._notifier.notify("error", f"{self._symbol} poll failed: {exc}")
        self._notifier.notify("stop", f"ORB live loop stopping for {self._symbol}")
