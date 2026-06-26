"""Live driver for the RSI-2 mean-reversion basket (the M2 strategy).

A *daily* counterpart to LiveTrader: instead of polling 15m bars and flattening by
the close, it acts once per day in a late-session window, holding each basket leg
overnight behind a wide catastrophe stop. Decisions come entirely from RSI2Session
(shared with the backtest); this module only handles seeding daily bars, the
once-a-day decision pass, per-leg execution, and restart reconciliation (which —
unlike ORB — never force-flattens, because holding overnight is by design).
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from ict_bot.ops.live_trader import next_boundary, timeframe_minutes
from ict_bot.strategy.rsi2_session import M2Action

_ET = "America/New_York"
_UTC = dt.timezone.utc
log = logging.getLogger(__name__)


class M2Trader:
    """Drives the RSI-2 basket live: seed -> once-daily decide -> hold overnight."""

    def __init__(self, symbols, feed, broker, state, notifier, session, *,
                 timeframe: str = "1d", seed_days: int = 400, poll_buffer_s: int = 20,
                 heartbeat_hours: float = 1.0, decision_after: str = "15:45",
                 stop_pct: float = 0.10) -> None:
        self._symbols = list(symbols)
        self._feed = feed
        self._broker = broker
        self._state = state
        self._notifier = notifier
        self._session = session
        self._timeframe = timeframe
        self._interval_min = timeframe_minutes("15m")  # wake every 15m to check the clock
        self._seed_days = seed_days
        self._poll_buffer_s = poll_buffer_s
        self._heartbeat_hours = heartbeat_hours
        self._decision_after = decision_after
        self._stop_pct = stop_pct
        self._frames: dict[str, pd.DataFrame] = {}
        self._entries: dict[str, dict] = {}
        self._acted_on: dt.date | None = None
        self._stopped = False
        self._last_heartbeat: dt.datetime | None = None

    # --- lifecycle -------------------------------------------------------
    def request_stop(self) -> None:
        self._stopped = True

    def _target_shares(self, equity: float, price: float) -> int:
        if price <= 0:
            return 0
        return int((equity / len(self._symbols)) / price)

    # --- seeding + reconcile --------------------------------------------
    def seed(self) -> None:
        end = dt.datetime.now(_UTC)
        start = end - dt.timedelta(days=self._seed_days)
        for sym in self._symbols:
            bars = self._feed.get_historical_bars(sym, self._timeframe, start, end)
            if bars is not None and not bars.empty:
                self._frames[sym] = bars.sort_index()
        self.reconcile()
        log.info("seeded M2 basket %s", self._symbols)

    def reconcile(self) -> None:
        """Rebuild per-symbol tracking from the broker (+ state). Never flattens —
        the basket holds overnight by design."""
        for sym in self._symbols:
            pos = self._broker.get_position(sym)
            if pos is None:
                self._entries.pop(sym, None)
                continue
            sig = self._state.last_signal(sym)
            if sig is not None:
                self._entries[sym] = {"qty": sig["qty"], "entry_price": sig["entry"],
                                      "entry_ts": pd.Timestamp(sig["ts"]),
                                      "stop": sig["stop"]}
            else:
                self._entries[sym] = {"qty": pos.qty, "entry_price": pos.avg_entry_price,
                                      "entry_ts": pd.Timestamp(dt.datetime.now(_UTC)),
                                      "stop": 0.0}

    # --- the once-a-day decision pass -----------------------------------
    def _in_window(self, now: dt.datetime) -> bool:
        et = pd.Timestamp(now).tz_convert(_ET)
        hhmm = f"{et.hour:02d}:{et.minute:02d}"
        return hhmm >= self._decision_after and self._acted_on != et.date()

    def decide_and_act(self, now: dt.datetime | None = None) -> None:
        now = now or dt.datetime.now(_UTC)
        if not self._in_window(now):
            return
        equity = self._broker.get_equity()
        for sym in self._symbols:
            frame = self._frames.get(sym)
            if frame is None or frame.empty:
                continue
            close = frame["close"]
            price = float(close.iloc[-1])
            has_pos = self._broker.get_position(sym) is not None
            action = self._session.decide(close, has_pos)
            if action is M2Action.ENTER:
                self._enter(sym, price, equity, now)
            elif action is M2Action.EXIT:
                self._exit(sym, price, now)
        self._acted_on = pd.Timestamp(now).tz_convert(_ET).date()

    def _enter(self, symbol: str, price: float, equity: float, now) -> None:
        et_date = pd.Timestamp(now).tz_convert(_ET).date()
        if self._state.has_traded_today(symbol, et_date):
            return
        qty = self._target_shares(equity, price)
        if qty <= 0:
            return
        stop = round(price * (1 - self._stop_pct), 2)
        order_id = self._broker.submit_entry_with_stop(symbol, "long", qty, stop, None)
        self._state.record_signal(ts=pd.Timestamp(now).isoformat(), et_date=et_date,
                                  symbol=symbol, direction="long", entry=price,
                                  stop=stop, target=None, qty=qty, order_id=order_id)
        self._entries[symbol] = {"qty": qty, "entry_price": price,
                                 "entry_ts": pd.Timestamp(now), "stop": stop}
        self._notifier.notify("signal", f"{symbol} long x{qty} @ {price:.2f} stop {stop:.2f}")

    def _exit(self, symbol: str, price: float, now, reason: str = "rsi-recover") -> None:
        self._broker.cancel_orders(symbol)
        self._broker.close_position(symbol)
        e = self._entries.pop(symbol, None)
        if e is not None:
            pnl = (price - e["entry_price"]) * e["qty"]
            self._state.record_trade(symbol=symbol, entry_ts=e["entry_ts"].isoformat(),
                                     exit_ts=pd.Timestamp(now).isoformat(), direction="long",
                                     entry_price=e["entry_price"], exit_price=price,
                                     qty=e["qty"], pnl=pnl)
            self._notifier.notify("close", f"{symbol} {reason} pnl={pnl:.2f}")

    def flatten_all(self) -> None:
        """Close every basket leg (+ cancel its stop). Used by the M2->ORB switch."""
        now = dt.datetime.now(_UTC)
        for sym in self._symbols:
            if self._broker.get_position(sym) is not None:
                frame = self._frames.get(sym)
                price = float(frame["close"].iloc[-1]) if frame is not None and not frame.empty \
                    else (self._entries.get(sym, {}).get("entry_price", 0.0))
                self._exit(sym, price, now, reason="flatten")

    # --- service loop ----------------------------------------------------
    def _heartbeat_due(self, now: dt.datetime) -> bool:
        if self._heartbeat_hours <= 0 or self._last_heartbeat is None:
            return False
        if (now - self._last_heartbeat).total_seconds() >= self._heartbeat_hours * 3600:
            self._last_heartbeat = now
            return True
        return False

    def _emit_heartbeat(self) -> None:
        held = [s for s in self._symbols if self._broker.get_position(s) is not None]
        self._notifier.notify("heartbeat", f"M2 alive — holding {held or 'flat'}")

    def _interruptible_sleep(self, seconds: float, sleep_fn) -> None:
        remaining = seconds
        while remaining > 0 and not self._stopped:
            sleep_fn(min(1.0, remaining))
            remaining -= 1.0

    def run(self, *, max_iterations: int | None = None,
            sleep_fn=time.sleep, now_fn=None) -> None:
        now_fn = now_fn or (lambda: dt.datetime.now(_UTC))
        self.seed()
        self._notifier.notify("start", f"M2 basket live loop up for {self._symbols}")
        self._last_heartbeat = now_fn()
        iterations = 0
        while not self._stopped:
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1
            now = now_fn()
            target = next_boundary(now, self._interval_min, self._poll_buffer_s)
            self._interruptible_sleep(max((target - now).total_seconds(), 0), sleep_fn)
            if self._stopped:
                break
            try:
                tick = now_fn()
                if self._heartbeat_due(tick):
                    self._emit_heartbeat()
                if self._broker.is_market_open():
                    self.seed()  # refresh daily frames
                    self.decide_and_act(tick)
            except Exception as exc:  # never let one bad cycle kill the loop
                log.exception("M2 cycle failed")
                self._notifier.notify("error", f"M2 cycle failed: {exc}")
        self._notifier.notify("stop", f"M2 basket loop stopping for {self._symbols}")
