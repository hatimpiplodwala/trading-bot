"""Dry-run broker — same surface as :class:`~ict_bot.broker.alpaca.AlpacaBroker`,
but it never submits anything. It logs and records intended orders and tracks a
notional position so the live loop can be exercised end-to-end (enter -> hold ->
flatten) on real market data with zero risk before going to the paper account.
"""

from __future__ import annotations

import logging

from ict_bot.broker.base import Position

log = logging.getLogger(__name__)


class DryRunBroker:
    """In-memory stand-in broker for ``--dry-run``."""

    def __init__(
        self, equity: float = 100_000.0, market_open: bool = True,
        equity_fn=None, market_open_fn=None,
    ) -> None:
        # Equity and the market clock can delegate to a real (read-only) broker so
        # a dry-run sizes against the real account and respects real market hours;
        # positions are always the in-memory simulated ones.
        self._equity = float(equity)
        self._market_open = bool(market_open)
        self._equity_fn = equity_fn
        self._market_open_fn = market_open_fn
        self._positions: dict[str, Position] = {}
        self.orders: list[dict] = []
        self._seq = 0

    def get_equity(self) -> float:
        return float(self._equity_fn()) if self._equity_fn else self._equity

    def is_market_open(self) -> bool:
        return bool(self._market_open_fn()) if self._market_open_fn else self._market_open

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def submit_entry_with_stop(
        self, symbol: str, direction: str, qty: int, stop: float,
        target: float | None = None,
    ) -> str:
        self._seq += 1
        oid = f"dryrun-{self._seq}"
        self._positions[symbol] = Position(symbol, int(qty), direction)
        self.orders.append({
            "kind": "entry", "id": oid, "symbol": symbol, "direction": direction,
            "qty": int(qty), "stop": stop, "target": target,
        })
        log.info("DRY-RUN entry %s %s x%d stop=%.2f target=%s",
                 direction, symbol, int(qty), stop, target)
        return oid

    def cancel_orders(self, symbol: str) -> None:
        self.orders.append({"kind": "cancel", "symbol": symbol})
        log.info("DRY-RUN cancel open orders %s", symbol)

    def close_position(self, symbol: str) -> str:
        self._seq += 1
        oid = f"dryrun-{self._seq}"
        self._positions.pop(symbol, None)
        self.orders.append({"kind": "close", "id": oid, "symbol": symbol})
        log.info("DRY-RUN close %s", symbol)
        return oid
