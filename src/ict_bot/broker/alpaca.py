"""Alpaca broker integration (Phase 7) — paper account.

Wraps ``TradingClient`` behind the small :class:`~ict_bot.broker.base.Broker`
surface the live loop uses. ORB's default has no take-profit (exit is the timed
flatten), so a plain breakout entry is an **OTO**: a market entry that, once
filled, leaves a resting stop at the broker — so a crashed bot can't strand an
unprotected position. When a target is configured we submit a full **bracket**.

Constraints (Gotcha #9): integer shares only (enforced by ``risk.position_size``)
and regular-hours entries only (gated by the loop via :meth:`is_market_open`).
"""

from __future__ import annotations

import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, PositionSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from ict_bot.broker.base import Position

log = logging.getLogger(__name__)


def build_entry_order(
    symbol: str, direction: str, qty: int, stop: float, target: float | None = None
) -> MarketOrderRequest:
    """Build the market entry order — OTO (entry+stop) or bracket (entry+stop+TP)."""
    side = OrderSide.BUY if direction == "long" else OrderSide.SELL
    stop_leg = StopLossRequest(stop_price=round(float(stop), 2))
    if target is None:
        return MarketOrderRequest(
            symbol=symbol, qty=int(qty), side=side, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OTO, stop_loss=stop_leg,
        )
    return MarketOrderRequest(
        symbol=symbol, qty=int(qty), side=side, time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET, stop_loss=stop_leg,
        take_profit=TakeProfitRequest(limit_price=round(float(target), 2)),
    )


class AlpacaBroker:
    """Paper-account broker over Alpaca's ``TradingClient``."""

    def __init__(self, api_key: str, secret: str, paper: bool = True) -> None:
        self._client = TradingClient(api_key, secret, paper=paper)

    def get_equity(self) -> float:
        return float(self._client.get_account().equity)

    def get_position(self, symbol: str) -> Position | None:
        try:
            p = self._client.get_open_position(symbol)
        except Exception:  # no open position -> Alpaca raises
            return None
        side = "long" if p.side == PositionSide.LONG else "short"
        return Position(
            symbol=symbol, qty=abs(int(float(p.qty))), side=side,
            avg_entry_price=float(p.avg_entry_price),
        )

    def submit_entry_with_stop(
        self, symbol: str, direction: str, qty: int, stop: float,
        target: float | None = None,
    ) -> str:
        order = self._client.submit_order(
            build_entry_order(symbol, direction, qty, stop, target)
        )
        log.info("submitted %s %s x%d stop=%.2f target=%s id=%s",
                 direction, symbol, int(qty), stop, target, order.id)
        return str(order.id)

    def close_position(self, symbol: str) -> str:
        order = self._client.close_position(symbol)
        log.info("closed %s id=%s", symbol, getattr(order, "id", ""))
        return str(getattr(order, "id", ""))

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)
