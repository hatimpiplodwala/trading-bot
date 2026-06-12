"""Broker layer: order-request construction (pure) and the dry-run broker.

The live Alpaca calls are validated by a manual dry-run during market hours, not
here. These tests pin the order *shape* (OTO market+stop for ORB's no-target case,
bracket when a target is configured) and the DryRunBroker's bookkeeping, both
without touching the network.
"""

from __future__ import annotations

from alpaca.trading.enums import OrderClass, OrderSide

from ict_bot.broker.alpaca import build_entry_order
from ict_bot.broker.dryrun import DryRunBroker


def test_build_entry_order_long_no_target_is_oto_with_stop():
    req = build_entry_order("SPY", "long", 199, stop=499.5, target=None)
    assert req.symbol == "SPY"
    assert int(req.qty) == 199
    assert req.side == OrderSide.BUY
    assert req.order_class == OrderClass.OTO          # market entry + resting stop
    assert float(req.stop_loss.stop_price) == 499.5
    assert req.take_profit is None


def test_build_entry_order_short_with_target_is_bracket():
    req = build_entry_order("SPY", "short", 50, stop=505.0, target=495.0)
    assert req.side == OrderSide.SELL
    assert req.order_class == OrderClass.BRACKET
    assert float(req.stop_loss.stop_price) == 505.0
    assert float(req.take_profit.limit_price) == 495.0


def test_dryrun_tracks_position_and_records_orders():
    b = DryRunBroker(equity=100_000.0, market_open=True)
    assert b.get_position("SPY") is None

    oid = b.submit_entry_with_stop("SPY", "long", 199, stop=499.5)
    assert oid
    pos = b.get_position("SPY")
    assert pos is not None and pos.qty == 199 and pos.side == "long"

    b.close_position("SPY")
    assert b.get_position("SPY") is None
    assert [o["kind"] for o in b.orders] == ["entry", "close"]


def test_dryrun_equity_and_clock():
    b = DryRunBroker(equity=50_000.0, market_open=False)
    assert b.get_equity() == 50_000.0
    assert b.is_market_open() is False


def test_dryrun_delegates_equity_and_clock_to_real_broker():
    # In --dry-run we borrow the real account's equity + market clock (read-only).
    b = DryRunBroker(equity_fn=lambda: 123_456.0, market_open_fn=lambda: True)
    assert b.get_equity() == 123_456.0
    assert b.is_market_open() is True
