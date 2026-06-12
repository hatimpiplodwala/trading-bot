"""Broker interface shared by the live Alpaca broker and the dry-run broker.

The live loop talks to *a broker* through this small surface, so it never knows
whether orders hit Alpaca's paper account or are merely logged. Keeping the
interface tiny is what lets the dry-run broker stand in for the real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Position:
    """An open position, normalized across brokers. ``qty`` is always positive."""

    symbol: str
    qty: int
    side: str  # "long" | "short"
    avg_entry_price: float = 0.0


class Broker(Protocol):
    """Minimal broker surface the live ORB loop depends on."""

    def get_equity(self) -> float: ...

    def get_position(self, symbol: str) -> Position | None: ...

    def submit_entry_with_stop(
        self, symbol: str, direction: str, qty: int, stop: float,
        target: float | None = None,
    ) -> str: ...

    def close_position(self, symbol: str) -> str: ...

    def is_market_open(self) -> bool: ...
