"""Premium/discount zones (Phase 2).

A dealing range is defined by the last major swing high and low. Within it:
equilibrium = 50%, premium = above it, discount = below it. OTE (Optimal Trade
Entry) is the 62-79% retracement band — deep discount for longs, premium for
shorts. Pure math; no smc dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

# OTE retracement bounds (fraction of the leg retraced).
_OTE_LOW = 0.62
_OTE_HIGH = 0.79


@dataclass(frozen=True)
class DealingRange:
    """Dealing range between a swing ``low`` and ``high`` (low < high)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"low ({self.low}) must be < high ({self.high})")

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def equilibrium(self) -> float:
        return (self.low + self.high) / 2

    def level_at(self, fraction: float) -> float:
        """Price at ``fraction`` of the range measured from the low (0=low, 1=high)."""
        return self.low + fraction * self.size

    def position(self, price: float) -> float:
        """Fractional position of ``price`` within the range (0=low, 1=high)."""
        return (price - self.low) / self.size

    def zone(self, price: float) -> str:
        pos = self.position(price)
        if pos > 0.5:
            return "premium"
        if pos < 0.5:
            return "discount"
        return "equilibrium"

    def in_premium(self, price: float) -> bool:
        return self.position(price) > 0.5

    def in_discount(self, price: float) -> bool:
        return self.position(price) < 0.5

    def ote(self, direction: str) -> tuple[float, float]:
        """OTE price band (low, high) for a ``"bullish"`` or ``"bearish"`` setup."""
        if direction == "bullish":
            # Retrace down from the high: 79% -> deepest, 62% -> shallowest.
            return (self.level_at(1 - _OTE_HIGH), self.level_at(1 - _OTE_LOW))
        if direction == "bearish":
            # Retrace up from the low.
            return (self.level_at(_OTE_LOW), self.level_at(_OTE_HIGH))
        raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")

    def in_ote(self, price: float, direction: str) -> bool:
        lo, hi = self.ote(direction)
        return lo <= price <= hi
