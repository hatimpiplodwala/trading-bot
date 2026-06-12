"""Market structure (Phase 2).

Wraps ``smc.swing_highs_lows()`` and ``smc.bos_choch()``. Exposes a
``MarketStructure`` dataclass with current bias (bullish/bearish/neutral) and
the last structure level (BOS / CHoCH / MSS).
"""

from __future__ import annotations
