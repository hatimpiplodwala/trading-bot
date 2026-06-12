"""Liquidity (Phase 2).

Wraps ``smc.liquidity()`` (BSL/SSL, sweeps/raids) and adds equal-highs/lows
detection by clustering swing points within 0.1x ATR tolerance.
"""

from __future__ import annotations
