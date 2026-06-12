"""SMT divergence (Phase 3, v1 core).

Detects correlation breaks across SPY vs QQQ vs IWM: e.g. SPY makes a higher
high but QQQ does not (bearish SMT), or vice versa. Requires time-aligned bars
across all three symbols and inherits the IEX data-quality caveat threefold
(validate QQQ/IWM extremes in the Phase 1 check). QQQ/IWM are divergence input
only — never tradeable instruments in v1 (Gotcha #8).
"""

from __future__ import annotations
