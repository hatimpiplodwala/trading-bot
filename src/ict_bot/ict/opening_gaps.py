"""Opening gaps — NDOG / NWOG (DEFERRED to v2).

The SPY cash ETF only gaps at the 09:30 ET open: NDOG = prior close -> today's
09:30 open; NWOG = Fri close -> Mon 09:30 open. Must EXCLUDE dividend ex-date
gaps (corporate-action, not liquidity). Only build once wired into the Phase 4
confluence scorer; otherwise leave as a stub (see prd.md v1 scope note).
"""

from __future__ import annotations
