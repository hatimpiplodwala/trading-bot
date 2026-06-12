"""Opening range (DEFERRED to v2).

Opening Range = high/low of 09:30-10:00 ET. NOTE: the ICT "Midnight Open"
(00:00 ET) is intentionally NOT implemented — there is no print when the SPY
cash ETF is closed (it is an ES-futures concept; cut from v1). See prd.md.
"""

from __future__ import annotations
