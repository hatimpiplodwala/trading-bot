"""Risk & position sizing (Phase 4).

``position_size(equity, entry, stop, risk_pct)`` -> WHOLE shares (Alpaca rejects
bracket/OCO orders on fractional shares — Gotcha #9); round down and sanity-cap.
ATR14 via ``pandas_ta.atr(length=14)``; stop = entry +/- max(1.5 x ATR14,
structural invalidation). Daily loss limit halts new entries at -2% realized P&L
per ET day. Max concurrent positions: 1.
"""

from __future__ import annotations
