"""Alpaca broker integration (Phase 7).

``TradingClient`` (paper). Methods: ``submit_bracket_order(symbol, qty, entry,
stop, target)``, ``get_positions()``, ``get_account()``, ``cancel_all()``.
Constraints: integer shares only and regular-hours entries only (Gotcha #9).
"""

from __future__ import annotations
