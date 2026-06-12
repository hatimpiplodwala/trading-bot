"""Telegram alerts (Phase 7).

Sends on: signal generated, order filled, order rejected, hourly heartbeat,
daily summary, error. Journal entries are post-trade and not alerted (keep
Telegram noise under ~20 msgs/day).
"""

from __future__ import annotations
