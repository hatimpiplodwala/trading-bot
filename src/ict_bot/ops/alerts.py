"""Notifications (Phase 7) — v1 is log-only; Telegram deferred.

A one-method :class:`Notifier` surface so the live loop can announce signals,
fills, flattens, errors and the daily summary without caring where they go. v1
ships :class:`LogNotifier` (writes to the rotating log + console). A
``TelegramNotifier`` can implement the same ``notify`` later with no loop changes.
"""

from __future__ import annotations

import logging
from typing import Protocol


class Notifier(Protocol):
    def notify(self, event: str, message: str) -> None: ...


class LogNotifier:
    """Notifier that records events to the standard logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger("ictbot.alerts")

    def notify(self, event: str, message: str) -> None:
        self._log.info("[%s] %s", event, message)
