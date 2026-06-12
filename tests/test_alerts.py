"""Notifier — v1 is log-only (Telegram deferred), behind a tiny interface."""

from __future__ import annotations

import logging

from ict_bot.ops.alerts import LogNotifier


def test_log_notifier_emits_event_and_message(caplog):
    notifier = LogNotifier()
    with caplog.at_level(logging.INFO):
        notifier.notify("signal", "SPY long x199 @ 500.00")
    messages = [r.getMessage() for r in caplog.records]
    assert any("signal" in m and "SPY long x199 @ 500.00" in m for m in messages)
