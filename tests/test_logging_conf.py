"""Logging config — JSON formatting plus optional daily-rotating file output."""

from __future__ import annotations

import logging

from ict_bot.ops.logging_conf import configure_logging


def test_configure_logging_writes_rotating_file(tmp_path):
    logfile = tmp_path / "ictbot.log"
    configure_logging(logfile=str(logfile))
    try:
        logging.getLogger("phase8.test").info("hello-phase-8")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert logfile.exists()
        assert "hello-phase-8" in logfile.read_text(encoding="utf-8")
    finally:
        configure_logging()  # detach the file handler so tmp_path can be cleaned up
