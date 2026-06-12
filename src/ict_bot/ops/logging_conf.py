"""Structured JSON logging (Phase 0).

All log records carry a UTC ISO-8601 timestamp; the bot reasons in UTC
everywhere. Human-facing display converts to GST (Asia/Dubai, UTC+4) via
:func:`format_gst`. Uses the stdlib only (no structlog dependency) so the
dependency set stays exactly as the build plan specifies.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import logging.handlers
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = _dt.timezone.utc
GST = ZoneInfo("Asia/Dubai")  # Gulf Standard Time, UTC+4, no DST

# Standard LogRecord attributes; anything else passed via ``extra=`` is treated
# as a structured field and merged into the JSON payload.
_RESERVED = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
    | {"message", "asctime", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object with a UTC timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": _dt.datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int | str = logging.INFO, logfile: str | None = None) -> None:
    """Install the JSON formatter on the root logger (idempotent).

    Always logs to the console. When ``logfile`` is given (the live service),
    also writes to a daily-rotating file keeping 30 days of history.
    """
    formatter = JsonFormatter()
    root = logging.getLogger()
    for handler in root.handlers[:]:  # release any prior handlers (incl. file handles)
        handler.close()
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if logfile:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            path, when="midnight", backupCount=30, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)
    # pandas_ta_classic warns on short windows (< ATR length); compute_atr already
    # falls back to a manual Wilder ATR, so silence the expected noise.
    logging.getLogger("pandas_ta_classic").setLevel(logging.ERROR)


def utcnow() -> _dt.datetime:
    """Timezone-aware current time in UTC."""
    return _dt.datetime.now(UTC)


def format_gst(ts: _dt.datetime, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format a datetime in GST for human-facing display.

    Naive datetimes are assumed to be UTC.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(GST).strftime(fmt)
