"""NY kill zones (Phase 3).

Session windows are defined in ET wall-clock and resolved against a UTC
timestamp via the ET tzinfo, so DST is handled automatically (Gotcha #4 — never
hardcode a fixed offset). Pre-market is observe-only in v1 (Gotcha #9).

Note: a market-holiday calendar is NOT applied in v1 — these functions report
regular hours on holidays. Add ``pandas_market_calendars`` if that matters later.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

# current_kill_zone priority: silver bullet wins inside NY AM; pre-market last.
_PRIORITY = ("silver_bullet", "ny_am", "ny_pm", "ny_premarket")


def _parse(hhmm: str) -> dt.time:
    hours, minutes = hhmm.split(":")
    return dt.time(int(hours), int(minutes))


@dataclass(frozen=True)
class _Window:
    start: dt.time
    end: dt.time
    tradeable: bool


class Sessions:
    """DST-aware NY kill-zone classifier over UTC timestamps."""

    def __init__(
        self,
        kill_zones: dict[str, dict],
        regular_hours: dict[str, str],
        market_tz: str = "America/New_York",
    ) -> None:
        self.tz = ZoneInfo(market_tz)
        self.zones = {
            name: _Window(_parse(z["start"]), _parse(z["end"]), z.get("tradeable", False))
            for name, z in kill_zones.items()
        }
        self.regular = _Window(_parse(regular_hours["start"]), _parse(regular_hours["end"]), True)

    # --- helpers ---------------------------------------------------------
    def _et(self, ts: dt.datetime) -> dt.datetime:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be tz-aware (UTC)")
        return ts.astimezone(self.tz)

    @staticmethod
    def _within(et: dt.datetime, window: _Window) -> bool:
        if et.weekday() >= 5:  # Saturday/Sunday
            return False
        return window.start <= et.timetz().replace(tzinfo=None) < window.end

    def _in_zone(self, ts: dt.datetime, name: str) -> bool:
        return self._within(self._et(ts), self.zones[name])

    # --- public API ------------------------------------------------------
    def is_silver_bullet(self, ts: dt.datetime) -> bool:
        return self._in_zone(ts, "silver_bullet")

    def is_ny_am(self, ts: dt.datetime) -> bool:
        return self._in_zone(ts, "ny_am")

    def is_ny_pm(self, ts: dt.datetime) -> bool:
        return self._in_zone(ts, "ny_pm")

    def is_premarket(self, ts: dt.datetime) -> bool:
        return self._in_zone(ts, "ny_premarket")

    def is_regular_hours(self, ts: dt.datetime) -> bool:
        return self._within(self._et(ts), self.regular)

    def current_kill_zone(self, ts: dt.datetime) -> str | None:
        et = self._et(ts)
        for name in _PRIORITY:
            if name in self.zones and self._within(et, self.zones[name]):
                return name
        return None

    def is_tradeable(self, ts: dt.datetime) -> bool:
        """In a tradeable kill zone (excludes pre-market and weekends)."""
        zone = self.current_kill_zone(ts)
        return zone is not None and self.zones[zone].tradeable

    @classmethod
    def from_settings(cls, settings: dict) -> "Sessions":
        return cls(settings["kill_zones"], settings["regular_hours"])
