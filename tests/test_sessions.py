"""Kill-zone session tests (Phase 3) — DST correctness is the headline.

Windows are defined in ET wall-clock; the same UTC hour maps to a different ET
hour across the DST boundary, and the code must resolve that via the ET tzinfo
(Gotcha #4). Pre-market is observe-only (not tradeable).
"""

from __future__ import annotations

import datetime as dt

import pytest

from ict_bot.ict.sessions import Sessions

UTC = dt.timezone.utc

KILL_ZONES = {
    "ny_premarket": {"start": "07:00", "end": "09:30", "tradeable": False},
    "ny_am": {"start": "09:30", "end": "12:00", "tradeable": True},
    "silver_bullet": {"start": "10:00", "end": "11:00", "tradeable": True},
    "ny_pm": {"start": "13:30", "end": "16:00", "tradeable": True},
}
REGULAR_HOURS = {"start": "09:30", "end": "16:00"}


@pytest.fixture
def sessions() -> Sessions:
    return Sessions(KILL_ZONES, REGULAR_HOURS)


def test_silver_bullet_in_summer_edt(sessions):
    # 2026-07-15 14:00 UTC = 10:00 EDT (UTC-4) -> silver bullet.
    ts = dt.datetime(2026, 7, 15, 14, 0, tzinfo=UTC)
    assert sessions.is_silver_bullet(ts)
    assert sessions.current_kill_zone(ts) == "silver_bullet"


def test_silver_bullet_in_winter_est(sessions):
    # 2026-01-15 15:00 UTC = 10:00 EST (UTC-5) -> silver bullet.
    ts = dt.datetime(2026, 1, 15, 15, 0, tzinfo=UTC)
    assert sessions.is_silver_bullet(ts)


def test_same_utc_hour_differs_across_dst(sessions):
    # 14:00 UTC is 10:00 ET in summer (silver bullet) but 09:00 ET in winter (pre-market).
    summer = dt.datetime(2026, 7, 15, 14, 0, tzinfo=UTC)
    winter = dt.datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
    assert sessions.is_silver_bullet(summer)
    assert not sessions.is_silver_bullet(winter)
    assert sessions.current_kill_zone(winter) == "ny_premarket"


def test_ny_am_excludes_silver_bullet_window_by_priority(sessions):
    # 09:45 ET is NY AM but not silver bullet.
    ts = dt.datetime(2026, 7, 15, 13, 45, tzinfo=UTC)  # 09:45 EDT
    assert sessions.is_ny_am(ts)
    assert not sessions.is_silver_bullet(ts)
    assert sessions.current_kill_zone(ts) == "ny_am"


def test_ny_pm(sessions):
    ts = dt.datetime(2026, 7, 15, 18, 0, tzinfo=UTC)  # 14:00 EDT
    assert sessions.is_ny_pm(ts)
    assert sessions.current_kill_zone(ts) == "ny_pm"


def test_regular_hours_boundaries(sessions):
    open_edt = dt.datetime(2026, 7, 15, 13, 30, tzinfo=UTC)  # 09:30 EDT
    just_before = dt.datetime(2026, 7, 15, 13, 29, tzinfo=UTC)  # 09:29 EDT
    close_edt = dt.datetime(2026, 7, 15, 20, 0, tzinfo=UTC)  # 16:00 EDT
    assert sessions.is_regular_hours(open_edt)
    assert not sessions.is_regular_hours(just_before)  # pre-market
    assert not sessions.is_regular_hours(close_edt)  # end-exclusive


def test_premarket_not_tradeable(sessions):
    ts = dt.datetime(2026, 7, 15, 12, 0, tzinfo=UTC)  # 08:00 EDT
    assert sessions.current_kill_zone(ts) == "ny_premarket"
    assert not sessions.is_tradeable(ts)


def test_silver_bullet_is_tradeable(sessions):
    ts = dt.datetime(2026, 7, 15, 14, 30, tzinfo=UTC)  # 10:30 EDT
    assert sessions.is_tradeable(ts)


def test_weekend_has_no_kill_zone(sessions):
    sat = dt.datetime(2026, 7, 18, 14, 30, tzinfo=UTC)  # Saturday 10:30 EDT
    assert sessions.current_kill_zone(sat) is None
    assert not sessions.is_silver_bullet(sat)
    assert not sessions.is_tradeable(sat)


def test_requires_tz_aware(sessions):
    with pytest.raises(ValueError):
        sessions.is_silver_bullet(dt.datetime(2026, 7, 15, 14, 0))  # naive
