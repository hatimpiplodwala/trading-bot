"""IEX data-quality comparison tests (Phase 1, Gotcha #5).

Pure logic behind scripts/validate_iex.py: compare daily high/low extremes from
the IEX feed against a consolidated reference and produce a PASS/REVIEW verdict.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.data.quality import compare_daily_extremes, iex_verdict


def _daily(dates: list[str], highs: list[float], lows: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="timestamp")
    return pd.DataFrame({"high": highs, "low": lows}, index=idx)


def test_identical_frames_zero_deviation():
    iex = _daily(["2026-06-10", "2026-06-11"], [101.0, 102.0], [99.0, 100.0])
    stats = compare_daily_extremes(iex, iex)
    assert stats["days"] == 2
    assert stats["high_dev_pct"]["max"] == 0.0
    assert stats["low_dev_pct"]["max"] == 0.0


def test_known_deviation_computed():
    iex = _daily(["2026-06-10"], [101.0], [99.0])
    # Reference high is 100 -> IEX 101 is +1% deviation; low 100 vs 99 -> 1%.
    ref = _daily(["2026-06-10"], [100.0], [100.0])
    stats = compare_daily_extremes(iex, ref)
    assert stats["days"] == 1
    assert round(stats["high_dev_pct"]["max"], 4) == 1.0
    assert round(stats["low_dev_pct"]["max"], 4) == 1.0


def test_only_overlapping_dates_compared():
    iex = _daily(["2026-06-10", "2026-06-11"], [101.0, 102.0], [99.0, 100.0])
    ref = _daily(["2026-06-11", "2026-06-12"], [102.0, 103.0], [100.0, 101.0])
    stats = compare_daily_extremes(iex, ref)
    assert stats["days"] == 1  # only 2026-06-11 overlaps


def test_verdict_pass_when_within_threshold():
    stats = {"high_dev_pct": {"median": 0.02}, "low_dev_pct": {"median": 0.03}}
    assert iex_verdict(stats, threshold_pct=0.1) == "PASS"


def test_verdict_review_when_exceeding_threshold():
    stats = {"high_dev_pct": {"median": 0.5}, "low_dev_pct": {"median": 0.03}}
    assert iex_verdict(stats, threshold_pct=0.1) == "REVIEW"
