"""IEX data-quality comparison (Phase 1, Gotcha #5).

ICT signals depend on precise daily high/low extremes (where sweeps register).
This compares the IEX feed's daily extremes against a consolidated reference
(e.g. Yahoo) and yields a PASS/REVIEW verdict. Used by scripts/validate_iex.py.
"""

from __future__ import annotations

import pandas as pd


def _to_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True).normalize()
    return out


def compare_daily_extremes(iex: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Compare daily high/low between IEX and a reference on overlapping dates.

    Returns absolute percent-deviation summary stats (median, max) for both the
    high and the low, plus the number of overlapping days compared.
    """
    a = _to_dates(iex)
    b = _to_dates(reference)
    joined = a[["high", "low"]].join(
        b[["high", "low"]], lsuffix="_iex", rsuffix="_ref", how="inner"
    )

    high_dev = (joined["high_iex"] - joined["high_ref"]).abs() / joined["high_ref"] * 100
    low_dev = (joined["low_iex"] - joined["low_ref"]).abs() / joined["low_ref"] * 100

    def _stats(s: pd.Series) -> dict:
        if s.empty:
            return {"median": 0.0, "max": 0.0}
        return {"median": float(s.median()), "max": float(s.max())}

    return {
        "days": int(len(joined)),
        "high_dev_pct": _stats(high_dev),
        "low_dev_pct": _stats(low_dev),
    }


def iex_verdict(stats: dict, threshold_pct: float = 0.1) -> str:
    """PASS if both median high/low deviations are within ``threshold_pct``."""
    worst_median = max(
        stats["high_dev_pct"]["median"], stats["low_dev_pct"]["median"]
    )
    return "PASS" if worst_median <= threshold_pct else "REVIEW"
