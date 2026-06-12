"""FVG detection tests (Phase 2).

smc.fvg is exercised on real data in the integration smoke; here we TDD the
transform from smc's positional output to timestamped FVG dataclasses, including
mitigation state. smc encodes an unmitigated gap as MitigatedIndex 0 (or NaN);
a positive integer is the bar position where the gap was filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ict_bot.ict.fvg import FVG, fvgs_from_frame

TS = pd.date_range("2026-06-12 09:30", periods=20, freq="15min", tz="UTC")


def _frame(rows: dict[int, tuple]) -> pd.DataFrame:
    """rows: {pos: (FVG, Top, Bottom, MitigatedIndex)}."""
    f = pd.DataFrame(
        {"FVG": np.nan, "Top": np.nan, "Bottom": np.nan, "MitigatedIndex": np.nan},
        index=range(len(TS)),
    )
    for pos, (fvg, top, bottom, mit) in rows.items():
        f.loc[pos, ["FVG", "Top", "Bottom", "MitigatedIndex"]] = [fvg, top, bottom, mit]
    return f


def test_unmitigated_bullish_fvg():
    fvgs = fvgs_from_frame(_frame({5: (1.0, 756.3, 756.1, 0.0)}), TS)
    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg == FVG(
        type="bullish", top=756.3, bottom=756.1,
        formed_at=TS[5], mitigated=False, mitigated_at=None,
    )


def test_mitigated_bearish_fvg_maps_timestamp():
    fvgs = fvgs_from_frame(_frame({11: (-1.0, 756.84, 756.64, 16.0)}), TS)
    fvg = fvgs[0]
    assert fvg.type == "bearish"
    assert fvg.mitigated is True
    assert fvg.mitigated_at == TS[16]


def test_nan_mitigated_index_is_unmitigated():
    fvgs = fvgs_from_frame(_frame({3: (1.0, 100.0, 99.0, np.nan)}), TS)
    assert fvgs[0].mitigated is False
    assert fvgs[0].mitigated_at is None


def test_rows_without_fvg_are_skipped_and_order_preserved():
    fvgs = fvgs_from_frame(
        _frame({2: (1.0, 100.0, 99.0, 0.0), 9: (-1.0, 110.0, 109.0, 12.0)}), TS
    )
    assert [f.formed_at for f in fvgs] == [TS[2], TS[9]]


def test_empty_frame_returns_empty():
    assert fvgs_from_frame(_frame({}), TS) == []
