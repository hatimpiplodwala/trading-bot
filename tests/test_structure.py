"""Market structure tests (Phase 2).

The smc swing/BOS/CHOCH detection is exercised on real data in an integration
smoke; here we TDD our interpretation: deriving current bias + last structure
level from smc's bos_choch output, counting only structure that has actually
been broken (no lookahead on unbroken levels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ict_bot.ict.structure import MarketStructure, interpret_structure


def _bc(rows: list[dict]) -> pd.DataFrame:
    """Build a bos_choch-shaped frame; rows give BOS/CHOCH/Level/BrokenIndex."""
    n = max((r["i"] for r in rows), default=0) + 5
    frame = pd.DataFrame(
        {
            "BOS": np.nan,
            "CHOCH": np.nan,
            "Level": np.nan,
            "BrokenIndex": np.nan,
        },
        index=range(n),
    )
    for r in rows:
        frame.loc[r["i"], "BOS"] = r.get("BOS", np.nan)
        frame.loc[r["i"], "CHOCH"] = r.get("CHOCH", np.nan)
        frame.loc[r["i"], "Level"] = r.get("Level", np.nan)
        frame.loc[r["i"], "BrokenIndex"] = r.get("BrokenIndex", np.nan)
    return frame


def test_no_events_is_neutral():
    ms = interpret_structure(_bc([]))
    assert ms == MarketStructure(bias="neutral", last_event=None, last_level=None, broken_index=None)


def test_single_bullish_bos():
    ms = interpret_structure(_bc([{"i": 2, "BOS": 1.0, "Level": 105.0, "BrokenIndex": 10.0}]))
    assert ms.bias == "bullish"
    assert ms.last_event == "BOS"
    assert ms.last_level == 105.0
    assert ms.broken_index == 10


def test_single_bearish_bos():
    ms = interpret_structure(_bc([{"i": 2, "BOS": -1.0, "Level": 95.0, "BrokenIndex": 8.0}]))
    assert ms.bias == "bearish"


def test_most_recently_broken_event_wins():
    # Bullish BOS broken at 10, then bearish CHOCH broken at 20 -> bearish now.
    ms = interpret_structure(_bc([
        {"i": 1, "BOS": 1.0, "Level": 105.0, "BrokenIndex": 10.0},
        {"i": 5, "CHOCH": -1.0, "Level": 98.0, "BrokenIndex": 20.0},
    ]))
    assert ms.bias == "bearish"
    assert ms.last_event == "CHOCH"
    assert ms.last_level == 98.0
    assert ms.broken_index == 20


def test_recency_is_by_broken_index_not_row():
    # CHOCH sits at a later row but was broken earlier; BOS broken later wins.
    ms = interpret_structure(_bc([
        {"i": 1, "BOS": 1.0, "Level": 105.0, "BrokenIndex": 30.0},
        {"i": 9, "CHOCH": -1.0, "Level": 98.0, "BrokenIndex": 10.0},
    ]))
    assert ms.bias == "bullish"
    assert ms.broken_index == 30


def test_unbroken_events_are_ignored():
    # Event present but never broken (NaN BrokenIndex) -> still neutral.
    ms = interpret_structure(_bc([{"i": 2, "BOS": 1.0, "Level": 105.0}]))
    assert ms.bias == "neutral"
