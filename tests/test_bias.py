"""HTF bias tests (Phase 3).

Daily structure leads; the 1H only vetoes on direct contradiction. The
CHoCH -> bias flip itself is covered by structure.interpret_structure; here we
test the Daily+1H combination rule and the long/short gating.
"""

from __future__ import annotations

import pytest

from ict_bot.ict.bias import HTFBias, combine_bias


@pytest.mark.parametrize(
    "daily,h1,expected",
    [
        ("bullish", "bullish", "bullish"),
        ("bullish", "neutral", "bullish"),   # daily leads when 1H is neutral
        ("bullish", "bearish", "neutral"),   # direct conflict -> stand aside
        ("bearish", "bearish", "bearish"),
        ("bearish", "neutral", "bearish"),
        ("bearish", "bullish", "neutral"),
        ("neutral", "bullish", "neutral"),   # no daily bias -> neutral
        ("neutral", "neutral", "neutral"),
    ],
)
def test_combine_bias(daily, h1, expected):
    assert combine_bias(daily, h1) == expected


def test_allows_long_only_when_bullish():
    assert HTFBias("bullish", "bullish", "bullish").allows_long()
    assert not HTFBias("bullish", "bullish", "bullish").allows_short()


def test_allows_short_only_when_bearish():
    b = HTFBias("bearish", "bearish", "neutral")
    assert b.allows_short()
    assert not b.allows_long()


def test_neutral_allows_neither():
    n = HTFBias("neutral", "bullish", "bearish")
    assert not n.allows_long()
    assert not n.allows_short()
