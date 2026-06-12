"""Integration smoke for the Phase 2 detectors against real stored bars.

Skips if the local DuckDB/Parquet store has no SPY 15m data (it is gitignored,
so a fresh clone has none). Verifies the smc wrappers run end-to-end and that
already-formed FVGs do not repaint when more bars arrive (no lookahead).
"""

from __future__ import annotations

import pytest

from ict_bot.config import REPO_ROOT
from ict_bot.data.store import BarStore
from ict_bot.ict.fvg import FVG, detect_fvgs
from ict_bot.ict.liquidity import detect_equal_levels, detect_liquidity
from ict_bot.ict.order_blocks import OrderBlock, detect_order_blocks
from ict_bot.ict.structure import market_structure


@pytest.fixture(scope="module")
def spy_15m():
    df = BarStore(REPO_ROOT / "data" / "parquet").read_bars("SPY", "15m")
    if len(df) < 500:
        pytest.skip("no local SPY 15m data (run scripts/download_history.py)")
    return df


def test_market_structure_runs(spy_15m):
    ms = market_structure(spy_15m.tail(1000), swing_length=20)
    assert ms.bias in {"bullish", "bearish", "neutral"}


def test_detectors_return_expected_types(spy_15m):
    window = spy_15m.tail(1000)
    fvgs = detect_fvgs(window)
    obs = detect_order_blocks(window, swing_length=20)
    assert fvgs and all(isinstance(f, FVG) for f in fvgs)
    assert all(isinstance(o, OrderBlock) for o in obs)
    assert all(o.state in {"active", "mitigated", "breaker"} for o in obs)
    assert detect_liquidity(window, swing_length=20) is not None
    # Equal levels may be empty depending on the window, but must not error.
    assert isinstance(detect_equal_levels(window, swing_length=20), list)


def test_formed_fvgs_do_not_repaint(spy_15m):
    short = spy_15m.iloc[:300]
    longer = spy_15m.iloc[:400]

    cutoff = short.index[-5]  # only FVGs fully formed well before the cutoff
    short_fvgs = {
        (f.formed_at, f.type, f.top, f.bottom)
        for f in detect_fvgs(short)
        if f.formed_at < cutoff
    }
    long_fvgs = {
        (f.formed_at, f.type, f.top, f.bottom) for f in detect_fvgs(longer)
    }
    # Every FVG formed in the short window still exists, unchanged, in the longer
    # window — formation never depends on future bars.
    assert short_fvgs and short_fvgs <= long_fvgs
