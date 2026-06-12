"""Integration smoke for Phase 3 (bias + SMT) against real stored bars.

Skips if the local store has no data. Verifies the smc-backed wrappers run
end-to-end and return valid values.
"""

from __future__ import annotations

import pytest

from ict_bot.config import REPO_ROOT
from ict_bot.data.store import BarStore
from ict_bot.ict.bias import HTFBias, htf_bias
from ict_bot.ict.smt import SMTSignal, detect_smt


@pytest.fixture(scope="module")
def store():
    return BarStore(REPO_ROOT / "data" / "parquet")


def test_htf_bias_runs(store):
    daily = store.read_bars("SPY", "1d")
    h1 = store.read_bars("SPY", "1h")
    if len(daily) < 60 or len(h1) < 200:
        pytest.skip("no local SPY 1d/1h data")
    bias = htf_bias(daily, h1.tail(1500), swing_length=20)
    assert isinstance(bias, HTFBias)
    assert bias.bias in {"bullish", "bearish", "neutral"}
    # Gating is consistent with the resolved bias.
    assert bias.allows_long() == (bias.bias == "bullish")


def test_smt_runs_against_qqq_iwm(store):
    spy = store.read_bars("SPY", "15m")
    qqq = store.read_bars("QQQ", "15m")
    iwm = store.read_bars("IWM", "15m")
    if min(len(spy), len(qqq), len(iwm)) < 500:
        pytest.skip("no local SPY/QQQ/IWM 15m data")
    signal = detect_smt(spy.tail(1000), {"QQQ": qqq, "IWM": iwm}, swing_length=20)
    assert isinstance(signal, SMTSignal)
    assert signal.type in {"bullish", "bearish", None}
    if signal.type is not None:
        assert signal.reference in {"QQQ", "IWM"}
