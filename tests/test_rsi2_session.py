import numpy as np
import pandas as pd
from ict_bot.strategy.rsi2_session import M2Action, RSI2Session


def _series(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def _session():
    return RSI2Session(entry_t=10, exit_t=60, trend_len=200, exit_sma_len=5)


def test_enter_when_flat_and_oversold():
    close = _series(list(np.linspace(100, 300, 250)) + [291, 285, 279])
    assert _session().decide(close, has_position=False) is M2Action.ENTER


def test_none_when_flat_no_signal():
    close = _series(list(np.linspace(100, 300, 260)))  # trending up, not oversold
    assert _session().decide(close, has_position=False) is M2Action.NONE


def test_exit_when_holding_and_recovered():
    close = _series(list(np.linspace(100, 120, 30)))   # above SMA5 -> exit
    assert _session().decide(close, has_position=True) is M2Action.EXIT


def test_hold_when_holding_no_exit():
    close = _series(list(np.linspace(300, 250, 250)) + [240, 233, 228])  # still depressed
    assert _session().decide(close, has_position=True) is M2Action.HOLD


def test_parity_with_backtest_position_series():
    # Stepping decide() bar-by-bar reproduces the backtest's pos_rsi2 state machine.
    import scripts.research_strategies as rs
    rng = np.random.default_rng(0)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 400)) + np.linspace(0, 40, 400))
    backtest_pos = rs.pos_rsi2(close, entry_t=10).to_numpy()
    s, holding, live = _session(), False, []
    for i in range(len(close)):
        act = s.decide(close.iloc[: i + 1], has_position=holding)
        holding = act in (M2Action.ENTER, M2Action.HOLD)
        live.append(1.0 if holding else 0.0)
    assert np.array_equal(np.array(live), backtest_pos)
