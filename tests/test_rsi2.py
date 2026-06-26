import numpy as np
import pandas as pd
from ict_bot.strategy.rsi2 import rsi, is_entry, is_exit


def _series(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def test_rsi_bounds_and_direction():
    # One counter-tick at the start so the opposite EWM is nonzero (a strictly
    # monotonic series divides by zero -> filled to 50, a degenerate-data edge case
    # real prices never hit). The rsi here is byte-identical to the backtest's.
    up = _series([10, 9] + list(range(10, 70)))    # rises after one down -> RSI near 100
    down = _series([60, 61] + list(range(60, 1, -1)))  # falls after one up -> RSI near 0
    assert rsi(up, 2).iloc[-1] > 90
    assert rsi(down, 2).iloc[-1] < 10


def test_is_entry_requires_oversold_in_uptrend():
    # 250 rising bars (close > SMA200), then a sharp 3-bar drop -> RSI(2) low.
    base = list(np.linspace(100, 300, 250))
    dip = [base[-1] * 0.97, base[-1] * 0.95, base[-1] * 0.93]
    close = _series(base + dip)
    assert is_entry(close, entry_t=10, trend_len=200) is True


def test_is_entry_false_below_trend():
    close = _series(list(np.linspace(300, 100, 260)))  # downtrend: close < SMA200
    assert is_entry(close, entry_t=10, trend_len=200) is False


def test_is_exit_on_recovery_above_sma5():
    close = _series(list(np.linspace(100, 120, 30)))   # last close above SMA5
    assert is_exit(close, exit_t=60, exit_sma_len=5) is True
