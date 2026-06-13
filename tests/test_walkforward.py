"""Walk-forward rolling-window generator."""

from __future__ import annotations

import pandas as pd

from ict_bot.backtest.walkforward import walk_forward_windows


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_rolling_windows_advance_by_step_and_end_at_range_end():
    wins = walk_forward_windows(_ts("2022-01-01"), _ts("2024-01-01"),
                                window_months=12, step_months=6)
    assert wins[0] == (_ts("2022-01-01"), _ts("2023-01-01"))
    assert wins[1][0] == _ts("2022-07-01")          # advanced 6 months
    assert wins[-1][1] == _ts("2024-01-01")         # last window reaches the end
    assert all(s >= _ts("2022-01-01") and e <= _ts("2024-01-01") for s, e in wins)


def test_final_partial_window_is_clipped_to_end():
    wins = walk_forward_windows(_ts("2022-01-01"), _ts("2023-04-01"),
                                window_months=12, step_months=6)
    assert wins[0][1] == _ts("2023-01-01")
    assert wins[-1][1] == _ts("2023-04-01")          # clipped tail still reaches end
