"""Walk-forward evaluation of the fixed ensemble.

The ensemble selects no parameters from the data, so the whole history is already
out-of-sample. Walk-forward here means *evaluation*, not optimization: slide a
rolling window across the series and measure the same fixed ensemble in each, to
show the edge is stable across regimes rather than carried by one lucky stretch.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.backtest.ensemble import ensemble_backtest


def walk_forward_windows(
    start: pd.Timestamp, end: pd.Timestamp, window_months: int = 12, step_months: int = 6
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rolling ``[start, end)`` windows; the final one is clipped to ``end``."""
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        win_end = cur + pd.DateOffset(months=window_months)
        if win_end >= end:
            windows.append((cur, end))
            break
        windows.append((cur, win_end))
        cur = cur + pd.DateOffset(months=step_months)
    return windows


def walk_forward_ensemble(
    symbol_df: pd.DataFrame,
    settings: dict,
    or_windows: list[str] | None = None,
    cash: float = 100_000.0,
    fill_mode: str = "next_open",
    window_months: int = 12,
    step_months: int = 6,
    min_bars: int = 500,
) -> list[dict]:
    """Run the ensemble on each rolling window; return per-window metrics."""
    windows = walk_forward_windows(symbol_df.index[0], symbol_df.index[-1],
                                   window_months, step_months)
    out: list[dict] = []
    for start, end in windows:
        seg = symbol_df[(symbol_df.index >= start) & (symbol_df.index < end)]
        if len(seg) < min_bars:
            continue
        res = ensemble_backtest(seg, settings, or_windows, cash, fill_mode)
        out.append({"start": start, "end": end, "metrics": res["metrics"]})
    return out
