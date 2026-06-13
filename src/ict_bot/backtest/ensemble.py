"""Capital-split ensemble of ORB opening-range-window variants.

Instead of betting on one "best" opening-range length (a curve-fit waiting to
happen — the hardening showed 60-min killed SPY and 15-min killed QQQ), run several
windows at once, each on cash/N, and hold the blend. That is N independent
sub-accounts; the portfolio is the sum of their equity curves (see
:mod:`ict_bot.backtest.portfolio`). No parameter is *selected* from the data, so
there is nothing to overfit — the ensemble itself is the fixed strategy that
walk-forward then evaluates.
"""

from __future__ import annotations

import copy

import pandas as pd

from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.backtest.portfolio import combine_equity_curves, portfolio_metrics

# 15- / 30- / 60-minute opening ranges (start fixed at 09:30).
DEFAULT_OR_WINDOWS = ["09:45", "10:00", "10:30"]


def ensemble_backtest(
    symbol_df: pd.DataFrame,
    settings: dict,
    or_windows: list[str] | None = None,
    cash: float = 100_000.0,
    fill_mode: str = "next_open",
) -> dict:
    """Run the capital-split ensemble on one symbol; return equity, trades, metrics."""
    or_windows = or_windows or DEFAULT_OR_WINDOWS
    sub_cash = cash / len(or_windows)

    curves, trade_frames = [], []
    for end in or_windows:
        cfg = copy.deepcopy(settings)
        cfg["opening_range"]["end"] = end
        stats = run_orb_backtest(symbol_df, cfg, cash=sub_cash, fill_mode=fill_mode)
        curves.append(stats["_equity_curve"]["Equity"])
        variant_trades = stats["_trades"]
        if len(variant_trades):
            trade_frames.append(variant_trades)

    equity = combine_equity_curves(curves)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return {"equity": equity, "trades": trades, "metrics": portfolio_metrics(equity, trades)}
