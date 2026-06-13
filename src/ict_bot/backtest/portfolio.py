"""Portfolio aggregation for the capital-split ensemble.

The ensemble runs N ORB variants (different opening-range windows) each on cash/N,
which is exactly N sub-accounts. The portfolio equity is the *sum* of their equity
curves; portfolio risk/return metrics come from that summed curve, while trade-level
stats (PF, win rate, avg-R) come from the concatenated trade list. Pure functions —
no backtesting import — so they unit-test on synthetic curves.
"""

from __future__ import annotations

import pandas as pd

from ict_bot.backtest.metrics import _avg_r_multiple

_DAYS_PER_MONTH = 30.44


def combine_equity_curves(curves: list[pd.Series]) -> pd.Series:
    """Sum sub-account equity curves, aligning on the union index (forward-filled)."""
    if not curves:
        raise ValueError("combine_equity_curves needs at least one curve")
    # ffill carries each curve forward; bfill seeds leading gaps with that curve's
    # initial capital (an asset that starts a few minutes later is idle-at-cash, not
    # absent — otherwise the portfolio "starts" below full capital and the return is
    # inflated by the step when the later asset's curve appears).
    aligned = pd.concat(curves, axis=1).sort_index().ffill().bfill()
    return aligned.sum(axis=1)


def _profit_factor(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return float("nan")
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    if losses <= 0:
        return float("nan") if gains <= 0 else float("inf")
    return gains / losses


def portfolio_metrics(
    equity: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252
) -> dict:
    """Reporting metrics for a portfolio equity curve + its combined trade list.

    Same dict shape as :func:`ict_bot.backtest.metrics.extract_metrics`, so the OOS
    gate and report renderer work unchanged. Sharpe is computed on daily-resampled
    equity (ORB is flat by the close, so daily steps capture the strategy).
    """
    eq = equity.dropna()
    start, end = float(eq.iloc[0]), float(eq.iloc[-1])
    total_return = (end / start - 1.0) * 100 if start > 0 else 0.0
    drawdown = float(((eq / eq.cummax() - 1.0).min()) * 100)

    daily = eq.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = (
        float(rets.mean() / rets.std() * (periods_per_year ** 0.5))
        if len(rets) > 1 and rets.std() > 0 else 0.0
    )

    n = 0 if trades is None else len(trades)
    pnl = trades["PnL"] if (n and "PnL" in trades) else pd.Series(dtype=float)
    span_days = (eq.index[-1] - eq.index[0]).days or 1

    return {
        "total_return_pct": total_return,
        "win_rate_pct": float((pnl > 0).mean() * 100) if n else 0.0,
        "profit_factor": _profit_factor(pnl),
        "max_drawdown_pct": drawdown,
        "sharpe": sharpe,
        "avg_r_multiple": _avg_r_multiple(trades) if n else 0.0,
        "num_trades": int(n),
        "trades_per_month": n / (span_days / _DAYS_PER_MONTH),
    }
