"""Backtest harness (Phase 5).

Wraps the live signal + risk engine in a ``backtesting.py`` Strategy, fed
bar-by-bar. The SAME code paths as live (no separate logic). Enforces a
train/test split: tune on the earlier ~18 months, judge on an untouched ~6-month
out-of-sample window. Adds a 1-cent slippage assumption (Gotcha #7).
"""

from __future__ import annotations
