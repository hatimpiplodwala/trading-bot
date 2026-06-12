"""ORB backtest integration test (real SPY 15m data).

Confirms the strategy wiring honors its invariants: at most one trade per ET
session, every trade intraday (same-day entry/exit, no overnight), and entries
only after the opening range completes (>= 10:00 ET) and before the late-entry
cutoff. Skips when the local store lacks data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ict_bot.backtest.orb_runner import run_orb_backtest
from ict_bot.config import REPO_ROOT, load_settings
from ict_bot.data.store import BarStore


@pytest.fixture(scope="module")
def store():
    return BarStore(REPO_ROOT / "data" / "parquet")


def test_orb_trades_are_intraday_one_per_day(store):
    spy15 = store.read_bars("SPY", "15m")
    if len(spy15) < 1000:
        pytest.skip("insufficient local SPY 15m data")

    segment = spy15[spy15.index >= spy15.index[-1] - pd.Timedelta(days=40)]
    settings = load_settings()
    stats = run_orb_backtest(segment, settings, cash=100_000.0)

    assert "# Trades" in stats.index
    trades = stats._trades
    if len(trades) == 0:
        pytest.skip("no ORB trades in window")

    et = "America/New_York"
    entry_et = trades["EntryTime"].dt.tz_convert(et)
    exit_et = trades["ExitTime"].dt.tz_convert(et)

    # Intraday only: same ET session for entry and exit.
    assert (entry_et.dt.date == exit_et.dt.date).all()
    # At most one trade per ET day.
    assert entry_et.dt.date.value_counts().max() <= 1
    # Entries only after the opening range closes and before the late cutoff.
    range_end = settings["opening_range"]["end"]
    no_entry = settings["exits"]["no_entry_after"]
    em = entry_et.dt.hour * 60 + entry_et.dt.minute
    assert (em >= int(range_end[:2]) * 60 + int(range_end[3:])).all()
    assert (em < int(no_entry[:2]) * 60 + int(no_entry[3:])).all()


def test_orb_next_open_fills_stay_intraday(store):
    """The realistic next-bar-open fill model must not leak overnight.

    Filling entries (and the flatten) at the *next* bar's open instead of the
    signal bar's close is the pessimistic execution model. It must still honor
    the flat-by-close invariant: every trade enters and exits the same ET
    session, at most one per day.
    """
    spy15 = store.read_bars("SPY", "15m")
    if len(spy15) < 1000:
        pytest.skip("insufficient local SPY 15m data")

    segment = spy15[spy15.index >= spy15.index[-1] - pd.Timedelta(days=40)]
    settings = load_settings()
    stats = run_orb_backtest(segment, settings, cash=100_000.0, fill_mode="next_open")

    trades = stats._trades
    if len(trades) == 0:
        pytest.skip("no ORB trades in window")

    et = "America/New_York"
    entry_et = trades["EntryTime"].dt.tz_convert(et)
    exit_et = trades["ExitTime"].dt.tz_convert(et)

    # No overnight holds: same ET session for entry and exit.
    assert (entry_et.dt.date == exit_et.dt.date).all()
    # At most one trade per ET day.
    assert entry_et.dt.date.value_counts().max() <= 1
