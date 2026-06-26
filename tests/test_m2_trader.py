import datetime as dt
import numpy as np
import pandas as pd
import pytest

from ict_bot.broker.dryrun import DryRunBroker
from ict_bot.broker.base import Position
from ict_bot.ops.alerts import LogNotifier
from ict_bot.ops.state import StateStore
from ict_bot.ops.m2_trader import M2Trader
from ict_bot.strategy.rsi2_session import RSI2Session

_ET = "America/New_York"
_UTC = dt.timezone.utc


def _daily(vals):
    idx = pd.date_range("2024-01-02 21:00", periods=len(vals), freq="D", tz="UTC")
    return pd.DataFrame({"open": vals, "high": vals, "low": vals,
                         "close": vals, "volume": 1}, index=idx)


class FakeFeed:
    def __init__(self, frames):
        self._frames = frames

    def get_historical_bars(self, symbol, timeframe, start, end):
        return self._frames[symbol]


def _oversold_uptrend():
    return list(np.linspace(100, 300, 250)) + [291, 285, 279]


def _trader(frames, broker=None):
    session = RSI2Session(entry_t=10, exit_t=60, trend_len=200, exit_sma_len=5)
    return M2Trader(
        list(frames), FakeFeed(frames), broker or DryRunBroker(equity=9000.0),
        StateStore(":memory:"), LogNotifier(), session,
        decision_after="15:45", stop_pct=0.10,
    )


def test_target_shares_equal_weight():
    t = _trader({"SPY": _daily(_oversold_uptrend())})
    # equity 9000 / 1 symbol / price 90 -> 100 shares
    assert t._target_shares(9000.0, 90.0) == 100


def test_enters_long_with_catastrophe_stop_when_oversold():
    broker = DryRunBroker(equity=9000.0)
    t = _trader({"SPY": _daily(_oversold_uptrend())}, broker)
    t.seed()
    now = pd.Timestamp("2024-09-09 19:50", tz="UTC")  # 15:50 ET
    t.decide_and_act(now=now)
    entry = [o for o in broker.orders if o["kind"] == "entry"]
    assert len(entry) == 1
    assert entry[0]["symbol"] == "SPY" and entry[0]["direction"] == "long"
    price = 279.0
    assert entry[0]["stop"] == pytest.approx(price * 0.90, rel=1e-3)
    assert entry[0]["target"] is None  # OTO entry+stop, no take-profit


def test_no_action_before_decision_window():
    broker = DryRunBroker(equity=9000.0)
    t = _trader({"SPY": _daily(_oversold_uptrend())}, broker)
    t.seed()
    t.decide_and_act(now=pd.Timestamp("2024-09-09 17:00", tz="UTC"))  # 13:00 ET
    assert [o for o in broker.orders if o["kind"] == "entry"] == []


def test_exits_when_recovered_and_holding():
    broker = DryRunBroker(equity=9000.0)
    broker._positions["SPY"] = Position("SPY", 100, "long")  # already holding
    recovered = list(np.linspace(100, 120, 30))
    t = _trader({"SPY": _daily(recovered)}, broker)
    t.seed()
    t.decide_and_act(now=pd.Timestamp("2024-09-09 19:50", tz="UTC"))
    assert any(o["kind"] == "close" and o["symbol"] == "SPY" for o in broker.orders)


def test_reconcile_does_not_flatten_overnight_hold():
    broker = DryRunBroker(equity=9000.0)
    broker._positions["SPY"] = Position("SPY", 100, "long")
    t = _trader({"SPY": _daily(_oversold_uptrend())}, broker)
    t.seed()  # seed calls reconcile
    assert broker.get_position("SPY") is not None  # NOT force-flattened (unlike ORB)


def test_flatten_all_closes_every_leg():
    broker = DryRunBroker(equity=9000.0)
    broker._positions["SPY"] = Position("SPY", 50, "long")
    broker._positions["QQQ"] = Position("QQQ", 20, "long")
    frames = {"SPY": _daily(_oversold_uptrend()), "QQQ": _daily(_oversold_uptrend())}
    t = _trader(frames, broker)
    t.flatten_all()
    assert broker.get_position("SPY") is None
    assert broker.get_position("QQQ") is None


def test_build_trader_selects_by_strategy_key(monkeypatch):
    from ict_bot.main import build_trader
    from ict_bot.config import load_settings
    from ict_bot.ops.live_trader import LiveTrader

    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")
    settings = load_settings()
    settings["strategy"] = "m2"
    assert isinstance(build_trader(settings, dry_run=True), M2Trader)
    settings["strategy"] = "orb"
    assert isinstance(build_trader(settings, dry_run=True), LiveTrader)
