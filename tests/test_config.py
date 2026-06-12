"""Phase 0: config loading and logging helpers work."""

from __future__ import annotations

import datetime as dt


def test_settings_load():
    from ict_bot.config import load_settings

    settings = load_settings()
    assert settings["symbol"] == "SPY"
    assert settings["risk"]["risk_per_trade"] == 0.005
    # A sane threshold (tune-proof: the exact value moves across backtest cycles).
    assert 0 < settings["signal"]["min_confluence_score"] <= 100


def test_env_loader_runs(tmp_path, monkeypatch):
    from ict_bot.config import load_env, require_env

    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    load_env(env_file)
    assert require_env("ALPACA_API_KEY") == "test-key"


def test_format_gst_is_utc_plus_4():
    from ict_bot.ops.logging_conf import format_gst

    noon_utc = dt.datetime(2026, 6, 12, 12, 0, tzinfo=dt.timezone.utc)
    # GST is UTC+4 year-round (no DST) -> 16:00.
    assert format_gst(noon_utc, fmt="%H:%M") == "16:00"
