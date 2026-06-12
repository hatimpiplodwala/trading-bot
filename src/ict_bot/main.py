"""Service entry point — the ORB live paper loop (Phase 7).

Wires data feed -> ORBSession decision core -> broker, with durable SQLite state
and log notifications. No LLM layer (Phase 6 dropped from v1 — it was off the
trade path). Run it during US market hours:

    uv run python -m ict_bot.main --dry-run --once   # smoke test: one poll, no orders
    uv run python -m ict_bot.main --dry-run          # full session, logs intended orders
    uv run python -m ict_bot.main                    # live on the Alpaca PAPER account
"""

from __future__ import annotations

import argparse
import signal

from ict_bot.broker.alpaca import AlpacaBroker
from ict_bot.broker.dryrun import DryRunBroker
from ict_bot.config import REPO_ROOT, load_env, load_settings, require_env
from ict_bot.data.feed import AlpacaFeed
from ict_bot.ops.alerts import LogNotifier
from ict_bot.ops.live_trader import LiveTrader
from ict_bot.ops.logging_conf import configure_logging
from ict_bot.ops.state import StateStore
from ict_bot.strategy.orb_session import ORBSession


def build_trader(settings: dict, *, dry_run: bool) -> LiveTrader:
    """Assemble the live trader from settings + environment (Alpaca paper)."""
    load_env()
    api_key, secret = require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET")
    broker_cfg = settings.get("broker", {})
    live_cfg = settings.get("live", {})

    feed = AlpacaFeed(api_key, secret)
    real_broker = AlpacaBroker(api_key, secret, paper=broker_cfg.get("paper", True))
    if dry_run:
        # Log intended orders; borrow the real account's equity + clock (read-only).
        broker = DryRunBroker(
            equity_fn=real_broker.get_equity, market_open_fn=real_broker.is_market_open
        )
    else:
        broker = real_broker

    db_path = REPO_ROOT / broker_cfg.get("state_db", "data/state.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    state = StateStore(str(db_path))

    return LiveTrader(
        settings["symbol"], feed, broker, state, LogNotifier(),
        ORBSession.from_settings(settings),
        timeframe=settings["entry_timeframe"],
        seed_days=live_cfg.get("seed_days", 4),
        poll_buffer_s=live_cfg.get("poll_buffer_seconds", 20),
        heartbeat_hours=settings.get("alerts", {}).get("heartbeat_hours", 1),
        max_frame_bars=settings.get("service", {}).get("max_frame_bars", 500),
    )


def _install_signal_handlers(trader: LiveTrader) -> None:
    """Route Ctrl-C / Ctrl-Break / SIGTERM to a graceful stop (NSSM stop, too)."""
    def _handler(_signum, _frame):
        trader.request_stop()

    for name in ("SIGINT", "SIGBREAK", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # not settable on this platform/thread
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="ORB live paper-trading loop.")
    parser.add_argument("--dry-run", action="store_true",
                        help="log intended orders; never submit to the broker")
    parser.add_argument("--once", action="store_true",
                        help="seed + a single poll, then exit (smoke test)")
    args = parser.parse_args()

    settings = load_settings()
    # Smoke runs stay console-only; the long-running service also writes a
    # daily-rotating log file.
    logfile = None if args.once else str(
        REPO_ROOT / settings.get("service", {}).get("log_file", "logs/ictbot.log")
    )
    configure_logging(logfile=logfile)

    trader = build_trader(settings, dry_run=args.dry_run)
    if args.once:
        trader.seed()
        trader.poll_once()
    else:
        _install_signal_handlers(trader)
        trader.run()


if __name__ == "__main__":
    main()
