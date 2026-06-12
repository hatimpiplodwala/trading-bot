"""Live-stream smoke test (Phase 1 acceptance).

Subscribes to live IEX bars and prints each one to stdout. Run during US market
hours for >=10 minutes to confirm the WebSocket feed works:

    uv run python scripts/stream_smoke.py
    uv run python scripts/stream_smoke.py --symbols SPY QQQ IWM
"""

from __future__ import annotations

import argparse
from typing import Any

from ict_bot.config import load_env, load_settings, require_env
from ict_bot.data.feed import AlpacaFeed
from ict_bot.ops.logging_conf import configure_logging, format_gst


def main() -> None:
    load_env()
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Print live bars from Alpaca IEX.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[settings["symbol"], *settings.get("smt_reference_symbols", [])],
    )
    args = parser.parse_args()

    configure_logging()
    feed = AlpacaFeed(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET"))

    async def on_bar(row: dict[str, Any]) -> None:
        gst = format_gst(row["timestamp"])
        print(
            f"[{gst}] {row['symbol']} "
            f"O={row['open']} H={row['high']} L={row['low']} "
            f"C={row['close']} V={row['volume']}"
        )

    print(f"Streaming {args.symbols} (Ctrl+C to stop)...")
    feed.stream_bars(args.symbols, on_bar)


if __name__ == "__main__":
    main()
