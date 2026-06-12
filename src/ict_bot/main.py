"""Service entry point — the live event loop.

Wires data feed -> ICT detectors -> signal/risk engine -> broker, with Telegram
alerts and a post-trade LLM journal. Implemented in Phase 7; this is a scaffold.
"""

from __future__ import annotations


def main() -> None:
    """Start the bot. Placeholder until the Phase 7 live loop lands."""
    print("ict-bot: scaffold only (Phase 0). See prd.md for the build plan.")


if __name__ == "__main__":
    main()
