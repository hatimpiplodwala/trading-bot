# ICT Paper-Trading Bot

Automated **paper-trading** bot for **SPY on Alpaca** using an ICT methodology
stack with ATR14 confluence, a local Qwen3:4b LLM journal layer, and Telegram
alerts. See [prd.md](prd.md) for the full phased build plan.

> Paper trading only. No live capital until the forward test is sustained-profitable.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (installs and pins Python 3.11 itself).

```bash
uv sync                      # create .venv (Python 3.11) and install deps
cp .env.example .env         # then fill in your keys (see below)
uv run pytest                # run the test suite
```

### Credentials (`.env`)

| Variable | Where to get it |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET` | Alpaca **paper** account at <https://app.alpaca.markets/> |
| `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` | `@BotFather` (Phase 7) |

`.env` is gitignored — never commit real keys.

## Phase 1 — Data pipeline

Built: `data/feed.py` (Alpaca IEX historical + WebSocket stream), `data/store.py`
(DuckDB/Parquet, partitioned `symbol/timeframe/YYYY-MM`), `data/resample.py`
(no-lookahead multi-timeframe aggregation), `data/quality.py` (IEX validity).

```bash
# 1. Bootstrap ~2 years of SPY + QQQ/IWM history into data/parquet/
uv run python scripts/download_history.py

# 2. GATE (Gotcha #5): confirm IEX daily extremes track a consolidated reference.
#    Exits non-zero on REVIEW — resolve before trusting detectors in Phase 2.
uv run python scripts/validate_iex.py

# 3. Confirm the live WebSocket feed (run during US market hours, >=10 min)
uv run python scripts/stream_smoke.py
```

Bars are indexed by **open time in UTC** everywhere; the resampler emits a
higher-timeframe bar only once it is fully closed (the forming bar is dropped) —
this is the single source of truth shared by the live loop and the backtest, so
the no-lookahead guarantee is identical in both. The `4h` timeframe is derived by
resampling, never fetched.
