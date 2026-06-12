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

## Phase 2 — Core ICT detection

Detectors in `src/ict_bot/ict/` wrap [`smartmoneyconcepts`](https://github.com/joshyattridge/smart-money-concepts)
and add the lifecycle/interpretation logic ICT needs:

- `structure.py` — swings + BOS/CHOCH → `MarketStructure` bias (only *broken* structure sets bias).
- `fvg.py` — fair value gaps with mitigation state.
- `order_blocks.py` — OBs with an `active → mitigated → breaker` state machine.
- `liquidity.py` — smc BSL/SSL/sweeps + equal-highs/lows clustering within `0.1 × ATR`.
- `zones.py` — dealing range: premium/discount/equilibrium and the OTE band (pure math).

All smc imports go through `ict/_smc.py`, which suppresses the library's import
banner (it crashes cp1252 Windows consoles). Detectors are pure functions of the
window passed in, so the no-lookahead guarantee is enforced at the call site;
a test confirms already-formed FVGs never repaint when more bars arrive.

```bash
# Visual sanity check — renders an interactive chart with FVGs, OBs, swings, structure
uv run python scripts/plot_setups.py        # writes data/charts/SPY_15m_setups.html
```

## Phase 3 — Sessions, bias & SMT

- `sessions.py` — DST-aware kill zones (Silver Bullet, NY AM/PM, pre-market) resolved
  against ET wall-clock; pre-market is observe-only (Gotcha #9).
- `bias.py` — HTF bias from Daily + 1H structure. Daily leads; the 1H only vetoes on a
  direct conflict, so a mixed read stands aside (`neutral`).
- `smt.py` — SMT divergence of SPY against QQQ/IWM: a higher high (or lower low) the
  reference fails to confirm. References are divergence input only — never traded (Gotcha #8).

## Phase 4 — Signal & risk engine

`src/ict_bot/strategy/` turns detector output into sized trade candidates on one code
path shared by the live loop and the backtest:

- `confluence.py` — scores a setup 0–100 from the confluences present (weights in
  `config/settings.yaml`); minimum to consider is **60**. The eight weights sum to 120,
  so the score is clamped to 100.
- `risk.py` — `position_size` (risk-budget ÷ per-share-risk, **floored to whole shares**
  and capped by a notional fraction of equity — the cap binds often on a ~$570 name);
  `compute_atr` (ATR14 via `pandas_ta_classic` with a manual Wilder fallback); `stop_loss`
  = entry ± `max(1.5×ATR, structural distance)`; `take_profit` at an R-multiple; and
  `DailyLossLimit`, the −2%/ET-day realized-P&L circuit breaker.
- `signal.py` — `SignalEngine.evaluate` scores a `SetupContext`, applies the daily gate,
  sizes the position, and emits a `CandidateSignal` (or suppresses it). `build_setup_context`
  wires the ICT detectors over bounded rolling windows (live-realistic, no lookahead).

Per-confluence detection is deliberately a modest v1 heuristic; its ICT fidelity is the
job of the Phase 5 out-of-sample backtest, not unit assertions. A month-long replay of
real SPY 15m bars confirms the engine is selective rather than firing on every bar.

## Phase 5 — Backtest harness

`src/ict_bot/backtest/` runs the **same** signal/risk code through
[`backtesting.py`](https://kernc.github.io/backtesting.py/) — no separate backtest logic:

- `runner.py` — `ICTStrategy` drives the engine bar-by-bar. SPY entry-TF bars are the
  primary series; Daily/1H and the QQQ/IWM references travel as config and are sliced to
  `<= ts` each bar (no lookahead, enforced by a test). One position at a time; the daily
  −2% loss limit is fed from realized trade P&L. Slippage of `slippage_cents`/share is
  modelled as a spread.
- `split.py` — `train_oos_split` reserves the most recent `oos_months` **untouched** as
  the out-of-sample verdict set (Gotcha #6); the rest is in-sample tuning.
- `metrics.py` — maps backtesting stats to our metric set (incl. avg R-multiple and
  trades/month) and renders an in-sample vs. OOS report. **OOS gate: PF > 1.2 and max
  drawdown < 15%** — in-sample alone proves nothing.

```bash
# Runs in-sample + OOS, writes backtest/results/YYYY-MM-DD.md (gitignored artifact)
uv run python scripts/run_backtest.py
```

Tuning rule: iterate weights/ATR multiples against in-sample only; the OOS run is the
verdict, not the tuning target. If OOS fails after ≤3 in-sample cycles, the strategy
needs structural rework — not more parameter tweaks.
