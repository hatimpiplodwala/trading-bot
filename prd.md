# ICT Paper-Trading Bot — Build Plan for Claude Code

## Project Goal

Build an automated paper-trading bot that trades **SPY on Alpaca's paper account** using a full **ICT (Inner Circle Trader) methodology stack** with **ATR14 confluence**, accompanied by a **local Qwen3:4b LLM journal layer via Ollama** (post-trade reflection; the trade-blocking "veto" use is deferred — see Phase 6 and Gotcha #3). Must run as a low-footprint background service.

## Locked-In Tech Decisions (Do Not Re-Litigate)

| Concern | Choice |
|---|---|
| Market / instrument | SPY (S&P 500 ETF) only — single instrument for v1 |
| Broker / data | Alpaca paper trading (`alpaca-py` SDK), free IEX real-time data |
| Language | Python 3.11+ |
| ICT detection | `smartmoneyconcepts` library (joshyattridge) + custom modules for what it doesn't cover |
| Indicators | `pandas-ta-classic` (NOT TA-Lib — avoid the C build) |
| Storage | DuckDB + Parquet for OHLC; SQLite for trade journal/state |
| LLM | Qwen3 4B via Ollama (`ollama pull qwen3:4b`) — role: **post-trade journal** for v1; trade-blocking veto deferred to v2 (see Gotcha #3) |
| Backtesting | `backtesting.py` |
| Alerting | Telegram Bot API |
| Service | systemd (Linux) / NSSM (Windows) — implement Linux first |
| Timezone | User is in Sharjah, UAE (GST, UTC+4) — all logs in UTC, display in GST |

## Hardware Context (Affects LLM Config)

- Dell XPS 9510 — i7-11800H, 32 GB RAM, RTX 3050 Laptop **4 GB VRAM**
- Qwen3:4b at default Q4 quant ≈ 2.6 GB weights → fits 4 GB VRAM with ~1.3 GB headroom for KV cache
- Must stay under **2.5 GB VRAM steady-state** to leave the user GPU room for other work

## ICT Concepts to Implement (Full Stack)

The bot must detect and use ALL of these:

1. **Market structure** — swing highs/lows, BOS, CHoCH, MSS
2. **Order blocks** — bullish/bearish OB, mitigation blocks, breaker blocks
3. **Fair Value Gaps (FVG)** — 3-candle imbalances, mitigation tracking
4. **Liquidity** — buy-side/sell-side liquidity, equal highs/lows, sweeps/raids
5. **Premium/Discount zones** — dealing range, equilibrium, OTE (62–79% fib)
6. **Kill zones (NY-centric for SPY)**:
   - NY pre-market: 07:00–09:30 ET — ⚠️ **observe only, do not trade in v1** (bracket orders + IEX data both behave poorly pre-open; see Gotcha #9)
   - NY AM session: 09:30–12:00 ET
   - **Silver Bullet: 10:00–11:00 ET** (highest-priority window)
   - NY PM session: 13:30–16:00 ET
7. **HTF bias** — Daily and 1H bias gates 15m/5m entries
8. **SMT divergence** — SPY vs QQQ vs IWM correlation breaks
9. **Power of 3** — Accumulation / Manipulation / Distribution
10. **Judas swing** — false move at session open that sweeps liquidity
11. **NDOG / NWOG** — New Day / New Week Opening Gaps. ⚠️ The SPY *cash ETF* only gaps at the **09:30 ET open** (it doesn't trade overnight), so "opening gap" = prior close → today's 09:30 open. Exclude **dividend ex-date gaps** — those are corporate-action gaps, not liquidity gaps, and NDOG will misread them.
12. **Midnight Open & Opening Range** — ⚠️ **Midnight Open (00:00 ET) is inapplicable to the SPY cash ETF** — the market is closed at midnight, there is no print. This is an ES-futures concept. **Cut from v1** (or proxy via ES futures, which we are *not* subscribed to). **Opening Range (09:30–10:00 ET high/low) is valid — keep that part.**
13. **ATR14 confluence** — for SL distance, position sizing, volatility filter

> **Concepts-vs-scoring gap (read this):** Several concepts above (Power of 3, Judas swing, NDOG/NWOG, midnight open) get their own modules but are **not** referenced by the Phase 4 confluence scorer as written. A built-but-unwired detector adds maintenance and lookahead surface for zero signal value. For each: either **wire it into the Phase 4 scorer** or **defer it to v2** and don't build the module in v1. See the v1 scope note at the end of the phased plan.

## Project Structure to Create

```
ict_bot/
├── pyproject.toml              # use uv or poetry; pin Python 3.11
├── README.md
├── .env.example                # ALPACA_API_KEY, ALPACA_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
├── .gitignore                  # must include .env, data/, logs/, *.db
├── config/
│   └── settings.yaml           # symbol, timeframes, risk %, kill-zone times
├── src/ict_bot/
│   ├── __init__.py
│   ├── main.py                 # event loop entry point
│   ├── data/
│   │   ├── feed.py             # Alpaca StockDataStream + historical bars
│   │   ├── store.py            # DuckDB/Parquet read/write
│   │   └── resample.py         # multi-timeframe aggregation
│   ├── ict/
│   │   ├── structure.py        # swings, BOS, CHoCH, MSS (wraps smc)
│   │   ├── order_blocks.py     # OB + mitigation + breaker state
│   │   ├── fvg.py              # FVG detection + mitigation
│   │   ├── liquidity.py        # BSL/SSL, equal H/L, sweeps
│   │   ├── zones.py            # premium/discount, OTE
│   │   ├── sessions.py         # NY kill zones, Silver Bullet
│   │   ├── opening_gaps.py     # NDOG / NWOG (09:30 open only; v2 unless wired into scoring)
│   │   ├── opening_range.py    # 09:30–10:00 ET range (midnight-open cut: N/A to SPY ETF)
│   │   ├── smt.py              # SPY/QQQ/IWM divergence
│   │   └── bias.py             # HTF bias engine
│   ├── strategy/
│   │   ├── signal.py           # combine confluences → candidate signal
│   │   ├── risk.py             # ATR position sizing, daily loss limit
│   │   └── confluence.py       # scoring system for setup quality
│   ├── llm/
│   │   ├── client.py           # Ollama client wrapper
│   │   ├── prompts.py          # journal prompt template (veto template = v2)
│   │   ├── journal.py          # post-trade LLM-written journal (v1)
│   │   └── veto.py             # DEFERRED to v2 — only if fed a news/sentiment feed
│   ├── broker/
│   │   └── alpaca.py           # order placement, account state (paper)
│   ├── ops/
│   │   ├── alerts.py           # Telegram client
│   │   ├── logging_conf.py     # structured JSON logging
│   │   └── state.py            # SQLite state store (open positions, daily P&L)
│   └── backtest/
│       └── runner.py           # backtesting.py harness
├── tests/
│   ├── test_fvg.py
│   ├── test_order_blocks.py
│   ├── test_structure.py
│   ├── test_resample.py        # CRITICAL: no lookahead bias
│   └── fixtures/               # known-good OHLC fixtures with hand-labeled setups
├── scripts/
│   ├── download_history.py     # bootstrap historical SPY data
│   ├── benchmark_llm.py        # measure tok/s on this machine
│   └── plot_setups.py          # visualize detections for sanity-check
└── service/
    ├── ictbot.service          # systemd unit
    └── nssm_install.bat        # Windows fallback
```

## Phased Implementation Plan

Work through phases sequentially. **Do not start a phase until the previous phase's acceptance criteria pass.** After each phase, run tests and report status before moving on.

---

### Phase 0: Bootstrap (Day 1)

**Tasks:**
1. Initialize repo with `uv init` (preferred) or `poetry init`. Python 3.11.
2. Install core deps: `alpaca-py`, `pandas`, `pandas-ta-classic`, `smartmoneyconcepts`, `duckdb`, `pyarrow`, `ollama`, `pyyaml`, `python-dotenv`, `pytest`, `httpx`, `backtesting`.
3. Create the directory structure above (empty modules with docstrings).
4. Write `.env.example` and `.gitignore`.
5. Set up `config/settings.yaml` with: symbol=`SPY`, timeframes=`[5m, 15m, 1h, 4h, 1d]`, risk_per_trade=`0.005` (0.5%), daily_loss_limit=`0.02` (2%), kill_zones (in ET, with DST handling).
6. Configure structured logging (`structlog` or stdlib `logging` with JSON formatter). All timestamps in UTC; display helpers convert to GST.

**Acceptance:** `python -c "import ict_bot"` works; `pytest` runs (zero tests yet, but framework live); `.env` loading works.

---

### Phase 1: Data Pipeline (Days 2–3)

**Tasks:**
1. `data/feed.py`: Implement `AlpacaFeed` class with:
   - `get_historical_bars(symbol, timeframe, start, end)` using `StockHistoricalDataClient`
   - `stream_bars(symbol, on_bar_callback)` using `StockDataStream` (WebSocket)
   - Use IEX feed (free tier) — set `feed=DataFeed.IEX`
2. `data/store.py`: DuckDB wrapper. Store OHLC in Parquet partitioned by `symbol/timeframe/year-month`. Provide `read_bars(symbol, timeframe, start, end) -> pd.DataFrame` and `append_bars(df)`.
3. `data/resample.py`: Resample base 1m or 5m bars to higher TFs using pandas. **Critical:** right-edge close, never include forming bars. Add a test that proves no lookahead.
4. `scripts/download_history.py`: bootstrap script — pull 2 years of SPY data at 5m, 15m, 1h, 1d.
5. **IEX data-validity check (gating — see Gotcha #5):** `scripts/validate_iex.py` — diff a few days of IEX bar highs/lows against a SIP/consolidated reference (TradingView, Yahoo, or a borrowed SIP sample). ICT is built on precise swing extremes, equal highs/lows, and liquidity sweeps — exactly the wick tips a single-exchange feed distorts. Quantify the divergence at swing points before trusting any detector built on this data.

**Acceptance:**
- Run `python scripts/download_history.py` → fills `data/parquet/SPY/...`
- `test_resample.py` passes — proves a 15m bar at time T only contains 5m bars that closed by T
- Live stream prints incoming bars to stdout for ≥10 minutes during market hours
- **IEX swing extremes materially match the SIP reference** (sweeps/equal-H/L line up), **OR** a documented decision is recorded to move to paid SIP (Alpaca Algo Trader Plus, ~$99/mo) or an alternate data source. Do not start Phase 2 until this is resolved — the whole strategy's validity rests on it.

---

### Phase 2: Core ICT Detection (Days 4–7)

**Tasks:**
1. `ict/structure.py`: Wrap `smc.swing_highs_lows()` and `smc.bos_choch()`. Add `MarketStructure` dataclass with current bias (`bullish`/`bearish`/`neutral`) and last structure level.
2. `ict/fvg.py`: Wrap `smc.fvg(join_consecutive=True)`. Track mitigation state across bars. Return list of `FVG` dataclasses with `top, bottom, type, formed_at, mitigated, mitigated_at`.
3. `ict/order_blocks.py`: Wrap `smc.ob()`. Implement mitigation tracking — when price closes through an OB, mark as `breaker_block`. State machine: `active` → `mitigated` → `breaker`.
4. `ict/liquidity.py`: Wrap `smc.liquidity()`. Add equal-highs/lows detection (cluster swing points within 0.1×ATR tolerance).
5. `ict/zones.py`: Compute dealing range from last major HH↔LL. Return premium (>50%), equilibrium (50%), discount (<50%), OTE (62–79% retracement).
6. **Tests:** For each detector, create hand-labeled OHLC fixtures in `tests/fixtures/` and verify detections match.

**Acceptance:**
- `scripts/plot_setups.py` renders SPY 15m chart with FVGs, OBs, swings, structure breaks drawn — sanity-check visually against TradingView
- All detector unit tests pass
- No detector ever uses data from after the bar being evaluated

---

### Phase 3: Sessions, Bias, Gaps (Days 8–9)

**Tasks:**
1. `ict/sessions.py`: Implement NY kill zones in ET with DST awareness. Functions: `is_silver_bullet(ts)`, `is_ny_am(ts)`, `is_ny_pm(ts)`, `current_kill_zone(ts)`. Convert from UTC.
2. `ict/bias.py`: `HTFBias` class — combine Daily + 1H structure to produce one of `bullish/bearish/neutral`. Long signals only allowed in bullish bias; shorts only in bearish.
3. `ict/opening_gaps.py`: NDOG (prev close → today's 09:30 open), NWOG (Fri close → Mon 09:30 open). Track whether filled. **Exclude dividend ex-date gaps.** *(Only build if it will be wired into Phase 4 scoring — otherwise defer to v2.)*
4. `ict/opening_range.py`: Opening Range = high/low of 09:30–10:00 ET. **Do NOT implement a 00:00 ET "midnight open" for the SPY ETF — no print exists when the market is closed** (ES-futures concept; cut from v1).
5. `ict/smt.py`: Pull QQQ + IWM bars. Detect divergence: SPY makes HH but QQQ doesn't (bearish SMT), or vice versa.

**Acceptance:**
- Session functions return correct kill zone for a series of test timestamps across DST boundary
- HTF bias correctly flips on a hand-crafted CHoCH fixture
- SMT divergence flags a known historical example

---

### Phase 4: Signal & Risk Engine (Days 10–12)

**Tasks:**
1. `strategy/confluence.py`: Score a setup 0–100 based on present confluences:
   - HTF bias aligned: +25
   - In Silver Bullet window: +20
   - In any NY kill zone: +10
   - Entry in discount (long) / premium (short): +15
   - OTE retracement: +10
   - FVG + OB overlap at entry: +15
   - SMT divergence supporting direction: +10
   - Liquidity sweep just before entry: +15
   - Min threshold to consider: **60**
2. `strategy/risk.py`:
   - `position_size(equity, entry, stop, risk_pct)` — risk per trade
   - ATR14 from `pandas_ta.atr(length=14)`
   - Stop = entry ± `max(1.5 × ATR14, structural_invalidation_level)`
   - Daily loss limit: track realized P&L per ET day; halt new entries at -2%
   - Max concurrent positions: 1 (single instrument)
3. `strategy/signal.py`: On each closed bar, run all detectors → check confluences → if score ≥ 60 and risk checks pass, emit `CandidateSignal` dataclass.

**Acceptance:**
- Replay 1 month of historical SPY 15m data through the signal engine → produces a reasonable number of signals (target: 5–20/month, not 500)
- Daily loss limit halts trading correctly in a synthetic losing-streak test

---

### Phase 5: Backtest Harness (Days 13–15)

**Tasks:**
1. `backtest/runner.py`: Wrap the signal + risk engine in a `backtesting.py` Strategy. Feed it historical data bar-by-bar. **The exact same code paths** as live — no separate logic.
2. Add explicit lookahead-bias test: confirm `Strategy.next()` sees only data up to `self.data.index[-1]`.
3. **Train/test split (mandatory — see Gotcha #6).** Split the 2 years of data: tune **only** on the earlier ~18 months (in-sample); reserve the most recent ~6 months **untouched** as out-of-sample (OOS). Never look at OOS while tuning.
4. Run the backtest and report on **both** in-sample and OOS: total return, win rate, profit factor, max drawdown, Sharpe, avg R-multiple, trades/month.
5. Iterate on confluence weights and ATR multiples against the **in-sample** set only — but **do not overfit**; keep changes minimal and motivated. The OOS run is the verdict, not the tuning target.

**Acceptance:**
- Backtest runs end-to-end on 2 years of SPY data with a clean in-sample / OOS split
- Results report generated to `backtest/results/YYYY-MM-DD.md` showing in-sample **and** OOS metrics side by side
- **Profit factor > 1.2 and max drawdown < 15% on the out-of-sample window** (in-sample alone proves nothing). If OOS fails after ≤3 in-sample tuning cycles, the strategy needs structural rework — not more parameter tweaks.

---

### Phase 6: LLM Journal Layer (Days 16–18)

**Design decision (see Gotcha #3):** For v1 the LLM is **journal-only**. A trade-blocking veto is intentionally *deferred*, because Qwen3:4b handed a JSON blob of confluences the deterministic engine already computed — with `recent_news: null` and no market data of its own — has no information edge to veto on. It would either rubber-stamp (the design already defaults to approve) or remove good trades on a hallucinated reason. Post-trade reflection, by contrast, is a natural fit for a small local model. **Only build the veto if/when it is fed genuinely new information** (a news/sentiment feed) the engine lacks — that is a v2 item, not v1.

**Tasks:**
1. Install Ollama; pull model: `ollama pull qwen3:4b`
2. Configure Ollama for background efficiency:
   ```bash
   # Linux: /etc/systemd/system/ollama.service.d/override.conf
   [Service]
   Environment="OLLAMA_KEEP_ALIVE=-1"
   Environment="OLLAMA_NUM_PARALLEL=1"
   Environment="OLLAMA_MAX_LOADED_MODELS=1"
   ```
3. `llm/client.py`: thin wrapper around `ollama.Client()`. Health check method. Timeout = 30s.
4. `llm/prompts.py`: Build the **journal** prompt template. Input is the closed-trade context (setup, confluences present, entry/stop/target, actual outcome, R-multiple, kill zone). System prompt asks for a short reflective note on what worked / what didn't.
5. `llm/journal.py`: After a trade closes, call the LLM with the trade outcome → write a reflective journal entry to SQLite. If the model is unreachable or returns garbage, log and skip — the journal is never on the trade path and must never block or delay trading.
6. `scripts/benchmark_llm.py`: measure tok/s, VRAM, time-to-first-token. Confirm <2.5 GB VRAM steady-state and acceptable journal latency.
7. *(Deferred to v2)* `llm/veto.py` + veto prompt: only if a news/sentiment input is added. If built, it must parse+validate JSON and **default to APPROVE on any malformed response, timeout, or crash** — never block a valid deterministic signal.

**Acceptance:**
- `ollama ps` shows `qwen3:4b` with `UNTIL: Forever`
- Benchmark report: <2.5 GB VRAM, ≥15 tok/s
- 10 sample closed trades produce journal entries written to SQLite
- Killing Ollama mid-run does not affect trading (journal entries simply skipped and logged)

---

### Phase 7: Broker Integration & Live Paper Loop (Days 19–21)

**Tasks:**
1. `broker/alpaca.py`: `TradingClient` (paper). Methods: `submit_bracket_order(symbol, qty, entry, stop, target)`, `get_positions()`, `get_account()`, `cancel_all()`.
   - ⚠️ **Integer shares only.** Alpaca does **not** allow bracket/OCO orders on fractional shares — `submit_bracket_order` with a fractional qty will be rejected. Round position size down to whole shares (see `risk.position_size`).
   - ⚠️ **Regular-hours only for v1.** Bracket orders behave poorly in extended hours; do not submit entries outside 09:30–16:00 ET (see Gotcha on pre-market).
2. `ops/state.py`: SQLite tables — `positions`, `daily_pnl`, `signals`, `journal`. *(Add `veto_log` only if/when the v2 veto is built.)*
3. `ops/alerts.py`: Telegram client. Send on: signal generated, order filled, order rejected, daily summary, error. *(Journal entries are post-trade and need not be alerted — keep Telegram noise down.)*
4. **Integer-share rounding in `strategy/risk.position_size`** (also affects Phase 4): round computed qty **down** to a whole share. On a ~$570 instrument with a 0.5% risk budget this can distort the realized risk %, so add a sanity cap (e.g. reject if rounded qty is 0, or if notional exceeds a configured fraction of equity).
5. `main.py`: Wire it all together (no trade-blocking veto in v1):
   ```
   on bar close (5m):
     update detectors with new bar
     if signal_engine.has_signal():
       signal = signal_engine.get()
       if risk.allowed(signal) and sessions.is_regular_hours(now):
         qty = risk.position_size(...)        # whole shares; skip if 0
         order = broker.submit_bracket_order(symbol, qty, entry, stop, target)
         alerts.notify_signal(signal, order)
         state.record(...)
   on trade close:
     journal.write_entry(trade)               # LLM reflection; never blocks
     alerts.notify_close(trade)
   ```
5. Heartbeat: every hour during market hours, log + Telegram "still alive" with open positions and daily P&L.
6. Error handling: WebSocket reconnect with exponential backoff; if Ollama unreachable, log + skip the journal entry (trading is unaffected — LLM is off the trade path); if Alpaca rejects, log + alert + halt.

**Acceptance:**
- Bot runs for one full NY session without crashing
- All signals/decisions/orders logged
- Telegram receives heartbeats and trade alerts
- Daily summary message at 16:05 ET

---

### Phase 8: Background Service & Hardening (Days 22–23)

**Tasks:**
1. `service/ictbot.service`: systemd unit with `Restart=always`, `Nice=10`, `CPUWeight=50`, env file pointing to `.env`. Document Windows NSSM equivalent.
2. Log rotation: rotate logs daily, keep 30 days.
3. Graceful shutdown: SIGTERM → cancel open orders? No — leave them; just stop accepting new signals and close streams cleanly.
4. Add a `--dry-run` flag that runs everything except `broker.submit_*` (logs what it would have done).
5. Stress test: run for 1 full week on paper. Monitor RAM/VRAM drift, log volume, Telegram noise level.

**Acceptance:**
- `systemctl status ictbot` shows active and stable for 1 week
- No memory leaks (RSS stable)
- No more than ~20 Telegram messages/day average

---

### Phase 9: Forward Test (Weeks 4–6)

Run on paper for **at least 3 weeks**. Do not modify strategy mid-run. After 3 weeks, compare live results to backtest expectations. If live underperforms backtest by >30%, debug — common culprits: slippage assumptions, IEX vs SIP data gaps, kill-zone timezone bugs.

**Do not consider live capital until forward test is sustained-profitable over multiple weeks.**

---

## v1 Scope & Realistic Timeline

**The day-level estimates above are optimistic.** A 13-concept ICT stack with detectors, tests, backtest harness, and a live loop is realistically a **multi-month** effort, not ~3 weeks. Treat the "Day N" labels as ordering, not a schedule.

**Build a lean, validated v1 first.** Prove the core engine end-to-end with the highest-signal concepts before adding the long tail:

- **v1 core:** market structure, FVG, order blocks, kill zones (regular hours), HTF bias, ATR risk, and **SMT divergence** (SPY vs QQQ/IWM). Wire *only these* into the Phase 4 confluence scorer. ⚠️ SMT inherits the IEX data-quality caveat threefold — validate QQQ/IWM extremes alongside SPY in the Phase 1 check, and confirm the three feeds' bars are time-aligned before trusting a divergence.
- **Defer to v2** (build only after v1 clears out-of-sample backtest + forward test): Power of 3, Judas swing, NDOG/NWOG, opening range, and the LLM veto. Don't build a detector module that isn't wired into scoring.

A smaller engine you can fully reason about and validate beats a 13-concept stack whose interactions you can't audit.

**Honest expectation.** A fully-mechanical ICT system clearing **PF > 1.2 out-of-sample on SPY** is an *unproven hypothesis*. The most likely Phase 5 outcome is "it doesn't beat buy-and-hold" — and that's fine: this plan's discipline (lookahead tests, OOS split, forward test before capital) exists precisely to discover that **cheaply and honestly**, before risking money. Success is a validated *answer*, not a guaranteed profitable bot.

---

## Critical Gotchas (Read Before Coding)

1. **No lookahead bias.** Every detector must only see data up to and including the current closed bar. Forming bars are forbidden. Resampling is the #1 source of leakage — test it explicitly.
2. **Order block subjectivity.** OBs are the most discretionary ICT concept. Pin the exact rule (the `smartmoneyconcepts` definition) and don't second-guess against guru charts.
3. **LLM is journal-only in v1 — and never on the trade path.** A 4B model fed only the engine's own confluence summary (no market data, no news) has no edge to veto on; it would rubber-stamp or hallucinate. Keep it to post-trade reflection. A trade-blocking veto is a v2 item and only justified if given genuinely new information (news/sentiment). If a veto is ever added, it must default-to-APPROVE on any parse error, timeout, or crash — never let the LLM block a valid deterministic signal.
4. **DST kills kill zones.** Compute session windows from UTC and current ET DST offset, not from hardcoded GST times.
5. **IEX vs SIP — validate before trusting the stack (not just a footnote).** Free Alpaca data is IEX-only (~2–3% of consolidated volume). ICT signals are built on precise sweeps, equal highs/lows, and swing extremes — the exact wick tips and prints a single-exchange tape gets wrong. A sweep on SIP may not show on IEX and vice-versa, so you can build everything perfectly and still detect setups in noise. Run the Phase 1 validity check; if IEX swing extremes diverge materially from a SIP reference, move to paid SIP or an alternate source rather than proceeding on faith.
6. **Don't overfit — validate out-of-sample.** Tuning ~8 confluence weights + a threshold on a single in-sample year *is* overfitting; that PF will clear 1.2 almost by construction and means nothing. The hold-out split (Phase 5) is the guard: tune on in-sample only, judge on the untouched OOS window. Cap in-sample tuning at 3 cycles — if OOS still fails, the strategy needs structural rework, not parameter tweaks.
7. **Alpaca paper fills are optimistic.** Add a 1-cent slippage assumption in the backtest harness.
8. **Single-symbol v1.** SPY only. Do not expand to QQQ/IWM as tradeable instruments in v1 — they're only used for SMT divergence input.
9. **Integer shares + regular hours for orders.** Alpaca rejects bracket/OCO orders on fractional shares — round position size down to whole shares and sanity-cap. And bracket orders behave poorly outside 09:30–16:00 ET, so the pre-market kill zone is **observe-only** in v1; submit entries during regular hours only.

## Reference Material

- Alpaca Python SDK: https://docs.alpaca.markets/docs/getting-started
- `smartmoneyconcepts`: https://github.com/joshyattridge/smart-money-concepts
- `backtesting.py`: https://kernc.github.io/backtesting.py/
- Ollama Python client: https://github.com/ollama/ollama-python

## Definition of Done (v1)

- [ ] All 9 phases complete with acceptance criteria met
- [ ] 3+ weeks of stable forward-test on Alpaca paper
- [ ] Tests passing; coverage on ICT detectors ≥ 70%
- [ ] Bot runs as a service and survives a reboot
- [ ] Telegram alerts working
- [ ] LLM journal layer measured at <2.5 GB VRAM steady-state; writes entries on trade close without ever blocking trading
- [ ] README documents setup, running, and stopping the bot

## Status Reporting

After each phase, report to the user:
- ✅ Acceptance criteria passed / ❌ blocked
- Files changed
- Any deviations from this plan (with reasoning)
- Next phase to start
