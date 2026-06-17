# Opening Range Breakout — an honest intraday trading bot

An automated **paper-trading** bot that trades a single, well-understood edge —
the **Opening Range Breakout** — on SPY through Alpaca's paper account. It runs as
a background service, decides on closed 15-minute bars, and is always flat by the
close.

What makes this project worth reading isn't the bot — it's the **discipline**. Every
strategy here had to clear an out-of-sample gate before it was trusted, and the
ones that didn't were thrown out, even after a lot of work went into them.

> **Paper trading only.** No real capital. The bot trades Alpaca's paper account;
> nothing here is financial advice.

---

## The story

This started as an **ICT (Inner Circle Trader)** bot — a full mechanical stack of
market-structure, fair-value-gap, order-block and liquidity-sweep "confluences"
scored 0–100. It was built end to end (detectors, a signal engine, a backtest
harness) and then put on trial against a strict rule: **tune only on the first 18
months, and let the untouched last 6 months be the verdict.**

It failed. Repeatedly.

- Parameter tuning: in-sample profit factor 1.13 → out-of-sample **0.83**.
- A structural rework (real stops and liquidity targets): in-sample 1.70 → OOS **0.18**.
- Per-confluence attribution showed every signal *flipped sign* out-of-sample — the
  "high-conviction" setups were the ones that lost money.

So ICT was abandoned. That negative result — proven cheaply, before any money was
at risk — is the most valuable thing the project produced.

The pivot was to **Opening Range Breakout**, a published intraday momentum effect:
let the market define its range in the first 30 minutes, then trade the first
decisive break of it. With **zero tuning**, ORB cleared the same out-of-sample gate
that ICT never could. It was then hardened against the assumptions most likely to
flatter it, and only then wired up to trade live (paper).

---

## How the bot trades

Each ET session, on every closed 15-minute bar:

1. **09:30–10:00 — build the range.** Track the high and low of the opening bars.
2. **From 10:00 — watch for a break.** The first bar to *close* above the range goes
   long; below it goes short. (A close, not a wick — fake-outs that get rejected
   intrabar don't count.)
3. **Size and stop.** Risk a fixed fraction of equity; the stop sits at the opposite
   side of the range (floored at ½·ATR so a tight range can't make a hair-trigger
   stop). Whole shares, no leverage.
4. **One trade per day. No new entries after 15:30. Flat by 15:45** — never held
   overnight.

The single most important design choice: **the live bot and the backtest run the
exact same decision code** ([`orb_session.py`](src/ict_bot/strategy/orb_session.py)
is shared by both). The backtest isn't a separate model of the strategy — it *is*
the strategy, replayed on history. That's what makes the validation mean something.

---

## What the validation actually says

On SPY, ~4.5 years (2022–2026), realistic next-bar-open fills:

- **Clears the gate** (profit factor > 1.2, max drawdown < 15%) **out-of-sample** —
  last 6 months: PF 1.37, Sharpe 1.79, −2.2% drawdown; full 4.5 years: PF 1.21,
  +53%, −5.7% drawdown.
- **Survives the 2022 bear** — because it trades both directions, the down-trend fed
  it rather than killing it. This retired the biggest fear ("only works in a bull
  market").
- **Survives realistic fills** — the edge barely moves when entries fill at the next
  bar's open instead of the signal bar's close.

And, just as importantly, what the validation says *doesn't* work:

- **A parameter ensemble** (blending 15/30/60-minute ranges) *hurt* — averaging in
  weaker windows diluted the good one.
- **Cross-asset diversification** (QQQ, IWM, TLT bonds, GLD gold) found **no new edge**
  to diversify into — only SPY independently passes. Adding low-correlation assets
  cuts drawdown but erodes the edge, so it was rejected.

The honest bottom line: **the edge is real but thin and regime-dependent.** Its value
is risk-adjusted (low drawdown, no overnight exposure), not spectacular returns.
This is a forward-test candidate, not a finished money machine.

---

## Architecture

| Layer | Modules | What it does |
| --- | --- | --- |
| **Strategy** | [`strategy/orb.py`](src/ict_bot/strategy/orb.py), [`orb_session.py`](src/ict_bot/strategy/orb_session.py), [`risk.py`](src/ict_bot/strategy/risk.py) | Pure breakout/stop/target math; the stateful per-session decision core; fixed-fractional sizing + daily-loss limit. |
| **Backtest** | [`backtest/`](src/ict_bot/backtest/) | Drives the *same* decision core through [backtesting.py](https://kernc.github.io/backtesting.py/); train/OOS split, ensemble + walk-forward + cross-asset screening tools. |
| **Live loop** | [`ops/live_trader.py`](src/ict_bot/ops/live_trader.py), [`broker/`](src/ict_bot/broker/), [`ops/state.py`](src/ict_bot/ops/state.py) | Seeds history, wakes on each bar close, executes via Alpaca (market entry + **resting broker-side stop**), durable one-trade-per-day, graceful shutdown. |
| **Service** | [`main.py`](src/ict_bot/main.py), [`service/`](service/) | CLI entry point and a Windows NSSM service wrapper (auto-restart, daily-rotating logs). |

Data is Alpaca's free IEX feed, stored as partitioned Parquet, indexed by bar
**open time in UTC** everywhere.

---

## Try it

Requires [uv](https://docs.astral.sh/uv/) (it installs and pins Python 3.11).

```bash
uv sync                              # create .venv + install deps
cp .env.example .env                 # add your Alpaca PAPER keys
uv run pytest                        # ~200 tests

uv run python scripts/download_history.py    # bootstrap history into data/parquet/
uv run python scripts/run_orb_backtest.py    # the out-of-sample report

uv run python -m ict_bot.main --dry-run --once   # smoke test: one poll, no orders
uv run python -m ict_bot.main --dry-run          # watch a live session, zero orders
uv run python -m ict_bot.main                    # trade the Alpaca PAPER account
```

Running it unattended as a Windows service: see [`service/README.md`](service/README.md).

`ALPACA_API_KEY` / `ALPACA_SECRET` come from an Alpaca **paper** account and live in
`.env`, which is gitignored — never commit keys.

---

## Status & known limitations

Phases 0–8 of the build are complete; what remains is the forward test — *running*
the bot on paper and comparing it to the ledger.

Held to the same honesty as the rest of the project:

- **Free IEX data** is ~2–3% of consolidated volume; the opening-range extremes it
  reports can differ from the true tape.
- A code-review pass caught pre/post-market bars leaking into the opening range and
  added a shared regular-hours filter (09:30–16:00 ET) used by both the backtest and
  the live loop; the numbers above are on that clean data.
- The edge is **thin and regime-dependent** — expect uneven months and treat the
  paper forward-test as the real gate before ever considering capital.
