# Running the bot as a Windows service (NSSM)

The live ORB loop (`python -m ict_bot.main`) is a long-running process that sleeps
between 15-minute bar boundaries and only trades during regular US market hours.
NSSM runs it as a proper Windows service: auto-start, auto-restart on crash, and a
**graceful** stop.

> ⚠️ This trades the Alpaca **paper** account (`broker.paper: true`). No real capital.

## Prerequisites

1. `uv sync` has created `.venv` in the repo root.
2. `.env` in the repo root holds `ALPACA_API_KEY` and `ALPACA_SECRET` (the bot loads
   it on startup; it is never committed).
3. `nssm.exe` downloaded from <https://nssm.cc/download>.
4. **Smoke-test first**, before installing the service:
   ```powershell
   uv run python -m ict_bot.main --dry-run --once   # one poll, logs only, no orders
   ```

## Install

```powershell
powershell -ExecutionPolicy Bypass -File service\install-service.ps1 -NssmPath C:\tools\nssm.exe
```

This registers the `ictbot` service with:

- **AppDirectory** = repo root, command = `.venv\Scripts\python.exe -m ict_bot.main`
- **Auto-start** at boot, **auto-restart** 10 s after any unexpected exit
- **Graceful stop**: NSSM sends Ctrl-C and waits 15 s. The bot catches it
  (`request_stop`), finishes the current poll, and exits — **leaving any open
  position and its resting broker stop in place**.

## Operate

```powershell
& C:\tools\nssm.exe start  ictbot
& C:\tools\nssm.exe stop   ictbot     # graceful
& C:\tools\nssm.exe restart ictbot
& C:\tools\nssm.exe remove ictbot confirm
```

Or manage it from `services.msc` (look for "ORB paper-trading bot").

## Logs & state

- **Logs:** `logs\ictbot.log` — structured JSON, daily rotation, 30-day retention.
  Watch live with `Get-Content logs\ictbot.log -Wait -Tail 20`.
- **State:** `data\state.db` (SQLite) — the durable one-trade-per-day guard and the
  trade log. Alpaca's paper account is the authoritative ledger for fills.
- **Heartbeat:** an hourly "alive" line reports whether a position is open.

## Soak test (the Phase 8 acceptance)

Let the service run a full week on paper. Confirm:

- It survives nights/weekends (it just sleeps and skips when the market is closed).
- RSS stays flat (the rolling bar frame is capped at `service.max_frame_bars`).
- Log volume is reasonable and `logs\ictbot.log` rotates daily.
- One trade per day max, always flat by the close.
