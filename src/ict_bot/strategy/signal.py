"""Signal generation (Phase 4).

On each closed bar: run detectors -> check confluences -> if score >= 60 and risk
checks pass, emit a ``CandidateSignal`` dataclass. The exact same code path is
reused by the backtest harness (Phase 5) — no separate live/backtest logic.
"""

from __future__ import annotations
