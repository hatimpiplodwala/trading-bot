"""Post-trade journal (Phase 6, v1).

After a trade closes, call the LLM with the outcome and write a reflective
journal entry to SQLite. Never blocks or delays trading: on any error or
malformed output, log and skip the entry.
"""

from __future__ import annotations
