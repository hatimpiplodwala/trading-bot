"""LLM veto gate — DEFERRED to v2.

Intentionally NOT used in v1: a 4B model handed only the engine's own confluence
summary (no market data, no news) has no information edge to veto on (Gotcha
#3). Build this ONLY if fed genuinely new information (news/sentiment). If built,
it MUST default-to-APPROVE on any malformed response, timeout, or crash — never
let the LLM block a valid deterministic signal.
"""

from __future__ import annotations
