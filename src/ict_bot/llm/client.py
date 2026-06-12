"""Ollama client wrapper (Phase 6).

Thin wrapper around ``ollama.Client()`` with a health check and a 30s timeout.
If Ollama is unreachable, callers log and skip — trading is never blocked.
"""

from __future__ import annotations
