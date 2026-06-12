"""Benchmark the local Qwen3:4b model (Phase 6).

Measures tok/s, VRAM, and time-to-first-token for the journal workload. Target:
<2.5 GB VRAM steady-state, >=15 tok/s.

Run with: ``uv run python scripts/benchmark_llm.py``
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implemented in Phase 6.")


if __name__ == "__main__":
    main()
