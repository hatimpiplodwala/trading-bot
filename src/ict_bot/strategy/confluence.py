"""Confluence scoring (Phase 4).

Scores a setup 0-100 from the confluences present (HTF bias aligned, Silver
Bullet / kill zone, discount/premium, OTE, FVG+OB overlap, SMT, liquidity
sweep). Minimum threshold to consider a setup: 60. Weights are a starting
hypothesis to be validated out-of-sample, not ground truth (Gotcha #6).

The eight weights sum to 120, but the score is clamped to the documented 0-100
range. The engine only consumes the threshold test (>= 60), so clamping the top
end loses no signal value.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

MAX_SCORE = 100


@dataclass(frozen=True)
class Confluences:
    """Which confluences are present at a candidate entry.

    Field names mirror the ``signal.weights`` keys in settings.yaml exactly, so
    a present confluence always maps to a weight (guarded by a test).
    """

    htf_aligned: bool = False
    silver_bullet: bool = False
    ny_kill_zone: bool = False
    discount_premium: bool = False
    ote: bool = False
    fvg_ob_overlap: bool = False
    smt_divergence: bool = False
    liquidity_sweep: bool = False

    def active(self) -> tuple[str, ...]:
        """Names of the confluences that are present."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name))


def score_confluences(confluences: Confluences, weights: dict[str, int]) -> int:
    """Sum the weights of present confluences, clamped to 0-100."""
    raw = sum(weights[name] for name in confluences.active())
    return min(raw, MAX_SCORE)


def meets_threshold(score: int, min_score: int) -> bool:
    """Whether a score clears the minimum to consider a setup."""
    return score >= min_score
