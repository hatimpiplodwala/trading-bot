"""Confluence scoring (Phase 4).

Scores a setup 0-100 from the confluences present (HTF bias aligned, Silver
Bullet / kill zone, discount/premium, OTE, FVG+OB overlap, SMT, liquidity
sweep). Minimum threshold to consider a setup: 60. Weights are a starting
hypothesis to be validated out-of-sample, not ground truth (Gotcha #6).
"""

from __future__ import annotations
