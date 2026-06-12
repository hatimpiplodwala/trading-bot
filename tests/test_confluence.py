"""Confluence scoring tests (Phase 4).

A setup's score is the sum of the weights of the confluences present, capped at
100 (the weights sum to 120, but the PRD scores 0-100). The engine only cares
whether the score clears the min threshold (60); ranking beyond that is unused.
"""

from __future__ import annotations

import pytest

from ict_bot.config import load_settings
from ict_bot.strategy.confluence import (
    Confluences,
    meets_threshold,
    score_confluences,
)

# A fixed copy of the settings weights so the rule is tested independently of
# config drift; test_weight_keys_match_settings guards that they stay aligned.
WEIGHTS = {
    "htf_aligned": 25,
    "silver_bullet": 20,
    "ny_kill_zone": 10,
    "discount_premium": 15,
    "ote": 10,
    "fvg_ob_overlap": 15,
    "smt_divergence": 10,
    "liquidity_sweep": 15,
}


def test_no_confluences_scores_zero():
    assert score_confluences(Confluences(), WEIGHTS) == 0


def test_single_confluence_scores_its_weight():
    assert score_confluences(Confluences(htf_aligned=True), WEIGHTS) == 25
    assert score_confluences(Confluences(ote=True), WEIGHTS) == 10


def test_score_sums_active_weights():
    conf = Confluences(htf_aligned=True, fvg_ob_overlap=True, liquidity_sweep=True)
    assert score_confluences(conf, WEIGHTS) == 25 + 15 + 15


def test_score_caps_at_100():
    everything = Confluences(
        htf_aligned=True,
        silver_bullet=True,
        ny_kill_zone=True,
        discount_premium=True,
        ote=True,
        fvg_ob_overlap=True,
        smt_divergence=True,
        liquidity_sweep=True,
    )
    # Raw sum is 120; capped to the documented 0-100 range.
    assert score_confluences(everything, WEIGHTS) == 100


def test_active_lists_only_present_confluences():
    conf = Confluences(silver_bullet=True, smt_divergence=True)
    assert set(conf.active()) == {"silver_bullet", "smt_divergence"}


def test_meets_threshold():
    assert meets_threshold(60, 60)
    assert meets_threshold(75, 60)
    assert not meets_threshold(59, 60)


def test_weight_keys_match_settings():
    # The dataclass fields must cover exactly the settings weight keys, or a
    # confluence would be silently unscorable.
    settings = load_settings()
    weight_keys = set(settings["signal"]["weights"])
    field_keys = set(Confluences().active.__self__.__dataclass_fields__)
    assert field_keys == weight_keys
