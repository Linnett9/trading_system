from __future__ import annotations

from core.research.ml.overlays.overlay import (
    overlay_decision_rule,
    should_reduce_exposure,
    simulate_shadow_overlay,
)


def test_champion_success_reduces_below_threshold():
    reduce_when, rule = overlay_decision_rule("champion_success")

    assert rule == "reduce_exposure_when_champion_success_probability_lt_threshold"
    assert should_reduce_exposure(0.49, 0.50, reduce_when)
    assert not should_reduce_exposure(0.50, 0.50, reduce_when)


def test_should_reduce_exposure_reduces_above_or_equal_threshold():
    reduce_when, rule = overlay_decision_rule("should_reduce_exposure")

    assert rule == "reduce_exposure_when_should_reduce_exposure_probability_gte_threshold"
    assert should_reduce_exposure(0.50, 0.50, reduce_when)
    assert not should_reduce_exposure(0.49, 0.50, reduce_when)


def test_should_reduce_exposure_threshold_is_not_inverted():
    equity = {"2024-01-01": 100.0, "2024-01-02": 90.0, "2024-01-03": 81.0}
    probabilities = {"2024-01-01": 0.60, "2024-01-02": 0.60}

    low_threshold = simulate_shadow_overlay(
        equity,
        probabilities,
        threshold=0.50,
        reduced_exposure=0.5,
        reduce_when="above_or_equal_threshold",
    )
    high_threshold = simulate_shadow_overlay(
        equity,
        probabilities,
        threshold=0.70,
        reduced_exposure=0.5,
        reduce_when="above_or_equal_threshold",
    )

    assert low_threshold is not None
    assert high_threshold is not None
    assert low_threshold.reduced_exposure_days == 1
    assert high_threshold.reduced_exposure_days == 0
