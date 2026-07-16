from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.research.ml.probabilistic_metrics import (
    ProbabilisticInputError,
    canonical_json,
    classification_metrics,
    distribution_negative_log_likelihood,
    empirical_crps,
    matched_metric_comparison,
    prediction_interval_metrics,
    probabilistic_input,
    quantile_metrics,
)


def _base(n=4, *, targets=None, weights=None, model="model"):
    return probabilistic_input(
        [f"2024-01-{index + 1:02d}" for index in range(n)],
        targets if targets is not None else list(range(n)),
        prediction_type="generic", model_id=model, sample_weights=weights,
        target_unit="return",
    )


def _logical(result):
    return {key: value for key, value in result.items() if key != "creation_metadata"}


def test_perfect_and_uniform_multiclass_scores():
    base = _base(targets=[0, 1, 2, 1])
    perfect = classification_metrics(base, class_probabilities=np.eye(3)[[0, 1, 2, 1]])
    uniform = classification_metrics(base, class_probabilities=np.ones((4, 3)) / 3)
    assert perfect["valid"]
    assert perfect["metric_values"]["negative_log_likelihood"] == pytest.approx(0)
    assert perfect["metric_values"]["multiclass_brier_score"] == pytest.approx(0)
    assert perfect["metric_values"]["ranked_probability_score"] == pytest.approx(0)
    assert uniform["metric_values"]["negative_log_likelihood"] == pytest.approx(math.log(3))
    assert uniform["metric_values"]["multiclass_brier_score"] > 0


@pytest.mark.parametrize("probabilities,reason", [
    ([[0.4, 0.4], [0.5, 0.5]], "SUM"),
    ([[-0.1, 1.1], [0.5, 0.5]], "OUT_OF_BOUNDS"),
])
def test_invalid_probability_contract(probabilities, reason):
    result = classification_metrics(_base(n=2, targets=[0, 1]), class_probabilities=probabilities)
    assert not result["valid"]
    assert reason in result["blocking_reasons"][0]


def test_log_loss_clipping_is_deterministic():
    base = _base(n=2, targets=[0, 1])
    probabilities = [[0.0, 1.0], [1.0, 0.0]]
    first = classification_metrics(base, class_probabilities=probabilities, clipping_epsilon=1e-6)
    second = classification_metrics(base, class_probabilities=probabilities, clipping_epsilon=1e-6)
    assert first["metric_values"]["negative_log_likelihood"] == pytest.approx(-math.log(1e-6))
    assert _logical(first) == _logical(second)


def test_ranked_probability_score_respects_order_and_expected_relevance():
    base = _base(n=2, targets=[2, 2])
    close = classification_metrics(base, class_probabilities=[[0.0, 0.4, 0.6], [0.0, 0.4, 0.6]])
    far = classification_metrics(base, class_probabilities=[[0.4, 0.0, 0.6], [0.4, 0.0, 0.6]])
    assert close["metric_values"]["ranked_probability_score"] < far["metric_values"]["ranked_probability_score"]
    assert close["metric_values"]["expected_relevance"] == pytest.approx([1.6, 1.6])
    assert close["metric_values"]["top_class_accuracy_is_primary"] is False


def test_calibration_policy_distinguishes_calibrated_and_miscalibrated():
    targets = [1] * 8 + [0] * 2
    base = _base(n=10, targets=targets)
    calibrated = classification_metrics(base, class_probabilities=[[0.2, 0.8]] * 10, calibration_bins=5)
    miscalibrated = classification_metrics(base, class_probabilities=[[0.01, 0.99]] * 10, calibration_bins=5)
    assert calibrated["metric_values"]["expected_calibration_error"] < miscalibrated["metric_values"]["expected_calibration_error"]
    assert calibrated["metric_values"]["calibration_bins"] == classification_metrics(base, class_probabilities=[[0.2, 0.8]] * 10, calibration_bins=5)["metric_values"]["calibration_bins"]


def test_weighted_classification_aggregation():
    base = _base(n=2, targets=[0, 1], weights=[100, 1])
    weighted = classification_metrics(base, class_probabilities=[[1, 0], [1, 0]])
    unweighted = classification_metrics(_base(n=2, targets=[0, 1]), class_probabilities=[[1, 0], [1, 0]])
    assert weighted["metric_values"]["negative_log_likelihood"] < unweighted["metric_values"]["negative_log_likelihood"]
    assert weighted["weighting_policy"] == "explicit_nonnegative_sample_weights"


def test_pinball_loss_calibration_and_equality():
    base = _base(n=3, targets=[0, 1, 2])
    result = quantile_metrics(base, quantile_levels=[0.25, 0.5, 0.75], quantile_forecasts=[
        [0, 0, 0], [0, 1, 2], [1, 2, 3],
    ])
    assert result["valid"]
    assert result["metric_values"]["per_quantile"]["0.25"]["pinball_loss"] == pytest.approx(1 / 6)
    assert result["metric_values"]["per_quantile"]["0.5"]["pinball_loss"] == pytest.approx(0)
    assert result["metric_values"]["quantile_crossing_count"] == 0


def test_quantile_crossing_report_and_strict_rejection():
    base = _base(n=2, targets=[0, 1])
    forecasts = [[1, 0], [2, 1]]
    diagnostic = quantile_metrics(base, quantile_levels=[0.25, 0.75], quantile_forecasts=forecasts)
    strict = quantile_metrics(base, quantile_levels=[0.25, 0.75], quantile_forecasts=forecasts, require_non_crossing=True)
    assert diagnostic["metric_values"]["quantile_crossing_count"] == 2
    assert diagnostic["metric_values"]["quantile_crossing_severity"] == pytest.approx(2)
    assert "QUANTILE_CROSSING_PRESENT" in diagnostic["warnings"]
    assert strict["status"] == "INVALID_INPUT"
    duplicate = quantile_metrics(base, quantile_levels=[0.5, 0.5], quantile_forecasts=forecasts)
    assert duplicate["status"] == "INVALID_INPUT"


def test_interval_coverage_width_score_and_undercoverage():
    base = _base(n=4, targets=[0, 1, 2, 3])
    covered = prediction_interval_metrics(base, lower_bounds=[-1, 0, 1, 2], upper_bounds=[1, 2, 3, 4], nominal_coverage=0.8, target_buckets=["low", "low", "high", "high"])
    narrow = prediction_interval_metrics(base, lower_bounds=[0, 1, 2, 4], upper_bounds=[0, 1, 2, 4], nominal_coverage=0.8)
    wide = prediction_interval_metrics(base, lower_bounds=[-100] * 4, upper_bounds=[100] * 4, nominal_coverage=0.8)
    assert covered["metric_values"]["empirical_coverage"] == 1
    assert covered["metric_values"]["conditional_coverage"]["low"]["coverage"] == 1
    assert narrow["metric_values"]["under_coverage_count"] == 1
    assert narrow["metric_values"]["upper_bound_miss_count"] == 0
    assert narrow["metric_values"]["lower_bound_miss_count"] == 1
    assert wide["metric_values"]["empirical_coverage"] == 1
    assert wide["metric_values"]["mean_interval_width"] > covered["metric_values"]["mean_interval_width"]
    assert wide["metric_values"]["mean_interval_score"] > covered["metric_values"]["mean_interval_score"]


def test_empirical_sample_crps_exact_and_weighted():
    base = _base(n=2, targets=[0, 2], weights=[3, 1])
    result = empirical_crps(base, predictive_samples=[[0, 0], [0, 2]])
    assert result["valid"]
    assert result["metric_values"]["observation_crps"] == pytest.approx([0, 0.5])
    assert result["metric_values"]["mean_crps"] == pytest.approx(0.125)
    insufficient = empirical_crps(base, predictive_samples=[[0], [2]])
    assert insufficient["status"] == "INSUFFICIENT_DATA"


def test_gaussian_nll_and_distribution_failures():
    base = _base(n=2, targets=[0, 1])
    valid = distribution_negative_log_likelihood(base, family="gaussian", parameters={"mean": [0, 1], "scale": [1, 1]})
    assert valid["valid"]
    assert valid["metric_values"]["mean_negative_log_likelihood"] == pytest.approx(0.5 * math.log(2 * math.pi))
    invalid = distribution_negative_log_likelihood(base, family="gaussian", parameters={"mean": [0, 1], "scale": [1, 0]})
    unsupported = distribution_negative_log_likelihood(base, family="student_t", parameters={})
    assert invalid["status"] == "INVALID_INPUT"
    assert unsupported["status"] == "UNSUPPORTED_PREDICTION_TYPE"


def test_matched_comparison_success_and_population_rejection():
    candidate = classification_metrics(_base(n=2, targets=[0, 1], model="a"), class_probabilities=[[0.9, 0.1], [0.1, 0.9]])
    benchmark = classification_metrics(_base(n=2, targets=[0, 1], model="b"), class_probabilities=[[0.6, 0.4], [0.4, 0.6]])
    comparison = matched_metric_comparison(candidate, benchmark, metric_path="negative_log_likelihood", lower_is_better=True)
    assert comparison["valid"]
    assert comparison["metric_values"]["improvement_value"] > 0
    assert comparison["metric_values"]["block_bootstrap_compatible"]
    other = classification_metrics(_base(n=3, targets=[0, 1, 0]), class_probabilities=[[0.5, 0.5]] * 3)
    mismatch = matched_metric_comparison(candidate, other, metric_path="negative_log_likelihood", lower_is_better=True)
    assert mismatch["status"] == "UNMATCHED_POPULATION"


def test_nonfinite_and_immature_inputs_fail_closed():
    with pytest.raises(ProbabilisticInputError, match="NON_FINITE"):
        _base(n=2, targets=[0, math.nan])
    with pytest.raises(ProbabilisticInputError, match="AFTER_TARGET_MATURITY"):
        probabilistic_input(
            ["a"], [0], prediction_type="classification", model_id="m",
            target_maturity_timestamps=["2024-01-02"],
            prediction_availability_timestamps=["2024-01-03"],
        )


def test_json_ordering_and_logical_checksum_are_stable():
    base = _base(n=2, targets=[0, 1])
    first = classification_metrics(base, class_probabilities=[[0.8, 0.2], [0.2, 0.8]])
    second = classification_metrics(base, class_probabilities=[[0.8, 0.2], [0.2, 0.8]])
    assert _logical(first) == _logical(second)
    encoded = canonical_json(_logical(first))
    assert encoded == canonical_json(json.loads(encoded))
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
