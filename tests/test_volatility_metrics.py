from __future__ import annotations

import copy
import math

import pytest

from core.research.ml.volatility_metrics import (
    VolatilityMetricInputError,
    canonical_json,
    compare_volatility_results,
    convert_volatility_representation,
    forecast_error_metrics,
    verify_volatility_comparison,
    verify_volatility_conversion,
    verify_volatility_result,
    volatility_metric_input,
    volatility_target_metrics,
)


def _input(**overrides):
    values = {
        "observation_ids": ["O1", "O2", "O3", "O4"],
        "forecast_values": [1.0, 2.0, 3.0, 4.0],
        "realised_values": [1.0, 2.0, 3.0, 4.0],
        "forecast_representation": "variance",
        "realised_representation": "variance",
        "model_identity": "synthetic_model",
        "horizon_identity": "five_sessions",
        "annualisation_factor": 252.0,
        "value_unit": "return_squared",
        "forecast_availability_timestamps": ["2025-01-01T09:00:00Z"] * 4,
        "decision_cutoff_timestamps": ["2025-01-01T10:00:00Z"] * 4,
        "realised_maturity_timestamps": ["2025-01-06T10:00:00Z"] * 4,
    }
    values.update(overrides)
    return volatility_metric_input(**values)


def test_perfect_variance_forecast_and_qlike():
    result = forecast_error_metrics(_input())
    assert result["valid"]
    assert result["aggregate_metric_values"]["mse"] == 0
    assert result["aggregate_metric_values"]["rmse"] == 0
    assert result["aggregate_metric_values"]["mae"] == 0
    assert result["aggregate_metric_values"]["qlike"]["weighted_mean"] == pytest.approx(0)


def test_perfect_volatility_forecast():
    data = _input(
        forecast_values=[0.1, 0.2, 0.3, 0.4],
        realised_values=[0.1, 0.2, 0.3, 0.4],
        forecast_representation="volatility",
        realised_representation="volatility",
        value_unit="daily_volatility",
    )
    result = forecast_error_metrics(data)
    assert result["aggregate_metric_values"]["mse"] == 0
    assert result["aggregate_metric_values"]["qlike"] is None


def test_qlike_known_numeric_under_and_over_forecasts():
    under = forecast_error_metrics(_input(forecast_values=[0.5] * 4, realised_values=[1.0] * 4))
    over = forecast_error_metrics(_input(forecast_values=[2.0] * 4, realised_values=[1.0] * 4))
    assert under["aggregate_metric_values"]["qlike"]["weighted_mean"] == pytest.approx(2 - math.log(2) - 1)
    assert over["aggregate_metric_values"]["qlike"]["weighted_mean"] == pytest.approx(0.5 - math.log(0.5) - 1)
    assert under["aggregate_metric_values"]["qlike"]["weighted_mean"] != over["aggregate_metric_values"]["qlike"]["weighted_mean"]


def test_zero_forecast_rejected_or_explicitly_floored():
    data = _input(forecast_values=[0.0, 1.0, 1.0, 1.0], realised_values=[1.0] * 4)
    blocked = forecast_error_metrics(data)
    assert blocked["blocking_reasons"] == ["QLIKE_FORECAST_VARIANCE_NOT_POSITIVE"]
    floored = forecast_error_metrics(data, qlike_forecast_floor=0.1)
    assert floored["valid"]
    assert "QLIKE_FORECAST_FLOOR_APPLIED" in floored["warnings"]


def test_zero_realised_variance_policy_is_explicit_rejection_for_ratio_qlike():
    result = forecast_error_metrics(_input(realised_values=[0.0, 1.0, 1.0, 1.0]))
    assert result["status"] == "INVALID_INPUT"
    assert result["blocking_reasons"] == ["QLIKE_REALISED_VARIANCE_ZERO_UNDEFINED"]


def test_known_mse_rmse_mae_and_bias_orientation():
    data = _input(
        forecast_values=[1.0, 3.0, 1.0, 3.0],
        realised_values=[2.0, 1.0, 2.0, 1.0],
    )
    values = forecast_error_metrics(data)["aggregate_metric_values"]
    assert values["mse"] == pytest.approx(2.5)
    assert values["rmse"] == pytest.approx(math.sqrt(2.5))
    assert values["mae"] == pytest.approx(1.5)
    assert values["mean_signed_error"] == pytest.approx(0.5)
    assert values["under_prediction_count"] == 2
    assert values["over_prediction_count"] == 2


def test_weighted_metrics_are_correct():
    data = _input(
        forecast_values=[2.0, 2.0, 2.0, 2.0],
        realised_values=[1.0, 2.0, 2.0, 2.0],
        sample_weights=[4.0, 1.0, 1.0, 1.0],
    )
    values = forecast_error_metrics(data)["aggregate_metric_values"]
    assert values["mse"] == pytest.approx(4 / 7)
    assert values["mae"] == pytest.approx(4 / 7)


def test_calibration_slope_and_insufficient_intercept_regression():
    result = forecast_error_metrics(_input(realised_values=[2.0, 4.0, 6.0, 8.0]))
    assert result["aggregate_metric_values"]["calibration_slope_through_origin"] == pytest.approx(2)
    short = volatility_metric_input(
        ["O1", "O2"], [1.0, 2.0], [2.0, 4.0],
        forecast_representation="variance", realised_representation="variance",
        model_identity="m", horizon_identity="one_session",
        annualisation_factor=252, value_unit="return_squared",
    )
    assert forecast_error_metrics(short)["aggregate_metric_values"]["calibration_status"] == "INSUFFICIENT_DATA"


def test_variance_and_volatility_annualisation_conversions():
    variance = convert_volatility_representation(
        ["O1"], [0.01], source_representation="variance",
        destination_representation="annualised_variance",
        annualisation_factor=252, source_unit="daily_variance", destination_unit="annual_variance",
    )
    volatility = convert_volatility_representation(
        ["O1"], [0.1], source_representation="volatility",
        destination_representation="annualised_volatility",
        annualisation_factor=252, source_unit="daily_volatility", destination_unit="annual_volatility",
    )
    assert variance["converted_values"] == pytest.approx([2.52])
    assert volatility["converted_values"] == pytest.approx([0.1 * math.sqrt(252)])
    assert verify_volatility_conversion(variance)["valid"]
    changed = copy.deepcopy(variance)
    changed["converted_values"][0] += 1
    assert not verify_volatility_conversion(changed)["valid"]


def test_invalid_annualisation_factor_and_representation_mismatch():
    with pytest.raises(VolatilityMetricInputError, match="ANNUALISATION_FACTOR_INVALID"):
        convert_volatility_representation(
            ["O1"], [1], source_representation="variance",
            destination_representation="volatility", annualisation_factor=0,
            source_unit="x", destination_unit="y",
        )
    result = forecast_error_metrics(_input(realised_representation="volatility"))
    assert result["status"] == "INCOMPATIBLE_REPRESENTATION"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("forecast_values", [-1.0, 1.0, 1.0, 1.0], "FORECAST_NEGATIVE"),
        ("realised_values", [float("nan"), 1.0, 1.0, 1.0], "REALISED_NON_FINITE"),
        ("observation_ids", ["O1", "O1", "O3", "O4"], "OBSERVATION_IDENTITIES_NOT_UNIQUE"),
    ],
)
def test_input_validation(field, value, reason):
    with pytest.raises(VolatilityMetricInputError, match=reason):
        _input(**{field: value})


def test_forecast_timing_and_maturity_are_enforced():
    with pytest.raises(VolatilityMetricInputError, match="FORECAST_AVAILABLE_AFTER_DECISION_CUTOFF"):
        _input(forecast_availability_timestamps=["2025-01-02T00:00:00Z"] * 4)
    with pytest.raises(VolatilityMetricInputError, match="REALISED_VALUE_NOT_MATURE_AFTER_FORECAST"):
        _input(realised_maturity_timestamps=["2025-01-01T08:00:00Z"] * 4)


def test_matched_comparison_and_mismatch_gates():
    candidate = forecast_error_metrics(_input(forecast_values=[1.1, 2.1, 3.1, 4.1]))
    benchmark = forecast_error_metrics(_input(model_identity="benchmark"))
    comparison = compare_volatility_results(candidate, benchmark, metric_path="mse")
    assert comparison["valid"]
    assert comparison["absolute_difference_candidate_minus_benchmark"] > 0
    assert verify_volatility_comparison(candidate, benchmark, comparison)["valid"]
    changed_comparison = copy.deepcopy(comparison)
    changed_comparison["candidate_metric"] += 1
    assert not verify_volatility_comparison(candidate, benchmark, changed_comparison)["valid"]
    mismatched = copy.deepcopy(benchmark)
    mismatched["population_checksum"] = "changed"
    blocked = compare_volatility_results(candidate, mismatched, metric_path="mse")
    assert blocked["status"] == "UNMATCHED_POPULATION"
    mismatched = copy.deepcopy(benchmark)
    mismatched["horizon_identity"] = "twenty_two_sessions"
    blocked = compare_volatility_results(candidate, mismatched, metric_path="mse")
    assert blocked["blocking_reasons"] == ["COMPARISON_HORIZON_MISMATCH"]


def _target_input(**overrides):
    values = {
        "forecast_values": [0.1] * 4,
        "realised_values": [0.1] * 4,
        "forecast_representation": "volatility",
        "realised_representation": "volatility",
        "value_unit": "annualised_volatility",
        "volatility_target": 0.1,
        "realised_portfolio_volatility": [0.1] * 4,
    }
    values.update(overrides)
    return _input(**values)


def test_target_perfect_tracking_overshoot_undershoot_and_tolerance():
    perfect = volatility_target_metrics(_target_input())
    assert perfect["aggregate_metric_values"]["mean_absolute_target_error"] == 0
    mixed = volatility_target_metrics(
        _target_input(realised_portfolio_volatility=[0.11, 0.09, 0.10, 0.12]),
        tolerance_bands=[0.1, 0.2],
    )
    values = mixed["aggregate_metric_values"]
    assert values["overshoot_frequency"] == pytest.approx(0.5)
    assert values["undershoot_frequency"] == pytest.approx(0.25)
    assert values["maximum_overshoot"] == pytest.approx(0.02)
    assert values["maximum_undershoot"] == pytest.approx(0.01)
    assert values["proportion_within_tolerance"]["0.1"] == pytest.approx(0.75)


def test_zero_target_blocks_percentage_metrics():
    result = volatility_target_metrics(_target_input(volatility_target=[0.1, 0.0, 0.1, 0.1]))
    assert result["blocking_reasons"] == ["VOLATILITY_TARGET_MUST_BE_POSITIVE"]


def test_overlapping_horizon_warning():
    result = forecast_error_metrics(_input(overlapping_horizons=True))
    assert "OVERLAPPING_HORIZON_OUTCOMES" in result["warnings"]


def test_independent_verification_and_mutations():
    data = _input()
    result = forecast_error_metrics(data)
    assert verify_volatility_result(data, result)["valid"]
    changed = copy.deepcopy(data)
    changed["forecast_values"][0] += 0.5
    assert not verify_volatility_result(changed, result)["valid"]
    target_data = _target_input()
    target_result = volatility_target_metrics(target_data)
    changed_target = copy.deepcopy(target_data)
    changed_target["volatility_target"][0] += 0.01
    assert not verify_volatility_result(changed_target, target_result)["valid"]
    changed_result = copy.deepcopy(result)
    changed_result["aggregate_metric_values"]["mse"] = 123
    assert not verify_volatility_result(data, changed_result)["valid"]


def test_stable_json_and_timestamp_independent_logical_checksum():
    result = forecast_error_metrics(_input())
    changed = copy.deepcopy(result)
    changed["creation_metadata"]["created_at"] = "different"
    assert result["logical_result_checksum"] == changed["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
