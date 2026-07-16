from __future__ import annotations

import copy

import numpy as np
import pytest

from core.research.ml.stock_level.huber_selector import (
    HuberSelectorError,
    canonical_json,
    coefficient_stability,
    compare_huber_with_ols,
    fit_huber_selector,
    huber_selector_input,
    verify_huber_result,
)


FEATURES = ["constant", "quality", "value"]


def _rows(*, outlier=False, validation_date="2024-02-01"):
    rows = []
    for index in range(12):
        quality = float(index)
        value = float((index % 4) - 1)
        target = 0.4 * quality - 0.2 * value + 1.0
        if outlier and index == 11:
            target = 1000.0
        rows.append({
            "row_id": f"T{index:02d}", "asset_id": f"A{index:02d}",
            "decision_timestamp": f"2024-01-{index + 1:02d}T10:00:00Z",
            "feature_availability_timestamp": f"2024-01-{index + 1:02d}T09:00:00Z",
            "feature_ids": FEATURES, "feature_values": [1.0, quality, value],
            "target_value": target, "target_maturity_timestamp": "2024-01-20T00:00:00Z",
            "sample_weight": 1.0, "split": "TRAINING",
        })
    for index in range(6):
        quality = float(index + 2)
        value = float((index % 3) - 1)
        rows.append({
            "row_id": f"V{index:02d}", "asset_id": f"Z{index:02d}",
            "decision_timestamp": validation_date,
            "feature_availability_timestamp": validation_date,
            "feature_ids": FEATURES, "feature_values": [1.0, quality, value],
            "target_value": 0.4 * quality - 0.2 * value + 1.0,
            "target_maturity_timestamp": "2024-02-15T00:00:00Z",
            "sample_weight": 1.0, "split": "VALIDATION",
        })
    return rows


def _input(rows=None):
    return huber_selector_input(
        rows or _rows(), target_horizon="ten_sessions",
        target_contract_identity="forward_return_10d_v1",
        feature_schema_identity="synthetic_features_v1",
        training_fold_identity="training_fold_v1",
        validation_fold_identity="validation_fold_v1",
        dataset_identity="synthetic_dataset_v1",
        source_population_checksum="synthetic_population",
    )


def test_deterministic_repeated_fit_and_estimator_identity():
    data = _input()
    first = fit_huber_selector(data)
    second = fit_huber_selector(data)
    assert first["status"] == "READY"
    assert first["model_checksum"] == second["model_checksum"]
    assert first["prediction_checksum"] == second["prediction_checksum"]
    assert first["estimator_identity"] == "sklearn.linear_model.HuberRegressor"
    assert first["dependency_version"] == "1.6.1"


def test_valid_temporal_split_and_known_linear_relation():
    result = fit_huber_selector(_input())
    assert result["valid"]
    assert result["observation_counts"] == {"training": 12, "validation": 6}
    assert result["diagnostic_summary"]["training_residual_summary"]["standard_deviation"] < 1e-4
    assert result["diagnostic_summary"]["prediction_dispersion"] > 0


def test_validation_before_training_and_maturity_leakage_rejected():
    rows = _rows(validation_date="2024-01-05T10:00:00Z")
    rows.sort(key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))
    assert fit_huber_selector(_input(rows))["status"] == "TEMPORAL_VIOLATION"
    rows = _rows()
    rows[0]["target_maturity_timestamp"] = "2024-03-01T00:00:00Z"
    assert fit_huber_selector(_input(rows))["blocking_reasons"] == ["TRAINING_TARGET_NOT_MATURE_BY_VALIDATION"]


def test_feature_order_mismatch_duplicate_and_nonfinite_inputs():
    rows = _rows()
    rows[-1]["feature_ids"] = ["constant", "quality", "zzz"]
    with pytest.raises(HuberSelectorError, match="FEATURE_ORDER_MISMATCH"):
        _input(rows)
    rows = _rows()
    rows[-1]["row_id"] = rows[-2]["row_id"]
    with pytest.raises(HuberSelectorError, match="ROW_IDENTITIES_NOT_UNIQUE"):
        _input(rows)
    rows = _rows()
    rows[0]["feature_values"][1] = np.nan
    with pytest.raises(HuberSelectorError, match="FEATURE_VALUE_NON_FINITE"):
        _input(rows)
    rows = _rows()
    rows[0]["target_value"] = np.inf
    with pytest.raises(HuberSelectorError, match="TARGET_VALUE_NON_FINITE"):
        _input(rows)
    rows = _rows()
    rows[0]["sample_weight"] = 0
    with pytest.raises(HuberSelectorError, match="SAMPLE_WEIGHT_INVALID"):
        _input(rows)


def test_constant_feature_handling_is_explicit():
    result = fit_huber_selector(_input())
    assert result["preprocessing"]["constant_feature_ids"] == ["constant"]
    assert result["preprocessing"]["scale"][0] == 1.0


def test_inadequate_training_and_empty_validation():
    rows = _rows()
    short = [row for row in rows if row["split"] == "VALIDATION"] + [
        row for row in rows if row["split"] == "TRAINING"
    ][:2]
    short.sort(key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))
    assert fit_huber_selector(_input(short))["status"] == "INSUFFICIENT_DATA"
    training_only = [row for row in rows if row["split"] == "TRAINING"]
    assert fit_huber_selector(_input(training_only))["blocking_reasons"] == ["VALIDATION_SAMPLE_EMPTY"]


def test_stable_prediction_order_and_canonical_tie_breaking():
    result = fit_huber_selector(_input())
    predictions = result["predictions"]
    scores_by_rank = sorted(predictions, key=lambda row: row["within_date_rank"])
    assert [row["predicted_return"] for row in scores_by_rank] == sorted(
        [row["predicted_return"] for row in predictions], reverse=True
    )
    rows = _rows()
    for row in rows:
        if row["split"] == "VALIDATION":
            row["feature_values"] = [1.0, 2.0, 0.0]
    tied = fit_huber_selector(_input(rows), minimum_rank_diversity=1)
    ranked = sorted(tied["predictions"], key=lambda row: row["within_date_rank"])
    assert [row["asset_id"] for row in ranked] == sorted(row["asset_id"] for row in ranked)


def test_huber_is_less_sensitive_than_ols_to_extreme_target():
    comparison = compare_huber_with_ols(_input(), _input(_rows(outlier=True)))
    assert comparison["valid"]
    assert comparison["huber_prediction_change_from_outlier"] < comparison["ols_prediction_change_from_outlier"]
    assert comparison["huber_coefficient_change_from_outlier"] < comparison["ols_coefficient_change_from_outlier"]


def test_explicit_epsilon_alpha_and_controlled_nonconvergence():
    result = fit_huber_selector(_input(), epsilon=1.5, alpha=0.01)
    assert result["model"]["epsilon"] == 1.5
    assert result["model"]["alpha"] == 0.01
    blocked = fit_huber_selector(_input(), maximum_iterations=1)
    assert blocked["status"] == "NON_CONVERGENCE"


def test_coefficient_stability_and_schema_mismatch():
    first = fit_huber_selector(_input())
    rows = _rows()
    for row in rows:
        if row["split"] == "TRAINING":
            row["target_value"] += 0.001
    second = fit_huber_selector(_input(rows))
    stability = coefficient_stability([first, second])
    assert stability["valid"]
    assert stability["ordered_feature_ids"] == FEATURES
    changed = copy.deepcopy(second)
    changed["model"]["feature_schema_checksum"] = "different"
    assert coefficient_stability([first, changed])["status"] == "FEATURE_SCHEMA_MISMATCH"


def test_prediction_verifier_and_mutations():
    data = _input()
    result = fit_huber_selector(data)
    assert verify_huber_result(data, result)["valid"]
    mutations = []
    changed = copy.deepcopy(result); changed["model"]["coefficient_vector"][1] += 1; mutations.append(changed)
    changed = copy.deepcopy(result); changed["predictions"][0]["predicted_return"] += 1; mutations.append(changed)
    changed = copy.deepcopy(result); changed["predictions"][0]["within_date_rank"] += 1; mutations.append(changed)
    changed = copy.deepcopy(result); changed["model"]["training_cutoff"] = "changed"; mutations.append(changed)
    for changed in mutations:
        assert not verify_huber_result(data, changed)["valid"]


def test_verifier_detects_input_feature_target_row_order_and_maturity_changes():
    data = _input()
    result = fit_huber_selector(data)
    for mutator in (
        lambda value: value["rows"][0]["feature_values"].__setitem__(1, 999),
        lambda value: value["rows"][0].__setitem__("target_value", 999),
        lambda value: value["rows"][0].__setitem__("row_id", "changed"),
        lambda value: value["rows"][0].__setitem__("target_maturity_timestamp", "2024-01-21T00:00:00Z"),
    ):
        changed = copy.deepcopy(data)
        mutator(changed)
        assert not verify_huber_result(changed, result)["valid"]


def test_stable_canonical_json_and_creation_timestamp_identity():
    result = fit_huber_selector(_input())
    changed = copy.deepcopy(result)
    changed["creation_metadata"]["created_at"] = "different"
    assert result["logical_result_checksum"] == changed["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
