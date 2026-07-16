from __future__ import annotations

import copy

import numpy as np
import pytest

from core.research.ml.stock_level.multi_horizon_linear_selector import (
    COMBINATION_WEIGHTS,
    HORIZON_IDS,
    MultiHorizonError,
    canonical_json,
    fit_multi_horizon_linear_selector,
    multi_horizon_linear_input,
    multi_horizon_target_contract,
    ordered_logit_adapter,
    verify_multi_horizon_result,
)


FEATURES = ["noise", "signal"]


def _rows(*, signal_type="persistent"):
    coefficients = {
        "persistent": {"return_1s": 1.0, "return_5s": 1.2, "return_10s": 1.4, "return_20s": 1.6},
        "short": {"return_1s": 1.5, "return_5s": 0.5, "return_10s": 0.0, "return_20s": 0.0},
        "reversal": {"return_1s": 1.2, "return_5s": 0.5, "return_10s": -0.5, "return_20s": -1.2},
        "none": {"return_1s": 0.0, "return_5s": 0.0, "return_10s": 0.0, "return_20s": 0.0},
    }[signal_type]
    rows = []
    for index in range(10):
        signal = float(index - 4)
        targets = {
            horizon: coefficient * signal + 0.01 * (index % 2)
            for horizon, coefficient in coefficients.items()
        }
        maturities = {
            "return_1s": "2024-01-11T00:00:00Z",
            "return_5s": "2024-01-12T00:00:00Z",
            "return_10s": "2024-01-13T00:00:00Z",
            "return_20s": "2024-01-14T00:00:00Z" if index < 8 else "2024-01-20T00:00:00Z",
        }
        states = {horizon: "MATURE" for horizon in HORIZON_IDS}
        if index >= 8:
            states["return_20s"] = "IMMATURE"
        rows.append({
            "row_id": f"T{index:02d}", "asset_id": f"A{index:02d}",
            "decision_timestamp": f"2024-01-{index + 1:02d}T10:00:00Z",
            "feature_availability_timestamp": f"2024-01-{index + 1:02d}T09:00:00Z",
            "feature_ids": FEATURES, "feature_values": [float(index % 3), signal],
            "target_values": targets, "target_maturity_timestamps": maturities,
            "target_availability_state": states, "sample_weight": 1.0, "split": "TRAINING",
        })
    for index in range(6):
        signal = float(index - 2)
        targets = {horizon: coefficient * signal for horizon, coefficient in coefficients.items()}
        rows.append({
            "row_id": f"V{index:02d}", "asset_id": f"Z{index:02d}",
            "decision_timestamp": "2024-02-01T10:00:00Z",
            "feature_availability_timestamp": "2024-02-01T09:00:00Z",
            "feature_ids": FEATURES, "feature_values": [float(index % 3), signal],
            "target_values": targets,
            "target_maturity_timestamps": {
                "return_1s": "2024-02-02T00:00:00Z", "return_5s": "2024-02-06T00:00:00Z",
                "return_10s": "2024-02-11T00:00:00Z", "return_20s": "2024-02-21T00:00:00Z",
            },
            "target_availability_state": {horizon: "MATURE" for horizon in HORIZON_IDS},
            "sample_weight": 1.0, "split": "VALIDATION",
        })
    return rows


def _input(rows=None, target_contract=None):
    return multi_horizon_linear_input(
        rows or _rows(), target_contract=target_contract or multi_horizon_target_contract(),
        feature_schema_identity="synthetic_features_v1", dataset_identity="synthetic_dataset",
        fold_identity="fold_v1", source_population_checksum="population",
    )


def _fit(data=None, **kwargs):
    return fit_multi_horizon_linear_selector(
        data or _input(), training_cutoff="2024-01-15T00:00:00Z", **kwargs
    )


def test_exact_horizon_contract_and_duplicate_or_missing_rejection():
    contract = multi_horizon_target_contract()
    assert [(row["horizon_id"], row["horizon_sessions"]) for row in contract["horizons"]] == [
        ("return_1s", 1), ("return_5s", 5), ("return_10s", 10), ("return_20s", 20),
    ]
    with pytest.raises(MultiHorizonError, match="EXACT_HORIZON_PANEL_REQUIRED"):
        multi_horizon_target_contract(contract["horizons"][:-1])
    duplicated = copy.deepcopy(contract["horizons"])
    duplicated[-1]["horizon_id"] = "return_10s"
    with pytest.raises(MultiHorizonError, match="DUPLICATE_HORIZON_ID"):
        multi_horizon_target_contract(duplicated)


def test_deterministic_ridge_and_elastic_net_fitting():
    ridge_first = _fit(model_families=["ridge"])
    ridge_second = _fit(model_families=["ridge"])
    elastic_first = _fit(model_families=["elastic_net"])
    elastic_second = _fit(model_families=["elastic_net"])
    assert ridge_first["model_checksum"] if "model_checksum" in ridge_first else ridge_first["models"][0]["model_checksum"]
    assert ridge_first["logical_result_checksum"] == ridge_second["logical_result_checksum"]
    assert elastic_first["logical_result_checksum"] == elastic_second["logical_result_checksum"]
    assert {model["estimator_identity"] for model in ridge_first["models"]} == {"sklearn.linear_model.Ridge"}
    assert {model["estimator_identity"] for model in elastic_first["models"]} == {"sklearn.linear_model.ElasticNet"}


def test_horizon_maturity_evidence_and_long_population_exclusion():
    result = _fit()
    evidence = result["target_maturity_evidence"]
    assert set(evidence) == set(HORIZON_IDS)
    assert result["training_populations"]["return_1s"]["eligible_count"] == 10
    assert result["training_populations"]["return_5s"]["eligible_count"] == 10
    assert result["training_populations"]["return_10s"]["eligible_count"] == 10
    assert result["training_populations"]["return_20s"]["eligible_count"] == 8
    assert result["training_populations"]["return_20s"]["excluded_immature_row_ids"] == ["T08", "T09"]


def test_temporal_leakage_feature_schema_and_nonfinite_validation():
    assert fit_multi_horizon_linear_selector(
        _input(), training_cutoff="2024-02-01T10:00:00Z"
    )["status"] == "TEMPORAL_VIOLATION"
    rows = _rows(); rows[-1]["feature_ids"] = ["noise", "zzz"]
    with pytest.raises(MultiHorizonError, match="FEATURE_ORDER_MISMATCH"):
        _input(rows)
    rows = _rows(); rows[0]["feature_values"][0] = np.nan
    with pytest.raises(MultiHorizonError, match="FEATURE_NON_FINITE"):
        _input(rows)
    rows = _rows(); rows[0]["target_values"]["return_5s"] = np.inf
    with pytest.raises(MultiHorizonError, match="MATURE_TARGET_NON_FINITE"):
        _input(rows)


def test_horizon_specific_preprocessing_uses_each_eligible_population():
    result = _fit()
    assert result["preprocessing"]["return_1s"]["training_population_checksum"] != result["preprocessing"]["return_20s"]["training_population_checksum"]
    long_rows = [
        row for row in _input()["rows"]
        if row["row_id"] in result["training_populations"]["return_20s"]["eligible_row_ids"]
    ]
    expected = np.mean([row["feature_values"] for row in long_rows], axis=0)
    assert result["preprocessing"]["return_20s"]["location"] == pytest.approx(expected)


def test_deterministic_rank_ties():
    rows = _rows()
    for row in rows:
        if row["split"] == "VALIDATION":
            row["feature_values"] = [1.0, 1.0]
    result = _fit(_input(rows), minimum_rank_diversity=1)
    subset = [
        row for row in result["predictions"]
        if row["horizon_id"] == "return_1s" and row["model_family"] == "ridge"
    ]
    ranked = sorted(subset, key=lambda row: row["within_date_rank"])
    assert [row["asset_id"] for row in ranked] == sorted(row["asset_id"] for row in ranked)


def test_fixed_combined_weights_and_all_horizons_available():
    result = _fit()
    assert result["status"] == "READY"
    assert result["missing_horizon_policy"] == "ALL_HORIZONS_AVAILABLE"
    assert result["combined_scores"][0]["combination_weights"] == COMBINATION_WEIGHTS
    scores = result["combined_scores"][0]["horizon_scores"]
    assert result["combined_scores"][0]["short_term_score"] == pytest.approx(
        0.6 * scores["return_1s"] + 0.4 * scores["return_5s"]
    )


def test_persistent_short_lived_reversal_and_no_signal_fixtures():
    persistent = _fit(_input(_rows(signal_type="persistent")))
    short = _fit(_input(_rows(signal_type="short")))
    reversal = _fit(_input(_rows(signal_type="reversal")))
    no_signal = _fit(_input(_rows(signal_type="none")), minimum_rank_diversity=1)
    persistent_mean = persistent["diagnostics"]["cross_horizon"]["persistence_distribution"]["mean"]
    reversal_disagreement = reversal["diagnostics"]["cross_horizon"]["disagreement_distribution"]["mean"]
    persistent_disagreement = persistent["diagnostics"]["cross_horizon"]["disagreement_distribution"]["mean"]
    assert persistent_mean > 0.8
    assert reversal_disagreement > persistent_disagreement
    assert abs(short["combined_scores"][-1]["short_term_score"]) >= abs(short["combined_scores"][-1]["long_term_score"])
    assert no_signal["valid"]


def test_missing_horizon_and_insufficient_horizon_policies():
    rows = _rows()
    for row in rows:
        if row["split"] == "TRAINING":
            row["target_availability_state"]["return_20s"] = "IMMATURE"
            row["target_maturity_timestamps"]["return_20s"] = "2024-03-01T00:00:00Z"
    partial = _fit(_input(rows))
    assert partial["status"] == "PARTIALLY_AVAILABLE"
    assert partial["missing_horizon_policy"] == "LONG_HORIZON_MISSING"
    for row in rows:
        if row["split"] == "TRAINING":
            for horizon in ("return_5s", "return_10s"):
                row["target_availability_state"][horizon] = "IMMATURE"
                row["target_maturity_timestamps"][horizon] = "2024-03-01T00:00:00Z"
    insufficient = _fit(_input(rows))
    assert insufficient["status"] == "INSUFFICIENT_HORIZONS"


def test_cross_horizon_coefficient_and_top_k_diagnostics():
    diagnostics = _fit()["diagnostics"]["cross_horizon"]
    assert diagnostics["ridge_coefficient_similarity"]
    assert diagnostics["ridge_coefficient_sign_consistency"]
    assert diagnostics["top_3_overlap"]
    assert diagnostics["rank_correlation"]


def test_ordered_logit_adapter_identity_and_target_mismatch():
    target = multi_horizon_target_contract()["horizons"][2]
    adapter = ordered_logit_adapter(
        horizon_id="return_10s", class_probabilities=[[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
        expected_relevance=[1.3, 0.7], row_ids=["A", "B"],
        model_identity="ordered_logit_fixture", fold_identity="fold_v1",
        target_checksum=target["target_checksum"], expected_target_checksum=target["target_checksum"],
    )
    assert adapter["prediction_semantics"] == "expected_ordinal_relevance"
    with pytest.raises(MultiHorizonError, match="ORDERED_ADAPTER_TARGET_MISMATCH"):
        ordered_logit_adapter(
            horizon_id="return_10s", class_probabilities=[[1.0]], expected_relevance=[0],
            row_ids=["A"], model_identity="x", fold_identity="f",
            target_checksum="changed", expected_target_checksum=target["target_checksum"],
        )


def test_verifier_success_and_mutations():
    data = _input()
    result = _fit(data)
    assert verify_multi_horizon_result(data, result)["valid"]
    changed_data = copy.deepcopy(data)
    changed_data["rows"][0]["target_maturity_timestamps"]["return_20s"] = "2024-01-13T00:00:00Z"
    assert not verify_multi_horizon_result(changed_data, result)["valid"]
    mutations = []
    changed = copy.deepcopy(result); changed["training_populations"]["return_20s"]["eligible_row_ids"].append("X"); mutations.append(changed)
    changed = copy.deepcopy(result); changed["models"][0]["coefficient_vector"][0] += 1; mutations.append(changed)
    changed = copy.deepcopy(result); changed["predictions"][0]["score"] += 1; mutations.append(changed)
    changed = copy.deepcopy(result); changed["configuration"]["combined_score_weights"]["short_term"]["return_1s"] = 0.5; mutations.append(changed)
    changed = copy.deepcopy(result); changed["combined_scores"][0]["persistence_score"] = 0; mutations.append(changed)
    for changed in mutations:
        assert not verify_multi_horizon_result(data, changed)["valid"]


def test_stable_canonical_json_and_checksums():
    result = _fit()
    changed = copy.deepcopy(result)
    changed["creation_metadata"]["created_at"] = "different"
    assert result["logical_result_checksum"] == changed["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
