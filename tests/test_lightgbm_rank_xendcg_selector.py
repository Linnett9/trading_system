from __future__ import annotations

import copy
from importlib import metadata

import pytest

from core.research.ml.ranking_labels import canonical_json, grouped_ranking_dataset
from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
    compare_integer_label_contracts,
    fit_synthetic_rank_xendcg_selector,
    fixed_rank_xendcg_configuration,
    validate_rank_xendcg_input,
    verify_rank_xendcg_result,
)


def _rows(label_type="quintile_integer", fixture="clear"):
    rows = []
    dates = ("2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z", "2026-01-03T10:00:00Z", "2026-01-04T10:00:00Z")
    for date_index, decision in enumerate(dates):
        for asset_index in range(6):
            signal = asset_index
            if fixture == "nonlinear":
                signal = (asset_index - 2.5) ** 2
            elif fixture == "context":
                signal = asset_index + (0.25 * date_index if asset_index % 2 else -0.25 * date_index)
            elif fixture == "no_signal":
                signal = date_index % 2
            relevance = asset_index if label_type == "decile_integer" else min(asset_index, 4)
            if fixture == "nonlinear":
                relevance = min(int(signal), 4 if label_type == "quintile_integer" else 9)
            if fixture == "tied":
                relevance = asset_index // 2
            rows.append({
                "row_id": f"D{date_index}A{asset_index}", "asset_id": f"A{asset_index}",
                "decision_date": decision, "feature_names": ["context", "noise", "signal"],
                "feature_values": [float(date_index), 0.0, float(signal)],
                "feature_availability_timestamp": decision, "label": relevance,
                "target_maturity_timestamp": "2026-01-05T00:00:00Z",
                "split_role": "TRAINING" if date_index < 2 else "VALIDATION",
            })
    return rows


def _dataset(label_type="quintile_integer", fixture="clear"):
    contract = (
        "within_date_quintile_relevance_v1"
        if label_type == "quintile_integer"
        else "within_date_decile_relevance_v1"
    )
    return grouped_ranking_dataset(
        _rows(label_type, fixture), label_type=label_type,
        feature_schema_identity="synthetic_rank_features_v1",
        target_contract_identity="synthetic_mature_return_v1",
        ranking_label_contract_identity=contract, split_identity="synthetic_fold_v1",
        allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
    )


def _fit(dataset=None, **kwargs):
    return fit_synthetic_rank_xendcg_selector(
        dataset or _dataset(), training_cutoff="2026-01-10T00:00:00Z", **kwargs,
    )


def test_exact_dependency_objective_and_fixed_configuration():
    assert metadata.version("lightgbm") == "4.6.0"
    config = fixed_rank_xendcg_configuration(num_threads=1)["parameters"]
    assert config["objective"] == "rank_xendcg"
    assert config["metric"] == "ndcg"
    assert config["n_jobs"] == 1
    assert config["deterministic"] is True


@pytest.mark.parametrize("threads", [1, 2])
def test_deterministic_bounded_fit_predictions_and_importance(threads):
    first, second = _fit(num_threads=threads), _fit(num_threads=threads)
    assert first["valid"] and second["valid"]
    assert first["prediction_contract"] == second["prediction_contract"]
    assert first["diagnostics"]["feature_importance"] == second["diagnostics"]["feature_importance"]
    assert first["thread_identity"] == f"bounded_threads:{threads}"


@pytest.mark.parametrize("label_type", ["quintile_integer", "decile_integer"])
def test_supported_integer_label_contracts(label_type):
    result = _fit(_dataset(label_type))
    assert result["status"] == "READY"
    assert result["prediction_contract"]["row_count"] == 12


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda rows: rows[0].__setitem__("label", 0.5), "UNSUPPORTED_LABEL_CONTRACT"),
        (lambda rows: rows[0].__setitem__("label", -1), "UNSUPPORTED_LABEL_CONTRACT"),
        (lambda rows: rows[0].__setitem__("label", None), "UNSUPPORTED_LABEL_CONTRACT"),
    ],
)
def test_fractional_negative_and_missing_labels_rejected(mutation, expected):
    rows = _rows()
    mutation(rows)
    dataset = grouped_ranking_dataset(
        rows, label_type="quintile_integer", feature_schema_identity="synthetic_rank_features_v1",
        target_contract_identity="synthetic_mature_return_v1",
        ranking_label_contract_identity="within_date_quintile_relevance_v1",
        split_identity="synthetic_fold_v1", allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
    )
    assert not dataset["valid"] or _fit(dataset)["status"] == expected


def test_continuous_percentile_contract_rejected_before_fit():
    dataset = grouped_ranking_dataset(
        [{**row, "label": row["label"] / 4} for row in _rows()],
        label_type="continuous_percentile", feature_schema_identity="synthetic_rank_features_v1",
        target_contract_identity="synthetic_mature_return_v1",
        ranking_label_contract_identity="continuous_percentile_relevance_v1",
        split_identity="synthetic_fold_v1", allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
    )
    assert _fit(dataset)["status"] == "UNSUPPORTED_LABEL_CONTRACT"


def test_group_sum_empty_noncontiguous_and_split_overlap_rejected():
    for mutate in (
        lambda data: data["group_size_vector"].__setitem__(0, 5),
        lambda data: data["group_size_vector"].__setitem__(0, 0),
        lambda data: data["groups"][0].__setitem__("start_position", 1),
        lambda data: data["rows"][0].__setitem__("split_role", "VALIDATION"),
    ):
        dataset = copy.deepcopy(_dataset())
        mutate(dataset)
        result = _fit(dataset)
        assert result["status"] in {"GROUP_STRUCTURE_INVALID", "SPLIT_OVERLAP", "INVALID_INPUT"}


def test_immature_training_target_rejected():
    rows = _rows()
    rows[0]["target_maturity_timestamp"] = "2026-01-20T00:00:00Z"
    dataset = grouped_ranking_dataset(
        rows, label_type="quintile_integer", feature_schema_identity="synthetic_rank_features_v1",
        target_contract_identity="synthetic_mature_return_v1",
        ranking_label_contract_identity="within_date_quintile_relevance_v1",
        split_identity="synthetic_fold_v1", allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
    )
    assert _fit(dataset)["status"] == "IMMATURE_TARGET"


@pytest.mark.parametrize("fixture", ["clear", "nonlinear", "context", "tied", "no_signal"])
def test_required_synthetic_fixtures_are_finite_grouped_and_reproducible(fixture):
    result = _fit(_dataset(fixture=fixture))
    assert result["valid"]
    validation = result["diagnostics"]["ranking"]["validation"]
    assert validation["finite_scores"]
    assert validation["query_group_count"] == 2
    assert result["diagnostics"]["repeatability"]["prediction_level"]
    if fixture == "clear":
        signal = next(row for row in result["diagnostics"]["feature_importance"]["features"] if row["feature_id"] == "signal")
        assert signal["gain_importance"] > 0


def test_prediction_ranks_are_complete_and_ties_use_canonical_assets():
    result = _fit(_dataset(fixture="tied"))
    predictions = result["prediction_contract"]["rows"]
    for decision in sorted({row["decision_date"] for row in predictions}):
        group = [row for row in predictions if row["decision_date"] == decision]
        assert sorted(row["within_date_rank"] for row in group) == list(range(1, 7))
        equal_score = [row for row in group if row["raw_score"] == group[0]["raw_score"]]
        if len(equal_score) > 1:
            assert [row["asset_id"] for row in equal_score] == sorted(row["asset_id"] for row in equal_score)


def test_feature_and_tree_diagnostics_include_zero_importance_and_model_shape():
    result = _fit()
    importance = result["diagnostics"]["feature_importance"]
    assert len(importance["features"]) == 3
    assert importance["zero_importance_feature_count"] >= 1
    tree = result["diagnostics"]["tree_model"]
    assert tree["number_of_trees"] <= tree["configured_estimator_count"]
    assert tree["model_byte_size"] > 0


def test_model_save_reload_and_prediction_equality(tmp_path):
    result = _fit(serialisation_directory=tmp_path)
    assert result["diagnostics"]["serialisation"] == {"saved": True, "reloaded": True}
    assert (tmp_path / "lightgbm_rank_xendcg_selector.txt").is_file()
    assert verify_rank_xendcg_result(_dataset(), result, serialisation_directory=tmp_path)["valid"]


def test_quintile_decile_contract_comparison():
    result = compare_integer_label_contracts(
        _dataset("quintile_integer"), _dataset("decile_integer"),
        training_cutoff="2026-01-10T00:00:00Z",
    )
    assert result["valid"]
    assert -1 <= result["score_rank_correlation"] <= 1
    assert 0 <= result["top_3_overlap"] <= 1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda dataset, result: dataset["rows"][0]["feature_values"].__setitem__(0, 99.0),
        lambda dataset, result: dataset["rows"][0].__setitem__("label", 4),
        lambda dataset, result: dataset["group_size_vector"].__setitem__(0, 5),
        lambda dataset, result: result["model_contract"]["parameters"].__setitem__("num_leaves", 99),
        lambda dataset, result: result["prediction_contract"]["rows"][0].__setitem__("raw_score", 99.0),
        lambda dataset, result: result["prediction_contract"]["rows"][0].__setitem__("within_date_rank", 99),
        lambda dataset, result: result["model_contract"].__setitem__("logical_model_checksum", "0" * 64),
    ],
)
def test_verifier_detects_contract_mutations(mutator, tmp_path):
    dataset = _dataset()
    result = _fit(dataset, serialisation_directory=tmp_path)
    changed_dataset, changed_result = copy.deepcopy(dataset), copy.deepcopy(result)
    mutator(changed_dataset, changed_result)
    verification = verify_rank_xendcg_result(changed_dataset, changed_result, serialisation_directory=tmp_path)
    assert not verification["valid"]


def test_stable_canonical_json_and_logical_checksum_excludes_runtime_metadata():
    first, second = _fit(), _fit()
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    changed = copy.deepcopy(first)
    changed["runtime_metadata"]["training_duration_seconds"] = 999
    assert changed["logical_result_checksum"] == first["logical_result_checksum"]
