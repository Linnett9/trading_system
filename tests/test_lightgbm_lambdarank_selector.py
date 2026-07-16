from __future__ import annotations

import copy
from importlib import metadata

import pytest

from core.research.ml.ranking_labels import canonical_json, grouped_ranking_dataset
from core.research.ml.stock_level.lightgbm_lambdarank_selector import (
    compare_lambdarank_with_rank_xendcg,
    fit_synthetic_lambdarank_selector,
    fixed_lambdarank_configuration,
    label_gain_policy,
    validate_lambdarank_input,
    verify_lambdarank_result,
)


def _rows(label_type="quintile_integer", fixture="clear"):
    rows = []
    for date_index, decision in enumerate((
        "2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z",
        "2026-01-03T10:00:00Z", "2026-01-04T10:00:00Z",
    )):
        for asset_index in range(6):
            signal = asset_index
            if fixture == "nonlinear":
                signal = (asset_index - 2.5) ** 2
            elif fixture == "context":
                signal = asset_index + (0.2 * date_index if asset_index % 2 else -0.2 * date_index)
            elif fixture == "no_signal":
                signal = date_index % 2
            label = asset_index if label_type == "decile_integer" else min(asset_index, 4)
            if fixture == "nonlinear":
                label = min(int(signal), 9 if label_type == "decile_integer" else 4)
            if fixture == "tied":
                label = asset_index // 2
            rows.append({
                "row_id": f"D{date_index}A{asset_index}", "asset_id": f"A{asset_index}",
                "decision_date": decision, "feature_names": ["context", "noise", "signal"],
                "feature_values": [float(date_index), 0.0, float(signal)],
                "feature_availability_timestamp": decision, "label": label,
                "target_maturity_timestamp": "2026-01-05T00:00:00Z",
                "split_role": "TRAINING" if date_index < 2 else "VALIDATION",
            })
    return rows


def _dataset(label_type="quintile_integer", fixture="clear"):
    contract = "within_date_quintile_relevance_v1" if label_type == "quintile_integer" else "within_date_decile_relevance_v1"
    return grouped_ranking_dataset(
        _rows(label_type, fixture), label_type=label_type,
        feature_schema_identity="synthetic_rank_features_v1",
        target_contract_identity="synthetic_mature_return_v1",
        ranking_label_contract_identity=contract, split_identity="synthetic_fold_v1",
        allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
    )


def _fit(dataset=None, **kwargs):
    return fit_synthetic_lambdarank_selector(
        dataset or _dataset(), training_cutoff="2026-01-10T00:00:00Z", **kwargs,
    )


def test_dependency_objective_and_fixed_configuration():
    assert metadata.version("lightgbm") == "4.6.0"
    config = fixed_lambdarank_configuration(
        label_contract="within_date_quintile_relevance_v1", num_threads=1,
    )
    assert config["parameters"]["objective"] == "lambdarank"
    assert config["parameters"]["n_estimators"] == 24
    assert config["parameters"]["n_jobs"] == 1
    assert config["objective_difference_from_rank_xendcg"] == ["objective", "label_gain"]


def test_quintile_and_decile_gain_policies():
    quintile = label_gain_policy("within_date_quintile_relevance_v1")
    decile = label_gain_policy("within_date_decile_relevance_v1")
    assert quintile["ordered_relevance_levels"] == list(range(5))
    assert quintile["gain_values"] == [0, 1, 3, 7, 15]
    assert decile["ordered_relevance_levels"] == list(range(10))
    assert decile["gain_values"][-1] == 511


@pytest.mark.parametrize("threads", [1, 2])
def test_one_and_two_thread_deterministic_fitting(threads):
    first, second = _fit(num_threads=threads), _fit(num_threads=threads)
    assert first["valid"] and second["valid"]
    assert first["prediction_contract"] == second["prediction_contract"]
    assert first["diagnostics"]["feature_importance"] == second["diagnostics"]["feature_importance"]


@pytest.mark.parametrize("label_type", ["quintile_integer", "decile_integer"])
def test_valid_integer_relevance_contracts(label_type):
    result = _fit(_dataset(label_type))
    assert result["status"] == "READY"
    assert result["prediction_contract"]["row_count"] == 12


def test_gain_policy_mismatch_and_out_of_range_relevance_rejected():
    expected = label_gain_policy("within_date_quintile_relevance_v1")
    changed = copy.deepcopy(expected)
    changed["gain_values"][1] = 2
    assert _fit(gain_policy=changed)["status"] == "LABEL_GAIN_MISMATCH"
    dataset = copy.deepcopy(_dataset())
    dataset["rows"][0]["label"] = 5
    assert _fit(dataset)["status"] in {"INVALID_INPUT", "LABEL_GAIN_MISMATCH"}


def test_continuous_fractional_negative_and_missing_relevance_rejected():
    continuous = grouped_ranking_dataset(
        [{**row, "label": row["label"] / 4} for row in _rows()],
        label_type="continuous_percentile", feature_schema_identity="synthetic_rank_features_v1",
        target_contract_identity="synthetic_mature_return_v1",
        ranking_label_contract_identity="continuous_percentile_relevance_v1",
        split_identity="synthetic_fold_v1", allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
    )
    assert _fit(continuous)["status"] == "UNSUPPORTED_LABEL_CONTRACT"
    for value in (0.5, -1, None):
        rows = _rows()
        rows[0]["label"] = value
        built = grouped_ranking_dataset(
            rows, label_type="quintile_integer", feature_schema_identity="synthetic_rank_features_v1",
            target_contract_identity="synthetic_mature_return_v1",
            ranking_label_contract_identity="within_date_quintile_relevance_v1",
            split_identity="synthetic_fold_v1", allowed_cutoff="2026-02-01T00:00:00Z", minimum_group_size=5,
        )
        assert not built["valid"] or not _fit(built)["valid"]


def test_invalid_groups_split_and_maturity_rejected():
    mutations = (
        lambda data: data["group_size_vector"].__setitem__(0, 5),
        lambda data: data["group_size_vector"].__setitem__(0, 0),
        lambda data: data["groups"][0].__setitem__("start_position", 1),
        lambda data: data["rows"][0].__setitem__("split_role", "VALIDATION"),
        lambda data: data["rows"][0].__setitem__("target_maturity_timestamp", "2026-01-20T00:00:00Z"),
    )
    for mutate in mutations:
        dataset = copy.deepcopy(_dataset())
        mutate(dataset)
        assert not _fit(dataset)["valid"]


@pytest.mark.parametrize("fixture", ["clear", "nonlinear", "context", "tied", "no_signal"])
def test_synthetic_fixture_diagnostics_are_finite_and_grouped(fixture):
    result = _fit(_dataset(fixture=fixture))
    assert result["valid"]
    validation = result["diagnostics"]["validation"]
    assert validation["finite_scores"]
    assert validation["query_group_count"] == 2
    assert "gain_distribution" in validation
    if fixture == "clear":
        signal = next(row for row in result["diagnostics"]["feature_importance"]["features"] if row["feature_id"] == "signal")
        assert signal["gain_importance"] > 0


def test_prediction_population_ranks_and_ties_are_deterministic():
    result = _fit(_dataset(fixture="tied"))
    rows = result["prediction_contract"]["rows"]
    assert len(rows) == 12
    for decision in sorted({row["decision_date"] for row in rows}):
        group = [row for row in rows if row["decision_date"] == decision]
        assert [row["within_date_rank"] for row in group] == list(range(1, 7))


def test_feature_tree_serialisation_and_reload(tmp_path):
    result = _fit(serialisation_directory=tmp_path)
    assert result["diagnostics"]["serialisation"] == {"saved": True, "reloaded": True}
    assert len(result["diagnostics"]["feature_importance"]["features"]) == 3
    assert result["diagnostics"]["tree_model"]["model_byte_size"] > 0
    assert (tmp_path / "lightgbm_lambdarank_selector.txt").is_file()
    assert verify_lambdarank_result(_dataset(), result, serialisation_directory=tmp_path)["valid"]


def test_matched_rank_xendcg_comparison_population_and_metrics():
    comparison = compare_lambdarank_with_rank_xendcg(
        _dataset(), training_cutoff="2026-01-10T00:00:00Z",
    )
    assert comparison["valid"]
    assert comparison["lambdarank"]["ndcg"]
    assert comparison["rank_xendcg"]["ndcg"]
    assert -1 <= comparison["score_spearman_correlation"] <= 1
    assert comparison["matched_non_objective_parameters"]


def test_comparison_population_mismatch_is_visible(monkeypatch):
    import core.research.ml.stock_level.lightgbm_lambdarank_selector as owner
    original = owner.fit_synthetic_rank_xendcg_selector

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["population_checksums"]["validation"] = "CHANGED"
        return result

    monkeypatch.setattr(owner, "fit_synthetic_rank_xendcg_selector", changed)
    result = owner.compare_lambdarank_with_rank_xendcg(
        _dataset(), training_cutoff="2026-01-10T00:00:00Z",
    )
    assert result["status"] == "COMPARISON_POPULATION_MISMATCH"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda dataset, result: dataset["rows"][0]["feature_values"].__setitem__(0, 99.0),
        lambda dataset, result: dataset["rows"][0].__setitem__("label", 4),
        lambda dataset, result: result["input_contract"]["gain_policy"]["gain_values"].__setitem__(1, 2),
        lambda dataset, result: dataset["group_size_vector"].__setitem__(0, 5),
        lambda dataset, result: result["model_contract"]["parameters"].__setitem__("num_leaves", 99),
        lambda dataset, result: result["prediction_contract"]["rows"][0].__setitem__("raw_score", 99.0),
        lambda dataset, result: result["prediction_contract"]["rows"][0].__setitem__("within_date_rank", 99),
        lambda dataset, result: result["model_contract"].__setitem__("logical_model_checksum", "0" * 64),
    ],
)
def test_verifier_detects_mutations(mutator, tmp_path):
    dataset, result = _dataset(), _fit(serialisation_directory=tmp_path)
    changed_dataset, changed_result = copy.deepcopy(dataset), copy.deepcopy(result)
    mutator(changed_dataset, changed_result)
    assert not verify_lambdarank_result(
        changed_dataset, changed_result, serialisation_directory=tmp_path,
    )["valid"]


def test_stable_canonical_json_and_checksums():
    first, second = _fit(), _fit()
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
