from __future__ import annotations

import copy

import pytest

from core.research.ml.ranking_labels import (
    canonical_json,
    continuous_percentile_relevance,
    existing_relevance_compatibility,
    generic_group_mapping,
    grouped_ranking_dataset,
    lightgbm_group_export,
    pairwise_ranking_dataset,
    pairwise_return_margin,
    verify_framework_exports,
    verify_grouped_dataset,
    verify_pairwise_result,
    verify_percentile_result,
    xgboost_qid_export,
)


def _label_rows(values=(0.0, 1.0, 1.0, 3.0, 4.0), dates=("2026-01-01T10:00:00Z",)):
    rows = []
    for date_index, decision in enumerate(dates):
        for index, value in enumerate(values):
            rows.append({
                "row_id": f"R{date_index}{index}", "asset_id": f"A{index}",
                "decision_date": decision, "realised_target": float(value),
                "target_maturity_timestamp": "2026-02-01T00:00:00Z",
            })
    return rows


def _percentile(rows=None, minimum=5):
    return continuous_percentile_relevance(
        rows or _label_rows(), target_contract_identity="forward_return_10d",
        maturity_cutoff="2026-03-01T00:00:00Z", minimum_group_size=minimum,
    )


def _dataset_rows(label_type="continuous_percentile"):
    rows = []
    for date_index, decision in enumerate(("2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z")):
        role = "TRAINING" if date_index == 0 else "VALIDATION"
        for index in range(5):
            label = index / 4
            if label_type == "quintile_integer":
                label = index
            elif label_type == "decile_integer":
                label = index * 2
            rows.append({
                "row_id": f"D{date_index}{index}", "asset_id": f"A{index}",
                "decision_date": decision, "feature_names": ["f1", "f2"],
                "feature_values": [float(index), float(date_index)],
                "feature_availability_timestamp": decision,
                "label": label, "target_maturity_timestamp": "2026-01-03T00:00:00Z",
                "split_role": role,
            })
    return rows


def _dataset(rows=None, label_type="continuous_percentile", minimum=5):
    return grouped_ranking_dataset(
        rows or _dataset_rows(label_type), label_type=label_type,
        feature_schema_identity="feature_schema_v1",
        target_contract_identity="forward_return_10d",
        ranking_label_contract_identity="ranking_label_v1",
        split_identity="fold_v1", allowed_cutoff="2026-02-01T00:00:00Z",
        minimum_group_size=minimum,
    )


def test_percentile_direction_range_and_economic_ties():
    result = _percentile()
    labels = {row["row_id"]: row["continuous_percentile_relevance"] for row in result["labels"]}
    assert labels["R00"] == 0
    assert labels["R04"] == 1
    assert labels["R01"] == labels["R02"] == pytest.approx(0.375)
    assert all(0 <= value <= 1 for value in labels.values())


def test_canonical_order_does_not_change_tied_labels_and_repeated_result_is_stable():
    rows = _label_rows()
    first = _percentile(rows)
    tied = [row for row in first["labels"] if row["realised_target"] == 1]
    assert len({row["continuous_percentile_relevance"] for row in tied}) == 1
    assert first["logical_result_checksum"] == _percentile(copy.deepcopy(rows))["logical_result_checksum"]


def test_missing_immature_and_undersized_percentile_groups_fail():
    rows = _label_rows(); rows[0]["realised_target"] = None
    assert _percentile(rows)["status"] == "MISSING_TARGET"
    rows = _label_rows(); rows[0]["target_maturity_timestamp"] = "2026-04-01T00:00:00Z"
    assert _percentile(rows)["status"] == "IMMATURE_TARGET"
    assert _percentile(_label_rows(values=(1, 2, 3)), minimum=5)["status"] == "INSUFFICIENT_GROUP_SIZE"


def test_pair_orientation_exact_margin_submargin_and_tie_exclusion():
    rows = _label_rows(values=(0.0, 0.5, 1.0, 1.0, 2.0))
    result = pairwise_return_margin(
        rows, target_contract_identity="forward_return_10d",
        maturity_cutoff="2026-03-01T00:00:00Z",
        minimum_return_margin=1.0, maximum_pairs_per_date=100,
    )
    assert result["valid"]
    assert result["excluded_tie_count"] == 1
    assert result["excluded_submargin_count"] > 0
    assert any(pair["realised_return_difference"] == 1.0 for pair in result["pairs"])
    assert all(pair["realised_return_difference"] >= 1.0 for pair in result["pairs"])
    assert all(pair["winner_asset_id"] != pair["loser_asset_id"] for pair in result["pairs"])


def test_pairs_never_cross_dates_or_duplicate_reverse_orientation():
    result = pairwise_return_margin(
        _label_rows(values=(0, 1, 2, 3, 4), dates=("2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z")),
        target_contract_identity="forward_return_10d", maturity_cutoff="2026-03-01T00:00:00Z",
        minimum_return_margin=0.0, maximum_pairs_per_date=100,
    )
    pairs = {(row["decision_date"], row["winner_row_id"], row["loser_row_id"]) for row in result["pairs"]}
    assert len(pairs) == result["pair_count"]
    assert not any((date, loser, winner) in pairs for date, winner, loser in pairs)


def test_pair_budget_is_deterministic_and_checksum_stable():
    kwargs = {
        "target_contract_identity": "forward_return_10d",
        "maturity_cutoff": "2026-03-01T00:00:00Z",
        "minimum_return_margin": 0.0, "maximum_pairs_per_date": 3,
    }
    first = pairwise_return_margin(_label_rows(values=(0, 1, 2, 3, 4)), **kwargs)
    second = pairwise_return_margin(_label_rows(values=(0, 1, 2, 3, 4)), **kwargs)
    assert first["pair_count"] == 3
    assert first["pairs"] == second["pairs"]
    assert first["pair_population_checksum"] == second["pair_population_checksum"]


@pytest.mark.parametrize(
    ("contract_id", "label_type", "labels"),
    [
        ("within_date_quintile_relevance_v1", "quintile_integer", [0, 1, 2, 3, 4]),
        ("within_date_decile_relevance_v1", "decile_integer", [0, 2, 4, 6, 9]),
    ],
)
def test_existing_quintile_and_decile_compatibility(contract_id, label_type, labels):
    rows = _label_rows()
    output = {"contract_id": contract_id, "labels_by_row_id": {row["row_id"]: value for row, value in zip(rows, labels)}}
    result = existing_relevance_compatibility(
        output, rows, expected_contract_id=contract_id,
        target_contract_identity="forward_return_10d",
        maturity_cutoff="2026-03-01T00:00:00Z", minimum_group_size=5,
    )
    assert result["status"] == "READY"
    dataset_rows = _dataset_rows(label_type)
    assert _dataset(dataset_rows, label_type)["valid"]


def test_legacy_compatibility_is_visible_and_integer_labels_are_validated():
    rows = _label_rows()
    output = {"contract_id": "within_date_quintile_relevance_v1", "labels_by_row_id": {row["row_id"]: index for index, row in enumerate(rows)}}
    result = existing_relevance_compatibility(
        output, rows, expected_contract_id="within_date_quintile_relevance_v1",
        target_contract_identity=None, maturity_cutoff=None, minimum_group_size=5,
    )
    assert result["status"] == "LEGACY_COMPATIBLE"
    bad = _dataset_rows("quintile_integer"); bad[0]["label"] = 5
    assert _dataset(bad, "quintile_integer")["status"] == "LABEL_CONTRACT_MISMATCH"


def test_group_contiguity_canonical_order_and_explicit_small_group_failure():
    rows = list(reversed(_dataset_rows()))
    result = _dataset(rows)
    assert result["valid"]
    assert [row["decision_date"] for row in result["rows"]] == sorted(row["decision_date"] for row in result["rows"])
    for group in result["groups"]:
        members = result["rows"][group["start_position"]:group["end_position_exclusive"]]
        assert [row["asset_id"] for row in members] == sorted(row["asset_id"] for row in members)
        assert len({row["decision_date"] for row in members}) == 1
    assert _dataset(_dataset_rows()[:3], minimum=5)["status"] == "INSUFFICIENT_GROUP_SIZE"


def test_feature_availability_maturity_and_split_overlap_fail_closed():
    rows = _dataset_rows(); rows[0]["feature_availability_timestamp"] = "2026-01-02T00:00:00Z"
    assert _dataset(rows)["status"] == "INVALID_INPUT"
    rows = _dataset_rows(); rows[0]["target_maturity_timestamp"] = "2026-03-01T00:00:00Z"
    assert _dataset(rows)["status"] == "IMMATURE_TARGET"
    rows = _dataset_rows(); rows[0]["split_role"] = "VALIDATION"
    assert _dataset(rows)["status"] == "SPLIT_OVERLAP"


def test_framework_exports_and_generic_mapping_encode_identical_queries():
    dataset = _dataset()
    lightgbm = lightgbm_group_export(dataset)
    xgboost = xgboost_qid_export(dataset)
    mapping = generic_group_mapping(dataset)
    assert lightgbm["group_size_vector"] == [5, 5]
    assert xgboost["qid_vector"] == [0] * 5 + [1] * 5
    assert verify_framework_exports(dataset, lightgbm, xgboost)["valid"]
    assert [row["group_relative_position"] for row in mapping["rows"][:5]] == list(range(5))


def test_dataset_feature_target_label_and_dataset_checksums_exist():
    dataset = _dataset()
    for field in (
        "feature_schema_checksum", "target_contract_checksum",
        "ranking_label_contract_checksum", "ordered_label_checksum",
        "group_size_vector_checksum", "dataset_checksum",
    ):
        assert len(dataset[field]) == 64


def test_pairwise_dataset_remains_separate():
    pair_result = pairwise_return_margin(
        _label_rows(values=(0, 1, 2, 3, 4)),
        target_contract_identity="forward_return_10d",
        maturity_cutoff="2026-03-01T00:00:00Z",
        minimum_return_margin=1, maximum_pairs_per_date=4,
    )
    dataset = pairwise_ranking_dataset(
        pair_result, split_identity="fold_v1",
        feature_difference_representation_identity="winner_minus_loser_scaled_features_v1",
    )
    assert dataset["valid"]
    assert dataset["pair_count"] == 4
    assert "rows" not in dataset


def test_verifiers_detect_label_pair_group_qid_and_split_mutations():
    label_rows = _label_rows()
    percentile = _percentile(label_rows)
    assert verify_percentile_result(label_rows, percentile)["valid"]
    changed = copy.deepcopy(percentile); changed["labels"][0]["continuous_percentile_relevance"] += 0.1
    assert not verify_percentile_result(label_rows, changed)["valid"]
    pair = pairwise_return_margin(
        label_rows, target_contract_identity="forward_return_10d",
        maturity_cutoff="2026-03-01T00:00:00Z", minimum_return_margin=0.5,
        maximum_pairs_per_date=5,
    )
    assert verify_pairwise_result(label_rows, pair)["valid"]
    changed_pair = copy.deepcopy(pair)
    changed_pair["pairs"][0]["winner_row_id"], changed_pair["pairs"][0]["loser_row_id"] = (
        changed_pair["pairs"][0]["loser_row_id"], changed_pair["pairs"][0]["winner_row_id"]
    )
    assert not verify_pairwise_result(label_rows, changed_pair)["valid"]
    rows, dataset = _dataset_rows(), _dataset()
    assert verify_grouped_dataset(rows, dataset)["valid"]
    changed_dataset = copy.deepcopy(dataset); changed_dataset["rows"][0]["split_role"] = "VALIDATION"
    assert not verify_grouped_dataset(rows, changed_dataset)["valid"]
    lightgbm, xgboost = lightgbm_group_export(dataset), xgboost_qid_export(dataset)
    changed_lgb = copy.deepcopy(lightgbm); changed_lgb["group_size_vector"] = [4, 6]
    assert not verify_framework_exports(dataset, changed_lgb, xgboost)["valid"]
    changed_xgb = copy.deepcopy(xgboost); changed_xgb["qid_vector"][4] = 1
    assert not verify_framework_exports(dataset, lightgbm, changed_xgb)["valid"]


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("asset_id", lambda row: row.__setitem__("asset_id", "CHANGED")),
        ("decision_date", lambda row: row.__setitem__("decision_date", "2026-01-09T10:00:00Z")),
        ("feature_value", lambda row: row["feature_values"].__setitem__(0, 999.0)),
        ("feature_order", lambda row: (
            row.__setitem__("feature_names", list(reversed(row["feature_names"]))),
            row.__setitem__("feature_values", list(reversed(row["feature_values"]))),
        )),
        ("target_maturity", lambda row: row.__setitem__("target_maturity_timestamp", "2026-01-31T00:00:00Z")),
    ],
)
def test_grouped_verifier_detects_row_evidence_mutations(field, mutator):
    rows, dataset = _dataset_rows(), _dataset()
    changed = copy.deepcopy(dataset)
    mutator(changed["rows"][0])
    assert not verify_grouped_dataset(rows, changed)["valid"], field


def test_verifiers_detect_target_margin_priority_and_dataset_checksum_mutations():
    label_rows = _label_rows()
    percentile = _percentile(label_rows)
    changed_source = copy.deepcopy(label_rows)
    changed_source[0]["realised_target"] = -1.0
    assert not verify_percentile_result(changed_source, percentile)["valid"]

    pair = pairwise_return_margin(
        label_rows, target_contract_identity="forward_return_10d",
        maturity_cutoff="2026-03-01T00:00:00Z", minimum_return_margin=0.5,
        maximum_pairs_per_date=5,
    )
    changed_margin = copy.deepcopy(pair)
    changed_margin["pairs"][0]["realised_return_difference"] += 0.25
    assert not verify_pairwise_result(label_rows, changed_margin)["valid"]
    changed_priority = copy.deepcopy(pair)
    changed_priority["pairs"][0]["selection_priority"] = "0" * 64
    assert not verify_pairwise_result(label_rows, changed_priority)["valid"]

    rows, dataset = _dataset_rows(), _dataset()
    changed_checksum = copy.deepcopy(dataset)
    changed_checksum["dataset_checksum"] = "0" * 64
    assert not verify_grouped_dataset(rows, changed_checksum)["valid"]


def test_stable_canonical_json_and_creation_timestamp_identity():
    result = _percentile()
    changed = copy.deepcopy(result); changed["creation_metadata"]["created_at"] = "different"
    assert result["logical_result_checksum"] == changed["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
