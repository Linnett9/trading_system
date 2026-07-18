from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence


PERCENTILE_CONTRACT = "continuous_percentile_relevance_v1"
PAIRWISE_CONTRACT = "pairwise_return_margin_v1"
GROUPED_DATASET_CONTRACT = "grouped_ranking_dataset_v1"
PAIRWISE_DATASET_CONTRACT = "pairwise_ranking_dataset_v1"
LIGHTGBM_EXPORT_CONTRACT = "grouped_ranking_lightgbm_export_v1"
XGBOOST_EXPORT_CONTRACT = "grouped_ranking_xgboost_export_v1"
GENERIC_EXPORT_CONTRACT = "grouped_ranking_mapping_v1"
COMPATIBILITY_CONTRACT = "existing_relevance_compatibility_v1"
INTEGER_RELEVANCE_CONTRACT = "mature_training_integer_relevance.v1"
STATUSES = {
    "READY", "LEGACY_COMPATIBLE", "INVALID_INPUT", "INSUFFICIENT_GROUP_SIZE",
    "IMMATURE_TARGET", "MISSING_TARGET", "LABEL_CONTRACT_MISMATCH",
    "FEATURE_SCHEMA_MISMATCH", "SPLIT_OVERLAP", "NONCONTIGUOUS_GROUP",
    "PAIR_MARGIN_VIOLATION", "PAIR_BUDGET_EXCEEDED", "CHECKSUM_MISMATCH",
    "NUMERICAL_FAILURE",
}


class RankingLabelError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def mature_training_integer_relevance(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_contract_identity: str,
    maturity_cutoff: str,
    bins: int = 5,
    minimum_group_size: int | None = None,
) -> dict[str, Any]:
    """Build registered integer relevance after fail-closed maturity filtering."""
    if bins not in {5, 10}:
        raise RankingLabelError(
            "LABEL_CONTRACT_MISMATCH", "INTEGER_RELEVANCE_BINS_UNSUPPORTED"
        )
    if any(str(row.get("split_role") or "") != "TRAINING" for row in rows):
        raise RankingLabelError(
            "SPLIT_OVERLAP", "RELEVANCE_THRESHOLDS_REQUIRE_TRAINING_ROWS_ONLY"
        )
    ordered_source = sorted(
        rows,
        key=lambda row: (
            str(row.get("decision_date") or ""),
            str(row.get("asset_id") or ""),
            str(row.get("row_id") or ""),
        ),
    )
    mature = _label_source_rows(ordered_source, maturity_cutoff)
    from core.research.ml.ranking import relevance_labels
    result = relevance_labels(
        [
            {
                "row_id": row["row_id"],
                "decision_timestamp": row["decision_date"],
                "actual_forward_return_10d": row["realised_target"],
            }
            for row in mature
        ],
        bins=bins,
        minimum_group_size=minimum_group_size,
    )
    logical = {
        "contract_version": INTEGER_RELEVANCE_CONTRACT,
        "status": "READY",
        "valid": True,
        "target_contract_identity": target_contract_identity,
        "maturity_cutoff": maturity_cutoff,
        "source_role": "TRAINING",
        "threshold_population_policy": (
            "within-decision-date training rows mature by cutoff only"
        ),
        "missing_or_nonfinite_target_policy": "fail_closed",
        "label_contract_identity": result["contract_id"],
        "label_range": [0, bins - 1],
        "tie_policy": result["tie_policy"],
        "labels_by_row_id": result["labels_by_row_id"],
        "class_distributions": result["class_distributions"],
        "source_population_checksum": canonical_hash(mature),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def continuous_percentile_relevance(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_contract_identity: str,
    maturity_cutoff: str,
    minimum_group_size: int = 5,
) -> dict[str, Any]:
    try:
        source = _label_source_rows(rows, maturity_cutoff)
        grouped = _groups(source)
        labels = []
        group_evidence = []
        for decision, group in grouped.items():
            if len(group) < minimum_group_size:
                raise RankingLabelError("INSUFFICIENT_GROUP_SIZE", f"GROUP_TOO_SMALL:{decision}")
            economic = sorted(group, key=lambda row: (row["realised_target"], row["asset_id"], row["row_id"]))
            positions = {}
            cursor = 0
            while cursor < len(economic):
                end = cursor + 1
                while end < len(economic) and economic[end]["realised_target"] == economic[cursor]["realised_target"]:
                    end += 1
                average_position = (cursor + end - 1) / 2
                percentile = average_position / (len(economic) - 1)
                for row in economic[cursor:end]:
                    positions[row["row_id"]] = percentile
                cursor = end
            ordered = sorted(group, key=lambda row: (row["asset_id"], row["row_id"]))
            for row in ordered:
                labels.append({
                    "decision_date": decision, "asset_id": row["asset_id"], "row_id": row["row_id"],
                    "realised_target": row["realised_target"],
                    "target_maturity_timestamp": row["target_maturity_timestamp"],
                    "continuous_percentile_relevance": float(positions[row["row_id"]]),
                    "target_contract_identity": target_contract_identity,
                })
            group_evidence.append({
                "decision_date": decision, "row_count": len(group),
                "row_population_checksum": canonical_hash([row["row_id"] for row in ordered]),
                "asset_population_checksum": canonical_hash([row["asset_id"] for row in ordered]),
            })
        logical = {
            "contract_version": PERCENTILE_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "minimum_group_size": int(minimum_group_size), "missingness_policy": "fail_closed",
            "tie_policy": "equal_targets_receive_average_zero_based_rank_divided_by_group_size_minus_one",
            "direction_policy": "lower_return_toward_zero_higher_return_toward_one",
            "target_contract_identity": str(target_contract_identity),
            "maturity_cutoff": str(maturity_cutoff),
            "labels": labels, "row_count": len(labels), "group_count": len(grouped),
            "label_population_checksum": canonical_hash(labels),
            "grouped_date_checksum": canonical_hash(group_evidence),
            "source_population_checksum": canonical_hash(source),
        }
        logical["contract_checksum"] = canonical_hash({key: value for key, value in logical.items() if key not in {"labels", "label_population_checksum"}})
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except RankingLabelError as exc:
        return _blocked(PERCENTILE_CONTRACT, exc)


def pairwise_return_margin(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_contract_identity: str,
    maturity_cutoff: str,
    minimum_return_margin: float,
    maximum_pairs_per_date: int,
) -> dict[str, Any]:
    try:
        margin = float(minimum_return_margin)
        budget = int(maximum_pairs_per_date)
        if not math.isfinite(margin) or margin < 0 or budget < 1:
            raise RankingLabelError("INVALID_INPUT", "PAIR_CONFIGURATION_INVALID")
        source = _label_source_rows(rows, maturity_cutoff)
        grouped = _groups(source)
        pairs, evidence = [], []
        total_ties = total_submargin = total_candidates = 0
        for decision, group in grouped.items():
            ordered = sorted(group, key=lambda row: (row["asset_id"], row["row_id"]))
            candidates = []
            tie_count = submargin_count = 0
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1:]:
                    difference = left["realised_target"] - right["realised_target"]
                    if difference == 0:
                        tie_count += 1
                        continue
                    if abs(difference) < margin:
                        submargin_count += 1
                        continue
                    winner, loser = (left, right) if difference > 0 else (right, left)
                    realised_difference = winner["realised_target"] - loser["realised_target"]
                    pair_id = f"{decision}|{winner['asset_id']}|{loser['asset_id']}|{winner['row_id']}|{loser['row_id']}"
                    priority = canonical_hash({"contract": PAIRWISE_CONTRACT, "pair_id": pair_id, "margin": margin})
                    pair = {
                        "pair_id": pair_id, "decision_date": decision,
                        "winner_asset_id": winner["asset_id"], "loser_asset_id": loser["asset_id"],
                        "winner_row_id": winner["row_id"], "loser_row_id": loser["row_id"],
                        "realised_return_difference": realised_difference,
                        "configured_minimum_margin": margin, "selection_priority": priority,
                        "target_contract_identity": target_contract_identity,
                        "winner_target_maturity_timestamp": winner["target_maturity_timestamp"],
                        "loser_target_maturity_timestamp": loser["target_maturity_timestamp"],
                    }
                    pair["pair_checksum"] = canonical_hash(pair)
                    candidates.append(pair)
            selected = sorted(candidates, key=lambda row: (row["selection_priority"], row["pair_id"]))[:budget]
            selected = sorted(selected, key=lambda row: row["pair_id"])
            pairs.extend(selected)
            evidence.append({
                "decision_date": decision, "source_row_count": len(group),
                "candidate_pair_count": len(candidates), "selected_pair_count": len(selected),
                "excluded_tie_count": tie_count, "excluded_submargin_count": submargin_count,
                "pair_group_checksum": canonical_hash(selected),
            })
            total_ties += tie_count
            total_submargin += submargin_count
            total_candidates += len(candidates)
        logical = {
            "contract_version": PAIRWISE_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "target_contract_identity": str(target_contract_identity),
            "maturity_cutoff": str(maturity_cutoff),
            "minimum_return_margin": margin, "maximum_pairs_per_date": budget,
            "selection_policy": "ascending_sha256_priority_then_pair_id_without_random_sampling",
            "pairs": pairs, "pair_count": len(pairs), "candidate_pair_count": total_candidates,
            "excluded_tie_count": total_ties, "excluded_submargin_count": total_submargin,
            "group_evidence": evidence,
            "pair_population_checksum": canonical_hash(pairs),
            "grouped_date_checksum": canonical_hash(evidence),
            "source_population_checksum": canonical_hash(source),
        }
        logical["configuration_checksum"] = canonical_hash({
            "target_contract_identity": target_contract_identity, "maturity_cutoff": maturity_cutoff,
            "minimum_return_margin": margin, "maximum_pairs_per_date": budget,
            "selection_policy": logical["selection_policy"],
        })
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except RankingLabelError as exc:
        return _blocked(PAIRWISE_CONTRACT, exc)


def grouped_ranking_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_type: str,
    feature_schema_identity: str,
    target_contract_identity: str,
    ranking_label_contract_identity: str,
    split_identity: str,
    allowed_cutoff: str,
    minimum_group_size: int,
) -> dict[str, Any]:
    try:
        if label_type not in {"quintile_integer", "decile_integer", "continuous_percentile"}:
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "LABEL_TYPE_UNSUPPORTED")
        normalised = []
        feature_names = None
        seen_rows, seen_asset_dates = set(), set()
        for raw in rows:
            row_id, asset = str(raw.get("row_id") or ""), str(raw.get("asset_id") or "")
            decision, role = str(raw.get("decision_date") or ""), str(raw.get("split_role") or "")
            features = [float(value) for value in raw.get("feature_values", ())]
            names = [str(value) for value in raw.get("feature_names", ())]
            label = raw.get("label")
            availability = str(raw.get("feature_availability_timestamp") or "")
            maturity = str(raw.get("target_maturity_timestamp") or "")
            if not row_id or not asset or role not in {"TRAINING", "VALIDATION"}:
                raise RankingLabelError("INVALID_INPUT", "DATASET_ROW_IDENTITY_INVALID")
            if row_id in seen_rows or (decision, asset) in seen_asset_dates:
                raise RankingLabelError("SPLIT_OVERLAP", "DUPLICATE_ROW_OR_ASSET_DATE")
            seen_rows.add(row_id); seen_asset_dates.add((decision, asset))
            if names != sorted(names) or len(names) != len(features) or len(names) != len(set(names)):
                raise RankingLabelError("FEATURE_SCHEMA_MISMATCH", "FEATURE_ORDER_OR_DIMENSION_INVALID")
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RankingLabelError("FEATURE_SCHEMA_MISMATCH", "FEATURE_ORDER_MISMATCH")
            if not all(math.isfinite(value) for value in features):
                raise RankingLabelError("INVALID_INPUT", "FEATURE_NON_FINITE")
            label = _validate_label(label, label_type)
            if _time(availability) > _time(decision):
                raise RankingLabelError("INVALID_INPUT", "FEATURE_AVAILABLE_AFTER_DECISION")
            if _time(maturity) > _time(allowed_cutoff):
                raise RankingLabelError("IMMATURE_TARGET", "TARGET_MATURES_AFTER_ALLOWED_CUTOFF")
            normalised.append({
                "row_id": row_id, "asset_id": asset, "decision_date": decision,
                "feature_names": names, "feature_values": features,
                "feature_availability_timestamp": availability, "label": label,
                "label_type": label_type, "target_maturity_timestamp": maturity,
                "split_identity": split_identity, "split_role": role,
                "group_id": f"decision_date:{decision}",
            })
        ordered = sorted(normalised, key=lambda row: (row["decision_date"], row["asset_id"], row["row_id"]))
        groups = _dataset_groups(ordered, minimum_group_size)
        roles_by_date = {}
        for row in ordered:
            roles_by_date.setdefault(row["decision_date"], set()).add(row["split_role"])
        if any(len(roles) != 1 for roles in roles_by_date.values()):
            raise RankingLabelError("SPLIT_OVERLAP", "DECISION_DATE_MIXES_SPLIT_ROLES")
        training_dates = [date for date, roles in roles_by_date.items() if roles == {"TRAINING"}]
        validation_dates = [date for date, roles in roles_by_date.items() if roles == {"VALIDATION"}]
        if training_dates and validation_dates and max(training_dates) >= min(validation_dates):
            raise RankingLabelError("SPLIT_OVERLAP", "TRAINING_VALIDATION_DATE_OVERLAP")
        group_sizes = [group["group_size"] for group in groups]
        logical = {
            "contract_version": GROUPED_DATASET_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "rows": ordered, "row_count": len(ordered), "group_count": len(groups),
            "feature_count": len(feature_names or []), "feature_names": feature_names or [],
            "label_type": label_type, "split_identity": split_identity,
            "feature_schema_identity": str(feature_schema_identity),
            "target_contract_identity": str(target_contract_identity),
            "ranking_label_contract_identity": str(ranking_label_contract_identity),
            "minimum_group_size": int(minimum_group_size), "groups": groups,
            "group_size_vector": group_sizes,
            "feature_schema_checksum": canonical_hash({"identity": feature_schema_identity, "features": feature_names}),
            "target_contract_checksum": canonical_hash({"identity": target_contract_identity}),
            "ranking_label_contract_checksum": canonical_hash({"identity": ranking_label_contract_identity, "label_type": label_type}),
            "ordered_row_population_checksum": canonical_hash([row["row_id"] for row in ordered]),
            "ordered_label_checksum": canonical_hash([row["label"] for row in ordered]),
            "decision_date_population_checksum": canonical_hash(sorted(roles_by_date)),
            "group_size_vector_checksum": canonical_hash(group_sizes),
            "split_checksum": canonical_hash({
                "split_identity": split_identity,
                "roles": [(row["row_id"], row["split_role"]) for row in ordered],
            }),
            "allowed_cutoff": allowed_cutoff,
        }
        logical["dataset_checksum"] = canonical_hash(logical)
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except RankingLabelError as exc:
        return _blocked(GROUPED_DATASET_CONTRACT, exc)


def lightgbm_group_export(dataset: Mapping[str, Any]) -> dict[str, Any]:
    _ready_dataset(dataset)
    logical = {
        "contract_version": LIGHTGBM_EXPORT_CONTRACT,
        "feature_matrix": [row["feature_values"] for row in dataset["rows"]],
        "labels": [row["label"] for row in dataset["rows"]],
        "group_size_vector": list(dataset["group_size_vector"]),
        "ordered_row_ids": [row["row_id"] for row in dataset["rows"]],
        "ordered_group_identities": [group["group_id"] for group in dataset["groups"]],
        "feature_names": list(dataset["feature_names"]),
        "dataset_checksum": dataset["dataset_checksum"],
    }
    logical["export_checksum"] = canonical_hash(logical)
    return logical


def xgboost_qid_export(dataset: Mapping[str, Any]) -> dict[str, Any]:
    _ready_dataset(dataset)
    qid = []
    for group_index, size in enumerate(dataset["group_size_vector"]):
        qid.extend([group_index] * size)
    logical = {
        "contract_version": XGBOOST_EXPORT_CONTRACT,
        "feature_matrix": [row["feature_values"] for row in dataset["rows"]],
        "labels": [row["label"] for row in dataset["rows"]],
        "qid_vector": qid,
        "ordered_row_ids": [row["row_id"] for row in dataset["rows"]],
        "ordered_group_identities": [group["group_id"] for group in dataset["groups"]],
        "feature_names": list(dataset["feature_names"]),
        "dataset_checksum": dataset["dataset_checksum"],
    }
    logical["export_checksum"] = canonical_hash(logical)
    return logical


def generic_group_mapping(dataset: Mapping[str, Any]) -> dict[str, Any]:
    _ready_dataset(dataset)
    mappings = []
    position = 0
    for group in dataset["groups"]:
        for relative in range(group["group_size"]):
            row = dataset["rows"][position]
            mappings.append({
                "row_position": position, "row_id": row["row_id"], "asset_id": row["asset_id"],
                "decision_date": row["decision_date"], "group_id": group["group_id"],
                "group_relative_position": relative, "label": row["label"],
                "split_role": row["split_role"],
            })
            position += 1
    logical = {"contract_version": GENERIC_EXPORT_CONTRACT, "rows": mappings, "dataset_checksum": dataset["dataset_checksum"]}
    logical["mapping_checksum"] = canonical_hash(logical)
    return logical


def existing_relevance_compatibility(
    existing_output: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    *,
    expected_contract_id: str,
    target_contract_identity: str | None,
    maturity_cutoff: str | None,
    minimum_group_size: int,
) -> dict[str, Any]:
    try:
        if expected_contract_id not in {"within_date_quintile_relevance_v1", "within_date_decile_relevance_v1"}:
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "EXISTING_CONTRACT_UNSUPPORTED")
        if existing_output.get("contract_id") != expected_contract_id:
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "EXISTING_CONTRACT_ID_MISMATCH")
        labels = dict(existing_output.get("labels_by_row_id") or {})
        expected_range = range(5) if "quintile" in expected_contract_id else range(10)
        if set(labels) != {str(row["row_id"]) for row in source_rows}:
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "EXISTING_LABEL_POPULATION_MISMATCH")
        if any(not isinstance(value, int) or value not in expected_range for value in labels.values()):
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "EXISTING_INTEGER_LABEL_INVALID")
        grouped = _groups(_label_source_rows(source_rows, maturity_cutoff) if maturity_cutoff else _legacy_source_rows(source_rows))
        if any(len(group) < minimum_group_size for group in grouped.values()):
            raise RankingLabelError("INSUFFICIENT_GROUP_SIZE", "EXISTING_GROUP_TOO_SMALL")
        legacy = target_contract_identity is None or maturity_cutoff is None
        logical = {
            "contract_version": COMPATIBILITY_CONTRACT,
            "status": "LEGACY_COMPATIBLE" if legacy else "READY", "valid": True,
            "blocking_reasons": [], "warnings": ["REQUIRED_METADATA_NOT_BOUND"] if legacy else [],
            "existing_contract_id": expected_contract_id,
            "target_contract_identity": target_contract_identity,
            "maturity_cutoff": maturity_cutoff, "row_count": len(source_rows),
            "group_count": len(grouped), "label_type": "ordinal_integer",
            "label_population_checksum": canonical_hash(sorted(labels.items())),
            "grouped_date_checksum": canonical_hash([
                (decision, [row["row_id"] for row in sorted(group, key=lambda row: (row["asset_id"], row["row_id"]))])
                for decision, group in grouped.items()
            ]),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return logical
    except RankingLabelError as exc:
        return _blocked(COMPATIBILITY_CONTRACT, exc)


def pairwise_ranking_dataset(
    pair_result: Mapping[str, Any],
    *,
    split_identity: str,
    feature_difference_representation_identity: str,
    pair_weight: float = 1.0,
) -> dict[str, Any]:
    if not pair_result.get("valid"):
        return _blocked(PAIRWISE_DATASET_CONTRACT, RankingLabelError("INVALID_INPUT", "PAIR_RESULT_INVALID"))
    if not math.isfinite(pair_weight) or pair_weight <= 0:
        return _blocked(PAIRWISE_DATASET_CONTRACT, RankingLabelError("INVALID_INPUT", "PAIR_WEIGHT_INVALID"))
    rows = [{
        "pair_id": pair["pair_id"], "decision_date": pair["decision_date"],
        "winner_row_id": pair["winner_row_id"], "loser_row_id": pair["loser_row_id"],
        "winner_asset_id": pair["winner_asset_id"], "loser_asset_id": pair["loser_asset_id"],
        "return_margin": pair["realised_return_difference"], "pair_weight": float(pair_weight),
        "pair_checksum": pair["pair_checksum"],
    } for pair in pair_result["pairs"]]
    logical = {
        "contract_version": PAIRWISE_DATASET_CONTRACT, "status": "READY", "valid": True,
        "blocking_reasons": [], "warnings": [], "pairs": rows,
        "ordered_pair_ids": [row["pair_id"] for row in rows], "pair_count": len(rows),
        "split_identity": split_identity,
        "feature_difference_representation_identity": feature_difference_representation_identity,
        "pair_population_checksum": canonical_hash(rows),
        "target_contract_checksum": canonical_hash({"identity": pair_result["target_contract_identity"]}),
    }
    logical["pairwise_dataset_checksum"] = canonical_hash(logical)
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def verify_percentile_result(rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> dict[str, Any]:
    expected = continuous_percentile_relevance(
        rows, target_contract_identity=result["target_contract_identity"],
        maturity_cutoff=result["maturity_cutoff"], minimum_group_size=result["minimum_group_size"],
    )
    return _verification(result, expected, "percentile")


def verify_pairwise_result(rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> dict[str, Any]:
    expected = pairwise_return_margin(
        rows, target_contract_identity=result["target_contract_identity"],
        maturity_cutoff=result["maturity_cutoff"],
        minimum_return_margin=result["minimum_return_margin"],
        maximum_pairs_per_date=result["maximum_pairs_per_date"],
    )
    return _verification(result, expected, "pairwise")


def verify_grouped_dataset(rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> dict[str, Any]:
    expected = grouped_ranking_dataset(
        rows, label_type=result["label_type"],
        feature_schema_identity=result["feature_schema_identity"],
        target_contract_identity=result["target_contract_identity"],
        ranking_label_contract_identity=result["ranking_label_contract_identity"],
        split_identity=result["split_identity"], allowed_cutoff=result["allowed_cutoff"],
        minimum_group_size=result["minimum_group_size"],
    )
    reasons = []
    independently_recomputed = (
        "rows", "groups", "group_size_vector", "feature_schema_checksum",
        "target_contract_checksum", "ranking_label_contract_checksum",
        "ordered_row_population_checksum", "ordered_label_checksum",
        "decision_date_population_checksum", "group_size_vector_checksum",
        "split_checksum", "dataset_checksum", "logical_result_checksum",
    )
    for field in independently_recomputed:
        if result.get(field) != expected.get(field):
            reasons.append(f"{field.upper()}_MISMATCH")
    return {"contract_version": "grouped_ranking_dataset_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def verify_framework_exports(dataset: Mapping[str, Any], lightgbm: Mapping[str, Any], xgboost: Mapping[str, Any]) -> dict[str, Any]:
    expected_lgb = lightgbm_group_export(dataset)
    expected_xgb = xgboost_qid_export(dataset)
    reasons = []
    if lightgbm != expected_lgb:
        reasons.append("LIGHTGBM_EXPORT_MISMATCH")
    if xgboost != expected_xgb:
        reasons.append("XGBOOST_EXPORT_MISMATCH")
    decoded_sizes = []
    qid = xgboost.get("qid_vector", [])
    if qid:
        current, count = qid[0], 0
        for value in qid:
            if value != current:
                decoded_sizes.append(count); current, count = value, 0
            count += 1
        decoded_sizes.append(count)
    if decoded_sizes != lightgbm.get("group_size_vector", []):
        reasons.append("QUERY_STRUCTURE_MISMATCH")
    return {"contract_version": "grouped_ranking_export_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def _label_source_rows(rows, cutoff):
    normalised, seen_rows, seen_asset_dates = [], set(), set()
    for raw in rows:
        row_id, asset = str(raw.get("row_id") or ""), str(raw.get("asset_id") or "")
        decision = str(raw.get("decision_date") or "")
        target = raw.get("realised_target")
        maturity = str(raw.get("target_maturity_timestamp") or "")
        if not row_id or not asset or not decision:
            raise RankingLabelError("INVALID_INPUT", "LABEL_ROW_IDENTITY_MISSING")
        if row_id in seen_rows or (decision, asset) in seen_asset_dates:
            raise RankingLabelError("INVALID_INPUT", "DUPLICATE_ROW_OR_ASSET_DATE")
        if target is None:
            raise RankingLabelError("MISSING_TARGET", "REALISED_TARGET_MISSING")
        target = float(target)
        if not math.isfinite(target):
            raise RankingLabelError("INVALID_INPUT", "REALISED_TARGET_NON_FINITE")
        if _time(maturity) > _time(cutoff):
            raise RankingLabelError("IMMATURE_TARGET", f"TARGET_IMMATURE:{row_id}")
        seen_rows.add(row_id); seen_asset_dates.add((decision, asset))
        normalised.append({
            "row_id": row_id, "asset_id": asset, "decision_date": decision,
            "realised_target": target, "target_maturity_timestamp": maturity,
        })
    ordered = sorted(normalised, key=lambda row: (row["decision_date"], row["asset_id"], row["row_id"]))
    if normalised != ordered:
        raise RankingLabelError("INVALID_INPUT", "LABEL_ROWS_NOT_DETERMINISTICALLY_ORDERED")
    return normalised


def _legacy_source_rows(rows):
    normalised = []
    for row in rows:
        normalised.append({
            "row_id": str(row["row_id"]), "asset_id": str(row.get("asset_id") or row["row_id"]),
            "decision_date": str(row.get("decision_date") or row.get("decision_timestamp")),
            "realised_target": float(row.get("realised_target", row.get("actual_forward_return_10d"))),
            "target_maturity_timestamp": str(row.get("target_maturity_timestamp") or ""),
        })
    return sorted(normalised, key=lambda row: (row["decision_date"], row["asset_id"], row["row_id"]))


def _groups(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["decision_date"], []).append(row)
    return {decision: grouped[decision] for decision in sorted(grouped)}


def _dataset_groups(rows, minimum):
    groups, cursor = [], 0
    for decision in sorted({row["decision_date"] for row in rows}):
        members = [row for row in rows if row["decision_date"] == decision]
        if len(members) < minimum:
            raise RankingLabelError("INSUFFICIENT_GROUP_SIZE", f"GROUP_TOO_SMALL:{decision}")
        groups.append({
            "group_id": f"decision_date:{decision}", "decision_date": decision,
            "start_position": cursor, "end_position_exclusive": cursor + len(members),
            "group_size": len(members),
            "row_population_checksum": canonical_hash([row["row_id"] for row in members]),
        })
        cursor += len(members)
    if cursor != len(rows):
        raise RankingLabelError("NONCONTIGUOUS_GROUP", "GROUP_COVERAGE_MISMATCH")
    return groups


def _validate_label(value, label_type):
    if label_type == "quintile_integer":
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "QUINTILE_LABEL_INVALID")
        return value
    if label_type == "decile_integer":
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 9:
            raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "DECILE_LABEL_INVALID")
        return value
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RankingLabelError("LABEL_CONTRACT_MISMATCH", "PERCENTILE_LABEL_INVALID")
    return value


def _ready_dataset(dataset):
    if not dataset.get("valid") or dataset.get("contract_version") != GROUPED_DATASET_CONTRACT:
        raise RankingLabelError("INVALID_INPUT", "GROUPED_DATASET_INVALID")


def _verification(actual, expected, owner):
    fields = (
        "status", "labels" if owner == "percentile" else "pairs",
        "label_population_checksum" if owner == "percentile" else "pair_population_checksum",
        "grouped_date_checksum", "source_population_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if actual.get(field) != expected.get(field)]
    return {"contract_version": f"{owner}_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def _blocked(contract, error):
    logical = {
        "contract_version": contract,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
        "row_count": 0, "group_count": 0, "pair_count": 0, "feature_count": 0,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RankingLabelError("INVALID_INPUT", "TIMESTAMP_INVALID") from exc


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
