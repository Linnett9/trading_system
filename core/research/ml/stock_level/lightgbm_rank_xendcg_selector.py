from __future__ import annotations

import json
import math
import statistics
import time
import warnings
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from core.research.ml.lightgbm_ranking_preflight import deterministic_ranker_configuration
from core.research.ml.ranking_labels import GROUPED_DATASET_CONTRACT, canonical_hash, canonical_json


INPUT_CONTRACT = "lightgbm_rank_xendcg_input_v1"
CONFIG_CONTRACT = "lightgbm_rank_xendcg_configuration_v1"
MODEL_CONTRACT = "lightgbm_rank_xendcg_model_v1"
PREDICTION_CONTRACT = "lightgbm_rank_xendcg_prediction_v1"
RESULT_CONTRACT = "lightgbm_rank_xendcg_result_v1"
VERIFICATION_CONTRACT = "lightgbm_rank_xendcg_verification_v1"
SUPPORTED_LABEL_CONTRACTS = {
    "within_date_quintile_relevance_v1",
    "within_date_decile_relevance_v1",
}
STATUSES = {
    "READY", "DEPENDENCY_UNAVAILABLE", "DEPENDENCY_MISMATCH", "OBJECTIVE_UNAVAILABLE",
    "INVALID_INPUT", "UNSUPPORTED_LABEL_CONTRACT", "GROUP_STRUCTURE_INVALID",
    "FEATURE_SCHEMA_MISMATCH", "TARGET_CONTRACT_MISMATCH", "SPLIT_OVERLAP",
    "IMMATURE_TARGET", "INSUFFICIENT_DATA", "NONFINITE_PREDICTION",
    "INCOMPLETE_PREDICTION_POPULATION", "NONDETERMINISTIC_RESULT",
    "SERIALISATION_FAILURE", "MODEL_RELOAD_FAILURE", "NUMERICAL_FAILURE",
}


class SelectorError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def fixed_rank_xendcg_configuration(*, num_threads: int = 1) -> dict[str, Any]:
    base = deterministic_ranker_configuration(objective="rank_xendcg", num_threads=num_threads)
    base.update({
        "metric": "ndcg",
        "eval_at": [1, 3, 5],
        "n_estimators": 24,
        "learning_rate": 0.08,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 2,
        "reg_alpha": 0.05,
        "reg_lambda": 0.20,
    })
    return {"contract_version": CONFIG_CONTRACT, "parameters": base}


def validate_rank_xendcg_input(
    dataset: Mapping[str, Any],
    *,
    training_cutoff: str,
    source_commit: str | None = None,
    registered_integer_label_contracts: Sequence[str] = (),
) -> dict[str, Any]:
    try:
        if dataset.get("contract_version") != GROUPED_DATASET_CONTRACT or not dataset.get("valid"):
            raise SelectorError("INVALID_INPUT", "GROUPED_DATASET_NOT_READY")
        label_type = dataset.get("label_type")
        label_contract = str(dataset.get("ranking_label_contract_identity") or "")
        allowed = SUPPORTED_LABEL_CONTRACTS | {str(value) for value in registered_integer_label_contracts}
        if label_type not in {"quintile_integer", "decile_integer"} or label_contract not in allowed:
            raise SelectorError("UNSUPPORTED_LABEL_CONTRACT", "INTEGER_REGISTERED_RELEVANCE_REQUIRED")
        rows = list(dataset.get("rows") or ())
        groups = list(dataset.get("groups") or ())
        if not rows or not groups:
            raise SelectorError("INSUFFICIENT_DATA", "ROWS_OR_GROUPS_EMPTY")
        expected_order = sorted(rows, key=lambda row: (row["decision_date"], row["asset_id"], row["row_id"]))
        if rows != expected_order:
            raise SelectorError("GROUP_STRUCTURE_INVALID", "ROWS_NOT_CANONICALLY_ORDERED")
        if dataset.get("group_size_vector") != [group.get("group_size") for group in groups]:
            raise SelectorError("GROUP_STRUCTURE_INVALID", "GROUP_SIZE_VECTOR_MISMATCH")
        if sum(dataset["group_size_vector"]) != len(rows) or any(size <= 0 for size in dataset["group_size_vector"]):
            raise SelectorError("GROUP_STRUCTURE_INVALID", "GROUP_SIZE_SUM_OR_EMPTY_GROUP")
        cursor = 0
        train_rows, validation_rows, train_groups, validation_groups = [], [], [], []
        for group in groups:
            size = int(group["group_size"])
            members = rows[cursor:cursor + size]
            if (
                group.get("start_position") != cursor
                or group.get("end_position_exclusive") != cursor + size
                or len(members) != size
                or len({row["decision_date"] for row in members}) != 1
                or members[0]["decision_date"] != group["decision_date"]
            ):
                raise SelectorError("GROUP_STRUCTURE_INVALID", "NONCONTIGUOUS_OR_MIXED_GROUP")
            roles = {row["split_role"] for row in members}
            if len(roles) != 1:
                raise SelectorError("SPLIT_OVERLAP", "GROUP_MIXES_SPLIT_ROLES")
            destination_rows, destination_groups = (
                (train_rows, train_groups) if roles == {"TRAINING"} else
                (validation_rows, validation_groups) if roles == {"VALIDATION"} else (None, None)
            )
            if destination_rows is None:
                raise SelectorError("INVALID_INPUT", "SPLIT_ROLE_INVALID")
            destination_rows.extend(members)
            destination_groups.append(size)
            cursor += size
        if cursor != len(rows):
            raise SelectorError("GROUP_STRUCTURE_INVALID", "GROUP_COVERAGE_INCOMPLETE")
        if not train_rows or not validation_rows or not train_groups or not validation_groups:
            raise SelectorError("INSUFFICIENT_DATA", "TRAINING_OR_VALIDATION_POPULATION_EMPTY")
        if {row["row_id"] for row in train_rows} & {row["row_id"] for row in validation_rows}:
            raise SelectorError("SPLIT_OVERLAP", "ROW_ID_SPLIT_OVERLAP")
        feature_names = list(dataset.get("feature_names") or ())
        if not feature_names:
            raise SelectorError("FEATURE_SCHEMA_MISMATCH", "FEATURE_SCHEMA_EMPTY")
        for row in rows:
            if row.get("feature_names") != feature_names or len(row.get("feature_values") or ()) != len(feature_names):
                raise SelectorError("FEATURE_SCHEMA_MISMATCH", "FEATURE_ORDER_OR_DIMENSION_MISMATCH")
            if not all(math.isfinite(float(value)) for value in row["feature_values"]):
                raise SelectorError("NUMERICAL_FAILURE", "FEATURE_NONFINITE")
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise SelectorError("UNSUPPORTED_LABEL_CONTRACT", "NONNEGATIVE_INTEGER_LABEL_REQUIRED")
        cutoff = _time(training_cutoff)
        training_maturities = [_time(row["target_maturity_timestamp"]) for row in train_rows]
        if any(value > cutoff for value in training_maturities):
            raise SelectorError("IMMATURE_TARGET", "TRAINING_LABEL_MATURES_AFTER_CUTOFF")
        if dataset.get("ordered_row_population_checksum") != canonical_hash([row["row_id"] for row in rows]):
            raise SelectorError("INVALID_INPUT", "ROW_POPULATION_CHECKSUM_MISMATCH")
        if dataset.get("ordered_label_checksum") != canonical_hash([row["label"] for row in rows]):
            raise SelectorError("INVALID_INPUT", "LABEL_CHECKSUM_MISMATCH")
        if dataset.get("group_size_vector_checksum") != canonical_hash(dataset["group_size_vector"]):
            raise SelectorError("GROUP_STRUCTURE_INVALID", "GROUP_CHECKSUM_MISMATCH")
        if dataset.get("feature_schema_checksum") != canonical_hash({
            "identity": dataset.get("feature_schema_identity"), "features": feature_names,
        }):
            raise SelectorError("FEATURE_SCHEMA_MISMATCH", "FEATURE_SCHEMA_CHECKSUM_MISMATCH")
        if dataset.get("target_contract_checksum") != canonical_hash({"identity": dataset.get("target_contract_identity")}):
            raise SelectorError("TARGET_CONTRACT_MISMATCH", "TARGET_CONTRACT_CHECKSUM_MISMATCH")
        dataset_logical = {
            key: value for key, value in dataset.items()
            if key not in {"creation_metadata", "dataset_checksum", "logical_result_checksum"}
        }
        if dataset.get("dataset_checksum") != canonical_hash(dataset_logical):
            raise SelectorError("INVALID_INPUT", "GROUPED_DATASET_CHECKSUM_MISMATCH")
        logical = {
            "contract_version": INPUT_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "grouped_dataset_contract": dataset["contract_version"],
            "grouped_dataset_checksum": dataset["dataset_checksum"],
            "ranking_label_contract_identity": label_contract,
            "ranking_label_contract_checksum": dataset["ranking_label_contract_checksum"],
            "target_contract_identity": dataset["target_contract_identity"],
            "target_contract_checksum": dataset["target_contract_checksum"],
            "feature_schema_identity": dataset["feature_schema_identity"],
            "feature_schema_checksum": dataset["feature_schema_checksum"],
            "split_identity": dataset["split_identity"],
            "ordered_row_population_checksum": dataset["ordered_row_population_checksum"],
            "ordered_label_checksum": dataset["ordered_label_checksum"],
            "group_size_vector_checksum": dataset["group_size_vector_checksum"],
            "training_population_checksum": canonical_hash([row["row_id"] for row in train_rows]),
            "validation_population_checksum": canonical_hash([row["row_id"] for row in validation_rows]),
            "training_count": len(train_rows), "validation_count": len(validation_rows),
            "training_group_count": len(train_groups), "validation_group_count": len(validation_groups),
            "training_group_sizes": train_groups, "validation_group_sizes": validation_groups,
            "training_cutoff": training_cutoff,
            "maximum_training_label_maturity_timestamp": max(row["target_maturity_timestamp"] for row in train_rows),
            "feature_availability_cutoff": max(row["feature_availability_timestamp"] for row in rows),
            "source_commit": source_commit,
        }
        logical["input_checksum"] = canonical_hash(logical)
        return logical
    except SelectorError as exc:
        return _blocked(INPUT_CONTRACT, exc)


def fit_synthetic_rank_xendcg_selector(
    dataset: Mapping[str, Any],
    *,
    training_cutoff: str,
    num_threads: int = 1,
    serialisation_directory: str | Path | None = None,
    source_commit: str | None = None,
    registered_integer_label_contracts: Sequence[str] = (),
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if lgb.__version__ != "4.6.0":
            raise SelectorError("DEPENDENCY_MISMATCH", f"LIGHTGBM_VERSION:{lgb.__version__}")
        input_contract = validate_rank_xendcg_input(
            dataset, training_cutoff=training_cutoff, source_commit=source_commit,
            registered_integer_label_contracts=registered_integer_label_contracts,
        )
        if not input_contract.get("valid"):
            raise SelectorError(input_contract["status"], input_contract["blocking_reasons"][0])
        configuration = fixed_rank_xendcg_configuration(num_threads=num_threads)
        parameters = configuration["parameters"]
        train_rows = [row for row in dataset["rows"] if row["split_role"] == "TRAINING"]
        validation_rows = [row for row in dataset["rows"] if row["split_role"] == "VALIDATION"]
        x_train = np.asarray([row["feature_values"] for row in train_rows], dtype=float)
        y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
        x_validation = np.asarray([row["feature_values"] for row in validation_rows], dtype=float)
        y_validation = np.asarray([row["label"] for row in validation_rows], dtype=int)
        fit_started = time.perf_counter()
        first = _fit(parameters, x_train, y_train, input_contract["training_group_sizes"])
        first_fit_seconds = time.perf_counter() - fit_started
        second = _fit(parameters, x_train, y_train, input_contract["training_group_sizes"])
        prediction_started = time.perf_counter()
        first_scores = _predict(first, x_validation)
        second_scores = _predict(second, x_validation)
        prediction_seconds = time.perf_counter() - prediction_started
        if not np.isfinite(first_scores).all():
            raise SelectorError("NONFINITE_PREDICTION", "VALIDATION_SCORE_NONFINITE")
        if len(first_scores) != len(validation_rows):
            raise SelectorError("INCOMPLETE_PREDICTION_POPULATION", "VALIDATION_SCORE_COUNT_MISMATCH")
        if not np.allclose(first_scores, second_scores, rtol=0.0, atol=1e-12):
            raise SelectorError("NONDETERMINISTIC_RESULT", "REPEATED_PREDICTIONS_DIFFER")
        first_importance = _feature_importance(first, dataset["feature_names"])
        second_importance = _feature_importance(second, dataset["feature_names"])
        if first_importance != second_importance:
            raise SelectorError("NONDETERMINISTIC_RESULT", "REPEATED_IMPORTANCE_DIFFERS")
        model_text = first.booster_.model_to_string()
        second_model_text = second.booster_.model_to_string()
        byte_deterministic = model_text == second_model_text
        serialisation_state, reload_state, model_path = False, False, None
        if serialisation_directory is not None:
            directory = Path(serialisation_directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "lightgbm_rank_xendcg_selector.txt"
            try:
                first.booster_.save_model(str(path))
                saved_text = path.read_text(encoding="utf-8")
            except Exception as exc:
                raise SelectorError("SERIALISATION_FAILURE", "MODEL_SERIALISATION_FAILED") from exc
            if canonical_hash(saved_text) != canonical_hash(model_text):
                raise SelectorError("SERIALISATION_FAILURE", "SAVED_MODEL_CHECKSUM_MISMATCH")
            serialisation_state = True
            model_path = str(path)
            try:
                reloaded_scores = np.asarray(lgb.Booster(model_file=str(path)).predict(x_validation), dtype=float)
            except Exception as exc:
                raise SelectorError("MODEL_RELOAD_FAILURE", "MODEL_RELOAD_FAILED") from exc
            reload_state = bool(np.allclose(first_scores, reloaded_scores, rtol=0.0, atol=1e-12))
            if not reload_state:
                raise SelectorError("MODEL_RELOAD_FAILURE", "RELOAD_PREDICTIONS_DIFFER")
        model_identity = {
            "contract_version": MODEL_CONTRACT,
            "model_id": "lightgbm_rank_xendcg_synthetic_selector_v1",
            "objective": "rank_xendcg",
            "dependency_identity": "lightgbm==4.6.0",
            "parameters": parameters,
            "thread_policy": f"bounded_threads:{num_threads}",
            "seed_policy": "fixed_1729_all_supported_seeds",
            "feature_schema_checksum": dataset["feature_schema_checksum"],
            "target_contract_checksum": dataset["target_contract_checksum"],
            "ranking_label_checksum": dataset["ranking_label_contract_checksum"],
            "grouped_dataset_checksum": dataset["dataset_checksum"],
            "group_size_vector_checksum": dataset["group_size_vector_checksum"],
            "training_population_checksum": input_contract["training_population_checksum"],
            "validation_population_checksum": input_contract["validation_population_checksum"],
            "training_cutoff": training_cutoff,
            "maximum_label_maturity_timestamp": input_contract["maximum_training_label_maturity_timestamp"],
            "source_commit": source_commit,
            "fitted_booster_identity": canonical_hash(first.booster_.dump_model()),
            "serialised_model_checksum": canonical_hash(model_text),
        }
        model_identity["logical_model_checksum"] = canonical_hash(model_identity)
        prediction_rows = _prediction_rows(
            validation_rows, first_scores, model_identity["logical_model_checksum"], dataset, input_contract,
        )
        prediction_checksum = canonical_hash(prediction_rows)
        for row in prediction_rows:
            row["prediction_population_checksum"] = prediction_checksum
        prediction_contract = {
            "contract_version": PREDICTION_CONTRACT,
            "rows": prediction_rows,
            "row_count": len(prediction_rows),
            "prediction_population_checksum": prediction_checksum,
        }
        ranking_diagnostics = {
            "training": _ranking_diagnostics(train_rows, _predict(first, x_train), input_contract["training_group_sizes"]),
            "validation": _ranking_diagnostics(validation_rows, first_scores, input_contract["validation_group_sizes"]),
        }
        tree_diagnostics = _tree_diagnostics(first, parameters, model_text)
        diagnostics = {
            "ranking": ranking_diagnostics,
            "feature_importance": first_importance,
            "tree_model": tree_diagnostics,
            "repeatability": {
                "prediction_level": True, "rank_level": _rank_signature(validation_rows, first_scores) == _rank_signature(validation_rows, second_scores),
                "feature_importance": True, "logical_model": canonical_hash(first.booster_.dump_model()) == canonical_hash(second.booster_.dump_model()),
                "byte_level_serialisation": byte_deterministic,
            },
            "serialisation": {"saved": serialisation_state, "reloaded": reload_state},
            "synthetic_only": True, "promotion_evidence": False,
        }
        logical = {
            "contract_version": RESULT_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": ["SYNTHETIC_RESULTS_ARE_NOT_PROMOTION_EVIDENCE"],
            "dependency_identity": "lightgbm==4.6.0", "objective": "rank_xendcg",
            "parameter_identity": canonical_hash(parameters),
            "thread_identity": f"bounded_threads:{num_threads}",
            "label_contract": input_contract["ranking_label_contract_identity"],
            "feature_count": len(dataset["feature_names"]),
            "training_count": len(train_rows), "validation_count": len(validation_rows),
            "training_group_count": len(input_contract["training_group_sizes"]),
            "validation_group_count": len(input_contract["validation_group_sizes"]),
            "population_checksums": {
                "training": input_contract["training_population_checksum"],
                "validation": input_contract["validation_population_checksum"],
                "group_sizes": dataset["group_size_vector_checksum"],
                "dataset": dataset["dataset_checksum"],
            },
            "input_contract": input_contract, "model_contract": model_identity,
            "prediction_contract": prediction_contract, "diagnostics": diagnostics,
            "model_checksum": model_identity["logical_model_checksum"],
            "prediction_checksum": prediction_checksum,
            "diagnostics_checksum": canonical_hash(diagnostics),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {
            **logical,
            "runtime_metadata": {
                "training_duration_seconds": first_fit_seconds,
                "prediction_duration_seconds": prediction_seconds,
                "total_duration_seconds": time.perf_counter() - started,
                "serialised_model_path": model_path,
            },
        }
    except SelectorError as exc:
        return _blocked(RESULT_CONTRACT, exc)


def compare_integer_label_contracts(
    quintile_dataset: Mapping[str, Any],
    decile_dataset: Mapping[str, Any],
    *,
    training_cutoff: str,
    num_threads: int = 1,
) -> dict[str, Any]:
    quintile = fit_synthetic_rank_xendcg_selector(
        quintile_dataset, training_cutoff=training_cutoff, num_threads=num_threads,
    )
    decile = fit_synthetic_rank_xendcg_selector(
        decile_dataset, training_cutoff=training_cutoff, num_threads=num_threads,
    )
    if not quintile.get("valid") or not decile.get("valid"):
        return _blocked(RESULT_CONTRACT, SelectorError("INVALID_INPUT", "LABEL_COMPARISON_FIT_FAILED"))
    q_scores = [row["raw_score"] for row in quintile["prediction_contract"]["rows"]]
    d_scores = [row["raw_score"] for row in decile["prediction_contract"]["rows"]]
    q_top = {row["row_id"] for row in sorted(quintile["prediction_contract"]["rows"], key=lambda row: row["within_date_rank"])[:3]}
    d_top = {row["row_id"] for row in sorted(decile["prediction_contract"]["rows"], key=lambda row: row["within_date_rank"])[:3]}
    q_gain = [row["normalised_gain_share"] for row in quintile["diagnostics"]["feature_importance"]["features"]]
    d_gain = [row["normalised_gain_share"] for row in decile["diagnostics"]["feature_importance"]["features"]]
    logical = {
        "contract_version": "lightgbm_rank_xendcg_label_comparison_v1",
        "status": "READY", "valid": True, "blocking_reasons": [], "warnings": ["NOT_A_MODEL_SELECTION_RESULT"],
        "quintile": {
            "label_distribution": quintile["diagnostics"]["ranking"]["validation"]["label_distribution"],
            "ndcg": quintile["diagnostics"]["ranking"]["validation"]["ndcg"],
            "score_standard_deviation": quintile["diagnostics"]["ranking"]["validation"]["score_standard_deviation"],
        },
        "decile": {
            "label_distribution": decile["diagnostics"]["ranking"]["validation"]["label_distribution"],
            "ndcg": decile["diagnostics"]["ranking"]["validation"]["ndcg"],
            "score_standard_deviation": decile["diagnostics"]["ranking"]["validation"]["score_standard_deviation"],
        },
        "score_rank_correlation": _spearman(q_scores, d_scores),
        "top_3_overlap": len(q_top & d_top) / max(1, len(q_top | d_top)),
        "feature_importance_similarity": _cosine(q_gain, d_gain),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def verify_rank_xendcg_result(
    dataset: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    serialisation_directory: str | Path | None = None,
) -> dict[str, Any]:
    expected = fit_synthetic_rank_xendcg_selector(
        dataset,
        training_cutoff=result["input_contract"]["training_cutoff"],
        num_threads=result["model_contract"]["parameters"]["n_jobs"],
        serialisation_directory=serialisation_directory,
        source_commit=result["input_contract"].get("source_commit"),
    )
    fields = (
        "status", "dependency_identity", "objective", "parameter_identity", "thread_identity",
        "label_contract", "population_checksums", "input_contract", "model_contract",
        "prediction_contract", "diagnostics", "model_checksum", "prediction_checksum",
        "diagnostics_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": VERIFICATION_CONTRACT, "valid": not reasons, "blocking_reasons": reasons}


def _fit(parameters, matrix, labels, groups):
    try:
        estimator_parameters = dict(parameters)
        eval_at = estimator_parameters.pop("eval_at")
        return lgb.LGBMRanker(**estimator_parameters).fit(matrix, labels, group=groups, eval_at=eval_at)
    except Exception as exc:
        raise SelectorError("OBJECTIVE_UNAVAILABLE", "RANK_XENDCG_FIT_FAILED") from exc


def _predict(model, matrix):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
        return np.asarray(model.predict(matrix), dtype=float)


def _prediction_rows(rows, scores, model_checksum, dataset, input_contract):
    output = []
    by_date = {}
    for row, score in zip(rows, scores):
        by_date.setdefault(row["decision_date"], []).append((row, float(score)))
    for decision in sorted(by_date):
        members = sorted(by_date[decision], key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]))
        count = len(members)
        for index, (row, score) in enumerate(members):
            output.append({
                "row_id": row["row_id"], "asset_id": row["asset_id"], "decision_date": decision,
                "raw_score": score, "within_date_rank": index + 1,
                "percentile_rank": 1.0 if count == 1 else 1.0 - index / (count - 1),
                "model_checksum": model_checksum, "grouped_dataset_checksum": dataset["dataset_checksum"],
                "feature_schema_checksum": dataset["feature_schema_checksum"],
                "target_contract_checksum": dataset["target_contract_checksum"],
                "label_contract_checksum": dataset["ranking_label_contract_checksum"],
                "split_identity": dataset["split_identity"], "training_cutoff": input_contract["training_cutoff"],
                "maximum_training_label_maturity": input_contract["maximum_training_label_maturity_timestamp"],
            })
    return output


def _ranking_diagnostics(rows, scores, group_sizes):
    labels = [int(row["label"]) for row in rows]
    sizes = sorted(group_sizes)
    tied_scores = largest_tie = rank_diversity = 0
    ndcg_values = {1: [], 3: [], 5: []}
    cursor = 0
    top_overlap = []
    for size in group_sizes:
        group_scores = list(map(float, scores[cursor:cursor + size]))
        group_labels = labels[cursor:cursor + size]
        counts = {}
        for score in group_scores:
            counts[score] = counts.get(score, 0) + 1
        ties = [count for count in counts.values() if count > 1]
        tied_scores += sum(ties)
        largest_tie = max([largest_tie, *ties])
        rank_diversity += len(counts)
        predicted = sorted(range(size), key=lambda index: (-group_scores[index], rows[cursor + index]["asset_id"]))
        realised = sorted(range(size), key=lambda index: (-group_labels[index], rows[cursor + index]["asset_id"]))
        for cutoff in ndcg_values:
            ndcg_values[cutoff].append(_ndcg(group_labels, predicted, cutoff))
        k = min(3, size)
        top_overlap.append(len(set(predicted[:k]) & set(realised[:k])) / k)
        cursor += size
    score_list = list(map(float, scores))
    distribution = {str(value): labels.count(value) for value in sorted(set(labels))}
    per_group_ndcg = ndcg_values[3]
    return {
        "row_count": len(rows), "query_group_count": len(group_sizes),
        "minimum_group_size": min(sizes), "median_group_size": statistics.median(sizes),
        "maximum_group_size": max(sizes), "feature_count": len(rows[0]["feature_values"]),
        "label_distribution": distribution, "score_mean": statistics.fmean(score_list),
        "score_standard_deviation": statistics.pstdev(score_list), "score_minimum": min(score_list),
        "score_maximum": max(score_list), "finite_scores": all(math.isfinite(value) for value in score_list),
        "tied_score_count": tied_scores, "largest_tied_score_group": largest_tie,
        "within_date_rank_diversity": rank_diversity,
        "ndcg": {str(k): statistics.fmean(values) for k, values in ndcg_values.items()},
        "spearman_rank_ic": _spearman(score_list, labels),
        "top_3_overlap": statistics.fmean(top_overlap),
        "group_ndcg_at_3_standard_deviation": statistics.pstdev(per_group_ndcg),
    }


def _feature_importance(model, feature_names):
    split = model.booster_.feature_importance("split").astype(int).tolist()
    gain = model.booster_.feature_importance("gain").astype(float).tolist()
    total = sum(gain)
    order = sorted(range(len(feature_names)), key=lambda index: (-gain[index], feature_names[index]))
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    features = [{
        "feature_id": feature_names[index], "split_importance": split[index],
        "gain_importance": gain[index], "normalised_gain_share": gain[index] / total if total else 0.0,
        "rank_by_gain": ranks[index], "zero_importance": split[index] == 0 and gain[index] == 0,
    } for index in range(len(feature_names))]
    shares = sorted((row["normalised_gain_share"] for row in features), reverse=True)
    return {
        "features": features, "used_feature_count": sum(not row["zero_importance"] for row in features),
        "zero_importance_feature_count": sum(row["zero_importance"] for row in features),
        "top_feature_concentration": shares[0] if shares else 0.0,
        "cumulative_top_5_gain_share": sum(shares[:5]),
        "interpretation": "TREE_IMPORTANCE_IS_NOT_CAUSAL_ATTRIBUTION",
    }


def _tree_diagnostics(model, parameters, model_text):
    dump = model.booster_.dump_model()
    depths, leaves = [], []
    for tree in dump.get("tree_info", []):
        leaves.append(int(tree.get("num_leaves", 0)))
        depths.append(_tree_depth(tree.get("tree_structure", {})))
    return {
        "number_of_trees": len(leaves), "best_iteration": int(model.best_iteration_ or 0),
        "configured_estimator_count": parameters["n_estimators"], "fitted_estimator_count": len(leaves),
        "tree_depth_distribution": depths, "leaf_count_distribution": leaves,
        "model_text_checksum": canonical_hash(model_text),
        "model_byte_size": len(model_text.encode("utf-8")),
    }


def _tree_depth(node):
    if "left_child" not in node and "right_child" not in node:
        return 0
    return 1 + max(_tree_depth(node["left_child"]), _tree_depth(node["right_child"]))


def _rank_signature(rows, scores):
    signature = []
    by_date = {}
    for row, score in zip(rows, scores):
        by_date.setdefault(row["decision_date"], []).append((row, float(score)))
    for decision in sorted(by_date):
        ordered = sorted(by_date[decision], key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]))
        signature.extend((row["row_id"], rank) for rank, (row, _) in enumerate(ordered, 1))
    return signature


def _ndcg(labels, predicted_order, cutoff):
    k = min(cutoff, len(labels))
    dcg = sum((2 ** labels[index] - 1) / math.log2(position + 2) for position, index in enumerate(predicted_order[:k]))
    ideal = sorted(labels, reverse=True)
    idcg = sum((2 ** value - 1) / math.log2(position + 2) for position, value in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 1.0


def _spearman(left, right):
    if len(left) < 2 or statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
        return 0.0
    return float(np.corrcoef(_average_ranks(left), _average_ranks(right))[0, 1])


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _cosine(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SelectorError("INVALID_INPUT", "TIMESTAMP_INVALID") from exc


def _blocked(contract, error):
    logical = {
        "contract_version": contract,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical
