from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from core.research.ml.ranking_labels import canonical_hash
from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
    _cosine,
    _feature_importance,
    _predict,
    _prediction_score_diagnostics,
    _rank_signature,
    _ranking_diagnostics,
    _spearman,
    _tree_diagnostics,
    fit_synthetic_rank_xendcg_selector,
    fixed_rank_xendcg_configuration,
    validate_rank_xendcg_input,
)


INPUT_CONTRACT = "lightgbm_lambdarank_input_v1"
GAIN_CONTRACT = "lightgbm_lambdarank_label_gain_v1"
CONFIG_CONTRACT = "lightgbm_lambdarank_configuration_v1"
MODEL_CONTRACT = "lightgbm_lambdarank_model_v1"
PREDICTION_CONTRACT = "lightgbm_lambdarank_prediction_v1"
RESULT_CONTRACT = "lightgbm_lambdarank_result_v1"
VERIFICATION_CONTRACT = "lightgbm_lambdarank_verification_v1"
COMPARISON_CONTRACT = "lightgbm_lambdarank_rank_xendcg_comparison_v1"
STATUSES = {
    "READY", "DEPENDENCY_UNAVAILABLE", "DEPENDENCY_MISMATCH", "OBJECTIVE_UNAVAILABLE",
    "INVALID_INPUT", "UNSUPPORTED_LABEL_CONTRACT", "LABEL_GAIN_MISMATCH",
    "GROUP_STRUCTURE_INVALID", "FEATURE_SCHEMA_MISMATCH", "TARGET_CONTRACT_MISMATCH",
    "SPLIT_OVERLAP", "IMMATURE_TARGET", "INSUFFICIENT_DATA", "NONFINITE_PREDICTION",
    "INCOMPLETE_PREDICTION_POPULATION", "NONDETERMINISTIC_RESULT",
    "SERIALISATION_FAILURE", "MODEL_RELOAD_FAILURE", "COMPARISON_POPULATION_MISMATCH",
    "NUMERICAL_FAILURE",
}


class LambdaRankError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def label_gain_policy(label_contract: str) -> dict[str, Any]:
    if label_contract == "within_date_quintile_relevance_v1":
        levels = list(range(5))
        policy_id = "exponential_gain_quintile_0_4_v1"
    elif label_contract == "within_date_decile_relevance_v1":
        levels = list(range(10))
        policy_id = "exponential_gain_decile_0_9_v1"
    else:
        raise LambdaRankError("LABEL_GAIN_MISMATCH", "NO_GAIN_POLICY_FOR_LABEL_CONTRACT")
    gains = [(2 ** level) - 1 for level in levels]
    logical = {
        "contract_version": GAIN_CONTRACT, "gain_policy_id": policy_id,
        "label_contract": label_contract, "ordered_relevance_levels": levels,
        "gain_values": gains, "maximum_supported_relevance": levels[-1],
    }
    logical["gain_checksum"] = canonical_hash(logical)
    return logical


def fixed_lambdarank_configuration(
    *,
    label_contract: str,
    num_threads: int = 1,
    device_type: str = "cpu",
) -> dict[str, Any]:
    gain = label_gain_policy(label_contract)
    parameters = dict(
        fixed_rank_xendcg_configuration(
            num_threads=num_threads,
            device_type=device_type,
        )["parameters"]
    )
    parameters["objective"] = "lambdarank"
    parameters["label_gain"] = gain["gain_values"]
    return {
        "contract_version": CONFIG_CONTRACT, "parameters": parameters,
        "gain_policy": gain,
        "objective_difference_from_rank_xendcg": ["objective", "label_gain"],
    }


def validate_lambdarank_input(
    dataset: Mapping[str, Any],
    *,
    training_cutoff: str,
    source_commit: str | None = None,
    gain_policy: Mapping[str, Any] | None = None,
    registered_integer_label_contracts: Sequence[str] = (),
) -> dict[str, Any]:
    baseline = validate_rank_xendcg_input(
        dataset, training_cutoff=training_cutoff, source_commit=source_commit,
        registered_integer_label_contracts=registered_integer_label_contracts,
    )
    if not baseline.get("valid"):
        return _blocked(INPUT_CONTRACT, LambdaRankError(baseline["status"], baseline["blocking_reasons"][0]))
    try:
        expected_gain = label_gain_policy(baseline["ranking_label_contract_identity"])
        supplied = dict(gain_policy) if gain_policy is not None else expected_gain
        if supplied != expected_gain:
            raise LambdaRankError("LABEL_GAIN_MISMATCH", "GAIN_POLICY_IDENTITY_OR_ORDER_MISMATCH")
        labels = [
            row["label"] for row in dataset["rows"]
            if row["split_role"] != "PREDICTION"
        ]
        if set(labels) - set(expected_gain["ordered_relevance_levels"]):
            raise LambdaRankError("LABEL_GAIN_MISMATCH", "LABEL_EXCEEDS_GAIN_TABLE")
        if set(range(max(labels) + 1)) - set(expected_gain["ordered_relevance_levels"]):
            raise LambdaRankError("LABEL_GAIN_MISMATCH", "RELEVANCE_LEVEL_MISSING_FROM_GAIN_POLICY")
        logical = {
            **baseline,
            "contract_version": INPUT_CONTRACT,
            "gain_policy": expected_gain,
        }
        logical.pop("input_checksum", None)
        logical["input_checksum"] = canonical_hash(logical)
        return logical
    except LambdaRankError as exc:
        return _blocked(INPUT_CONTRACT, exc)


def fit_synthetic_lambdarank_selector(
    dataset: Mapping[str, Any],
    *,
    training_cutoff: str,
    num_threads: int = 1,
    serialisation_directory: str | Path | None = None,
    source_commit: str | None = None,
    gain_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if lgb.__version__ != "4.6.0":
            raise LambdaRankError("DEPENDENCY_MISMATCH", f"LIGHTGBM_VERSION:{lgb.__version__}")
        input_contract = validate_lambdarank_input(
            dataset, training_cutoff=training_cutoff, source_commit=source_commit,
            gain_policy=gain_policy,
        )
        if not input_contract.get("valid"):
            raise LambdaRankError(input_contract["status"], input_contract["blocking_reasons"][0])
        configuration = fixed_lambdarank_configuration(
            label_contract=input_contract["ranking_label_contract_identity"],
            num_threads=num_threads,
        )
        parameters = configuration["parameters"]
        train_rows = [row for row in dataset["rows"] if row["split_role"] == "TRAINING"]
        validation_rows = [
            row for row in dataset["rows"]
            if row["split_role"] in {"PREDICTION", "VALIDATION"}
        ]
        x_train = np.asarray([row["feature_values"] for row in train_rows], dtype=float)
        y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
        x_validation = np.asarray([row["feature_values"] for row in validation_rows], dtype=float)
        fit_started = time.perf_counter()
        first = _fit(parameters, x_train, y_train, input_contract["training_group_sizes"])
        fit_seconds = time.perf_counter() - fit_started
        second = _fit(parameters, x_train, y_train, input_contract["training_group_sizes"])
        prediction_started = time.perf_counter()
        first_scores, second_scores = _predict(first, x_validation), _predict(second, x_validation)
        prediction_seconds = time.perf_counter() - prediction_started
        if not np.isfinite(first_scores).all():
            raise LambdaRankError("NONFINITE_PREDICTION", "VALIDATION_SCORE_NONFINITE")
        if len(first_scores) != len(validation_rows):
            raise LambdaRankError("INCOMPLETE_PREDICTION_POPULATION", "VALIDATION_SCORE_COUNT_MISMATCH")
        if not np.allclose(first_scores, second_scores, rtol=0.0, atol=1e-12):
            raise LambdaRankError("NONDETERMINISTIC_RESULT", "REPEATED_PREDICTIONS_DIFFER")
        first_importance = _feature_importance(first, dataset["feature_names"])
        if first_importance != _feature_importance(second, dataset["feature_names"]):
            raise LambdaRankError("NONDETERMINISTIC_RESULT", "REPEATED_IMPORTANCE_DIFFERS")
        model_text, second_text = first.booster_.model_to_string(), second.booster_.model_to_string()
        saved = reloaded = False
        model_path = None
        if serialisation_directory is not None:
            directory = Path(serialisation_directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "lightgbm_lambdarank_selector.txt"
            try:
                first.booster_.save_model(str(path))
                saved_text = path.read_text(encoding="utf-8")
            except Exception as exc:
                raise LambdaRankError("SERIALISATION_FAILURE", "MODEL_SERIALISATION_FAILED") from exc
            if canonical_hash(saved_text) != canonical_hash(model_text):
                raise LambdaRankError("SERIALISATION_FAILURE", "SAVED_MODEL_CHECKSUM_MISMATCH")
            saved, model_path = True, str(path)
            try:
                reload_scores = np.asarray(lgb.Booster(model_file=str(path)).predict(x_validation), dtype=float)
            except Exception as exc:
                raise LambdaRankError("MODEL_RELOAD_FAILURE", "MODEL_RELOAD_FAILED") from exc
            reloaded = bool(np.allclose(first_scores, reload_scores, rtol=0.0, atol=1e-12))
            if not reloaded:
                raise LambdaRankError("MODEL_RELOAD_FAILURE", "RELOAD_PREDICTIONS_DIFFER")
        model = {
            "contract_version": MODEL_CONTRACT,
            "model_id": "lightgbm_lambdarank_synthetic_challenger_v1",
            "objective": "lambdarank", "dependency_identity": "lightgbm==4.6.0",
            "parameters": parameters, "gain_policy": configuration["gain_policy"],
            "thread_policy": f"bounded_threads:{num_threads}",
            "seed_policy": "fixed_1729_all_supported_seeds",
            "feature_schema_checksum": dataset["feature_schema_checksum"],
            "target_contract_checksum": dataset["target_contract_checksum"],
            "label_contract_checksum": dataset["ranking_label_contract_checksum"],
            "grouped_dataset_checksum": dataset["dataset_checksum"],
            "group_vector_checksum": dataset["group_size_vector_checksum"],
            "training_population_checksum": input_contract["training_population_checksum"],
            "validation_population_checksum": input_contract["validation_population_checksum"],
            "training_cutoff": training_cutoff,
            "maximum_label_maturity": input_contract["maximum_training_label_maturity_timestamp"],
            "source_commit": source_commit,
            "fitted_booster_identity": canonical_hash(first.booster_.dump_model()),
            "serialised_model_checksum": canonical_hash(model_text),
        }
        model["logical_model_checksum"] = canonical_hash(model)
        predictions = _prediction_rows(validation_rows, first_scores, model, dataset, input_contract)
        prediction_checksum = canonical_hash(predictions)
        for row in predictions:
            row["prediction_population_checksum"] = prediction_checksum
        prediction_contract = {
            "contract_version": PREDICTION_CONTRACT, "rows": predictions,
            "row_count": len(predictions), "prediction_population_checksum": prediction_checksum,
        }
        diagnostics = {
            "training": _with_gain_distribution(
                _ranking_diagnostics(train_rows, _predict(first, x_train), input_contract["training_group_sizes"]),
                train_rows, configuration["gain_policy"],
            ),
            **(
                {
                    "validation": _with_gain_distribution(
                        _ranking_diagnostics(
                            validation_rows, first_scores,
                            input_contract["validation_group_sizes"],
                        ),
                        validation_rows, configuration["gain_policy"],
                    )
                }
                if all("label" in row for row in validation_rows)
                else {
                    "prediction": _prediction_score_diagnostics(
                        validation_rows, first_scores,
                        input_contract["validation_group_sizes"],
                    )
                }
            ),
            "feature_importance": first_importance,
            "tree_model": _tree_diagnostics(first, parameters, model_text),
            "repeatability": {
                "prediction_level": True,
                "rank_level": _rank_signature(validation_rows, first_scores) == _rank_signature(validation_rows, second_scores),
                "feature_importance": True,
                "logical_model": canonical_hash(first.booster_.dump_model()) == canonical_hash(second.booster_.dump_model()),
                "byte_level_serialisation": model_text == second_text,
            },
            "serialisation": {"saved": saved, "reloaded": reloaded},
            "synthetic_only": True, "promotion_evidence": False,
        }
        logical = {
            "contract_version": RESULT_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": ["SYNTHETIC_OBJECTIVE_COMPARISON_ONLY"],
            "dependency_identity": "lightgbm==4.6.0", "objective": "lambdarank",
            "parameter_identity": canonical_hash(parameters),
            "thread_identity": f"bounded_threads:{num_threads}",
            "label_contract": input_contract["ranking_label_contract_identity"],
            "gain_policy_checksum": configuration["gain_policy"]["gain_checksum"],
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
            "input_contract": input_contract, "model_contract": model,
            "prediction_contract": prediction_contract, "diagnostics": diagnostics,
            "model_checksum": model["logical_model_checksum"],
            "prediction_checksum": prediction_checksum,
            "diagnostics_checksum": canonical_hash(diagnostics),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {
            **logical,
            "runtime_metadata": {
                "fit_duration_seconds": fit_seconds,
                "prediction_duration_seconds": prediction_seconds,
                "total_duration_seconds": time.perf_counter() - started,
                "serialised_model_path": model_path,
            },
        }
    except LambdaRankError as exc:
        return _blocked(RESULT_CONTRACT, exc)


def compare_lambdarank_with_rank_xendcg(
    dataset: Mapping[str, Any],
    *,
    training_cutoff: str,
    num_threads: int = 1,
) -> dict[str, Any]:
    lambdarank = fit_synthetic_lambdarank_selector(
        dataset, training_cutoff=training_cutoff, num_threads=num_threads,
    )
    xendcg = fit_synthetic_rank_xendcg_selector(
        dataset, training_cutoff=training_cutoff, num_threads=num_threads,
    )
    if not lambdarank.get("valid") or not xendcg.get("valid"):
        return _blocked(COMPARISON_CONTRACT, LambdaRankError("INVALID_INPUT", "MATCHED_FIT_FAILED"))
    if lambdarank["population_checksums"] != xendcg["population_checksums"]:
        return _blocked(COMPARISON_CONTRACT, LambdaRankError("COMPARISON_POPULATION_MISMATCH", "POPULATION_CHECKSUMS_DIFFER"))
    lambda_rows, xendcg_rows = lambdarank["prediction_contract"]["rows"], xendcg["prediction_contract"]["rows"]
    lambda_by_id = {row["row_id"]: row for row in lambda_rows}
    xendcg_by_id = {row["row_id"]: row for row in xendcg_rows}
    if set(lambda_by_id) != set(xendcg_by_id):
        return _blocked(COMPARISON_CONTRACT, LambdaRankError("COMPARISON_POPULATION_MISMATCH", "PREDICTION_ROWS_DIFFER"))
    ordered_ids = sorted(lambda_by_id)
    lambda_scores = [lambda_by_id[row_id]["raw_score"] for row_id in ordered_ids]
    xendcg_scores = [xendcg_by_id[row_id]["raw_score"] for row_id in ordered_ids]
    lambda_top = {row["row_id"] for row in sorted(lambda_rows, key=lambda row: (row["decision_date"], row["within_date_rank"]))[:3]}
    xendcg_top = {row["row_id"] for row in sorted(xendcg_rows, key=lambda row: (row["decision_date"], row["within_date_rank"]))[:3]}
    lambda_gain = [row["normalised_gain_share"] for row in lambdarank["diagnostics"]["feature_importance"]["features"]]
    xendcg_gain = [row["normalised_gain_share"] for row in xendcg["diagnostics"]["feature_importance"]["features"]]
    logical = {
        "contract_version": COMPARISON_CONTRACT, "status": "READY", "valid": True,
        "blocking_reasons": [], "warnings": [
            "SYNTHETIC_OBJECTIVE_COMPARISON_DOES_NOT_SELECT_A_PRODUCTION_WINNER",
            "REAL_COMPARISON_REQUIRES_STRICT_OOS_MULTI_REGIME_AND_AFTER_COST_PORTFOLIO_EVIDENCE",
        ],
        "population_checksums": lambdarank["population_checksums"],
        "matched_non_objective_parameters": _matched_parameters(
            lambdarank["model_contract"]["parameters"], xendcg["model_contract"]["parameters"],
        ),
        "lambdarank": _comparison_metrics(lambdarank),
        "rank_xendcg": _comparison_metrics(xendcg),
        "score_spearman_correlation": _spearman(lambda_scores, xendcg_scores),
        "top_3_overlap": len(lambda_top & xendcg_top) / max(1, len(lambda_top | xendcg_top)),
        "feature_importance_similarity": _cosine(lambda_gain, xendcg_gain),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {
        **logical,
        "runtime_comparison": {
            "lambdarank_fit_duration_seconds": lambdarank["runtime_metadata"]["fit_duration_seconds"],
            "lambdarank_prediction_duration_seconds": lambdarank["runtime_metadata"]["prediction_duration_seconds"],
            "rank_xendcg_fit_duration_seconds": xendcg["runtime_metadata"]["training_duration_seconds"],
            "rank_xendcg_prediction_duration_seconds": xendcg["runtime_metadata"]["prediction_duration_seconds"],
        },
    }


def verify_lambdarank_result(
    dataset: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    serialisation_directory: str | Path | None = None,
) -> dict[str, Any]:
    expected = fit_synthetic_lambdarank_selector(
        dataset,
        training_cutoff=result["input_contract"]["training_cutoff"],
        num_threads=result["model_contract"]["parameters"]["n_jobs"],
        serialisation_directory=serialisation_directory,
        source_commit=result["input_contract"].get("source_commit"),
        gain_policy=result["input_contract"]["gain_policy"],
    )
    fields = (
        "status", "dependency_identity", "objective", "parameter_identity", "thread_identity",
        "label_contract", "gain_policy_checksum", "population_checksums", "input_contract",
        "model_contract", "prediction_contract", "diagnostics", "model_checksum",
        "prediction_checksum", "diagnostics_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": VERIFICATION_CONTRACT, "valid": not reasons, "blocking_reasons": reasons}


def _fit(parameters, matrix, labels, groups):
    estimator = dict(parameters)
    eval_at = estimator.pop("eval_at")
    try:
        return lgb.LGBMRanker(**estimator).fit(matrix, labels, group=groups, eval_at=eval_at)
    except Exception as exc:
        raise LambdaRankError("OBJECTIVE_UNAVAILABLE", "LAMBDARANK_FIT_FAILED") from exc


def _prediction_rows(rows, scores, model, dataset, input_contract):
    output = []
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row["decision_date"], []).append((row, float(score)))
    for decision in sorted(grouped):
        ordered = sorted(grouped[decision], key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]))
        count = len(ordered)
        for index, (row, score) in enumerate(ordered):
            output.append({
                "row_id": row["row_id"], "asset_id": row["asset_id"], "decision_date": decision,
                "raw_score": score, "within_date_rank": index + 1,
                "percentile_rank": 1.0 if count == 1 else 1.0 - index / (count - 1),
                "model_checksum": model["logical_model_checksum"],
                "grouped_dataset_checksum": dataset["dataset_checksum"],
                "feature_schema_checksum": dataset["feature_schema_checksum"],
                "target_contract_checksum": dataset["target_contract_checksum"],
                "label_contract_checksum": dataset["ranking_label_contract_checksum"],
                "gain_policy_checksum": model["gain_policy"]["gain_checksum"],
                "split_identity": dataset["split_identity"], "training_cutoff": input_contract["training_cutoff"],
                "maximum_training_label_maturity": input_contract["maximum_training_label_maturity_timestamp"],
            })
    return output


def _with_gain_distribution(diagnostics, rows, gain_policy):
    gains = dict(zip(gain_policy["ordered_relevance_levels"], gain_policy["gain_values"]))
    distribution = {}
    for row in rows:
        gain = str(gains[row["label"]])
        distribution[gain] = distribution.get(gain, 0) + 1
    return {**diagnostics, "gain_distribution": distribution}


def _matched_parameters(lambdarank, xendcg):
    ignored = {"objective", "label_gain"}
    left = {key: value for key, value in lambdarank.items() if key not in ignored}
    right = {key: value for key, value in xendcg.items() if key not in ignored}
    if left != right:
        raise LambdaRankError("COMPARISON_POPULATION_MISMATCH", "NON_OBJECTIVE_PARAMETERS_DIFFER")
    return canonical_hash(left)


def _comparison_metrics(result):
    validation = result["diagnostics"]["validation"] if result["objective"] == "lambdarank" else result["diagnostics"]["ranking"]["validation"]
    importance = result["diagnostics"]["feature_importance"]
    tree = result["diagnostics"]["tree_model"]
    return {
        "ndcg": validation["ndcg"], "score_standard_deviation": validation["score_standard_deviation"],
        "tied_score_count": validation["tied_score_count"],
        "group_ndcg_at_3_standard_deviation": validation["group_ndcg_at_3_standard_deviation"],
        "zero_importance_feature_count": importance["zero_importance_feature_count"],
        "model_byte_size": tree["model_byte_size"],
        "deterministic_repeatability": result["diagnostics"]["repeatability"],
    }


def _blocked(contract, error):
    logical = {
        "contract_version": contract,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical
