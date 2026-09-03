from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.stock_level.lightgbm_lambdarank_selector import (
    _fit as fit_lambdarank,
    fixed_lambdarank_configuration,
    validate_lambdarank_input,
)
from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
    _fit as fit_rank_xendcg,
    fixed_rank_xendcg_configuration,
    validate_rank_xendcg_input,
)


PRODUCTION_RESULT_CONTRACT = "production_lightgbm_selector_result.v1"
REQUIRED_CONTEXT = (
    "selector_dataset_identity", "selector_dataset_checksum",
    "operational_input_identity", "operational_input_checksum",
    "campaign_identity", "production_plan_job_checksum",
    "model_registry_identity", "ranking_contract_identity",
    "grouped_query_contract", "relevance_label_contract",
    "target_contract", "horizon_contract", "fold_identity",
    "training_boundary_identity", "outcome_maturity_cutoff",
    "purge_sessions", "embargo_sessions", "feature_schema",
    "ordered_feature_checksum", "model_configuration_checksum",
    "seed", "source_commit",
)


@dataclass(frozen=True)
class FittedLightGBMRanker:
    estimator: Any
    feature_order: tuple[str, ...]
    feature_schema_identity: str
    feature_schema_checksum: str
    configuration: Mapping[str, Any]
    input_contract: Mapping[str, Any]
    group_evidence: Mapping[str, Any]
    ranking_label_evidence: Mapping[str, Any]


def fit_production_lightgbm_selector(
    dataset: Mapping[str, Any],
    *,
    model_id: str,
    authoritative_context: Mapping[str, Any],
    dependency_preflight: Mapping[str, Any],
    estimator_fitters: Mapping[str, Any] | None = None,
    fitted_model_callback: Callable[[FittedLightGBMRanker], None] | None = None,
) -> dict[str, Any]:
    """Fit one authoritative grouped ranker; publication remains Wave-4 owned."""
    missing = [
        field for field in REQUIRED_CONTEXT
        if authoritative_context.get(field) in (None, "")
    ]
    if missing:
        return _blocked("MANDATORY_IDENTITY_MISSING:" + ",".join(missing))
    if dependency_preflight.get("status") != "READY":
        return _blocked("DEPENDENCY_PREFLIGHT_NOT_READY")
    if authoritative_context["fold_identity"] != dataset.get("split_identity"):
        return _blocked("FOLD_IDENTITY_MISMATCH")
    if authoritative_context["operational_input_checksum"] != dataset.get(
        "dataset_checksum"
    ):
        return _blocked("OPERATIONAL_INPUT_CHECKSUM_MISMATCH")
    if authoritative_context["relevance_label_contract"] != dataset.get(
        "ranking_label_contract_identity"
    ):
        return _blocked("RELEVANCE_CONTRACT_MISMATCH")
    if authoritative_context["feature_schema"] != dataset.get(
        "feature_schema_identity"
    ):
        return _blocked("FEATURE_SCHEMA_IDENTITY_MISMATCH")
    if authoritative_context["ordered_feature_checksum"] != canonical_hash(
        list(dataset.get("feature_names") or ())
    ):
        return _blocked("ORDERED_FEATURE_CHECKSUM_MISMATCH")
    if int(authoritative_context["seed"]) != 1729:
        return _blocked("SEED_IDENTITY_MISMATCH")
    if (
        int(authoritative_context["purge_sessions"]) != 20
        or int(authoritative_context["embargo_sessions"]) != 5
    ):
        return _blocked("PURGE_OR_EMBARGO_POLICY_MISMATCH")

    resolver = RegistryResolver(load_registry_bundle())
    model_entry = resolver.resolve(
        "selector_models", model_id, role="selector"
    ).entry
    target_entry = resolver.resolve(
        "target_contracts",
        str(authoritative_context["target_contract"]),
        role="selector",
    ).entry
    payload = model_entry.payload
    if authoritative_context["model_registry_identity"] != model_entry.entry_hash:
        return _blocked("MODEL_REGISTRY_IDENTITY_MISMATCH")
    if authoritative_context["ranking_contract_identity"] != payload.get(
        "ranking_problem_contract"
    ):
        return _blocked("RANKING_CONTRACT_IDENTITY_MISMATCH")
    if authoritative_context["grouped_query_contract"] != payload.get(
        "grouped_query_contract"
    ):
        return _blocked("GROUPED_QUERY_CONTRACT_MISMATCH")
    if authoritative_context["relevance_label_contract"] != payload.get(
        "relevance_contract"
    ):
        return _blocked("REGISTERED_RELEVANCE_CONTRACT_MISMATCH")
    if dataset.get("target_contract_identity") != target_entry.canonical_id:
        return _blocked("TARGET_CONTRACT_IDENTITY_MISMATCH")

    cutoff = str(authoritative_context["outcome_maturity_cutoff"])
    label_contract = str(dataset.get("ranking_label_contract_identity") or "")
    try:
        device_type, runtime_policy = _lightgbm_runtime_policy(dependency_preflight)
    except ValueError as exc:
        return _blocked(str(exc))
    if model_id == "lightgbm_rank_xendcg":
        input_contract = validate_rank_xendcg_input(
            dataset, training_cutoff=cutoff,
            source_commit=str(authoritative_context["source_commit"]),
        )
        configuration = fixed_rank_xendcg_configuration(num_threads=1, device_type=device_type)
        default_fitter = fit_rank_xendcg
        objective = "rank_xendcg"
    elif model_id == "lightgbm_lambdarank":
        input_contract = validate_lambdarank_input(
            dataset, training_cutoff=cutoff,
            source_commit=str(authoritative_context["source_commit"]),
        )
        configuration = fixed_lambdarank_configuration(
            label_contract=label_contract, num_threads=1, device_type=device_type
        )
        default_fitter = fit_lambdarank
        objective = "lambdarank"
    else:
        return _blocked("UNSUPPORTED_LIGHTGBM_MODEL")
    if not input_contract.get("valid"):
        return _blocked(
            str(input_contract.get("blocking_reasons", ["INVALID_INPUT"])[0]),
            status=str(input_contract.get("status") or "INVALID_INPUT"),
        )
    if canonical_hash(configuration) != authoritative_context[
        "model_configuration_checksum"
    ]:
        return _blocked("MODEL_CONFIGURATION_CHECKSUM_MISMATCH")
    parameters = dict(configuration["parameters"])
    if parameters.get("objective") != objective or parameters.get("n_jobs") != 1:
        return _blocked("OBJECTIVE_OR_THREAD_POLICY_MISMATCH")
    if dependency_preflight.get("objective") not in (None, objective):
        return _blocked("DEPENDENCY_OBJECTIVE_MISMATCH")

    training = [
        row for row in dataset["rows"] if row["split_role"] == "TRAINING"
    ]
    prediction = [
        row for row in dataset["rows"] if row["split_role"] == "VALIDATION"
    ]
    fitter = (estimator_fitters or {}).get(model_id, default_fitter)
    model = fitter(
        parameters,
        np.asarray([row["feature_values"] for row in training], dtype=float),
        np.asarray([row["label"] for row in training], dtype=int),
        input_contract["training_group_sizes"],
    )
    if fitted_model_callback is not None:
        training_group_dates = list(dict.fromkeys(
            str(row["decision_date"]) for row in training
        ))
        training_labels = [int(row["label"]) for row in training]
        fitted_payload = FittedLightGBMRanker(
            estimator=model,
            feature_order=tuple(dataset["feature_names"]),
            feature_schema_identity=dataset["feature_schema_identity"],
            feature_schema_checksum=dataset["feature_schema_checksum"],
            configuration=configuration,
            input_contract=input_contract,
            group_evidence={
                "source_group_contract_identity": dataset["contract_version"],
                "grouped_query_contract": authoritative_context[
                    "grouped_query_contract"
                ],
                "deterministic_group_ordering": (
                    "decision_date_ascending_then_asset_id_then_row_id"
                ),
                "training_group_dates": training_group_dates,
                "training_group_sizes": list(
                    input_contract["training_group_sizes"]
                ),
                "training_group_row_count": len(training),
                "ordered_training_membership_checksum": canonical_hash([
                    {
                        "row_id": row["row_id"],
                        "asset_id": row["asset_id"],
                        "query": row["decision_date"],
                    }
                    for row in training
                ]),
            },
            ranking_label_evidence={
                "raw_ranking_outcome_identity": dataset[
                    "target_contract_identity"
                ],
                "relevance_label_contract": dataset[
                    "ranking_label_contract_identity"
                ],
                "ranking_label_contract_checksum": dataset[
                    "ranking_label_contract_checksum"
                ],
                "ordered_training_label_checksum": canonical_hash(
                    training_labels
                ),
                "ordered_relevance_levels": sorted(set(training_labels)),
                "label_distribution": {
                    str(value): training_labels.count(value)
                    for value in sorted(set(training_labels))
                },
                "label_count": len(training_labels),
                "maturity_policy": (
                    "training_labels_mature_at_or_before_training_cutoff"
                ),
                "maximum_training_label_maturity_timestamp": input_contract[
                    "maximum_training_label_maturity_timestamp"
                ],
                "training_only_label_claim": True,
                "published_prediction_rows_unlabeled": True,
            },
        )
    scores = np.asarray(
        model.predict(
            np.asarray([row["feature_values"] for row in prediction], dtype=float)
        ),
        dtype=float,
    )
    if len(scores) != len(prediction) or not np.isfinite(scores).all():
        return _blocked("INVALID_PREDICTION_POPULATION")
    rows = _prediction_rows(
        prediction, scores, authoritative_context, input_contract
    )
    if fitted_model_callback is not None:
        fitted_model_callback(fitted_payload)
    capability = {
        "production_owner": True,
        "synthetic_only": False,
        "strict_oos_capable": True,
        "authoritative_input_required": True,
        "campaign_execution_eligible": True,
        "promotion_evidence": False,
        "promoted": False,
    }
    result = {
        "contract_version": PRODUCTION_RESULT_CONTRACT,
        "status": "READY", "valid": True, "blocking_reasons": [],
        "model_id": model_id, "objective": objective,
        "dependency_identity": (
            "lightgbm==" + str(dependency_preflight["lightgbm_version"])
        ),
        "configuration": configuration,
        "lightgbm_runtime_policy": runtime_policy,
        "authoritative_context": dict(authoritative_context),
        "input_contract": input_contract,
        "prediction_contract": {
            "contract_version": (
                "strict_oos_lightgbm_component_predictions.v1"
            ),
            "rows": rows, "row_count": len(rows),
            "ordered_population_checksum": canonical_hash(
                [row["row_id"] for row in rows]
            ),
        },
        "capability_evidence": capability,
        **capability,
    }
    result["logical_result_checksum"] = canonical_hash(result)
    return result


def _prediction_rows(rows, scores, context, input_contract):
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(str(row["decision_date"]), []).append(
            (row, float(score))
        )
    output = []
    for decision_date in sorted(grouped):
        ranked = sorted(
            grouped[decision_date],
            key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]),
        )
        for rank, (row, score) in enumerate(ranked, start=1):
            output.append({
                "row_id": row["row_id"], "asset_id": row["asset_id"],
                "symbol": row.get("symbol", row["asset_id"]),
                "decision_date": decision_date, "raw_score": score,
                "within_date_rank": rank,
                "fold_identity": context["fold_identity"],
                "campaign_identity": context["campaign_identity"],
                "production_plan_job_checksum": context[
                    "production_plan_job_checksum"
                ],
                "maximum_training_label_maturity": input_contract[
                    "maximum_training_label_maturity_timestamp"
                ],
            })
    return output


def _lightgbm_runtime_policy(dependency_preflight: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    prefer_gpu = bool(
        dependency_preflight.get("lightgbm_prefer_gpu")
        or dependency_preflight.get("prefer_gpu")
    )
    gpu_supported = bool(
        dependency_preflight.get("lightgbm_gpu_supported")
        or dependency_preflight.get("gpu_supported")
    )
    fallback_allowed = bool(dependency_preflight.get("lightgbm_safe_cpu_fallback", True))
    if prefer_gpu and gpu_supported:
        return "gpu", {
            "preference": "GPU",
            "selected_device_type": "gpu",
            "safe_cpu_fallback_used": False,
            "fallback_reason": "",
        }
    if prefer_gpu and not gpu_supported:
        if not fallback_allowed:
            raise ValueError("LIGHTGBM_GPU_REQUESTED_WITHOUT_SAFE_CPU_FALLBACK")
        return "cpu", {
            "preference": "GPU",
            "selected_device_type": "cpu",
            "safe_cpu_fallback_used": True,
            "fallback_reason": str(
                dependency_preflight.get("lightgbm_gpu_fallback_reason")
                or "LIGHTGBM_GPU_NOT_SUPPORTED_BY_CURRENT_BUILD_OR_DRIVER"
            ),
        }
    return "cpu", {
        "preference": "CPU",
        "selected_device_type": "cpu",
        "safe_cpu_fallback_used": False,
        "fallback_reason": "",
    }


def _blocked(reason: str, *, status: str = "INVALID_INPUT") -> dict[str, Any]:
    result = {
        "contract_version": PRODUCTION_RESULT_CONTRACT,
        "status": status, "valid": False, "blocking_reasons": [reason],
        "production_owner": True, "synthetic_only": False,
        "strict_oos_capable": True, "authoritative_input_required": True,
        "campaign_execution_eligible": False,
        "promotion_evidence": False, "promoted": False,
    }
    result["logical_result_checksum"] = canonical_hash(result)
    return result
