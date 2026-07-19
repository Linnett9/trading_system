from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.research.ml.artifact_lineage import VERIFIED_STRICT_OOS, build_artifact_link, verify_selector_artifact
from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash
from core.research.ml.lightgbm_ranking_preflight import (
    SUPPORTED_OBJECTIVES,
    deterministic_ranker_configuration,
)
from core.research.ml.ranking import OrderedLogitRanker
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.stock_level.contextual_elastic_net_selector import fit_contextual_elastic_net
from core.research.ml.stock_level.huber_selector import fit_huber_selector
from core.research.ml.stock_level.lightgbm_production_selector import (
    FittedLightGBMRanker,
)
from core.research.ml.stock_level.multi_horizon_linear_selector import (
    FittedMultiHorizonMember,
    HORIZON_IDS,
    fit_multi_horizon_linear_selector,
)
from core.research.ml.stock_level.selector_target_identity import (
    validate_selector_target_identity,
)
from core.research.ml.stock_level.selector_multihorizon_model_artifacts import (
    publish_selector_multihorizon_package,
)
from core.research.ml.stock_level.selector_lightgbm_model_artifacts import (
    publish_selector_lightgbm_model_package,
)
from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
    publish_selector_sklearn_model_package,
)


PUBLICATION_CONTRACT = "ordinary_selector_publication.v1"
COMPONENT_CONTRACT = "authoritative_selector_component_v1"
AGGREGATION_CONTRACT = "wave4_multi_horizon_evidence.v1"
MODEL_FAMILIES = {
    "huber", "contextual_elastic_net", "multi_horizon_ridge",
    "multi_horizon_elastic_net", "multi_horizon_ordered_logit",
    "lightgbm_rank_xendcg", "lightgbm_lambdarank",
}
HORIZON_TARGETS = {
    "return_1s": "forward_return_1d", "return_5s": "forward_return_5d",
    "return_10s": "forward_return_10d", "return_20s": "forward_return_20d",
}
LIGHTGBM_PREFLIGHT_CONTRACT = "lightgbm_ranking_dependency_preflight_v2"


def assess_lightgbm_ranking_dependency(
    *,
    objective: str,
    num_threads: int = 1,
    importer=import_module,
) -> dict[str, Any]:
    status, reasons, version, dependency = "READY", [], None, None
    try:
        dependency = importer("lightgbm")
    except (ImportError, ModuleNotFoundError):
        status, reasons = "MISSING_DEPENDENCY", ["LIGHTGBM_IMPORT_FAILED"]
    if dependency is not None:
        version = str(getattr(dependency, "__version__", ""))
        if version != "4.6.0":
            status, reasons = "UNSUPPORTED_VERSION", [
                "LIGHTGBM_VERSION_REQUIRED:4.6.0"
            ]
        elif not callable(getattr(dependency, "LGBMRanker", None)):
            status, reasons = "UNSUPPORTED_OBJECTIVE", [
                "LGBMRANKER_API_MISSING"
            ]
    try:
        configuration = deterministic_ranker_configuration(
            objective=objective, num_threads=num_threads
        )
    except ValueError as exc:
        configuration = None
        status = (
            "UNSUPPORTED_OBJECTIVE"
            if objective not in SUPPORTED_OBJECTIVES
            else "INVALID_CONFIGURATION"
        )
        reasons.append(str(exc))
    if configuration is not None and configuration["n_jobs"] != 1:
        status = "INVALID_CONFIGURATION"
        reasons.append("INNER_N_JOBS_MUST_EQUAL_ONE")
    logical = {
        "contract_version": LIGHTGBM_PREFLIGHT_CONTRACT,
        "status": status,
        "valid": status == "READY",
        "blocking_reasons": sorted(set(reasons)),
        "dependency_available": dependency is not None,
        "lightgbm_version": version,
        "required_lightgbm_version": "4.6.0",
        "objective": objective,
        "objective_supported_by_registered_version": (
            status == "READY" and objective in SUPPORTED_OBJECTIVES
        ),
        "grouped_ranker_api_available": (
            dependency is not None
            and callable(getattr(dependency, "LGBMRanker", None))
        ),
        "runtime_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "deterministic_configuration": configuration,
        "inner_n_jobs": configuration.get("n_jobs") if configuration else None,
        "fitting_performed": False,
    }
    logical["configuration_checksum"] = (
        canonical_hash(configuration) if configuration else None
    )
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def publish_wave4_component(
    *,
    model_id: str,
    prediction_date: str,
    fit_input: Mapping[str, Any],
    output_root: Path,
    parent_gate: Mapping[str, Any],
    ledger_path: Path,
    horizon_id: str | None = None,
    interaction_contract: Mapping[str, Any] | None = None,
    fit_options: Mapping[str, Any] | None = None,
    campaign_identity: str | None = None,
    production_plan_job_checksum: str | None = None,
    plan_job_identity: str | None = None,
    declared_component_runner: str | None = None,
    resolved_runtime_owner: str | None = None,
    operational_input_identity: str | None = None,
    component_owner: Path | None = None,
) -> dict[str, Any]:
    """Fit and atomically publish one synthetic-or-production-ready Wave 4 component."""
    if model_id not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported Wave 4 model: {model_id}")
    if parent_gate.get("gate_contract_version") != "selector_parent_publication_gate.v1" or parent_gate.get("status") != "READY":
        raise ValueError("Parent gate not ready")
    if model_id.startswith("multi_horizon_") and horizon_id not in HORIZON_IDS:
        raise ValueError("Explicit registered horizon is required")
    if not model_id.startswith("multi_horizon_") and horizon_id is not None:
        raise ValueError("Horizon is only valid for a multi-horizon component")

    resolver = RegistryResolver(load_registry_bundle())
    model = resolver.resolve("selector_models", model_id, role="selector")
    model_payload = model.entry.payload
    target_identity = validate_selector_target_identity(
        economic_target_id=model_payload.get("economic_target_id"),
        target_provenance_contract_version=model_payload.get(
            "target_provenance_contract_version"
        ),
    )
    dependency_preflight = None
    if model_id.startswith("lightgbm_"):
        if not campaign_identity or not production_plan_job_checksum:
            raise ValueError(
                "LightGBM components require campaign and plan-job identity"
            )
        dependency_preflight = assess_lightgbm_ranking_dependency(
            objective=str(model_payload["objective"]), num_threads=1
        )
        if dependency_preflight["status"] != "READY":
            raise ValueError(
                "LightGBM dependency preflight not ready: "
                + dependency_preflight["status"]
            )
    target_id = (
        HORIZON_TARGETS[horizon_id]
        if horizon_id
        else target_identity.economic_target_id
    )
    target = resolver.resolve("target_contracts", target_id, role="selector")
    runner_evidence = {
        "campaign_identity": campaign_identity or "legacy_direct_wave4.v1",
        "plan_job_identity": plan_job_identity or (
            f"wave4:{prediction_date}:{model_id}:{horizon_id or 'default'}"
        ),
        "production_plan_job_checksum": (
            production_plan_job_checksum or "legacy-direct-wave4"
        ),
        "declared_component_runner": declared_component_runner or (
            "core.research.ml.stock_level.wave4_selector_integration:"
            "publish_wave4_component"
        ),
        "resolved_runtime_owner": resolved_runtime_owner or (
            "core.research.ml.stock_level.wave4_selector_integration:"
            "publish_wave4_component"
        ),
        "operational_input_identity": operational_input_identity or (
            (fit_options or {}).get("operational_input_identity")
            or f"legacy-fit-input:{fit_input.get('logical_input_checksum')}"
        ),
    }
    identity = {
        "model_id": model.canonical_id, "model_entry_hash": model.entry.entry_hash,
        "horizon_id": horizon_id, "target_contract": target.canonical_id,
        **target_identity.as_dict(),
        "target_entry_hash": target.entry.entry_hash, "prediction_date": prediction_date,
        "dataset_id": parent_gate["selector_dataset_id"],
        "dataset_checksum": parent_gate["selector_dataset_artifact_checksum"],
        "feature_schema": model.entry.payload["feature_schema"],
        "parent_gate_checksum": parent_gate["logical_checksum"],
        "fit_input_checksum": fit_input.get("logical_input_checksum"),
        "fit_options": dict(fit_options or {}),
        **runner_evidence,
    }
    if dependency_preflight:
        identity.update(
            {
                "ranking_objective_identity": model_payload[
                    "objective_identity"
                ],
                "ranking_problem_contract": model_payload[
                    "ranking_problem_contract"
                ],
                "relevance_contract": model_payload["relevance_contract"],
                "grouped_query_contract": model_payload[
                    "grouped_query_contract"
                ],
                "fitting_configuration_checksum": model_payload[
                    "fitting_configuration_checksum"
                ],
                "dependency_preflight_identity": dependency_preflight[
                    "contract_version"
                ],
                "dependency_version": dependency_preflight[
                    "lightgbm_version"
                ],
                "dependency_preflight_checksum": dependency_preflight[
                    "logical_result_checksum"
                ],
                "seed": 1729,
                "inner_n_jobs": 1,
            }
        )
    spec_hash = experiment_spec_hash(identity)
    run_id = f"wave4-{spec_hash[:20].lower()}"
    owner = component_owner or (
        output_root / f"model={model_id}"
        / (
            f"horizon={horizon_id}/date={prediction_date}"
            if horizon_id else f"date={prediction_date}"
        )
    )
    existing = _compatible(owner, identity)
    if existing:
        _event(ledger_path, spec_hash, run_id, "SKIPPED_COMPLETE", model_id, horizon_id, identity, (str(owner / "manifest.json"),))
        return {"status": "SKIPPED_COMPLETE", "manifest_path": str(owner / "manifest.json")}
    if owner.exists():
        raise FileExistsError(f"Incompatible component exists: {owner}")

    _event(ledger_path, spec_hash, run_id, "STARTED", model_id, horizon_id, identity)
    temp = owner.with_name(f".{owner.name}.{uuid.uuid4().hex}.tmp")
    try:
        publication_option_names = {
            "operational_input_identity", "operational_input_checksum",
            "training_boundary_identity", "training_cutoff",
            "purge_sessions", "embargo_sessions", "source_commit",
        }
        model_fit_options = {
            key: value for key, value in (fit_options or {}).items()
            if key not in publication_option_names
        }
        captured_fitted_models: list[dict[str, Any]] = []
        captured_multihorizon_members: list[FittedMultiHorizonMember] = []
        captured_lightgbm_models: list[FittedLightGBMRanker] = []
        if model_id in {"huber", "contextual_elastic_net"}:
            model_fit_options["fitted_model_callback"] = (
                lambda **payload: captured_fitted_models.append(dict(payload))
            )
        elif model_id in {
            "multi_horizon_ridge",
            "multi_horizon_elastic_net",
        }:
            model_fit_options["fitted_member_callback"] = (
                captured_multihorizon_members.append
            )
        elif model_id in {
            "lightgbm_rank_xendcg",
            "lightgbm_lambdarank",
        }:
            model_fit_options["fitted_model_callback"] = (
                captured_lightgbm_models.append
            )
        result, predictions, diagnostics = _fit_and_select(
            model_id, horizon_id, fit_input, interaction_contract,
            model_fit_options, authoritative_context=(
                {
                    "selector_dataset_identity": parent_gate[
                        "selector_dataset_id"
                    ],
                    "selector_dataset_checksum": parent_gate[
                        "selector_dataset_artifact_checksum"
                    ],
                    "operational_input_identity": (fit_options or {}).get(
                        "operational_input_identity"
                    ),
                    "operational_input_checksum": (fit_options or {}).get(
                        "operational_input_checksum"
                    ),
                    "campaign_identity": campaign_identity,
                    "production_plan_job_checksum": production_plan_job_checksum,
                    "model_registry_identity": model.entry.entry_hash,
                    "ranking_contract_identity": model_payload.get(
                        "ranking_problem_contract"
                    ),
                    "grouped_query_contract": model_payload.get(
                        "grouped_query_contract"
                    ),
                    "relevance_label_contract": model_payload.get(
                        "relevance_contract"
                    ),
                    "target_contract": target.canonical_id,
                    "horizon_contract": horizon_id or "return_10s",
                    "fold_identity": fit_input.get("split_identity"),
                    "training_boundary_identity": (fit_options or {}).get(
                        "training_boundary_identity"
                    ),
                    "outcome_maturity_cutoff": (fit_options or {}).get(
                        "training_cutoff"
                    ),
                    "purge_sessions": (fit_options or {}).get("purge_sessions"),
                    "embargo_sessions": (fit_options or {}).get(
                        "embargo_sessions"
                    ),
                    "feature_schema": fit_input.get(
                        "feature_schema_identity"
                    ),
                    "ordered_feature_checksum": canonical_hash(
                        list(fit_input.get("feature_names") or ())
                    ),
                    "model_configuration_checksum": model_payload.get(
                        "fitting_configuration_checksum"
                    ),
                    "seed": 1729,
                    "source_commit": (fit_options or {}).get("source_commit"),
                }
                if model_id.startswith("lightgbm_") else None
            ),
            dependency_preflight=dependency_preflight,
        )
        if model_id.startswith("lightgbm_") and {
            str(row.get("decision_date") or "") for row in predictions
        } != {prediction_date}:
            raise ValueError(
                "LightGBM validation rows must own the component prediction date"
            )
        if not result.get("valid") or not predictions:
            reason = ";".join(result.get("blocking_reasons") or [str(result.get("status"))])
            _event(ledger_path, spec_hash, run_id, "REJECTED", model_id, horizon_id, identity, rejection_summary=reason)
            raise ValueError(f"Wave 4 fit rejected: {reason}")
        output_rows = _component_rows(predictions, model_id, horizon_id, prediction_date, identity)
        if not all(math.isfinite(float(row["selector_score"])) for row in output_rows):
            raise ValueError("Nonfinite selector score")
        temp.mkdir(parents=True, exist_ok=False)
        _write_csv(temp / "predictions.csv", output_rows)
        prediction_checksum = _sha256(temp / "predictions.csv")
        population_checksum = canonical_hash([row["row_id"] for row in output_rows])
        training_start, training_cutoff, label_max = _temporal_fields(fit_input, result, horizon_id)
        if not (training_start < training_cutoff < prediction_date and label_max <= prediction_date):
            raise ValueError("Temporal legality failed")
        final_predictions = owner / "predictions.csv"
        link = build_artifact_link(
            artifact_kind="ORDINARY_SELECTOR_PREDICTION",
            artifact_id=f"wave4-selector:{canonical_hash(identity)}",
            artifact_manifest_path=owner / "manifest.json",
            artifact_path=final_predictions, artifact_checksum=prediction_checksum,
            experiment_spec_hash=spec_hash, experiment_run_id=run_id,
            source_commit=_git_commit(), canonical_model_or_policy_id=model_id,
            model_or_policy_entry_hash=model.entry.entry_hash,
            dataset_id=parent_gate["selector_dataset_id"],
            dataset_checksum=parent_gate["selector_dataset_artifact_checksum"],
            row_population_hash=population_checksum,
            feature_schema_hash=parent_gate["selector_feature_schema_checksum"],
            target_contract_hash=target.entry.entry_hash,
            decision_start=prediction_date, decision_end=prediction_date,
            training_cutoff=training_cutoff,
            maximum_label_available_timestamp=label_max,
            strict_oos_claim=True,
            strict_oos_evidence={
                "prediction_quality_passed": True, "row_population_verified": True,
                "temporal_legality_checked": True,
            }, completion_status="complete",
        )
        verification = verify_selector_artifact(link); link.update(verification.to_dict())
        if verification.status != VERIFIED_STRICT_OOS:
            raise ValueError(f"Artifact verification failed: {verification.reason_codes}")
        metrics = {
            "publication_contract_version": PUBLICATION_CONTRACT,
            "model_id": model_id, "horizon_id": horizon_id,
            "prediction_date": prediction_date, "fit_status": result["status"],
            "configuration": result.get("configuration", {}),
            "diagnostics": diagnostics,
            "gate_w4_evidence": {
                "multi_regime_rank_ic": None, "portfolio_utility_reference": None,
                "turnover_reference": None, "stability_reference": "metrics.json",
                "incremental_information_comparison": None,
                "model_to_baseline_rank_correlation": None,
                "rejected_date_count": 0, "effective_experiment_count": 1,
                "gate_passed": False,
            },
        }
        _write_json(temp / "metrics.json", metrics)
        shared_model_artifact = None
        if model_id in {"huber", "contextual_elastic_net"}:
            if len(captured_fitted_models) != 1:
                raise ValueError("PREPROCESSING_EVIDENCE_MISSING")
            fitted = captured_fitted_models[0]
            shared_model_artifact = publish_selector_sklearn_model_package(
                component_root=temp,
                published_component_root=owner,
                estimator=fitted["estimator"],
                preprocessing=fitted["preprocessing"],
                feature_order=fitted["feature_order"],
                model_id=model_id,
                model_family=model_id,
                model_configuration=fitted["model_configuration"],
                random_seed=fitted["random_seed"],
                training_boundary=fitted["training_boundary"],
                training_population_checksum=fitted[
                    "training_population_checksum"
                ],
                target_horizon_identity=target.canonical_id,
                economic_target_id=target_identity.economic_target_id,
                target_provenance_contract_version=(
                    target_identity.target_provenance_contract_version
                ),
                prediction_path=temp / "predictions.csv",
                prediction_schema=sorted(
                    {key for row in output_rows for key in row}
                ),
                prediction_count=len(output_rows),
                input_population_checksum=str(
                    fit_input.get("logical_input_checksum")
                    or fit_input.get("input_checksum")
                ),
                output_population_checksum=population_checksum,
                campaign_identity=runner_evidence["campaign_identity"],
                plan_job_identity=runner_evidence["plan_job_identity"],
                component_identity=canonical_hash(identity),
                component_runner=runner_evidence[
                    "declared_component_runner"
                ],
                runtime_owner=runner_evidence["resolved_runtime_owner"],
                implementation_owner=(
                    f"{type(fitted['estimator']).__module__}."
                    f"{type(fitted['estimator']).__qualname__}"
                ),
                decision_date=prediction_date,
                fold_identity=str(fitted["fold_identity"]),
                training_row_artifact_identity=(
                    f"{runner_evidence['operational_input_identity']}:training"
                ),
                prediction_row_artifact_identity=(
                    f"{runner_evidence['operational_input_identity']}:prediction"
                ),
                source_schema_guarantee_identity=str(
                    parent_gate["selector_feature_schema_checksum"]
                ),
                input_package_identity=runner_evidence[
                    "operational_input_identity"
                ],
                source_git_commit=_git_commit(),
                contextual_evidence=fitted["contextual_evidence"],
            )
        elif model_id in {
            "multi_horizon_ridge",
            "multi_horizon_elastic_net",
        }:
            shared_model_artifact = publish_selector_multihorizon_package(
                component_root=temp,
                published_component_root=owner,
                fitted_members=captured_multihorizon_members,
                fit_result=result,
                selected_component_rows=output_rows,
                model_id=model_id,
                campaign_identity=runner_evidence["campaign_identity"],
                plan_job_identity=runner_evidence["plan_job_identity"],
                component_identity=canonical_hash(identity),
                component_runner=runner_evidence[
                    "declared_component_runner"
                ],
                runtime_owner=runner_evidence["resolved_runtime_owner"],
                decision_date=prediction_date,
                training_row_artifact_identity=(
                    f"{runner_evidence['operational_input_identity']}:training"
                ),
                prediction_row_artifact_identity=(
                    f"{runner_evidence['operational_input_identity']}:prediction"
                ),
                input_package_identity=runner_evidence[
                    "operational_input_identity"
                ],
                source_schema_guarantee_identity=str(
                    parent_gate["selector_feature_schema_checksum"]
                ),
                input_population_checksum=str(
                    fit_input.get("logical_input_checksum")
                    or fit_input.get("input_checksum")
                ),
                source_git_commit=_git_commit(),
                economic_target_id=target_identity.economic_target_id,
                target_provenance_contract_version=(
                    target_identity.target_provenance_contract_version
                ),
            )
        elif model_id in {
            "lightgbm_rank_xendcg",
            "lightgbm_lambdarank",
        }:
            if len(captured_lightgbm_models) != 1:
                raise ValueError("LIGHTGBM_NATIVE_MODEL_MISSING")
            fitted = captured_lightgbm_models[0]
            shared_model_artifact = publish_selector_lightgbm_model_package(
                component_root=temp,
                published_component_root=owner,
                estimator=fitted.estimator,
                feature_order=fitted.feature_order,
                feature_schema_identity=fitted.feature_schema_identity,
                feature_schema_checksum=fitted.feature_schema_checksum,
                source_schema_guarantee_identity=str(
                    parent_gate["selector_feature_schema_checksum"]
                ),
                configuration=fitted.configuration,
                input_contract=fitted.input_contract,
                group_evidence=fitted.group_evidence,
                ranking_label_evidence=fitted.ranking_label_evidence,
                model_id=model_id,
                prediction_path=temp / "predictions.csv",
                prediction_schema=sorted(
                    {key for row in output_rows for key in row}
                ),
                prediction_count=len(output_rows),
                output_population_checksum=population_checksum,
                campaign_identity=runner_evidence["campaign_identity"],
                plan_job_identity=runner_evidence["plan_job_identity"],
                component_identity=canonical_hash(identity),
                component_runner=runner_evidence[
                    "declared_component_runner"
                ],
                runtime_owner=runner_evidence["resolved_runtime_owner"],
                decision_date=prediction_date,
                horizon_identity=horizon_id or "return_10s",
                training_row_artifact_identity=(
                    f"{runner_evidence['operational_input_identity']}:training"
                ),
                prediction_row_artifact_identity=(
                    f"{runner_evidence['operational_input_identity']}:prediction"
                ),
                input_package_identity=runner_evidence[
                    "operational_input_identity"
                ],
                input_population_checksum=str(
                    fit_input.get("dataset_checksum")
                    or fit_input.get("logical_input_checksum")
                    or fit_input.get("input_checksum")
                ),
                source_git_commit=_git_commit(),
                lightgbm_version=str(
                    dependency_preflight["lightgbm_version"]
                ),
                economic_target_id=target_identity.economic_target_id,
                target_provenance_contract_version=(
                    target_identity.target_provenance_contract_version
                ),
            )
        manifest = {
            "component_schema_version": COMPONENT_CONTRACT,
            "publication_contract_version": PUBLICATION_CONTRACT,
            "selector_model_identity": model_id,
            "selector_model_version": model.entry.entry_hash,
            "component_subtype": f"{model_id}__{horizon_id}" if horizon_id else model_id,
            "horizon_id": horizon_id,
            "horizon_sessions": int(horizon_id.removeprefix("return_").removesuffix("s")) if horizon_id else 10,
            "prediction_date": prediction_date,
            "training_start": training_start, "training_cutoff": training_cutoff,
            "training_label_available_timestamp_max": label_max,
            "fold_identity": fit_input.get("fold_identity") or fit_input.get("validation_fold_identity"),
            "frozen_selector_dataset_identity": {
                "dataset_id": parent_gate["selector_dataset_id"],
                "dataset_checksum": parent_gate["selector_dataset_artifact_checksum"],
            },
            "symbol_registry_identity": parent_gate["canonical_registry_id"],
            "daily_stock_spine_identity": parent_gate["daily_spine_id"],
            "feature_contract_version": model.entry.payload["feature_schema"],
            "interaction_contract_version": (
                interaction_contract.get("contract_version") if interaction_contract else None
            ),
            "economic_target_id": target_identity.economic_target_id,
            "target_provenance_contract_version": (
                target_identity.target_provenance_contract_version
            ),
            "legacy_target_contract": model_payload.get("target_contract"),
            "ranking_contract_version": model.entry.payload.get("ranking_problem_contract") or "ranking_metric_contract_v1",
            "relevance_contract_version": model.entry.payload.get("relevance_contract"),
            "ranking_objective_identity": model.entry.payload.get(
                "objective_identity"
            ),
            "grouped_query_contract": model.entry.payload.get(
                "grouped_query_contract"
            ),
            "model_configuration_checksum": model.entry.payload.get(
                "fitting_configuration_checksum"
            ),
            "dependency_preflight": dependency_preflight,
            "production_capability_evidence": result.get(
                "capability_evidence"
            ),
            "prediction_row_count": len(output_rows),
            "prediction_population_checksum": population_checksum,
            "prediction_artifact_path": str(final_predictions),
            "prediction_checksum": prediction_checksum,
            "artifact_link": link, "publication_status": "complete",
            "validation_status": VERIFIED_STRICT_OOS, "git_commit": _git_commit(),
            "non_production_smoke": False,
            "parent_gate_logical_checksum": parent_gate["logical_checksum"],
            "experiment_spec_hash": spec_hash, "experiment_run_id": run_id,
            **runner_evidence,
            "metrics_path": str(owner / "metrics.json"),
            "shared_model_artifact": shared_model_artifact,
        }
        manifest["wave4_identity"] = identity
        manifest["manifest_checksum"] = canonical_hash(manifest)
        _write_json(temp / "manifest.json", manifest)
        owner.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, owner)
        _event(ledger_path, spec_hash, run_id, "COMPLETED", model_id, horizon_id, identity,
               (str(owner / "manifest.json"), str(owner / "predictions.csv"), str(owner / "metrics.json")),
               metadata={"metrics_path": str(owner / "metrics.json")})
        return {"status": "COMPLETED", "manifest_path": str(owner / "manifest.json")}
    except (ValueError, FileExistsError):
        if temp.exists(): shutil.rmtree(temp)
        raise
    except BaseException as exc:
        if temp.exists(): shutil.rmtree(temp)
        _event(ledger_path, spec_hash, run_id, "FAILED", model_id, horizon_id, identity,
               error_summary=f"{type(exc).__name__}: {exc}")
        raise


def build_multi_horizon_evidence(component_manifests: Sequence[Path]) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(component_manifests)]
    horizons = {row.get("horizon_id") for row in rows}
    if horizons != set(HORIZON_IDS):
        raise ValueError("All four horizon components are required")
    prediction_rows = {}
    for manifest in rows:
        with Path(manifest["prediction_artifact_path"]).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                prediction_rows.setdefault(row["row_id"], {})[manifest["horizon_id"]] = float(row["selector_score"])
    if not prediction_rows or any(set(values) != set(HORIZON_IDS) for values in prediction_rows.values()):
        raise ValueError("Missing horizon prediction")
    evidence = []
    for row_id, scores in sorted(prediction_rows.items()):
        ordered = [scores[horizon] for horizon in HORIZON_IDS]
        ranks = _percentile_ranks(ordered)
        sign_agreement = max(sum(value >= 0 for value in ordered), sum(value <= 0 for value in ordered)) / 4
        persistence = 0.5 * sign_agreement + 0.5 * max(0.0, 1.0 - float(np.std(ranks)))
        disagreement = (
            0.4 * (1.0 - sign_agreement)
            + 0.4 * (max(ranks) - min(ranks))
            + 0.2 * float(np.sign(ordered[0]) != np.sign(ordered[-1]))
        )
        evidence.append({
            "row_id": row_id, "short_term_score": scores["return_1s"],
            "medium_term_score": 0.5 * (scores["return_5s"] + scores["return_10s"]),
            "long_term_score": scores["return_20s"], "persistence_score": persistence,
            "horizon_disagreement": disagreement, "horizon_scores": scores,
        })
    logical = {
        "contract_version": AGGREGATION_CONTRACT,
        "persistence_equation": "0.5*sign_agreement + 0.5*max(0,1-std(horizon_percentile_ranks))",
        "disagreement_equation": "0.4*sign_disagreement + 0.4*rank_range + 0.2*short_long_sign_conflict",
        "components": [row["manifest_checksum"] for row in rows],
        "rows": evidence,
    }
    logical["logical_checksum"] = canonical_hash(logical)
    return logical


def _fit_and_select(
    model_id, horizon_id, fit_input, interaction_contract, options,
    *, authoritative_context=None, dependency_preflight=None,
):
    if model_id == "huber":
        result = fit_huber_selector(fit_input, **options)
        return result, result.get("predictions", []), result.get("diagnostic_summary", {})
    if model_id == "contextual_elastic_net":
        if interaction_contract is None: raise ValueError("Contextual interaction contract is required")
        result = fit_contextual_elastic_net(fit_input, interaction_contract, **options)
        return result, result.get("predictions", []), result.get("coefficient_diagnostics", {})
    if model_id in {"multi_horizon_ridge", "multi_horizon_elastic_net"}:
        family = "ridge" if model_id.endswith("ridge") else "elastic_net"
        result = fit_multi_horizon_linear_selector(fit_input, model_families=(family,), **options)
        selected = [row for row in result.get("predictions", []) if row["horizon_id"] == horizon_id]
        return result, selected, result.get("diagnostics", {}).get("per_horizon", {}).get(horizon_id, {})
    if model_id in {
        "lightgbm_rank_xendcg", "lightgbm_lambdarank"
    }:
        from core.research.ml.stock_level.lightgbm_production_selector import (
            fit_production_lightgbm_selector,
        )
        result = fit_production_lightgbm_selector(
            fit_input,
            model_id=model_id,
            authoritative_context=authoritative_context or {},
            dependency_preflight=dependency_preflight or {},
            fitted_model_callback=options.get("fitted_model_callback"),
        )
        return (
            result,
            result.get("prediction_contract", {}).get("rows", []),
            result.get("capability_evidence", {}),
        )
    return _fit_ordered_horizon(fit_input, horizon_id, options)


def _fit_ordered_horizon(data, horizon, options):
    rows = list(data["rows"])
    cutoff = str(options.get("training_cutoff"))
    training = [row for row in rows if row["split"] == "TRAINING"
                and row["target_availability_state"][horizon] == "MATURE"
                and row["target_maturity_timestamps"][horizon] <= cutoff
                and row["decision_timestamp"] < cutoff]
    validation = [row for row in rows if row["split"] == "VALIDATION"]
    if len(training) < int(options.get("minimum_training_rows", 5)) or not validation:
        return {"status": "INSUFFICIENT_DATA", "valid": False, "blocking_reasons": ["HORIZON_POPULATION_INADEQUATE"]}, [], {}
    ranker = OrderedLogitRanker(
        bins=5, max_iter=int(options.get("maximum_iterations", 500)),
        tolerance=float(options.get("tolerance", 1e-8)),
    )
    x_train = [row["feature_values"] for row in training]
    ranker.fit(
        x_train, [row["target_values"][horizon] for row in training],
        groups=[row["decision_timestamp"] for row in training],
        row_ids=[row["row_id"] for row in training],
    )
    x_validation = [row["feature_values"] for row in validation]
    probabilities = ranker.predict_proba(x_validation)
    scores = ranker.predict(x_validation)
    predictions = []
    order = np.lexsort((
        np.asarray([row["asset_id"] for row in validation]),
        -np.asarray(scores, dtype=float),
    ))
    ranks = np.empty(len(validation), dtype=int); ranks[order] = np.arange(1, len(validation) + 1)
    for index, row in enumerate(validation):
        item = {
            "row_id": row["row_id"], "asset_id": row["asset_id"],
            "decision_timestamp": row["decision_timestamp"], "score": float(scores[index]),
            "within_date_rank": int(ranks[index]), "horizon_id": horizon,
            "maximum_label_maturity_timestamp": max(row["target_maturity_timestamps"][horizon] for row in training),
            "training_cutoff": cutoff,
            "ordered_logit_predicted_relevance_class": int(np.argmax(probabilities[index])),
        }
        for cls in range(5): item[f"ordered_logit_probability_{cls}"] = float(probabilities[index, cls])
        predictions.append(item)
    result = {
        "status": "READY", "valid": True, "blocking_reasons": [],
        "configuration": dict(options), "predictions": predictions,
    }
    return result, predictions, ranker.diagnostics


def _component_rows(rows, model, horizon, date, identity):
    output = []
    for raw in rows:
        score = raw.get(
            "predicted_return",
            raw.get("continuous_score", raw.get("score", raw.get("raw_score"))),
        )
        row = {
            "row_id": str(raw["row_id"]), "asset_id": str(raw["asset_id"]),
            "symbol": str(raw.get("symbol", raw["asset_id"])), "prediction_date": date,
            "model_id": model, "horizon_id": horizon or "",
            "selector_score": float(score), "deterministic_rank": int(raw.get("within_date_rank", 0)),
            "dataset_identity": identity["dataset_id"],
            "feature_contract_identity": identity["feature_schema"],
            "economic_target_id": identity["economic_target_id"],
            "target_provenance_contract_version": identity[
                "target_provenance_contract_version"
            ],
            "fold_identity": str(raw.get("fold_identity", "")),
        }
        for key, value in raw.items():
            if key.startswith("ordered_logit_"): row[key] = value
        if model == "multi_horizon_ordered_logit":
            row["ordered_logit_expected_relevance"] = float(score)
        output.append(row)
    return sorted(output, key=lambda row: (row["prediction_date"], row["deterministic_rank"], row["asset_id"], row["row_id"]))


def _temporal_fields(data, result, horizon):
    if data.get("contract_version") == "grouped_ranking_dataset_v1":
        training = [
            row for row in data["rows"]
            if row["split_role"] == "TRAINING"
        ]
        return (
            min(str(row["decision_date"]) for row in training),
            str(result["input_contract"]["training_cutoff"]),
            max(
                str(row["target_maturity_timestamp"]) for row in training
            ),
        )
    training = [row for row in data["rows"] if row["split"] == "TRAINING"]
    training_start = min(str(row["decision_timestamp"]) for row in training)
    training_cutoff = str(result.get("configuration", {}).get("training_cutoff") or max(row["decision_timestamp"] for row in training))
    if horizon:
        mature = [row["target_maturity_timestamps"][horizon] for row in training
                  if row["target_availability_state"][horizon] == "MATURE"
                  and row["target_maturity_timestamps"][horizon] <= training_cutoff]
    else:
        mature = [row["target_maturity_timestamp"] for row in training]
    return training_start, training_cutoff, max(mature)


def _event(path, spec, run, status, model, horizon, identity, artifacts=(), **kwargs):
    append_ledger_event(
        path, experiment_spec_hash_value=spec, experiment_run_id=run,
        event_status=status, artifact_kind="SELECTOR_PREDICTION_PARTITION",
        canonical_model_id=model, requested_model_id=model,
        registry_hashes={
            "model_entry_hash": identity["model_entry_hash"],
            "target_entry_hash": identity["target_entry_hash"],
        }, source_commit=_git_commit(), artifact_paths=artifacts,
        metadata={"horizon": horizon, "dataset_id": identity["dataset_id"], **kwargs.pop("metadata", {})},
        **kwargs,
    )


def _compatible(owner, identity):
    try:
        manifest = json.loads((owner / "manifest.json").read_text(encoding="utf-8"))
        return (
            manifest.get("publication_status") == "complete"
            and manifest.get("wave4_identity") == identity
            and _sha256(owner / "predictions.csv") == manifest.get("prediction_checksum")
        )
    except (OSError, json.JSONDecodeError):
        return False


def _percentile_ranks(values):
    order = np.argsort(np.argsort(values))
    return (order / max(len(values) - 1, 1)).tolist()


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
