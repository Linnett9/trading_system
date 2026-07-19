from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.compute.artifact_contracts import (
    FITTED_MODEL_CONTRACT,
    PREDICTION_BINDING_CONTRACT,
    ArtifactRole,
    ArtifactType,
    build_artifact_manifest,
    canonical_checksum,
    validate_prediction_binding,
)
from core.research.compute.artifact_storage import (
    publish_artifact_package,
    validate_artifact_package,
)
from core.research.compute.model_artifacts import (
    validate_model_artifact_manifest,
)
from core.research.ml.stock_level.selector_target_identity import (
    validate_selector_prediction_target_binding,
    validate_selector_target_identity,
)


SUPPORTED_MODELS = {
    "lightgbm_rank_xendcg": "rank_xendcg",
    "lightgbm_lambdarank": "lambdarank",
}
PREPROCESSING_NOT_APPLICABLE = "NOT_APPLICABLE_DIRECT_FEATURE_MATRIX"


def publish_selector_lightgbm_model_package(
    *,
    component_root: Path,
    published_component_root: Path,
    estimator: Any,
    feature_order: Sequence[str],
    feature_schema_identity: str,
    feature_schema_checksum: str,
    source_schema_guarantee_identity: str,
    configuration: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    group_evidence: Mapping[str, Any],
    ranking_label_evidence: Mapping[str, Any],
    model_id: str,
    prediction_path: Path,
    prediction_schema: Sequence[str],
    prediction_count: int,
    output_population_checksum: str,
    campaign_identity: str,
    plan_job_identity: str,
    component_identity: str,
    component_runner: str,
    runtime_owner: str,
    decision_date: str,
    horizon_identity: str,
    training_row_artifact_identity: str,
    prediction_row_artifact_identity: str,
    input_package_identity: str,
    input_population_checksum: str,
    source_git_commit: str,
    lightgbm_version: str,
    economic_target_id: str,
    target_provenance_contract_version: str,
) -> dict[str, Any]:
    target_identity = validate_selector_target_identity(
        economic_target_id=economic_target_id,
        target_provenance_contract_version=target_provenance_contract_version,
    )
    objective = SUPPORTED_MODELS.get(model_id)
    if objective is None:
        raise ValueError("LIGHTGBM_CONFIGURATION_MISMATCH")
    ordered_features = [str(value) for value in feature_order]
    if not ordered_features or len(ordered_features) != len(set(ordered_features)):
        raise ValueError("FEATURE_ORDER_MISSING")
    if feature_schema_checksum in (None, ""):
        raise ValueError("FEATURE_ORDER_MISSING")
    parameters = dict(configuration.get("parameters") or {})
    if (
        parameters.get("objective") != objective
        or parameters.get("n_jobs") != 1
        or parameters.get("metric") in (None, "")
    ):
        raise ValueError("LIGHTGBM_CONFIGURATION_MISMATCH")

    group_metadata = _validated_group_evidence(group_evidence)
    label_metadata = _validated_label_evidence(ranking_label_evidence)
    gain_policy: Mapping[str, Any] | str
    if objective == "lambdarank":
        gain_policy = dict(configuration.get("gain_policy") or {})
        _validate_gain_policy(gain_policy)
    else:
        if configuration.get("gain_policy") is not None:
            raise ValueError("LAMBDARANK_GAIN_POLICY_MISMATCH")
        gain_policy = "NOT_APPLICABLE_RANK_XENDCG"

    native_bytes, native_owner, wrapper_identity, num_trees, best_iteration = (
        _native_model_bytes(estimator)
    )
    native_checksum = hashlib.sha256(native_bytes).hexdigest()
    prediction_checksum, actual_schema, prediction_rows = _prediction_evidence(
        prediction_path
    )
    actual_count = len(prediction_rows)
    if actual_schema != list(prediction_schema):
        raise ValueError("PREDICTION_SCHEMA_MISMATCH")
    if actual_count != prediction_count:
        raise ValueError("PREDICTION_COUNT_MISMATCH")
    forbidden = {"label", "relevance", "ranking_label"}
    if forbidden.intersection(actual_schema):
        raise ValueError("PREDICTION_SCHEMA_MISMATCH")
    required_prediction = {
        "row_id", "asset_id", "symbol", "prediction_date", "model_id",
        "horizon_id", "selector_score", "deterministic_rank",
    }
    if not required_prediction.issubset(actual_schema):
        raise ValueError("PREDICTION_SCHEMA_MISMATCH")
    if (
        any(
            row["prediction_date"] != decision_date
            or row["model_id"] != model_id
            or row["horizon_id"] != horizon_identity
            or not row["symbol"]
            for row in prediction_rows
        )
        or len({row["row_id"] for row in prediction_rows}) != actual_count
        or prediction_rows != sorted(
            prediction_rows,
            key=lambda row: (
                row["prediction_date"],
                int(row["deterministic_rank"]),
                row["asset_id"],
                row["row_id"],
            ),
        )
    ):
        raise ValueError("PREDICTION_SCHEMA_MISMATCH")
    if canonical_checksum(
        [row["row_id"] for row in prediction_rows]
    ) != str(output_population_checksum).lower():
        raise ValueError("PREDICTION_COUNT_MISMATCH")

    run_id = os.environ.get("SELECTOR_COMPUTE_RUN_ID") or component_identity
    attempt_id = (
        os.environ.get("SELECTOR_COMPUTE_ATTEMPT_ID")
        or canonical_checksum({
            "component": component_identity,
            "plan_job": plan_job_identity,
        })
    )
    run_identity = os.environ.get("SELECTOR_COMPUTE_RUN_IDENTITY")
    configuration_checksum = canonical_checksum(configuration)
    feature_order_checksum = canonical_checksum(ordered_features)
    group_identity = canonical_checksum(group_metadata)
    label_identity = canonical_checksum(label_metadata)
    gain_identity = (
        gain_policy["gain_checksum"]
        if isinstance(gain_policy, Mapping)
        else str(gain_policy)
    )
    model_id_value = "selector-lightgbm:" + canonical_checksum({
        "campaign": campaign_identity,
        "job": plan_job_identity,
        "component": component_identity,
        "model": model_id,
        "native_checksum": native_checksum,
        "feature_order": feature_order_checksum,
        "groups": group_identity,
        "labels": label_identity,
        "gain": gain_identity,
        **target_identity.as_dict(),
    })
    model_metadata = {
        "model_id": model_id,
        "implementation_owner": (
            "core.research.ml.stock_level.lightgbm_production_selector:"
            "fit_production_lightgbm_selector"
        ),
        "model_family": objective,
        "model_configuration": dict(configuration),
        "model_configuration_checksum": configuration_checksum,
        "random_seed": 1729,
        "training_boundary": {
            "training_cutoff": input_contract["training_cutoff"],
            "maximum_training_label_maturity_timestamp": input_contract[
                "maximum_training_label_maturity_timestamp"
            ],
            "fold_identity": input_contract["split_identity"],
        },
        "training_population_checksum": input_contract[
            "training_population_checksum"
        ],
        "target_horizon_identity": horizon_identity,
        **target_identity.as_dict(),
        "feature_order": ordered_features,
        "feature_order_checksum": feature_order_checksum,
        "feature_profile_identity": feature_schema_identity,
        "feature_profile_checksum": feature_schema_checksum,
        "categorical_feature_names": [],
        "categorical_feature_indices": [],
        "categorical_feature_identity": canonical_checksum([]),
        "feature_validation_policy": "EXACT_ORDERED_NATIVE_FEATURES",
        "preprocessing_identity": PREPROCESSING_NOT_APPLICABLE,
        "lightgbm_version": lightgbm_version,
        "native_format": "LIGHTGBM_MODEL_TEXT",
        "native_model_file": "model/lightgbm_model.txt",
        "native_model_sha256": native_checksum,
        "native_booster_owner": native_owner,
        "model_wrapper_identity": wrapper_identity,
        "objective": objective,
        "metrics": [parameters["metric"]],
        "booster_type": parameters.get("boosting_type", "gbdt"),
        "num_trees": num_trees,
        "best_iteration": best_iteration,
        "prediction_iteration_policy": (
            "BEST_ITERATION_IF_POSITIVE_ELSE_ALL_FITTED_TREES"
        ),
        "prediction_raw_score": False,
        "prediction_score_normalisation": "NONE",
        "configured_iterations": parameters.get("n_estimators"),
        "label_gain_policy": gain_policy,
        "label_gain_policy_identity": gain_identity,
        "grouped_ranking_configuration": group_metadata,
        "group_query_identity": group_identity,
        "ranking_label_evidence": label_metadata,
        "ranking_label_identity": label_identity,
        "campaign_identity": campaign_identity,
        "plan_job_identity": plan_job_identity,
        "component_identity": component_identity,
        "component_runner": component_runner,
        "runtime_owner": runtime_owner,
        "decision_date": decision_date,
        "horizon_identity": horizon_identity,
        "training_row_artifact_identity": training_row_artifact_identity,
        "prediction_row_artifact_identity": prediction_row_artifact_identity,
        "input_package_identity": input_package_identity,
        "source_schema_guarantee_identity": source_schema_guarantee_identity,
        "source_git_commit": source_git_commit,
        "run_identity": run_identity,
        "promotion_state": False,
    }
    dependencies = _dependencies(lightgbm_version)
    model_template = build_artifact_manifest(
        artifact_id=model_id_value,
        artifact_type=ArtifactType.FITTED_MODEL.value,
        artifact_subtype="SELECTOR_LIGHTGBM_NATIVE_MODEL",
        artifact_role=ArtifactRole.RESEARCH_FOLD_MODEL.value,
        pipeline="selector",
        stage="stage10_component",
        run_id=run_id,
        attempt_id=attempt_id,
        dataset_input_ancestry=[{
            "identity": input_package_identity,
            "checksum": input_population_checksum,
        }],
        source_artifacts=[{
            "identity": training_row_artifact_identity,
            "checksum": input_contract["training_population_checksum"],
        }],
        configuration_identity=configuration_checksum,
        configuration_checksum=configuration_checksum,
        source_git_commit=source_git_commit,
        serialization_handler="LIGHTGBM_NATIVE",
        feature_schema_identity=source_schema_guarantee_identity,
        dependency_versions=dependencies,
        claims={
            "fitting_performed": True,
            "prediction_performed": True,
            "evaluation_performed": False,
            "promoted": False,
            "production_data_used": False,
        },
        fitted_model_contract_version=FITTED_MODEL_CONTRACT,
        model_metadata=model_metadata,
    )
    model_root = component_root / "shared_model_artifact" / "model"
    model_status, model_manifest = publish_artifact_package(
        model_root,
        model_template,
        {
            "model/lightgbm_model.txt": native_bytes,
            "metadata/feature_schema.json": _json_bytes({
                "feature_order": ordered_features,
                "feature_order_checksum": feature_order_checksum,
                "feature_profile_identity": feature_schema_identity,
                "feature_profile_checksum": feature_schema_checksum,
            }),
            "metadata/ranking_groups.json": _json_bytes(group_metadata),
            "metadata/ranking_labels.json": _json_bytes(label_metadata),
            "metadata/configuration.json": _json_bytes(dict(configuration)),
            "metadata/dependency_versions.json": _json_bytes(dependencies),
        },
    )
    validate_lightgbm_model_package(model_root)

    binding = {
        "contract_version": PREDICTION_BINDING_CONTRACT,
        "fitted_model_artifact_identity": model_manifest["artifact_id"],
        "fitted_model_artifact_checksum": model_manifest["logical_checksum"],
        "fitted_model_package_checksum": model_manifest["package_checksum"],
        "preprocessing_identity": PREPROCESSING_NOT_APPLICABLE,
        "feature_order_identity": feature_order_checksum,
        "group_query_identity": group_identity,
        "ranking_label_identity": label_identity,
        "gain_policy_identity": gain_identity,
        "input_population_checksum": input_population_checksum,
        "output_population_checksum": output_population_checksum,
        "prediction_artifact_identity": (
            "selector-lightgbm-prediction:" + canonical_checksum({
                "component": component_identity,
                "checksum": prediction_checksum,
            })
        ),
        "prediction_artifact_checksum": prediction_checksum,
        "prediction_schema": actual_schema,
        "prediction_count": actual_count,
        "campaign_identity": campaign_identity,
        "component_identity": component_identity,
        "plan_job_identity": plan_job_identity,
        "decision_date": decision_date,
        "horizon": horizon_identity,
        **target_identity.as_dict(),
        "source_git_commit": source_git_commit,
    }
    prediction_template = build_artifact_manifest(
        artifact_id=binding["prediction_artifact_identity"],
        artifact_type=ArtifactType.PREDICTION_ARTIFACT.value,
        artifact_subtype="SELECTOR_LIGHTGBM_PREDICTION_BINDING",
        artifact_role=ArtifactRole.RESEARCH_PREDICTIONS.value,
        pipeline="selector",
        stage="stage10_component",
        run_id=run_id,
        attempt_id=attempt_id,
        dataset_input_ancestry=[{
            "identity": input_package_identity,
            "checksum": input_population_checksum,
        }],
        source_artifacts=[{
            "identity": model_manifest["artifact_id"],
            "checksum": model_manifest["package_checksum"],
        }, {
            "identity": prediction_row_artifact_identity,
            "checksum": prediction_checksum,
        }],
        configuration_identity=configuration_checksum,
        configuration_checksum=configuration_checksum,
        source_git_commit=source_git_commit,
        serialization_handler="GENERIC_STAGE_FILES",
        feature_schema_identity=source_schema_guarantee_identity,
        dependency_versions=dependencies,
        claims={
            "fitting_performed": False,
            "prediction_performed": True,
            "evaluation_performed": False,
            "promoted": False,
            "production_data_used": False,
        },
        prediction_model_binding=binding,
    )
    prediction_root = component_root / "shared_model_artifact" / "prediction"
    prediction_status, prediction_manifest = publish_artifact_package(
        prediction_root,
        prediction_template,
        {"metadata/prediction_binding.json": _json_bytes(binding)},
    )
    validate_prediction_binding(prediction_manifest, model_manifest)
    validate_selector_prediction_target_binding(binding, model_metadata)
    public_shared = published_component_root / "shared_model_artifact"
    return {
        "completion_status": "COMPLETE",
        "compatible_skip_status": (
            "SKIPPED_COMPATIBLE"
            if model_status == prediction_status == "SKIPPED_COMPATIBLE"
            else "COMPLETE"
        ),
        "artifact_identity": model_manifest["artifact_id"],
        "package_checksum": model_manifest["package_checksum"],
        "preprocessing_identity": PREPROCESSING_NOT_APPLICABLE,
        "feature_order_identity": feature_order_checksum,
        "group_query_identity": group_identity,
        "ranking_label_identity": label_identity,
        "gain_policy_identity": gain_identity,
        "prediction_binding_identity": canonical_checksum(binding),
        "prediction_artifact_identity": prediction_manifest["artifact_id"],
        "model_package_path": str(public_shared / "model"),
        "prediction_package_path": str(public_shared / "prediction"),
    }


def validate_lightgbm_model_package(model_root: Path) -> Mapping[str, Any]:
    try:
        manifest = validate_artifact_package(model_root)
        validate_model_artifact_manifest(manifest)
    except (OSError, ValueError) as exc:
        raise ValueError("CORRUPT_MODEL_ARTIFACT") from exc
    metadata = manifest["model_metadata"]
    if manifest["serialization_handler"] != "LIGHTGBM_NATIVE":
        raise ValueError("INCOMPATIBLE_MODEL_ARTIFACT")
    native = model_root / str(metadata.get("native_model_file") or "")
    if not native.is_file():
        raise ValueError("LIGHTGBM_NATIVE_MODEL_MISSING")
    if hashlib.sha256(native.read_bytes()).hexdigest() != metadata.get(
        "native_model_sha256"
    ):
        raise ValueError("CORRUPT_MODEL_ARTIFACT")
    _validated_group_evidence(metadata.get("grouped_ranking_configuration") or {})
    _validated_label_evidence(metadata.get("ranking_label_evidence") or {})
    if metadata["objective"] == "lambdarank":
        _validate_gain_policy(metadata.get("label_gain_policy") or {})
    return manifest


def resolve_lightgbm_package(
    *,
    component_root: Path,
    job: Mapping[str, Any],
    component_manifest: Mapping[str, Any],
    run_identity: str,
) -> Mapping[str, Any] | None:
    model_root = component_root / "shared_model_artifact" / "model"
    prediction_root = component_root / "shared_model_artifact" / "prediction"
    if not model_root.exists() or not prediction_root.exists():
        return None
    model = validate_lightgbm_model_package(model_root)
    try:
        prediction = validate_artifact_package(prediction_root)
        validate_prediction_binding(prediction, model)
    except (OSError, ValueError) as exc:
        raise ValueError("PREDICTION_BINDING_MISSING") from exc
    metadata = model["model_metadata"]
    if (
        metadata.get("model_id") != job.get("model_id")
        or metadata.get("campaign_identity") != job.get("campaign_identity")
        or metadata.get("plan_job_identity") != job.get("job_id")
        or metadata.get("decision_date") != job.get("prediction_date")
        or metadata.get("component_identity")
        != component_manifest.get("wave4_identity", {}).get(
            "component_identity", metadata.get("component_identity")
        )
        or (
            metadata.get("run_identity") is not None
            and metadata.get("run_identity") != run_identity
        )
    ):
        raise ValueError("INCOMPATIBLE_MODEL_ARTIFACT")
    binding = prediction["prediction_model_binding"]
    return {
        "completion_status": "COMPLETE",
        "compatible_skip_status": "SKIPPED_COMPATIBLE",
        "artifact_identity": model["artifact_id"],
        "package_checksum": model["package_checksum"],
        "preprocessing_identity": metadata["preprocessing_identity"],
        "feature_order_identity": metadata["feature_order_checksum"],
        "group_query_identity": metadata["group_query_identity"],
        "ranking_label_identity": metadata["ranking_label_identity"],
        "gain_policy_identity": metadata["label_gain_policy_identity"],
        "prediction_binding_identity": canonical_checksum(binding),
        "prediction_artifact_identity": prediction["artifact_id"],
        "model_package_path": str(model_root),
        "prediction_package_path": str(prediction_root),
    }


def _native_model_bytes(
    estimator: Any,
) -> tuple[bytes, str, str, int, int]:
    booster = estimator if callable(
        getattr(estimator, "model_to_string", None)
    ) else getattr(estimator, "booster_", None)
    if booster is None or not callable(getattr(booster, "model_to_string", None)):
        raise ValueError("LIGHTGBM_NATIVE_MODEL_MISSING")
    try:
        text = booster.model_to_string()
        native = text.encode("utf-8")
    except Exception as exc:
        raise ValueError("LIGHTGBM_NATIVE_SERIALIZATION_FAILED") from exc
    if not native:
        raise ValueError("LIGHTGBM_NATIVE_SERIALIZATION_FAILED")
    tree_getter = getattr(booster, "num_trees", None)
    num_trees = int(tree_getter()) if callable(tree_getter) else int(
        getattr(estimator, "n_estimators_", 0) or 0
    )
    return (
        native,
        f"{type(booster).__module__}.{type(booster).__qualname__}",
        f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        num_trees,
        int(getattr(estimator, "best_iteration_", 0) or 0),
    )


def _validated_group_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    sizes = [int(size) for size in result.get("training_group_sizes") or ()]
    dates = [str(date) for date in result.get("training_group_dates") or ()]
    if (
        not sizes or len(sizes) != len(dates) or any(size <= 0 for size in sizes)
        or sum(sizes) != int(result.get("training_group_row_count") or -1)
        or not result.get("ordered_training_membership_checksum")
    ):
        raise ValueError("RANKING_GROUP_EVIDENCE_MISSING")
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("RANKING_GROUP_CHECKSUM_MISMATCH")
    result.update({
        "group_size_vector_checksum": canonical_checksum(sizes),
        "group_query_checksum": canonical_checksum({
            "dates": dates, "sizes": sizes,
        }),
        "group_count": len(sizes),
        "minimum_group_size": min(sizes),
        "maximum_group_size": max(sizes),
    })
    return result


def _validated_label_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    levels = result.get("ordered_relevance_levels")
    distribution = result.get("label_distribution")
    if (
        not isinstance(levels, Sequence) or not levels
        or list(levels) != sorted(set(levels))
        or not isinstance(distribution, Mapping)
        or sum(int(count) for count in distribution.values())
        != int(result.get("label_count") or -1)
        or not result.get("training_only_label_claim")
        or not result.get("published_prediction_rows_unlabeled")
    ):
        raise ValueError("RANKING_LABEL_EVIDENCE_MISSING")
    return result


def _validate_gain_policy(value: Mapping[str, Any]) -> None:
    levels = list(value.get("ordered_relevance_levels") or ())
    gains = list(value.get("gain_values") or ())
    logical = {key: item for key, item in value.items() if key != "gain_checksum"}
    if (
        not levels or len(levels) != len(gains)
        or levels != sorted(set(levels))
        or str(value.get("gain_checksum") or "").lower()
        != canonical_checksum(logical)
    ):
        raise ValueError("LAMBDARANK_GAIN_POLICY_MISMATCH")


def _prediction_evidence(
    path: Path,
) -> tuple[str, list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError("PREDICTION_BINDING_MISSING")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        schema = list(reader.fieldnames or ())
        rows = list(reader)
    return hashlib.sha256(path.read_bytes()).hexdigest(), schema, rows


def _dependencies(lightgbm_version: str) -> dict[str, str]:
    def version(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "UNAVAILABLE"

    return {
        "python": platform.python_version(),
        "lightgbm": lightgbm_version,
        "numpy": version("numpy"),
        "scikit-learn": version("scikit-learn"),
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
