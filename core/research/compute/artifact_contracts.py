from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

ARTIFACT_MANIFEST_CONTRACT = "compute_artifact_manifest.v1"
FITTED_MODEL_CONTRACT = "compute_fitted_model_artifact.v1"
STAGE_ARTIFACT_CONTRACT = "compute_stage_artifact.v1"
PREDICTION_BINDING_CONTRACT = "compute_prediction_model_binding.v1"


class ArtifactType(str, Enum):
    FITTED_MODEL = "FITTED_MODEL"
    PREPROCESSING_STATE = "PREPROCESSING_STATE"
    TRAINING_CHECKPOINT = "TRAINING_CHECKPOINT"
    PREDICTION_ARTIFACT = "PREDICTION_ARTIFACT"
    DATASET_ARTIFACT = "DATASET_ARTIFACT"
    FEATURE_STORE_ARTIFACT = "FEATURE_STORE_ARTIFACT"
    EVALUATION_ARTIFACT = "EVALUATION_ARTIFACT"
    REPLAY_ARTIFACT = "REPLAY_ARTIFACT"
    DATA_STAGE_ARTIFACT = "DATA_STAGE_ARTIFACT"
    ENSEMBLE_ARTIFACT = "ENSEMBLE_ARTIFACT"
    EXTERNAL_MODEL_REFERENCE = "EXTERNAL_MODEL_REFERENCE"


class ArtifactRole(str, Enum):
    RESEARCH_FOLD_MODEL = "RESEARCH_FOLD_MODEL"
    TRAINING_CHECKPOINT = "TRAINING_CHECKPOINT"
    FINAL_REFIT_MODEL = "FINAL_REFIT_MODEL"
    PROMOTED_MODEL = "PROMOTED_MODEL"
    RESEARCH_PREDICTIONS = "RESEARCH_PREDICTIONS"
    EVALUATION_ONLY = "EVALUATION_ONLY"
    REFERENCE_DATA = "REFERENCE_DATA"


class ArtifactStatus(str, Enum):
    PLANNED = "PLANNED"
    WRITING = "WRITING"
    COMPLETE = "COMPLETE"
    SKIPPED_COMPATIBLE = "SKIPPED_COMPATIBLE"
    INCOMPATIBLE_EXISTING = "INCOMPATIBLE_EXISTING"
    PARTIAL = "PARTIAL"
    CORRUPT = "CORRUPT"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


SERIALIZATION_HANDLERS = {
    "SKLEARN_PIPELINE", "LIGHTGBM_NATIVE", "PYTORCH_STATE_DICT",
    "EXTERNAL_PINNED_MODEL_REFERENCE", "ENSEMBLE_MANIFEST",
    "GENERIC_STAGE_FILES",
}


def canonical_checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def logical_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(manifest))
    for field in (
        "created_timestamp", "logical_checksum", "package_checksum",
        "completion_status", "atomic_publication_evidence", "artifact_root",
        "file_inventory",
        "compatibility_identity",
    ):
        payload.pop(field, None)
    return payload


def manifest_logical_checksum(manifest: Mapping[str, Any]) -> str:
    return canonical_checksum(logical_manifest_payload(manifest))


def build_artifact_manifest(
    *,
    artifact_id: str,
    artifact_type: str,
    artifact_subtype: str,
    artifact_role: str,
    pipeline: str,
    stage: str,
    run_id: str,
    attempt_id: str,
    dataset_input_ancestry: Sequence[Mapping[str, Any]],
    source_artifacts: Sequence[Mapping[str, Any]],
    configuration_identity: str,
    configuration_checksum: str,
    source_git_commit: str,
    serialization_handler: str,
    feature_schema_identity: str | None = None,
    dependency_versions: Mapping[str, str] | None = None,
    claims: Mapping[str, bool] | None = None,
    completion_status: str = ArtifactStatus.PLANNED.value,
    created_timestamp: str | None = None,
    **optional: Any,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "contract_version": ARTIFACT_MANIFEST_CONTRACT,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "artifact_subtype": artifact_subtype,
        "artifact_role": artifact_role,
        "pipeline": pipeline,
        "stage": stage,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "dataset_input_ancestry": [dict(row) for row in dataset_input_ancestry],
        "source_artifacts": [dict(row) for row in source_artifacts],
        "feature_schema_identity": feature_schema_identity,
        "configuration_identity": configuration_identity,
        "configuration_checksum": configuration_checksum,
        "source_git_commit": source_git_commit,
        "python_version": sys.version.split()[0],
        "dependency_versions": {
            "python": sys.version.split()[0],
            **dict(dependency_versions or {}),
        },
        "serialization_handler": serialization_handler,
        "file_inventory": [],
        "completion_status": completion_status,
        "created_timestamp": created_timestamp or datetime.now(timezone.utc).isoformat(),
        "atomic_publication_evidence": {"applicable": True, "published": False},
        "promotion_status": "NOT_PROMOTED",
        "compatibility_identity": "",
        "claims": {
            "fitting_performed": False,
            "prediction_performed": False,
            "evaluation_performed": False,
            "promoted": False,
            "production_data_used": False,
            **dict(claims or {}),
        },
        "resource_evidence": {
            "applicable": False,
            "machine_profile_identity": None,
            "resource_request_identity": None,
            "resource_lease_identity": None,
            "telemetry_artifact_identity": None,
            "resource_summary_identity": None,
        },
        **optional,
    }
    manifest["logical_checksum"] = manifest_logical_checksum(manifest)
    manifest["compatibility_identity"] = manifest["logical_checksum"]
    validate_artifact_manifest(manifest, allow_incomplete=True)
    return manifest


def validate_artifact_manifest(
    manifest: Mapping[str, Any], *, allow_incomplete: bool = False
) -> None:
    required = (
        "artifact_id", "artifact_type", "artifact_subtype", "artifact_role",
        "pipeline", "stage", "run_id", "attempt_id",
        "configuration_identity", "configuration_checksum",
        "source_git_commit", "python_version", "serialization_handler",
        "claims", "resource_evidence",
    )
    if manifest.get("contract_version") != ARTIFACT_MANIFEST_CONTRACT:
        raise ValueError("Unsupported artifact manifest contract")
    missing = [field for field in required if manifest.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Artifact manifest fields missing: {','.join(missing)}")
    if manifest["artifact_type"] not in {row.value for row in ArtifactType}:
        raise ValueError("Unsupported artifact type")
    if manifest["artifact_role"] not in {row.value for row in ArtifactRole}:
        raise ValueError("Unsupported artifact role")
    if manifest["serialization_handler"] not in SERIALIZATION_HANDLERS:
        raise ValueError("Unsupported serialization handler")
    if not manifest.get("dataset_input_ancestry") and not manifest.get("source_artifacts"):
        raise ValueError("Artifact ancestry is required")
    claims = manifest.get("claims")
    if not isinstance(claims, Mapping):
        raise ValueError("Artifact claims are required")
    if (
        manifest["artifact_role"] == ArtifactRole.RESEARCH_FOLD_MODEL.value
        and claims.get("promoted")
    ):
        raise ValueError("Research fold model cannot claim promotion")
    if (
        manifest["artifact_role"] == ArtifactRole.EVALUATION_ONLY.value
        and claims.get("fitting_performed")
    ):
        raise ValueError("Evaluation-only artifact cannot claim fitting")
    if manifest["artifact_role"] != ArtifactRole.PROMOTED_MODEL.value and (
        claims.get("promoted") or manifest.get("promotion_status") == "PROMOTED"
    ):
        raise ValueError("Artifact role is not eligible for promoted status")
    status = manifest.get("completion_status")
    if status not in {row.value for row in ArtifactStatus}:
        raise ValueError("Unsupported artifact status")
    if not allow_incomplete and status not in {
        ArtifactStatus.COMPLETE.value, ArtifactStatus.SKIPPED_COMPATIBLE.value
    }:
        raise ValueError("Artifact package is not complete")
    if manifest.get("logical_checksum") != manifest_logical_checksum(manifest):
        raise ValueError("Artifact manifest logical checksum mismatch")
    _validate_resource_evidence(manifest.get("resource_evidence"))


def validate_prediction_binding(
    prediction_manifest: Mapping[str, Any],
    fitted_model_manifest: Mapping[str, Any] | None,
) -> None:
    if fitted_model_manifest is None:
        raise ValueError("Prediction requires a fitted-model artifact")
    validate_artifact_manifest(fitted_model_manifest)
    if fitted_model_manifest.get("artifact_type") != ArtifactType.FITTED_MODEL.value:
        raise ValueError("Prediction binding target is not a fitted model")
    binding = prediction_manifest.get("prediction_model_binding")
    required = (
        "fitted_model_artifact_identity", "fitted_model_artifact_checksum",
        "preprocessing_identity", "input_population_checksum",
        "output_population_checksum", "prediction_schema",
        "prediction_count", "source_git_commit",
    )
    if not isinstance(binding, Mapping) or any(
        binding.get(field) in (None, "") for field in required
    ):
        raise ValueError("Prediction-to-model binding is incomplete")
    if binding["fitted_model_artifact_identity"] != fitted_model_manifest["artifact_id"]:
        raise ValueError("Prediction fitted-model identity mismatch")
    if binding["fitted_model_artifact_checksum"] != fitted_model_manifest["logical_checksum"]:
        raise ValueError("Prediction fitted-model checksum mismatch")


def build_stage_artifact_manifest(
    *, stage_owner: str, output_counts: Mapping[str, int],
    schema_identity: str, coverage_evidence: Mapping[str, Any],
    resumability_evidence: Mapping[str, Any], **common: Any,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        serialization_handler="GENERIC_STAGE_FILES",
        stage_contract_version=STAGE_ARTIFACT_CONTRACT,
        stage_metadata={
            "stage_owner": stage_owner,
            "output_counts": dict(output_counts),
            "schema_identity": schema_identity,
            "coverage_evidence": dict(coverage_evidence),
            "resumability_evidence": dict(resumability_evidence),
            "no_model_applicability": True,
        },
        **common,
    )
    validate_stage_artifact_manifest(manifest)
    return manifest


def validate_stage_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    metadata = manifest.get("stage_metadata")
    if manifest.get("stage_contract_version") != STAGE_ARTIFACT_CONTRACT:
        raise ValueError("Stage artifact contract mismatch")
    if not isinstance(metadata, Mapping) or any(
        not metadata.get(field)
        for field in (
            "stage_owner", "output_counts", "schema_identity",
            "coverage_evidence", "resumability_evidence",
        )
    ):
        raise ValueError("Stage artifact metadata is incomplete")


def _validate_resource_evidence(evidence: Any) -> None:
    if not isinstance(evidence, Mapping) or "applicable" not in evidence:
        raise ValueError("Resource evidence applicability is required")
    if evidence["applicable"] and not evidence.get("machine_profile_identity"):
        raise ValueError("Applicable resource evidence requires a machine profile")
