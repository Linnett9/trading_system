from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_contracts import (
    FITTED_MODEL_CONTRACT,
    ArtifactRole,
    ArtifactType,
    canonical_checksum,
    validate_artifact_manifest,
)


def validate_model_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    validate_artifact_manifest(
        manifest,
        allow_incomplete=manifest.get("completion_status") not in {
            "COMPLETE", "SKIPPED_COMPATIBLE"
        },
    )
    handler = manifest.get("serialization_handler")
    metadata = manifest.get("model_metadata")
    if handler == "EXTERNAL_PINNED_MODEL_REFERENCE":
        _validate_external(metadata)
        return
    if manifest.get("fitted_model_contract_version") != FITTED_MODEL_CONTRACT:
        raise ValueError("Fitted-model artifact contract mismatch")
    if manifest.get("artifact_type") not in {
        ArtifactType.FITTED_MODEL.value,
        ArtifactType.TRAINING_CHECKPOINT.value,
        ArtifactType.ENSEMBLE_ARTIFACT.value,
    }:
        raise ValueError("Invalid model artifact type")
    if not isinstance(metadata, Mapping):
        raise ValueError("Model metadata is required")
    required = (
        "model_id", "implementation_owner", "model_family",
        "model_configuration", "model_configuration_checksum", "random_seed",
        "training_boundary", "training_population_checksum",
        "target_horizon_identity", "feature_order",
        "feature_order_checksum", "preprocessing_identity",
    )
    if any(metadata.get(field) in (None, "") for field in required):
        raise ValueError("Fitted model metadata or preprocessing/schema is incomplete")
    if metadata["feature_order_checksum"] != canonical_checksum(
        list(metadata["feature_order"])
    ):
        raise ValueError("Model feature order checksum mismatch")
    if handler == "SKLEARN_PIPELINE":
        _validate_sklearn(metadata)
    elif handler == "LIGHTGBM_NATIVE":
        _validate_lightgbm(metadata)
    elif handler == "PYTORCH_STATE_DICT":
        _validate_pytorch(metadata, manifest)
    elif handler == "ENSEMBLE_MANIFEST":
        _validate_ensemble(metadata)
    else:
        raise ValueError("Handler is not valid for a model artifact")


def compare_artifact_compatibility(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    fields = (
        "artifact_type", "artifact_subtype", "artifact_role",
        "serialization_handler", "logical_checksum", "package_checksum",
    )
    reasons = tuple(
        f"{field.upper()}_MISMATCH"
        for field in fields if expected.get(field) != actual.get(field)
    )
    return not reasons, reasons


def inspect_model_metadata(package_root: Path) -> dict[str, Any]:
    from .artifact_storage import validate_artifact_package

    manifest = validate_artifact_package(package_root)
    validate_model_artifact_manifest(manifest)
    return {
        key: manifest.get(key)
        for key in (
            "artifact_id", "artifact_type", "artifact_role",
            "serialization_handler", "model_metadata",
            "dependency_versions", "logical_checksum", "package_checksum",
        )
    }


def read_trusted_model_bytes(
    package_root: Path, relative_path: str, *, trusted_artifact: bool
) -> bytes:
    if not trusted_artifact:
        raise PermissionError(
            "Model deserialization requires explicit trusted-artifact policy"
        )
    from .artifact_storage import validate_artifact_package

    manifest = validate_artifact_package(package_root)
    validate_model_artifact_manifest(manifest)
    owned = _owned_path(package_root, relative_path)
    declared = {
        row["relative_path"] for row in manifest.get("file_inventory", [])
    }
    if relative_path not in declared:
        raise ValueError("Model path is not owned by the artifact package")
    return owned.read_bytes()


def _validate_sklearn(metadata: Mapping[str, Any]) -> None:
    if not metadata.get("sklearn_version"):
        raise ValueError("Sklearn version is required")
    if not (
        metadata.get("preprocessing_embedded")
        or metadata.get("preprocessing_file")
    ):
        raise ValueError("Sklearn package requires fitted preprocessing")
    if metadata.get("fitted_feature_count") != len(metadata["feature_order"]):
        raise ValueError("Sklearn fitted feature count mismatch")


def _validate_lightgbm(metadata: Mapping[str, Any]) -> None:
    required = (
        "lightgbm_version", "native_format", "objective", "metrics",
        "num_trees", "label_gain_policy", "grouped_ranking_configuration",
    )
    if any(metadata.get(field) in (None, "") for field in required):
        raise ValueError("LightGBM native metadata is incomplete")


def _validate_pytorch(
    metadata: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    required = (
        "pytorch_version", "architecture_owner", "architecture_configuration",
        "state_dict_file", "dtype", "device_at_save", "channel_order",
        "lookback", "tensor_schema", "target_heads",
    )
    if any(metadata.get(field) in (None, "") for field in required):
        raise ValueError("PyTorch state_dict metadata is incomplete")
    if (
        manifest.get("artifact_role") == ArtifactRole.RESEARCH_FOLD_MODEL.value
        and metadata.get("optimizer_state_required")
    ):
        raise ValueError("Inference fold model cannot require optimiser state")


def _validate_external(metadata: Any) -> None:
    required = (
        "provider", "repository", "revision", "tokenizer_revision",
        "configuration_checksum", "scoring_configuration",
    )
    if not isinstance(metadata, Mapping) or any(
        metadata.get(field) in (None, "") for field in required
    ):
        raise ValueError("External pinned model reference is incomplete")


def _validate_ensemble(metadata: Mapping[str, Any]) -> None:
    members = metadata.get("ordered_members")
    weights = metadata.get("weights")
    if (
        not isinstance(members, Sequence) or not members
        or not isinstance(weights, Sequence) or len(members) != len(weights)
        or not metadata.get("combination_rule")
        or not metadata.get("required_prediction_schema")
    ):
        raise ValueError("Ensemble metadata is incomplete")
    if any(
        not isinstance(row, Mapping)
        or not row.get("artifact_identity")
        or not row.get("artifact_checksum")
        for row in members
    ):
        raise ValueError("Ensemble member binding is incomplete")


def _owned_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Artifact path escapes owned directory") from exc
    return path
