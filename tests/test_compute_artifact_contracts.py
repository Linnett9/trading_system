from __future__ import annotations

from copy import deepcopy

import pytest

from core.research.compute.artifact_contracts import (
    ArtifactRole,
    ArtifactType,
    build_artifact_manifest,
    build_stage_artifact_manifest,
    manifest_logical_checksum,
    validate_artifact_manifest,
    validate_prediction_binding,
)


def common(**overrides):
    values = {
        "artifact_id": "artifact-1",
        "artifact_type": ArtifactType.DATASET_ARTIFACT.value,
        "artifact_subtype": "STRICT_OOS_ROWS",
        "artifact_role": ArtifactRole.REFERENCE_DATA.value,
        "pipeline": "selector",
        "stage": "inputs",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "dataset_input_ancestry": [{"identity": "source", "checksum": "a" * 64}],
        "source_artifacts": [],
        "configuration_identity": "config-v1",
        "configuration_checksum": "b" * 64,
        "source_git_commit": "commit",
        "serialization_handler": "GENERIC_STAGE_FILES",
        "feature_schema_identity": "schema-v1",
    }
    values.update(overrides)
    return build_artifact_manifest(**values)


def test_common_manifest_roles_ancestry_and_resource_evidence() -> None:
    manifest = common()
    validate_artifact_manifest(manifest, allow_incomplete=True)
    with pytest.raises(ValueError, match="ancestry"):
        common(dataset_input_ancestry=[], source_artifacts=[])
    with pytest.raises(ValueError, match="Research fold"):
        common(
            artifact_type=ArtifactType.FITTED_MODEL.value,
            artifact_role=ArtifactRole.RESEARCH_FOLD_MODEL.value,
            claims={"promoted": True},
        )
    with pytest.raises(ValueError, match="Evaluation-only"):
        common(
            artifact_type=ArtifactType.EVALUATION_ARTIFACT.value,
            artifact_role=ArtifactRole.EVALUATION_ONLY.value,
            claims={"fitting_performed": True},
        )
    resource = common()
    resource["resource_evidence"] = {
        "applicable": True, "machine_profile_identity": "profile",
        "resource_request_identity": "request", "resource_lease_identity": "lease",
        "telemetry_artifact_identity": "telemetry",
        "resource_summary_identity": "summary",
    }
    resource["logical_checksum"] = manifest_logical_checksum(resource)
    validate_artifact_manifest(resource, allow_incomplete=True)


def test_stage_artifact_has_no_model_requirement() -> None:
    stage = build_stage_artifact_manifest(
        stage_owner="daily_builder",
        output_counts={"rows": 10, "partitions": 2},
        schema_identity="daily-v2",
        coverage_evidence={"start": "2020-01-01", "end": "2020-01-10"},
        resumability_evidence={"resume_key": "partition"},
        **{
            key: value for key, value in {
                "artifact_id": "stage-1",
                "artifact_type": ArtifactType.DATA_STAGE_ARTIFACT.value,
                "artifact_subtype": "ARCHIVE_FINALISATION",
                "artifact_role": ArtifactRole.REFERENCE_DATA.value,
                "pipeline": "data",
                "stage": "finalise",
                "run_id": "run",
                "attempt_id": "attempt",
                "dataset_input_ancestry": [{"identity": "raw", "checksum": "1"}],
                "source_artifacts": [],
                "configuration_identity": "cfg",
                "configuration_checksum": "2",
                "source_git_commit": "commit",
            }.items()
        },
    )
    assert stage["stage_metadata"]["no_model_applicability"] is True
    assert "model_metadata" not in stage


def test_prediction_requires_exact_completed_model_binding() -> None:
    model = common(
        artifact_id="model-1",
        artifact_type=ArtifactType.FITTED_MODEL.value,
        artifact_role=ArtifactRole.RESEARCH_FOLD_MODEL.value,
        serialization_handler="SKLEARN_PIPELINE",
        completion_status="COMPLETE",
    )
    prediction = common(
        artifact_id="predictions",
        artifact_type=ArtifactType.PREDICTION_ARTIFACT.value,
        artifact_role=ArtifactRole.RESEARCH_PREDICTIONS.value,
        claims={"prediction_performed": True},
    )
    prediction["prediction_model_binding"] = {
        "contract_version": "compute_prediction_model_binding.v1",
        "fitted_model_artifact_identity": "model-1",
        "fitted_model_artifact_checksum": model["logical_checksum"],
        "preprocessing_identity": "prep",
        "input_population_checksum": "input",
        "output_population_checksum": "output",
        "prediction_schema": "pred-v1",
        "prediction_count": 10,
        "source_git_commit": "commit",
    }
    with pytest.raises(ValueError, match="requires a fitted-model"):
        validate_prediction_binding(prediction, None)
    validate_prediction_binding(prediction, model)
    changed = deepcopy(prediction)
    changed["prediction_model_binding"]["fitted_model_artifact_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        validate_prediction_binding(changed, model)
