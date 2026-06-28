from __future__ import annotations

from core.research.ml import artifacts
from core.research.ml.artifacts import artifact_schema
from core.research.ml.artifacts import artifact_validator


def test_prediction_artifact_schema_exports_from_artifacts_package():
    assert artifacts.ARTIFACT_SCHEMA_VERSION == artifact_schema.ARTIFACT_SCHEMA_VERSION
    assert artifacts.REQUIRED_PREDICTION_ARTIFACT_COLUMNS is (
        artifact_schema.REQUIRED_PREDICTION_ARTIFACT_COLUMNS
    )
    assert artifacts.is_allowed_predicted_column is (
        artifact_schema.is_allowed_predicted_column
    )


def test_prediction_artifact_validator_exports_from_artifacts_package():
    assert artifacts.PredictionArtifactValidationResult is (
        artifact_validator.PredictionArtifactValidationResult
    )
    assert artifacts.validate_prediction_artifacts is (
        artifact_validator.validate_prediction_artifacts
    )
    assert artifacts.validate_prediction_artifact_dirs is (
        artifact_validator.validate_prediction_artifact_dirs
    )
