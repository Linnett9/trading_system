from __future__ import annotations

from typing import Any

__all__ = [
    "ALLOWED_OPTIONAL_PREDICTION_COLUMNS",
    "ARTIFACT_SCHEMA_VERSION",
    "MLCoreArtifactWriter",
    "MLExperimentPathBuilder",
    "MLExperimentPaths",
    "MLFeatureCache",
    "PredictionArtifactValidationResult",
    "REQUIRED_PREDICTION_ARTIFACT_COLUMNS",
    "is_allowed_predicted_column",
    "validate_prediction_artifact_dirs",
    "validate_prediction_artifacts",
]


def __getattr__(name: str) -> Any:
    if name == "MLCoreArtifactWriter":
        from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter

        return MLCoreArtifactWriter
    if name in {"MLExperimentPathBuilder", "MLExperimentPaths"}:
        from core.research.ml.artifacts.experiment_paths import (
            MLExperimentPathBuilder,
            MLExperimentPaths,
        )

        return {
            "MLExperimentPathBuilder": MLExperimentPathBuilder,
            "MLExperimentPaths": MLExperimentPaths,
        }[name]
    if name == "MLFeatureCache":
        from core.research.ml.artifacts.feature_cache import MLFeatureCache

        return MLFeatureCache
    if name in {
        "ALLOWED_OPTIONAL_PREDICTION_COLUMNS",
        "ARTIFACT_SCHEMA_VERSION",
        "REQUIRED_PREDICTION_ARTIFACT_COLUMNS",
        "is_allowed_predicted_column",
    }:
        from core.research.ml.artifacts.artifact_schema import (
            ALLOWED_OPTIONAL_PREDICTION_COLUMNS,
            ARTIFACT_SCHEMA_VERSION,
            REQUIRED_PREDICTION_ARTIFACT_COLUMNS,
            is_allowed_predicted_column,
        )

        return {
            "ALLOWED_OPTIONAL_PREDICTION_COLUMNS": (
                ALLOWED_OPTIONAL_PREDICTION_COLUMNS
            ),
            "ARTIFACT_SCHEMA_VERSION": ARTIFACT_SCHEMA_VERSION,
            "REQUIRED_PREDICTION_ARTIFACT_COLUMNS": (
                REQUIRED_PREDICTION_ARTIFACT_COLUMNS
            ),
            "is_allowed_predicted_column": is_allowed_predicted_column,
        }[name]
    if name in {
        "PredictionArtifactValidationResult",
        "validate_prediction_artifact_dirs",
        "validate_prediction_artifacts",
    }:
        from core.research.ml.artifacts.artifact_validator import (
            PredictionArtifactValidationResult,
            validate_prediction_artifact_dirs,
            validate_prediction_artifacts,
        )

        return {
            "PredictionArtifactValidationResult": (
                PredictionArtifactValidationResult
            ),
            "validate_prediction_artifact_dirs": validate_prediction_artifact_dirs,
            "validate_prediction_artifacts": validate_prediction_artifacts,
        }[name]
    raise AttributeError(name)
