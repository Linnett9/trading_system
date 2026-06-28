from __future__ import annotations

from typing import Any

__all__ = [
    "MLCoreArtifactWriter",
    "MLExperimentPathBuilder",
    "MLExperimentPaths",
    "MLFeatureCache",
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
    raise AttributeError(name)
