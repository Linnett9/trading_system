from __future__ import annotations

from typing import Any

__all__ = [
    "MLDatasetPipeline",
    "MLDatasetPreparation",
    "MLFeaturePipeline",
    "MLFeaturePipelineResult",
    "MLLabelPipeline",
    "MLModelPipeline",
    "MLModelPrediction",
]


def __getattr__(name: str) -> Any:
    if name in {"MLDatasetPipeline", "MLDatasetPreparation"}:
        from core.research.ml.pipelines.dataset_pipeline import (
            MLDatasetPipeline,
            MLDatasetPreparation,
        )

        return {
            "MLDatasetPipeline": MLDatasetPipeline,
            "MLDatasetPreparation": MLDatasetPreparation,
        }[name]
    if name in {"MLFeaturePipeline", "MLFeaturePipelineResult"}:
        from core.research.ml.pipelines.feature_pipeline import (
            MLFeaturePipeline,
            MLFeaturePipelineResult,
        )

        return {
            "MLFeaturePipeline": MLFeaturePipeline,
            "MLFeaturePipelineResult": MLFeaturePipelineResult,
        }[name]
    if name == "MLLabelPipeline":
        from core.research.ml.pipelines.label_pipeline import MLLabelPipeline

        return MLLabelPipeline
    if name in {"MLModelPipeline", "MLModelPrediction"}:
        from core.research.ml.pipelines.model_pipeline import (
            MLModelPipeline,
            MLModelPrediction,
        )

        return {
            "MLModelPipeline": MLModelPipeline,
            "MLModelPrediction": MLModelPrediction,
        }[name]
    raise AttributeError(name)
