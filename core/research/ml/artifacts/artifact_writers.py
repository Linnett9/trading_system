from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.research.ml.artifacts.artifact_writer_basic import MLArtifactBasicWritersMixin
from core.research.ml.artifacts.artifact_writer_hashing import MLArtifactHashingMixin
from core.research.ml.artifacts.artifact_writer_predictions import MLPredictionArtifactWriterMixin
from core.research.ml.artifacts.artifact_writer_stats import MLArtifactStatsMixin
from core.research.ml.config import MLExperimentConfig
from core.research.ml.pipelines import MLModelPipeline


class MLCoreArtifactWriter(
    MLArtifactBasicWritersMixin,
    MLPredictionArtifactWriterMixin,
    MLArtifactHashingMixin,
    MLArtifactStatsMixin,
):
    """Write core ML experiment artifacts without controlling the experiment flow."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        *,
        research_label: str,
        model_pipeline: MLModelPipeline | None = None,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._research_label = research_label
        self._model_pipeline = model_pipeline or MLModelPipeline(
            config,
            experiment_config,
        )
    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}
