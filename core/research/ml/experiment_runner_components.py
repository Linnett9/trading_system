from __future__ import annotations

from typing import Any

from core.research.ml.artifacts import (
    MLCoreArtifactWriter,
    MLExperimentPathBuilder,
    MLExperimentPaths,
    MLFeatureCache,
)
from core.research.ml.config import MLExperimentConfig
from core.research.ml.pipelines import (
    MLDatasetPipeline,
    MLFeaturePipeline,
    MLModelPipeline,
    MLRebalancePipeline,
)
from core.research.ml.reports import (
    MLCalibrationReportWriter,
    MLDiagnosticReportWriter,
    MLOverlayReportWriter,
)


class MLExperimentRunnerComponentMixin:
    def _experiment_path_builder(self) -> MLExperimentPathBuilder:
        return MLExperimentPathBuilder(self.config, self.experiment_config)

    def _experiment_paths(self) -> MLExperimentPaths:
        return self._experiment_path_builder().build()

    def _feature_cache(self) -> MLFeatureCache:
        return MLFeatureCache(self.config)

    def _feature_pipeline(self) -> MLFeaturePipeline:
        return MLFeaturePipeline(
            self.config,
            self.experiment_config,
            feed=self.feed,
            research_label=self.research_label,
            feature_cache=self._feature_cache(),
            path_builder=self._experiment_path_builder(),
        )

    def _dataset_pipeline(self) -> MLDatasetPipeline:
        return MLDatasetPipeline(self.experiment_config)

    def _model_pipeline(self) -> MLModelPipeline:
        return MLModelPipeline(self.config, self.experiment_config)

    def _rebalance_pipeline(self) -> MLRebalancePipeline:
        return MLRebalancePipeline(
            self.config,
            self.experiment_config,
            champion_equity_curve=self._champion_equity_curve,
            champion_selections=self._champion_selections,
            feature_cache=self._feature_cache(),
        )

    def _artifact_writer(self) -> MLCoreArtifactWriter:
        return MLCoreArtifactWriter(
            self.config,
            self.experiment_config,
            research_label=self.research_label,
            model_pipeline=self._model_pipeline(),
        )

    def _calibration_report_writer(self) -> MLCalibrationReportWriter:
        return MLCalibrationReportWriter(
            self.config,
            self.experiment_config,
            model_pipeline=self._model_pipeline(),
        )

    def _diagnostic_report_writer(self) -> MLDiagnosticReportWriter:
        return MLDiagnosticReportWriter(
            self.config,
            self.experiment_config,
            model_pipeline=self._model_pipeline(),
        )

    def _overlay_report_writer(self) -> MLOverlayReportWriter:
        return MLOverlayReportWriter(
            self.config,
            self.experiment_config,
            self._champion_equity_curve,
            self._champion_rebalance_dates,
            model_pipeline=self._model_pipeline(),
        )
