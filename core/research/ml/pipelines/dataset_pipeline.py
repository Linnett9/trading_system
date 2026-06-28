from __future__ import annotations

from dataclasses import dataclass

from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset, build_dataset
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.labels import MLLabelBuildResult
from core.research.ml.validation import ChronologicalSplit, chronological_holdout


@dataclass(frozen=True)
class MLDatasetPreparation:
    dataset: MLDataset
    split: ChronologicalSplit


class MLDatasetPipeline:
    """Prepare the ML dataset and chronological validation split."""

    def __init__(self, experiment_config: MLExperimentConfig) -> None:
        self._experiment_config = experiment_config

    def prepare(
        self,
        feature_result: MLFeatureBuildResult,
        label_result: MLLabelBuildResult,
    ) -> MLDatasetPreparation:
        dataset = self.build_dataset(feature_result, label_result)
        return MLDatasetPreparation(
            dataset=dataset,
            split=self.split(dataset),
        )

    def build_dataset(
        self,
        feature_result: MLFeatureBuildResult,
        label_result: MLLabelBuildResult,
    ) -> MLDataset:
        return build_dataset(
            feature_result.rows,
            label_result.rows,
            label_name=label_result.label_name,
        )

    def split(self, dataset: MLDataset) -> ChronologicalSplit:
        return chronological_holdout(
            dataset,
            test_fraction=self._experiment_config.test_fraction,
            train_start=self._experiment_config.train_start,
            train_end=self._experiment_config.train_end,
            test_start=self._experiment_config.test_start,
            test_end=self._experiment_config.test_end,
        )
