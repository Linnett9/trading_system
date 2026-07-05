from __future__ import annotations

from typing import Any

from core.research.ml.data.datasets import MLDataset
from core.research.ml.pipelines import MLModelPipeline
from core.research.ml.validation import ChronologicalSplit


class MLExperimentRunnerModelMixin:
    def _split_dataset(self, dataset: MLDataset) -> ChronologicalSplit:
        return self._dataset_pipeline().split(dataset)

    def _set_model_sequence_context(self, model: Any, dataset: MLDataset) -> None:
        self._model_pipeline().set_sequence_context(model, dataset)

    def _fit_research_model(self, model: Any, dataset: MLDataset) -> None:
        self._model_pipeline().fit(model, dataset)

    def _predict_research_model(
        self,
        model: Any,
        dataset: MLDataset,
        *,
        prediction_context: MLDataset | None = None,
    ) -> tuple[list[float], list[dict[str, float]]]:
        prediction = self._model_pipeline().predict(
            model,
            dataset,
            prediction_context=prediction_context,
        )
        return prediction.probabilities, prediction.auxiliary_predictions

    def _prediction_context(self, split: ChronologicalSplit) -> MLDataset:
        return self._model_pipeline().prediction_context(split)

    @staticmethod
    def _tail_rows(rows: list[Any], sample_count: int) -> list[Any]:
        return MLModelPipeline.tail_rows(rows, sample_count)

    @staticmethod
    def _concat_datasets(left: MLDataset, right: MLDataset) -> MLDataset:
        return MLModelPipeline.concat_datasets(left, right)

    def _model_component_predictions(
        self,
        model: Any,
        features: list[dict[str, float]],
    ) -> list[dict[str, float]] | None:
        return MLModelPipeline.model_component_predictions(model, features)

    def _component_probability(self, row: dict[str, float]) -> float:
        return MLModelPipeline.component_probability(row)

    def _safe_component_auxiliary_predictions(
        self,
        row: dict[str, float],
    ) -> dict[str, float]:
        return MLModelPipeline.safe_component_auxiliary_predictions(row)

    def _multitask_enabled(self) -> bool:
        return self._model_pipeline().multitask_enabled()

    def _multitask_regression_targets(self) -> list[str]:
        return self._model_pipeline().multitask_regression_targets()

    def _auxiliary_targets_for_dataset(
        self,
        dataset: MLDataset,
    ) -> dict[str, list[float | None]]:
        return self._model_pipeline().auxiliary_targets_for_dataset(dataset)

    def _model_filename(self) -> str:
        return self._experiment_path_builder().model_filename()

    def _class_weight(self) -> str | None:
        return self._model_pipeline().class_weight()

    def _predictions_from_probabilities(self, probabilities: list[float]) -> list[int]:
        return self._model_pipeline().predictions_from_probabilities(probabilities)
