from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.models import build_ml_model
from core.research.ml.validation import ChronologicalSplit


@dataclass(frozen=True)
class MLModelPrediction:
    probabilities: list[float]
    auxiliary_predictions: list[dict[str, float]]


class MLModelPipeline:
    """Build, train, and predict research ML models."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config

    def build_model(self) -> Any:
        return build_ml_model(
            self._experiment_config.model_type,
            random_seed=self._experiment_config.random_seed,
            class_weight=self.class_weight(),
            model_config=self._ml_config(),
        )

    def fit(self, model: Any, dataset: MLDataset) -> None:
        self.set_sequence_context(model, dataset)
        fit_multitask = getattr(model, "fit_multitask", None)
        if callable(fit_multitask) and self.multitask_enabled():
            fit_multitask(
                dataset.features,
                dataset.labels,
                self.auxiliary_targets_for_dataset(dataset),
            )
            return
        model.fit(dataset.features, dataset.labels)

    def predict(
        self,
        model: Any,
        dataset: MLDataset,
        *,
        prediction_context: MLDataset | None = None,
    ) -> MLModelPrediction:
        context = prediction_context or dataset
        self.set_sequence_context(model, context)
        predict_multitask = getattr(model, "predict_multitask", None)
        if callable(predict_multitask) and self.multitask_enabled():
            predictions = self.tail_rows(
                predict_multitask(context.features),
                dataset.sample_count,
            )
            probabilities = [
                float(row.get("probability_should_reduce_exposure", 0.5))
                for row in predictions
            ]
            auxiliary_predictions = [
                {
                    key: float(value)
                    for key, value in row.items()
                    if key.startswith("predicted_")
                }
                for row in predictions
            ]
            return MLModelPrediction(probabilities, auxiliary_predictions)
        component_predictions = self.model_component_predictions(
            model,
            context.features,
        )
        if component_predictions is not None:
            component_predictions = self.tail_rows(
                component_predictions,
                dataset.sample_count,
            )
            probabilities = [
                self.component_probability(row)
                for row in component_predictions
            ]
            auxiliary_predictions = [
                self.safe_component_auxiliary_predictions(row)
                for row in component_predictions
            ]
            return MLModelPrediction(probabilities, auxiliary_predictions)
        probabilities = self.tail_rows(
            model.predict_proba(context.features),
            dataset.sample_count,
        )
        return MLModelPrediction(probabilities, [{} for _ in probabilities])

    def prediction_context(self, split: ChronologicalSplit) -> MLDataset:
        return self.concat_datasets(split.train, split.test)

    @staticmethod
    def set_sequence_context(model: Any, dataset: MLDataset) -> None:
        setter = getattr(model, "set_sequence_context", None)
        if callable(setter):
            setter(metadata=dataset.metadata, feature_dates=dataset.feature_dates)

    @staticmethod
    def tail_rows(rows: list[Any], sample_count: int) -> list[Any]:
        if sample_count <= 0:
            return []
        return list(rows[-sample_count:])

    @staticmethod
    def concat_datasets(left: MLDataset, right: MLDataset) -> MLDataset:
        return MLDataset(
            features=[*left.features, *right.features],
            labels=[*left.labels, *right.labels],
            feature_dates=[*left.feature_dates, *right.feature_dates],
            label_start_dates=[*left.label_start_dates, *right.label_start_dates],
            label_end_dates=[*left.label_end_dates, *right.label_end_dates],
            feature_ids=[*left.feature_ids, *right.feature_ids]
            if left.feature_ids or right.feature_ids
            else [],
            metadata=[*left.metadata, *right.metadata]
            if left.metadata or right.metadata
            else [],
            auxiliary_targets=[*left.auxiliary_targets, *right.auxiliary_targets]
            if left.auxiliary_targets or right.auxiliary_targets
            else [],
        )

    @staticmethod
    def model_component_predictions(
        model: Any,
        features: list[dict[str, float]],
    ) -> list[dict[str, float]] | None:
        for method_name in (
            "predict_components",
            "predict_context",
            "predict_tft_outputs",
            "predict_news_components",
        ):
            method = getattr(model, method_name, None)
            if callable(method):
                return method(features)
        return None

    @staticmethod
    def component_probability(row: dict[str, float]) -> float:
        for name in (
            "probability_should_reduce_exposure",
            "market_regime_probability_risk_off",
            "news_probability_should_reduce_exposure",
        ):
            if name in row:
                return float(row[name])
        return 0.5

    @staticmethod
    def safe_component_auxiliary_predictions(
        row: dict[str, float],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        direct_mappings = {
            "trend_score": "predicted_trend_score",
            "regime_score": "predicted_regime_score",
            "size_multiplier": "predicted_size_multiplier",
            "rank_score": "predicted_rank_score",
            "risk_multiplier": "predicted_context_risk_multiplier",
        }
        for key, value in row.items():
            if key.startswith("predicted_"):
                values[key] = float(value)
            elif key in direct_mappings:
                values[direct_mappings[key]] = float(value)
        return values

    def multitask_enabled(self) -> bool:
        ml_config = self._ml_config()
        return bool(
            ml_config.get("multitask_enabled", False)
            or self._experiment_config.model_type == "multitask_transformer"
        )

    def multitask_regression_targets(self) -> list[str]:
        configured = self._ml_config().get("multitask_regression_targets", [])
        return [str(value) for value in configured]

    def auxiliary_targets_for_dataset(
        self,
        dataset: MLDataset,
    ) -> dict[str, list[float | None]]:
        targets = self.multitask_regression_targets()
        if not targets:
            return {}
        return {
            target: [
                (
                    dataset.auxiliary_targets[index].get(target)
                    if dataset.auxiliary_targets
                    and index < len(dataset.auxiliary_targets)
                    and dataset.auxiliary_targets[index]
                    else None
                )
                for index in range(dataset.sample_count)
            ]
            for target in targets
        }

    def class_weight(self) -> str | None:
        return "balanced" if self._experiment_config.class_weight_balanced else None

    def predictions_from_probabilities(self, probabilities: list[float]) -> list[int]:
        return [
            int(probability >= self._experiment_config.decision_threshold)
            for probability in probabilities
        ]

    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}
