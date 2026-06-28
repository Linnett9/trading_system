from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.artifacts import MLCoreArtifactWriter
from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.diagnostics import (
    build_ranking_diagnostics,
    probability_summary,
    rolling_base_rate_probabilities,
)
from core.research.ml.evaluation import classification_metrics
from core.research.ml.pipelines import MLModelPipeline
from core.research.ml.validation import rolling_walk_forward


class MLDiagnosticReportWriter:
    """Write non-overlay ML diagnostic and model-comparison reports."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        *,
        model_pipeline: MLModelPipeline | None = None,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._model_pipeline = model_pipeline or MLModelPipeline(
            config,
            experiment_config,
        )

    def write_walk_forward_metrics(self, path: Path, dataset: MLDataset) -> None:
        folds = rolling_walk_forward(
            dataset,
            fold_count=self._experiment_config.walk_forward_folds,
        )
        payload_folds = []
        for fold in folds:
            model = self._model_pipeline.build_model()
            self._model_pipeline.fit(model, fold.split.train)
            prediction = self._model_pipeline.predict(
                model,
                fold.split.test,
                prediction_context=self._model_pipeline.prediction_context(fold.split),
            )
            predictions = self._model_pipeline.predictions_from_probabilities(
                prediction.probabilities
            )
            payload_folds.append({
                "fold": fold.fold_number,
                "train_sample_count": fold.split.train.sample_count,
                "test_sample_count": fold.split.test.sample_count,
                "test_start_date": fold.split.test_start_date,
                "purged_train_samples": fold.split.purged_train_samples,
                "metrics": classification_metrics(fold.split.test.labels, predictions),
                "baselines": self.baseline_metrics(fold.split),
            })
        path.write_text(json.dumps({
            "model_type": self._experiment_config.model_type,
            "fold_count": len(payload_folds),
            "folds": payload_folds,
            "research_only": True,
        }, indent=2), encoding="utf-8")

    def write_threshold_sweep(
        self,
        path: Path,
        dataset: MLDataset,
        probabilities: list[float],
    ) -> None:
        thresholds = self.thresholds()
        path.write_text(json.dumps({
            "evaluation": "holdout_only",
            "thresholds": [
                {
                    "threshold": threshold,
                    "metrics": classification_metrics(
                        dataset.labels,
                        [int(value >= threshold) for value in probabilities],
                    ),
                }
                for threshold in thresholds
            ],
        }, indent=2), encoding="utf-8")

    def write_model_comparison(self, path: Path, dataset: MLDataset) -> None:
        folds = rolling_walk_forward(
            dataset,
            self._experiment_config.walk_forward_folds,
        )
        thresholds = self.thresholds()
        model_types = self._config.get("ml", {}).get(
            "comparison_models",
            ["logistic_regression", "random_forest", "gradient_boosting"],
        )
        models = []
        for model_type in model_types:
            threshold_metrics = {threshold: [] for threshold in thresholds}
            model_pipeline = self.model_pipeline_for(str(model_type))
            for fold in folds:
                model = model_pipeline.build_model()
                model_pipeline.fit(model, fold.split.train)
                prediction = model_pipeline.predict(
                    model,
                    fold.split.test,
                    prediction_context=model_pipeline.prediction_context(fold.split),
                )
                for threshold in thresholds:
                    threshold_metrics[threshold].append(classification_metrics(
                        fold.split.test.labels,
                        [
                            int(value >= threshold)
                            for value in prediction.probabilities
                        ],
                    ))
            models.append({
                "model_type": model_type,
                "thresholds": [
                    {
                        "threshold": threshold,
                        "mean_balanced_accuracy": self.mean_metric(
                            threshold_metrics[threshold], "balanced_accuracy"
                        ),
                        "mean_precision": self.mean_metric(
                            threshold_metrics[threshold], "precision"
                        ),
                        "mean_recall": self.mean_metric(
                            threshold_metrics[threshold], "recall"
                        ),
                    }
                    for threshold in thresholds
                ],
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "fold_count": len(folds),
            "models": models,
            "research_only": True,
        }, indent=2), encoding="utf-8")

    def write_baseline_model_comparison(
        self,
        path: Path,
        dataset: MLDataset,
    ) -> None:
        model_types = list(self._config.get("ml", {}).get(
            "comparison_models",
            ["logistic_regression", "random_forest", "gradient_boosting"],
        ))
        fold_payloads = []
        summaries_by_name: dict[str, list[dict]] = {}
        for fold in rolling_walk_forward(
            dataset,
            fold_count=self._experiment_config.walk_forward_folds,
        ):
            static_probability = (
                sum(fold.split.train.labels) / fold.split.train.sample_count
            )
            static_probabilities = [static_probability] * fold.split.test.sample_count
            static_summary = probability_summary(
                fold.split.test.labels,
                static_probabilities,
                decision_threshold=self._experiment_config.decision_threshold,
            )
            rolling_probabilities = rolling_base_rate_probabilities(
                fold.split.train.labels,
                fold.split.train.label_end_dates,
                fold.split.test.feature_dates,
                fold.split.test.labels,
                fold.split.test.label_end_dates,
                lookback_samples=self.rolling_base_rate_lookback_samples(),
            )
            baseline_summaries = {
                "static_base_rate": static_summary,
                "rolling_base_rate": probability_summary(
                    fold.split.test.labels,
                    rolling_probabilities,
                    decision_threshold=self._experiment_config.decision_threshold,
                    reference_brier_score=static_summary["brier_score"],
                ),
                "always_positive": probability_summary(
                    fold.split.test.labels,
                    [1.0] * fold.split.test.sample_count,
                    decision_threshold=self._experiment_config.decision_threshold,
                    reference_brier_score=static_summary["brier_score"],
                ),
            }
            model_summaries = []
            for model_type in model_types:
                model_pipeline = self.model_pipeline_for(str(model_type))
                model = model_pipeline.build_model()
                model_pipeline.fit(model, fold.split.train)
                prediction = model_pipeline.predict(
                    model,
                    fold.split.test,
                    prediction_context=model_pipeline.prediction_context(fold.split),
                )
                summary = probability_summary(
                    fold.split.test.labels,
                    prediction.probabilities,
                    decision_threshold=self._experiment_config.decision_threshold,
                    reference_brier_score=baseline_summaries["rolling_base_rate"][
                        "brier_score"
                    ],
                )
                summary["brier_skill_vs_static_base_rate"] = (
                    1 - summary["brier_score"] / static_summary["brier_score"]
                    if static_summary["brier_score"] else None
                )
                summaries_by_name.setdefault(model_type, []).append(summary)
                model_summaries.append({"model_type": model_type, **summary})
            for name, summary in baseline_summaries.items():
                summaries_by_name.setdefault(name, []).append(summary)
            fold_payloads.append({
                "fold": fold.fold_number,
                "test_start_date": fold.split.test_start_date,
                "baselines": baseline_summaries,
                "models": model_summaries,
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "rolling_base_rate_lookback_samples": (
                self.rolling_base_rate_lookback_samples()
            ),
            "fold_count": len(fold_payloads),
            "folds": fold_payloads,
            "mean_metrics_by_predictor": {
                name: self.mean_probability_summary(summaries)
                for name, summaries in summaries_by_name.items()
            },
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def write_ranking_diagnostics(
        self,
        path: Path,
        dataset: MLDataset,
        outcomes_by_feature_date: dict[str, dict[str, float | None]],
    ) -> None:
        folds = []
        all_labels: list[int] = []
        all_probabilities: list[float] = []
        all_outcomes: list[dict[str, float | None]] = []
        for fold in rolling_walk_forward(
            dataset,
            fold_count=self._experiment_config.walk_forward_folds,
        ):
            model = self._model_pipeline.build_model()
            self._model_pipeline.fit(model, fold.split.train)
            prediction = self._model_pipeline.predict(
                model,
                fold.split.test,
                prediction_context=self._model_pipeline.prediction_context(fold.split),
            )
            outcomes = [
                outcomes_by_feature_date.get(feature_date, {})
                for feature_date in fold.split.test.feature_dates
            ]
            all_labels.extend(fold.split.test.labels)
            all_probabilities.extend(prediction.probabilities)
            all_outcomes.extend(outcomes)
            folds.append({
                "fold": fold.fold_number,
                "test_start_date": fold.split.test_start_date,
                "diagnostics": build_ranking_diagnostics(
                    fold.split.test.labels,
                    prediction.probabilities,
                    outcomes,
                    quantile_count=self.ranking_quantile_count(),
                ),
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "model_type": self._experiment_config.model_type,
            "quantile_count": self.ranking_quantile_count(),
            "fold_count": len(folds),
            "folds": folds,
            "pooled_out_of_sample_diagnostics": build_ranking_diagnostics(
                all_labels,
                all_probabilities,
                all_outcomes,
                quantile_count=self.ranking_quantile_count(),
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def model_pipeline_for(self, model_type: str) -> MLModelPipeline:
        return MLModelPipeline(
            self._config,
            replace(self._experiment_config, model_type=model_type),
        )

    def baseline_metrics(self, split) -> dict[str, dict]:
        return MLCoreArtifactWriter(
            self._config,
            self._experiment_config,
            research_label=str(
                self._config.get("ml", {}).get(
                    "research_label",
                    "UNSPECIFIED_RESEARCH",
                )
            ),
            model_pipeline=self._model_pipeline,
        ).baseline_metrics(split)

    def mean_probability_summary(self, summaries: list[dict]) -> dict[str, float | None]:
        keys = (
            "brier_score",
            "brier_skill_vs_reference",
            "brier_skill_vs_static_base_rate",
            "roc_auc",
            "positive_prediction_rate",
        )
        return {
            key: self.mean_metric(summaries, key)
            for key in keys
        }

    def rolling_base_rate_lookback_samples(self) -> int:
        return int(
            self._config.get("ml", {}).get(
                "rolling_base_rate_lookback_samples", 252
            )
        )

    def ranking_quantile_count(self) -> int:
        return int(self._config.get("ml", {}).get("ranking_quantile_count", 5))

    @staticmethod
    def thresholds() -> list[float]:
        return [round(value / 100, 2) for value in range(20, 85, 5)]

    @staticmethod
    def mean_metric(metrics: list[dict], key: str) -> float | None:
        values = [item[key] for item in metrics if item.get(key) is not None]
        return sum(values) / len(values) if values else None
