from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.calibration import (
    build_probability_calibration,
    compare_calibration_methods,
)
from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.pipelines import MLModelPipeline
from core.research.ml.validation import ChronologicalSplit, rolling_walk_forward


class MLCalibrationReportWriter:
    """Write research-only probability calibration reports."""

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

    def write_probability_calibration(
        self,
        path: Path,
        labels: list[int],
        probabilities: list[float],
    ) -> None:
        path.write_text(json.dumps({
            "evaluation": "chronological_holdout",
            "model_type": self._experiment_config.model_type,
            "calibration": build_probability_calibration(
                labels,
                probabilities,
                bin_count=self.calibration_bin_count(),
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def write_calibrated_probability_calibration(
        self,
        path: Path,
        split: ChronologicalSplit,
        raw_probabilities: list[float],
    ) -> None:
        train_model = self._model_pipeline.build_model()
        self._model_pipeline.fit(train_model, split.train)
        train_prediction = self._model_pipeline.predict(train_model, split.train)
        comparison = compare_calibration_methods(
            split.train.labels,
            train_prediction.probabilities,
            split.test.labels,
            raw_probabilities,
            bin_count=self.calibration_bin_count(),
        )
        raw_calibration = build_probability_calibration(
            split.test.labels,
            raw_probabilities,
            bin_count=self.calibration_bin_count(),
        )
        best_method = comparison.get("best_method_by_brier")
        best_calibration = (
            comparison.get("methods", {})
            .get(str(best_method), {})
            .get("calibration", {})
        )
        path.write_text(json.dumps({
            "evaluation": "chronological_holdout_calibration_method_comparison",
            "model_type": self._experiment_config.model_type,
            "label_type": self._experiment_config.label_type,
            "calibration_methods": ["raw", "platt", "isotonic", "temperature_scaling"],
            "best_method_by_brier": best_method,
            "raw_calibration": raw_calibration,
            "best_calibration": best_calibration,
            "method_comparison": comparison,
            "raw_brier_score": raw_calibration.get("brier_score"),
            "best_brier_score": best_calibration.get("brier_score"),
            "brier_delta_best_minus_raw": (
                best_calibration.get("brier_score") - raw_calibration.get("brier_score")
                if raw_calibration.get("brier_score") is not None
                and best_calibration.get("brier_score") is not None
                else None
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def write_walk_forward_probability_calibration(
        self,
        path: Path,
        dataset: MLDataset,
    ) -> None:
        fold_payloads = []
        all_labels: list[int] = []
        all_probabilities: list[float] = []
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
            all_labels.extend(fold.split.test.labels)
            all_probabilities.extend(prediction.probabilities)
            fold_payloads.append({
                "fold": fold.fold_number,
                "test_start_date": fold.split.test_start_date,
                "calibration": build_probability_calibration(
                    fold.split.test.labels,
                    prediction.probabilities,
                    bin_count=self.calibration_bin_count(),
                ),
            })
        path.write_text(json.dumps({
            "evaluation": "purged_walk_forward",
            "model_type": self._experiment_config.model_type,
            "fold_count": len(fold_payloads),
            "folds": fold_payloads,
            "pooled_out_of_sample_calibration": build_probability_calibration(
                all_labels,
                all_probabilities,
                bin_count=self.calibration_bin_count(),
            ),
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def calibration_bin_count(self) -> int:
        return int(self._config.get("ml", {}).get("calibration_bin_count", 10))
