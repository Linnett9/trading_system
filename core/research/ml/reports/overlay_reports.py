from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.models import build_ml_model
from core.research.ml.overlay import overlay_decision_rule, simulate_shadow_overlay
from core.research.ml.pipelines import MLModelPipeline
from core.research.ml.validation import ChronologicalSplit, rolling_walk_forward


class MLOverlayReportWriter:
    """Write research-only overlay and shadow overlay reports."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        champion_equity_curve: list[Any],
        champion_rebalance_dates: set[str],
        *,
        model_pipeline: MLModelPipeline | None = None,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._champion_equity_curve = champion_equity_curve
        self._champion_rebalance_dates = champion_rebalance_dates
        self._model_pipeline = model_pipeline or MLModelPipeline(
            config,
            experiment_config,
        )

    def write_overlay_model_comparison(self, path: Path, dataset: MLDataset) -> None:
        config = self._ml_config()
        reduce_when, decision_rule = overlay_decision_rule(
            self._experiment_config.label_type
        )
        model_types = self.unique_strings(
            config.get(
                "overlay_comparison_models",
                [
                    self._experiment_config.model_type,
                    "logistic_regression",
                    "random_forest",
                    "gradient_boosting",
                ],
            )
        )
        thresholds = [
            float(value)
            for value in config.get(
                "overlay_comparison_thresholds",
                config.get("shadow_thresholds", [0.10, 0.15, 0.20, 0.25]),
            )
        ]
        reduced_exposures = [
            float(value)
            for value in config.get(
                "overlay_comparison_reduced_exposures",
                config.get("shadow_reduced_exposures", [0.70, 0.80, 0.90]),
            )
        ]
        transaction_cost_bps = float(config.get("shadow_transaction_cost_bps", 5.0))
        equity_by_date = self.equity_by_date()
        folds = rolling_walk_forward(
            dataset,
            self._experiment_config.walk_forward_folds,
        )
        model_payloads = []
        for model_type in model_types:
            scenarios = []
            for threshold in thresholds:
                for reduced_exposure in reduced_exposures:
                    fold_payloads = []
                    for fold in folds:
                        try:
                            model = build_ml_model(
                                model_type,
                                random_seed=self._experiment_config.random_seed,
                                class_weight=self._model_pipeline.class_weight(),
                                model_config=self._ml_config(),
                            )
                            self._model_pipeline.fit(model, fold.split.train)
                            prediction = self._model_pipeline.predict(
                                model,
                                fold.split.test,
                                prediction_context=(
                                    self._model_pipeline.prediction_context(fold.split)
                                ),
                            )
                            result = simulate_shadow_overlay(
                                equity_by_date,
                                dict(
                                    zip(
                                        fold.split.test.feature_dates,
                                        prediction.probabilities,
                                    )
                                ),
                                threshold,
                                reduced_exposure,
                                rebalance_dates=self._champion_rebalance_dates,
                                transaction_cost_bps=transaction_cost_bps,
                                reduce_when=reduce_when,
                            )
                        except Exception as exc:  # research report should record, not crash comparison
                            fold_payloads.append({
                                "fold": fold.fold_number,
                                "skipped": True,
                                "reason": str(exc),
                            })
                            continue
                        if result is None:
                            fold_payloads.append({
                                "fold": fold.fold_number,
                                "skipped": True,
                                "reason": "overlay_result_unavailable",
                            })
                            continue
                        payload = {
                            "fold": fold.fold_number,
                            **result.__dict__,
                        }
                        payload["return_delta"] = (
                            payload["overlay_total_return"] - payload["base_total_return"]
                        )
                        payload["max_drawdown_delta"] = (
                            payload["overlay_max_drawdown"] - payload["base_max_drawdown"]
                        )
                        fold_payloads.append(payload)
                    scenarios.append({
                        "decision_threshold": threshold,
                        "reduced_exposure": reduced_exposure,
                        "folds": fold_payloads,
                        "summary": self.overlay_fold_summary(fold_payloads),
                    })
            model_payloads.append({"model_type": model_type, "scenarios": scenarios})
        path.write_text(json.dumps({
            "mode": "overlay_model_comparison_research_only",
            "label_type": self._experiment_config.label_type,
            "overlay_probability": self.overlay_probability_label(),
            "overlay_decision_rule": decision_rule,
            "fold_count": len(folds),
            "rebalance_only": True,
            "transaction_cost_bps": transaction_cost_bps,
            "models": model_payloads,
            "research_only": True,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def write_shadow_overlay(self, path: Path, dataset: MLDataset) -> None:
        config = self._ml_config()
        reduce_when, decision_rule = overlay_decision_rule(
            self._experiment_config.label_type
        )
        model_type = str(config.get("shadow_model_type", "gradient_boosting"))
        thresholds = config.get("shadow_thresholds", [0.10, 0.15, 0.20, 0.25])
        reduced_exposures = config.get("shadow_reduced_exposures", [0.70, 0.80, 0.90])
        transaction_cost_bps = float(config.get("shadow_transaction_cost_bps", 5.0))
        equity_by_date = self.equity_by_date()
        scenarios = [
            {"decision_threshold": float(threshold), "reduced_exposure": float(exposure), "folds": []}
            for threshold in thresholds
            for exposure in reduced_exposures
        ]
        for fold in rolling_walk_forward(
            dataset,
            self._experiment_config.walk_forward_folds,
        ):
            model = build_ml_model(
                model_type,
                random_seed=self._experiment_config.random_seed,
                model_config=self._ml_config(),
            )
            self._model_pipeline.fit(model, fold.split.train)
            prediction = self._model_pipeline.predict(
                model,
                fold.split.test,
                prediction_context=self._model_pipeline.prediction_context(fold.split),
            )
            probabilities_by_date = dict(
                zip(fold.split.test.feature_dates, prediction.probabilities)
            )
            for scenario in scenarios:
                result = simulate_shadow_overlay(
                    equity_by_date,
                    probabilities_by_date,
                    scenario["decision_threshold"],
                    scenario["reduced_exposure"],
                    rebalance_dates=self._champion_rebalance_dates,
                    transaction_cost_bps=transaction_cost_bps,
                    reduce_when=reduce_when,
                )
                if result is not None:
                    scenario["folds"].append({
                        "fold": fold.fold_number,
                        **result.__dict__,
                    })
        path.write_text(json.dumps({
            "mode": "shadow_research_only",
            "model_type": model_type,
            "rebalance_only": True,
            "overlay_probability": self.overlay_probability_label(),
            "overlay_decision_rule": decision_rule,
            "transaction_cost_bps": transaction_cost_bps,
            "scenarios": scenarios,
            "trading_impact": "none",
        }, indent=2), encoding="utf-8")

    def write_holdout_shadow_overlay(
        self,
        path: Path,
        split: ChronologicalSplit,
    ) -> None:
        config = self._ml_config()
        reduce_when, decision_rule = overlay_decision_rule(
            self._experiment_config.label_type
        )
        model_type = str(config.get("shadow_model_type", "gradient_boosting"))
        threshold = float(config.get("shadow_holdout_threshold", 0.20))
        reduced_exposure = float(
            config.get("shadow_holdout_reduced_exposure", 0.70)
        )
        transaction_cost_bps = float(config.get("shadow_transaction_cost_bps", 5.0))
        model = build_ml_model(
            model_type,
            random_seed=self._experiment_config.random_seed,
            model_config=self._ml_config(),
        )
        self._model_pipeline.fit(model, split.train)
        prediction = self._model_pipeline.predict(
            model,
            split.test,
            prediction_context=self._model_pipeline.prediction_context(split),
        )
        result = simulate_shadow_overlay(
            self.equity_by_date(),
            dict(zip(split.test.feature_dates, prediction.probabilities)),
            threshold,
            reduced_exposure,
            rebalance_dates=self._champion_rebalance_dates,
            transaction_cost_bps=transaction_cost_bps,
            reduce_when=reduce_when,
        )
        path.write_text(json.dumps({
            "mode": "final_holdout_shadow_research_only",
            "model_type": model_type,
            "decision_threshold": threshold,
            "reduced_exposure": reduced_exposure,
            "rebalance_only": True,
            "overlay_probability": self.overlay_probability_label(),
            "overlay_decision_rule": decision_rule,
            "transaction_cost_bps": transaction_cost_bps,
            "test_start_date": split.test_start_date,
            "result": result.__dict__ if result is not None else None,
            "trading_impact": "none",
            "candidate_frozen_before_holdout": True,
        }, indent=2), encoding="utf-8")

    def overlay_probability_label(self) -> str:
        return f"{self._experiment_config.label_type}_probability"

    def overlay_fold_summary(self, folds: list[dict]) -> dict[str, float | int | None]:
        valid_folds = [fold for fold in folds if not fold.get("skipped")]
        return {
            "valid_fold_count": len(valid_folds),
            "skipped_fold_count": len(folds) - len(valid_folds),
            "mean_base_total_return": self.mean_metric(valid_folds, "base_total_return"),
            "mean_overlay_total_return": self.mean_metric(valid_folds, "overlay_total_return"),
            "mean_return_delta": self.mean_metric(valid_folds, "return_delta"),
            "mean_base_max_drawdown": self.mean_metric(valid_folds, "base_max_drawdown"),
            "mean_overlay_max_drawdown": self.mean_metric(valid_folds, "overlay_max_drawdown"),
            "mean_max_drawdown_delta": self.mean_metric(valid_folds, "max_drawdown_delta"),
            "mean_reduced_exposure_days": self.mean_metric(valid_folds, "reduced_exposure_days"),
            "mean_overlay_turnover": self.mean_metric(valid_folds, "overlay_turnover"),
        }

    def equity_by_date(self) -> dict[str, float]:
        return {
            point.timestamp.date().isoformat(): point.equity
            for point in self._champion_equity_curve
        }

    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}

    @staticmethod
    def unique_strings(values: list[Any]) -> list[str]:
        seen = set()
        output = []
        for value in values:
            text = str(value)
            if text not in seen:
                seen.add(text)
                output.append(text)
        return output

    @staticmethod
    def mean_metric(metrics: list[dict], key: str) -> float | None:
        values = [item[key] for item in metrics if item.get(key) is not None]
        return sum(values) / len(values) if values else None
