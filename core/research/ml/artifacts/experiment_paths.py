from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.config import MLExperimentConfig


@dataclass(frozen=True)
class MLExperimentPaths:
    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path
    confusion_matrix_path: Path
    metadata_path: Path
    model_path: Path
    features_path: Path
    feature_summary_path: Path
    labels_path: Path
    dataset_path: Path
    dataset_audit_path: Path
    walk_forward_metrics_path: Path
    threshold_sweep_path: Path
    model_comparison_path: Path
    shadow_overlay_path: Path
    holdout_shadow_overlay_path: Path
    rebalance_dataset_path: Path
    rebalance_dataset_audit_path: Path
    history_coverage_path: Path
    drawdown_event_review_path: Path
    rule_exposure_study_path: Path
    probability_calibration_path: Path
    walk_forward_probability_calibration_path: Path
    baseline_model_comparison_path: Path
    ranking_diagnostics_path: Path
    calibrated_probability_calibration_path: Path
    overlay_model_comparison_path: Path
    prediction_artifacts_path: Path
    prediction_artifacts_metadata_path: Path
    html_report_path: Path

    def result_kwargs(self) -> dict[str, Path]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class MLExperimentPathBuilder:
    """Construct filesystem paths for ML experiment artifacts."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config

    def build(self) -> MLExperimentPaths:
        output_dir = Path(self._experiment_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return MLExperimentPaths(
            output_dir=output_dir,
            metrics_path=output_dir / "metrics.json",
            predictions_path=output_dir / "predictions.csv",
            feature_importance_path=output_dir / "feature_importance.csv",
            confusion_matrix_path=output_dir / "confusion_matrix.csv",
            metadata_path=output_dir / "metadata.json",
            model_path=output_dir / self.model_filename(),
            features_path=self.features_path(),
            feature_summary_path=output_dir / "feature_summary.json",
            labels_path=self.labels_path(),
            dataset_path=self.dataset_path(),
            dataset_audit_path=output_dir / "dataset_audit.json",
            walk_forward_metrics_path=output_dir / "walk_forward_metrics.json",
            threshold_sweep_path=output_dir / "threshold_sweep.json",
            model_comparison_path=output_dir / "model_comparison.json",
            shadow_overlay_path=output_dir / "shadow_overlay.json",
            holdout_shadow_overlay_path=output_dir / "holdout_shadow_overlay.json",
            rebalance_dataset_path=self.rebalance_dataset_path(),
            rebalance_dataset_audit_path=output_dir / "rebalance_dataset_audit.json",
            history_coverage_path=output_dir / "history_coverage.json",
            drawdown_event_review_path=output_dir / "drawdown_event_review.json",
            rule_exposure_study_path=output_dir / "rule_exposure_study.json",
            probability_calibration_path=output_dir / "probability_calibration.json",
            walk_forward_probability_calibration_path=(
                output_dir / "walk_forward_probability_calibration.json"
            ),
            baseline_model_comparison_path=(
                output_dir / "baseline_model_comparison.json"
            ),
            ranking_diagnostics_path=output_dir / "ranking_diagnostics.json",
            calibrated_probability_calibration_path=(
                output_dir / "calibrated_probability_calibration.json"
            ),
            overlay_model_comparison_path=output_dir / "overlay_model_comparison.json",
            prediction_artifacts_path=output_dir / "prediction_artifacts.csv",
            prediction_artifacts_metadata_path=output_dir / "prediction_artifacts.json",
            html_report_path=output_dir / "research_report.html",
        )

    def features_path(self) -> Path:
        return self._cache_dir() / "features.csv"

    def labels_path(self) -> Path:
        filename = f"labels_{self._experiment_config.label_type}.csv"
        return self._cache_dir() / filename

    def dataset_path(self) -> Path:
        filename = f"dataset_{self._experiment_config.label_type}.csv"
        return self._cache_dir() / filename

    def rebalance_dataset_path(self) -> Path:
        if self._experiment_config.label_type == "should_reduce_exposure":
            return self._cache_dir() / "expanded_rebalance_dataset.csv"
        return self._cache_dir() / "champion_rebalance_dataset.csv"

    def model_filename(self) -> str:
        if self._experiment_config.model_type == "noop":
            return "model.json"
        if self._experiment_config.model_type in {
            "transformer",
            "patchtst",
            "dlinear",
            "itransformer",
            "market_context_encoder",
            "momentum_transformer",
            "multitask_transformer",
            "news_analysis_transformer",
            "temporal_fusion_transformer",
        }:
            return "model.pt"
        return "model.joblib"

    def _cache_dir(self) -> Path:
        cache_config = self._config.get("cache", {})
        return Path(cache_config.get("ml_dir", "cache/ml"))
