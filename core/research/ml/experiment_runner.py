from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import write_dataset
from core.research.ml.experiment_result import MLExperimentResult
from core.research.ml.overlays.drawdown_review import write_drawdown_event_review
from core.research.ml.html_report import write_research_html_report
from core.research.ml.features.features import write_feature_rows
from core.research.ml.features.labels import write_label_rows
from core.research.ml.experiment_runner_components import (
    MLExperimentRunnerComponentMixin,
)
from core.research.ml.experiment_runner_features import MLExperimentRunnerFeatureMixin
from core.research.ml.experiment_runner_model import MLExperimentRunnerModelMixin
from core.research.ml.experiment_runner_reporting import MLExperimentRunnerReportingMixin


class MLExperimentRunner(
    MLExperimentRunnerReportingMixin,
    MLExperimentRunnerModelMixin,
    MLExperimentRunnerFeatureMixin,
    MLExperimentRunnerComponentMixin,
):
    """Research-only ML runner. It does not affect trading decisions."""

    def __init__(self, config: dict[str, Any], feed: Any = None):
        self.config = config
        self.feed = feed
        self.experiment_config = MLExperimentConfig.from_config(config)
        self.research_label = str(
            config.get("ml", {}).get("research_label", "UNSPECIFIED_RESEARCH")
        )
        self._champion_equity_curve = []
        self._champion_rebalance_dates: set[str] = set()
        self._champion_selections: list[Any] = []
        self._history_data_metadata: dict[str, dict] = {}

    def build_expanded_rebalance_dataset(self) -> tuple[Path, Path, int]:
        feature_result, candles_by_symbol = self._build_features()
        expanded = self._build_expanded_rebalance_features(
            feature_result,
            candles_by_symbol,
        )
        dataset_path = self._rebalance_dataset_path()
        audit_path = Path(
            self.config.get("ml", {}).get(
                "expanded_rebalance_audit_path",
                "reports/ml/expanded_rebalance_dataset_audit.json",
            )
        )
        return dataset_path, audit_path, len(expanded.rows)

    def run(self) -> MLExperimentResult:
        feature_result, candles_by_symbol = self._build_features()
        if self.experiment_config.label_type == "should_reduce_exposure":
            feature_result = self._build_expanded_rebalance_features(
                feature_result,
                candles_by_symbol,
            )
        label_result = self._build_labels(feature_result, candles_by_symbol)
        prepared_dataset = self._dataset_pipeline().prepare(
            feature_result,
            label_result,
        )
        dataset = prepared_dataset.dataset
        split = prepared_dataset.split
        model = self._model_pipeline().build_model()
        self._fit_research_model(model, split.train)
        probabilities, auxiliary_predictions = self._predict_research_model(
            model,
            split.test,
            prediction_context=self._prediction_context(split),
        )
        predictions = self._predictions_from_probabilities(probabilities)

        paths = self._experiment_paths()

        write_feature_rows(paths.features_path, feature_result.rows)
        write_label_rows(paths.labels_path, label_result.rows, label_result.label_name)
        write_dataset(paths.dataset_path, dataset, label_name=label_result.label_name)
        self._write_feature_summary(paths.feature_summary_path, feature_result)
        self._write_dataset_audit(paths.dataset_audit_path, dataset, label_result)
        self._write_walk_forward_metrics(paths.walk_forward_metrics_path, dataset)
        self._write_probability_calibration(
            paths.probability_calibration_path,
            split.test.labels,
            probabilities,
        )
        self._write_calibrated_probability_calibration(
            paths.calibrated_probability_calibration_path,
            split,
            probabilities,
        )
        self._write_walk_forward_probability_calibration(
            paths.walk_forward_probability_calibration_path,
            dataset,
        )
        self._write_baseline_model_comparison(
            paths.baseline_model_comparison_path,
            dataset,
        )
        self._write_ranking_diagnostics(
            paths.ranking_diagnostics_path,
            dataset,
            self._outcomes_by_feature_date(label_result, candles_by_symbol),
        )
        self._write_threshold_sweep(
            paths.threshold_sweep_path,
            split.test,
            probabilities,
        )
        self._write_model_comparison(paths.model_comparison_path, dataset)
        self._write_shadow_overlay(paths.shadow_overlay_path, dataset)
        self._write_overlay_model_comparison(
            paths.overlay_model_comparison_path,
            dataset,
        )
        self._write_holdout_shadow_overlay(
            paths.holdout_shadow_overlay_path,
            split,
        )
        rebalance_rows = self._write_rebalance_dataset(
            paths.rebalance_dataset_path,
            paths.rebalance_dataset_audit_path,
            feature_result.rows,
            candles_by_symbol,
            paths.rule_exposure_study_path,
        )
        write_drawdown_event_review(
            paths.drawdown_event_review_path,
            rebalance_rows,
        )

        prediction_artifact_provenance = self._prediction_artifact_provenance(
            dataset,
            split,
        )
        self._write_metrics(paths.metrics_path, dataset, split, predictions)
        self._write_predictions(
            paths.predictions_path,
            split.test,
            predictions,
            probabilities,
        )
        self._write_feature_importance(
            paths.feature_importance_path,
            model.feature_importances(),
        )
        self._write_confusion_matrix(
            paths.confusion_matrix_path,
            split.test,
            predictions,
        )
        self._write_metadata(paths.metadata_path, dataset, split)
        self._write_prediction_artifacts(
            paths.prediction_artifacts_path,
            paths.prediction_artifacts_metadata_path,
            dataset,
            split,
            probabilities,
            auxiliary_predictions,
            dataset_hash=str(prediction_artifact_provenance["dataset_hash"]),
            source_dataset_row_count=int(
                prediction_artifact_provenance["source_dataset_row_count"]
            ),
            train_sample_count=int(prediction_artifact_provenance["train_sample_count"]),
            test_sample_count=int(prediction_artifact_provenance["test_sample_count"]),
            generated_at=str(prediction_artifact_provenance["generated_at"]),
        )
        model.save(paths.model_path)
        self._annotate_report_artifacts(paths.output_dir)
        write_research_html_report(paths.html_report_path, paths.output_dir)

        return MLExperimentResult(**paths.result_kwargs())
