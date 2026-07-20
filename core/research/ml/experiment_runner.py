from __future__ import annotations

import json
import logging
import csv
from pathlib import Path
from typing import Any

from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import MLDataset, write_dataset
from core.research.ml.exposure_input import validate_exposure_input_resolution
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
from core.research.ml.immutable_runs import (
    deterministic_run_id,
    preserve_immutable_run,
)
from core.research.ml.models.torch_checkpointing import checkpoint_enabled
from core.research.ml.validation import ChronologicalSplit

LOGGER = logging.getLogger(__name__)


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
            self._exposure_input_identity = validate_exposure_input_resolution(
                self.config
            )
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
        temporal_audit: dict[str, Any] | None = None
        if self.experiment_config.label_type == "should_reduce_exposure":
            model, split, probabilities, auxiliary_predictions, temporal_audit = (
                self._fit_predict_exposure_walk_forward(dataset)
            )
        else:
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
        if temporal_audit is not None:
            self._write_exposure_temporal_audit(paths.output_dir, temporal_audit)
        model.save(paths.model_path)
        if checkpoint_enabled(model) and paths.model_path.suffix == ".pt":
            final_model_path = paths.output_dir / "final_model.pt"
            model.save(final_model_path)
            LOGGER.info(
                "torch_checkpoint_event",
                extra={
                    "event": "final_inference_model_written",
                    "model_type": self.experiment_config.model_type,
                    "path": str(final_model_path),
                },
            )
        self._annotate_report_artifacts(paths.output_dir)
        write_research_html_report(paths.html_report_path, paths.output_dir)
        self._preserve_immutable_exposure_run(paths)

        return MLExperimentResult(**paths.result_kwargs())

    def _fit_predict_exposure_walk_forward(
        self,
        dataset: MLDataset,
    ) -> tuple[Any, ChronologicalSplit, list[float], list[dict[str, float]], dict[str, Any]]:
        minimum_rows = int(self.config.get("ml", {}).get("exposure_minimum_training_rows", 3))
        minimum_positive = int(self.config.get("ml", {}).get("exposure_minimum_positive_labels", 1))
        minimum_negative = int(self.config.get("ml", {}).get("exposure_minimum_negative_labels", 1))
        unique_dates = sorted(set(dataset.feature_dates))
        probabilities_by_index: dict[int, float] = {}
        auxiliary_by_index: dict[int, dict[str, float]] = {}
        train_index_union: set[int] = set()
        fold_rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        last_model: Any | None = None
        for fold_id, decision_timestamp in enumerate(unique_dates, start=1):
            decision_indices = [
                index for index, value in enumerate(dataset.feature_dates)
                if value == decision_timestamp
            ]
            train_indices = [
                index for index in range(dataset.sample_count)
                if dataset.feature_dates[index] < decision_timestamp
                and self._label_available_timestamp(dataset, index) <= decision_timestamp
            ]
            positives = sum(dataset.labels[index] for index in train_indices)
            negatives = len(train_indices) - positives
            if (
                len(train_indices) < minimum_rows
                or positives < minimum_positive
                or negatives < minimum_negative
            ):
                skipped.append({
                    "fold_id": fold_id,
                    "decision_timestamp": decision_timestamp,
                    "skipped_row_count": len(decision_indices),
                    "reason": "minimum_mature_history_not_met",
                    "candidate_training_row_count": len(train_indices),
                    "positive_labels": positives,
                    "negative_labels": negatives,
                })
                continue
            train = self._slice_dataset(dataset, train_indices)
            test = self._slice_dataset(dataset, decision_indices)
            model = self._model_pipeline().build_model()
            self._fit_research_model(model, train)
            prediction = self._model_pipeline().predict(
                model,
                test,
                prediction_context=self._concat_datasets(train, test),
            )
            for index, probability, auxiliary in zip(
                decision_indices,
                prediction.probabilities,
                prediction.auxiliary_predictions,
            ):
                probabilities_by_index[index] = probability
                auxiliary_by_index[index] = auxiliary
            train_index_union.update(train_indices)
            last_model = model
            fold_rows.append({
                "workflow": "exposure_base_model_walk_forward",
                "model_name": self.experiment_config.model_type,
                "fold_id": fold_id,
                "decision_timestamp": decision_timestamp,
                "retraining_occurred": True,
                "training_row_count": len(train_indices),
                "training_date_minimum": min(dataset.feature_dates[index] for index in train_indices),
                "training_date_maximum": max(dataset.feature_dates[index] for index in train_indices),
                "maximum_training_label_available_timestamp": max(
                    self._label_available_timestamp(dataset, index) for index in train_indices
                ),
                "validation_row_count": 0,
                "validation_date_minimum": None,
                "validation_date_maximum": None,
                "maximum_validation_label_available_timestamp": None,
                "purged_row_count": sum(
                    1 for index in range(dataset.sample_count)
                    if dataset.feature_dates[index] < decision_timestamp
                    and self._label_available_timestamp(dataset, index) > decision_timestamp
                ),
                "embargoed_row_count": 0,
                "skipped_row_count": 0,
                "preprocessing_fit_row_count": len(train_indices),
                "oos_prediction_row_count": len(decision_indices),
                "leakage_assertions": {
                    "training_labels_matured": True,
                    "decision_rows_excluded_from_training": not set(decision_indices) & set(train_indices),
                    "validation_rows_excluded_from_training": True,
                },
            })
        if not probabilities_by_index or last_model is None:
            raise ValueError("Exposure walk-forward produced no OOS predictions")
        ordered_prediction_indices = sorted(probabilities_by_index, key=lambda index: (dataset.feature_dates[index], dataset.feature_ids[index] if dataset.feature_ids else str(index)))
        split = ChronologicalSplit(
            train=self._slice_dataset(dataset, sorted(train_index_union)),
            test=self._slice_dataset(dataset, ordered_prediction_indices),
            test_start_date=dataset.feature_dates[ordered_prediction_indices[0]],
            purged_train_samples=sum(row["purged_row_count"] for row in fold_rows),
        )
        temporal_audit = {
            "version": 1,
            "workflow": "exposure_base_model_walk_forward",
            "temporal_policy": {
                "training_window_type": "expanding",
                "training_eligibility_rule": "label_available_timestamp <= decision_timestamp",
                "minimum_training_rows": minimum_rows,
                "minimum_positive_labels": minimum_positive,
                "minimum_negative_labels": minimum_negative,
                "validation_policy": "none",
            },
            "checkpoint_identity_policy": {
                "fold_specific": True,
                "identity_fields": [
                    "dataset_hash",
                    "model_input_hash",
                    "feature_columns",
                    "resolved_config_hash",
                    "run_identity",
                    "temporal_policy",
                ],
                "resume_across_different_decision_fold_allowed": False,
            },
            "folds": fold_rows,
            "skipped_decisions": skipped,
            "leakage_checks_passed": all(
                row["maximum_training_label_available_timestamp"] <= row["decision_timestamp"]
                and row["leakage_assertions"]["decision_rows_excluded_from_training"]
                for row in fold_rows
            ),
        }
        return (
            last_model,
            split,
            [probabilities_by_index[index] for index in ordered_prediction_indices],
            [auxiliary_by_index.get(index, {}) for index in ordered_prediction_indices],
            temporal_audit,
        )

    @staticmethod
    def _label_available_timestamp(dataset: MLDataset, index: int) -> str:
        if dataset.metadata and index < len(dataset.metadata):
            value = dataset.metadata[index].get("label_available_timestamp")
            if value:
                return str(value)
        return dataset.label_end_dates[index]

    @staticmethod
    def _slice_dataset(dataset: MLDataset, indices: list[int]) -> MLDataset:
        return MLDataset(
            features=[dataset.features[index] for index in indices],
            labels=[dataset.labels[index] for index in indices],
            feature_dates=[dataset.feature_dates[index] for index in indices],
            label_start_dates=[dataset.label_start_dates[index] for index in indices],
            label_end_dates=[dataset.label_end_dates[index] for index in indices],
            feature_ids=[dataset.feature_ids[index] for index in indices] if dataset.feature_ids else [],
            metadata=[dataset.metadata[index] for index in indices] if dataset.metadata else [],
            auxiliary_targets=[dataset.auxiliary_targets[index] for index in indices] if dataset.auxiliary_targets else [],
        )

    @staticmethod
    def _write_exposure_temporal_audit(output_dir: Path, audit: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "exposure_temporal_audit.json").write_text(
            json.dumps(audit, indent=2),
            encoding="utf-8",
        )
        rows = audit.get("folds", [])
        if not rows:
            return
        path = output_dir / "exposure_temporal_folds.csv"
        fieldnames = [key for key in rows[0] if key != "leakage_assertions"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    def _preserve_immutable_exposure_run(self, paths: Any) -> None:
        if self.experiment_config.label_type != "should_reduce_exposure":
            return
        try:
            metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(metadata, dict) or metadata.get("run_status") != "complete":
            return
        identity_keys = (
            "config_hash",
            "dataset_hash",
            "model_input_contract_version",
            "model_input_hash",
            "feature_columns",
            "feature_count",
            "sample_count",
            "feature_date_min",
            "feature_date_max",
            "training_date_min",
            "training_date_max",
            "target_label_name",
            "label_type",
            "model_name",
            "model_type",
            "feature_set",
            "model_input_source_path",
        )
        identity = {key: metadata.get(key) for key in identity_keys}
        run_id = deterministic_run_id("exposure_ml", identity)
        preserve_immutable_run(
            output_dir=paths.output_dir,
            run_id=run_id,
            kind="exposure_ml",
            identity=identity,
            artifact_paths=(
                paths.metrics_path,
                paths.metadata_path,
                paths.predictions_path,
                paths.model_path,
                paths.prediction_artifacts_path,
                paths.prediction_artifacts_metadata_path,
                paths.dataset_audit_path,
            ),
            extra_manifest={
                "model_type": self.experiment_config.model_type,
                "label_type": self.experiment_config.label_type,
            },
        )
