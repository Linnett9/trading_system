from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

from core.research.ml.data.datasets import MODEL_INPUT_CONTRACT_VERSION, MLDataset
from core.research.ml.features.features import MLFeatureBuildResult
from core.research.ml.features.labels import MLLabelBuildResult
from core.research.ml.metrics.evaluation import classification_metrics
from core.research.ml.validation import ChronologicalSplit


class MLArtifactBasicWritersMixin:
    def write_feature_summary(
        self,
        path: Path,
        feature_result: MLFeatureBuildResult,
    ) -> None:
        rows = feature_result.rows
        numeric_columns = [
            name for name in (rows[0] if rows else {})
            if name != "feature_date" and self.is_numeric_column(rows, name)
        ]
        summary = {
            "row_count": len(rows),
            "dropped_rows_insufficient_lookback": feature_result.dropped_rows,
            "date_range": feature_result.date_range,
            "missing_values": {
                name: sum(row.get(name) is None for row in rows)
                for name in numeric_columns
            },
            "means": {
                name: sum(float(row[name]) for row in rows) / len(rows)
                for name in numeric_columns
            } if rows else {},
            "standard_deviations": {
                name: self.standard_deviation([float(row[name]) for row in rows])
                for name in numeric_columns
            } if rows else {},
            "correlation_matrix": {
                left: {
                    right: self.correlation(
                        [float(row[left]) for row in rows],
                        [float(row[right]) for row in rows],
                    )
                    for right in numeric_columns
                }
                for left in numeric_columns
            } if rows else {},
        }
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def write_dataset_audit(
        self,
        path: Path,
        dataset: MLDataset,
        label_result: MLLabelBuildResult,
    ) -> None:
        positive_labels = sum(dataset.labels)
        sample_count = dataset.sample_count
        payload = {
            "sample_count": sample_count,
            "feature_count": dataset.feature_count,
            "date_coverage": (
                [dataset.feature_dates[0], dataset.feature_dates[-1]]
                if dataset.feature_dates
                else None
            ),
            "class_balance": {
                "positive": positive_labels,
                "negative": sample_count - positive_labels,
                "positive_rate": positive_labels / sample_count if sample_count else None,
            },
            "dropped_rows_insufficient_label_horizon": (
                label_result.dropped_rows_insufficient_horizon
            ),
            "leakage_check_passed": all(
                feature_date < label_start <= label_end
                for feature_date, label_start, label_end in zip(
                    dataset.feature_dates,
                    dataset.label_start_dates,
                    dataset.label_end_dates,
                )
            ),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    def write_metrics(
        self,
        path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
        predictions: list[int],
    ) -> None:
        metrics = classification_metrics(split.test.labels, predictions)
        dataset_hash = self.source_dataset_hash(dataset)
        payload = {
            "mode": "research",
            "model_type": self._experiment_config.model_type,
            "feature_set": self._experiment_config.feature_set,
            "label_type": self._experiment_config.label_type,
            "decision_threshold": self._experiment_config.decision_threshold,
            "class_weight": self._model_pipeline.class_weight(),
            "train_sample_count": split.train.sample_count,
            "test_sample_count": split.test.sample_count,
            "source_dataset_row_count": dataset.sample_count,
            "dataset_hash": dataset_hash,
            "feature_count": split.train.feature_count,
            "test_start_date": split.test_start_date,
            "purged_train_samples": split.purged_train_samples,
            "metrics": metrics,
            "baselines": self.baseline_metrics(split),
            "note": (
                "Research-only out-of-sample evaluation; ML does not affect "
                "trading decisions."
            ),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    def baseline_metrics(self, split: ChronologicalSplit) -> dict[str, dict]:
        no_op_predictions = [0] * split.test.sample_count
        majority_class = int(
            sum(split.train.labels) >= (split.train.sample_count / 2)
        ) if split.train.sample_count else 0
        majority_predictions = [majority_class] * split.test.sample_count
        return {
            "noop": classification_metrics(split.test.labels, no_op_predictions),
            "majority_class": {
                "predicted_class": majority_class,
                "metrics": classification_metrics(
                    split.test.labels,
                    majority_predictions,
                ),
            },
        }
    def write_predictions(
        self,
        path: Path,
        dataset: MLDataset,
        predictions: list[int],
        probabilities: list[float],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "row",
                    "feature_date",
                    "label_start_date",
                    "label_end_date",
                    "prediction",
                    "probability",
                    "label",
                ],
            )
            writer.writeheader()
            for index, prediction in enumerate(predictions):
                writer.writerow({
                    "row": index,
                    "feature_date": dataset.feature_dates[index],
                    "label_start_date": dataset.label_start_dates[index],
                    "label_end_date": dataset.label_end_dates[index],
                    "prediction": prediction,
                    "probability": probabilities[index],
                    "label": dataset.labels[index],
                })
    def write_feature_importance(
        self,
        path: Path,
        feature_importances: dict[str, float],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["feature", "importance"])
            writer.writeheader()
            for feature, importance in sorted(
                feature_importances.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                writer.writerow({"feature": feature, "importance": importance})
    def write_confusion_matrix(
        self,
        path: Path,
        dataset: MLDataset,
        predictions: list[int],
    ) -> None:
        counts = {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for actual, prediction in zip(dataset.labels, predictions):
            if actual == prediction == 1:
                counts["true_positive"] += 1
            elif actual == prediction == 0:
                counts["true_negative"] += 1
            elif actual == 0 and prediction == 1:
                counts["false_positive"] += 1
            elif actual == 1 and prediction == 0:
                counts["false_negative"] += 1

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["bucket", "count"])
            writer.writeheader()
            for bucket, count in counts.items():
                writer.writerow({"bucket": bucket, "count": count})
    def write_metadata(
        self,
        path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
    ) -> None:
        dataset_hash = self.source_dataset_hash(dataset)
        feature_columns = self.model_input_feature_columns(dataset)
        feature_date_min = min(dataset.feature_dates) if dataset.feature_dates else None
        feature_date_max = max(dataset.feature_dates) if dataset.feature_dates else None
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": self.hash_payload(self._config),
            "data_hash": dataset_hash,
            "dataset_hash": dataset_hash,
            "model_input_contract_version": MODEL_INPUT_CONTRACT_VERSION,
            "model_input_hash": self.model_input_hash(dataset),
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "sample_count": dataset.sample_count,
            "feature_date_min": feature_date_min,
            "feature_date_max": feature_date_max,
            "training_date_min": feature_date_min,
            "training_date_max": feature_date_max,
            "model_input_source_path": self.model_input_source_path(),
            "source_dataset_row_count": dataset.sample_count,
            "git_commit": self.git_commit(),
            "model_name": self._experiment_config.model_type,
            "model_type": self._experiment_config.model_type,
            "feature_set": self._experiment_config.feature_set,
            "label_type": self._experiment_config.label_type,
            "target_label_name": self._experiment_config.label_type,
            "random_seed": self._experiment_config.random_seed,
            "experiment_config": self._experiment_config.to_dict(),
            "validation": {
                "method": "purged_chronological_holdout",
                "train_sample_count": split.train.sample_count,
                "test_sample_count": split.test.sample_count,
                "test_start_date": split.test_start_date,
                "purged_train_samples": split.purged_train_samples,
            },
            "research_only": True,
            "run_status": "complete",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
