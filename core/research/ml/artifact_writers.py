from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from core.research.ml.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.evaluation import classification_metrics
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.labels import MLLabelBuildResult
from core.research.ml.model_pipeline import MLModelPipeline
from core.research.ml.validation import ChronologicalSplit, rolling_walk_forward


class MLCoreArtifactWriter:
    """Write core ML experiment artifacts without controlling the experiment flow."""

    def __init__(
        self,
        config: Mapping[str, Any],
        experiment_config: MLExperimentConfig,
        *,
        research_label: str,
        model_pipeline: MLModelPipeline | None = None,
    ) -> None:
        self._config = config
        self._experiment_config = experiment_config
        self._research_label = research_label
        self._model_pipeline = model_pipeline or MLModelPipeline(
            config,
            experiment_config,
        )

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
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": self.hash_payload(self._config),
            "data_hash": dataset_hash,
            "dataset_hash": dataset_hash,
            "source_dataset_row_count": dataset.sample_count,
            "git_commit": self.git_commit(),
            "model_type": self._experiment_config.model_type,
            "feature_set": self._experiment_config.feature_set,
            "label_type": self._experiment_config.label_type,
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
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_prediction_artifacts(
        self,
        csv_path: Path,
        metadata_path: Path,
        dataset: MLDataset,
        split: ChronologicalSplit,
        holdout_probabilities: list[float],
        holdout_auxiliary_predictions: list[dict[str, float]] | None = None,
        *,
        dataset_hash: str | None = None,
        source_dataset_row_count: int | None = None,
        train_sample_count: int | None = None,
        test_sample_count: int | None = None,
        generated_at: str | None = None,
    ) -> None:
        rows = []
        provenance = self.prediction_artifact_provenance(
            dataset,
            split,
            dataset_hash=dataset_hash,
            source_dataset_row_count=source_dataset_row_count,
            train_sample_count=train_sample_count,
            test_sample_count=test_sample_count,
            generated_at=generated_at,
        )
        provenance = {
            "source_dataset_row_count": int(provenance["source_dataset_row_count"]),
            "train_sample_count": int(provenance["train_sample_count"]),
            "test_sample_count": int(provenance["test_sample_count"]),
            "generated_at": str(provenance["generated_at"]),
            "dataset_hash": str(provenance["dataset_hash"]),
        }
        for fold in rolling_walk_forward(
            dataset,
            self._experiment_config.walk_forward_folds,
        ):
            model = self._model_pipeline.build_model()
            self._model_pipeline.fit(model, fold.split.train)
            prediction = self._model_pipeline.predict(
                model,
                fold.split.test,
                prediction_context=self._model_pipeline.prediction_context(fold.split),
            )
            rows.extend(
                self.prediction_artifact_rows(
                    fold.split.test,
                    prediction.probabilities,
                    prediction.auxiliary_predictions,
                    split_name="out_of_fold",
                    fold=fold.fold_number,
                    provenance=provenance,
                )
            )
        rows.extend(
            self.prediction_artifact_rows(
                split.test,
                holdout_probabilities,
                holdout_auxiliary_predictions,
                split_name="holdout",
                fold="holdout",
                provenance=provenance,
            )
        )
        auxiliary_fieldnames = self.prediction_artifact_auxiliary_fieldnames(rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "artifact_schema_version",
            "profile",
            "model_name",
            "date",
            "prediction_date",
            "symbol",
            "rebalance_date",
            "feature_id",
            "variant_id",
            "config_path",
            "model_type",
            "label_type",
            "split",
            "fold",
            "actual_label",
            "predicted_probability",
            "raw_probability",
            "calibrated_probability",
            "prediction",
            "decision_threshold",
            "source_dataset_row_count",
            "train_sample_count",
            "test_sample_count",
            "generated_at",
            "dataset_hash",
            "research_label",
            *auxiliary_fieldnames,
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        metadata_path.write_text(json.dumps({
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "profile": self._ml_config().get("profile", ""),
            "model_name": self.prediction_artifact_model_name(),
            "model_type": self._experiment_config.model_type,
            "label_type": self._experiment_config.label_type,
            "feature_set": self._experiment_config.feature_set,
            "config_path": self._ml_config().get("config_path", ""),
            "config_hash": self.hash_payload(self._config),
            "data_hash": provenance["dataset_hash"],
            "dataset_hash": provenance["dataset_hash"],
            "source_dataset_row_count": provenance["source_dataset_row_count"],
            "train_sample_count": provenance["train_sample_count"],
            "test_sample_count": provenance["test_sample_count"],
            "generated_at": provenance["generated_at"],
            "git_commit": self.git_commit(),
            "validation_method": "rolling_walk_forward_out_of_fold_plus_holdout",
            "row_count": len(rows),
            "auxiliary_targets": self._model_pipeline.multitask_regression_targets(),
            "auxiliary_prediction_columns": [
                name for name in auxiliary_fieldnames if name.startswith("predicted_")
            ],
            "auxiliary_actual_columns": [
                name for name in auxiliary_fieldnames if name.startswith("actual_")
            ],
            "trading_impact": "none",
            "research_only": True,
        }, indent=2), encoding="utf-8")

    def prediction_artifact_provenance(
        self,
        dataset: MLDataset,
        split: ChronologicalSplit,
        *,
        dataset_hash: str | None = None,
        source_dataset_row_count: int | None = None,
        train_sample_count: int | None = None,
        test_sample_count: int | None = None,
        generated_at: str | None = None,
    ) -> dict[str, str | int]:
        return {
            "dataset_hash": dataset_hash or self.dataset_hash(dataset),
            "source_dataset_row_count": (
                dataset.sample_count
                if source_dataset_row_count is None
                else source_dataset_row_count
            ),
            "train_sample_count": (
                split.train.sample_count
                if train_sample_count is None
                else train_sample_count
            ),
            "test_sample_count": (
                split.test.sample_count
                if test_sample_count is None
                else test_sample_count
            ),
            "generated_at": generated_at or datetime.utcnow().isoformat() + "Z",
        }

    def prediction_artifact_rows(
        self,
        dataset: MLDataset,
        probabilities: list[float],
        auxiliary_predictions: list[dict[str, float]] | None,
        split_name: str,
        fold: int | str,
        provenance: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        provenance = provenance or {}
        auxiliary_predictions = auxiliary_predictions or [{} for _ in probabilities]
        for index, probability in enumerate(probabilities):
            metadata = dataset.metadata[index] if dataset.metadata else {}
            feature_id = (
                dataset.feature_ids[index]
                if dataset.feature_ids
                else dataset.feature_dates[index]
            )
            row = {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "profile": self._ml_config().get("profile", ""),
                "model_name": self.prediction_artifact_model_name(),
                "date": dataset.feature_dates[index],
                "prediction_date": dataset.feature_dates[index],
                "symbol": metadata.get("symbol", ""),
                "rebalance_date": metadata.get(
                    "rebalance_date",
                    dataset.feature_dates[index],
                ),
                "feature_id": feature_id,
                "variant_id": metadata.get("variant_id", ""),
                "config_path": self._ml_config().get("config_path", ""),
                "model_type": self._experiment_config.model_type,
                "label_type": self._experiment_config.label_type,
                "split": split_name,
                "fold": fold,
                "actual_label": dataset.labels[index],
                "predicted_probability": float(probability),
                "raw_probability": float(probability),
                "calibrated_probability": "",
                "prediction": int(
                    probability >= self._experiment_config.decision_threshold
                ),
                "decision_threshold": self._experiment_config.decision_threshold,
                "source_dataset_row_count": provenance.get(
                    "source_dataset_row_count", ""
                ),
                "train_sample_count": provenance.get("train_sample_count", ""),
                "test_sample_count": provenance.get("test_sample_count", ""),
                "generated_at": provenance.get("generated_at", ""),
                "dataset_hash": provenance.get("dataset_hash", ""),
                "research_label": self._research_label,
            }
            row.update(
                self.prediction_artifact_auxiliary_values(
                    dataset,
                    index,
                    auxiliary_predictions[index]
                    if index < len(auxiliary_predictions)
                    else {},
                )
            )
            rows.append(row)
        return rows

    def prediction_artifact_model_name(self) -> str:
        ml_config = self._ml_config()
        return str(
            ml_config.get("model_name")
            or ml_config.get("research_label")
            or self._experiment_config.model_type
        )

    def prediction_artifact_auxiliary_values(
        self,
        dataset: MLDataset,
        index: int,
        auxiliary_prediction: dict[str, float],
    ) -> dict[str, float | str]:
        values: dict[str, float | str] = {}
        targets = self._model_pipeline.multitask_regression_targets()
        actuals = dataset.auxiliary_targets[index] if dataset.auxiliary_targets else {}
        for target in targets:
            prediction_key = f"predicted_{target}"
            actual_key = f"actual_{target}"
            if prediction_key in auxiliary_prediction:
                values[prediction_key] = float(auxiliary_prediction[prediction_key])
            else:
                values[prediction_key] = ""
            actual_value = actuals.get(target) if actuals else None
            values[actual_key] = "" if actual_value is None else float(actual_value)
        for key, value in auxiliary_prediction.items():
            if key.startswith("predicted_") and key not in values:
                values[key] = float(value)
        return values

    @staticmethod
    def prediction_artifact_auxiliary_fieldnames(
        rows: list[dict[str, Any]],
    ) -> list[str]:
        names: list[str] = []
        for row in rows:
            for name in row:
                if name == "actual_label":
                    continue
                if name == "predicted_probability":
                    continue
                if (
                    (name.startswith("predicted_") or name.startswith("actual_"))
                    and name not in names
                ):
                    names.append(name)
        return sorted(names)

    def dataset_hash(self, dataset: MLDataset) -> str:
        return self.source_dataset_hash(dataset)

    def source_dataset_hash(self, dataset: MLDataset) -> str:
        return self.hash_payload(self.source_dataset_identity(dataset))

    def source_dataset_identity(self, dataset: MLDataset) -> dict[str, Any]:
        rows = []
        for index in range(dataset.sample_count):
            metadata = dataset.metadata[index] if index < len(dataset.metadata) else {}
            rows.append({
                "feature_date": dataset.feature_dates[index],
                "label_start_date": dataset.label_start_dates[index],
                "label_end_date": dataset.label_end_dates[index],
                "label": dataset.labels[index],
                "rebalance_date": metadata.get(
                    "rebalance_date",
                    dataset.feature_dates[index],
                ),
                "variant_id": metadata.get("variant_id", ""),
                "symbol": metadata.get("symbol", ""),
                "selected_symbols": metadata.get("selected_symbols", ""),
                "variant_universe": metadata.get("variant_universe", ""),
                "variant_rebalance_frequency": metadata.get(
                    "variant_rebalance_frequency",
                    "",
                ),
                "variant_weighting": metadata.get("variant_weighting", ""),
            })
        rows.sort(key=lambda row: tuple(str(value) for value in row.values()))
        return {
            "label_type": self._experiment_config.label_type,
            "rows": rows,
        }

    def model_input_hash(self, dataset: MLDataset) -> str:
        return self.hash_payload({
            "features": dataset.features,
            "labels": dataset.labels,
            "feature_ids": dataset.feature_ids,
            "feature_dates": dataset.feature_dates,
            "label_start_dates": dataset.label_start_dates,
            "label_end_dates": dataset.label_end_dates,
            "auxiliary_targets": dataset.auxiliary_targets,
        })

    @staticmethod
    def hash_payload(payload: Any) -> str:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    @staticmethod
    def standard_deviation(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        average = sum(values) / len(values)
        return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5

    @staticmethod
    def correlation(left: list[float], right: list[float]) -> float:
        if len(left) < 2 or len(left) != len(right):
            return 0.0
        left_average = sum(left) / len(left)
        right_average = sum(right) / len(right)
        numerator = sum(
            (left_value - left_average) * (right_value - right_average)
            for left_value, right_value in zip(left, right)
        )
        left_scale = sum((value - left_average) ** 2 for value in left) ** 0.5
        right_scale = sum((value - right_average) ** 2 for value in right) ** 0.5
        if left_scale == 0 or right_scale == 0:
            return 0.0
        return numerator / (left_scale * right_scale)

    @staticmethod
    def is_numeric_column(
        rows: list[dict[str, float | str]],
        name: str,
    ) -> bool:
        for row in rows:
            try:
                float(row[name])
            except (KeyError, TypeError, ValueError):
                return False
        return True

    def _ml_config(self) -> Mapping[str, Any]:
        return self._config.get("ml", {}) or {}
