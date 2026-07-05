from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from core.research.ml.artifacts.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.data.datasets import MLDataset
from core.research.ml.validation import ChronologicalSplit, rolling_walk_forward


class MLPredictionArtifactWriterMixin:
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
