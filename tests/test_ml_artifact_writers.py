from __future__ import annotations

import csv
import json

import pytest

from core.research.ml.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.artifact_writers import MLCoreArtifactWriter
from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.labels import MLLabelBuildResult
from core.research.ml.validation import ChronologicalSplit


def test_core_artifact_writer_preserves_summary_audit_and_holdout_files(tmp_path):
    writer = _writer()
    feature_result = MLFeatureBuildResult(
        rows=[
            {"feature_date": "2024-01-01", "alpha": 1.0, "beta": 2.0},
            {"feature_date": "2024-01-02", "alpha": 3.0, "beta": 4.0},
        ],
        dropped_rows=5,
        date_range=("2024-01-01", "2024-01-02"),
    )
    dataset = _dataset(
        features=[{"alpha": 1.0}, {"alpha": 3.0}],
        labels=[1, 0],
        feature_dates=["2024-01-01", "2024-01-02"],
    )
    split = ChronologicalSplit(
        train=_dataset(
            features=[{"alpha": 1.0}],
            labels=[1],
            feature_dates=["2024-01-01"],
        ),
        test=_dataset(
            features=[{"alpha": 3.0}],
            labels=[0],
            feature_dates=["2024-01-02"],
        ),
        test_start_date="2024-01-02",
        purged_train_samples=0,
    )
    label_result = MLLabelBuildResult(
        rows=[],
        dropped_rows_insufficient_horizon=2,
        label_name="risk_regime",
    )

    writer.write_feature_summary(tmp_path / "feature_summary.json", feature_result)
    writer.write_dataset_audit(tmp_path / "dataset_audit.json", dataset, label_result)
    writer.write_metrics(tmp_path / "metrics.json", dataset, split, [0])
    writer.write_predictions(tmp_path / "predictions.csv", split.test, [0], [0.4])
    writer.write_feature_importance(
        tmp_path / "feature_importance.csv",
        {"slow": 0.2, "fast": 0.9},
    )
    writer.write_confusion_matrix(tmp_path / "confusion_matrix.csv", split.test, [0])
    writer.write_metadata(tmp_path / "metadata.json", dataset, split)

    summary = json.loads((tmp_path / "feature_summary.json").read_text())
    assert summary["row_count"] == 2
    assert summary["dropped_rows_insufficient_lookback"] == 5
    assert summary["standard_deviations"]["alpha"] == 1.0
    assert summary["correlation_matrix"]["alpha"]["beta"] == pytest.approx(1.0)

    audit = json.loads((tmp_path / "dataset_audit.json").read_text())
    assert audit["sample_count"] == 2
    assert audit["class_balance"]["positive_rate"] == 0.5
    assert audit["dropped_rows_insufficient_label_horizon"] == 2
    assert audit["leakage_check_passed"] is False

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["mode"] == "research"
    assert metrics["class_weight"] == "balanced"
    assert metrics["baselines"]["majority_class"]["predicted_class"] == 1

    with (tmp_path / "predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {
                "row": "0",
                "feature_date": "2024-01-02",
                "label_start_date": "2024-01-02",
                "label_end_date": "2024-01-02",
                "prediction": "0",
                "probability": "0.4",
                "label": "0",
            }
        ]

    with (tmp_path / "feature_importance.csv").open("r", encoding="utf-8", newline="") as handle:
        assert [row["feature"] for row in csv.DictReader(handle)] == ["fast", "slow"]

    with (tmp_path / "confusion_matrix.csv").open("r", encoding="utf-8", newline="") as handle:
        confusion_rows = list(csv.DictReader(handle))
    assert confusion_rows == [
        {"bucket": "true_positive", "count": "0"},
        {"bucket": "true_negative", "count": "1"},
        {"bucket": "false_positive", "count": "0"},
        {"bucket": "false_negative", "count": "0"},
    ]

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["dataset_hash"] == metrics["dataset_hash"]
    assert metadata["validation"]["method"] == "purged_chronological_holdout"
    assert metadata["research_only"] is True


def test_core_artifact_writer_preserves_prediction_artifact_schema(tmp_path):
    config = {
        "ml": {
            "model_type": "noop",
            "walk_forward_folds": 1,
            "decision_threshold": 0.5,
            "profile": "unit",
            "config_path": "config.yaml",
            "multitask_regression_targets": ["future_drawdown"],
        }
    }
    experiment_config = MLExperimentConfig.from_config(config)
    writer = MLCoreArtifactWriter(
        config,
        experiment_config,
        research_label="UNIT_RESEARCH",
    )
    dataset = _dataset(
        features=[{"alpha": float(index)} for index in range(6)],
        labels=[0, 1, 0, 1, 0, 1],
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(6)],
        metadata=[
            {"symbol": "SPY", "variant_id": f"v{index}"}
            for index in range(6)
        ],
        auxiliary_targets=[
            {"future_drawdown": -0.01 * index}
            for index in range(6)
        ],
    )
    split = ChronologicalSplit(
        train=_slice_dataset(dataset, [0, 1, 2, 3]),
        test=_slice_dataset(dataset, [4, 5]),
        test_start_date="2024-01-05",
        purged_train_samples=0,
    )

    writer.write_prediction_artifacts(
        tmp_path / "prediction_artifacts.csv",
        tmp_path / "prediction_artifacts.json",
        dataset,
        split,
        [0.25, 0.75],
        [{"predicted_future_drawdown": -0.05}, {"predicted_extra": 0.4}],
        dataset_hash="dataset-hash",
        generated_at="2026-01-01T00:00:00Z",
    )

    with (tmp_path / "prediction_artifacts.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    assert fieldnames[:6] == [
        "artifact_schema_version",
        "profile",
        "model_name",
        "date",
        "prediction_date",
        "symbol",
    ]
    assert "actual_future_drawdown" in fieldnames
    assert "predicted_extra" in fieldnames
    assert rows[-2]["split"] == "holdout"
    assert rows[-2]["predicted_probability"] == "0.25"
    assert rows[-2]["actual_future_drawdown"] == "-0.04"
    assert rows[-1]["predicted_extra"] == "0.4"
    assert {row["artifact_schema_version"] for row in rows} == {
        ARTIFACT_SCHEMA_VERSION
    }
    assert {row["dataset_hash"] for row in rows} == {"dataset-hash"}
    assert {row["research_label"] for row in rows} == {"UNIT_RESEARCH"}

    metadata = json.loads((tmp_path / "prediction_artifacts.json").read_text())
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["dataset_hash"] == "dataset-hash"
    assert metadata["validation_method"] == (
        "rolling_walk_forward_out_of_fold_plus_holdout"
    )
    assert metadata["row_count"] == len(rows)
    assert metadata["auxiliary_targets"] == ["future_drawdown"]
    assert "predicted_extra" in metadata["auxiliary_prediction_columns"]
    assert "actual_future_drawdown" in metadata["auxiliary_actual_columns"]


def _writer() -> MLCoreArtifactWriter:
    config = {"ml": {"model_type": "noop"}}
    return MLCoreArtifactWriter(
        config,
        MLExperimentConfig.from_config(config),
        research_label="UNIT_RESEARCH",
    )


def _dataset(
    features: list[dict[str, float]],
    labels: list[int],
    feature_dates: list[str],
    *,
    metadata: list[dict[str, str]] | None = None,
    auxiliary_targets: list[dict[str, float | None]] | None = None,
) -> MLDataset:
    return MLDataset(
        features=features,
        labels=labels,
        feature_dates=feature_dates,
        label_start_dates=feature_dates,
        label_end_dates=feature_dates,
        feature_ids=feature_dates,
        metadata=metadata or [{} for _ in features],
        auxiliary_targets=auxiliary_targets or [{} for _ in features],
    )


def _slice_dataset(dataset: MLDataset, indices: list[int]) -> MLDataset:
    return MLDataset(
        features=[dataset.features[index] for index in indices],
        labels=[dataset.labels[index] for index in indices],
        feature_dates=[dataset.feature_dates[index] for index in indices],
        label_start_dates=[dataset.label_start_dates[index] for index in indices],
        label_end_dates=[dataset.label_end_dates[index] for index in indices],
        feature_ids=[dataset.feature_ids[index] for index in indices],
        metadata=[dataset.metadata[index] for index in indices],
        auxiliary_targets=[dataset.auxiliary_targets[index] for index in indices],
    )
