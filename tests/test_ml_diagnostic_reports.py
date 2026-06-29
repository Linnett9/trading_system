from __future__ import annotations

import json

from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import MLDataset
from core.research.ml.pipelines import MLModelPipeline, MLModelPrediction
from core.research.ml.reports import MLDiagnosticReportWriter


def test_diagnostic_report_writer_preserves_threshold_sweep_payload(tmp_path):
    writer = _writer()
    dataset = _dataset([0, 1, 1, 0])

    writer.write_threshold_sweep(
        tmp_path / "threshold_sweep.json",
        dataset,
        [0.10, 0.30, 0.60, 0.90],
    )

    payload = json.loads((tmp_path / "threshold_sweep.json").read_text())
    assert payload["evaluation"] == "holdout_only"
    assert [row["threshold"] for row in payload["thresholds"]][:3] == [
        0.2,
        0.25,
        0.3,
    ]
    assert payload["thresholds"][-1]["threshold"] == 0.8
    assert payload["thresholds"][0]["metrics"]["samples"] == 4


def test_diagnostic_report_writer_preserves_walk_forward_metrics_payload(tmp_path):
    writer = _writer({"walk_forward_folds": 2})
    dataset = _dataset([0, 1, 0, 1, 0, 1, 0, 1])

    writer.write_walk_forward_metrics(
        tmp_path / "walk_forward_metrics.json",
        dataset,
    )

    payload = json.loads((tmp_path / "walk_forward_metrics.json").read_text())
    assert payload["model_type"] == "noop"
    assert payload["fold_count"] == 2
    assert payload["research_only"] is True
    assert payload["folds"][0] == {
        "fold": 1,
        "train_sample_count": 4,
        "test_sample_count": 2,
        "test_start_date": "2024-03-05",
        "purged_train_samples": 0,
        "metrics": {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2 / 3,
            "balanced_accuracy": 0.5,
            "samples": 2,
        },
        "baselines": {
            "noop": {
                "accuracy": 0.5,
                "precision": None,
                "recall": 0.0,
                "f1": None,
                "balanced_accuracy": 0.5,
                "samples": 2,
            },
            "majority_class": {
                "predicted_class": 1,
                "metrics": {
                    "accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 2 / 3,
                    "balanced_accuracy": 0.5,
                    "samples": 2,
                },
            },
        },
    }


def test_walk_forward_metrics_uses_model_pipeline_and_prediction_context(tmp_path):
    config = {"ml": {"model_type": "noop", "walk_forward_folds": 2}}
    model_pipeline = _RecordingModelPipeline()
    writer = MLDiagnosticReportWriter(
        config,
        MLExperimentConfig.from_config(config),
        model_pipeline=model_pipeline,
    )
    dataset = _dataset([0, 1, 0, 1, 0, 1, 0, 1])

    writer.write_walk_forward_metrics(
        tmp_path / "walk_forward_metrics.json",
        dataset,
    )

    payload = json.loads((tmp_path / "walk_forward_metrics.json").read_text())
    assert list(payload) == ["model_type", "fold_count", "folds", "research_only"]
    assert list(payload["folds"][0]) == [
        "fold",
        "train_sample_count",
        "test_sample_count",
        "test_start_date",
        "purged_train_samples",
        "metrics",
        "baselines",
    ]
    assert model_pipeline.built_model_numbers == [1, 2]
    assert model_pipeline.fit_sample_counts == [4, 6]
    assert model_pipeline.predict_sample_counts == [2, 2]
    assert model_pipeline.predict_context_sample_counts == [6, 8]
    assert model_pipeline.predict_context_start_dates == ["2024-03-01", "2024-03-01"]
    assert payload["folds"][0]["metrics"]["balanced_accuracy"] == 1.0
    assert set(payload["folds"][0]["baselines"]) == {"noop", "majority_class"}


def test_diagnostic_report_writer_preserves_model_comparison_payload(tmp_path):
    writer = _writer({"comparison_models": ["noop"], "walk_forward_folds": 2})
    dataset = _dataset([0, 1, 0, 1, 0, 1, 0, 1])

    writer.write_model_comparison(tmp_path / "model_comparison.json", dataset)

    payload = json.loads((tmp_path / "model_comparison.json").read_text())
    assert payload["evaluation"] == "purged_walk_forward"
    assert payload["fold_count"] == 2
    assert payload["research_only"] is True
    assert [model["model_type"] for model in payload["models"]] == ["noop"]
    assert payload["models"][0]["thresholds"][0] == {
        "threshold": 0.2,
        "mean_balanced_accuracy": 0.5,
        "mean_precision": 0.5,
        "mean_recall": 1.0,
    }


def test_diagnostic_report_writer_preserves_baseline_comparison_payload(tmp_path):
    writer = _writer({
        "comparison_models": ["noop"],
        "walk_forward_folds": 2,
        "rolling_base_rate_lookback_samples": 3,
    })
    dataset = _dataset([0, 1, 0, 1, 0, 1, 0, 1])

    writer.write_baseline_model_comparison(
        tmp_path / "baseline_model_comparison.json",
        dataset,
    )

    payload = json.loads((tmp_path / "baseline_model_comparison.json").read_text())
    assert payload["evaluation"] == "purged_walk_forward"
    assert payload["rolling_base_rate_lookback_samples"] == 3
    assert payload["fold_count"] == 2
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert set(payload["folds"][0]["baselines"]) == {
        "static_base_rate",
        "rolling_base_rate",
        "always_positive",
    }
    assert payload["folds"][0]["models"][0]["model_type"] == "noop"
    assert "noop" in payload["mean_metrics_by_predictor"]


def test_diagnostic_report_writer_preserves_ranking_diagnostics_payload(tmp_path):
    writer = _writer({"ranking_quantile_count": 2, "walk_forward_folds": 2})
    dataset = _dataset([0, 1, 0, 1, 0, 1, 0, 1])
    outcomes_by_date = {
        feature_date: {
            "strategy_return": index / 100,
            "excess_spy_return": index / 200,
            "drawdown_event": float(index % 2),
        }
        for index, feature_date in enumerate(dataset.feature_dates)
    }

    writer.write_ranking_diagnostics(
        tmp_path / "ranking_diagnostics.json",
        dataset,
        outcomes_by_date,
    )

    payload = json.loads((tmp_path / "ranking_diagnostics.json").read_text())
    assert payload["evaluation"] == "purged_walk_forward"
    assert payload["model_type"] == "noop"
    assert payload["quantile_count"] == 2
    assert payload["fold_count"] == 2
    assert [fold["fold"] for fold in payload["folds"]] == [1, 2]
    assert payload["pooled_out_of_sample_diagnostics"]["sample_count"] == 4
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"


def _writer(ml_overrides: dict | None = None) -> MLDiagnosticReportWriter:
    config = {"ml": {"model_type": "noop", **(ml_overrides or {})}}
    return MLDiagnosticReportWriter(
        config,
        MLExperimentConfig.from_config(config),
    )


def _dataset(labels: list[int]) -> MLDataset:
    feature_dates = [
        f"2024-03-{index + 1:02d}"
        for index in range(len(labels))
    ]
    return MLDataset(
        features=[{"x": float(index)} for index in range(len(labels))],
        labels=labels,
        feature_dates=feature_dates,
        label_start_dates=feature_dates,
        label_end_dates=feature_dates,
        feature_ids=feature_dates,
        metadata=[{} for _ in labels],
        auxiliary_targets=[{} for _ in labels],
    )


class _RecordingModelPipeline:
    def __init__(self) -> None:
        self.built_model_numbers: list[int] = []
        self.fit_sample_counts: list[int] = []
        self.predict_sample_counts: list[int] = []
        self.predict_context_sample_counts: list[int] = []
        self.predict_context_start_dates: list[str] = []

    def build_model(self) -> dict[str, int]:
        model = {"number": len(self.built_model_numbers) + 1}
        self.built_model_numbers.append(model["number"])
        return model

    def fit(self, model: dict[str, int], dataset: MLDataset) -> None:
        self.fit_sample_counts.append(dataset.sample_count)

    def prediction_context(self, split) -> MLDataset:
        return MLModelPipeline.concat_datasets(split.train, split.test)

    def predict(
        self,
        model: dict[str, int],
        dataset: MLDataset,
        *,
        prediction_context: MLDataset | None = None,
    ) -> MLModelPrediction:
        self.predict_sample_counts.append(dataset.sample_count)
        context = prediction_context or dataset
        self.predict_context_sample_counts.append(context.sample_count)
        self.predict_context_start_dates.append(context.feature_dates[0])
        probabilities = [0.1 if label == 0 else 0.9 for label in dataset.labels]
        return MLModelPrediction(probabilities, [{} for _ in probabilities])

    def predictions_from_probabilities(self, probabilities: list[float]) -> list[int]:
        return [int(probability >= 0.5) for probability in probabilities]
