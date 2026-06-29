from __future__ import annotations

from datetime import datetime, timedelta
import json
from types import SimpleNamespace

from core.research.ml.config import MLExperimentConfig
from core.research.ml.data.datasets import MLDataset
from core.research.ml.reports import MLOverlayReportWriter
from core.research.ml.validation import ChronologicalSplit


def test_overlay_report_writer_preserves_shadow_overlay_payload(tmp_path):
    writer = _writer({
        "shadow_model_type": "noop",
        "shadow_thresholds": [0.6],
        "shadow_reduced_exposures": [0.7],
        "shadow_transaction_cost_bps": 0.0,
        "walk_forward_folds": 2,
    })
    dataset = _dataset(8)

    writer.write_shadow_overlay(tmp_path / "shadow_overlay.json", dataset)

    payload = json.loads((tmp_path / "shadow_overlay.json").read_text())
    assert payload["mode"] == "shadow_research_only"
    assert payload["model_type"] == "noop"
    assert payload["rebalance_only"] is True
    assert payload["overlay_probability"] == "champion_success_probability"
    assert payload["overlay_decision_rule"] == (
        "reduce_exposure_when_champion_success_probability_lt_threshold"
    )
    assert payload["transaction_cost_bps"] == 0.0
    assert payload["trading_impact"] == "none"
    assert payload["scenarios"][0]["decision_threshold"] == 0.6
    assert payload["scenarios"][0]["reduced_exposure"] == 0.7
    assert [fold["fold"] for fold in payload["scenarios"][0]["folds"]] == [1, 2]


def test_overlay_report_writer_preserves_holdout_shadow_payload(tmp_path):
    writer = _writer({
        "shadow_model_type": "noop",
        "shadow_holdout_threshold": 0.6,
        "shadow_holdout_reduced_exposure": 0.7,
        "shadow_transaction_cost_bps": 0.0,
    })
    dataset = _dataset(8)
    split = ChronologicalSplit(
        train=_slice_dataset(dataset, [0, 1, 2, 3]),
        test=_slice_dataset(dataset, [4, 5, 6, 7]),
        test_start_date="2024-04-05",
        purged_train_samples=0,
    )

    writer.write_holdout_shadow_overlay(
        tmp_path / "holdout_shadow_overlay.json",
        split,
    )

    payload = json.loads((tmp_path / "holdout_shadow_overlay.json").read_text())
    assert payload["mode"] == "final_holdout_shadow_research_only"
    assert payload["model_type"] == "noop"
    assert payload["decision_threshold"] == 0.6
    assert payload["reduced_exposure"] == 0.7
    assert payload["rebalance_only"] is True
    assert payload["overlay_probability"] == "champion_success_probability"
    assert payload["transaction_cost_bps"] == 0.0
    assert payload["test_start_date"] == "2024-04-05"
    assert payload["result"]["evaluated_days"] == 3
    assert payload["trading_impact"] == "none"
    assert payload["candidate_frozen_before_holdout"] is True


def test_overlay_report_writer_preserves_model_comparison_payload(tmp_path):
    writer = _writer({
        "overlay_comparison_models": ["noop"],
        "overlay_comparison_thresholds": [0.6],
        "overlay_comparison_reduced_exposures": [0.7],
        "shadow_transaction_cost_bps": 0.0,
        "walk_forward_folds": 2,
    })
    dataset = _dataset(8)

    writer.write_overlay_model_comparison(
        tmp_path / "overlay_model_comparison.json",
        dataset,
    )

    payload = json.loads((tmp_path / "overlay_model_comparison.json").read_text())
    assert payload["mode"] == "overlay_model_comparison_research_only"
    assert payload["label_type"] == "champion_success"
    assert payload["overlay_probability"] == "champion_success_probability"
    assert payload["fold_count"] == 2
    assert payload["rebalance_only"] is True
    assert payload["transaction_cost_bps"] == 0.0
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    scenario = payload["models"][0]["scenarios"][0]
    assert payload["models"][0]["model_type"] == "noop"
    assert scenario["decision_threshold"] == 0.6
    assert scenario["reduced_exposure"] == 0.7
    assert scenario["summary"]["valid_fold_count"] == 2
    assert scenario["summary"]["skipped_fold_count"] == 0
    assert "return_delta" in scenario["folds"][0]
    assert "max_drawdown_delta" in scenario["folds"][0]


def test_overlay_report_writer_records_model_comparison_exceptions(tmp_path):
    writer = _writer({
        "overlay_comparison_models": ["unsupported_model"],
        "overlay_comparison_thresholds": [0.6],
        "overlay_comparison_reduced_exposures": [0.7],
        "walk_forward_folds": 2,
    })
    dataset = _dataset(8)

    writer.write_overlay_model_comparison(
        tmp_path / "overlay_model_comparison.json",
        dataset,
    )

    payload = json.loads((tmp_path / "overlay_model_comparison.json").read_text())
    folds = payload["models"][0]["scenarios"][0]["folds"]
    assert [fold["fold"] for fold in folds] == [1, 2]
    assert all(fold["skipped"] is True for fold in folds)
    assert all("Unsupported ml.model_type" in fold["reason"] for fold in folds)
    assert payload["models"][0]["scenarios"][0]["summary"] == {
        "valid_fold_count": 0,
        "skipped_fold_count": 2,
        "mean_base_total_return": None,
        "mean_overlay_total_return": None,
        "mean_return_delta": None,
        "mean_base_max_drawdown": None,
        "mean_overlay_max_drawdown": None,
        "mean_max_drawdown_delta": None,
        "mean_reduced_exposure_days": None,
        "mean_overlay_turnover": None,
    }


def _writer(ml_overrides: dict) -> MLOverlayReportWriter:
    config = {"ml": {"model_type": "noop", **ml_overrides}}
    return MLOverlayReportWriter(
        config,
        MLExperimentConfig.from_config(config),
        _equity_curve(8),
        {
            f"2024-04-{index + 1:02d}"
            for index in range(8)
        },
    )


def _equity_curve(count: int) -> list[SimpleNamespace]:
    start = datetime(2024, 4, 1)
    return [
        SimpleNamespace(
            timestamp=start + timedelta(days=index),
            equity=100.0 + index,
        )
        for index in range(count)
    ]


def _dataset(count: int) -> MLDataset:
    feature_dates = [
        f"2024-04-{index + 1:02d}"
        for index in range(count)
    ]
    return MLDataset(
        features=[{"x": float(index)} for index in range(count)],
        labels=[index % 2 for index in range(count)],
        feature_dates=feature_dates,
        label_start_dates=feature_dates,
        label_end_dates=feature_dates,
        feature_ids=feature_dates,
        metadata=[{} for _ in range(count)],
        auxiliary_targets=[{} for _ in range(count)],
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
