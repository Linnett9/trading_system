from __future__ import annotations

import json

from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.reports import MLCalibrationReportWriter
from core.research.ml.validation import ChronologicalSplit


def test_calibration_report_writer_preserves_holdout_payload(tmp_path):
    writer = _writer({"calibration_bin_count": 2})

    writer.write_probability_calibration(
        tmp_path / "probability_calibration.json",
        [0, 1],
        [0.25, 0.75],
    )

    payload = json.loads((tmp_path / "probability_calibration.json").read_text())
    assert payload["evaluation"] == "chronological_holdout"
    assert payload["model_type"] == "noop"
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["calibration"]["sample_count"] == 2
    assert payload["calibration"]["brier_score"] == 0.0625
    assert len(payload["calibration"]["bins"]) == 2


def test_calibration_report_writer_preserves_calibrated_comparison_payload(tmp_path):
    writer = _writer({"calibration_bin_count": 2})
    split = ChronologicalSplit(
        train=_dataset([0, 1, 0, 1], "2024-01"),
        test=_dataset([0, 1], "2024-02"),
        test_start_date="2024-02-01",
        purged_train_samples=0,
    )

    writer.write_calibrated_probability_calibration(
        tmp_path / "calibrated_probability_calibration.json",
        split,
        [0.25, 0.75],
    )

    payload = json.loads(
        (tmp_path / "calibrated_probability_calibration.json").read_text()
    )
    assert payload["evaluation"] == (
        "chronological_holdout_calibration_method_comparison"
    )
    assert payload["model_type"] == "noop"
    assert payload["label_type"] == "champion_success"
    assert payload["calibration_methods"] == [
        "raw",
        "platt",
        "isotonic",
        "temperature_scaling",
    ]
    assert payload["best_method_by_brier"] in payload["method_comparison"]["methods"]
    assert payload["raw_brier_score"] == payload["raw_calibration"]["brier_score"]
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"


def test_calibration_report_writer_preserves_walk_forward_payload(tmp_path):
    writer = _writer({"calibration_bin_count": 2, "walk_forward_folds": 2})
    dataset = _dataset([0, 1, 0, 1, 0, 1, 0, 1], "2024-03")

    writer.write_walk_forward_probability_calibration(
        tmp_path / "walk_forward_probability_calibration.json",
        dataset,
    )

    payload = json.loads(
        (tmp_path / "walk_forward_probability_calibration.json").read_text()
    )
    assert payload["evaluation"] == "purged_walk_forward"
    assert payload["model_type"] == "noop"
    assert payload["fold_count"] == 2
    assert [fold["fold"] for fold in payload["folds"]] == [1, 2]
    assert payload["pooled_out_of_sample_calibration"]["sample_count"] == 4
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"


def _writer(ml_overrides: dict) -> MLCalibrationReportWriter:
    config = {"ml": {"model_type": "noop", **ml_overrides}}
    return MLCalibrationReportWriter(
        config,
        MLExperimentConfig.from_config(config),
    )


def _dataset(labels: list[int], prefix: str) -> MLDataset:
    feature_dates = [
        f"{prefix}-{index + 1:02d}"
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
