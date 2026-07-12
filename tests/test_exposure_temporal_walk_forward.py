from __future__ import annotations

import json

from core.research.ml.data.datasets import MLDataset
from core.research.ml.experiment_runner import MLExperimentRunner


def test_exposure_walk_forward_uses_only_matured_labels_and_writes_audit(tmp_path):
    dates = [f"2024-01-{day:02d}" for day in range(1, 9)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    metadata = [
        {
            "feature_timestamp": date,
            "decision_timestamp": date,
            "label_available_timestamp": f"2024-01-{day + 1:02d}",
        }
        for day, date in enumerate(dates, start=1)
    ]
    metadata[1]["label_available_timestamp"] = "2024-01-07"
    dataset = MLDataset(
        features=[
            {"momentum": float(index), "volatility": float(index % 3)}
            for index in range(len(dates))
        ],
        labels=labels,
        feature_dates=dates,
        label_start_dates=dates,
        label_end_dates=[f"2024-01-{day + 1:02d}" for day in range(1, 9)],
        feature_ids=[f"row-{index}" for index in range(len(dates))],
        metadata=metadata,
    )
    runner = MLExperimentRunner({
        "ml": {
            "label_type": "should_reduce_exposure",
            "model_type": "logistic_regression",
            "output_dir": str(tmp_path),
            "exposure_minimum_training_rows": 2,
            "exposure_minimum_positive_labels": 1,
            "exposure_minimum_negative_labels": 1,
        }
    })

    _, split, probabilities, _, audit = runner._fit_predict_exposure_walk_forward(dataset)
    runner._write_exposure_temporal_audit(tmp_path, audit)

    assert split.test.sample_count == len(probabilities)
    assert audit["leakage_checks_passed"] is True
    assert audit["checkpoint_identity_policy"]["resume_across_different_decision_fold_allowed"] is False
    assert any(fold["purged_row_count"] > 0 for fold in audit["folds"])
    assert all(
        fold["maximum_training_label_available_timestamp"] <= fold["decision_timestamp"]
        for fold in audit["folds"]
    )
    written = json.loads(
        (tmp_path / "exposure_temporal_audit.json").read_text(encoding="utf-8")
    )
    assert written["temporal_policy"]["training_eligibility_rule"] == (
        "label_available_timestamp <= decision_timestamp"
    )
    assert (tmp_path / "exposure_temporal_folds.csv").exists()
