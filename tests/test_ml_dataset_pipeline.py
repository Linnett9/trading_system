from __future__ import annotations

from core.research.ml.config import MLExperimentConfig
from core.research.ml.dataset_pipeline import MLDatasetPipeline
from core.research.ml.features import MLFeatureBuildResult
from core.research.ml.labels import MLLabelBuildResult


def test_dataset_pipeline_builds_dataset_and_split():
    config = {
        "ml": {
            "label_type": "risk_regime",
            "test_fraction": 0.25,
        }
    }
    pipeline = MLDatasetPipeline(MLExperimentConfig.from_config(config))
    feature_result = MLFeatureBuildResult(
        rows=[
            {"feature_date": "2024-01-01", "momentum": 0.1},
            {"feature_date": "2024-01-02", "momentum": 0.2},
            {"feature_date": "2024-01-03", "momentum": 0.3},
            {"feature_date": "2024-01-04", "momentum": 0.4},
        ],
        dropped_rows=0,
        date_range=("2024-01-01", "2024-01-04"),
    )
    label_result = MLLabelBuildResult(
        rows=[
            _label("2024-01-01", "2024-01-02", "2024-01-02", 1),
            _label("2024-01-02", "2024-01-03", "2024-01-03", 0),
            _label("2024-01-03", "2024-01-04", "2024-01-05", 1),
            _label("2024-01-04", "2024-01-05", "2024-01-06", 0),
        ],
        dropped_rows_insufficient_horizon=0,
        label_name="risk_regime",
    )

    prepared = pipeline.prepare(feature_result, label_result)

    assert prepared.dataset.sample_count == 4
    assert prepared.dataset.features[0] == {"momentum": 0.1}
    assert prepared.dataset.labels == [1, 0, 1, 0]
    assert prepared.split.train.sample_count == 2
    assert prepared.split.test.sample_count == 1
    assert prepared.split.test_start_date == "2024-01-04"


def test_dataset_pipeline_respects_explicit_split_dates():
    config = {
        "ml": {
            "label_type": "risk_regime",
            "train_end": "2024-01-02",
            "test_start": "2024-01-03",
        }
    }
    pipeline = MLDatasetPipeline(MLExperimentConfig.from_config(config))
    feature_result = MLFeatureBuildResult(
        rows=[
            {"feature_date": "2024-01-01", "momentum": 0.1},
            {"feature_date": "2024-01-02", "momentum": 0.2},
            {"feature_date": "2024-01-03", "momentum": 0.3},
        ],
        dropped_rows=0,
        date_range=("2024-01-01", "2024-01-03"),
    )
    label_result = MLLabelBuildResult(
        rows=[
            _label("2024-01-01", "2024-01-02", "2024-01-02", 1),
            _label("2024-01-02", "2024-01-03", "2024-01-04", 0),
            _label("2024-01-03", "2024-01-04", "2024-01-05", 1),
        ],
        dropped_rows_insufficient_horizon=0,
        label_name="risk_regime",
    )

    split = pipeline.prepare(feature_result, label_result).split

    assert split.train.feature_dates == ["2024-01-01"]
    assert split.test.feature_dates == ["2024-01-03"]
    assert split.purged_train_samples == 1


def _label(
    feature_date: str,
    label_start_date: str,
    label_end_date: str,
    value: int,
) -> dict[str, int | str]:
    return {
        "feature_date": feature_date,
        "label_start_date": label_start_date,
        "label_end_date": label_end_date,
        "risk_regime": value,
    }
