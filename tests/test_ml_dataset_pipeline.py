from __future__ import annotations

import csv

from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import (
    MLDataset,
    build_dataset,
    dataset_leakage_audit,
    write_dataset,
)
from core.research.ml.pipelines.dataset_pipeline import MLDatasetPipeline
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


def test_write_dataset_preserves_dynamic_feature_columns(tmp_path):
    dataset = MLDataset(
        features=[
            {"spy_return_1m": 0.1},
            {"1h_spy_return_last_bar": 0.2, "5m_spy_volume_last_bar": 100.0},
        ],
        labels=[1, 0],
        feature_dates=["2024-01-01", "2024-01-02"],
        label_start_dates=["2024-01-02", "2024-01-03"],
        label_end_dates=["2024-01-10", "2024-01-11"],
    )
    path = tmp_path / "dataset.csv"

    write_dataset(path, dataset, label_name="should_reduce_exposure")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "feature_date",
        "label_start_date",
        "label_end_date",
        "should_reduce_exposure",
        "1h_spy_return_last_bar",
        "5m_spy_volume_last_bar",
        "spy_return_1m",
    ]
    assert rows[0]["1h_spy_return_last_bar"] == ""
    assert rows[1]["1h_spy_return_last_bar"] == "0.2"


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


def test_target_derived_outcomes_are_excluded_from_predictors():
    feature = {
        "feature_id": "one",
        "feature_date": "2024-01-01",
        "safe_momentum": 0.2,
        "should_reduce_exposure": 1,
        "champion_excess_return": -0.2,
        "volatility_adjusted_excess_return": -2.0,
        "future_max_drawdown": -0.3,
        "future_custom_outcome": 99.0,
        "forward_return_5d": -0.1,
    }
    label = {
        "feature_id": "one",
        "feature_date": "2024-01-01",
        "label_start_date": "2024-01-02",
        "label_end_date": "2024-02-01",
        "should_reduce_exposure": 1,
    }

    dataset = build_dataset([feature], [label], "should_reduce_exposure")

    assert dataset.features == [{"safe_momentum": 0.2}]
    assert dataset_leakage_audit(dataset)["leakage_check_passed"] is True


def test_leakage_audit_fails_for_deliberately_inserted_target_component():
    dataset = MLDataset(
        features=[{"volatility_adjusted_excess_return": -2.0}],
        labels=[1],
        feature_dates=["2024-01-01"],
        label_start_dates=["2024-01-02"],
        label_end_dates=["2024-02-01"],
    )

    audit = dataset_leakage_audit(dataset)

    assert audit["leakage_check_passed"] is False
    assert audit["forbidden_predictor_columns"] == [
        "volatility_adjusted_excess_return"
    ]


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
