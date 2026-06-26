from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import pytest

from core.entities.candle import Candle
from core.research.ml.datasets import MLDataset
from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.validation import chronological_holdout


def test_prediction_artifacts_include_standard_columns(tmp_path):
    dataset = MLDataset(
        features=[{"x": float(index)} for index in range(12)],
        labels=[0, 1] * 6,
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(12)],
        label_start_dates=[f"2024-01-{index + 2:02d}" for index in range(12)],
        label_end_dates=[f"2024-01-{index + 2:02d}" for index in range(12)],
        feature_ids=[f"id-{index}" for index in range(12)],
        metadata=[{"rebalance_date": f"2024-01-{index + 1:02d}", "variant_id": "v"} for index in range(12)],
    )
    runner = MLExperimentRunner({
        "ml": {
            "model_type": "logistic_regression",
            "label_type": "should_reduce_exposure",
            "feature_set": "expanded_rebalance_v1",
            "walk_forward_folds": 2,
        }
    })
    split = chronological_holdout(dataset, test_fraction=0.25)

    runner._write_prediction_artifacts(
        tmp_path / "prediction_artifacts.csv",
        tmp_path / "prediction_artifacts.json",
        dataset,
        split,
        [0.2, 0.8, 0.7],
    )

    with (tmp_path / "prediction_artifacts.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads((tmp_path / "prediction_artifacts.json").read_text())

    assert rows
    assert "feature_id" in rows[0]
    assert "variant_id" in rows[0]
    assert "raw_probability" in rows[0]
    assert rows[0]["source_dataset_row_count"] == "12"
    assert rows[0]["train_sample_count"] == str(split.train.sample_count)
    assert rows[0]["test_sample_count"] == str(split.test.sample_count)
    assert rows[0]["generated_at"]
    assert rows[0]["dataset_hash"]
    assert metadata["validation_method"] == "rolling_walk_forward_out_of_fold_plus_holdout"
    assert metadata["trading_impact"] == "none"
    assert metadata["source_dataset_row_count"] == 12
    assert metadata["train_sample_count"] == split.train.sample_count
    assert metadata["test_sample_count"] == split.test.sample_count
    assert metadata["generated_at"]
    assert metadata["dataset_hash"] == rows[0]["dataset_hash"]
    assert metadata["data_hash"] == metadata["dataset_hash"]


def test_prediction_artifacts_overwrite_with_new_dataset_hash_when_dataset_changes(tmp_path):
    runner = MLExperimentRunner({
        "ml": {
            "model_type": "logistic_regression",
            "label_type": "should_reduce_exposure",
            "feature_set": "expanded_rebalance_v1",
            "walk_forward_folds": 2,
        }
    })
    csv_path = tmp_path / "prediction_artifacts.csv"
    metadata_path = tmp_path / "prediction_artifacts.json"

    first_dataset = _dataset(12)
    first_split = chronological_holdout(first_dataset, test_fraction=0.25)
    runner._write_prediction_artifacts(
        csv_path,
        metadata_path,
        first_dataset,
        first_split,
        [0.5] * first_split.test.sample_count,
    )
    first_metadata = json.loads(metadata_path.read_text())

    second_dataset = _dataset(14)
    second_split = chronological_holdout(second_dataset, test_fraction=0.25)
    runner._write_prediction_artifacts(
        csv_path,
        metadata_path,
        second_dataset,
        second_split,
        [0.5] * second_split.test.sample_count,
    )
    second_metadata = json.loads(metadata_path.read_text())

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert first_metadata["source_dataset_row_count"] == 12
    assert second_metadata["source_dataset_row_count"] == 14
    assert second_metadata["dataset_hash"] != first_metadata["dataset_hash"]
    assert {row["dataset_hash"] for row in rows} == {second_metadata["dataset_hash"]}
    assert {row["source_dataset_row_count"] for row in rows} == {"14"}


def test_ml_experiment_runner_writes_prediction_artifacts_with_runtime_provenance(tmp_path):
    report_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(cache_dir)},
            "reports": {"ml_dir": str(report_dir)},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "noop",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)


def test_itransformer_runner_writes_prediction_artifacts_with_provenance(tmp_path):
    pytest.importorskip("torch")
    report_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(cache_dir)},
            "reports": {"ml_dir": str(report_dir)},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "itransformer",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "overlay_comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
                "walk_forward_folds": 1,
                "sequence_length": 8,
                "itransformer_sequence_length": 8,
                "itransformer_d_model": 8,
                "itransformer_heads": 2,
                "itransformer_layers": 1,
                "itransformer_feedforward": 16,
                "itransformer_epochs": 1,
                "itransformer_batch_size": 8,
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)
    assert result.model_path.name == "model.pt"


def test_momentum_transformer_runner_writes_prediction_artifacts_with_provenance(tmp_path):
    pytest.importorskip("torch")
    report_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(cache_dir)},
            "reports": {"ml_dir": str(report_dir)},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "momentum_transformer",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "overlay_comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
                "walk_forward_folds": 1,
                "sequence_length": 8,
                "momentum_transformer_sequence_length": 8,
                "momentum_transformer_d_model": 8,
                "momentum_transformer_heads": 2,
                "momentum_transformer_layers": 1,
                "momentum_transformer_feedforward": 16,
                "momentum_transformer_epochs": 1,
                "momentum_transformer_batch_size": 8,
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)
    assert result.model_path.name == "model.pt"


def test_multitask_transformer_runner_writes_prediction_artifacts_with_provenance(tmp_path):
    pytest.importorskip("torch")
    report_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(cache_dir)},
            "reports": {"ml_dir": str(report_dir)},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "multitask_transformer",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "overlay_comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
                "walk_forward_folds": 1,
                "sequence_length": 8,
                "multitask_transformer_sequence_length": 8,
                "multitask_transformer_d_model": 8,
                "multitask_transformer_heads": 2,
                "multitask_transformer_layers": 1,
                "multitask_transformer_feedforward": 16,
                "multitask_transformer_epochs": 1,
                "multitask_transformer_batch_size": 8,
                "multitask_regression_targets": ["forward_return_5d"],
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)
    assert result.model_path.name == "model.pt"


def test_market_context_encoder_runner_writes_prediction_artifacts_with_provenance(tmp_path):
    pytest.importorskip("torch")
    report_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(cache_dir)},
            "reports": {"ml_dir": str(report_dir)},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "market_context_encoder",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "overlay_comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
                "walk_forward_folds": 1,
                "sequence_length": 8,
                "market_context_sequence_length": 8,
                "market_context_hidden_size": 8,
                "market_context_epochs": 1,
                "market_context_batch_size": 8,
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)
    assert result.model_path.name == "model.pt"


def test_news_analysis_transformer_runner_writes_prediction_artifacts_with_provenance(tmp_path):
    pytest.importorskip("torch")
    report_dir = tmp_path / "reports"
    cache_dir = tmp_path / "cache"
    candles_by_symbol = {
        symbol: _candles(symbol, 400, start_price)
        for symbol, start_price in (("SPY", 100.0), ("QQQ", 200.0), ("AAPL", 150.0))
    }
    runner = MLExperimentRunner(
        {
            "backtest": {
                "years": 2,
                "timeframe": "1Day",
                "starting_equity": 10_000.0,
            },
            "cache": {"enabled": False, "ml_dir": str(cache_dir)},
            "reports": {"ml_dir": str(report_dir)},
            "research": {"dual_momentum": {"symbols": ["AAPL", "SPY", "QQQ"]}},
            "ml": {
                "model_type": "news_analysis_transformer",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "overlay_comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
                "walk_forward_folds": 1,
                "sequence_length": 8,
                "news_transformer_sequence_length": 8,
                "news_transformer_d_model": 8,
                "news_transformer_heads": 2,
                "news_transformer_layers": 1,
                "news_transformer_feedforward": 16,
                "news_transformer_epochs": 1,
                "news_transformer_batch_size": 8,
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)
    assert result.model_path.name == "model.pt"


def _assert_prediction_artifact_provenance(result) -> None:
    with result.dataset_path.open("r", encoding="utf-8", newline="") as handle:
        expected_dataset_rows = len(list(csv.DictReader(handle)))
    with result.prediction_artifacts_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        artifact_rows = list(reader)
    metadata = json.loads(
        result.prediction_artifacts_metadata_path.read_text(encoding="utf-8")
    )

    assert "dataset_hash" in fieldnames
    assert "source_dataset_row_count" in fieldnames
    assert "train_sample_count" in fieldnames
    assert "test_sample_count" in fieldnames
    assert "generated_at" in fieldnames
    assert artifact_rows
    assert all(row["dataset_hash"] for row in artifact_rows)
    assert {row["source_dataset_row_count"] for row in artifact_rows} == {
        str(expected_dataset_rows)
    }
    assert all(row["train_sample_count"] for row in artifact_rows)
    assert all(row["test_sample_count"] for row in artifact_rows)
    assert all(row["generated_at"] for row in artifact_rows)
    assert metadata["dataset_hash"] == artifact_rows[0]["dataset_hash"]
    assert metadata["source_dataset_row_count"] == expected_dataset_rows
    assert metadata["train_sample_count"]
    assert metadata["test_sample_count"]
    assert metadata["generated_at"]


def _dataset(sample_count: int) -> MLDataset:
    return MLDataset(
        features=[{"x": float(index)} for index in range(sample_count)],
        labels=[index % 2 for index in range(sample_count)],
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(sample_count)],
        label_start_dates=[f"2024-01-{index + 1:02d}" for index in range(sample_count)],
        label_end_dates=[f"2024-01-{index + 1:02d}" for index in range(sample_count)],
        feature_ids=[f"id-{index}" for index in range(sample_count)],
        metadata=[
            {"rebalance_date": f"2024-01-{index + 1:02d}", "variant_id": "v"}
            for index in range(sample_count)
        ],
    )


def _candles(symbol: str, count: int, start_price: float) -> list[Candle]:
    start = datetime(2024, 1, 1)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=start_price + index,
            high=start_price + index,
            low=start_price + index,
            close=start_price + index,
            volume=1_000.0,
        )
        for index in range(count)
    ]


class _StaticFeed:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]]):
        self.candles_by_symbol = candles_by_symbol

    def get_historical_bars(self, symbol, timeframe, start, end):
        return self.candles_by_symbol[symbol]
