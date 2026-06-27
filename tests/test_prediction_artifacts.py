from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import pytest

from core.research.ml.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.artifact_validator import (
    validate_prediction_artifact_dirs,
    validate_prediction_artifacts,
)
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
    assert rows[0]["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert "profile" in rows[0]
    assert rows[0]["model_name"] == "logistic_regression"
    assert "config_path" in rows[0]
    assert rows[0]["prediction_date"] == rows[0]["date"]
    assert "symbol" in rows[0]
    assert rows[0]["predicted_probability"] == rows[0]["raw_probability"]
    assert "feature_id" in rows[0]
    assert "variant_id" in rows[0]
    assert "raw_probability" in rows[0]
    assert rows[0]["source_dataset_row_count"] == "12"
    assert rows[0]["train_sample_count"] == str(split.train.sample_count)
    assert rows[0]["test_sample_count"] == str(split.test.sample_count)
    assert rows[0]["generated_at"]
    assert rows[0]["dataset_hash"]
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["model_name"] == "logistic_regression"
    assert metadata["validation_method"] == "rolling_walk_forward_out_of_fold_plus_holdout"
    assert metadata["trading_impact"] == "none"
    assert metadata["source_dataset_row_count"] == 12
    assert metadata["train_sample_count"] == split.train.sample_count
    assert metadata["test_sample_count"] == split.test.sample_count
    assert metadata["generated_at"]
    assert metadata["dataset_hash"] == rows[0]["dataset_hash"]
    assert metadata["data_hash"] == metadata["dataset_hash"]
    validation = validate_prediction_artifacts(
        tmp_path / "prediction_artifacts.csv",
        tmp_path / "prediction_artifacts.json",
    )
    assert validation.dataset_hash == metadata["dataset_hash"]
    assert validation.legacy_warnings == ()


def test_prediction_artifact_validator_flags_legacy_csv(tmp_path):
    csv_path = tmp_path / "prediction_artifacts.csv"
    metadata_path = tmp_path / "prediction_artifacts.json"
    csv_path.write_text(
        "feature_id,model_type,split,dataset_hash\n"
        "a,dlinear,holdout,dataset-a\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({"dataset_hash": "dataset-a"}),
        encoding="utf-8",
    )

    result = validate_prediction_artifacts(csv_path, metadata_path)

    assert result.legacy_warnings
    assert any("legacy_prediction_artifact" in item for item in result.legacy_warnings)


def test_prediction_artifact_validator_rejects_missing_dataset_hash(tmp_path):
    csv_path = tmp_path / "prediction_artifacts.csv"
    metadata_path = tmp_path / "prediction_artifacts.json"
    csv_path.write_text(
        "feature_id,model_type,split,dataset_hash\n"
        "a,dlinear,holdout,\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({"dataset_hash": "dataset-a"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing dataset_hash"):
        validate_prediction_artifacts(csv_path, metadata_path)


def test_prediction_artifact_validator_rejects_mixed_hashes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_minimal_modern_artifact(first, "dlinear", "hash-a")
    _write_minimal_modern_artifact(second, "patchtst", "hash-b")

    with pytest.raises(RuntimeError, match="different dataset hashes"):
        validate_prediction_artifact_dirs([first, second])


def test_prediction_artifact_validator_reports_missing_dirs_cleanly(tmp_path):
    missing = tmp_path / "missing_model"

    with pytest.raises(
        RuntimeError,
        match="Prediction artifact directories do not exist",
    ):
        validate_prediction_artifact_dirs([missing])


def test_same_source_dataset_hash_across_different_model_inputs(tmp_path):
    first_dataset = _dataset(12)
    reversed_indexes = list(reversed(range(12)))
    second_dataset = MLDataset(
        features=[
            {"x": float(index), "model_specific_feature": float(index * 10)}
            for index in reversed_indexes
        ],
        labels=[first_dataset.labels[index] for index in reversed_indexes],
        feature_dates=[first_dataset.feature_dates[index] for index in reversed_indexes],
        label_start_dates=[first_dataset.label_start_dates[index] for index in reversed_indexes],
        label_end_dates=[first_dataset.label_end_dates[index] for index in reversed_indexes],
        feature_ids=[f"model-specific-{index}" for index in reversed_indexes],
        metadata=[first_dataset.metadata[index] for index in reversed_indexes],
        auxiliary_targets=[
            {"forward_return_5d": float(index) / 100.0}
            for index in reversed_indexes
        ],
    )
    first_runner = MLExperimentRunner({
        "ml": {
            "model_type": "noop",
            "label_type": "should_reduce_exposure",
            "feature_set": "expanded_rebalance_v1",
            "walk_forward_folds": 1,
        }
    })
    second_runner = MLExperimentRunner({
        "ml": {
            "model_type": "logistic_regression",
            "label_type": "should_reduce_exposure",
            "feature_set": "model_specific_feature_set",
            "walk_forward_folds": 1,
        }
    })

    first_split = chronological_holdout(first_dataset, test_fraction=0.25)
    second_split = chronological_holdout(second_dataset, test_fraction=0.25)
    first_runner._write_prediction_artifacts(
        tmp_path / "first" / "prediction_artifacts.csv",
        tmp_path / "first" / "prediction_artifacts.json",
        first_dataset,
        first_split,
        [0.5] * first_split.test.sample_count,
    )
    second_runner._write_prediction_artifacts(
        tmp_path / "second" / "prediction_artifacts.csv",
        tmp_path / "second" / "prediction_artifacts.json",
        second_dataset,
        second_split,
        [0.5] * second_split.test.sample_count,
    )

    first_metadata = json.loads(
        (tmp_path / "first" / "prediction_artifacts.json").read_text()
    )
    second_metadata = json.loads(
        (tmp_path / "second" / "prediction_artifacts.json").read_text()
    )

    assert first_metadata["dataset_hash"] == second_metadata["dataset_hash"]


def test_sequence_prediction_uses_train_history_for_holdout_context():
    runner = MLExperimentRunner({
        "ml": {
            "model_type": "noop",
            "label_type": "should_reduce_exposure",
            "feature_set": "expanded_rebalance_v1",
        }
    })
    dataset = MLDataset(
        features=[{"x": float(index)} for index in range(5)],
        labels=[0, 1, 0, 1, 0],
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(5)],
        label_start_dates=[f"2024-01-{index + 2:02d}" for index in range(5)],
        label_end_dates=[f"2024-01-{index + 2:02d}" for index in range(5)],
        feature_ids=[f"id-{index}" for index in range(5)],
        metadata=[{"variant_id": "same_variant"} for _ in range(5)],
    )
    split = chronological_holdout(dataset, test_fraction=0.4)
    model = _TwoRowContextModel()

    probabilities, _ = runner._predict_research_model(
        model,
        split.test,
        prediction_context=runner._prediction_context(split),
    )

    assert probabilities == [0.8, 0.8]


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
    fieldnames, rows = _prediction_artifact_rows(result)
    assert "predicted_trend_score" in fieldnames
    assert "predicted_regime_score" in fieldnames
    assert "predicted_size_multiplier" in fieldnames
    assert any(row["predicted_size_multiplier"] != "" for row in rows)


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
    with result.prediction_artifacts_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    metadata = json.loads(
        result.prediction_artifacts_metadata_path.read_text(encoding="utf-8")
    )

    assert "predicted_forward_return_5d" in fieldnames
    assert "actual_forward_return_5d" in fieldnames
    assert fieldnames.count("actual_label") == 1
    assert any(row["predicted_forward_return_5d"] != "" for row in rows)
    assert metadata["auxiliary_targets"] == ["forward_return_5d"]
    assert metadata["auxiliary_prediction_columns"] == ["predicted_forward_return_5d"]
    assert metadata["auxiliary_actual_columns"] == ["actual_forward_return_5d"]


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
    fieldnames, rows = _prediction_artifact_rows(result)
    assert "predicted_context_risk_multiplier" in fieldnames
    assert any(row["predicted_context_risk_multiplier"] != "" for row in rows)


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


def test_temporal_fusion_transformer_runner_writes_prediction_artifacts_with_provenance(tmp_path):
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
                "model_type": "temporal_fusion_transformer",
                "output_dir": str(report_dir),
                "comparison_models": ["noop"],
                "overlay_comparison_models": ["noop"],
                "shadow_model_type": "noop",
                "minimum_history_years": 1,
                "walk_forward_folds": 1,
                "sequence_length": 8,
                "tft_encoder_length": 8,
                "tft_hidden_size": 8,
                "tft_attention_heads": 2,
                "tft_epochs": 1,
                "tft_batch_size": 8,
                "tft_known_future_features": ["day_of_week", "month"],
            },
        },
        feed=_StaticFeed(candles_by_symbol),
    )

    result = runner.run()

    _assert_prediction_artifact_provenance(result)
    assert result.model_path.name == "model.pt"
    fieldnames, rows = _prediction_artifact_rows(result)
    assert "predicted_forward_return_5d" in fieldnames
    assert "predicted_forward_return_10d" in fieldnames
    assert "predicted_future_volatility" in fieldnames
    assert "predicted_future_drawdown" in fieldnames
    assert any(row["predicted_future_volatility"] != "" for row in rows)


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
    assert "artifact_schema_version" in fieldnames
    assert "profile" in fieldnames
    assert "model_name" in fieldnames
    assert "config_path" in fieldnames
    assert "prediction_date" in fieldnames
    assert "symbol" in fieldnames
    assert "predicted_probability" in fieldnames
    assert "source_dataset_row_count" in fieldnames
    assert "train_sample_count" in fieldnames
    assert "test_sample_count" in fieldnames
    assert "generated_at" in fieldnames
    assert artifact_rows
    assert {row["artifact_schema_version"] for row in artifact_rows} == {
        ARTIFACT_SCHEMA_VERSION
    }
    assert all(row["prediction_date"] for row in artifact_rows)
    assert all(row["predicted_probability"] for row in artifact_rows)
    assert all(row["dataset_hash"] for row in artifact_rows)
    assert {row["source_dataset_row_count"] for row in artifact_rows} == {
        str(expected_dataset_rows)
    }
    assert all(row["train_sample_count"] for row in artifact_rows)
    assert all(row["test_sample_count"] for row in artifact_rows)
    assert all(row["generated_at"] for row in artifact_rows)
    assert metadata["dataset_hash"] == artifact_rows[0]["dataset_hash"]
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert metadata["source_dataset_row_count"] == expected_dataset_rows
    assert metadata["train_sample_count"]
    assert metadata["test_sample_count"]
    assert metadata["generated_at"]


def _prediction_artifact_rows(result) -> tuple[list[str], list[dict[str, str]]]:
    with result.prediction_artifacts_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_minimal_modern_artifact(
    path,
    model_type: str,
    dataset_hash: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    row = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "profile": "test",
        "model_name": model_type,
        "model_type": model_type,
        "config_path": "configs/research/test.yaml",
        "dataset_hash": dataset_hash,
        "source_dataset_row_count": "1",
        "train_sample_count": "1",
        "prediction_date": "2024-01-01",
        "symbol": "",
        "rebalance_date": "2024-01-01",
        "actual_label": "0",
        "predicted_probability": "0.5",
        "feature_id": "feature-a",
        "split": "holdout",
    }
    with (path / "prediction_artifacts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (path / "prediction_artifacts.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "model_type": model_type,
                "dataset_hash": dataset_hash,
            }
        ),
        encoding="utf-8",
    )


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


class _TwoRowContextModel:
    def __init__(self) -> None:
        self._group_ids: list[str] = []

    def set_sequence_context(
        self,
        metadata: list[dict[str, str]] | None = None,
        feature_dates: list[str] | None = None,
    ) -> None:
        del feature_dates
        self._group_ids = [
            str(row.get("variant_id", "global"))
            for row in metadata or []
        ]

    def predict_proba(self, rows: list[dict[str, float]]) -> list[float]:
        probabilities = []
        for index, _ in enumerate(rows):
            has_history = (
                index > 0
                and index < len(self._group_ids)
                and self._group_ids[index] == self._group_ids[index - 1]
            )
            probabilities.append(0.8 if has_history else 0.2)
        return probabilities


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
