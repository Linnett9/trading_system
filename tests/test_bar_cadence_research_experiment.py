from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.stock_level.bar_cadence_research_experiment import (
    _resolve_models,
    build_bar_cadence_research_experiment,
    build_every_bar_oos_predictions,
    build_replay_grid_from_predictions,
)
from core.research.ml.stock_level.stock_alpha_model_sets import FULL_SEQUENCE_MODELS
from core.research.ml.stock_level.bar_targets import (
    add_forward_return_targets,
    label_is_mature,
    target_column_name,
)
from core.research.ml.stock_level.feature_bank_adapter import (
    FeatureBankSlice,
    load_feature_bank_slice,
)


def test_feature_bank_adapter_filters_orders_and_reports_metadata(tmp_path: Path):
    path = tmp_path / "stock_features_1Day.parquet"
    rows = list(reversed(_feature_rows(symbols=("AAA", "BBB"), count=8)))
    pq.write_table(pa.Table.from_pylist(rows), path)

    loaded = load_feature_bank_slice(
        "1Day",
        path=path,
        symbols=["BBB"],
        start="2024-01-03T00:00:00+00:00",
        end="2024-01-06T00:00:00+00:00",
        columns=["feature_a", "feature_b"],
    )

    assert [row["symbol"] for row in loaded.rows] == ["BBB"] * 4
    assert [row["timestamp"].day for row in loaded.rows] == [3, 4, 5, 6]
    assert loaded.metadata_columns == ("timestamp", "symbol", "timeframe")
    assert set(loaded.feature_columns) == {"open", "high", "low", "close", "volume", "feature_a", "feature_b"}
    assert loaded.label_columns == ()
    assert loaded.metadata["source_row_count"] == 16
    assert loaded.metadata["row_count"] == 4
    assert loaded.metadata["duplicate_key_count"] == 0


def test_target_generation_uses_bar_horizon_and_label_maturity():
    rows = _feature_rows(symbols=("AAA",), count=5)
    targeted, metadata = add_forward_return_targets(rows, horizon_bars=2)
    target = target_column_name(2)

    assert targeted[0][target] == pytest.approx((102.5 / 100.5) - 1.0)
    assert targeted[0]["label_maturity_timestamp_2b"] == targeted[2]["timestamp"]
    assert targeted[-1][target] is None
    assert metadata["target_row_count"] == 3
    assert label_is_mature(
        targeted[0],
        horizon_bars=2,
        fit_cutoff=targeted[2]["timestamp"],
    )
    assert not label_is_mature(
        targeted[1],
        horizon_bars=2,
        fit_cutoff=targeted[2]["timestamp"],
    )


def test_intraday_target_generation_can_drop_cross_session_horizons():
    rows = [
        _target_row("2024-01-02T15:55:00+00:00", 100.0),
        _target_row("2024-01-02T16:00:00+00:00", 101.0),
        _target_row("2024-01-03T09:30:00+00:00", 103.0),
    ]

    targeted, metadata = add_forward_return_targets(
        rows,
        horizon_bars=1,
        allow_cross_session_horizon=False,
        expected_bar_seconds=300,
        allow_missing_intermediate_bars=False,
    )
    target = target_column_name(1)

    assert targeted[0][target] == pytest.approx(0.01)
    assert targeted[1][target] is None
    assert metadata["cross_session_target_count"] == 1
    assert metadata["dropped_cross_session_target_count"] == 1
    assert metadata["missing_intermediate_gap_count"] == 1
    assert metadata["target_row_count"] == 1


def test_intraday_target_generation_can_allow_next_session_horizons():
    rows = [
        _target_row("2024-01-02T16:00:00+00:00", 101.0),
        _target_row("2024-01-03T09:30:00+00:00", 103.0),
    ]

    targeted, metadata = add_forward_return_targets(
        rows,
        horizon_bars=1,
        allow_cross_session_horizon=True,
        expected_bar_seconds=300,
        allow_missing_intermediate_bars=True,
    )

    assert targeted[0][target_column_name(1)] == pytest.approx((103.0 / 101.0) - 1.0)
    assert metadata["cross_session_target_count"] == 1
    assert metadata["dropped_cross_session_target_count"] == 0


def test_intraday_target_generation_can_drop_missing_intermediate_bars():
    rows = [
        _target_row("2024-01-02T09:30:00+00:00", 100.0),
        _target_row("2024-01-02T09:40:00+00:00", 102.0),
        _target_row("2024-01-02T09:45:00+00:00", 103.0),
    ]

    targeted, metadata = add_forward_return_targets(
        rows,
        horizon_bars=1,
        expected_bar_seconds=300,
        allow_missing_intermediate_bars=False,
    )

    assert targeted[0][target_column_name(1)] is None
    assert targeted[1][target_column_name(1)] == pytest.approx((103.0 / 102.0) - 1.0)
    assert metadata["missing_intermediate_gap_count"] == 1
    assert metadata["dropped_missing_intermediate_target_count"] == 1


def test_scheduled_refit_scores_every_bar_and_keeps_model_frozen_between_refits():
    feature_slice = FeatureBankSlice(
        rows=_feature_rows(symbols=("AAA", "BBB", "CCC"), count=16),
        metadata={"timeframe": "1Day"},
        metadata_columns=("timestamp", "symbol", "timeframe"),
        feature_columns=("feature_a", "feature_b", "open", "close"),
        label_columns=(),
    )

    predictions, payload = build_every_bar_oos_predictions(
        feature_slice,
        horizon_bars=1,
        model_name="ridge",
        refit_frequency_bars=3,
        min_train_rows=9,
    )

    assert predictions
    assert payload["fit_count"] >= 2
    assert payload["model_frozen_between_refits"] is True
    keys = {
        (row["timeframe"], row["model"], row["timestamp"], row["symbol"])
        for row in predictions
    }
    assert len(keys) == len(predictions)
    by_timestamp = {}
    for row in predictions:
        by_timestamp.setdefault(row["timestamp"], []).append(row)
        assert row["intended_execution_timestamp"] is None or row["intended_execution_timestamp"] > row["timestamp"]
        assert row["fit_cutoff_timestamp"] <= row["timestamp"]
    assert all(len(rows) == 3 for rows in by_timestamp.values())
    refits = sorted({row["refit_id"] for row in predictions})
    assert refits[0] == 1
    assert len(refits) >= 2


def test_end_to_end_research_experiment_connects_predictions_to_replay(tmp_path: Path):
    path = tmp_path / "stock_features_1Day.parquet"
    pq.write_table(pa.Table.from_pylist(_feature_rows(symbols=("AAA", "BBB", "CCC"), count=18)), path)
    config = {
        "ml": {
            "stock_bar_cadence_feature_bank_path": str(path),
            "stock_bar_cadence_replay_timeframe": "1Day",
            "stock_bar_cadence_symbols": ["AAA", "BBB", "CCC"],
            "stock_bar_cadence_target_horizon_bars": 1,
            "stock_bar_cadence_refit_frequency_bars": 3,
            "stock_bar_cadence_min_train_rows": 9,
            "stock_bar_cadence_replay_top_n": 1,
            "stock_bar_cadence_replay_max_position_weight": 1.0,
            "stock_bar_cadence_replay_cost_bps": 5,
            "stock_bar_cadence_replay_slippage_bps": 5,
        }
    }

    result = build_bar_cadence_research_experiment(config)

    assert result.predictions
    assert result.replay.periods
    assert result.payload["prediction_report"]["duplicate_prediction_key_count"] == 0
    assert result.payload["replay_summary"]["period_count"] > 0
    assert result.payload["replay_summary"]["transaction_cost_drag"] > 0
    assert result.payload["replay_parallelism"]["stateful_execution_workers"] == 1


def test_parallel_worker_setting_preserves_prediction_keys_and_replay_metrics(tmp_path: Path):
    path = tmp_path / "stock_features_1Day.parquet"
    pq.write_table(pa.Table.from_pylist(_feature_rows(symbols=("AAA", "BBB", "CCC"), count=18)), path)
    base = {
        "ml": {
            "stock_bar_cadence_feature_bank_path": str(path),
            "stock_bar_cadence_replay_timeframe": "1Day",
            "stock_bar_cadence_symbols": ["AAA", "BBB", "CCC"],
            "stock_bar_cadence_target_horizon_bars": 1,
            "stock_bar_cadence_refit_frequency_bars": 3,
            "stock_bar_cadence_min_train_rows": 9,
            "stock_bar_cadence_replay_top_n": 1,
            "stock_bar_cadence_replay_max_position_weight": 1.0,
        }
    }
    sequential = build_bar_cadence_research_experiment({**base, "ml": {**base["ml"], "stock_bar_cadence_replay_n_jobs": 1}})
    parallel = build_bar_cadence_research_experiment({**base, "ml": {**base["ml"], "stock_bar_cadence_replay_n_jobs": 3}})

    seq_keys = [(row["timeframe"], row["model"], row["timestamp"], row["symbol"]) for row in sequential.predictions]
    par_keys = [(row["timeframe"], row["model"], row["timestamp"], row["symbol"]) for row in parallel.predictions]
    assert par_keys == seq_keys
    assert parallel.replay.summary == sequential.replay.summary
    assert parallel.replay.periods == sequential.replay.periods
    assert parallel.replay.decisions == sequential.replay.decisions
    assert parallel.replay.payload["parallelism"]["effective_workers"] == 1
    assert parallel.replay.payload["parallelism"]["independent_unit_workers"] == 3


def test_experiment_uses_stock_ranker_model_registry_for_multiple_tabular_models(tmp_path: Path):
    path = tmp_path / "stock_features_1Day.parquet"
    pq.write_table(pa.Table.from_pylist(_feature_rows(symbols=("AAA", "BBB", "CCC"), count=18)), path)

    result = build_bar_cadence_research_experiment(
        {
            "ml": {
                "stock_bar_cadence_feature_bank_path": str(path),
                "stock_bar_cadence_replay_timeframe": "1Day",
                "stock_bar_cadence_models": ["ridge", "elastic_net"],
                "stock_bar_cadence_symbols": ["AAA", "BBB", "CCC"],
                "stock_bar_cadence_target_horizon_bars": 1,
                "stock_bar_cadence_refit_frequency_bars": 3,
                "stock_bar_cadence_min_train_rows": 9,
                "stock_bar_cadence_replay_top_n": 1,
                "stock_bar_cadence_replay_max_position_weight": 1.0,
                "stock_bar_cadence_replay_n_jobs": 2,
            }
        }
    )

    assert set(result.payload["models"]) == {"ridge", "elastic_net"}
    assert set(result.payload["prediction_report_by_model"]) == {"ridge", "elastic_net"}
    assert set(result.payload["replay_summary_by_model"]) == {"ridge", "elastic_net"}
    assert {
        row["model"]
        for row in result.predictions
    } == {"ridge", "elastic_net"}
    assert result.payload["independent_unit_parallelism"]["effective_workers"] == 2
    assert result.payload["prediction_report_by_model"]["elastic_net"]["model_registry_family"] == "tabular_regressor"


def test_sequence_model_can_score_the_feature_bank_artifact_interface():
    pytest.importorskip("torch")
    feature_slice = FeatureBankSlice(
        rows=_feature_rows(symbols=("AAA", "BBB"), count=10),
        metadata={"timeframe": "1Day"},
        metadata_columns=("timestamp", "symbol", "timeframe"),
        feature_columns=("feature_a", "feature_b", "open", "close"),
        label_columns=(),
    )

    predictions, payload = build_every_bar_oos_predictions(
        feature_slice,
        horizon_bars=1,
        model_name="dlinear",
        refit_frequency_bars=3,
        min_train_rows=4,
        sequence_length=2,
        sequence_epochs=1,
        sequence_batch_size=4,
        torch_num_threads=1,
    )

    assert predictions
    assert payload["fit_count"] >= 1
    assert payload["model_registry_family"] == "sequence_regressor"
    assert all(row["model"] == "dlinear" for row in predictions)


def test_bar_cadence_model_set_resolves_existing_deep_model_families():
    models, metadata = _resolve_models(
        {
            "stock_bar_cadence_model_set": "full",
            "stock_bar_cadence_include_sequence_models": True,
        }
    )

    for model_name in FULL_SEQUENCE_MODELS:
        assert model_name in models
    assert metadata["effective_model_set"] == "full"


def test_replay_grid_reuses_fixed_predictions_for_cost_and_top_n_scenarios():
    predictions = []
    for row in _feature_rows(symbols=("AAA", "BBB"), count=5):
        predictions.append(
            {
                **row,
                "model": "ridge",
                "prediction": 1.0 if row["symbol"] == "AAA" else 0.5,
            }
        )

    grid = build_replay_grid_from_predictions(
        predictions,
        timeframe="1Day",
        scenarios=[
            {"top_n": 1, "max_position_weight": 1.0, "cost_bps": 0, "slippage_bps": 0},
            {"top_n": 2, "max_position_weight": 0.5, "cost_bps": 10, "slippage_bps": 5},
        ],
        max_workers=2,
    )

    assert grid["enabled"] is True
    assert grid["scenario_count"] == 2
    assert grid["parallelism"]["effective_workers"] == 2
    assert grid["results"][0]["summary"]["trade_count"] > 0
    assert grid["results"][1]["summary"]["cost_drag"] > 0.0


def _feature_rows(*, symbols=("AAA", "BBB"), count=10):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        timestamp = start + timedelta(days=index)
        for symbol_index, symbol in enumerate(symbols):
            base = 100.0 + index * (1.0 + symbol_index * 0.2) + symbol_index * 5.0
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "timeframe": "1Day",
                    "open": base,
                    "high": base + 1.0,
                    "low": base - 1.0,
                    "close": base + 0.5,
                    "volume": 1000.0 + index * 10 + symbol_index,
                    "feature_a": float(index + symbol_index),
                    "feature_b": float((index % 4) - symbol_index),
                }
            )
    return rows


def _target_row(timestamp: str, close: float) -> dict:
    return {
        "timestamp": datetime.fromisoformat(timestamp),
        "symbol": "AAA",
        "timeframe": "5m",
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
        "feature_a": close,
        "feature_b": close / 10.0,
    }
