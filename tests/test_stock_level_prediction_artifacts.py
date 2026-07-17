from __future__ import annotations

import csv
import hashlib
import inspect
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from infrastructure.data.market_sessions import EASTERN, next_trading_session, rth_close_for_date, trading_sessions
from core.research.ml.stock_level import stock_level_prediction_artifacts
from core.research.ml.stock_level.prediction_artifacts.service import (
    build_stock_level_prediction_artifacts,
    write_stock_level_prediction_artifacts,
)
from core.research.ml.stock_level.prediction_artifacts import service as artifact_service
from core.research.ml.stock_level.prediction_artifacts.sources import _universe_symbols
from core.research.ml.stock_level.prediction_artifacts.types import (
    TARGET_PROVENANCE_CONTRACT_VERSION,
)
from core.research.ml.stock_level.stock_level_alpha_features_io import _write_enriched_csv
from core.research.ml.stock_level.stock_level_artifact_io import read_stock_level_artifact


def test_stock_level_artifacts_create_one_row_per_symbol_date():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-08")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": _closes(100.0),
            "BBB": _closes(50.0),
        },
    )

    keys = {(row["rebalance_date"], row["symbol"]) for row in rows}

    assert len(rows) == 4
    assert keys == {
        ("2024-01-01", "AAA"),
        ("2024-01-01", "BBB"),
        ("2024-01-08", "AAA"),
        ("2024-01-08", "BBB"),
    }
    assert audit["true_stock_level_rows"] is True
    assert audit["row_count"] == 4
    assert audit["symbol_count"] == 2
    assert audit["rebalance_date_count"] == 2


def test_missing_predictions_are_reported_explicitly():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": _closes(100.0),
            "BBB": _closes(50.0),
        },
    )

    assert all(row["predicted_probability"] == "" for row in rows)
    assert audit["missing_prediction_counts"]["predicted_probability"] == 2
    assert audit["missing_prediction_counts"]["predicted_momentum_20d"] == 2
    assert audit["artifact_rows_with_symbol_predictions"] == 0
    assert audit["suitable_for_true_stock_level_ranking_diagnostics"] is False
    assert "do not contain symbol-level predictions" in audit["suitability_reason"]


def test_baseline_features_use_only_prices_before_rebalance_date():
    before = _history_around_rebalance(after_jump=100.0)
    after_changed = _history_around_rebalance(after_jump=10_000.0)

    first_rows, first_audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-04-01")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": before},
    )
    second_rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-04-01")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": after_changed},
    )

    assert first_rows[0]["predicted_momentum_20d"] == second_rows[0][
        "predicted_momentum_20d"
    ]
    assert first_rows[0]["predicted_momentum_60d"] == second_rows[0][
        "predicted_momentum_60d"
    ]
    assert first_rows[0]["predicted_volatility_20d"] == second_rows[0][
        "predicted_volatility_20d"
    ]
    assert first_audit["populated_prediction_counts"]["predicted_momentum_20d"] == 1
    assert first_audit["usable_for_stock_level_ranking"] is True


def test_missing_early_history_baseline_features_are_reported():
    _, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-05")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _history_around_rebalance(after_jump=100.0)},
    )

    assert audit["missing_prediction_counts"]["predicted_momentum_120d"] == 1


def test_market_residual_is_generated_without_market_in_tradable_universe():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-12")], artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _closes(100.0), "SPY": _closes(200.0)},
        market_symbol="SPY",
    )
    expected_benchmark = (210.0 / 200.0) - 1.0
    expected_raw = (110.0 / 100.0) - 1.0
    assert rows[0]["benchmark_symbol"] == "SPY"
    assert rows[0]["actual_benchmark_return_10d"] == pytest.approx(expected_benchmark)
    assert rows[0]["actual_market_residual_return_10d"] != ""
    assert rows[0]["actual_market_residual_return_10d"] == pytest.approx(
        expected_raw - expected_benchmark
    )
    assert audit["market_residual_label_generation"]["market_symbol_loaded"] is True
    assert audit["market_residual_label_generation"]["market_symbol_is_tradable_candidate"] is False
    assert (
        audit["market_residual_label_generation"]["benchmark_return_column"]
        == "actual_benchmark_return_10d"
    )


def test_configured_benchmark_symbol_supplies_actual_benchmark_return():
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-12")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _closes(100.0), "QQQ": _closes(300.0)},
        market_symbol="QQQ",
    )

    assert rows[0]["benchmark_symbol"] == "QQQ"
    assert rows[0]["actual_benchmark_return_10d"] == pytest.approx(
        (310.0 / 300.0) - 1.0
    )


def test_forward_target_provenance_uses_same_future_observation_and_preserves_value():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-12")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _closes(100.0), "SPY": _closes(200.0)},
    )

    row = rows[0]

    assert row["actual_forward_return_10d"] == pytest.approx((110.0 / 100.0) - 1.0)
    assert row["target_provenance_contract_version"] == TARGET_PROVENANCE_CONTRACT_VERSION
    assert row["feature_timestamp"] == "2024-01-01"
    assert row["decision_timestamp"] == "2024-01-01"
    assert row["target_horizon"] == "10_trading_observations"
    assert row["target_observation_count"] == 10
    assert row["target_start_timestamp"] == "2024-01-01"
    assert row["label_start_timestamp"] == _close_ts("2024-01-02")
    assert row["label_end_timestamp"] == _close_ts("2024-01-11")
    assert row["label_available_timestamp"] == _close_ts("2024-01-12")
    assert row["benchmark_label_start_timestamp"] == row["label_start_timestamp"]
    assert row["benchmark_label_end_timestamp"] == row["label_end_timestamp"]
    assert row["benchmark_label_available_timestamp"] == row["label_available_timestamp"]
    assert audit["target_provenance_audit"]["complete_rows"] == 1


def test_forward_target_availability_uses_ordered_trading_calendar_not_calendar_offset():
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02"), _expanded("2024-01-17")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _business_day_closes(100.0), "SPY": _business_day_closes(200.0)},
    )

    row = next(row for row in rows if row["rebalance_date"] == "2024-01-02")

    assert row["actual_forward_return_10d"] == pytest.approx((110.0 / 100.0) - 1.0)
    assert row["label_end_timestamp"] == _close_ts("2024-01-16")
    assert row["label_available_timestamp"] == _close_ts("2024-01-17")


def test_missing_benchmark_data_leaves_benchmark_outcome_blank():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _closes(100.0)},
        market_symbol="QQQ",
    )

    assert rows[0]["benchmark_symbol"] == "QQQ"
    assert rows[0]["actual_benchmark_return_10d"] == ""
    assert rows[0]["actual_market_residual_return_10d"] == ""
    assert audit["market_residual_label_generation"]["market_symbol_loaded"] is False


def test_risk_aware_targets_are_populated_with_required_history():
    rebalance = (date(2024, 1, 1) + timedelta(days=30)).isoformat()
    later_decision = (date.fromisoformat(rebalance) + timedelta(days=11)).isoformat()
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded(rebalance), _expanded(later_decision)], artifact_rows=[], universe_symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": _long_closes(100, 1.0), "BBB": _long_closes(80, .4), "SPY": _long_closes(200, .3)},
    )
    rebalance_rows = [row for row in rows if row["rebalance_date"] == rebalance]
    assert all(row["actual_vol_adjusted_forward_return_10d"] != "" for row in rebalance_rows)
    assert all(row["actual_market_residual_return_10d"] != "" for row in rebalance_rows)
    assert all(row["actual_rank_normalized_forward_return_10d"] != "" for row in rebalance_rows)


def test_existing_artifact_level_files_are_preserved(tmp_path):
    output_dir = tmp_path / "reports" / "meta"
    cache_dir = tmp_path / "cache"
    universe_path = tmp_path / "universe.yaml"
    expanded_path = cache_dir / "expanded_rebalance_dataset.csv"
    old_artifact = output_dir / "prediction_artifacts.csv"
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    old_artifact.write_text("sentinel\n", encoding="utf-8")
    universe_path.write_text(
        yaml.safe_dump({"symbols": ["AAA", "BBB"]}),
        encoding="utf-8",
    )
    _write_csv(expanded_path, [_expanded("2024-01-01")])
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "rebalance_date,feature_id,symbol\n2024-01-01,feature-a,\n",
        encoding="utf-8",
    )

    paths = write_stock_level_prediction_artifacts(
        {
            "cache": {"ml_dir": str(cache_dir)},
            "ml": {
                "output_dir": str(output_dir),
                "expanded_rebalance_dataset_path": str(expanded_path),
                "expanded_rebalance_dataset": {
                    "universe_paths": [str(universe_path)],
                },
                "stooq_parquet_dir": str(tmp_path / "missing_parquet"),
            },
        }
    )

    assert paths.csv_path.exists()
    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert old_artifact.read_text(encoding="utf-8") == "sentinel\n"
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["existing_artifact_level_files_preserved"] is True
    artifact_rows = read_stock_level_artifact(paths.csv_path)
    assert artifact_rows[0]["benchmark_symbol"] == "SPY"


def test_dataset_workers_report_effective_parallelism_and_inner_thread_cap():
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-12")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": _closes(100.0), "BBB": _closes(50.0), "SPY": _closes(200.0)},
        dataset_workers=12,
        inner_thread_limit=1,
        executor_cls=ReversingExecutor,
    )

    assert [(row["rebalance_date"], row["symbol"]) for row in rows] == [
        ("2024-01-01", "AAA"),
        ("2024-01-01", "BBB"),
        ("2024-01-12", "AAA"),
        ("2024-01-12", "BBB"),
    ]
    parallelism = audit["dataset_parallelism"]
    assert parallelism["parallelism_owner"] == "stock_level_prediction_artifacts_symbol_tasks"
    assert parallelism["requested_workers"] == 12
    assert parallelism["effective_workers"] == 2
    assert parallelism["task_count"] == 2
    assert parallelism["completed_task_count"] == 2
    assert parallelism["failed_task_count"] == 0
    assert parallelism["inner_thread_limit"] == 1
    assert parallelism["nested_parallelism_prevented"] is True
    assert parallelism["worker_execution_mode"] == "process_pool"
    phase_names = {phase["phase_name"] for phase in audit["phase_timings"]}
    assert {
        "daily-grid construction",
        "symbol-data preparation",
        "symbol-task dispatch",
        "symbol-task execution",
        "worker result collection",
        "cross-sectional calculation",
        "deterministic sorting",
    }.issubset(phase_names)
    symbol_execution = next(
        phase for phase in audit["phase_timings"] if phase["phase_name"] == "symbol-task execution"
    )
    assert symbol_execution["requested_workers"] == 12
    assert symbol_execution["effective_workers"] == 2
    assert symbol_execution["execution_mode"] == "process_pool"


def test_symbol_task_diagnostics_jsonl_records_lifecycle_events(tmp_path):
    diagnostics_path = tmp_path / "stock_artifact_symbol_tasks.jsonl"

    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01"), _expanded("2024-01-12")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={"AAA": _closes(100.0), "BBB": _closes(50.0), "SPY": _closes(200.0)},
        dataset_workers=1,
        inner_thread_limit=1,
        diagnostics_path=diagnostics_path,
    )

    events = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    assert audit["dataset_parallelism"]["diagnostics_path"] == str(diagnostics_path)
    assert {event["event_type"] for event in events} == {
        "dispatched",
        "started",
        "completed",
    }
    assert [event["symbol"] for event in events if event["event_type"] == "completed"] == [
        "AAA",
        "BBB",
    ]
    completed = next(event for event in events if event["event_type"] == "completed")
    assert completed["pid"]
    assert completed["rows_read"] == 20
    assert completed["rows_emitted"] == 2
    assert completed["date_count"] == 2
    assert completed["seconds_elapsed"] >= 0.0


def test_symbol_partitions_resume_without_recomputing_completed_symbols(tmp_path):
    partition_dir = tmp_path / "partitions"
    diagnostics_path = tmp_path / "diagnostics.jsonl"
    kwargs = {
        "expanded_rows": [_expanded("2024-01-01"), _expanded("2024-01-12")],
        "artifact_rows": [],
        "universe_symbols": ["AAA", "BBB"],
        "closes_by_symbol": {"AAA": _closes(100.0), "BBB": _closes(50.0), "SPY": _closes(200.0)},
        "partition_dir": partition_dir,
        "diagnostics_path": diagnostics_path,
    }

    first_rows, first_audit = build_stock_level_prediction_artifacts(**kwargs, dataset_workers=1)
    second_rows, second_audit = build_stock_level_prediction_artifacts(**kwargs, dataset_workers=1)

    assert second_rows == first_rows
    assert first_audit["dataset_parallelism"]["written_partition_count"] == 2
    assert second_audit["dataset_parallelism"]["reused_partition_count"] == 2
    assert second_audit["dataset_parallelism"]["written_partition_count"] == 0
    events = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [event["event_type"] for event in events].count("partition_reused") == 2


def test_corrupt_symbol_partition_fails_closed(tmp_path):
    partition_dir = tmp_path / "partitions"
    partition_dir.mkdir()
    (partition_dir / "AAA.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt stock-level symbol partition"):
        build_stock_level_prediction_artifacts(
            expanded_rows=[_expanded("2024-01-01")],
            artifact_rows=[],
            universe_symbols=["AAA"],
            closes_by_symbol={"AAA": _closes(100.0), "SPY": _closes(200.0)},
            partition_dir=partition_dir,
        )


def test_duplicate_or_wrong_symbol_partition_is_rejected(tmp_path):
    partition_dir = tmp_path / "partitions"
    build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-01")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={"AAA": _closes(100.0), "SPY": _closes(200.0)},
        partition_dir=partition_dir,
    )
    payload = json.loads((partition_dir / "AAA.json").read_text(encoding="utf-8"))
    payload["symbol"] = "BBB"
    (partition_dir / "AAA.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="symbol mismatch"):
        build_stock_level_prediction_artifacts(
            expanded_rows=[_expanded("2024-01-01")],
            artifact_rows=[],
            universe_symbols=["AAA"],
            closes_by_symbol={"AAA": _closes(100.0), "SPY": _closes(200.0)},
            partition_dir=partition_dir,
        )


def test_partition_resume_keeps_deterministic_merge_order(tmp_path):
    partition_dir = tmp_path / "partitions"
    kwargs = {
        "expanded_rows": [_expanded("2024-01-01"), _expanded("2024-01-12")],
        "artifact_rows": [],
        "universe_symbols": ["BBB", "AAA"],
        "closes_by_symbol": {"AAA": _closes(100.0), "BBB": _closes(50.0), "SPY": _closes(200.0)},
        "partition_dir": partition_dir,
    }

    first_rows, _ = build_stock_level_prediction_artifacts(**kwargs, dataset_workers=12, executor_cls=ReversingExecutor)
    second_rows, _ = build_stock_level_prediction_artifacts(**kwargs, dataset_workers=1)

    assert second_rows == first_rows
    assert [(row["rebalance_date"], row["symbol"]) for row in second_rows] == [
        ("2024-01-01", "AAA"),
        ("2024-01-01", "BBB"),
        ("2024-01-12", "AAA"),
        ("2024-01-12", "BBB"),
    ]


def test_one_worker_and_twelve_worker_dataset_builds_are_equivalent():
    kwargs = {
        "expanded_rows": [_expanded("2024-01-01"), _expanded("2024-01-12")],
        "artifact_rows": [],
        "universe_symbols": ["BBB", "AAA"],
        "closes_by_symbol": {"AAA": _closes(100.0), "BBB": _closes(50.0), "SPY": _closes(200.0)},
    }
    one_rows, one_audit = build_stock_level_prediction_artifacts(
        **kwargs,
        dataset_workers=1,
        inner_thread_limit=1,
    )
    twelve_rows, twelve_audit = build_stock_level_prediction_artifacts(
        **kwargs,
        dataset_workers=12,
        inner_thread_limit=1,
        executor_cls=ReversingExecutor,
    )

    assert twelve_rows == one_rows
    assert twelve_audit["row_count"] == one_audit["row_count"]
    assert twelve_audit["symbol_count"] == one_audit["symbol_count"]
    assert twelve_audit["rebalance_date_count"] == one_audit["rebalance_date_count"]
    assert twelve_audit["target_provenance_audit"] == one_audit["target_provenance_audit"]


def test_duplicate_worker_results_fail_validation():
    with pytest.raises(ValueError, match="duplicates=\\['AAA'\\].*missing=\\['BBB'\\]"):
        build_stock_level_prediction_artifacts(
            expanded_rows=[_expanded("2024-01-01")],
            artifact_rows=[],
            universe_symbols=["AAA", "BBB"],
            closes_by_symbol={"AAA": _closes(100.0), "BBB": _closes(50.0)},
            dataset_workers=2,
            executor_cls=DuplicateExecutor,
        )


def test_missing_worker_results_fail_validation():
    with pytest.raises(ValueError, match="missing=\\['BBB'\\]"):
        build_stock_level_prediction_artifacts(
            expanded_rows=[_expanded("2024-01-01")],
            artifact_rows=[],
            universe_symbols=["AAA", "BBB"],
            closes_by_symbol={"AAA": _closes(100.0), "BBB": _closes(50.0)},
            dataset_workers=2,
            executor_cls=MissingExecutor,
        )


def test_symbol_task_timeout_fails_closed_with_pending_symbols(tmp_path):
    diagnostics_path = tmp_path / "timeout_diagnostics.jsonl"
    with pytest.raises(TimeoutError, match="pending_symbols=\\['AAA', 'BBB'\\]"):
        build_stock_level_prediction_artifacts(
            expanded_rows=[_expanded("2024-01-01")],
            artifact_rows=[],
            universe_symbols=["AAA", "BBB"],
            closes_by_symbol={"AAA": _closes(100.0), "BBB": _closes(50.0)},
            dataset_workers=2,
            task_timeout_seconds=0.01,
            progress_interval_seconds=1.0,
            diagnostics_path=diagnostics_path,
            executor_cls=NeverCompletesExecutor,
        )
    events = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["event_type"] for event in events].count("timeout") == 2


def test_worker_task_failure_prevents_canonical_publication(tmp_path, monkeypatch):
    output_dir = tmp_path / "reports" / "meta"
    cache_dir = tmp_path / "cache"
    universe_path = tmp_path / "universe.yaml"
    expanded_path = cache_dir / "expanded_rebalance_dataset.csv"
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    universe_path.write_text(yaml.safe_dump({"symbols": ["AAA"]}), encoding="utf-8")
    _write_csv(expanded_path, [_expanded("2024-01-01")])
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "rebalance_date,feature_id,symbol\n2024-01-01,feature-a,\n",
        encoding="utf-8",
    )

    def fail_build(**_kwargs):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(artifact_service, "build_stock_level_prediction_artifacts", fail_build)
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        artifact_service.write_stock_level_prediction_artifacts(
            {
                "cache": {"ml_dir": str(cache_dir)},
                "ml": {
                    "output_dir": str(output_dir),
                    "expanded_rebalance_dataset_path": str(expanded_path),
                    "expanded_rebalance_dataset": {"universe_paths": [str(universe_path)]},
                    "stooq_parquet_dir": str(tmp_path / "missing_parquet"),
                    "stock_level_dataset_workers": 2,
                },
            }
        )

    assert not (output_dir / "stock_level_prediction_artifacts.parquet").exists()
    assert not list(output_dir.glob("*.worker*.tmp"))
    failure = json.loads(
        (output_dir / "stock_artifact_consolidation_manifest.json").read_text()
    )
    assert failure["failure_phase"] == "symbol_task_execution"
    assert failure["alpha_enrichment_allowed"] is False


def test_interruption_after_partitions_before_publication_leaves_no_complete_artifact(tmp_path, monkeypatch):
    output_dir = tmp_path / "reports" / "meta"
    cache_dir = tmp_path / "cache"
    universe_path = tmp_path / "universe.yaml"
    expanded_path = cache_dir / "expanded_rebalance_dataset.csv"
    parquet_dir = tmp_path / "prices"
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    universe_path.write_text(yaml.safe_dump({"symbols": ["AAA"]}), encoding="utf-8")
    _write_csv(expanded_path, [_expanded("2024-01-01"), _expanded("2024-01-12")])
    (output_dir / "meta_auxiliary_predictions.csv").write_text(
        "rebalance_date,feature_id,symbol\n2024-01-01,feature-a,\n",
        encoding="utf-8",
    )
    _write_price_parquet(parquet_dir / "AAA.parquet", _closes(100.0))
    _write_price_parquet(parquet_dir / "SPY.parquet", _closes(200.0))

    def fail_publish(*_args, **_kwargs):
        raise RuntimeError("synthetic publication interruption")

    monkeypatch.setattr(
        artifact_service, "finalize_base_from_partitions", fail_publish,
    )
    with pytest.raises(RuntimeError, match="synthetic publication interruption"):
        artifact_service.write_stock_level_prediction_artifacts(
            {
                "cache": {"ml_dir": str(cache_dir)},
                "ml": {
                    "output_dir": str(output_dir),
                    "expanded_rebalance_dataset_path": str(expanded_path),
                    "expanded_rebalance_dataset": {"universe_paths": [str(universe_path)]},
                    "stooq_parquet_dir": str(parquet_dir),
                    "stock_level_artifact_format": "parquet",
                    "stock_level_dataset_workers": 1,
                },
            }
        )

    assert (output_dir / "stock_artifact_symbol_partitions" / "AAA.json").exists()
    assert not (output_dir / "stock_level_prediction_artifacts.parquet").exists()
    assert not (output_dir / "stock_level_prediction_artifacts.json").exists()


def test_daily_exchange_session_grid_generates_consecutive_sessions_with_timing():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 31))
    ]
    rows, audit = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02"), _expanded("2024-01-09")],
        artifact_rows=[],
        universe_symbols=["AAA", "BBB"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "BBB": _session_closes(sessions, 50.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-31",
        decision_grid_max_sessions=5,
        decision_grid_min_history_sessions=1,
    )

    dates = sorted({row["rebalance_date"] for row in rows})

    assert dates == sessions[1:6]
    assert audit["decision_frequency"] == "daily"
    assert audit["decision_grid"]["overlapping_targets"] is True
    assert len(rows) == 10
    assert all(row["decision_session_date"] == row["rebalance_date"] for row in rows)
    assert all(row["feature_data_cutoff_timestamp"] <= row["decision_timestamp"] for row in rows)
    assert all(
        row["first_actionable_session"]
        == next_trading_session(date.fromisoformat(row["rebalance_date"])).isoformat()
        for row in rows
    )
    assert all(row["context_asof_join_direction"] == "backward" for row in rows)
    assert rows[0]["context_source_timestamp"] == "2024-01-02"


def test_daily_targets_are_ten_trading_days_and_boundary_rows_are_explicit():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 2, 15))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-02-15",
        decision_grid_min_history_sessions=1,
    )

    realized = next(row for row in rows if row["target_status"] == "realized")
    boundary = rows[-1]

    assert realized["target_horizon"] == "10_trading_observations"
    assert realized["target_observation_count"] == 10
    assert realized["label_start_timestamp"] == _close_ts(sessions[sessions.index(realized["rebalance_date"]) + 1])
    assert realized["label_end_timestamp"] == _close_ts(sessions[sessions.index(realized["rebalance_date"]) + 10])
    assert realized["label_available_timestamp"] == _close_ts(sessions[sessions.index(realized["rebalance_date"]) + 11])
    assert realized["benchmark_label_end_timestamp"] == realized["label_end_timestamp"]
    assert (
        realized["benchmark_label_available_timestamp"]
        == realized["label_available_timestamp"]
    )
    assert boundary["target_status"] == "unrealized_boundary"
    assert boundary["actual_forward_return_10d"] == ""


def test_daily_target_metadata_uses_next_session_close_after_intraday_decision():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 25))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-25",
        decision_grid_min_history_sessions=0,
    )

    row = next(row for row in rows if row["rebalance_date"] == "2024-01-02")

    assert row["target_start_timestamp"] == "2024-01-02"
    assert row["decision_timestamp"] < row["label_start_timestamp"]
    assert row["label_start_timestamp"] == _close_ts("2024-01-03")
    assert row["label_start_timestamp"] <= row["label_end_timestamp"]
    assert row["label_end_timestamp"] <= row["label_available_timestamp"]


def test_daily_target_horizon_crosses_weekend_and_holiday_by_ordered_observations():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 12), date(2024, 2, 5))
    ]
    assert "2024-01-15" not in sessions  # MLK Day.
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-12")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 50.0),
            "SPY": _session_closes(sessions, 100.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-12",
        decision_grid_end_date="2024-02-05",
        decision_grid_min_history_sessions=0,
    )

    row = next(row for row in rows if row["rebalance_date"] == "2024-01-12")
    start_index = sessions.index("2024-01-12")
    end_session = sessions[start_index + 10]

    assert end_session == "2024-01-29"
    assert row["label_start_timestamp"] == _close_ts("2024-01-16")
    assert row["label_end_timestamp"] == _close_ts(end_session)
    assert row["actual_forward_return_10d"] == pytest.approx((60.0 / 50.0) - 1.0)


def test_missing_future_history_keeps_existing_unrealized_boundary_policy():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 12))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-12",
        decision_grid_min_history_sessions=0,
    )

    assert all(row["target_status"] == "unrealized_boundary" for row in rows)
    assert all(row["actual_forward_return_10d"] == "" for row in rows)


def test_benchmark_and_residual_targets_share_future_label_contract():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 25))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "QQQ": _session_closes(sessions, 300.0),
        },
        market_symbol="QQQ",
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-25",
        decision_grid_min_history_sessions=0,
    )

    row = next(row for row in rows if row["target_status"] == "realized")

    assert row["benchmark_label_start_timestamp"] == row["label_start_timestamp"]
    assert row["benchmark_label_end_timestamp"] == row["label_end_timestamp"]
    assert row["benchmark_label_available_timestamp"] == row["label_available_timestamp"]
    assert row["actual_market_residual_return_10d"] == pytest.approx(
        row["actual_forward_return_10d"] - row["actual_benchmark_return_10d"]
    )


def test_metadata_only_correction_preserves_numerical_target_and_changes_identity(tmp_path):
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 25))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-25",
        decision_grid_min_history_sessions=1,
    )
    row = next(row for row in rows if row["target_status"] == "realized")

    legacy = {
        **row,
        "target_provenance_contract_version": "stock_level_target_provenance_v1",
        "label_start_timestamp": row["label_start_timestamp"][:10],
        "label_end_timestamp": row["label_end_timestamp"][:10],
        "label_available_timestamp": row["label_available_timestamp"][:10],
        "benchmark_label_start_timestamp": row["benchmark_label_start_timestamp"][:10],
        "benchmark_label_end_timestamp": row["benchmark_label_end_timestamp"][:10],
        "benchmark_label_available_timestamp": row["benchmark_label_available_timestamp"][:10],
    }

    assert row["actual_forward_return_10d"] == pytest.approx(legacy["actual_forward_return_10d"])
    assert _logical_row_id(row) != _logical_row_id(legacy)


def test_genuine_horizon_change_changes_numerical_target():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 25))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-25",
        decision_grid_min_history_sessions=0,
    )

    row = next(row for row in rows if row["rebalance_date"] == "2024-01-02")
    start = 100.0
    ten_session = (110.0 / start) - 1.0
    nine_session = (109.0 / start) - 1.0

    assert row["actual_forward_return_10d"] == pytest.approx(ten_session)
    assert row["actual_forward_return_10d"] != pytest.approx(nine_session)


def test_enriched_writer_preserves_base_target_metadata_equality(tmp_path):
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 1, 25))
    ]
    rows, _ = build_stock_level_prediction_artifacts(
        expanded_rows=[_expanded("2024-01-02")],
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol={
            "AAA": _session_closes(sessions, 100.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-01-25",
        decision_grid_min_history_sessions=1,
    )
    base_row = next(row for row in rows if row["target_status"] == "realized")
    enriched_row = {**base_row, "_stock_above_200d_average": 1.0}
    path = tmp_path / "stock_level_prediction_artifacts_enriched.parquet"

    _write_enriched_csv(path, [base_row], [enriched_row], config={"ml": {"stock_level_artifact_format": "parquet"}})
    [written] = read_stock_level_artifact(path)

    for column in (
        "target_start_timestamp",
        "label_start_timestamp",
        "label_end_timestamp",
        "label_available_timestamp",
        "benchmark_target_start_timestamp",
        "benchmark_label_start_timestamp",
        "benchmark_label_end_timestamp",
        "benchmark_label_available_timestamp",
    ):
        assert _utc_text(written[column]) == _utc_text(base_row[column])


def test_future_price_and_future_context_mutations_do_not_change_earlier_daily_features():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 2, 15))
    ]
    base_context = [_expanded("2024-01-02"), _expanded("2024-01-16")]
    changed_context = [dict(row) for row in base_context]
    changed_context.append({**_expanded("2024-02-14"), "breadth_above_sma_200": "0.99"})
    base_prices = {
        "AAA": _session_closes(sessions, 100.0),
        "SPY": _session_closes(sessions, 200.0),
    }
    changed_prices = {
        symbol: {key: dict(value) for key, value in payload.items()}
        for symbol, payload in base_prices.items()
    }
    changed_prices["AAA"]["close"][sessions[-1]] = 99_999.0

    first, _ = build_stock_level_prediction_artifacts(
        expanded_rows=base_context,
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol=base_prices,
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-02-15",
        decision_grid_max_sessions=10,
        decision_grid_min_history_sessions=1,
    )
    second, _ = build_stock_level_prediction_artifacts(
        expanded_rows=changed_context,
        artifact_rows=[],
        universe_symbols=["AAA"],
        closes_by_symbol=changed_prices,
        decision_grid_frequency="daily",
        decision_grid_start_date="2024-01-02",
        decision_grid_end_date="2024-02-15",
        decision_grid_max_sessions=10,
        decision_grid_min_history_sessions=1,
    )

    comparable = [
        "predicted_momentum_20d",
        "predicted_momentum_60d",
        "predicted_risk_adjusted_momentum",
        "average_dollar_volume_21d",
        "breadth_above_sma_200",
        "context_source_timestamp",
    ]
    assert [{key: row[key] for key in comparable} for row in first] == [
        {key: row[key] for key in comparable} for row in second
    ]


def test_daily_one_worker_and_twelve_worker_semantic_content_match():
    sessions = [
        day.isoformat()
        for day in trading_sessions(date(2024, 1, 2), date(2024, 2, 15))
    ]
    kwargs = {
        "expanded_rows": [_expanded("2024-01-02"), _expanded("2024-01-16")],
        "artifact_rows": [],
        "universe_symbols": ["BBB", "AAA"],
        "closes_by_symbol": {
            "AAA": _session_closes(sessions, 100.0),
            "BBB": _session_closes(sessions, 50.0),
            "SPY": _session_closes(sessions, 200.0),
        },
        "decision_grid_frequency": "daily",
        "decision_grid_start_date": "2024-01-02",
        "decision_grid_end_date": "2024-02-15",
        "decision_grid_max_sessions": 12,
        "decision_grid_min_history_sessions": 1,
    }
    one_rows, one_audit = build_stock_level_prediction_artifacts(
        **kwargs,
        dataset_workers=1,
        inner_thread_limit=1,
    )
    twelve_rows, twelve_audit = build_stock_level_prediction_artifacts(
        **kwargs,
        dataset_workers=12,
        inner_thread_limit=1,
        executor_cls=ReversingExecutor,
    )

    assert twelve_rows == one_rows
    assert (
        twelve_audit["decision_grid"]["decision_grid_identity"]
        == one_audit["decision_grid"]["decision_grid_identity"]
    )
    assert twelve_audit["dataset_parallelism"]["effective_workers"] == 2
    assert twelve_audit["dataset_parallelism"]["worker_execution_mode"] == "process_pool"


def test_stock_alpha_artifact_universe_uses_diagnostic_limit_and_required_symbols(tmp_path):
    universe_path = tmp_path / "universe.yaml"
    symbols = ["AAA", "AAPL", "BBB", "CCC", "DDD", "EEE", "SPY", "ZZZ"]
    universe_path.write_text(yaml.safe_dump({"symbols": symbols}), encoding="utf-8")
    config = {
        "ml": {
            "stock_alpha_artifact_universe_paths": [str(universe_path)],
            "stock_alpha_artifact_max_symbols": 5,
            "stock_alpha_artifact_symbol_sample_method": "deterministic_hash",
            "stock_alpha_dev_required_symbols": ["SPY", "AAPL"],
            "expanded_rebalance_dataset": {
                "universe_paths": ["data/reference/universes/current_32.yaml"],
                "max_symbols": 1,
            },
        }
    }

    first = _universe_symbols(config)
    second = _universe_symbols(config)

    assert first == second
    assert len(first) == 5
    assert first[:2] == ["SPY", "AAPL"]
    assert set(first).issubset(set(symbols))


def test_stock_level_prediction_artifacts_has_no_operational_imports():
    source = inspect.getsource(stock_level_prediction_artifacts)

    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def test_normal_service_uses_partition_only_shared_streaming_owner():
    service_source = inspect.getsource(
        artifact_service.write_stock_level_prediction_artifacts
    )
    assert "partition_only=True" in service_source
    assert "finalize_base_from_partitions(" in service_source
    assert "write_stock_level_artifact(" not in service_source
    from scripts.recover_large_daily_stock_artifact import (
        finalize_base_from_partitions as consolidation_owner,
    )
    owner_source = inspect.getsource(consolidation_owner)
    assert "pq.ParquetWriter(" in owner_source
    assert "read_stock_level_artifact(" not in owner_source
    assert "pq.read_table(" not in owner_source
    assert ".to_pylist(" not in owner_source
    rows_source = inspect.getsource(
        build_stock_level_prediction_artifacts
    )
    assert "partition_only" in rows_source


def test_cancelled_future_blocks_partition_consolidation(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    with pytest.raises(RuntimeError, match="cancellation blocks consolidation"):
        build_stock_level_prediction_artifacts(
            expanded_rows=[_expanded("2024-01-01")],
            artifact_rows=[],
            universe_symbols=["AAA", "BBB"],
            closes_by_symbol={
                "AAA": _closes(100.0),
                "BBB": _closes(50.0),
                "SPY": _closes(200.0),
            },
            dataset_workers=2,
            diagnostics_path=diagnostics,
            partition_dir=tmp_path / "partitions",
            partition_only=True,
            executor_cls=CancelledExecutor,
        )
    events = [
        json.loads(line) for line in diagnostics.read_text().splitlines()
    ]
    assert any(row["event_type"] == "cancelled" for row in events)


def _expanded(date: str) -> dict[str, str]:
    return {
        "rebalance_date": date,
        "feature_date": date,
        "breadth_above_sma_200": "0.5",
        "spy_realized_volatility_21d": "0.1",
        "spy_realized_volatility_63d": "0.2",
        "spy_max_drawdown_63d": "-0.03",
        "spy_max_drawdown_126d": "-0.05",
    }


def _close_ts(session: str) -> str:
    close = rth_close_for_date(date.fromisoformat(session))
    assert close is not None
    return datetime.combine(date.fromisoformat(session), close, tzinfo=EASTERN).astimezone(
        timezone.utc
    ).isoformat().replace("+00:00", "Z")


def _utc_text(value: object) -> str:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _logical_row_id(row: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _closes(start: float) -> dict[str, dict[str, float]]:
    close = {}
    dollar_volume = {}
    for index in range(20):
        day = index + 1
        date = f"2024-01-{day:02d}"
        close[date] = start + index
        dollar_volume[date] = (start + index) * 1_000_000
    return {"close": close, "dollar_volume": dollar_volume}


def _session_closes(sessions: list[str], start: float) -> dict[str, dict[str, float]]:
    close = {
        session: start + index
        for index, session in enumerate(sessions)
    }
    return {
        "close": close,
        "dollar_volume": {
            session: value * 1_000_000
            for session, value in close.items()
        },
    }


def _business_day_closes(start: float) -> dict[str, dict[str, float]]:
    close = {}
    dollar_volume = {}
    current = date(2024, 1, 2)
    index = 0
    while len(close) < 20:
        if current.weekday() < 5:
            key = current.isoformat()
            close[key] = start + index
            dollar_volume[key] = (start + index) * 1_000_000
            index += 1
        current += timedelta(days=1)
    return {"close": close, "dollar_volume": dollar_volume}


def _long_closes(start: float, step: float) -> dict[str, dict[str, float]]:
    closes = {(date(2024, 1, 1) + timedelta(days=index)).isoformat(): start + step * index + (index % 3) * .05 for index in range(50)}
    return {"close": closes, "dollar_volume": {key: value * 1_000_000 for key, value in closes.items()}}


class ReversingExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def map(self, fn, tasks):
        return [fn(task) for task in reversed(list(tasks))]


class DuplicateExecutor(ReversingExecutor):
    def map(self, fn, tasks):
        task_list = list(tasks)
        return [fn(task_list[0]), fn(task_list[0])]


class MissingExecutor(ReversingExecutor):
    def map(self, fn, tasks):
        task_list = list(tasks)
        return [fn(task_list[0])]


class NeverCompletesExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.futures = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, _fn, _task):
        from concurrent.futures import Future

        future = Future()
        self.futures.append(future)
        return future


class CancelledExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.shutdown_called = False

    def submit(self, _fn, _task):
        from concurrent.futures import Future

        future = Future()
        future.cancel()
        future.set_running_or_notify_cancel()
        return future

    def shutdown(self, **_kwargs):
        self.shutdown_called = True


def _history_around_rebalance(after_jump: float) -> dict[str, dict[str, float]]:
    close = {}
    dollar_volume = {}
    for index in range(90):
        date = f"2024-01-{index + 1:02d}" if index < 31 else (
            f"2024-02-{index - 30:02d}" if index < 60 else f"2024-03-{index - 59:02d}"
        )
        value = 100.0 + index
        close[date] = value
        dollar_volume[date] = value * 1_000_000
    close["2024-04-02"] = after_jump
    dollar_volume["2024-04-02"] = after_jump * 1_000_000
    return {"close": close, "dollar_volume": dollar_volume}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_price_parquet(path: Path, payload: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp": date,
            "close": payload["close"][date],
            "volume": payload["dollar_volume"][date] / payload["close"][date],
        }
        for date in sorted(payload["close"])
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
