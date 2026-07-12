from __future__ import annotations

import csv
import inspect
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

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
    assert row["label_start_timestamp"] == "2024-01-02"
    assert row["label_end_timestamp"] == "2024-01-11"
    assert row["label_available_timestamp"] == "2024-01-12"
    assert row["benchmark_label_end_timestamp"] == "2024-01-11"
    assert row["benchmark_label_available_timestamp"] == "2024-01-12"
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
    assert row["label_end_timestamp"] == "2024-01-16"
    assert row["label_available_timestamp"] == "2024-01-17"


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


def _closes(start: float) -> dict[str, dict[str, float]]:
    close = {}
    dollar_volume = {}
    for index in range(20):
        day = index + 1
        date = f"2024-01-{day:02d}"
        close[date] = start + index
        dollar_volume[date] = (start + index) * 1_000_000
    return {"close": close, "dollar_volume": dollar_volume}


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
