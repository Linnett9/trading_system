import csv
import json
import subprocess
import sys

import pytest

from core.research.ml.stock_level.selector_target_tournament import (
    build_selector_target_tournament,
    discover_target_contracts,
    write_selector_target_tournament,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    read_stock_level_artifact,
    write_stock_level_artifact,
)


def _rows(date_count=7):
    rows = []
    dates = [f"2024-01-{day:02d}" for day in range(1, date_count + 1)]
    symbols = ("AAA", "BBB", "CCC")
    for date_index, rebalance_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            raw = (2 - symbol_index) * 0.01 + date_index * 0.001
            benchmark = 0.002
            rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "symbol": symbol,
                    "benchmark_symbol": "SPY",
                    "predicted_momentum_20d": 0.1 + symbol_index,
                    "predicted_momentum_60d": 0.2 + symbol_index,
                    "predicted_momentum_120d": 0.3 + symbol_index,
                    "predicted_volatility_20d": 0.05 + symbol_index * 0.01,
                    "predicted_drawdown_60d": -0.02 - symbol_index * 0.01,
                    "predicted_liquidity_score": 1.0 + symbol_index,
                    "predicted_risk_adjusted_momentum": 0.4 + symbol_index,
                    "actual_forward_return_5d": raw / 2,
                    "actual_forward_return_10d": raw,
                    "actual_future_volatility": 0.02,
                    "actual_future_drawdown": min(0.0, raw),
                    "actual_max_adverse_excursion": min(0.0, raw),
                    "actual_benchmark_return_10d": benchmark,
                    "actual_market_residual_return_10d": raw - benchmark,
                    "actual_vol_adjusted_forward_return_10d": raw / 0.05,
                    "actual_drawdown_adjusted_forward_return_10d": raw - abs(min(0.0, raw)),
                    "actual_rank_normalized_forward_return_10d": (2 - symbol_index) / 2,
                    "actual_top_decile_label_10d": int(symbol_index == 0),
                    "decision_session_date": rebalance_date,
                    "target_provenance_contract_version": "stock_level_target_provenance_v1",
                    "feature_timestamp": rebalance_date,
                    "feature_data_cutoff_timestamp": rebalance_date,
                    "decision_timestamp": rebalance_date,
                    "first_actionable_session": f"2024-01-{date_index + 2:02d}",
                    "decision_grid_version": "test-grid-v1",
                    "decision_grid_identity": "grid-123",
                    "exchange_calendar_identity": "XNYS-test",
                    "decision_frequency": "daily",
                    "target_horizon_trading_days": "10",
                    "overlapping_targets": "False",
                    "required_purge_horizon_trading_days": "10",
                    "target_horizon": "10_trading_observations",
                    "target_observation_count": "10",
                    "target_start_timestamp": rebalance_date,
                    "label_start_timestamp": rebalance_date,
                    "label_end_timestamp": f"2024-01-{date_index + 2:02d}",
                    "label_available_timestamp": f"2024-01-{date_index + 2:02d}",
                    "target_price_convention": "simple_close_to_close",
                    "benchmark_target_start_timestamp": rebalance_date,
                    "benchmark_label_start_timestamp": rebalance_date,
                    "benchmark_label_end_timestamp": f"2024-01-{date_index + 2:02d}",
                    "benchmark_label_available_timestamp": f"2024-01-{date_index + 2:02d}",
                    "target_status": "realized",
                }
            )
    return rows


def _settings(**overrides):
    settings = {
        "enabled": True,
        "comparison_mode": "target_intersection",
        "reference_target_id": "raw_return_10d",
        "target_ids": ["raw_return_10d", "market_residual_return_10d"],
        "targets": {},
        "model_ids": ["ridge", "elastic_net"],
        "feature_set_id": "stock_level_default_features",
        "seeds": [42],
        "plan_only": False,
        "source_dataset_path": "source.csv",
        "allow_csv_fallback": False,
        "expected_dataset": {},
        "output_dir": "unused",
        "write_predictions": True,
        "write_target_summary": True,
        "write_portfolio_promotion_report": True,
        "write_debug_csv": False,
        "maximum_decision_dates": None,
        "maximum_symbols": None,
        "maximum_folds": 1,
        "minimum_symbols_per_date": 2,
        "min_train_dates": 2,
        "test_window_dates": 2,
        "embargo_dates": 1,
        "include_engineered_features": False,
        "sklearn_n_jobs": 1,
        "promotion_config": {
            "enabled": True,
            "comparison_mode": "intersection",
            "candidate_types": ["single_model"],
            "fixed_policy": {
                "policy": "long_only_top_n_equal_weight",
                "top_n": 1,
                "cost_bps": 10.0,
                "slippage_bps": 5.0,
                "max_position_weight": 1.0,
                "min_position_weight": 0.0,
            },
            "ranking": {
                "primary_metric": "net_sharpe",
                "secondary_metrics": ["net_cagr", "max_drawdown"],
                "deterministic_tiebreak": "candidate_id",
            },
            "gates": {
                "minimum_oos_decision_dates": 1,
                "minimum_prediction_coverage": 0.95,
                "minimum_net_cagr": None,
                "minimum_net_sharpe": None,
                "maximum_drawdown": None,
                "maximum_annualized_turnover": None,
                "maximum_cost_drag": None,
                "minimum_positive_calendar_year_fraction": None,
                "require_outperformance_of_baseline": False,
            },
            "baseline_candidate_id": "baseline:momentum_120d",
            "multiple_testing": {"enabled": False, "method": "report_only"},
        },
    }
    settings.update(overrides)
    return settings


def test_target_discovery_finds_known_columns_and_rejects_classification():
    contracts = {row["target_id"]: row for row in discover_target_contracts(_rows(), _settings())}
    again = {row["target_id"]: row for row in discover_target_contracts(_rows(), _settings())}
    assert contracts["raw_return_10d"]["classification"] == "available_and_validated"
    assert contracts["market_residual_return_10d"]["target_column"] == "actual_market_residual_return_10d"
    assert contracts["top_decile_label_10d"]["classification"] == "incompatible_with_regression_training_path"
    assert contracts["raw_return_10d"]["contract_identity"] == again["raw_return_10d"]["contract_identity"]


def test_missing_target_is_reported_clearly():
    rows = _rows()
    for row in rows:
        row.pop("actual_market_residual_return_10d")
    contracts = {row["target_id"]: row for row in discover_target_contracts(rows, _settings())}
    assert contracts["market_residual_return_10d"]["classification"] == "unavailable"
    assert contracts["market_residual_return_10d"]["classification_reason"] == "target_column_missing"


def test_target_intersection_excludes_boundary_and_missing_target_rows():
    rows = _rows()
    rows[-1]["target_status"] = "unrealized_boundary"
    rows[0]["actual_market_residual_return_10d"] = ""
    payload = build_selector_target_tournament(
        rows,
        config={"ml": {}},
        source_path=None,
        settings=_settings(plan_only=True),
    )
    stats = payload["plan"]["target_eligibility"]
    assert stats["common_target_intersection_row_count"] == len(rows) - 2
    assert stats["boundary_rows_excluded"]["raw_return_10d"] == 1


def test_label_availability_and_shared_folds_are_recorded():
    payload = build_selector_target_tournament(
        _rows(),
        config={"ml": {}},
        source_path=None,
        settings=_settings(plan_only=True),
    )
    folds = payload["plan"]["shared_fold_plan"]["folds"]
    assert len(folds) == 1
    assert folds[0]["label_availability_guard_passed"] is True
    assert payload["plan"]["shared_fold_plan"]["fold_plan_identity"]
    assert payload["fit_count_plan"]["total_expected_fits"] == 4


def test_leakage_audit_excludes_realized_targets_from_features():
    payload = build_selector_target_tournament(
        _rows(),
        config={"ml": {}},
        source_path=None,
        settings=_settings(plan_only=True),
    )
    audit = payload["plan"]["leakage_audit"]
    assert audit["passed"] is True
    assert "actual_market_residual_return_10d" in audit["denied_outcome_columns"]
    assert not any(column.startswith("actual_") for column in audit["feature_columns"])


def test_run_writes_target_aware_predictions_and_integrates_portfolio_promotion():
    payload = build_selector_target_tournament(
        _rows(),
        config={"ml": {}},
        source_path=None,
        settings=_settings(),
    )
    candidate_ids = {row["candidate_id"] for row in payload["oos_predictions"]}
    assert "raw_return_10d::ridge::seed_42" in candidate_ids
    assert "market_residual_return_10d::elastic_net::seed_42" in candidate_ids
    assert len(candidate_ids) == 4
    assert payload["execution_reconciliation"]["status"] == "reconciled"
    assert payload["execution_reconciliation"]["completed_fits"] == payload["fit_count_plan"]["expected_base_fits"]
    assert payload["promotion_results"]["candidate_metrics"]
    assert all(row["actual_investable_return_10d"] != row["actual_selected_target"] or row["target_id"] == "raw_return_10d" for row in payload["oos_predictions"])
    assert all(row["actual_benchmark_return_10d"] == pytest.approx(0.002) for row in payload["oos_predictions"])
    assert payload["reference_target_deltas"]


def test_plan_only_writes_plan_without_predictions(tmp_path):
    source = tmp_path / "source.parquet"
    _write_parquet(source, _rows())
    paths = write_selector_target_tournament(
        {
            "ml": {
                "selector_target_tournament": {
                    "enabled": True,
                    "source_dataset_path": str(source),
                    "output_dir": str(tmp_path / "out"),
                    "plan_only": True,
                    "min_train_dates": 2,
                    "test_window_dates": 2,
                    "embargo_dates": 1,
                    "bounded": {"maximum_folds": 1},
                }
            }
        }
    )
    assert paths.plan_path.exists()
    assert paths.predictions_path is None
    report = json.loads(paths.report_json_path.read_text())
    assert report["status"] == "plan_only"
    assert report["training_performed"] is False
    assert report["real_artifact_audit"]["resolved_absolute_path"] == str(source.resolve())


def test_missing_source_dataset_fails_without_legacy_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="No legacy fallback is permitted"):
        write_selector_target_tournament(
            {
                "ml": {
                    "selector_target_tournament": {
                        "enabled": True,
                        "source_dataset_path": str(tmp_path / "missing.parquet"),
                        "output_dir": str(tmp_path / "out"),
                        "plan_only": True,
                    }
                }
            }
        )


def test_expected_dataset_assertions_block_training(tmp_path):
    source = tmp_path / "source.parquet"
    _write_parquet(source, _rows())
    payload = build_selector_target_tournament(
        _rows(),
        config={"ml": {}},
        source_path=source,
        settings=_settings(expected_dataset={"path": str(source), "minimum_rows": 999}),
    )
    assert payload["status"] == "blocked"
    assert "expected_dataset:minimum_rows" in payload["blockers"]
    assert payload["training_performed"] is False


def test_multiple_seeds_create_distinct_candidates_and_reconcile_fits():
    payload = build_selector_target_tournament(
        _rows(),
        config={"ml": {}},
        source_path=None,
        settings=_settings(seeds=[11, 22], model_ids=["ridge"], maximum_folds=1),
    )
    assert {row["candidate_id"] for row in payload["oos_predictions"]} == {
        "market_residual_return_10d::ridge::seed_11",
        "market_residual_return_10d::ridge::seed_22",
        "raw_return_10d::ridge::seed_11",
        "raw_return_10d::ridge::seed_22",
    }
    assert payload["fit_count_plan"]["expected_base_fits"] == 4
    assert payload["execution_reconciliation"]["completed_fits"] == 4
    assert payload["execution_reconciliation"]["all_configured_seeds_executed"] is True


def test_write_path_emits_canonical_zstd_parquet_predictions(tmp_path):
    source = tmp_path / "source.parquet"
    _write_parquet(source, _rows())
    paths = write_selector_target_tournament(
        {
            "ml": {
                "selector_target_tournament": {
                    "enabled": True,
                    "source_dataset_path": str(source),
                    "output_dir": str(tmp_path / "out"),
                    "min_train_dates": 2,
                    "test_window_dates": 2,
                    "embargo_dates": 1,
                    "bounded": {"maximum_folds": 1},
                }
            }
        }
    )
    assert paths.predictions_path is not None
    assert paths.predictions_path.suffix == ".parquet"
    prediction_rows = read_stock_level_artifact(paths.predictions_path, required_columns={"candidate_id", "strict_oos"})
    assert prediction_rows
    report = json.loads(paths.report_json_path.read_text())
    assert report["prediction_artifact_identity"]["compression"] == "zstd"
    assert report["promotion_results"]["candidate_metrics"]


def test_duplicate_source_rows_fail_closed():
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="unique by rebalance_date and symbol"):
        build_selector_target_tournament(
            rows,
            config={"ml": {}},
            source_path=None,
            settings=_settings(plan_only=True),
        )


def test_cli_mode_is_registered_feedless():
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "ml-selector-target-tournament", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ml-selector-target-tournament" in result.stdout


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_parquet(path, rows):
    write_stock_level_artifact(
        path,
        rows,
        fieldnames=list(rows[0]),
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
    )
