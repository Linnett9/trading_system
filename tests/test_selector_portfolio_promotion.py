import csv
import json
import subprocess
import sys

import pytest

from core.research.ml.stock_level.selector_portfolio_promotion import (
    build_selector_portfolio_promotion,
    discover_selector_portfolio_candidates,
    write_selector_portfolio_promotion,
)


def _rows():
    rows = []
    dates = ("2024-01-01", "2024-01-11", "2024-01-21", "2024-01-31")
    symbols = ("AAA", "BBB", "CCC")
    returns = {"AAA": 0.04, "BBB": 0.01, "CCC": -0.02}
    for fold_id, rebalance_date in enumerate(dates, start=1):
        for index, symbol in enumerate(symbols):
            rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "symbol": symbol,
                    "fold_id": str(fold_id),
                    "actual_forward_return_10d": returns[symbol],
                    "actual_benchmark_return_10d": 0.005,
                    "target_provenance_contract_version": "stock_level_target_provenance.v1",
                    "stock_level_predicted_forward_return_10d_ridge": 3 - index,
                    "stock_level_predicted_forward_return_10d_elastic_net": (
                        "" if rebalance_date == "2024-01-31" and symbol == "CCC" else 3 - index
                    ),
                    "predicted_momentum_120d": index,
                    "predicted_risk_adjusted_momentum": index,
                    "stock_level_ensemble_average_rank_score": 3 - index,
                    "market_regime": "bull" if fold_id <= 2 else "bear",
                }
            )
    return rows


def _benchmark():
    return {
        "walk_forward": {"out_of_sample_only": True},
        "completed_models": ["ridge", "elastic_net"],
        "feature_columns": ["feature_a", "feature_b"],
        "best_ml_model": {
            "name": "ridge",
            "signal_column": "stock_level_predicted_forward_return_10d_ridge",
            "mean_spearman_ic": 1.0,
        },
    }


def _promotion(**overrides):
    base = {
        "enabled": True,
        "comparison_mode": "intersection",
        "candidate_types": ["single_model", "baseline", "ensemble"],
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
            "secondary_metrics": ["net_cagr", "max_drawdown", "annualized_turnover"],
            "deterministic_tiebreak": "candidate_id",
        },
        "gates": {
            "minimum_oos_decision_dates": 4,
            "minimum_prediction_coverage": 0.95,
            "minimum_net_cagr": None,
            "minimum_net_sharpe": None,
            "maximum_drawdown": None,
            "maximum_annualized_turnover": None,
            "maximum_cost_drag": None,
            "minimum_positive_calendar_year_fraction": None,
            "require_outperformance_of_baseline": True,
        },
        "baseline_candidate_id": "baseline:momentum_120d",
        "multiple_testing": {"enabled": False, "method": "report_only"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def test_candidate_discovery_identifies_models_baselines_and_ensembles():
    candidates = discover_selector_portfolio_candidates(
        _rows(),
        benchmark=_benchmark(),
        promotion_config=_promotion(),
    )
    ids = {row["candidate_id"] for row in candidates}
    assert "single_model:ridge" in ids
    assert "baseline:momentum_120d" in ids
    assert "ensemble:average_rank_ensemble" in ids
    assert all(row["eligible_for_evaluation"] for row in candidates)


def test_candidate_discovery_marks_missing_prediction_column_ineligible():
    benchmark = _benchmark()
    benchmark["completed_models"] = ["ridge", "missing_model"]
    candidates = discover_selector_portfolio_candidates(
        _rows(),
        benchmark=benchmark,
        promotion_config=_promotion(candidate_types=["single_model"]),
    )
    missing = next(row for row in candidates if row["candidate_id"] == "single_model:missing_model")
    assert missing["eligible_for_evaluation"] is False
    assert "prediction_column_missing" in missing["ineligible_reasons"]


def test_intersection_mode_uses_identical_common_rows_and_discloses_exclusions():
    payload = build_selector_portfolio_promotion(
        _rows(),
        benchmark=_benchmark(),
        promotion_config=_promotion(candidate_types=["single_model"]),
    )
    stats = payload["common_row_statistics"]
    assert stats["common_decision_date_count"] == 4
    assert stats["common_row_count"] == 11
    assert stats["rows_excluded_per_candidate"]["single_model:ridge"] == 1
    replayed = {row["candidate_id"]: row["replayed_decision_dates"] for row in payload["candidate_metrics"]}
    assert replayed == {"single_model:elastic_net": 4, "single_model:ridge": 4}


def test_native_coverage_preserves_candidate_specific_rows():
    payload = build_selector_portfolio_promotion(
        _rows(),
        benchmark=_benchmark(),
        promotion_config=_promotion(comparison_mode="native_coverage", candidate_types=["single_model"]),
    )
    metrics = {row["candidate_id"]: row for row in payload["candidate_metrics"]}
    assert metrics["single_model:ridge"]["prediction_coverage"] == 1.0
    assert payload["common_row_statistics"]["rows_excluded_per_candidate"]["single_model:ridge"] == 0


def test_duplicate_prediction_rows_fail_closed():
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="Duplicate selector prediction row identity"):
        build_selector_portfolio_promotion(
            rows,
            benchmark=_benchmark(),
            promotion_config=_promotion(),
        )


def test_promotion_gates_fail_closed_and_do_not_recommend_ineligible_candidate():
    payload = build_selector_portfolio_promotion(
        _rows(),
        benchmark=_benchmark(),
        promotion_config=_promotion(gates={"minimum_oos_decision_dates": 250}),
    )
    assert payload["recommended_portfolio_candidate"] is None
    assert payload["promotion_status"] == "no_eligible_candidate"
    assert {row["overall_status"] for row in payload["gate_results"]} == {"insufficient_evidence"}


def test_eligible_candidates_are_ranked_deterministically_with_tiebreak():
    rows = _rows()
    for row in rows:
        row["stock_level_predicted_forward_return_10d_elastic_net"] = row[
            "stock_level_predicted_forward_return_10d_ridge"
        ]
    payload = build_selector_portfolio_promotion(
        rows,
        benchmark=_benchmark(),
        promotion_config=_promotion(
            candidate_types=["single_model"],
            gates={"require_outperformance_of_baseline": False},
        ),
    )
    assert payload["eligible_candidate_ranking"][0]["candidate_id"] == "single_model:elastic_net"
    assert payload["recommended_portfolio_candidate"] == "single_model:elastic_net"


def test_lineage_hashes_change_when_policy_or_config_changes(tmp_path):
    predictions = tmp_path / "predictions.csv"
    benchmark = tmp_path / "benchmark.json"
    _write_csv(predictions, _rows())
    benchmark.write_text(json.dumps(_benchmark()), encoding="utf-8")
    first = build_selector_portfolio_promotion(
        _rows(),
        benchmark=_benchmark(),
        promotion_config=_promotion(),
        predictions_path=predictions,
        benchmark_path=benchmark,
        config={"ml": {"selector_promotion": _promotion()}},
    )
    changed = _promotion()
    changed["fixed_policy"]["cost_bps"] = 25.0
    second = build_selector_portfolio_promotion(
        _rows(),
        benchmark=_benchmark(),
        promotion_config=changed,
        predictions_path=predictions,
        benchmark_path=benchmark,
        config={"ml": {"selector_promotion": changed}},
    )
    assert first["run_identity"]["prediction_artifact_sha256"]
    assert first["run_identity"]["policy_configuration_hash"] != second["run_identity"]["policy_configuration_hash"]
    assert first["run_identity"]["promotion_configuration_hash"] != second["run_identity"]["promotion_configuration_hash"]


def test_writer_outputs_json_markdown_and_tabular_reports(tmp_path):
    predictions = tmp_path / "predictions.csv"
    benchmark = tmp_path / "benchmark.json"
    _write_csv(predictions, _rows())
    benchmark.write_text(json.dumps(_benchmark()), encoding="utf-8")
    paths = write_selector_portfolio_promotion(
        {
            "ml": {
                "selector_promotion_predictions_path": str(predictions),
                "selector_promotion_benchmark_path": str(benchmark),
                "selector_promotion_output_dir": str(tmp_path / "promotion"),
                "selector_promotion": _promotion(),
            }
        }
    )
    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    assert paths.candidate_metrics_csv_path.exists()
    payload = json.loads(paths.json_path.read_text())
    assert payload["training_performed"] is False
    assert payload["trading_impact"] == "none"
    assert payload["best_forecast_model"]["name"] == "ridge"
    assert "recommended_portfolio_candidate" in payload


def test_cli_mode_is_registered_without_running_trading_side_effects():
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "ml-selector-portfolio-promotion", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ml-selector-portfolio-promotion" in result.stdout


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
