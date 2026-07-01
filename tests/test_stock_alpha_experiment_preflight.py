from __future__ import annotations

import csv
import json

from config.config_loader import load_config
from core.research.ml.stock_level.stock_alpha_experiment_preflight import (
    build_stock_alpha_experiment_preflight,
    write_stock_alpha_experiment_preflight,
)


def test_preflight_reports_missing_future_source_file_clearly():
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml", overlay_project_config=True)
    config["config_path"] = "config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml"

    payload = build_stock_alpha_experiment_preflight(config)

    assert payload["source_file_exists"] is False
    assert payload["source_exists"] is False
    assert payload["source_predictions_exists"] is False
    assert payload["safe_to_run"] is False
    assert payload["blocking_issues"] == ["configured source predictions file does not exist"]
    assert "configured source predictions file is missing required columns" not in payload["blocking_issues"]
    assert payload["required_columns_found"] == []
    assert payload["required_columns_missing"] == []
    assert payload["required_columns_not_checked_reason"] == "source_file_missing"
    assert payload["source_predictions_path"].endswith("stock_alpha_ensemble_average_rank_predictions.csv")
    assert payload["estimated_policy_count"] == 72
    assert payload["guardrail_failures"] == []


def test_preflight_marks_existing_portfolio_sweep_config_safe_when_source_exists(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_prediction_header(source)
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml", overlay_project_config=True)
    config["config_path"] = "config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml"
    config["ml"]["stock_alpha_report_root"] = str(tmp_path / "reports")
    config["ml"]["stock_alpha_portfolio_sweep_source_predictions_path"] = str(source)

    payload = build_stock_alpha_experiment_preflight(config)

    assert payload["source_file_exists"] is True
    assert payload["source_exists"] is True
    assert payload["source_predictions_exists"] is True
    assert payload["required_columns_missing"] == []
    assert payload["required_columns_not_checked_reason"] == ""
    assert payload["blocking_issues"] == []
    assert payload["estimated_policy_count"] == 48
    assert payload["safe_to_run"] is True


def test_preflight_writes_json_and_markdown(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_prediction_header(source)
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml", overlay_project_config=True)
    config["config_path"] = "config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml"
    config["ml"]["stock_alpha_report_root"] = str(tmp_path / "reports")
    config["ml"]["stock_alpha_portfolio_sweep_source_predictions_path"] = str(source)

    paths = write_stock_alpha_experiment_preflight(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["source_file_exists"] is True
    assert payload["safe_to_run"] is True
    assert "- Source exists: True" in markdown
    assert "- Safe to run: True" in markdown
    assert "- None" in markdown
    assert "STOCK-ALPHA" not in markdown


def test_preflight_existing_source_missing_columns_blocks_clearly(tmp_path):
    source = tmp_path / "predictions.csv"
    source.write_text("rebalance_date,symbol\n2024-01-01,AAA\n", encoding="utf-8")
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_portfolio_sweep_source_predictions_path"] = str(source)

    payload = build_stock_alpha_experiment_preflight(config)

    assert payload["source_file_exists"] is True
    assert payload["required_columns_not_checked_reason"] == ""
    assert "actual_forward_return_10d" in payload["required_columns_missing"]
    assert "configured source predictions file is missing required columns" in payload["blocking_issues"]
    assert payload["safe_to_run"] is False


def test_preflight_markdown_and_json_agree_for_missing_source(tmp_path):
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_full_enriched_coarse.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_report_root"] = str(tmp_path / "reports")

    paths = write_stock_alpha_experiment_preflight(config)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["source_file_exists"] is False
    assert "- Source exists: False" in markdown
    assert "- Safe to run: False" in markdown
    assert "configured source predictions file does not exist" in markdown


def test_preflight_blocks_unsafe_output_controls(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_prediction_header(source)
    config = load_config("config/config.stock_alpha_portfolio_sweep_ensemble_benchmark_fast_coarse.yaml", overlay_project_config=True)
    config["ml"]["stock_alpha_portfolio_sweep_source_predictions_path"] = str(source)
    config["ml"]["stock_alpha_portfolio_sweep_write_all_trades"] = True

    payload = build_stock_alpha_experiment_preflight(config)

    assert payload["safe_to_run"] is False
    assert payload["blocking_issues"] == []
    assert "all-policy trades output is enabled" in payload["warnings"]


def _write_prediction_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rebalance_date",
                "symbol",
                "actual_forward_return_10d",
                "stock_level_ensemble_average_rank_score",
                "stock_level_ensemble_trimmed_mean_rank_score",
                "predicted_momentum_120d",
                "predicted_risk_adjusted_momentum",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rebalance_date": "2024-01-01",
                "symbol": "AAA",
                "actual_forward_return_10d": "0.01",
                "stock_level_ensemble_average_rank_score": "0.6",
                "stock_level_ensemble_trimmed_mean_rank_score": "0.6",
                "predicted_momentum_120d": "0.5",
                "predicted_risk_adjusted_momentum": "0.7",
            }
        )
