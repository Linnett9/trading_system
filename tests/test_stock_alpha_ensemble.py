from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from config.config_loader import load_config
from core.research.ml.stock_level.stock_alpha_ensemble import (
    COMPONENT_COUNT_COLUMN,
    COMPONENT_COVERAGE_COLUMN,
    CONFIDENCE_SIGNAL_COLUMN,
    CONSENSUS_SIGNAL_COLUMN,
    DISAGREEMENT_NORMALIZED_COLUMN,
    DISAGREEMENT_RAW_COLUMN,
    DISAGREEMENT_SIGNAL_COLUMN,
    ENSEMBLE_SIGNAL_COLUMN,
    MEDIAN_RANK_SIGNAL_COLUMN,
    TRIMMED_MEAN_RANK_SIGNAL_COLUMN,
    build_average_rank_ensemble,
    write_stock_alpha_ensemble,
)
from core.research.ml.stock_level.stock_alpha_news_contract import validate_news_contract


def test_average_rank_normalizes_per_date():
    rows = [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "b": "10", "actual_forward_return_10d": "0.1"},
        {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "2", "b": "20", "actual_forward_return_10d": "0.2"},
        {"rebalance_date": "2024-01-02", "symbol": "AAA", "a": "100", "b": "1", "actual_forward_return_10d": "0.1"},
        {"rebalance_date": "2024-01-02", "symbol": "BBB", "a": "200", "b": "2", "actual_forward_return_10d": "0.2"},
    ]
    output, availability = build_average_rank_ensemble(
        rows,
        component_columns=["a", "b"],
        min_component_count=2,
        rank_normalization_method="percentile",
    )

    assert [row[ENSEMBLE_SIGNAL_COLUMN] for row in output] == [0.0, 1.0, 0.0, 1.0]
    assert all(row["component_columns_found"] == 2 for row in availability)


def test_average_rank_handles_missing_columns_and_min_component_count():
    rows = [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "actual_forward_return_10d": "0.1"},
        {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "2", "actual_forward_return_10d": "0.2"},
    ]
    output, _ = build_average_rank_ensemble(
        rows,
        component_columns=["a", "missing"],
        min_component_count=2,
        rank_normalization_method="percentile",
    )

    assert [row[ENSEMBLE_SIGNAL_COLUMN] for row in output] == ["", ""]
    assert [row[COMPONENT_COUNT_COLUMN] for row in output] == [1, 1]


def test_robust_rank_ensemble_columns_are_computed_per_date():
    rows = [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "b": "10", "c": "100", "d": "1000", "actual_forward_return_10d": "0.1"},
        {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "2", "b": "20", "c": "90", "d": "900", "actual_forward_return_10d": "0.2"},
        {"rebalance_date": "2024-01-01", "symbol": "CCC", "a": "3", "b": "30", "c": "80", "d": "800", "actual_forward_return_10d": "0.3"},
    ]
    output, _ = build_average_rank_ensemble(
        rows,
        component_columns=["a", "b", "c", "d"],
        min_component_count=2,
        rank_normalization_method="percentile",
        trim_extremes=1,
        trim_min_components=4,
    )

    middle = output[1]
    assert middle[COMPONENT_COUNT_COLUMN] == 4
    assert middle[COMPONENT_COVERAGE_COLUMN] == 1.0
    assert middle[ENSEMBLE_SIGNAL_COLUMN] == 0.5
    assert middle[MEDIAN_RANK_SIGNAL_COLUMN] == 0.5
    assert middle[TRIMMED_MEAN_RANK_SIGNAL_COLUMN] == 0.5
    assert middle[DISAGREEMENT_RAW_COLUMN] == 0.0
    assert middle[DISAGREEMENT_NORMALIZED_COLUMN] == 0.0
    assert middle[DISAGREEMENT_SIGNAL_COLUMN] == 0.0
    assert middle[CONSENSUS_SIGNAL_COLUMN] == 0.5
    assert middle[CONFIDENCE_SIGNAL_COLUMN] == 1.0


def test_disagreement_consensus_and_confidence_reflect_model_conflict():
    rows = [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "b": "1", "c": "3", "actual_forward_return_10d": "0.1"},
        {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "2", "b": "2", "c": "2", "actual_forward_return_10d": "0.2"},
        {"rebalance_date": "2024-01-01", "symbol": "CCC", "a": "3", "b": "3", "c": "1", "actual_forward_return_10d": "0.3"},
    ]
    output, _ = build_average_rank_ensemble(
        rows,
        component_columns=["a", "b", "c"],
        min_component_count=2,
        rank_normalization_method="percentile",
    )

    stable = output[1]
    conflicted = output[0]
    assert stable[DISAGREEMENT_RAW_COLUMN] == 0.0
    assert stable[CONFIDENCE_SIGNAL_COLUMN] == 1.0
    assert conflicted[DISAGREEMENT_RAW_COLUMN] > 0.0
    assert conflicted[DISAGREEMENT_NORMALIZED_COLUMN] == 1.0
    assert conflicted[CONFIDENCE_SIGNAL_COLUMN] == 0.0
    assert conflicted[CONSENSUS_SIGNAL_COLUMN] == 0.0


def test_trimmed_mean_can_fallback_to_missing_when_too_few_components():
    rows = [
        {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "b": "10"},
        {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "2", "b": "20"},
    ]
    output, _ = build_average_rank_ensemble(
        rows,
        component_columns=["a", "b"],
        min_component_count=2,
        rank_normalization_method="percentile",
        trim_min_components=4,
        trim_fallback="missing",
    )

    assert [row[TRIMMED_MEAN_RANK_SIGNAL_COLUMN] for row in output] == ["", ""]


def test_average_rank_rejects_target_leakage():
    with pytest.raises(ValueError, match="Target columns cannot be ensemble components"):
        build_average_rank_ensemble(
            [{"rebalance_date": "2024-01-01", "symbol": "AAA", "actual_forward_return_10d": "0.1"}],
            component_columns=["actual_forward_return_10d"],
            min_component_count=1,
            rank_normalization_method="percentile",
        )


def test_ensemble_writer_outputs_metadata_and_guardrails(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_predictions(source)
    paths = write_stock_alpha_ensemble(
        {
            "ml": {
                "stock_alpha_report_root": str(tmp_path / "ensemble"),
                "stock_alpha_run_size": "dev",
                "stock_alpha_ensemble_source_predictions_path": str(source),
                "stock_alpha_ensemble_component_signal_columns": ["a", "b", "missing"],
                "stock_alpha_ensemble_min_component_count": 2,
                "stock_alpha_ensemble_methods": [
                    "average_rank",
                    "median_rank",
                    "trimmed_mean_rank",
                    "consensus",
                    "confidence",
                ],
            }
        }
    )
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))

    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["production_validated"] is False
    assert payload["promotion_thresholds_changed"] is False
    assert payload["component_signal_columns_found"] == ["a", "b"]
    assert payload["component_signal_columns_missing"] == ["missing"]
    assert payload["ensemble_methods_enabled"] == [
        "average_rank",
        "median_rank",
        "trimmed_mean_rank",
        "consensus",
        "confidence",
    ]
    assert payload["eligible_rows_by_method"]["average_rank"] > 0
    assert payload["eligible_rows_by_method"]["confidence"] > 0
    assert paths.predictions_path.exists()
    assert paths.leaderboard_csv_path.exists()
    leaderboard = {row["name"]: row for row in payload["leaderboard"]}
    assert set(leaderboard) == {
        "average_rank_ensemble",
        "median_rank_ensemble",
        "trimmed_mean_rank_ensemble",
        "consensus_ensemble",
        "confidence_ensemble",
    }
    assert all(row["date_count"] > 0 for row in leaderboard.values())
    assert all(row["row_count"] > 0 for row in leaderboard.values())


def test_ensemble_writer_missing_source_fails_clearly(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(ValueError) as excinfo:
        write_stock_alpha_ensemble(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "ensemble"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_ensemble_source_predictions_path": str(missing),
                    "stock_alpha_ensemble_component_signal_columns": ["a", "b"],
                    "stock_alpha_ensemble_min_component_count": 2,
                }
            }
        )

    message = str(excinfo.value)
    assert "source predictions path does not exist" in message
    assert f"source_predictions_path={missing}" in message
    assert "requested_component_columns=['a', 'b']" in message
    assert "found_component_columns=[]" in message
    assert "missing_component_columns=['a', 'b']" in message
    assert "min_component_count=2" in message


def test_ensemble_writer_empty_source_fails_clearly(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_text("rebalance_date,symbol,a,b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source predictions CSV has no rows"):
        write_stock_alpha_ensemble(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "ensemble"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_ensemble_source_predictions_path": str(source),
                    "stock_alpha_ensemble_component_signal_columns": ["a", "b"],
                    "stock_alpha_ensemble_min_component_count": 2,
                }
            }
        )


def test_ensemble_writer_zero_found_components_fails_clearly(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_predictions(source)

    with pytest.raises(ValueError) as excinfo:
        write_stock_alpha_ensemble(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "ensemble"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_ensemble_source_predictions_path": str(source),
                    "stock_alpha_ensemble_component_signal_columns": ["missing_a", "missing_b"],
                    "stock_alpha_ensemble_min_component_count": 2,
                }
            }
        )

    message = str(excinfo.value)
    assert "fewer than min_component_count component columns were found globally" in message
    assert "found_component_columns=[]" in message
    assert "missing_component_columns=['missing_a', 'missing_b']" in message


def test_ensemble_writer_insufficient_components_fails_clearly(tmp_path):
    source = tmp_path / "predictions.csv"
    _write_predictions(source)

    with pytest.raises(ValueError) as excinfo:
        write_stock_alpha_ensemble(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "ensemble"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_ensemble_source_predictions_path": str(source),
                    "stock_alpha_ensemble_component_signal_columns": ["a", "missing"],
                    "stock_alpha_ensemble_min_component_count": 2,
                }
            }
        )

    message = str(excinfo.value)
    assert "fewer than min_component_count component columns were found globally" in message
    assert "found_component_columns=['a']" in message
    assert "missing_component_columns=['missing']" in message


def test_ensemble_writer_no_rows_satisfy_min_component_count_fails(tmp_path):
    source = tmp_path / "predictions.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rebalance_date", "symbol", "a", "b", "actual_forward_return_10d"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "b": "", "actual_forward_return_10d": "0.1"},
                {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "", "b": "2", "actual_forward_return_10d": "0.2"},
            ]
        )

    with pytest.raises(ValueError, match="no source rows satisfied min_component_count"):
        write_stock_alpha_ensemble(
            {
                "ml": {
                    "stock_alpha_report_root": str(tmp_path / "ensemble"),
                    "stock_alpha_run_size": "dev",
                    "stock_alpha_ensemble_source_predictions_path": str(source),
                    "stock_alpha_ensemble_component_signal_columns": ["a", "b"],
                    "stock_alpha_ensemble_min_component_count": 2,
                }
            }
        )


def test_ensemble_configs_load():
    dev = load_config("config/config.stock_alpha_ensemble_dev_diagnostic.yaml", overlay_project_config=True)
    benchmark = load_config("config/config.stock_alpha_ensemble_benchmark_fast.yaml", overlay_project_config=True)
    assert dev["ml"]["stock_alpha_ensemble_method"] == "average_rank"
    assert dev["ml"]["stock_alpha_ensemble_methods"] == [
        "average_rank",
        "median_rank",
        "trimmed_mean_rank",
        "consensus",
        "confidence",
    ]
    assert dev["ml"]["stock_alpha_ensemble_trim_extremes"] == 1
    assert dev["ml"]["stock_alpha_ensemble_trim_min_components"] == 4
    assert benchmark["ml"]["stock_alpha_run_size"] == "benchmark"
    assert "stacked_meta_model" in dev["ml"]["stock_alpha_ensemble_methods_future"]


def test_dev_diagnostic_config_points_to_fast_prediction_columns_if_present():
    config = load_config("config/config.stock_alpha_ensemble_dev_diagnostic.yaml", overlay_project_config=True)
    source = Path(config["ml"]["stock_alpha_ensemble_source_predictions_path"])
    if not source.exists():
        pytest.skip(f"diagnostic OOS predictions not present: {source}")

    with source.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    for column in [
        "stock_level_predicted_forward_return_10d_ridge",
        "stock_level_predicted_forward_return_10d_elastic_net",
        "stock_level_predicted_forward_return_10d_random_forest",
        "stock_level_predicted_forward_return_10d_gradient_boosting",
    ]:
        assert column in header


def test_news_transformer_unavailable_without_contract():
    validation = validate_news_contract(
        {"ml": {}},
        [{"rebalance_date": "2024-01-01", "symbol": "AAA"}],
    )
    assert validation.available is False
    assert "stock_alpha_news_contract_path" in validation.reason


def test_news_contract_rejects_future_articles_and_missing_fields(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text("article_id,symbol,published_at_utc\n1,AAA,2024-01-02T00:00:00Z\n", encoding="utf-8")
    missing = validate_news_contract({"ml": {"stock_news_contract_path": str(path)}}, [{"rebalance_date": "2024-01-01", "symbol": "AAA"}])
    assert missing.available is False
    assert missing.missing_fields
    _write_news_contract(path, published="2024-01-02T00:00:00Z", ingested="2024-01-02T00:00:00Z")
    future = validate_news_contract({"ml": {"stock_news_contract_path": str(path), "stock_news_min_symbol_coverage": 0.0, "stock_news_min_date_coverage": 0.0}}, [{"rebalance_date": "2024-01-01", "symbol": "AAA"}])
    assert future.available is False
    assert future.reason == "news contract contains future articles"


def test_news_contract_accepts_point_in_time_rows(tmp_path):
    path = tmp_path / "news.csv"
    _write_news_contract(path, published="2024-01-01T00:00:00Z", ingested="2024-01-01T00:00:00Z")
    validation = validate_news_contract(
        {"ml": {"stock_news_contract_path": str(path), "stock_news_min_symbol_coverage": 1.0, "stock_news_min_date_coverage": 1.0}},
        [{"rebalance_date": "2024-01-01", "symbol": "AAA"}],
    )
    assert validation.available is False
    assert validation.contract_valid is True


def _write_predictions(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rebalance_date", "symbol", "a", "b", "actual_forward_return_10d", "actual_future_volatility", "actual_future_drawdown"])
        writer.writeheader()
        writer.writerows(
            [
                {"rebalance_date": "2024-01-01", "symbol": "AAA", "a": "1", "b": "10", "actual_forward_return_10d": "0.1", "actual_future_volatility": "0.1", "actual_future_drawdown": "-0.1"},
                {"rebalance_date": "2024-01-01", "symbol": "BBB", "a": "2", "b": "20", "actual_forward_return_10d": "0.2", "actual_future_volatility": "0.1", "actual_future_drawdown": "-0.1"},
            ]
        )


def _write_news_contract(path, *, published, ingested):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "article_id",
                "symbol",
                "published_at_utc",
                "ingested_at",
                "source",
                "headline",
                "body_or_summary",
                "sentiment_score",
                "relevance_score",
                "novelty_score",
                "event_type",
                "language",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "article_id": "1",
                "symbol": "AAA",
                "published_at_utc": published,
                "ingested_at": ingested,
                "source": "vendor",
                "headline": "real headline",
                "body_or_summary": "real summary",
                "sentiment_score": "0.2",
                "relevance_score": "0.9",
                "novelty_score": "0.8",
                "event_type": "earnings",
                "language": "en",
            }
        )
