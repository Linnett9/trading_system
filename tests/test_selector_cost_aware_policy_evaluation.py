import json
import subprocess
import sys

import pytest

from core.research.ml.stock_level.selector_cost_aware_policy_evaluation import (
    build_selector_cost_aware_policy_evaluation,
    write_selector_cost_aware_policy_evaluation,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    read_stock_level_artifact,
    write_stock_level_artifact,
)


def _rows():
    scores = {
        "2024-01-01": {"AAA": 1.00, "BBB": 0.90, "CCC": 0.80, "DDD": 0.70},
        "2024-01-11": {"BBB": 1.00, "CCC": 0.91, "AAA": 0.90, "DDD": 0.70},
        "2024-01-21": {"CCC": 1.00, "DDD": 0.99, "BBB": 0.89, "AAA": 0.60},
    }
    rows = []
    for fold, (rebalance_date, by_symbol) in enumerate(scores.items(), start=1):
        for symbol, score in by_symbol.items():
            rows.append(
                {
                    "candidate_id": "single_model:test",
                    "rebalance_date": rebalance_date,
                    "symbol": symbol,
                    "fold_id": fold,
                    "prediction": score,
                    "actual_investable_return_10d": 0.01 if symbol in {"AAA", "BBB"} else 0.005,
                    "actual_benchmark_return_10d": 0.002,
                    "target_id": "raw_return_10d",
                    "target_contract_identity": "target-1",
                    "fold_plan_identity": "folds-1",
                    "strict_oos": True,
                }
            )
    return rows


def _settings(**overrides):
    settings = {
        "enabled": True,
        "prediction_artifact_path": "predictions.parquet",
        "candidate_id": "single_model:test",
        "prediction_column": "prediction",
        "prediction_semantics": "rank_score",
        "allow_csv_fallback": False,
        "output_dir": "unused",
        "top_n": 2,
        "cost_bps": 10.0,
        "slippage_bps": 5.0,
        "max_position_weight": 0.5,
        "min_position_weight": 0.0,
        "maximum_decision_dates": None,
        "maximum_symbols": None,
        "development_period": {},
        "evaluation_period": {},
        "policies": [
            {
                "policy_id": "exact_top_n",
                "construction_mode": "exact_top_n",
                "selection": {"target_holdings": 2, "entry_rank_max": 2, "retention_rank_max": 2},
                "trading": {"minimum_trade_weight": 0.0, "rebalance_fraction": 1.0},
                "edge_filter": {"enabled": False, "mode": "rank_only", "cost_multiplier": 1.0},
                "retention": {"enabled": False},
                "liquidity": {"enabled": False},
                "costs": {"reuse_replay_cost_model": True},
            },
            {
                "policy_id": "hysteresis_min_trade",
                "construction_mode": "cost_aware",
                "selection": {"target_holdings": 2, "entry_rank_max": 2, "retention_rank_max": 3},
                "trading": {"minimum_trade_weight": 0.0, "rebalance_fraction": 1.0},
                "edge_filter": {"enabled": False, "mode": "rank_only", "cost_multiplier": 1.0},
                "retention": {"enabled": True},
                "liquidity": {"enabled": False},
                "costs": {"reuse_replay_cost_model": True},
            },
            {
                "policy_id": "edge_partial",
                "construction_mode": "cost_aware",
                "selection": {"target_holdings": 2, "entry_rank_max": 2, "retention_rank_max": 3},
                "trading": {"minimum_trade_weight": 0.01, "rebalance_fraction": 0.5},
                "edge_filter": {"enabled": True, "mode": "standardized_score", "minimum_percentile_advantage": 0.50, "cost_multiplier": 1.5},
                "retention": {"enabled": True},
                "liquidity": {"enabled": False},
                "costs": {"reuse_replay_cost_model": True},
            },
        ],
    }
    settings.update(overrides)
    return settings


def test_hysteresis_retains_minor_rank_crossing_and_reduces_turnover():
    payload = build_selector_cost_aware_policy_evaluation(_rows(), config={"ml": {}}, settings=_settings(), source_path=None)
    decisions = [
        row
        for row in payload["decisions"]
        if row["policy_id"] == "hysteresis_min_trade"
        and row["rebalance_date"] == "2024-01-11"
        and row["symbol"] == "AAA"
    ]
    assert decisions[0]["decision_reason"] == "retained_within_band"
    metrics = {row["policy_id"]: row for row in payload["policy_metrics"]}
    assert metrics["hysteresis_min_trade"]["turnover_avoided_vs_baseline"] > 0
    assert metrics["hysteresis_min_trade"]["costs_avoided_vs_baseline"] > 0


def test_edge_filter_blocks_marginal_replacement_and_partial_rebalance_changes_turnover():
    payload = build_selector_cost_aware_policy_evaluation(_rows(), config={"ml": {}}, settings=_settings(), source_path=None)
    blocked = [
        row for row in payload["decisions"]
        if row["policy_id"] == "edge_partial" and row["decision_reason"] == "blocked_insufficient_edge"
    ]
    assert blocked
    partial = [
        row for row in payload["decisions"]
        if row["policy_id"] == "edge_partial" and row["decision_reason"] == "partial_rebalance"
    ]
    assert partial
    assert all(abs(float(row["weight_change"])) <= 0.25 for row in partial)


def test_return_calibrated_mode_fails_closed_for_rank_scores():
    settings = _settings()
    settings["policies"][1]["edge_filter"] = {"enabled": True, "mode": "return_calibrated", "cost_multiplier": 1.5}
    with pytest.raises(ValueError, match="return_calibrated edge filtering requires"):
        build_selector_cost_aware_policy_evaluation(_rows(), config={"ml": {}}, settings=settings, source_path=None)


def test_minimum_trade_size_suppresses_tiny_weight_changes():
    settings = _settings(
        policies=[
            _settings()["policies"][0],
            {
                **_settings()["policies"][1],
                "policy_id": "high_min_trade",
                "trading": {"minimum_trade_weight": 0.60, "rebalance_fraction": 0.5},
            },
        ]
    )
    payload = build_selector_cost_aware_policy_evaluation(_rows(), config={"ml": {}}, settings=settings, source_path=None)
    assert any(row["decision_reason"] == "blocked_below_trade_size" for row in payload["decisions"])


def test_writer_outputs_parquet_decisions_and_lineage(tmp_path):
    source = tmp_path / "predictions.parquet"
    write_stock_level_artifact(
        source,
        _rows(),
        fieldnames=list(_rows()[0]),
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
    )
    paths = write_selector_cost_aware_policy_evaluation(
        {
            "ml": {
                "selector_cost_aware_policy_evaluation": {
                    "enabled": True,
                    "prediction_artifact_path": str(source),
                    "candidate_id": "single_model:test",
                    "prediction_semantics": "rank_score",
                    "output_dir": str(tmp_path / "out"),
                    "top_n": 2,
                    "policies": _settings()["policies"],
                }
            }
        }
    )
    decisions = read_stock_level_artifact(paths.decisions_path, required_columns={"policy_id", "decision_reason"})
    assert decisions
    payload = json.loads(paths.comparison_json_path.read_text())
    assert payload["source_prediction_artifact_identity"]["sha256"]
    assert payload["training_performed"] is False


def test_cli_mode_is_registered_feedless():
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "ml-selector-cost-aware-policy-evaluation", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ml-selector-cost-aware-policy-evaluation" in result.stdout
