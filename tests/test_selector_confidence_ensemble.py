import json
import subprocess
import sys

import pytest

from core.research.ml.stock_level.selector_confidence_ensemble import (
    build_selector_confidence_ensemble,
    write_selector_confidence_ensemble,
)
from core.research.ml.stock_level.selector_cost_aware_policy_evaluation import (
    build_selector_cost_aware_policy_evaluation,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    read_stock_level_artifact,
    write_stock_level_artifact,
)


COMPONENTS = [
    "raw_return_10d::ridge::seed_42",
    "raw_return_10d::ridge::seed_1729",
    "raw_return_10d::elastic_net::seed_42",
    "raw_return_10d::elastic_net::seed_1729",
]


def _rows(include_benchmark=True):
    dates = ("2024-01-01", "2024-01-11")
    symbols = ("AAA", "BBB", "CCC", "DDD")
    base = {
        "raw_return_10d::ridge::seed_42": [4, 3, 2, 1],
        "raw_return_10d::ridge::seed_1729": [4.1, 2.9, 2.1, 1],
        "raw_return_10d::elastic_net::seed_42": [4, 3, 1, 2],
        "raw_return_10d::elastic_net::seed_1729": [1, 2, 3, 4],
    }
    rows = []
    for date_index, date in enumerate(dates):
        for component in COMPONENTS:
            for symbol_index, symbol in enumerate(symbols):
                value = base[component][symbol_index] + date_index * 0.01
                row = {
                    "candidate_id": component,
                    "target_id": "raw_return_10d",
                    "model_id": component.split("::")[1],
                    "seed": int(component.rsplit("_", 1)[1]),
                    "rebalance_date": date,
                    "decision_timestamp": date,
                    "symbol": symbol,
                    "prediction": value,
                    "actual_investable_return_10d": (4 - symbol_index) / 100,
                    "fold_id": f"fold_{date_index}",
                    "strict_oos": True,
                    "dataset_identity": "dataset-1",
                    "target_contract_identity": "target-1",
                    "fold_plan_identity": "folds-1",
                }
                if include_benchmark:
                    row["actual_benchmark_return_10d"] = 0.0 if date_index == 0 else 0.01
                rows.append(row)
    return rows


def _settings(**overrides):
    settings = {
        "enabled": True,
        "prediction_artifact_path": "predictions.parquet",
        "output_dir": "unused",
        "cohorts": [
            {
                "ensemble_id": "raw_return_tabular_rank_ensemble",
                "target_id": "raw_return_10d",
                "component_selection": {"candidate_ids": COMPONENTS},
                "score_normalisation": "cross_sectional_percentile",
                "aggregation_method": "mean_rank",
                "minimum_components_per_row": 3,
                "require_shared_dataset_identity": True,
                "require_shared_fold_plan_identity": True,
                "require_shared_target_contract": True,
            }
        ],
        "confidence": {"minimum_confidence": 0.30, "margin_weight": 1.0},
        "abstention": {
            "enabled": True,
            "minimum_confidence": 0.30,
            "maximum_disagreement": 0.70,
            "minimum_components": 3,
            "minimum_model_families": 2,
            "minimum_entry_margin": None,
        },
        "sizing": {"mode": "tiered", "tiers": [{"minimum_confidence": 0.80, "multiplier": 1.0}, {"minimum_confidence": 0.30, "multiplier": 0.5}]},
        "portfolio_policy": {
            "policy_id": "exact_top_n",
            "construction_mode": "exact_top_n",
            "selection": {"target_holdings": 2, "entry_rank_max": 2, "retention_rank_max": 2},
            "trading": {"minimum_trade_weight": 0.0, "rebalance_fraction": 1.0},
            "edge_filter": {"enabled": False, "mode": "rank_only", "cost_multiplier": 1.0},
            "retention": {"enabled": False},
            "liquidity": {"enabled": False},
            "costs": {"reuse_replay_cost_model": True},
        },
        "cost_bps": 10.0,
        "slippage_bps": 5.0,
        "max_position_weight": 0.5,
        "maximum_decision_dates": None,
        "maximum_symbols": None,
        "comparison_requires_benchmark": False,
    }
    settings.update(overrides)
    return settings


def test_compatible_components_build_contract_and_seed_aggregation():
    payload = build_selector_confidence_ensemble(_rows(), config={"ml": {}}, settings=_settings(), source_path=None)
    contract = payload["ensemble_contracts"][0]
    assert contract["component_models"] == ["elastic_net", "ridge"]
    assert contract["component_seeds"] == [42, 1729]
    assert payload["blockers"] == []
    first = next(row for row in payload["ensemble_predictions"] if row["symbol"] == "AAA")
    assert first["model_count"] == 2
    assert first["seed_count"] == 4
    assert 0.0 <= first["confidence"] <= 1.0


def test_mixed_fold_plan_blocks_cohort():
    rows = _rows()
    rows[0]["fold_plan_identity"] = "other-fold-plan"
    payload = build_selector_confidence_ensemble(rows, config={"ml": {}}, settings=_settings(), source_path=None)
    assert any("mixed_fold_plan_identities" in blocker for blocker in payload["blockers"])


def test_missing_components_reduce_confidence_and_can_abstain():
    rows = [row for row in _rows() if row["candidate_id"] != COMPONENTS[-1]]
    payload = build_selector_confidence_ensemble(rows, config={"ml": {}}, settings=_settings(), source_path=None)
    statuses = {row["abstention_status"] for row in payload["ensemble_predictions"]}
    assert "low_confidence" in statuses
    assert any(row["component_count"] == 3 for row in payload["ensemble_predictions"])


def test_benchmark_missing_is_not_zero_filled_in_cost_aware_adapter():
    rows = [
        {"candidate_id": "x", "rebalance_date": "2024-01-01", "symbol": "AAA", "fold_id": "1", "prediction": 1.0, "actual_investable_return_10d": 0.01},
        {"candidate_id": "x", "rebalance_date": "2024-01-01", "symbol": "BBB", "fold_id": "1", "prediction": 0.5, "actual_investable_return_10d": 0.02},
    ]
    payload = build_selector_cost_aware_policy_evaluation(
        rows,
        config={"ml": {}},
        source_path=None,
        settings={
            "enabled": True,
            "prediction_artifact_path": "",
            "candidate_id": "x",
            "prediction_column": "prediction",
            "prediction_semantics": "rank_score",
            "allow_csv_fallback": False,
            "output_dir": "",
            "top_n": 1,
            "cost_bps": 10,
            "slippage_bps": 5,
            "max_position_weight": 1.0,
            "min_position_weight": 0.0,
            "maximum_decision_dates": None,
            "maximum_symbols": None,
            "development_period": {},
            "evaluation_period": {},
            "policies": [_settings()["portfolio_policy"]],
        },
    )
    assert payload["benchmark_identity"]["benchmark_non_null_count"] == 0
    assert payload["benchmark_identity"]["benchmark_relative_metrics_available"] is False
    assert "benchmark_returns_unavailable; benchmark_relative_metrics_disabled" in payload["warnings"]


def test_writer_outputs_parquet_predictions_and_diagnostics(tmp_path):
    source = tmp_path / "predictions.parquet"
    write_stock_level_artifact(
        source,
        _rows(),
        fieldnames=list(_rows()[0]),
        config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
    )
    config = {"ml": {"selector_confidence_ensemble": {**_settings(), "prediction_artifact_path": str(source), "output_dir": str(tmp_path / "out")}}}
    paths = write_selector_confidence_ensemble(config)
    predictions = read_stock_level_artifact(paths.predictions_path, required_columns={"ensemble_id", "confidence", "abstention_status"})
    diagnostics = read_stock_level_artifact(paths.diagnostics_path, required_columns={"ensemble_id", "seed_dispersion", "model_dispersion"})
    assert predictions
    assert diagnostics
    payload = json.loads(paths.comparison_json_path.read_text())
    assert payload["benchmark_availability"]["benchmark_non_null_count"] > 0


def test_cli_mode_is_registered_feedless():
    result = subprocess.run(
        [sys.executable, "main.py", "--mode", "ml-selector-confidence-ensemble", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ml-selector-confidence-ensemble" in result.stdout
