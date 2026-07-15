from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from core.research.ml.stock_level.bounded_selector_runner import BoundedSelectorSettings, PredictionQualityError, _prediction_quality, _run_identity
from core.research.ml.stock_level.selector_feature_schema import load_feature_schema
from core.research.ml.stock_level.tree_selector_diagnostics import (
    boosting_routing, diagnostic_manifest_payload, feature_variation, filter_legal_training_window,
    forest_routing, temporal_legality_report, window_identity,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT_SCHEMA = ROOT / "config/selector_features/canonical_v2_daily_tabular_v1.json"
TREE_SCHEMA = ROOT / "config/selector_features/canonical_v2_daily_tree_cross_sectional_v1.json"
EXCLUDED = (
    "market_momentum_20d", "market_momentum_60d", "market_momentum_120d",
    "market_volatility_20d", "market_drawdown_60d", "market_distance_from_200d_average",
    "market_trend_state", "market_volatility_percentile",
)


def test_stock_specific_and_date_level_feature_reporting():
    assert feature_variation(np.array([1.0, 2.0, 3.0]))["classification"] == "stock_specific"
    common = feature_variation(np.array([2.0, 2.0, 2.0]))
    assert common["classification"] == "common_to_every_stock"
    assert common["constant"] is True
    assert feature_variation(np.array([1.0, np.nan, 2.0]), missing_count=1)["classification"] == "partly_stock_specific"


def test_feature_order_preserved_in_forest_routing():
    from sklearn.ensemble import RandomForestRegressor
    x = np.array([[0.0, 9.0], [0.0, 8.0], [1.0, 2.0], [1.0, 1.0]])
    model = RandomForestRegressor(n_estimators=1, max_depth=2, random_state=3).fit(x, [0.0, 0.0, 1.0, 1.0])
    report = forest_routing(model, x, ("first", "second"))
    assert all(split["feature_name"] == ("first", "second")[split["feature_index"]] for split in report[0]["splits"])
    assert report[0]["distinct_oos_leaf_ids"] >= 2


def test_boosting_stage_diversity_reporting():
    from sklearn.ensemble import GradientBoostingRegressor
    x = np.arange(20, dtype=float).reshape(-1, 1); y = (x[:, 0] > 9).astype(float)
    model = GradientBoostingRegressor(n_estimators=3, max_depth=1, random_state=2).fit(x, y)
    report = boosting_routing(model, x, ("signal",))
    assert len(report) == 3
    assert all(stage["distinct_stage_output_count"] >= 2 for stage in report)
    assert report[-1]["cumulative_prediction_quality"]["distinct_rank_count"] >= 2


def test_session_window_preserves_decision_and_label_legality():
    table = pa.Table.from_pylist([
        {"decision_session_date": "2024-01-01", "decision_timestamp": "2024-01-01 20:00", "label_available_timestamp": "2024-01-03 20:00"},
        {"decision_session_date": "2024-01-02", "decision_timestamp": "2024-01-02 20:00", "label_available_timestamp": "2024-01-10 20:00"},
        {"decision_session_date": "2024-01-03", "decision_timestamp": "2024-01-03 20:00", "label_available_timestamp": "2024-01-04 20:00"},
    ])
    result = filter_legal_training_window(table, decision_timestamp="2024-01-05 20:00", start_date="2024-01-02")
    assert result["decision_session_date"].to_pylist() == ["2024-01-03"]
    legality = temporal_legality_report(decision_timestamp="2024-01-05 20:00", training_decision_max="2024-01-03 20:00", training_label_available_max="2024-01-04 20:00")
    assert legality["decision_timestamp_guard_passed"] is True
    assert legality["label_availability_guard_passed"] is True


def test_window_identity_invalidates_incompatible_resume_identity():
    five_year = window_identity(requested_start_date="2021-06-25", trailing_sessions=None, resolved_start_date="2021-06-25", decision_date="2026-06-25")
    sessions = window_identity(requested_start_date=None, trailing_sessions=1260, resolved_start_date="2021-06-24", decision_date="2026-06-25")
    assert five_year != sessions
    assert five_year["contract_version"] == sessions["contract_version"]


def test_degenerate_predictions_remain_rejected():
    try:
        _prediction_quality([0.1, 0.1, 0.1], 3, require_dispersion=True)
    except PredictionQualityError as exc:
        assert exc.metrics["distinct_rank_count"] == 1
    else:
        raise AssertionError("degenerate prediction unexpectedly accepted")


def test_tree_schema_is_parent_minus_exactly_eight_in_parent_order():
    parent = load_feature_schema(PARENT_SCHEMA); child = load_feature_schema(TREE_SCHEMA)
    parent_names = [row["name"] for row in parent["features"]]; child_names = [row["name"] for row in child["features"]]
    assert child_names == [name for name in parent_names if name not in EXCLUDED]
    assert len(parent_names) == 29 and len(child_names) == 21
    assert tuple(child["excluded_fields"]) == EXCLUDED
    assert tuple(row["name"] for row in child["exclusions"]) == EXCLUDED
    assert all(row["reason"] for row in child["exclusions"])
    assert child["parent_schema"]["schema_hash"] == parent["schema_hash"]
    assert child["schema_hash"] != parent["schema_hash"]
    assert child["intended_model_family"] == ["random_forest", "gradient_boosting"]


def test_child_schema_hash_changes_runner_resume_identity(tmp_path: Path):
    dataset = tmp_path / "dataset"; dataset.mkdir(); (dataset / "feature_schema.json").write_text("{}", encoding="utf-8")
    settings = BoundedSelectorSettings.from_config({"ml": {"stock_selector_bounded": {"dataset_root": str(dataset), "output_root": str(tmp_path / "out"), "oos_end_date": "2026-06-25"}}})
    parent = load_feature_schema(PARENT_SCHEMA); child = load_feature_schema(TREE_SCHEMA)
    manifest = {"dataset_id": "test", "source_sha256": "source", "checksums": {"rows.parquet": "rows", "baseline_scores.parquet": "baseline"}}
    parent_selected = {"path": str(PARENT_SCHEMA), "contract_version": parent["contract_version"], "schema_hash": parent["schema_hash"]}
    child_selected = {"path": str(TREE_SCHEMA), "contract_version": child["contract_version"], "schema_hash": child["schema_hash"]}
    parent_identity = _run_identity(settings, manifest, tuple(row["name"] for row in parent["features"]), parent_selected)
    child_identity = _run_identity(settings, manifest, tuple(row["name"] for row in child["features"]), child_selected)
    assert parent_identity["config_hash"] != child_identity["config_hash"]
    assert child_identity["selected_feature_schema"]["schema_hash"] == child["schema_hash"]


def test_diagnostic_manifest_cannot_masquerade_as_completed_selector_result():
    payload = {"contract_version": "diag", "selector_completion_status": "not_published_diagnostic_only", "decision_date": "2026-06-25", "model_id": "random_forest", "selected_feature_schema": {}, "training_window": {}, "training_row_count": 10, "training_date_min": "2024-01-01", "training_date_max": "2024-01-02", "temporal_legality": {"decision_timestamp_guard_passed": True, "label_availability_guard_passed": True}, "prediction_quality_accepted": False, "prediction_quality": {}, "elapsed_seconds": 1.0}
    manifest = diagnostic_manifest_payload(payload)
    assert manifest["artifact_kind"] == "research_diagnostic"
    assert manifest["eligible_as_completed_selector_partition"] is False
    assert "completion_status" not in manifest
