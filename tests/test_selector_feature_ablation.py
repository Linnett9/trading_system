import json

import pytest

from application.cli_runtime import FEEDLESS_MODES
from core.research.ml.stock_level.selector_feature_ablation import (
    build_selector_feature_ablation,
    build_feature_family_contracts,
    feature_set_equivalence,
    resolve_feature_set_contracts,
    write_selector_feature_ablation,
)
from core.research.ml.stock_level.stock_level_alpha_features_audit import _audit
from core.research.ml.stock_level.stock_level_artifact_io import (
    read_stock_level_artifact,
    write_stock_level_artifact,
)


def _rows(date_count=8):
    rows = []
    dates = [f"2024-01-{day:02d}" for day in range(1, date_count + 1)]
    for date_index, rebalance_date in enumerate(dates):
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC")):
            raw = (2 - symbol_index) * 0.01 + date_index * 0.001
            rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "symbol": symbol,
                    "benchmark_symbol": "SPY",
                    "predicted_momentum_20d": 0.1 + symbol_index + date_index * 0.01,
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
                    "actual_benchmark_return_10d": 0.002,
                    "actual_market_residual_return_10d": raw - 0.002,
                    "actual_vol_adjusted_forward_return_10d": raw / 0.05,
                    "actual_drawdown_adjusted_forward_return_10d": raw,
                    "actual_rank_normalized_forward_return_10d": (2 - symbol_index) / 2,
                    "actual_top_decile_label_10d": int(symbol_index == 0),
                    "decision_session_date": rebalance_date,
                    "target_provenance_contract_version": "stock_level_target_provenance_v2",
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


def _enriched_rows(date_count=8):
    rows = _rows(date_count)
    for index, row in enumerate(rows):
        row.update(
            {
                "momentum_250d": 0.2 + index * 0.001,
                "momentum_acceleration": 0.01,
                "momentum_persistence": 0.7,
                "momentum_consistency": 0.8,
                "relative_momentum_vs_spy": 0.05,
                "relative_momentum_vs_sector": 0.01,
                "momentum_percentile": (index % 3) / 2,
                "distance_from_52_week_high": -0.05,
                "drawdown_recovery_days": 12.0,
                "rolling_max_drawdown_120d": -0.1,
                "ulcer_index": 0.03,
                "downside_deviation": 0.02,
                "volatility_percentile": 0.6,
                "volatility_trend": 0.1,
                "volatility_regime": 1.0,
                "ATR_percentile": 0.55,
                "sector_relative_strength": (2 - (index % 3)) / 2,
                "industry_relative_strength": "",
                "relative_momentum_vs_market_like": 999.0,
            }
        )
    return rows


def _settings(**overrides):
    settings = {
        "enabled": True,
        "source_dataset_path": "unused.parquet",
        "output_dir": "unused",
        "allow_csv_fallback": False,
        "include_engineered_features": False,
        "strict_roles": True,
        "comparison_mode": "strict_feature_intersection",
        "feature_sets": [
            {"feature_set_id": "momentum_only", "include_families": ["momentum_core"]},
            {"feature_set_id": "price_core", "include_families": ["momentum_core", "volatility", "drawdown", "liquidity"]},
        ],
        "feature_set_ids": ["momentum_only", "price_core"],
        "target_id": "raw_return_10d",
        "target_column": "actual_forward_return_10d",
        "model_ids": ["ridge", "elastic_net"],
        "seeds": [42],
        "plan_only": False,
        "min_train_dates": 3,
        "test_window_dates": 2,
        "embargo_dates": 1,
        "maximum_decision_dates": None,
        "maximum_symbols": None,
        "maximum_folds": 1,
        "minimum_symbols_per_date": 2,
        "sklearn_n_jobs": 1,
        "portfolio_top_n": 1,
        "cost_bps": 10.0,
        "slippage_bps": 5.0,
        "max_position_weight": 1.0,
        "min_position_weight": 0.0,
    }
    settings.update(overrides)
    return settings


def test_feature_set_identity_changes_with_membership():
    rows = _rows()
    roles = {name: "feature" for name in rows[0] if name.startswith("predicted_")}
    families = build_feature_family_contracts(rows, tuple(roles))
    one = resolve_feature_set_contracts(
        {"feature_sets": [{"feature_set_id": "x", "include_families": ["momentum_core"]}]},
        families,
        roles,
    )[0]
    two = resolve_feature_set_contracts(
        {"feature_sets": [{"feature_set_id": "x", "include_families": ["momentum_core", "volatility"]}]},
        families,
        roles,
    )[0]
    assert one["feature_set_hash"] != two["feature_set_hash"]


def test_target_columns_fail_closed_as_features():
    rows = _rows()
    settings = _settings(
        feature_sets=[{"feature_set_id": "bad", "include_columns": ["actual_forward_return_10d"]}],
        feature_set_ids=["bad"],
    )
    with pytest.raises(ValueError, match="not role=feature"):
        build_selector_feature_ablation(rows, config={"ml": {}}, settings=settings, source_path=None)


def test_plan_only_counts_fits_and_writes_no_predictions():
    payload = build_selector_feature_ablation(
        _rows(),
        config={"ml": {}},
        settings=_settings(plan_only=True),
        source_path=None,
    )
    assert payload["plan"]["expected_fits"] == 4
    assert payload["plan"]["fold_count"] == 1
    assert payload["oos_predictions"] == []
    assert payload["training_performed"] is False


def test_shared_fold_predictions_include_feature_set_identity():
    payload = build_selector_feature_ablation(
        _rows(),
        config={"ml": {}},
        settings=_settings(model_ids=["ridge"], feature_set_ids=["momentum_only", "price_core"]),
        source_path=None,
    )
    predictions = payload["oos_predictions"]
    assert {row["feature_set_id"] for row in predictions} == {"momentum_only", "price_core"}
    assert {row["fold_plan_identity"] for row in predictions} == {payload["shared_fold_plan"]["identity"]}
    assert all("::price_core::" in row["candidate_id"] or "::momentum_only::" in row["candidate_id"] for row in predictions)
    by_set = {}
    for row in predictions:
        by_set.setdefault(row["feature_set_id"], set()).add((row["fold_id"], row["rebalance_date"], row["symbol"]))
    assert by_set["momentum_only"] == by_set["price_core"]


def test_missingness_is_reported_and_not_zero_filled():
    rows = _rows()
    rows[0]["predicted_volatility_20d"] = None
    payload = build_selector_feature_ablation(
        rows,
        config={"ml": {}},
        settings=_settings(plan_only=True),
        source_path=None,
    )
    availability = [
        row for row in payload["availability"]
        if row["feature_set_id"] == "price_core" and row["feature"] == "predicted_volatility_20d"
    ]
    assert any(row["missing_count"] == 1 for row in availability)
    assert payload["matched_population"]["rows_excluded_per_feature_set"]["price_core"] == 1


def test_write_outputs_parquet_predictions(tmp_path):
    artifact = tmp_path / "source.parquet"
    write_stock_level_artifact(
        artifact,
        _rows(),
        fieldnames=list(_rows()[0]),
        config={"ml": {"stock_level_artifact_format": "parquet"}},
    )
    paths = write_selector_feature_ablation(
        {
            "ml": {
                "selector_feature_ablation": {
                    **_settings(source_dataset_path=str(artifact), output_dir=str(tmp_path / "out"), model_ids=["ridge"]),
                    "enabled": True,
                }
            }
        }
    )
    assert paths.predictions_path.exists()
    predictions = read_stock_level_artifact(paths.predictions_path, required_columns={"candidate_id", "feature_set_id", "preprocessing_identity"})
    assert predictions
    report = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
    assert report["diagnostic_status"] == "BOUNDED DIAGNOSTIC ONLY / NOT FEATURE PROMOTION EVIDENCE"
    assert paths.family_resolution_path.exists()
    assert paths.feature_set_equivalence_path.exists()
    assert paths.enrichment_contract_path.exists()


def test_known_enriched_columns_map_and_near_matches_do_not_enter():
    rows = _enriched_rows()
    available = (
        "predicted_momentum_20d",
        "predicted_momentum_60d",
        "predicted_momentum_120d",
        "predicted_risk_adjusted_momentum",
        "relative_momentum_vs_spy",
        "relative_momentum_vs_market_like",
        "sector_relative_strength",
        "momentum_percentile",
    )
    families = build_feature_family_contracts(rows, available, settings={"include_artifact_enriched_features": True})
    market = next(family for family in families if family["family_id"] == "market_relative")
    sector = next(family for family in families if family["family_id"] == "sector_relative")
    rank = next(family for family in families if family["family_id"] == "cross_sectional_rank")

    assert market["resolved_ordered_columns"] == ["relative_momentum_vs_spy"]
    assert "relative_momentum_vs_market_like" not in market["resolved_ordered_columns"]
    assert sector["resolved_ordered_columns"] == ["sector_relative_strength"]
    assert rank["resolved_ordered_columns"] == ["momentum_percentile"]


def test_required_empty_family_fails_and_optional_warns():
    rows = _rows()
    settings = _settings(
        include_artifact_enriched_features=True,
        empty_family_policy="fail",
        feature_sets=[{"feature_set_id": "bad", "include_families": ["market_relative"]}],
        feature_set_ids=["bad"],
    )
    with pytest.raises(ValueError, match="empty mandatory families"):
        build_selector_feature_ablation(rows, config={"ml": {}}, settings=settings, source_path=None)

    payload = build_selector_feature_ablation(
        rows,
        config={"ml": {}},
        settings=_settings(
            plan_only=True,
            include_artifact_enriched_features=True,
            empty_family_policy="fail",
            feature_sets=[{"feature_set_id": "ok", "include_families": ["momentum_core", "market_relative"], "optional_families": ["market_relative"]}],
            feature_set_ids=["ok"],
        ),
        source_path=None,
    )
    contract = payload["feature_set_contracts"][0]
    assert contract["empty_requested_families"] == []
    resolution = next(row for row in payload["family_resolution"] if row["family_id"] == "market_relative")
    assert resolution["resolution_reason"] == "NOT PRESENT IN SOURCE ARTIFACT"


def test_feature_set_equivalence_and_strict_block():
    feature_sets = [
        {"feature_set_id": "a", "ordered_feature_columns": ["x", "y"]},
        {"feature_set_id": "b", "ordered_feature_columns": ["x", "y"]},
        {"feature_set_id": "c", "ordered_feature_columns": ["x", "y", "z"]},
    ]
    relationships = feature_set_equivalence(feature_sets)
    assert relationships[0]["relationship"] == "identical_ordered_columns"
    assert relationships[1]["relationship"] == "strict_subset"

    with pytest.raises(ValueError, match="identical ordered columns"):
        build_selector_feature_ablation(
            _rows(),
            config={"ml": {}},
            settings=_settings(
                plan_only=True,
                fail_on_identical_feature_sets=True,
                feature_sets=[
                    {"feature_set_id": "a", "include_families": ["momentum_core"]},
                    {"feature_set_id": "b", "include_families": ["momentum_core"]},
                ],
                feature_set_ids=["a", "b"],
            ),
            source_path=None,
        )


def test_artifact_resident_enriched_flag_controls_resolution():
    rows = _enriched_rows()
    disabled = build_selector_feature_ablation(
        rows,
        config={"ml": {}},
        settings=_settings(
            plan_only=True,
            include_artifact_enriched_features=False,
            empty_family_policy="allow",
            feature_sets=[{"feature_set_id": "market", "include_families": ["momentum_core", "market_relative"]}],
            feature_set_ids=["market"],
        ),
        source_path=None,
    )
    enabled = build_selector_feature_ablation(
        rows,
        config={"ml": {}},
        settings=_settings(
            plan_only=True,
            include_artifact_enriched_features=True,
            empty_family_policy="fail",
            feature_sets=[{"feature_set_id": "market", "include_families": ["momentum_core", "market_relative"]}],
            feature_set_ids=["market"],
        ),
        source_path=None,
    )
    disabled_columns = disabled["feature_set_contracts"][0]["ordered_feature_columns"]
    enabled_columns = enabled["feature_set_contracts"][0]["ordered_feature_columns"]
    assert "relative_momentum_vs_spy" not in disabled_columns
    assert "relative_momentum_vs_spy" in enabled_columns


def test_enrichment_contract_status_is_explicit():
    audit = _audit(_rows(), _enriched_rows(), {}, "source.parquet", 1)
    assert audit["enrichment_status"] == "partially_enriched"
    assert audit["resolved_enriched_column_count"] > 0
    empty = _audit(_rows(), _rows(), {}, "source.parquet", 1)
    assert empty["enrichment_status"] == "no_additional_features"


def test_cli_mode_is_feedless():
    assert "ml-selector-feature-ablation" in FEEDLESS_MODES
