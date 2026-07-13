from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from core.research.framework.data import CsvRowRepository
from core.research.framework.ranking import CrossSectionalRankingEvaluator, finite_number
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.artifacts.artifact_writers import MLCoreArtifactWriter
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_level_alpha_features import ENGINEERED_FEATURE_COLUMNS
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    read_stock_level_artifact,
    write_stock_level_artifact,
)
from core.research.ml.stock_level.stock_level_portfolio_replay import _metrics, _replay
from core.research.ml.stock_level_benchmark_data import (
    _available_feature_columns,
    _build_oos_prediction_rows,
    _number,
    _prepare_rows,
    _validate_split_settings,
    _validate_unique_keys,
)
from core.research.ml.stock_level_benchmark_models import _model_factories
from core.research.ml.stock_level_benchmark_types import (
    AUXILIARY_TARGET_COLUMNS,
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_OUTPUT_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
)


SCHEMA_VERSION = "selector_feature_ablation_v1"
FEATURE_CONTRACT_VERSION = "selector_feature_contract_v1"
DIAGNOSTIC_STATUS = "BOUNDED DIAGNOSTIC ONLY / NOT FEATURE PROMOTION EVIDENCE"
TARGET_IDS = {"raw_return_10d": TARGET_COLUMN}
DEFENSIVE_OUTCOME_PREFIXES = ("actual_", "future_", "forward_", "target_", "label_")
DEFENSIVE_OUTCOME_SUBSTRINGS = ("return_10d", "return_5d", "drawdown", "volatility")
ENGINEERED_FEATURE_FAMILY_COLUMNS = {
    "momentum_extended": (
        "momentum_250d",
        "momentum_acceleration",
        "momentum_persistence",
        "momentum_consistency",
    ),
    "market_relative": ("relative_momentum_vs_spy",),
    "sector_relative": ("relative_momentum_vs_sector", "sector_relative_strength"),
    "industry_relative": (
        "relative_momentum_vs_industry",
        "industry_relative_strength",
        "industry_momentum_percentile",
    ),
    "cross_sectional_rank": ("momentum_percentile",),
    "market_context": (
        "market_momentum_20d",
        "market_momentum_60d",
        "market_momentum_120d",
        "market_volatility_20d",
        "market_drawdown_60d",
        "market_distance_from_200d_average",
        "market_trend_state",
        "market_volatility_percentile",
        "breadth_above_sma_200",
        "spy_realized_volatility_21d",
        "spy_realized_volatility_63d",
        "spy_max_drawdown_63d",
        "spy_max_drawdown_126d",
    ),
    "breadth": (
        "breadth_positive_momentum_20d",
        "breadth_positive_momentum_60d",
        "breadth_above_long_term_trend",
        "breadth_cross_sectional_median_return",
        "breadth_return_dispersion",
        "breadth_advance_decline_ratio",
        "breadth_coverage",
    ),
    "drawdown": (
        "predicted_drawdown_60d",
        "distance_from_52_week_high",
        "drawdown_recovery_days",
        "rolling_max_drawdown_120d",
        "ulcer_index",
    ),
    "volatility": (
        "predicted_volatility_20d",
        "downside_deviation",
        "volatility_percentile",
        "volatility_trend",
        "volatility_regime",
        "ATR_percentile",
    ),
    "regime_context": ("volatility_regime", "market_trend_state", "market_volatility_percentile"),
    "fundamental_growth": (
        "revenue_growth_yoy",
        "revenue_growth_qoq",
        "gross_profit_growth_yoy",
        "operating_income_growth_yoy",
        "net_income_growth_yoy",
        "eps_growth_yoy",
        "operating_cash_flow_growth_yoy",
        "asset_growth_yoy",
        "equity_growth_yoy",
        "growth_acceleration",
        "positive_growth_breadth",
    ),
    "fundamental_profitability": (
        "gross_margin",
        "operating_margin",
        "net_margin",
        "return_on_assets",
        "return_on_equity",
        "asset_turnover",
        "operating_cash_flow_to_assets",
        "free_cash_flow_margin",
        "cash_conversion",
    ),
    "fundamental_quality": (
        "total_accruals_to_assets",
        "cash_flow_to_net_income",
        "working_capital_accruals",
        "earnings_quality_score",
        "fundamental_coverage_count",
        "fundamental_missing_fraction",
        "restatement_indicator",
        "entity_mapping_quality",
    ),
    "fundamental_balance_sheet": (
        "debt_to_assets",
        "debt_to_equity",
        "net_debt_to_assets",
        "current_ratio",
        "cash_to_assets",
        "interest_coverage",
        "working_capital_to_assets",
    ),
    "fundamental_shareholder_actions": (
        "share_count_growth_yoy",
        "dilution_rate",
        "net_share_issuance",
        "repurchase_intensity",
        "dividend_payout",
    ),
    "fundamental_valuation": (
        "earnings_yield",
        "book_to_market",
        "sales_to_price",
        "free_cash_flow_yield",
    ),
    "fundamental_freshness": (
        "filing_recency_score",
        "fundamental_coverage_count",
        "fundamental_missing_fraction",
        "restatement_indicator",
        "entity_mapping_quality",
    ),
}


@dataclass(frozen=True)
class SelectorFeatureAblationPaths:
    output_dir: Path
    inventory_path: Path
    family_contracts_path: Path
    feature_set_contracts_path: Path
    availability_path: Path
    leakage_audit_path: Path
    plan_path: Path
    predictions_path: Path
    forecast_metrics_path: Path
    portfolio_metrics_path: Path
    pairwise_path: Path
    stability_path: Path
    redundancy_path: Path
    family_resolution_path: Path
    feature_set_equivalence_path: Path
    enrichment_contract_path: Path
    market_context_contract_path: Path
    breadth_contract_path: Path
    industry_mapping_audit_path: Path
    enriched_feature_coverage_path: Path
    pit_audit_path: Path
    enriched_validation_json_path: Path
    enriched_validation_markdown_path: Path
    report_json_path: Path
    report_markdown_path: Path


def write_selector_feature_ablation(config: Mapping[str, Any]) -> SelectorFeatureAblationPaths:
    settings = _settings(config)
    if not settings["enabled"]:
        raise ValueError("ml.selector_feature_ablation.enabled is false")
    source_path = Path(settings["source_dataset_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Selector feature-ablation source artifact does not exist: {source_path}")
    rows = read_stock_level_artifact(
        source_path,
        required_columns={"rebalance_date", "symbol", TARGET_COLUMN},
        allow_csv_fallback=bool(settings["allow_csv_fallback"]),
    )
    payload = build_selector_feature_ablation(
        rows,
        config=config,
        settings=settings,
        source_path=source_path,
    )
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = SelectorFeatureAblationPaths(
        output_dir=output_dir,
        inventory_path=output_dir / "selector_feature_inventory.csv",
        family_contracts_path=output_dir / "selector_feature_family_contracts.json",
        feature_set_contracts_path=output_dir / "selector_feature_set_contracts.json",
        availability_path=output_dir / "selector_feature_availability.csv",
        leakage_audit_path=output_dir / "selector_feature_leakage_audit.json",
        plan_path=output_dir / "selector_feature_ablation_plan.json",
        predictions_path=output_dir / "selector_feature_ablation_oos_predictions.parquet",
        forecast_metrics_path=output_dir / "selector_feature_ablation_forecast_metrics.csv",
        portfolio_metrics_path=output_dir / "selector_feature_ablation_portfolio_metrics.csv",
        pairwise_path=output_dir / "selector_feature_ablation_pairwise_comparisons.csv",
        stability_path=output_dir / "selector_feature_ablation_stability.csv",
        redundancy_path=output_dir / "selector_feature_redundancy_report.csv",
        family_resolution_path=output_dir / "selector_feature_family_resolution.csv",
        feature_set_equivalence_path=output_dir / "selector_feature_set_equivalence.csv",
        enrichment_contract_path=output_dir / "selector_enrichment_contract.json",
        market_context_contract_path=output_dir / "selector_market_context_contract.json",
        breadth_contract_path=output_dir / "selector_breadth_contract.json",
        industry_mapping_audit_path=output_dir / "selector_industry_mapping_audit.json",
        enriched_feature_coverage_path=output_dir / "selector_enriched_feature_coverage.csv",
        pit_audit_path=output_dir / "selector_enriched_feature_pit_audit.json",
        enriched_validation_json_path=output_dir / "selector_enriched_feature_validation_report.json",
        enriched_validation_markdown_path=output_dir / "selector_enriched_feature_validation_report.md",
        report_json_path=output_dir / "selector_feature_ablation_report.json",
        report_markdown_path=output_dir / "selector_feature_ablation_report.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_csv(paths.inventory_path, payload["feature_inventory"], fieldnames=_fields(payload["feature_inventory"], ["column_name"]))
    writer.write_json(paths.family_contracts_path, payload["feature_family_contracts"])
    writer.write_json(paths.feature_set_contracts_path, payload["feature_set_contracts"])
    writer.write_csv(paths.availability_path, payload["availability"], fieldnames=_fields(payload["availability"], ["feature_set_id", "feature"]))
    writer.write_json(paths.leakage_audit_path, payload["leakage_audit"])
    writer.write_json(paths.plan_path, payload["plan"])
    if not payload["plan_only"]:
        write_stock_level_artifact(
            paths.predictions_path,
            payload["oos_predictions"],
            fieldnames=_prediction_fields(payload["oos_predictions"]),
            config={"ml": {"stock_level_artifact_format": "parquet", "stock_level_parquet_compression": "zstd"}},
        )
    writer.write_csv(paths.forecast_metrics_path, payload["forecast_metrics"], fieldnames=_fields(payload["forecast_metrics"], ["candidate_id"]))
    writer.write_csv(paths.portfolio_metrics_path, payload["portfolio_metrics"], fieldnames=_fields(payload["portfolio_metrics"], ["candidate_id"]))
    writer.write_csv(paths.pairwise_path, payload["pairwise_comparisons"], fieldnames=_fields(payload["pairwise_comparisons"], ["comparison_id"]))
    writer.write_csv(paths.stability_path, payload["stability"], fieldnames=_fields(payload["stability"], ["feature_set_id", "segment_id"]))
    writer.write_csv(paths.redundancy_path, payload["redundancy"], fieldnames=_fields(payload["redundancy"], ["diagnostic_id"]))
    writer.write_csv(paths.family_resolution_path, payload["family_resolution"], fieldnames=_fields(payload["family_resolution"], ["family_id"]))
    writer.write_csv(paths.feature_set_equivalence_path, payload["feature_set_equivalence"], fieldnames=_fields(payload["feature_set_equivalence"], ["left_feature_set_id", "right_feature_set_id"]))
    writer.write_json(paths.enrichment_contract_path, payload["enrichment_contract"])
    writer.write_json(paths.market_context_contract_path, payload["market_context_contract"])
    writer.write_json(paths.breadth_contract_path, payload["breadth_contract"])
    writer.write_json(paths.industry_mapping_audit_path, payload["industry_mapping_audit"])
    writer.write_csv(paths.enriched_feature_coverage_path, payload["enriched_feature_coverage"], fieldnames=_fields(payload["enriched_feature_coverage"], ["family_id", "feature"]))
    writer.write_json(paths.pit_audit_path, payload["pit_audit"])
    writer.write_json(paths.enriched_validation_json_path, _enriched_validation_payload(payload))
    writer.write_markdown(paths.enriched_validation_markdown_path, _enriched_validation_markdown(payload))
    writer.write_json(paths.report_json_path, payload)
    writer.write_markdown(paths.report_markdown_path, _markdown(payload, paths))
    return paths


def build_selector_feature_ablation(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    source_path: Path | None,
) -> dict[str, Any]:
    _validate_split_settings(settings["min_train_dates"], settings["test_window_dates"], settings["embargo_dates"])
    _validate_unique_keys([{"rebalance_date": str(r.get("rebalance_date")), "symbol": str(r.get("symbol"))} for r in rows if r.get("rebalance_date") and r.get("symbol")])
    bounded_rows = _bounded_rows(rows, settings)
    if not bounded_rows:
        raise ValueError("No rows are available for selector feature ablation")
    available_features = _available_feature_columns(
        bounded_rows,
        include_engineered=bool(settings.get("include_artifact_enriched_features", settings.get("include_engineered_features", True))),
    )
    available_features = _with_artifact_resident_features(bounded_rows, available_features)
    roles = build_feature_roles(bounded_rows, available_features)
    inventory = feature_inventory(bounded_rows, roles, available_features)
    families = build_feature_family_contracts(bounded_rows, available_features, settings=settings)
    leakage_audit = audit_leakage(roles, available_features)
    feature_sets = resolve_feature_set_contracts(settings, families, roles)
    equivalence = feature_set_equivalence(feature_sets)
    if settings.get("fail_on_identical_feature_sets", False) and any(row["relationship"] == "identical_ordered_columns" for row in equivalence):
        raise ValueError("Configured selector feature sets resolved identical ordered columns")
    availability = availability_report(bounded_rows, feature_sets)
    pit_audit = pit_audit_report(bounded_rows, families)
    enrichment_contract = enrichment_contract_report(bounded_rows, families, source_path)
    family_coverage = enriched_feature_coverage(bounded_rows, families)
    matched_rows, matched = matched_population(bounded_rows, feature_sets, settings)
    if matched["common_row_count"] == 0:
        raise ValueError("Matched feature population is empty")
    prepared, excluded = _prepare_rows(matched_rows, tuple(sorted(matched["union_feature_columns"])))
    if excluded:
        matched["target_or_provenance_excluded_rows"] = excluded
    dates = sorted({row["rebalance_date"] for row in prepared})
    first_test_index = min(len(dates), int(settings["min_train_dates"]))
    folds, base_predictions = _build_oos_prediction_rows(
        prepared,
        dates,
        first_test_index=first_test_index,
        test_window_dates=int(settings["test_window_dates"]),
        embargo_dates=int(settings["embargo_dates"]),
    )
    maximum_folds = settings.get("maximum_folds")
    if maximum_folds:
        folds = folds[: int(maximum_folds)]
        allowed_fold_ids = {fold["fold_id"] for fold in folds}
        base_predictions = [row for row in base_predictions if row["fold_id"] in allowed_fold_ids]
    minimum_symbols = int(settings["minimum_symbols_per_date"])
    if any(_test_symbol_count(base_predictions, fold) < minimum_symbols for fold in folds):
        raise ValueError("Shared fold plan cannot satisfy minimum_symbols_per_date")
    fold_plan_identity = _hash({"folds": folds, "schema_version": SCHEMA_VERSION})
    plan = _plan(settings, feature_sets, folds, matched, fold_plan_identity)
    source_identity = _source_identity(source_path, rows)
    common_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "source_dataset_identity": source_identity,
        "feature_inventory": inventory,
        "feature_family_contracts": families,
        "feature_set_contracts": feature_sets,
        "family_resolution": [row for family in families for row in family.get("resolution", [])],
        "feature_set_equivalence": equivalence,
        "pit_audit": pit_audit,
        "enrichment_contract": enrichment_contract,
        "market_context_contract": market_context_contract_report(enrichment_contract, families),
        "breadth_contract": breadth_contract_report(enrichment_contract, families),
        "industry_mapping_audit": industry_mapping_audit(bounded_rows, families),
        "enriched_feature_coverage": family_coverage,
        "availability": availability,
        "leakage_audit": leakage_audit,
        "matched_population": matched,
        "shared_fold_plan": {"identity": fold_plan_identity, "folds": folds},
        "plan": plan,
        "plan_only": bool(settings["plan_only"]),
        "training_performed": False,
        "final_fit_performed": False,
        "trading_impact": "none",
        "paper_state_modified": False,
        "configuration_hash": _hash(settings),
        "code_commit": MLCoreArtifactWriter.git_commit(),
    }
    if settings["plan_only"]:
        return {
            **common_payload,
            "oos_predictions": [],
            "forecast_metrics": [],
            "portfolio_metrics": [],
            "pairwise_comparisons": [],
            "stability": [],
            "redundancy": redundancy_report(matched_rows, feature_sets, []),
        }
    predictions = train_feature_ablation_candidates(
        prepared,
        feature_sets,
        folds,
        base_predictions,
        settings,
        fold_plan_identity,
    )
    _validate_prediction_keys(predictions)
    forecast = forecast_metrics(predictions)
    portfolio = portfolio_metrics(predictions, settings)
    pairwise = pairwise_comparisons(forecast, portfolio, settings)
    stability = stability_report(predictions)
    redundancy = redundancy_report(matched_rows, feature_sets, predictions)
    return {
        **common_payload,
        "training_performed": True,
        "oos_predictions": predictions,
        "forecast_metrics": forecast,
        "portfolio_metrics": portfolio,
        "pairwise_comparisons": pairwise,
        "stability": stability,
        "redundancy": redundancy,
    }


def build_feature_roles(
    rows: Sequence[Mapping[str, Any]],
    available_features: Sequence[str],
) -> dict[str, str]:
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    roles = {column: "unsupported" for column in columns}
    for column in ("rebalance_date", "symbol", "benchmark_symbol"):
        if column in roles:
            roles[column] = "identifier"
    for column in TARGET_PROVENANCE_COLUMNS:
        if column in roles:
            roles[column] = "target provenance"
    for column in available_features:
        if column in roles:
            roles[column] = "feature"
    for column in CONTEXT_COLUMNS:
        if column in roles and column in available_features:
            roles[column] = "feature"
    for column in TARGET_OUTPUT_COLUMNS:
        if column in roles:
            roles[column] = "benchmark outcome" if column == "actual_benchmark_return_10d" else "outcome"
    for column in AUXILIARY_TARGET_COLUMNS:
        if column in roles:
            roles[column] = "outcome"
    if TARGET_COLUMN in roles:
        roles[TARGET_COLUMN] = "target"
    for column in columns:
        if column.startswith("actual_") and roles.get(column) in {"unsupported", "feature"}:
            roles[column] = "outcome"
        if column in {"decision_session_date", "first_actionable_session", "target_status"}:
            roles[column] = "decision-time metadata"
        if column.startswith("selector_") or column in {"fold_id", "candidate_id", "prediction"}:
            roles[column] = "diagnostic"
    return roles


def feature_inventory(
    rows: Sequence[Mapping[str, Any]],
    roles: Mapping[str, str],
    available_features: Sequence[str],
) -> list[dict[str, Any]]:
    first = next(iter(rows), {})
    result = []
    for column in sorted(roles):
        values = [row.get(column) for row in rows]
        family = _feature_family_for_column(column)
        result.append(
            {
                "column_name": column,
                "dtype": type(first.get(column)).__name__ if column in first else "unknown",
                "producer_module": _producer_module(column),
                "feature_family": family or "",
                "feature_description": _feature_description(column),
                "lookback_horizon": _lookback(column),
                "availability_timestamp": "feature_data_cutoff_timestamp" if column in available_features else "",
                "missing_value_semantics": "NaN preserved; fold-local median imputer in sklearn pipeline" if column in available_features else "",
                "point_in_time_status": _pit_status_for_family(family)["status"] if column in available_features else "",
                "current_enabled_status": column in available_features,
                "models_receiving_it": "configured tabular selector candidates" if column in available_features else "",
                "role": roles[column],
                "non_null_count": sum(1 for value in values if finite_number(value) is not None or (value not in (None, "") and roles[column] != "feature")),
            }
        )
    return result


def build_feature_family_contracts(
    rows: Sequence[Mapping[str, Any]],
    available_features: Sequence[str],
    settings: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = settings or {}
    available = set(available_features)
    family_columns = {
        "momentum_core": (
            "predicted_momentum_20d",
            "predicted_momentum_60d",
            "predicted_momentum_120d",
            "predicted_risk_adjusted_momentum",
        ),
        "momentum_extended": ENGINEERED_FEATURE_FAMILY_COLUMNS["momentum_extended"],
        "volatility": ENGINEERED_FEATURE_FAMILY_COLUMNS["volatility"],
        "drawdown": ENGINEERED_FEATURE_FAMILY_COLUMNS["drawdown"],
        "liquidity": ("predicted_liquidity_score",),
        "market_context": ENGINEERED_FEATURE_FAMILY_COLUMNS["market_context"],
        "market_relative": ENGINEERED_FEATURE_FAMILY_COLUMNS["market_relative"],
        "sector_relative": ENGINEERED_FEATURE_FAMILY_COLUMNS["sector_relative"],
        "industry_relative": ENGINEERED_FEATURE_FAMILY_COLUMNS["industry_relative"],
        "cross_sectional_rank": ENGINEERED_FEATURE_FAMILY_COLUMNS["cross_sectional_rank"],
        "regime_context": ENGINEERED_FEATURE_FAMILY_COLUMNS["regime_context"],
        "breadth": ENGINEERED_FEATURE_FAMILY_COLUMNS["breadth"],
        "fundamental_growth": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_growth"],
        "fundamental_profitability": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_profitability"],
        "fundamental_quality": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_quality"],
        "fundamental_balance_sheet": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_balance_sheet"],
        "fundamental_shareholder_actions": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_shareholder_actions"],
        "fundamental_valuation": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_valuation"],
        "fundamental_freshness": ENGINEERED_FEATURE_FAMILY_COLUMNS["fundamental_freshness"],
        "news": tuple(column for column in ENGINEERED_FEATURE_COLUMNS if ("news" in column or "sentiment" in column)),
    }
    source_columns = set().union(*(row.keys() for row in rows)) if rows else set()
    contracts = []
    for family_id, columns in family_columns.items():
        resolved = tuple(column for column in columns if column in available)
        present = tuple(column for column in columns if column in source_columns)
        missing = tuple(column for column in columns if column not in source_columns)
        pit = _pit_status_for_family(family_id)
        resolution = _family_resolution_rows(
            family_id,
            columns,
            present,
            missing,
            resolved,
            pit,
            settings,
        )
        payload = {
            "family_id": family_id,
            "family_version": FEATURE_CONTRACT_VERSION,
            "expected_columns": list(columns),
            "present_columns": list(present),
            "missing_columns": list(missing),
            "resolved_ordered_columns": list(resolved),
            "resolution": resolution,
            "producer_identity": _family_producer(family_id),
            "required_source_identities": ["stock_level_prediction_artifacts_enriched"],
            "lookback_requirements": _family_lookback(family_id),
            "minimum_history_requirements": "producer-defined; no cross-fold fitting before split",
            "point_in_time_availability_rule": pit["rule"],
            "point_in_time_status": pit["status"],
            "missing_value_policy": "preserve NaN; model pipeline imputes median fitted on training fold",
            "normalisation_policy": "model pipeline scaling fitted on training fold for scaled tabular models",
            "supported_model_types": ["ridge", "elastic_net", "random_forest", "gradient_boosting"],
            "blocked": pit["status"] == "BLOCKED",
        }
        payload["family_contract_hash"] = _hash(payload)
        contracts.append(payload)
    return contracts


def resolve_feature_set_contracts(
    settings: Mapping[str, Any],
    families: Sequence[Mapping[str, Any]],
    roles: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_family = {str(family["family_id"]): dict(family) for family in families}
    raw_sets = list(settings.get("feature_sets") or _default_feature_sets())
    empty_policy = str(settings.get("empty_family_policy", "fail")).lower()
    result = []
    seen_ids: set[str] = set()
    for raw in raw_sets:
        feature_set_id = str(raw["feature_set_id"])
        if feature_set_id in seen_ids:
            raise ValueError(f"Duplicate feature_set_id: {feature_set_id}")
        seen_ids.add(feature_set_id)
        columns: list[str] = []
        family_hashes = []
        blocked = []
        optional = {str(value) for value in raw.get("optional_families", [])}
        empty_requested = []
        for family_id in raw.get("include_families", []):
            family = by_family.get(str(family_id))
            if family is None:
                raise ValueError(f"Unknown feature family in feature set {feature_set_id}: {family_id}")
            if family["blocked"]:
                blocked.append(str(family_id))
            family_hashes.append(family["family_contract_hash"])
            resolved_columns = [str(column) for column in family["resolved_ordered_columns"]]
            if not resolved_columns and str(family_id) not in optional:
                empty_requested.append(str(family_id))
            columns.extend(resolved_columns)
        if blocked:
            raise ValueError(f"Feature set {feature_set_id} requested blocked families: {blocked}")
        if empty_requested and empty_policy == "fail":
            raise ValueError(f"Feature set {feature_set_id} requested empty mandatory families: {empty_requested}")
        columns.extend(str(column) for column in raw.get("include_columns", []))
        excluded = set(str(column) for column in raw.get("exclude_columns", []))
        ordered = tuple(column for column in _dedupe(columns) if column not in excluded)
        if not ordered:
            raise ValueError(f"Feature set {feature_set_id} resolved no feature columns")
        _validate_feature_columns(ordered, roles)
        payload = {
            "feature_set_id": feature_set_id,
            "ordered_feature_columns": list(ordered),
            "included_family_identities": family_hashes,
            "include_families": list(raw.get("include_families", [])),
            "optional_families": sorted(optional),
            "empty_requested_families": empty_requested,
            "empty_family_policy": empty_policy,
            "excluded_columns": sorted(excluded),
            "missingness_policy": "strict intersection for primary comparison; NaNs retained for fold-local imputation",
            "preprocessing_identity": _preprocessing_identity(ordered, "shared-plan"),
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
        }
        payload["feature_set_hash"] = _hash(payload)
        result.append(payload)
    return result


def audit_leakage(roles: Mapping[str, str], available_features: Sequence[str]) -> dict[str, Any]:
    blocked = []
    for column in available_features:
        role = roles.get(column, "unsupported")
        if role != "feature" or _defensive_outcome_like(column):
            blocked.append({"column": column, "role": role, "reason": "not_explicit_feature_or_outcome_like"})
    auto_numeric_routes = [
        "selector benchmark data uses FEATURE_COLUMNS plus optionally ENGINEERED_FEATURE_COLUMNS allowlist",
        "this ablation runner accepts only explicit feature-family or explicit-column contracts",
    ]
    return {
        "status": "PASS" if not blocked else "BLOCKED",
        "blocked_requested_features": blocked,
        "defensive_denylist_prefixes": list(DEFENSIVE_OUTCOME_PREFIXES),
        "auto_numeric_feature_discovery_routes": auto_numeric_routes,
        "unknown_columns_fail_in_strict_mode": True,
    }


def availability_report(
    rows: Sequence[Mapping[str, Any]],
    feature_sets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    periods = sorted({str(row.get("rebalance_date"))[:7] for row in rows if row.get("rebalance_date")})
    for feature_set in feature_sets:
        for feature in feature_set["ordered_feature_columns"]:
            for period in periods:
                scoped = [row for row in rows if str(row.get("rebalance_date", ""))[:7] == period]
                non_null = [row for row in scoped if finite_number(row.get(feature)) is not None]
                dates = sorted({str(row.get("rebalance_date")) for row in non_null})
                symbols = {str(row.get("symbol")).upper() for row in non_null if row.get("symbol")}
                result.append(
                    {
                        "feature_set_id": feature_set["feature_set_id"],
                        "feature": feature,
                        "decision_period": period,
                        "row_count": len(scoped),
                        "non_null_count": len(non_null),
                        "missing_count": len(scoped) - len(non_null),
                        "missing_fraction": (len(scoped) - len(non_null)) / len(scoped) if scoped else None,
                        "first_available_date": dates[0] if dates else None,
                        "last_available_date": dates[-1] if dates else None,
                        "symbol_coverage": len(symbols),
                        "date_coverage": len(dates),
                        "consecutive_missing_runs": _max_missing_run(scoped, feature),
                        "imputation_count": len(scoped) - len(non_null),
                        "rows_excluded": len(scoped) - len(non_null),
                        "missing_reason": "genuinely_unavailable_or_invalid_numeric" if len(scoped) != len(non_null) else "",
                    }
                )
    return result


def matched_population(
    rows: Sequence[Mapping[str, Any]],
    feature_sets: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    union_features = tuple(sorted({column for fs in feature_sets for column in fs["ordered_feature_columns"]}))
    common = []
    excluded_by_set = {fs["feature_set_id"]: 0 for fs in feature_sets}
    for row in rows:
        if finite_number(row.get(TARGET_COLUMN)) is None:
            continue
        include = True
        for feature_set in feature_sets:
            missing = [column for column in feature_set["ordered_feature_columns"] if finite_number(row.get(column)) is None]
            if missing:
                excluded_by_set[feature_set["feature_set_id"]] += 1
                include = False
        if include:
            common.append(dict(row))
    dates = sorted({str(row.get("rebalance_date")) for row in common})
    symbols = sorted({str(row.get("symbol")).upper() for row in common if row.get("symbol")})
    return common, {
        "mode": settings.get("comparison_mode", "strict_feature_intersection"),
        "common_row_count": len(common),
        "common_decision_date_count": len(dates),
        "common_symbol_count": len(symbols),
        "common_decision_dates": dates,
        "common_symbols": symbols,
        "rows_excluded_per_feature_set": excluded_by_set,
        "dates_excluded_per_feature_set": {
            fs["feature_set_id"]: len({str(row.get("rebalance_date")) for row in rows}) - len(dates)
            for fs in feature_sets
        },
        "union_feature_columns": list(union_features),
        "native_coverage_results_labelled_separately": True,
    }


def feature_set_equivalence(feature_sets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left_index, left in enumerate(feature_sets):
        left_ordered = tuple(left["ordered_feature_columns"])
        left_set = set(left_ordered)
        for right in feature_sets[left_index + 1:]:
            right_ordered = tuple(right["ordered_feature_columns"])
            right_set = set(right_ordered)
            if left_ordered == right_ordered:
                relationship = "identical_ordered_columns"
            elif left_set == right_set:
                relationship = "same_columns_different_order"
            elif left_set < right_set:
                relationship = "strict_subset"
            elif left_set > right_set:
                relationship = "strict_superset"
            elif left_set & right_set:
                relationship = "partial_overlap"
            else:
                relationship = "disjoint"
            rows.append(
                {
                    "left_feature_set_id": left["feature_set_id"],
                    "right_feature_set_id": right["feature_set_id"],
                    "relationship": relationship,
                    "overlap_count": len(left_set & right_set),
                    "left_column_count": len(left_ordered),
                    "right_column_count": len(right_ordered),
                    "identical_predictions_not_independent_evidence": relationship == "identical_ordered_columns",
                }
            )
    return rows


def pit_audit_report(rows: Sequence[Mapping[str, Any]], families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_rows = []
    for family in families:
        columns = family["resolved_ordered_columns"]
        checks = []
        for row in rows:
            decision = str(row.get("decision_timestamp") or row.get("rebalance_date") or "")
            cutoff = str(row.get("feature_data_cutoff_timestamp") or row.get("feature_timestamp") or decision)
            if decision and cutoff:
                checks.append(cutoff <= decision)
        family_rows.append(
            {
                "family_id": family["family_id"],
                "status": family["point_in_time_status"],
                "resolved_columns": columns,
                "availability_timestamp_le_decision": all(checks) if checks else None,
                "checked_row_count": len(checks),
                "rule": family["point_in_time_availability_rule"],
                "eligible_for_bounded_run": family["point_in_time_status"] in {"SAFE", "SAFE WITH CONDITIONS"} and all(checks or [True]),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "families": family_rows}


def enrichment_contract_report(rows: Sequence[Mapping[str, Any]], families: Sequence[Mapping[str, Any]], source_path: Path | None) -> dict[str, Any]:
    source_columns = set().union(*(row.keys() for row in rows)) if rows else set()
    enriched = [column for column in ENGINEERED_FEATURE_COLUMNS if column in source_columns]
    if not enriched:
        status = "no_additional_features"
    elif len(enriched) < len(ENGINEERED_FEATURE_COLUMNS):
        status = "partially_enriched"
    else:
        status = "enriched"
    return {
        "schema_version": "selector_enrichment_contract_v1",
        "enrichment_status": status,
        "source_dataset_path": str(source_path) if source_path else None,
        "resolved_enriched_columns": enriched,
        "resolved_enriched_column_count": len(enriched),
        "feature_family_identities": {
            family["family_id"]: family["family_contract_hash"]
            for family in families
            if family["resolved_ordered_columns"]
        },
        "row_count": len(rows),
        "symbol_count": len({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")}),
        "decision_date_count": len({str(row.get("rebalance_date")) for row in rows if row.get("rebalance_date")}),
        "decision_grid_identities": sorted({str(row.get("decision_grid_identity")) for row in rows if row.get("decision_grid_identity")})[:10],
        "universe_identity": "source_artifact_symbol_set",
    }


def enriched_feature_coverage(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for family in families:
        for feature in family["expected_columns"]:
            non_null = [
                row for row in rows
                if finite_number(row.get(feature)) is not None
            ]
            output.append(
                {
                    "family_id": family["family_id"],
                    "feature": feature,
                    "row_count": len(rows),
                    "non_null_count": len(non_null),
                    "missing_count": len(rows) - len(non_null),
                    "non_null_fraction": len(non_null) / len(rows) if rows else 0.0,
                    "all_null": len(non_null) == 0,
                    "date_coverage": len({str(row.get("rebalance_date")) for row in non_null if row.get("rebalance_date")}),
                    "symbol_coverage": len({str(row.get("symbol")).upper() for row in non_null if row.get("symbol")}),
                }
            )
    return output


def market_context_contract_report(
    enrichment_contract: Mapping[str, Any],
    families: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family = next((item for item in families if item["family_id"] == "market_context"), {})
    return {
        "contract_version": "selector_market_context_contract_v1",
        "benchmark_symbol": "SPY",
        "source_artifact_identity": enrichment_contract.get("source_dataset_path"),
        "price_adjustment_semantics": "same adjusted daily bars used by stock-level enrichment",
        "session_calendar_identity": "source artifact decision calendar",
        "feature_cutoff_rule": "market observation timestamp <= feature cutoff timestamp",
        "lookback_definitions": ["20d momentum", "60d momentum", "120d momentum", "20d volatility", "60d drawdown", "200d average distance"],
        "missing_session_behaviour": "missing remains missing; no future fill",
        "minimum_history": "200 prior observations for full context",
        "contract_hash": _hash(family),
        "resolved_columns": family.get("resolved_ordered_columns", []),
    }


def breadth_contract_report(
    enrichment_contract: Mapping[str, Any],
    families: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family = next((item for item in families if item["family_id"] == "breadth"), {})
    return {
        "contract_version": "selector_breadth_contract_v1",
        "eligible_universe_identity": "source artifact decision-date cross-section",
        "source_artifact_identity": enrichment_contract.get("source_dataset_path"),
        "feature_cutoff": "row feature cutoff / rebalance decision timestamp",
        "breadth_definitions": [
            "positive 20d momentum fraction",
            "positive 60d momentum fraction",
            "above 200d average fraction",
            "cross-sectional median 20d momentum",
            "20d momentum dispersion",
            "advance-decline ratio",
            "coverage",
        ],
        "survivorship_limitation": "bounded artifact uses source artifact symbols; historical delisted membership is not repaired here",
        "contract_hash": _hash(family),
        "resolved_columns": family.get("resolved_ordered_columns", []),
    }


def industry_mapping_audit(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    industry_values = [str(row.get("industry") or row.get("industry_id") or "").strip() for row in rows]
    mapped = [value for value in industry_values if value]
    family = next((item for item in families if item["family_id"] == "industry_relative"), {})
    return {
        "contract_version": "selector_industry_mapping_audit_v1",
        "mapping_source": "artifact industry column when present",
        "historically_versioned": False,
        "limitation": "static/current industry metadata only when supplied upstream; no symbol-change history repair",
        "row_count": len(rows),
        "mapped_row_count": len(mapped),
        "mapping_coverage": len(mapped) / len(rows) if rows else 0.0,
        "industry_count": len(set(mapped)),
        "resolved_columns": family.get("resolved_ordered_columns", []),
        "status": "available" if family.get("resolved_ordered_columns") else "blocked_or_missing",
    }


def train_feature_ablation_candidates(
    prepared: Sequence[Mapping[str, Any]],
    feature_sets: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    base_predictions: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    fold_plan_identity: str,
) -> list[dict[str, Any]]:
    by_key = {(row["rebalance_date"], row["symbol"]): dict(row) for row in prepared}
    predictions_by_fold = {}
    for row in base_predictions:
        predictions_by_fold.setdefault(row["fold_id"], []).append(row)
    outputs = []
    for feature_set in feature_sets:
        columns = tuple(feature_set["ordered_feature_columns"])
        for seed in settings["seeds"]:
            factories = _model_factories(int(seed), int(settings["sklearn_n_jobs"]))
            for model_id in settings["model_ids"]:
                if model_id not in factories:
                    raise ValueError(f"Unsupported feature ablation model_id: {model_id}")
                candidate_id = f"{settings['target_id']}::{feature_set['feature_set_id']}::{model_id}::seed_{seed}"
                for fold in folds:
                    train_rows = [
                        row for row in prepared
                        if str(fold["train_start_date"]) <= row["rebalance_date"] <= str(fold["train_end_date"])
                    ]
                    test_rows = [
                        by_key[(row["rebalance_date"], row["symbol"])]
                        for row in predictions_by_fold.get(fold["fold_id"], [])
                    ]
                    model = factories[model_id]()
                    x_train = [[row[column] for column in columns] for row in train_rows]
                    y_train = [row[TARGET_COLUMN] for row in train_rows]
                    model.fit(x_train, y_train)
                    x_test = [[row[column] for column in columns] for row in test_rows]
                    fold_predictions = model.predict(x_test) if x_test else []
                    preprocessing_identity = _preprocessing_identity(
                        columns,
                        _hash({
                            "fold_id": fold["fold_id"],
                            "train_start_date": fold["train_start_date"],
                            "train_end_date": fold["train_end_date"],
                            "train_row_count": len(train_rows),
                        }),
                    )
                    for source, prediction in zip(test_rows, fold_predictions):
                        outputs.append(
                            {
                                "candidate_id": candidate_id,
                                "target_id": settings["target_id"],
                                "feature_set_id": feature_set["feature_set_id"],
                                "feature_set_identity": feature_set["feature_set_hash"],
                                "ordered_feature_columns": json.dumps(list(columns), separators=(",", ":")),
                                "family_identities": json.dumps(feature_set["included_family_identities"], separators=(",", ":")),
                                "model_id": model_id,
                                "seed": int(seed),
                                "fold_id": fold["fold_id"],
                                "fold_plan_identity": fold_plan_identity,
                                "preprocessing_identity": preprocessing_identity,
                                "prediction": float(prediction),
                                "strict_oos": True,
                                "training_window_start": fold["train_start_date"],
                                "training_window_end": fold["train_end_date"],
                                "rebalance_date": source["rebalance_date"],
                                "symbol": source["symbol"],
                                "benchmark_symbol": source.get("benchmark_symbol", ""),
                                **{column: source.get(column) for column in TARGET_PROVENANCE_COLUMNS},
                                TARGET_COLUMN: source.get(TARGET_COLUMN),
                                **{column: source.get(column) for column in AUXILIARY_TARGET_COLUMNS},
                                **{column: source.get(column) for column in TARGET_OUTPUT_COLUMNS},
                            }
                        )
    return sorted(outputs, key=lambda row: (row["candidate_id"], row["rebalance_date"], row["symbol"]))


def forecast_metrics(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evaluator = CrossSectionalRankingEvaluator(target_column=TARGET_COLUMN)
    rows = []
    for candidate_id in sorted({row["candidate_id"] for row in predictions}):
        candidate_rows = [dict(row) for row in predictions if row["candidate_id"] == candidate_id]
        metric = evaluator.evaluate(candidate_rows, name=candidate_id, signal_column="prediction", kind="feature_ablation")
        values = [finite_number(row.get("prediction")) for row in candidate_rows]
        targets = [finite_number(row.get(TARGET_COLUMN)) for row in candidate_rows]
        hit = [
            1.0 if (pred is not None and target is not None and (pred >= 0) == (target >= 0)) else 0.0
            for pred, target in zip(values, targets)
            if pred is not None and target is not None
        ]
        metric.update(_candidate_parts(candidate_id))
        metric["median_spearman_ic"] = metric.get("mean_spearman_ic")
        metric["ic_information_ratio"] = metric.get("spread_sharpe")
        metric["pearson_correlation"] = metric.get("mean_pearson_ic")
        metric["hit_rate"] = mean(hit) if hit else None
        metric["prediction_coverage"] = sum(1 for value in values if value is not None) / len(candidate_rows) if candidate_rows else None
        metric["date_coverage"] = len({row["rebalance_date"] for row in candidate_rows})
        rows.append(metric)
    return rows


def portfolio_metrics(predictions: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for candidate_id in sorted({row["candidate_id"] for row in predictions}):
        rows = [dict(row) for row in predictions if row["candidate_id"] == candidate_id]
        for row in rows:
            row["selector_feature_signal"] = row["prediction"]
        periods, holdings = _replay(
            rows,
            "selector_feature_signal",
            "long_only_top_n_equal_weight",
            int(settings["portfolio_top_n"]),
            float(settings["cost_bps"]),
            float(settings["slippage_bps"]),
            float(settings["max_position_weight"]),
            float(settings["min_position_weight"]),
        )
        metric = _metrics("selector_feature_signal", "long_only_top_n_equal_weight", periods, holdings)
        metric["net_cumulative_return"] = metric.get("total_return")
        metric["net_cagr"] = metric.get("annualized_return")
        metric["net_sharpe"] = metric.get("sharpe")
        metric["annualized_turnover"] = metric.get("average_turnover")
        metric["total_cost_drag"] = metric.get("transaction_cost_drag")
        metric["trade_count"] = sum(
            1 for holding in holdings
            if (finite_number(holding.get("weight")) or 0.0) != 0.0
        )
        metric.update(_candidate_parts(candidate_id))
        metric["candidate_id"] = candidate_id
        metric["portfolio_policy_id"] = "exact_top_n"
        metric["cost_bps"] = float(settings["cost_bps"])
        metric["slippage_bps"] = float(settings["slippage_bps"])
        metric["coverage"] = len(rows)
        result.append(metric)
    return result


def pairwise_comparisons(
    forecast: Sequence[Mapping[str, Any]],
    portfolio: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    forecast_by = {row["candidate_id"]: row for row in forecast}
    portfolio_by = {row["candidate_id"]: row for row in portfolio}
    feature_sets = list(settings["feature_set_ids"])
    base = feature_sets[0]
    rows = []
    for model_id in settings["model_ids"]:
        for seed in settings["seeds"]:
            left = f"{settings['target_id']}::{base}::{model_id}::seed_{seed}"
            for feature_set_id in feature_sets[1:]:
                right = f"{settings['target_id']}::{feature_set_id}::{model_id}::seed_{seed}"
                if left not in forecast_by or right not in forecast_by:
                    continue
                row = {
                    "comparison_id": f"{right}__vs__{left}",
                    "baseline_candidate_id": left,
                    "candidate_id": right,
                    "model_id": model_id,
                    "seed": seed,
                    "baseline_feature_set_id": base,
                    "feature_set_id": feature_set_id,
                    "delta_ic": _delta(forecast_by[right], forecast_by[left], "mean_spearman_ic"),
                    "delta_top_minus_bottom_spread": _delta(forecast_by[right], forecast_by[left], "top_minus_bottom_spread"),
                    "delta_net_cagr": _delta(portfolio_by.get(right, {}), portfolio_by.get(left, {}), "net_cagr"),
                    "delta_net_sharpe": _delta(portfolio_by.get(right, {}), portfolio_by.get(left, {}), "net_sharpe"),
                    "delta_max_drawdown": _delta(portfolio_by.get(right, {}), portfolio_by.get(left, {}), "max_drawdown"),
                    "delta_turnover": _delta(portfolio_by.get(right, {}), portfolio_by.get(left, {}), "annualized_turnover"),
                    "delta_costs": _delta(portfolio_by.get(right, {}), portfolio_by.get(left, {}), "total_cost_drag"),
                    "delta_prediction_coverage": _delta(forecast_by[right], forecast_by[left], "prediction_coverage"),
                }
                rows.append(row)
    return rows


def redundancy_report(
    rows: Sequence[Mapping[str, Any]],
    feature_sets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    features = sorted({column for fs in feature_sets for column in fs["ordered_feature_columns"]})
    for left_index, left in enumerate(features):
        values = [finite_number(row.get(left)) for row in rows]
        finite = [value for value in values if value is not None]
        if finite and max(finite) - min(finite) <= 1e-12:
            result.append({"diagnostic_id": f"near_constant::{left}", "diagnostic_type": "near_constant_feature", "left": left, "right": "", "value": 1.0})
        missing_fraction = 1.0 - (len(finite) / len(rows) if rows else 0.0)
        if missing_fraction > 0.5:
            result.append({"diagnostic_id": f"high_missingness::{left}", "diagnostic_type": "high_missingness", "left": left, "right": "", "value": missing_fraction})
        for right in features[left_index + 1:]:
            corr = _correlation(rows, left, right)
            if corr is not None and abs(corr) >= 0.98:
                result.append({"diagnostic_id": f"feature_corr::{left}::{right}", "diagnostic_type": "near_duplicate_feature", "left": left, "right": right, "value": corr})
    for model_seed in sorted({_candidate_model_seed(row["candidate_id"]) for row in predictions}):
        candidates = sorted({row["candidate_id"] for row in predictions if _candidate_model_seed(row["candidate_id"]) == model_seed})
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1:]:
                corr = _candidate_rank_corr(predictions, left, right)
                if corr is not None:
                    result.append({"diagnostic_id": f"candidate_rank_corr::{left}::{right}", "diagnostic_type": "candidate_rank_correlation", "left": left, "right": right, "value": corr})
    return result


def stability_report(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    evaluator = CrossSectionalRankingEvaluator(target_column=TARGET_COLUMN)
    for candidate_id in sorted({row["candidate_id"] for row in predictions}):
        by_year: dict[str, list[dict[str, Any]]] = {}
        for row in predictions:
            if row["candidate_id"] == candidate_id:
                by_year.setdefault(str(row["rebalance_date"])[:4], []).append(dict(row))
        for year, year_rows in sorted(by_year.items()):
            metric = evaluator.evaluate(year_rows, name=candidate_id, signal_column="prediction", kind="feature_ablation_stability")
            parts = _candidate_parts(candidate_id)
            rows.append({
                "feature_set_id": parts["feature_set_id"],
                "candidate_id": candidate_id,
                "segment_type": "calendar_year",
                "segment_id": year,
                "row_count": len(year_rows),
                "date_count": metric.get("date_count"),
                "mean_spearman_ic": metric.get("mean_spearman_ic"),
                "top_minus_bottom_spread": metric.get("top_minus_bottom_spread"),
            })
    return rows


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw = dict(ml.get("selector_feature_ablation", {}) or {})
    output_dir = Path(raw.get("output_dir") or (stock_alpha_output_dir(config) / "selector_feature_ablation"))
    feature_sets = list(raw.get("feature_sets") or _default_feature_sets())
    target_id = str(raw.get("target_id", raw.get("reference_target_id", "raw_return_10d")))
    if target_id not in TARGET_IDS:
        raise ValueError(f"Unsupported selector feature ablation target_id: {target_id}")
    include_artifact_enriched = raw.get(
        "include_artifact_enriched_features",
        raw.get("include_engineered_features", True),
    )
    return {
        "enabled": bool(raw.get("enabled", False)),
        "source_dataset_path": str(raw.get("source_dataset_path") or raw.get("artifact_path") or ml.get("stock_level_prediction_artifacts_path", "")),
        "output_dir": str(output_dir),
        "allow_csv_fallback": bool(raw.get("allow_csv_fallback", ml.get("stock_level_allow_csv_artifact_fallback", False))),
        "include_engineered_features": bool(raw.get("include_engineered_features", include_artifact_enriched)),
        "include_artifact_enriched_features": bool(include_artifact_enriched),
        "include_runtime_engineered_features": bool(raw.get("include_runtime_engineered_features", False)),
        "strict_roles": bool(raw.get("strict_roles", True)),
        "empty_family_policy": str(raw.get("empty_family_policy", "fail")).lower(),
        "fail_on_identical_feature_sets": bool(raw.get("fail_on_identical_feature_sets", False)),
        "enriched_feature_coverage": dict(raw.get("enriched_feature_coverage", {}) or {}),
        "feature_sets": feature_sets,
        "feature_set_ids": [str(item["feature_set_id"]) for item in feature_sets],
        "target_id": target_id,
        "target_column": TARGET_IDS[target_id],
        "model_ids": list(raw.get("model_ids", ["ridge", "elastic_net"])),
        "seeds": list(raw.get("seeds", [ml.get("random_seed", 42)])),
        "plan_only": bool(raw.get("plan_only", False)),
        "comparison_mode": str(raw.get("comparison_mode", "strict_feature_intersection")),
        "min_train_dates": int(raw.get("min_train_dates", ml.get("stock_ranker_min_train_dates", 20))),
        "test_window_dates": int(raw.get("test_window_dates", ml.get("stock_ranker_test_window_dates", 5))),
        "embargo_dates": int(raw.get("embargo_dates", ml.get("stock_ranker_embargo_dates", 1))),
        "maximum_decision_dates": raw.get("maximum_decision_dates", dict(raw.get("bounded", {}) or {}).get("maximum_decision_dates")),
        "maximum_symbols": raw.get("maximum_symbols", dict(raw.get("bounded", {}) or {}).get("maximum_symbols")),
        "maximum_folds": raw.get("maximum_folds", dict(raw.get("bounded", {}) or {}).get("maximum_folds")),
        "minimum_symbols_per_date": int(raw.get("minimum_symbols_per_date", 2)),
        "sklearn_n_jobs": int(raw.get("sklearn_n_jobs", ml.get("sklearn_n_jobs", 1))),
        "portfolio_top_n": int(raw.get("portfolio_top_n", ml.get("stock_portfolio_replay_top_n", 25))),
        "cost_bps": float(raw.get("cost_bps", ml.get("stock_portfolio_replay_cost_bps", 10.0))),
        "slippage_bps": float(raw.get("slippage_bps", ml.get("stock_portfolio_replay_slippage_bps", 5.0))),
        "max_position_weight": float(raw.get("max_position_weight", ml.get("stock_portfolio_replay_max_position_weight", 0.05))),
        "min_position_weight": float(raw.get("min_position_weight", ml.get("stock_portfolio_replay_min_position_weight", 0.0))),
    }


def _default_feature_sets() -> list[dict[str, Any]]:
    return [
        {"feature_set_id": "momentum_only", "include_families": ["momentum_core"]},
        {"feature_set_id": "price_core", "include_families": ["momentum_core", "volatility", "drawdown", "liquidity"]},
        {
            "feature_set_id": "full_current_price",
            "include_families": [
                "momentum_core",
                "momentum_extended",
                "volatility",
                "drawdown",
                "liquidity",
                "market_relative",
                "sector_relative",
                "industry_relative",
                "cross_sectional_rank",
                "regime_context",
                "market_context",
                "breadth",
            ],
            "optional_families": ["industry_relative", "market_context", "breadth"],
        },
        {
            "feature_set_id": "price_plus_fundamental_quality",
            "include_families": ["momentum_core", "volatility", "drawdown", "liquidity", "fundamental_profitability", "fundamental_quality", "fundamental_balance_sheet", "fundamental_freshness"],
            "optional_families": ["fundamental_profitability", "fundamental_quality", "fundamental_balance_sheet", "fundamental_freshness"],
        },
        {
            "feature_set_id": "price_plus_fundamental_growth",
            "include_families": ["momentum_core", "volatility", "drawdown", "liquidity", "fundamental_growth", "fundamental_freshness"],
            "optional_families": ["fundamental_growth", "fundamental_freshness"],
        },
        {
            "feature_set_id": "price_plus_all_fundamentals",
            "include_families": ["momentum_core", "volatility", "drawdown", "liquidity", "fundamental_growth", "fundamental_profitability", "fundamental_quality", "fundamental_balance_sheet", "fundamental_shareholder_actions", "fundamental_valuation", "fundamental_freshness"],
            "optional_families": ["fundamental_growth", "fundamental_profitability", "fundamental_quality", "fundamental_balance_sheet", "fundamental_shareholder_actions", "fundamental_valuation", "fundamental_freshness"],
        },
        {
            "feature_set_id": "full_price_context_fundamentals",
            "include_families": ["momentum_core", "momentum_extended", "volatility", "drawdown", "liquidity", "market_relative", "sector_relative", "industry_relative", "cross_sectional_rank", "regime_context", "market_context", "breadth", "fundamental_growth", "fundamental_profitability", "fundamental_quality", "fundamental_balance_sheet", "fundamental_shareholder_actions", "fundamental_valuation", "fundamental_freshness"],
            "optional_families": ["industry_relative", "market_context", "breadth", "fundamental_growth", "fundamental_profitability", "fundamental_quality", "fundamental_balance_sheet", "fundamental_shareholder_actions", "fundamental_valuation", "fundamental_freshness"],
        },
    ]


def _bounded_rows(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    bounded = [dict(row) for row in rows]
    max_dates = settings.get("maximum_decision_dates")
    if max_dates:
        dates = sorted({str(row.get("rebalance_date")) for row in bounded if row.get("rebalance_date")})[-int(max_dates):]
        allowed = set(dates)
        bounded = [row for row in bounded if str(row.get("rebalance_date")) in allowed]
    max_symbols = settings.get("maximum_symbols")
    if max_symbols:
        symbols = sorted({str(row.get("symbol")).upper() for row in bounded if row.get("symbol")})[: int(max_symbols)]
        allowed_symbols = set(symbols)
        bounded = [row for row in bounded if str(row.get("symbol")).upper() in allowed_symbols]
    return sorted(bounded, key=lambda row: (str(row.get("rebalance_date")), str(row.get("symbol")).upper()))


def _validate_feature_columns(columns: Sequence[str], roles: Mapping[str, str]) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate feature columns are not allowed: {columns}")
    for column in columns:
        role = roles.get(column)
        if role != "feature":
            raise ValueError(f"Requested feature column is not role=feature: {column} role={role}")
        if _defensive_outcome_like(column):
            raise ValueError(f"Requested feature column matches defensive leakage denylist: {column}")


def _with_artifact_resident_features(
    rows: Sequence[Mapping[str, Any]],
    available_features: Sequence[str],
) -> tuple[str, ...]:
    available = list(available_features)
    known_artifact_features = tuple(
        column
        for family_id, columns in ENGINEERED_FEATURE_FAMILY_COLUMNS.items()
        if family_id.startswith("fundamental_")
        for column in columns
    )
    for column in known_artifact_features:
        if column not in available and any(finite_number(row.get(column)) for row in rows):
            available.append(column)
    return tuple(available)


def _defensive_outcome_like(column: str) -> bool:
    lowered = column.lower()
    if lowered.startswith("predicted_") or lowered.startswith("news_"):
        return False
    return lowered.startswith(DEFENSIVE_OUTCOME_PREFIXES) and any(part in lowered for part in DEFENSIVE_OUTCOME_SUBSTRINGS)


def _pit_status_for_family(family_id: str | None) -> dict[str, str]:
    if family_id == "news":
        return {"status": "BLOCKED", "rule": "requires first-seen and article-availability contract before model input"}
    if family_id in {"market_relative", "sector_relative", "industry_relative", "cross_sectional_rank", "breadth"}:
        return {"status": "SAFE WITH CONDITIONS", "rule": "must be decision-date local and use producer as-of joins only"}
    if family_id and family_id.startswith("fundamental_"):
        return {"status": "SAFE WITH CONDITIONS", "rule": "requires stock_fundamentals available_timestamp <= decision_timestamp and explicit entity mapping contract"}
    if family_id in {"momentum_core", "momentum_extended", "volatility", "drawdown", "liquidity", "market_context", "regime_context"}:
        return {"status": "SAFE", "rule": "feature_data_cutoff_timestamp <= decision_timestamp; generated from historical bars/context only"}
    return {"status": "AMBIGUOUS", "rule": "no family producer contract was resolved"}


def _feature_family_for_column(column: str) -> str:
    for family_id, columns in ENGINEERED_FEATURE_FAMILY_COLUMNS.items():
        if column in columns and not (family_id == "regime_context" and column == "volatility_regime"):
            return family_id
    if column in {"predicted_momentum_20d", "predicted_momentum_60d", "predicted_momentum_120d", "predicted_risk_adjusted_momentum"}:
        return "momentum_core"
    if column == "predicted_volatility_20d":
        return "volatility"
    if column == "predicted_drawdown_60d":
        return "drawdown"
    if column == "predicted_liquidity_score":
        return "liquidity"
    if column in CONTEXT_COLUMNS:
        return "market_context" if "breadth" not in column else "breadth"
    lowered = column.lower()
    if "sector" in lowered:
        return "sector_relative"
    if "industry" in lowered:
        return "industry_relative"
    if "market" in lowered or "spy" in lowered:
        return "market_relative"
    if "rank" in lowered:
        return "cross_sectional_rank"
    if "news" in lowered or "sentiment" in lowered:
        return "news"
    return ""


def _feature_description(column: str) -> str:
    descriptions = {
        "predicted_momentum_20d": "20-day price momentum predictor",
        "predicted_momentum_60d": "60-day price momentum predictor",
        "predicted_momentum_120d": "120-day price momentum predictor",
        "predicted_volatility_20d": "20-day realized volatility predictor",
        "predicted_drawdown_60d": "60-day drawdown predictor",
        "predicted_liquidity_score": "liquidity score predictor",
        "predicted_risk_adjusted_momentum": "momentum adjusted by realized risk",
    }
    return descriptions.get(column, "selector dataset column")


def _lookback(column: str) -> str:
    for token in ("20d", "60d", "120d", "21d", "63d", "126d", "200"):
        if token in column:
            return token
    return "producer-defined"


def _producer_module(column: str) -> str:
    if column in FEATURE_COLUMNS:
        return "core.research.ml.stock_level.prediction_artifacts.service"
    if column in ENGINEERED_FEATURE_COLUMNS or column in CONTEXT_COLUMNS:
        return "core.research.ml.stock_level.stock_level_alpha_features"
    return ""


def _family_producer(family_id: str) -> str:
    if family_id in {"momentum_core", "liquidity", "market_context"}:
        return "stock_level_prediction_artifacts_enriched"
    if family_id in {"momentum_extended", "market_relative", "sector_relative", "industry_relative", "cross_sectional_rank", "regime_context", "volatility", "drawdown"}:
        return "stock_level_alpha_features_builder"
    if family_id == "news":
        return "stock_alpha_news_contract"
    if family_id.startswith("fundamental_"):
        return "stock_fundamentals_pipeline"
    return "stock_level_alpha_features"


def _family_lookback(family_id: str) -> str:
    if family_id == "momentum_core":
        return "20/60/120 trading-day price history"
    if family_id == "volatility":
        return "20/60/252 trading-day realized volatility and ATR context"
    if family_id == "drawdown":
        return "60/120/252 trading-day drawdown context"
    if family_id == "momentum_extended":
        return "20/60/120/250 trading-day momentum context"
    if family_id == "market_context":
        return "21/63/126 trading-day market context where present"
    if family_id.startswith("fundamental_"):
        return "latest official filing facts available before decision timestamp; YOY features require prior comparable fiscal period"
    return "producer-defined"


def _preprocessing_identity(columns: Sequence[str], fit_population_identity: str) -> str:
    return _hash({
        "imputation_method": "SimpleImputer(strategy=median)",
        "scaling_method": "StandardScaler for ridge/elastic_net pipelines",
        "clipping_method": "none",
        "ordered_columns": list(columns),
        "fit_population_identity": fit_population_identity,
    })


def _plan(settings: Mapping[str, Any], feature_sets: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, Any]], matched: Mapping[str, Any], fold_plan_identity: str) -> dict[str, Any]:
    counts = {
        "feature_set_count": len(feature_sets),
        "target_count": 1,
        "model_count": len(settings["model_ids"]),
        "seed_count": len(settings["seeds"]),
        "fold_count": len(folds),
        "hyperparameter_candidate_count": 1,
    }
    expected = math.prod(counts.values())
    return {
        **counts,
        "expected_fits": expected,
        "feature_set_ids": [fs["feature_set_id"] for fs in feature_sets],
        "target_id": settings["target_id"],
        "model_ids": list(settings["model_ids"]),
        "seeds": list(settings["seeds"]),
        "fold_plan_identity": fold_plan_identity,
        "common_row_count": matched["common_row_count"],
        "plan_only": bool(settings["plan_only"]),
    }


def _source_identity(path: Path | None, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if path and path.exists():
        return artifact_identity(path)
    return {"row_count": len(rows), "resolved_artifact_path": str(path) if path else None}


def _test_symbol_count(base_predictions: Sequence[Mapping[str, Any]], fold: Mapping[str, Any]) -> int:
    return len({row["symbol"] for row in base_predictions if row["fold_id"] == fold["fold_id"]})


def _fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _prediction_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "candidate_id",
        "target_id",
        "feature_set_id",
        "feature_set_identity",
        "model_id",
        "seed",
        "fold_id",
        "fold_plan_identity",
        "preprocessing_identity",
        "rebalance_date",
        "symbol",
        "prediction",
        TARGET_COLUMN,
        "actual_benchmark_return_10d",
        "strict_oos",
    ]
    return _fields(rows, preferred)


def _validate_prediction_keys(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [(row["candidate_id"], row["fold_id"], row["rebalance_date"], row["symbol"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate selector feature ablation prediction rows detected")


def _candidate_parts(candidate_id: str) -> dict[str, Any]:
    target_id, feature_set_id, model_id, seed = candidate_id.split("::")
    return {"candidate_id": candidate_id, "target_id": target_id, "feature_set_id": feature_set_id, "model_id": model_id, "seed": seed.replace("seed_", "")}


def _candidate_model_seed(candidate_id: str) -> str:
    parts = _candidate_parts(candidate_id)
    return f"{parts['model_id']}::{parts['seed']}"


def _delta(right: Mapping[str, Any], left: Mapping[str, Any], column: str) -> float | None:
    r = finite_number(right.get(column))
    l = finite_number(left.get(column))
    return r - l if r is not None and l is not None else None


def _correlation(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float | None:
    pairs = [(finite_number(row.get(left)), finite_number(row.get(right))) for row in rows]
    clean = [(l, r) for l, r in pairs if l is not None and r is not None]
    if len(clean) < 2:
        return None
    left_values = [item[0] for item in clean]
    right_values = [item[1] for item in clean]
    return _pearson(left_values, right_values)


def _candidate_rank_corr(predictions: Sequence[Mapping[str, Any]], left: str, right: str) -> float | None:
    by_left = {(row["rebalance_date"], row["symbol"]): finite_number(row.get("prediction")) for row in predictions if row["candidate_id"] == left}
    by_right = {(row["rebalance_date"], row["symbol"]): finite_number(row.get("prediction")) for row in predictions if row["candidate_id"] == right}
    keys = sorted(set(by_left) & set(by_right))
    pairs = [(by_left[key], by_right[key]) for key in keys if by_left[key] is not None and by_right[key] is not None]
    if len(pairs) < 2:
        return None
    return _pearson([item[0] for item in pairs], [item[1] for item in pairs])


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right))
    left_var = sum((l - left_mean) ** 2 for l in left)
    right_var = sum((r - right_mean) ** 2 for r in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator > 0 else None


def _max_missing_run(rows: Sequence[Mapping[str, Any]], feature: str) -> int:
    max_run = 0
    current = 0
    for row in sorted(rows, key=lambda item: str(item.get("rebalance_date"))):
        if finite_number(row.get(feature)) is None:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _dedupe(values: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _family_resolution_rows(
    family_id: str,
    expected: Sequence[str],
    present: Sequence[str],
    missing: Sequence[str],
    resolved: Sequence[str],
    pit: Mapping[str, str],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    enabled = bool(settings.get("include_artifact_enriched_features", True))
    rows = []
    if not expected:
        reason = "IMPLEMENTATION ABSENT"
    elif pit["status"] == "BLOCKED":
        reason = "BLOCKED FOR PIT SAFETY"
    elif not enabled and any(column in ENGINEERED_FEATURE_COLUMNS or column in CONTEXT_COLUMNS for column in expected):
        reason = "DISABLED BY CONFIGURATION"
    elif present and not resolved:
        reason = "BLOCKED BY MISSING SOURCE DATA"
    elif not present:
        reason = "NOT PRESENT IN SOURCE ARTIFACT"
    elif resolved:
        reason = "RESOLVED"
    else:
        reason = "IMPLEMENTATION NOT REACHED"
    rows.append(
        {
            "family_id": family_id,
            "expected_columns": json.dumps(list(expected), separators=(",", ":")),
            "present_columns": json.dumps(list(present), separators=(",", ":")),
            "missing_columns": json.dumps(list(missing), separators=(",", ":")),
            "resolved_columns": json.dumps(list(resolved), separators=(",", ":")),
            "resolution_reason": reason,
            "point_in_time_status": pit["status"],
            "include_artifact_enriched_features": enabled,
            "include_runtime_engineered_features": bool(settings.get("include_runtime_engineered_features", False)),
        }
    )
    return rows


def _enriched_validation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "selector_enriched_feature_validation_report_v1",
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "source_dataset_identity": payload["source_dataset_identity"],
        "enrichment_contract": payload["enrichment_contract"],
        "family_resolution": payload["family_resolution"],
        "feature_set_equivalence": payload["feature_set_equivalence"],
        "pit_audit": payload["pit_audit"],
        "plan": payload["plan"],
        "training_performed": payload["training_performed"],
        "final_fit_performed": payload["final_fit_performed"],
        "trading_impact": payload["trading_impact"],
    }


def _enriched_validation_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Selector Enriched Feature Validation",
        "",
        DIAGNOSTIC_STATUS,
        "",
        f"Enrichment status: {payload['enrichment_contract']['enrichment_status']}",
        f"Resolved enriched columns: {payload['enrichment_contract']['resolved_enriched_column_count']}",
        "",
        "## Family Resolution",
    ]
    for row in payload["family_resolution"]:
        lines.append(f"- {row['family_id']}: {row['resolution_reason']} -> {row['resolved_columns']}")
    lines.extend(["", "## Feature-Set Equivalence"])
    for row in payload["feature_set_equivalence"]:
        lines.append(f"- {row['left_feature_set_id']} vs {row['right_feature_set_id']}: {row['relationship']}")
    lines.append("")
    return "\n".join(lines)


def _markdown(payload: Mapping[str, Any], paths: SelectorFeatureAblationPaths) -> str:
    plan = payload["plan"]
    forecast = payload["forecast_metrics"]
    portfolio = payload["portfolio_metrics"]
    lines = [
        "# Selector Feature Ablation",
        "",
        DIAGNOSTIC_STATUS,
        "",
        f"Source: {payload['source_dataset_identity'].get('resolved_artifact_path')}",
        f"Common rows: {payload['matched_population']['common_row_count']}",
        f"Shared folds: {plan['fold_count']}",
        f"Expected fits: {plan['expected_fits']}",
        f"Plan only: {payload['plan_only']}",
        "",
        "## Feature Sets",
    ]
    for contract in payload["feature_set_contracts"]:
        lines.append(f"- {contract['feature_set_id']}: {', '.join(contract['ordered_feature_columns'])}")
    lines.extend(["", "## Forecast Metrics"])
    for row in forecast:
        lines.append(f"- {row['candidate_id']}: mean_spearman_ic={row.get('mean_spearman_ic')} spread={row.get('top_minus_bottom_spread')}")
    lines.extend(["", "## Portfolio Metrics"])
    for row in portfolio:
        lines.append(f"- {row['candidate_id']}: net_cagr={row.get('net_cagr')} net_sharpe={row.get('net_sharpe')}")
    lines.extend(["", "## Artifacts", f"- Predictions: {paths.predictions_path}", f"- Pairwise: {paths.pairwise_path}", f"- Redundancy: {paths.redundancy_path}"])
    return "\n".join(lines) + "\n"
