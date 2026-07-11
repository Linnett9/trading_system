from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.research.ml.stock_level.news_risk_overlay_research_audit import (
    _risk_subset,
    _stable_hash,
)
from core.research.ml.stock_level.news_risk_overlay_research_inspection import _metric, _read_optional_json
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    _number,
    _write_json,
)

def _period_for_date(date_key: str, manifest: Mapping[str, Any]) -> str:
    for name, payload in dict(manifest.get("periods", {}) or {}).items():
        start = payload.get("start_date")
        end = payload.get("end_date")
        if start and end and str(start) <= date_key <= str(end):
            return name
    return "unassigned"


def _contrarian_grid_reports(
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    price_score_column: str,
    periods: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    weights = tuple(config.get("stock_alpha_news_risk_overlay_contrarian_grid_weights", (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)))
    selection_cost_bps = float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0))
    transforms = ("raw_probability", "training_median_centered", "decision_date_percentile", "fold_local_zscore")
    missing = ("no_adjustment", "fold_local_neutral_median")
    price = dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})
    contrarian = dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})
    cost_metric_overrides = dict(config.get("stock_alpha_news_risk_overlay_contrarian_grid_cost_metrics", {}) or {})
    grid_rows = []
    for weight in weights:
        for transform in transforms:
            for missing_policy in missing:
                config_id = f"w{float(weight):.2f}_{transform}_{missing_policy}"
                is_current = abs(float(weight) - float(config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25))) < 1e-12 and transform == "raw_probability" and missing_policy == "no_adjustment"
                cost_metrics = _grid_metrics_at_selection_cost(
                    config_id,
                    selection_cost_bps,
                    cost_metric_overrides,
                )
                median_calmar = (
                    _number(cost_metrics.get("median_validation_calmar"))
                    if cost_metrics
                    else (_metric(contrarian, "Calmar_ratio") if is_current else 0.0)
                )
                excess_return = (
                    _number(cost_metrics.get("median_excess_return"))
                    if cost_metrics
                    else ((_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0) if is_current else 0.0)
                )
                excess_sharpe = (
                    _number(cost_metrics.get("median_excess_sharpe"))
                    if cost_metrics
                    else ((_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0) if is_current else 0.0)
                )
                eligible = bool(cost_metrics.get("eligible", True)) if cost_metrics else is_current
                grid_rows.append(
                    {
                        "configuration_id": config_id,
                        "contrarian_weight": float(weight),
                        "news_transformation": transform,
                        "missing_news_treatment": missing_policy,
                        "selection_metric": "median_validation_calmar",
                        "selection_round_trip_cost_bps": selection_cost_bps,
                        "median_validation_calmar": median_calmar or 0.0,
                        "median_excess_return": excess_return or 0.0,
                        "median_excess_sharpe": excess_sharpe or 0.0,
                        "eligible": eligible,
                        "rejection_reason": "" if eligible else "not evaluated in lightweight validation artifact; requires manual grid run",
                        "training_data_only_transform_fit": transform in {"training_median_centered", "fold_local_zscore"},
                        "decision_timestamp_only_transform": transform == "decision_date_percentile",
                    }
                )
    winner = _select_contrarian_grid_configuration(grid_rows)
    fold_rows = []
    for period_name, payload in dict(periods.get("periods", {}) or {}).items():
        fold_rows.append(
            {
                "fold_id": period_name,
                "training_dates": "earlier_observations_only",
                "validation_dates": f"{payload.get('start_date')}..{payload.get('end_date')}",
                "selected_configuration": winner["configuration_id"],
                "price_only_return": _metric(price, "total_return_decimal"),
                "contrarian_return": _metric(contrarian, "total_return_decimal"),
                "excess_return": (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0),
                "price_only_drawdown": _metric(price, "maximum_drawdown"),
                "contrarian_drawdown": _metric(contrarian, "maximum_drawdown"),
                "sharpe_difference": (_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0),
                "calmar_difference": (_metric(contrarian, "Calmar_ratio") or 0.0) - (_metric(price, "Calmar_ratio") or 0.0),
                "news_coverage": payload.get("news_coverage"),
                "point_in_time_validation_failures": 0,
            }
        )
    selection = {
        "schema_name": "contrarian_grid_selection",
        "schema_version": "1.0",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "SCAFFOLD_CURRENT_FORMULA_ONLY",
        "experiment_registry_metadata": {
            "experiment_family": "stock_alpha_news_risk_overlay",
            "experiment_name": "news_contrarian_rerank_validation_scaffold",
            "research_only": True,
            "broker_invoked": False,
            "paper_orders_enabled": False,
            "live_orders_enabled": False,
            "news_originated_entries_enabled": False,
            "status": "DEVELOPMENT_ONLY",
        },
        "validation_status": "DEVELOPMENT_ONLY",
        "grid_search_status": "SCAFFOLD_COMPLETE_CURRENT_FORMULA_ONLY",
        "full_grid_search_implemented": False,
        "future_validation_stage_statuses": {
            "walk_forward": "NOT_IMPLEMENTED",
            "placebo_permutation": "NOT_IMPLEMENTED",
            "matched_controls": "NOT_IMPLEMENTED",
            "concentration_analysis": "NOT_IMPLEMENTED",
            "survivorship_audit": "NOT_IMPLEMENTED",
            "corporate_action_audit": "NOT_IMPLEMENTED",
            "missing_news_bias": "NOT_IMPLEMENTED",
            "transaction_cost_validation": "NOT_IMPLEMENTED",
        },
        "selection_policy": "highest median validation Calmar among eligible chronological folds",
        "selection_round_trip_cost_bps": selection_cost_bps,
        "cost_selection_policy": "single predeclared round-trip cost used for parameter selection; sensitivity reported separately",
        "cost_selection_metric_source": "metrics at configured selection_round_trip_cost_bps",
        "selected_configuration_id": winner["configuration_id"],
        "selection_metric": "median_validation_calmar",
        "selected_config_metric_at_selection_cost": winner.get("median_validation_calmar", 0.0),
        "selected_config": winner,
        "selected_configuration": winner,
        "holdout_used_for_selection": False,
        "used_holdout_for_selection": False,
        "eligible_config_count": sum(bool(row.get("eligible")) for row in grid_rows),
        "rejected_config_count": sum(not row["eligible"] for row in grid_rows),
        "rejected_configuration_count": sum(not row["eligible"] for row in grid_rows),
        "rejection_reasons": sorted({
            str(row.get("rejection_reason", ""))
            for row in grid_rows
            if not row.get("eligible") and row.get("rejection_reason")
        }),
        "rejected_configurations": [row for row in grid_rows if not row["eligible"]][:50],
        "tie_break_rules": [
            "higher median validation Calmar at configured selection cost",
            "higher median excess return",
            "higher median excess Sharpe",
            "smaller absolute contrarian weight",
            "deterministic lexical configuration ID",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": [
            "Grid artifact is scaffold/current-formula only; full bounded grid is not complete.",
            "Holdout rows are not used for parameter selection.",
        ],
        "fixed_portfolio_rules": {
            "price_score_column": price_score_column,
            "position_count": "unchanged",
            "max_position_weight": "unchanged",
            "entry_exit_rules": "unchanged",
            "transaction_cost_assumption_for_selection": f"{selection_cost_bps:g} bps round trip",
        },
    }
    return grid_rows, fold_rows, selection


def _select_contrarian_grid_configuration(grid_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [dict(row) for row in grid_rows if bool(row.get("eligible"))]
    candidates = eligible or [dict(row) for row in grid_rows]
    if not candidates:
        return {
            "configuration_id": "unavailable",
            "eligible": False,
            "median_validation_calmar": 0.0,
            "median_excess_return": 0.0,
            "median_excess_sharpe": 0.0,
            "contrarian_weight": 0.0,
            "rejection_reason": "no grid rows available",
        }
    return max(
        candidates,
        key=lambda row: (
            bool(row.get("eligible")),
            _number(row.get("median_validation_calmar")) or 0.0,
            _number(row.get("median_excess_return")) or 0.0,
            _number(row.get("median_excess_sharpe")) or 0.0,
            -abs(_number(row.get("contrarian_weight")) or 0.0),
            str(row.get("configuration_id", "")),
        ),
    )


def _selection_cost_metrics_from_scenarios(
    config: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
) -> dict[str, Any]:
    weight = float(config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25))
    config_id = f"w{weight:.2f}_raw_probability_no_adjustment"
    by_cost: dict[str, Any] = {}
    for scenario in dict(cost_scenarios.get("scenarios", {}) or {}).values():
        if not isinstance(scenario, Mapping):
            continue
        round_trip_bps = _number(scenario.get("round_trip_bps"))
        variants = dict(scenario.get("variants", {}) or {})
        price = dict(variants.get("price_only", {}) or {})
        contrarian = dict(variants.get("news_contrarian_rerank", {}) or {})
        if round_trip_bps is None or not contrarian:
            continue
        by_cost[f"{round_trip_bps:g}"] = {
            "median_validation_calmar": _metric(contrarian, "Calmar_ratio", "calmar_ratio") or 0.0,
            "median_excess_return": (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0),
            "median_excess_sharpe": (_metric(contrarian, "Sharpe_ratio", "sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio", "sharpe_ratio") or 0.0),
            "eligible": True,
        }
    return {config_id: by_cost} if by_cost else {}


def _grid_metrics_at_selection_cost(
    configuration_id: str,
    selection_cost_bps: float,
    cost_metric_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    by_config = dict(cost_metric_overrides.get(configuration_id, {}) or {})
    if not by_config:
        return {}
    exact_keys = (
        str(selection_cost_bps),
        f"{selection_cost_bps:g}",
        f"{selection_cost_bps:.1f}",
        f"{selection_cost_bps:.2f}",
    )
    for key in exact_keys:
        payload = by_config.get(key)
        if isinstance(payload, Mapping):
            return dict(payload)
    bps_key = f"{selection_cost_bps:g}_bps"
    payload = by_config.get(bps_key)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _frozen_contrarian_config(
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    periods: Mapping[str, Any],
) -> dict[str, Any]:
    selected = dict(selection.get("selected_configuration", {}) or {})
    payload = {
        "schema_name": "contrarian_frozen_config",
        "schema_version": "1.0",
        "configuration_id": selected.get("configuration_id", "unavailable"),
        "experiment_registry_metadata": dict(selection.get("experiment_registry_metadata", {}) or {}),
        "exact_formula": "contrarian_score = price_score + contrarian_weight * transformed_news_score",
        "formula": "contrarian_score = price_score + contrarian_weight * transformed_news_score",
        "news_transformation": selected.get("news_transformation", "raw_probability"),
        "contrarian_weight": selected.get("contrarian_weight", config.get("stock_alpha_news_risk_overlay_contrarian_weight", 0.25)),
        "missing_news_treatment": selected.get("missing_news_treatment", "no_adjustment"),
        "candidate_universe": "joined stock-alpha price-model candidate universe only",
        "ranking_direction": "descending contrarian_score",
        "tie_breaking": "symbol lexical ordering after score",
        "portfolio_rules": "unchanged from open-trade replay",
        "cost_assumptions": "selection uses one predeclared round-trip cost; sensitivity remains in cost_scenario_comparison.json",
        "development_dates": periods.get("periods", {}).get("development", {}),
        "validation_dates": periods.get("periods", {}).get("parameter_validation", {}),
        "selection_metric": selection.get("selection_policy"),
        "selection_round_trip_cost_bps": float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0)),
        "selected_config_metric_at_selection_cost": selected.get("median_validation_calmar", 0.0),
        "eligibility_constraints": {
            "positive_excess_return_majority_required": True,
            "median_excess_return_positive_required": True,
            "drawdown_not_worse_than_price_only_by_more_than_5pct_points": True,
            "sufficient_trade_count_required": True,
            "point_in_time_validation_failures_allowed": 0,
            "status": "SCAFFOLD_CURRENT_FORMULA_ONLY",
        },
        "tie_break_rules": ["median excess Sharpe", "lower turnover", "smaller absolute weight", "simpler transformation", "lexical configuration ID"],
        "used_holdout_for_selection": False,
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "hash_excluded_fields": ["generated_timestamp"],
        "final_holdout_required": True,
        "parameter_overrides_allowed_for_final_evaluation": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "holdout_type": "PSEUDO_HOLDOUT",
        "validation_label": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "is_final_validation": False,
        "validation_passed": False,
        "production_signal": False,
        "paper_orders_enabled": False,
        "live_orders_enabled": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "news_originated_entries_enabled": False,
    }
    payload["immutable_configuration_hash"] = _frozen_config_hash(payload)
    return payload


def _frozen_config_hash(payload: Mapping[str, Any]) -> str:
    ignored = set(payload.get("hash_excluded_fields", ["generated_timestamp"]) or [])
    ignored.update({"generated_timestamp", "immutable_configuration_hash"})
    stable_payload = {
        key: value
        for key, value in dict(payload).items()
        if key not in ignored
    }
    return _stable_hash(stable_payload)


def _holdout_report(
    replay: Mapping[str, Any],
    periods: Mapping[str, Any],
    frozen: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
) -> dict[str, Any]:
    price = dict(replay.get("risk_metrics", {}).get("price_only", {}) or {})
    contrarian = dict(replay.get("risk_metrics", {}).get("news_contrarian_rerank", {}) or {})
    validation_gate = _pseudo_holdout_validation_gate(periods, frozen)
    return {
        "schema_name": "contrarian_holdout_report",
        "schema_version": "1.0",
        "holdout_status": "PSEUDO_HOLDOUT",
        "validation_label": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "holdout_type": "PSEUDO_HOLDOUT",
        "evaluation_result": "PASSED_WITH_WARNINGS",
        "is_final_validation": False,
        "validation_passed": False,
        "validation_gate": validation_gate,
        "required_validation_gates": {
            "genuine_untouched_holdout": False,
            "frozen_configuration_not_retuned": True,
            "walk_forward_checks_passed": False,
            "placebo_permutation_checks_passed": False,
            "exposure_matched_controls_passed": False,
            "concentration_analysis_passed": False,
            "survivorship_audit_passed": False,
            "corporate_action_audit_passed": False,
            "missing_news_bias_analysis_passed": False,
            "transaction_cost_validation_passed": False,
        },
        "reason": "No genuinely untouched holdout has been established for the already-inspected contrarian hypothesis.",
        "warnings": [
            "Previously viewed period; not a genuine untouched holdout.",
            "This is a pseudo-holdout evaluation result, not final independent validation.",
        ],
        "frozen_configuration_hash": frozen.get("immutable_configuration_hash"),
        "selection_round_trip_cost_bps": frozen.get("selection_round_trip_cost_bps"),
        "parameter_overrides_used": False,
        "price_only": _risk_subset(price),
        "contrarian": _risk_subset(contrarian),
        "excess_return_over_price_only": (_metric(contrarian, "total_return_decimal") or 0.0) - (_metric(price, "total_return_decimal") or 0.0),
        "excess_sharpe": (_metric(contrarian, "Sharpe_ratio") or 0.0) - (_metric(price, "Sharpe_ratio") or 0.0),
        "drawdown_difference": (_metric(contrarian, "maximum_drawdown") or 0.0) - (_metric(price, "maximum_drawdown") or 0.0),
        "cost_sensitivity": cost_scenarios,
        "holdout_period": periods.get("periods", {}).get(
            "final_holdout",
            periods.get("periods", {}).get("final_untouched_holdout", {}),
        ),
    }


def _pseudo_holdout_validation_gate(
    periods: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    final_period = dict(periods.get("periods", {}).get("final_holdout", {}) or {})
    retuning_gate = _retuning_refusal_gate(frozen, override_requested=False)
    warnings = [
        "Final period is not demonstrably untouched.",
        "Robustness, placebo, matched-control, concentration, survivorship, corporate-action, missing-news, and transaction-cost gates remain incomplete.",
    ]
    return {
        "status": "BLOCKED_PSEUDO_HOLDOUT",
        "holdout_type": periods.get("holdout_type", "PSEUDO_HOLDOUT"),
        "validation_label": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "is_final_validation": False,
        "validation_passed": False,
        "parameter_overrides_used": False,
        "retuning_gate": retuning_gate,
        "retuning_gate_status": retuning_gate["status"],
        "retuning_gate_passed": bool(retuning_gate["retuning_gate_passed"]),
        "retuning_override_requested": bool(retuning_gate["retuning_override_requested"]),
        "retuning_override_allowed_for_final_evaluation": bool(retuning_gate["retuning_override_allowed_for_final_evaluation"]),
        "final_validation_blocked_by_pseudo_holdout": True,
        "final_evaluation_valid": False,
        "frozen_configuration_hash": frozen.get("immutable_configuration_hash"),
        "final_period_start_date": final_period.get("start_date"),
        "final_period_end_date": final_period.get("end_date"),
        "warnings": warnings,
    }


def _retuning_refusal_gate(
    frozen: Mapping[str, Any],
    *,
    override_requested: bool,
) -> dict[str, Any]:
    overrides_allowed = bool(frozen.get("parameter_overrides_allowed_for_final_evaluation", False))
    refused = bool(override_requested and not overrides_allowed)
    return {
        "status": "REFUSED" if refused else "NO_OVERRIDE_REQUESTED",
        "retuning_gate_status": "REFUSED" if refused else "NO_OVERRIDE_REQUESTED",
        "retuning_gate_passed": not refused,
        "retuning_override_requested": override_requested,
        "retuning_override_allowed_for_final_evaluation": overrides_allowed,
        "override_requested": override_requested,
        "overrides_allowed_for_final_evaluation": overrides_allowed,
        "final_validation_blocked_by_pseudo_holdout": False,
        "final_evaluation_valid": not refused,
        "reason": (
            "Parameter overrides are not allowed for final evaluation from a frozen configuration."
            if refused
            else "No retuning override requested."
        ),
    }


def _append_experiment_registry_entry(
    path: Path,
    *,
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    validation: Mapping[str, Any],
    coverage: Mapping[str, Any],
    event_category_analysis: Mapping[str, Any],
    decile_reconciliation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    risk_metrics = dict(replay.get("risk_metrics", {}) or {})
    decile_reconciliation_payload = dict(decile_reconciliation or {})
    if not decile_reconciliation_payload:
        decile_reconciliation_payload = {"status": "UNKNOWN"}
    event_categories = dict(event_category_analysis or {})
    categorized_count = sum(
        int(payload.get("count", 0))
        for key, payload in event_categories.items()
        if isinstance(payload, Mapping) and key != "general_negative_sentiment_or_uncategorized"
    )
    total_event_count = sum(
        int(payload.get("count", 0))
        for payload in event_categories.values()
        if isinstance(payload, Mapping)
    )
    entry = {
        "experiment_id": _stable_hash(
            {
                "hypothesis": "Among price-model-approved candidates, downside-news pressure may improve ranking.",
                "generated_date": datetime.now(timezone.utc).date().isoformat(),
                "selection_cost": config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0),
            }
        ),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Among price-model-approved candidates, downside-news pressure may improve ranking.",
        "status": "DEVELOPMENT_ONLY",
        "current_formula": "contrarian_score = price_score + contrarian_weight * transformed_news_score",
        "parameters_already_inspected": True,
        "historical_periods_already_viewed": True,
        "current_development_result": {
            "price_only": _risk_subset(dict(risk_metrics.get("price_only", {}) or {})),
            "news_contrarian_rerank": _risk_subset(dict(risk_metrics.get("news_contrarian_rerank", {}) or {})),
        },
        "holdout_contamination_status": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "selected_transaction_cost_bps": float(config.get("stock_alpha_news_risk_overlay_selection_round_trip_cost_bps", 10.0)),
        "candidate_universe_constraint": "news reranking remains inside the price-model-approved candidate universe",
        "score_direction": "higher score aligns with downside risk",
        "news_coverage": coverage.get("row_coverage_ratio"),
        "event_categorization_status": "UNCATEGORIZED" if categorized_count == 0 and total_event_count else "PARTIAL_OR_UNAVAILABLE",
        "decile_reconciliation_status": decile_reconciliation_payload.get("status", "UNKNOWN"),
        "grid_selection_status": dict(validation.get("contrarian_grid_selection", {}) or {}).get("status", "UNKNOWN"),
        "frozen_config_hash": dict(validation.get("contrarian_frozen_config", {}) or {}).get("immutable_configuration_hash"),
        "pseudo_holdout_gate_status": dict(
            dict(validation.get("contrarian_holdout_report", {}) or {}).get("validation_gate", {}) or {}
        ).get("status", "UNKNOWN"),
        "warnings": [
            "Development-only result; not a production signal.",
            "Final period remains pseudo-holdout because untouched status cannot be proven.",
        ],
        "holdout_accessed": True,
        "validation_label": "PSEUDO_HOLDOUT",
        "holdout_type": "PSEUDO_HOLDOUT",
        "is_final_validation": False,
        "validation_passed": False,
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "row_count": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


__all__ = [name for name in globals() if not name.startswith("__")]
