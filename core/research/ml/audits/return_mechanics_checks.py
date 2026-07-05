from __future__ import annotations

from pathlib import Path
from typing import Any

from core.research.ml.audits.return_mechanics_candidate import _requirement_columns
from core.research.ml.audits.return_mechanics_loading import _read_yaml
from core.research.ml.audits.return_mechanics_math import _number, _numbers_close


def _mechanics_summary(
    shadow: dict[str, Any],
    meta_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    *,
    default_cost_bps: float,
) -> dict[str, Any]:
    holdout_rows = [row for row in meta_rows if row.get("split") == "holdout"]
    holdout_dates = {row.get("rebalance_date") for row in holdout_rows}
    period_counts = [
        int(row.get("number_of_periods") or 0)
        for row in candidates
        if row.get("available")
    ]
    exact = [row for row in candidates if row.get("exact_period_path")]
    max_abs_period = max(
        (
            abs(float(record["net_return"]))
            for row in exact
            for record in row.get("top_20_contributing_rebalance_dates", [])
            + row.get("worst_20_contributing_rebalance_dates", [])
        ),
        default=0.0,
    )
    return {
        "returns_compounded_by_rebalance_date": True,
        "rows_aggregated_before_compounding": True,
        "aggregation_method": "mean_by_rebalance_date_for_return_and_exposure",
        "multiple_strategy_variants_compounded_as_independent_capital": False,
        "transaction_cost_method": (
            "cost = abs(exposure - previous_exposure) * bps / 10000; "
            "applied once per aggregated rebalance date"
        ),
        "default_transaction_cost_bps": default_cost_bps,
        "turnover_method": "cumulative absolute exposure change, not annualized",
        "return_unit_inference": _return_unit(max_abs_period),
        "meta_holdout_row_count": len(holdout_rows),
        "meta_holdout_unique_rebalance_dates": len(holdout_dates),
        "shadow_overlay_mode": shadow.get("mode"),
        "candidate_period_counts": period_counts,
        "all_available_candidates_share_period_count": len(set(period_counts)) <= 1,
    }


def _champion_baseline_audit(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    by_name = {row["candidate_name"]: row for row in candidates}
    champion = by_name.get("champion_baseline", {})
    full = by_name.get("always_full_exposure", {})
    metrics_equal = _candidate_metrics_equal(champion, full)
    champion_config_path = (
        config.get("research", {})
        .get("dual_momentum", {})
        .get("champion_config_path")
    )
    champion_config = _read_yaml(Path(champion_config_path)) if champion_config_path else {}
    return {
        "champion_baseline_equals_always_full_exposure": metrics_equal,
        "intended_by_current_allocation_code": metrics_equal,
        "current_code_meaning": (
            "diagnostic full-exposure baseline over champion_return_next_period"
        ),
        "champion_config_path": champion_config_path,
        "champion_config_id": champion_config.get("champion_id"),
        "champion_config_target_exposure": (
            champion_config.get("overrides", {}).get("target_exposure")
        ),
        "represents_full_frozen_champion_yaml_replay": False,
        "same_date_range_as_ml_policies": _same_range_as_policies(candidates),
        "transaction_costs_applied": champion.get("costs"),
        "reported_transaction_costs": champion.get(
            "reported_estimated_transaction_costs"
        ),
        "should_have_turnover_costs_flag": (
            "The frozen champion YAML has target exposure/rebalance mechanics, "
            "but the allocation diagnostic baseline is constant 1.0 exposure, "
            "so it has zero exposure turnover in this report."
        ),
        "comparison_contains_champion_baseline": any(
            row.get("policy_name") == "champion_baseline"
            for row in comparison.get("baselines", [])
        ),
    }


def _data_sanity_checks(
    config: dict[str, Any],
    expanded_audit: dict[str, Any],
    meta_audit: dict[str, Any],
    expanded_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    variants = expanded_audit.get("variants", [])
    available_symbols = [
        int(row.get("available_symbols"))
        for row in variants
        if isinstance(row, dict) and row.get("available_symbols") is not None
    ]
    profile = config.get("research_profile", {})
    champion = next(
        (row for row in candidates if row["candidate_name"] == "champion_baseline"),
        {},
    )
    return {
        "profile_name": profile.get("name") or config.get("ml", {}).get("profile"),
        "benchmark_universe_name": profile.get("universe"),
        "universe_paths": expanded_audit.get("universe_paths", []),
        "requested_universe_size": profile.get("max_symbols"),
        "actual_available_symbol_count": max(available_symbols) if available_symbols else None,
        "benchmark_used_379_available_symbols": (
            max(available_symbols) == 379 if available_symbols else None
        ),
        "date_range": {
            "start": champion.get("start_date"),
            "end": champion.get("end_date"),
        },
        "expanded_row_count": expanded_audit.get("row_count"),
        "expanded_variant_count": expanded_audit.get("variant_count"),
        "meta_row_count": meta_audit.get("row_count"),
        "meta_source_dataset_hash": meta_audit.get("source_dataset_hash"),
        "symbol_concentration": _symbol_concentration(
            expanded_rows,
            start_date=champion.get("start_date"),
            end_date=champion.get("end_date"),
        ),
        "stooq_price_adjustment_status": (
            "not_proven_from_artifacts; Stooq adapters read Close/close columns "
            "and no adjustment metadata was found in the audited artifacts"
        ),
    }


def _leakage_check(
    comparison: dict[str, Any],
    optimizer: dict[str, Any],
) -> dict[str, Any]:
    forecast_requirements = sorted({
        str(column)
        for collection in ("policies", "baselines")
        for row in comparison.get(collection, [])
        if isinstance(row, dict)
        for column in row.get("required_prediction_columns", [])
    })
    forecast_inputs = sorted(_requirement_columns(forecast_requirements))
    actual_forecasts = [
        column for column in forecast_inputs if column.startswith("actual_")
    ]
    protocol = str(optimizer.get("selection_protocol") or "")
    selected = optimizer.get("selected_policy")
    return {
        "optimizer_selection_protocol": protocol,
        "optimizer_selects_parameters_on_out_of_fold_data": "out_of_fold" in protocol,
        "holdout_evaluated_once_after_selection": (
            isinstance(selected, dict)
            and isinstance(selected.get("frozen_holdout_metrics"), dict)
        ),
        "forecast_input_requirements": forecast_requirements,
        "forecast_input_columns": forecast_inputs,
        "actual_columns_used_as_forecasts": actual_forecasts,
        "actual_columns_are_evaluation_only": not actual_forecasts,
        "forecast_inputs_are_predicted_or_meta_predicted": all(
            column == "predicted_probability"
            or column.startswith("predicted_")
            or column.startswith("meta_predicted_")
            or "|" in column
            for column in forecast_inputs
        ),
    }


def _symbol_concentration(
    rows: list[dict[str, str]],
    *,
    start_date: Any,
    end_date: Any,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    contributions: dict[str, float] = {}
    used_rows = 0
    for row in rows:
        date = str(row.get("rebalance_date") or "")
        if start_date and date < str(start_date):
            continue
        if end_date and date > str(end_date):
            continue
        symbols = [
            symbol.strip().upper()
            for symbol in str(row.get("selected_symbols", "")).split(",")
            if symbol.strip()
        ]
        if not symbols:
            continue
        used_rows += 1
        contribution = _number(row.get("champion_return_next_period")) or 0.0
        per_symbol = contribution / len(symbols)
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
            contributions[symbol] = contributions.get(symbol, 0.0) + per_symbol
    return {
        "method": "row_weighted_selected_symbol_frequency_and_equal_return_attribution",
        "rows_used": used_rows,
        "top_symbols_by_selection_count": [
            {"symbol": symbol, "count": count}
            for symbol, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        ],
        "top_symbols_by_approx_return_contribution": [
            {"symbol": symbol, "approx_return_contribution": contribution}
            for symbol, contribution in sorted(
                contributions.items(),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        ],
    }


def _scenario_matrix(
    candidates: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    names = {
        "champion_baseline",
        "return_only_allocation",
        "selected_bayesian_optimizer_diagnostic_policy",
        "meta_ensemble_allocation",
        "binary_exposure_overlay",
    }
    return {
        row["candidate_name"]: row.get(key, {})
        for row in candidates
        if row["candidate_name"] in names and row.get("available")
    }


def _global_red_flags(
    candidates: list[dict[str, Any]],
    mechanics: dict[str, Any],
    champion_audit: dict[str, Any],
    leakage_check: dict[str, Any],
) -> list[str]:
    flags = []
    if not mechanics["all_available_candidates_share_period_count"]:
        flags.append("candidate_period_counts_do_not_match")
    if not champion_audit["champion_baseline_equals_always_full_exposure"]:
        flags.append("champion_baseline_not_equal_to_always_full_exposure")
    if champion_audit["represents_full_frozen_champion_yaml_replay"] is False:
        flags.append("champion_baseline_is_full_exposure_diagnostic_not_yaml_replay")
    if leakage_check["actual_columns_used_as_forecasts"]:
        flags.append("actual_columns_used_as_forecast_inputs")
    for candidate in candidates:
        flags.extend(
            f"{candidate['candidate_name']}:{flag}"
            for flag in candidate.get("red_flags", [])
        )
    return sorted(set(flags))


def _candidate_metrics_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("total_return", "max_drawdown", "turnover", "costs", "exposure_mean")
    if not left or not right:
        return False
    return all(
        _numbers_close(left.get(key), right.get(key))
        for key in keys
    )


def _same_range_as_policies(candidates: list[dict[str, Any]]) -> bool:
    available = [row for row in candidates if row.get("available")]
    ranges = {
        (row.get("start_date"), row.get("end_date"))
        for row in available
        if row.get("exact_period_path")
    }
    return len(ranges) <= 1


def _return_unit(max_abs_period: float) -> str:
    if max_abs_period > 5.0:
        return "percent_style_or_multiplier_values_suspected"
    if max_abs_period > 1.0:
        return "large_decimal_simple_returns"
    return "decimal_simple_returns"
