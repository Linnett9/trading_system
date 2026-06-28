from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from core.research.performance_metrics import calmar_ratio
from core.research.ml.allocation_v2 import _variant_exposures
from core.research.ml.allocation_v2_variants import grid_variant


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}

AUDITED_CANDIDATES = (
    "champion_baseline",
    "always_full_exposure",
    "binary_exposure_overlay",
    "return_only_allocation",
    "risk_adjusted_allocation",
    "meta_ensemble_allocation",
    "best_grid_search_diagnostic_policy",
    "selected_bayesian_optimizer_diagnostic_policy",
)

CAP_SCENARIOS = {
    "cap_-50pct_+50pct": (-0.50, 0.50),
    "cap_-25pct_+25pct": (-0.25, 0.25),
    "cap_-10pct_+10pct": (-0.10, 0.10),
}

COST_SENSITIVITY_BPS = (5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class ReturnMechanicsAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_return_mechanics_audit(config: dict[str, Any]) -> ReturnMechanicsAuditPaths:
    ml_config = config.get("ml", {})
    output_dir = Path(
        ml_config.get(
            "output_dir",
            Path(config.get("reports", {}).get("ml_dir", "reports/ml"))
            / "regime_transformer_meta_ensemble_v1",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    default_cost_bps = float(ml_config.get("allocation_transaction_cost_bps", 5.0))

    paths = _artifact_paths(config, output_dir)
    shadow = _read_json(paths["shadow_overlay"])
    comparison = _read_json(paths["allocation_comparison"])
    optimizer = _read_json(paths["optimizer_results"])
    grid_search = _read_json(paths["grid_search"])
    expanded_audit = _read_json(paths["expanded_audit"])
    meta_audit = _read_json(paths["meta_audit"])
    meta_rows = _read_csv(paths["meta_dataset"])
    expanded_rows = _read_csv(paths["expanded_dataset"])
    auxiliary_rows = _read_csv(paths["meta_auxiliary_predictions"])
    selected_optimizer_path = _read_json(paths["selected_optimizer_exposure_path_json"])

    reported_metrics = _reported_metrics_by_candidate(comparison, optimizer)
    series_by_candidate = _load_shadow_series(
        shadow,
        comparison,
        default_cost_bps=default_cost_bps,
    )
    optimizer_series = _load_selected_optimizer_series(
        selected_optimizer_path,
        optimizer,
    ) or _reconstruct_optimizer_series(
        config=config,
        optimizer=optimizer,
        meta_rows=meta_rows,
        auxiliary_rows=auxiliary_rows,
    )
    if optimizer_series is not None:
        series_by_candidate[optimizer_series["candidate_name"]] = optimizer_series

    candidate_audits = []
    for name in AUDITED_CANDIDATES:
        series = series_by_candidate.get(name)
        if series is None:
            candidate_audits.append(_missing_candidate(name))
            continue
        candidate_audits.append(
            _candidate_audit(
                name,
                series,
                reported_metrics.get(name, {}),
                default_cost_bps=default_cost_bps,
            )
        )

    mechanics = _mechanics_summary(
        shadow,
        meta_rows,
        candidate_audits,
        default_cost_bps=default_cost_bps,
    )
    champion_audit = _champion_baseline_audit(
        config,
        candidate_audits,
        comparison,
    )
    data_sanity = _data_sanity_checks(
        config,
        expanded_audit,
        meta_audit,
        expanded_rows,
        candidate_audits,
    )
    leakage_check = _leakage_check(comparison, optimizer)
    payload = {
        "mode": "return_mechanics_audit_research_only",
        "audited_candidates": list(AUDITED_CANDIDATES),
        "source_artifacts": {key: str(path) for key, path in paths.items()},
        "mechanics": mechanics,
        "data_sanity": data_sanity,
        "leakage_check": leakage_check,
        "champion_baseline_audit": champion_audit,
        "candidates": candidate_audits,
        "capped_return_sensitivity": _scenario_matrix(
            candidate_audits,
            "capped_return_sensitivity",
        ),
        "cost_sensitivity": _scenario_matrix(candidate_audits, "cost_sensitivity"),
        "red_flags": _global_red_flags(
            candidate_audits,
            mechanics,
            champion_audit,
            leakage_check,
        ),
        **RESEARCH_METADATA,
    }

    audit_paths = ReturnMechanicsAuditPaths(
        csv_path=output_dir / "benchmark_return_audit.csv",
        json_path=output_dir / "benchmark_return_audit.json",
        markdown_path=output_dir / "benchmark_return_audit.md",
    )
    _write_csv(audit_paths.csv_path, candidate_audits)
    audit_paths.json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    audit_paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return audit_paths


def _artifact_paths(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    ml_config = config.get("ml", {})
    report_dir = Path(config.get("reports", {}).get("ml_dir", output_dir.parent))
    cache_dir = Path(config.get("cache", {}).get("ml_dir", "cache/ml"))
    return {
        "trading_research_leaderboard": output_dir / "trading_research_leaderboard.md",
        "trading_research_leaderboard_csv": output_dir / "trading_research_leaderboard.csv",
        "allocation_comparison": output_dir / "allocation_policy_comparison.json",
        "allocation_leaderboard": output_dir / "allocation_policy_leaderboard.md",
        "allocation_diagnostics": output_dir / "allocation_policy_diagnostics.json",
        "grid_search": output_dir / "allocation_policy_grid_search.json",
        "optimizer_results": output_dir / "allocation_optimizer_results.json",
        "optimizer_report": output_dir / "allocation_optimizer_report.md",
        "selected_optimizer_exposure_path_json": (
            output_dir / "selected_optimizer_exposure_path.json"
        ),
        "selected_optimizer_exposure_path_csv": (
            output_dir / "selected_optimizer_exposure_path.csv"
        ),
        "shadow_overlay": output_dir / "allocation_shadow_overlay.json",
        "meta_audit": output_dir / "meta_dataset_audit.json",
        "meta_dataset": Path(
            ml_config.get("meta_dataset_path", cache_dir / "meta_ensemble_dataset.csv")
        ),
        "meta_auxiliary_predictions": output_dir / "meta_auxiliary_predictions.csv",
        "expanded_dataset": Path(
            ml_config.get(
                "expanded_rebalance_dataset_path",
                cache_dir / "expanded_rebalance_dataset.csv",
            )
        ),
        "expanded_audit": Path(
            ml_config.get(
                "expanded_rebalance_audit_path",
                report_dir / "expanded_rebalance_dataset_audit.json",
            )
        ),
    }


def _load_shadow_series(
    shadow: dict[str, Any],
    comparison: dict[str, Any],
    *,
    default_cost_bps: float,
) -> dict[str, dict[str, Any]]:
    output = {}
    comparison_metrics = _reported_metrics_by_candidate(comparison, {})
    for collection in ("policies", "baselines"):
        payloads = shadow.get(collection, {})
        if not isinstance(payloads, dict):
            continue
        for candidate_name, payload in payloads.items():
            if not isinstance(payload, dict) or not payload.get("available", True):
                continue
            rows = payload.get("rows", [])
            if not isinstance(rows, list):
                continue
            cost_bps = _number(payload.get("transaction_cost_bps"))
            if cost_bps is None:
                cost_bps = _number(comparison.get("transaction_cost_bps"))
            if cost_bps is None:
                cost_bps = default_cost_bps
            output[str(candidate_name)] = {
                "candidate_name": str(candidate_name),
                "policy_kind": payload.get("policy_kind"),
                "forecast_source": payload.get("forecast_source"),
                "required_prediction_columns": payload.get(
                    "required_prediction_columns",
                    [],
                ),
                "transaction_cost_bps": float(cost_bps),
                "period_source": "allocation_shadow_overlay_exact",
                "exact_period_path": True,
                "rows": rows,
                "reported_metrics": comparison_metrics.get(str(candidate_name), {}),
            }
    return output


def _load_selected_optimizer_series(
    payload: dict[str, Any],
    optimizer: dict[str, Any],
) -> dict[str, Any] | None:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return None
    selected = optimizer.get("selected_policy")
    metrics = {}
    if isinstance(selected, dict):
        raw_metrics = selected.get("frozen_holdout_metrics") or selected.get(
            "holdout_metrics"
        )
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
    candidate_name = str(
        metrics.get("policy_name")
        or f"selected_{payload.get('sampler_used', 'unknown')}_optimizer_diagnostic_policy"
    )
    return {
        "candidate_name": candidate_name,
        "policy_kind": "optimizer_diagnostic",
        "forecast_source": metrics.get("forecast_source"),
        "required_prediction_columns": metrics.get("required_prediction_columns", []),
        "transaction_cost_bps": _number(
            rows[0].get("transaction_cost_bps")
        ) or _number(metrics.get("transaction_cost_bps")) or 5.0,
        "period_source": "selected_optimizer_exposure_path_exact",
        "exact_period_path": True,
        "rows": [
            {
                "date": row.get("rebalance_date"),
                "baseline_return": row.get("period_return"),
                "exposure": row.get("exposure"),
            }
            for row in rows
        ],
        "reported_metrics": metrics,
    }


def _reconstruct_optimizer_series(
    *,
    config: dict[str, Any],
    optimizer: dict[str, Any],
    meta_rows: list[dict[str, str]],
    auxiliary_rows: list[dict[str, str]],
) -> dict[str, Any] | None:
    selected = optimizer.get("selected_policy")
    if not isinstance(selected, dict):
        return None
    metrics = selected.get("frozen_holdout_metrics") or selected.get("holdout_metrics")
    if not isinstance(metrics, dict):
        return None
    candidate_name = str(
        metrics.get("policy_name")
        or f"selected_{optimizer.get('sampler_used', 'unknown')}_optimizer_diagnostic_policy"
    )
    parameters = selected.get("parameters") or selected.get("selected_params")
    if not isinstance(parameters, dict) or not meta_rows or not auxiliary_rows:
        return {
            "candidate_name": candidate_name,
            "policy_kind": "optimizer_diagnostic",
            "forecast_source": metrics.get("forecast_source"),
            "required_prediction_columns": metrics.get("required_prediction_columns", []),
            "transaction_cost_bps": float(metrics.get("transaction_cost_bps", 5.0)),
            "period_source": "allocation_optimizer_scalar_metrics_only",
            "exact_period_path": False,
            "rows": [],
            "reconstruction_warning": (
                "Selected optimizer period path is not persisted in existing artifacts."
            ),
            "reported_metrics": metrics,
        }
    holdout_by_feature_id = {
        row.get("feature_id"): row
        for row in meta_rows
        if row.get("feature_id") and row.get("split") == "holdout"
    }
    joined_rows = []
    for row in auxiliary_rows:
        feature_id = row.get("feature_id")
        source = holdout_by_feature_id.get(feature_id)
        if not source:
            continue
        joined = dict(source)
        joined.update(row)
        joined_rows.append(joined)
    if not joined_rows:
        return None
    variant = grid_variant(parameters)
    exposures = _variant_exposures(
        joined_rows,
        [],
        config.get("ml", {}),
        variant=variant,
        fit_rows=None,
    )
    rows = [
        {
            "date": row.get("rebalance_date", ""),
            "baseline_return": row.get("champion_return_next_period", 0.0),
            "exposure": exposure,
        }
        for row, exposure in zip(joined_rows, exposures)
    ]
    return {
        "candidate_name": candidate_name,
        "policy_kind": "optimizer_diagnostic",
        "forecast_source": metrics.get("forecast_source"),
        "required_prediction_columns": metrics.get("required_prediction_columns", []),
        "transaction_cost_bps": float(metrics.get("transaction_cost_bps", 5.0)),
        "period_source": "reconstructed_from_saved_holdout_auxiliary_predictions",
        "exact_period_path": False,
        "rows": rows,
        "reconstruction_warning": (
            "Exact optimizer fit rows/cross-fitted auxiliary forecasts are not "
            "persisted, so quantile thresholds are refit on holdout for the "
            "per-date audit path. Saved optimizer scalar holdout metrics remain "
            "the authoritative optimizer result."
        ),
        "reported_metrics": metrics,
    }


def _candidate_audit(
    candidate_name: str,
    series: dict[str, Any],
    reported_metrics: dict[str, Any],
    *,
    default_cost_bps: float,
) -> dict[str, Any]:
    cost_bps = _number(series.get("transaction_cost_bps"))
    if cost_bps is None:
        cost_bps = default_cost_bps
    period_rows = _aggregate_return_rows(series.get("rows", []))
    records = _equity_records(period_rows, cost_bps=float(cost_bps))
    summary = _records_summary(records)
    reported = {**series.get("reported_metrics", {}), **reported_metrics}
    total_delta = _metric_delta(summary.get("total_return"), reported.get("total_return"))
    red_flags = []
    if summary["largest_positive_period_contribution"] > 0.50:
        red_flags.append("period_return_above_50pct")
    if summary["largest_negative_period_contribution"] < -0.50:
        red_flags.append("period_return_below_-50pct")
    if total_delta is not None and abs(total_delta) > 1e-6:
        red_flags.append("recomputed_total_return_differs_from_reported")
    exposure_sanity = _exposure_sanity(records)
    if exposure_sanity["out_of_range_exposure_dates"]:
        red_flags.append("exposure_outside_0_1")
    return {
        "candidate_name": candidate_name,
        "available": bool(records) or bool(reported),
        "policy_kind": series.get("policy_kind"),
        "period_source": series.get("period_source"),
        "exact_period_path": bool(series.get("exact_period_path")),
        "reconstruction_warning": series.get("reconstruction_warning"),
        "forecast_source": series.get("forecast_source"),
        "required_prediction_columns": series.get("required_prediction_columns", []),
        "forecast_inputs_use_actual_columns": any(
            column.startswith("actual_")
            for column in _requirement_columns(
                series.get("required_prediction_columns", [])
            )
        ),
        "transaction_cost_bps": float(cost_bps),
        "reported_total_return": _number(reported.get("total_return")),
        "reported_max_drawdown": _number(reported.get("max_drawdown")),
        "reported_sharpe": _number(reported.get("sharpe")),
        "reported_turnover": _number(reported.get("turnover")),
        "reported_estimated_transaction_costs": _number(
            reported.get("estimated_transaction_costs")
        ),
        "total_return_delta_vs_reported": total_delta,
        "max_drawdown_delta_vs_reported": _metric_delta(
            summary.get("max_drawdown"),
            reported.get("max_drawdown"),
        ),
        **summary,
        "top_20_contributing_rebalance_dates": _top_records(records, reverse=True),
        "worst_20_contributing_rebalance_dates": _top_records(records, reverse=False),
        "return_concentration": _return_concentration(records),
        "capped_return_sensitivity": _capped_return_sensitivity(records),
        "cost_sensitivity": _cost_sensitivity(period_rows),
        "exposure_sanity_checks": exposure_sanity,
        "red_flags": red_flags,
        **RESEARCH_METADATA,
    }


def _aggregate_return_rows(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    by_date: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        date = str(row.get("date") or row.get("rebalance_date") or "")
        if not date:
            continue
        period_return = _finite_float(row.get("baseline_return", 0.0))
        exposure = _finite_float(row.get("exposure", 1.0))
        by_date.setdefault(date, []).append((period_return, exposure))
    return [
        {
            "date": date,
            "baseline_return": mean(value[0] for value in values),
            "exposure": mean(value[1] for value in values),
            "source_row_count": len(values),
        }
        for date, values in sorted(by_date.items())
    ]


def _equity_records(
    periods: list[dict[str, float | str]],
    *,
    cost_bps: float,
) -> list[dict[str, float | str]]:
    equity = 1.0
    previous_exposure = 1.0
    records = []
    for period in periods:
        exposure = float(period["exposure"])
        baseline_return = float(period["baseline_return"])
        turnover = abs(exposure - previous_exposure)
        cost = turnover * cost_bps / 10_000.0
        net_return = (baseline_return * exposure) - cost
        equity *= 1.0 + net_return
        records.append({
            "date": str(period["date"]),
            "baseline_return": baseline_return,
            "exposure": exposure,
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
            "source_row_count": int(period.get("source_row_count", 1)),
        })
        previous_exposure = exposure
    return records


def _records_summary(records: list[dict[str, float | str]]) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in records]
    exposures = [float(row["exposure"]) for row in records]
    equity_curve = [1.0] + [float(row["equity"]) for row in records]
    total = _compound_returns(returns)
    annualized = _annualized_return(total, records)
    drawdown = _max_drawdown(equity_curve)
    return {
        "start_date": records[0]["date"] if records else None,
        "end_date": records[-1]["date"] if records else None,
        "number_of_periods": len(records),
        "total_return": total,
        "compounded_return": total,
        "arithmetic_mean_period_return": mean(returns) if returns else None,
        "geometric_mean_period_return": _geometric_mean_return(total, len(returns)),
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "sharpe": _sharpe_ratio(returns, records),
        "sortino": _sortino_ratio(returns, records),
        "calmar": calmar_ratio(annualized if annualized is not None else total, drawdown),
        "turnover": sum(float(row["turnover"]) for row in records),
        "costs": sum(float(row["cost"]) for row in records),
        "exposure_mean": mean(exposures) if exposures else None,
        "exposure_median": median(exposures) if exposures else None,
        "exposure_min": min(exposures) if exposures else None,
        "exposure_max": max(exposures) if exposures else None,
        "largest_positive_period_contribution": max(returns, default=0.0),
        "largest_negative_period_contribution": min(returns, default=0.0),
    }


def _top_records(
    records: list[dict[str, float | str]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    sorted_records = sorted(
        records,
        key=lambda row: float(row["net_return"]),
        reverse=reverse,
    )[:20]
    return [
        {
            "date": row["date"],
            "net_return": row["net_return"],
            "baseline_return": row["baseline_return"],
            "exposure": row["exposure"],
            "turnover": row["turnover"],
            "cost": row["cost"],
            "equity": row["equity"],
            "source_row_count": row["source_row_count"],
        }
        for row in sorted_records
    ]


def _return_concentration(records: list[dict[str, float | str]]) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in records]
    positives = sorted([value for value in returns if value > 0.0], reverse=True)
    total_positive = sum(positives)
    total_return = _compound_returns(returns)
    output = {}
    for count in (1, 3, 5, 10, 20):
        top = positives[:count]
        compounded = _compound_returns(top)
        output[f"top_{count}"] = {
            "sum_net_return": sum(top),
            "share_of_positive_period_returns": (
                sum(top) / total_positive if total_positive else None
            ),
            "compounded_return_from_top_periods": compounded,
            "compounded_share_of_total_return": (
                compounded / total_return if total_return else None
            ),
        }
    return output


def _capped_return_sensitivity(
    records: list[dict[str, float | str]],
) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in records]
    output = {}
    for scenario, (minimum, maximum) in CAP_SCENARIOS.items():
        capped = [min(maximum, max(minimum, value)) for value in returns]
        output[scenario] = {
            "minimum_period_return": minimum,
            "maximum_period_return": maximum,
            "total_return": _compound_returns(capped),
            "max_drawdown": _max_drawdown([1.0] + _equity_curve(capped)),
            "largest_positive_period_contribution": max(capped, default=0.0),
            "largest_negative_period_contribution": min(capped, default=0.0),
        }
    return output


def _cost_sensitivity(
    periods: list[dict[str, float | str]],
) -> dict[str, Any]:
    output = {}
    for cost_bps in COST_SENSITIVITY_BPS:
        records = _equity_records(periods, cost_bps=cost_bps)
        output[f"{cost_bps:g}_bps"] = {
            "transaction_cost_bps": cost_bps,
            "total_return": _compound_returns(
                [float(row["net_return"]) for row in records]
            ),
            "max_drawdown": _max_drawdown(
                [1.0] + [float(row["equity"]) for row in records]
            ),
            "turnover": sum(float(row["turnover"]) for row in records),
            "costs": sum(float(row["cost"]) for row in records),
        }
    return output


def _exposure_sanity(records: list[dict[str, float | str]]) -> dict[str, Any]:
    exposures = [float(row["exposure"]) for row in records]
    changes = [
        abs(current - previous)
        for previous, current in zip(exposures, exposures[1:])
    ]
    return {
        "average_exposure": mean(exposures) if exposures else None,
        "number_of_exposure_changes": sum(
            not math.isclose(current, previous)
            for previous, current in zip(exposures, exposures[1:])
        ),
        "largest_exposure_jump": max(changes, default=0.0),
        "out_of_range_exposure_dates": [
            row["date"]
            for row in records
            if float(row["exposure"]) < 0.0 or float(row["exposure"]) > 1.0
        ],
        "exposure_below_zero": any(value < 0.0 for value in exposures),
        "exposure_above_one": any(value > 1.0 for value in exposures),
    }


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


def _requirement_columns(requirements: Any) -> list[str]:
    output = []
    for requirement in requirements or []:
        output.extend(
            column
            for column in str(requirement).split("|")
            if column
        )
    return output


def _missing_candidate(candidate_name: str) -> dict[str, Any]:
    return {
        "candidate_name": candidate_name,
        "available": False,
        "skip_reason": "candidate not found in allocation audit artifacts",
        "red_flags": ["missing_candidate_period_path"],
        **RESEARCH_METADATA,
    }


def _reported_metrics_by_candidate(
    comparison: dict[str, Any],
    optimizer: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output = {}
    for collection in ("policies", "baselines"):
        for row in comparison.get(collection, []):
            if isinstance(row, dict) and row.get("policy_name"):
                output[str(row["policy_name"])] = row
    selected = optimizer.get("selected_policy")
    if isinstance(selected, dict):
        metrics = selected.get("frozen_holdout_metrics") or selected.get(
            "holdout_metrics"
        )
        if isinstance(metrics, dict) and metrics.get("policy_name"):
            output[str(metrics["policy_name"])] = metrics
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_name",
        "available",
        "period_source",
        "exact_period_path",
        "start_date",
        "end_date",
        "number_of_periods",
        "total_return",
        "reported_total_return",
        "total_return_delta_vs_reported",
        "compounded_return",
        "arithmetic_mean_period_return",
        "geometric_mean_period_return",
        "annualized_return",
        "max_drawdown",
        "reported_max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "turnover",
        "costs",
        "exposure_mean",
        "exposure_median",
        "exposure_min",
        "exposure_max",
        "largest_positive_period_contribution",
        "largest_negative_period_contribution",
        "forecast_source",
        "transaction_cost_bps",
        "red_flags",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: (
                    json.dumps(row.get(name))
                    if isinstance(row.get(name), (list, dict))
                    else row.get(name)
                )
                for name in fieldnames
            })


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Return Mechanics Audit",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        "## Mechanics",
        "",
        f"- Aggregation: {payload['mechanics']['aggregation_method']}",
        f"- Return unit inference: {payload['mechanics']['return_unit_inference']}",
        f"- Transaction costs: {payload['mechanics']['transaction_cost_method']}",
        f"- Turnover: {payload['mechanics']['turnover_method']}",
        f"- Meta holdout rows: {payload['mechanics']['meta_holdout_row_count']}",
        f"- Meta holdout rebalance dates: {payload['mechanics']['meta_holdout_unique_rebalance_dates']}",
        "",
        "## Candidate Summary",
        "",
        "|candidate|periods|total return|reported total|drawdown|Sharpe|turnover|costs|mean exposure|largest + period|largest - period|flags|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            "|{name}|{periods}|{total}|{reported}|{drawdown}|{sharpe}|"
            "{turnover}|{costs}|{exposure}|{positive}|{negative}|{flags}|".format(
                name=row["candidate_name"],
                periods=row.get("number_of_periods", ""),
                total=_fmt(row.get("total_return")),
                reported=_fmt(row.get("reported_total_return")),
                drawdown=_fmt(row.get("max_drawdown")),
                sharpe=_fmt(row.get("sharpe")),
                turnover=_fmt(row.get("turnover")),
                costs=_fmt(row.get("costs")),
                exposure=_fmt(row.get("exposure_mean")),
                positive=_fmt(row.get("largest_positive_period_contribution")),
                negative=_fmt(row.get("largest_negative_period_contribution")),
                flags=", ".join(row.get("red_flags", [])) or "",
            )
        )
    lines.extend([
        "",
        "## Concentration",
        "",
    ])
    for row in payload["candidates"]:
        if not row.get("available"):
            continue
        top_1 = row.get("return_concentration", {}).get("top_1", {})
        top_20 = row.get("return_concentration", {}).get("top_20", {})
        lines.append(
            "- {name}: top 1 share of positive period returns={top1}, "
            "top 20 share={top20}".format(
                name=row["candidate_name"],
                top1=_fmt(top_1.get("share_of_positive_period_returns")),
                top20=_fmt(top_20.get("share_of_positive_period_returns")),
            )
        )
    lines.extend([
        "",
        "## Champion Baseline",
        "",
        f"- Equals always_full_exposure: {payload['champion_baseline_audit']['champion_baseline_equals_always_full_exposure']}",
        f"- Champion config: {payload['champion_baseline_audit'].get('champion_config_path')}",
        f"- Represents full YAML replay: {payload['champion_baseline_audit']['represents_full_frozen_champion_yaml_replay']}",
        f"- Note: {payload['champion_baseline_audit']['should_have_turnover_costs_flag']}",
        "",
        "## Leakage Check",
        "",
        f"- Optimizer protocol: {payload['leakage_check']['optimizer_selection_protocol']}",
        f"- Out-of-fold optimizer selection: {payload['leakage_check']['optimizer_selects_parameters_on_out_of_fold_data']}",
        f"- Actual columns used as forecasts: {payload['leakage_check']['actual_columns_used_as_forecasts']}",
        "",
        "## Red Flags",
        "",
    ])
    if payload["red_flags"]:
        lines.extend(f"- {flag}" for flag in payload["red_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Top Dates", ""])
    for row in payload["candidates"]:
        if not row.get("available"):
            continue
        lines.append(f"### {row['candidate_name']}")
        lines.append("")
        lines.append("|date|net return|baseline return|exposure|")
        lines.append("|---|---:|---:|---:|")
        for record in row.get("top_20_contributing_rebalance_dates", [])[:20]:
            lines.append(
                "|{date}|{net}|{base}|{exposure}|".format(
                    date=record["date"],
                    net=_fmt(record["net_return"]),
                    base=_fmt(record["baseline_return"]),
                    exposure=_fmt(record["exposure"]),
                )
            )
        lines.append("")
    lines.append("Research only. Trading impact: none. Production validated: false.")
    lines.append("")
    return "\n".join(lines)


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_float(value: Any) -> float:
    number = _number(value)
    if number is None:
        raise ValueError(f"Expected finite numeric value, got {value!r}")
    return number


def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    curve = []
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    return curve


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown


def _geometric_mean_return(total_return: float, periods: int) -> float | None:
    if periods <= 0 or total_return <= -1.0:
        return None
    return (1.0 + total_return) ** (1.0 / periods) - 1.0


def _annualized_return(
    total_return: float,
    records: list[dict[str, float | str]],
) -> float | None:
    if len(records) < 2 or total_return <= -1.0:
        return None
    try:
        dates = [datetime.fromisoformat(str(row["date"])[:10]) for row in records]
    except ValueError:
        return None
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
        if (current - previous).days > 0
    ]
    terminal_days = max(1, round(median(gaps))) if gaps else 0
    elapsed_days = (dates[-1] - dates[0]).days + terminal_days
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def _observed_periods_per_year(records: list[dict[str, float | str]]) -> float:
    if len(records) < 2:
        return 1.0
    try:
        dates = [datetime.fromisoformat(str(row["date"])[:10]) for row in records]
    except ValueError:
        return 1.0
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
        if (current - previous).days > 0
    ]
    terminal_days = max(1, round(median(gaps))) if gaps else 0
    elapsed_days = (dates[-1] - dates[0]).days + terminal_days
    if elapsed_days <= 0:
        return 1.0
    return max(1.0, len(records) * 365.25 / elapsed_days)


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(mean((value - average) ** 2 for value in values))


def _sharpe_ratio(
    returns: list[float],
    records: list[dict[str, float | str]],
) -> float:
    if not returns:
        return 0.0
    std = _population_std(returns)
    if std == 0.0:
        return 0.0
    return mean(returns) / std * math.sqrt(_observed_periods_per_year(records))


def _sortino_ratio(
    returns: list[float],
    records: list[dict[str, float | str]],
) -> float:
    if not returns:
        return 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if downside_deviation == 0.0:
        return 0.0
    return mean(returns) / downside_deviation * math.sqrt(
        _observed_periods_per_year(records)
    )


def _metric_delta(left: Any, right: Any) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return False
    return math.isclose(left_number, right_number, abs_tol=tolerance)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
