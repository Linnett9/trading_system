from __future__ import annotations

from typing import Any

from core.research.ml.audits.champion_baseline_audit_io import (
    _meta_output_dir,
    _read_csv,
)
from core.research.ml.audits.champion_baseline_audit_math import _number
from core.research.ml.audits.champion_baseline_audit_types import RESEARCH_METADATA


def _evaluation_periods(
    config: dict[str, Any],
    meta_rows: list[dict[str, str]],
    expanded_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    holdout_dates = _configured_evaluation_dates(config, meta_rows)
    end_by_date = {}
    for row in expanded_rows:
        date = row.get("rebalance_date")
        if date in holdout_dates and row.get("outcome_end_date"):
            end_by_date[date] = row["outcome_end_date"]
    return [
        {"rebalance_date": date, "outcome_end_date": end_by_date[date]}
        for date in sorted(end_by_date)
    ]


def _configured_evaluation_dates(
    config: dict[str, Any],
    meta_rows: list[dict[str, str]],
) -> set[str]:
    horizon_config = (
        config.get("ml", {}).get("meta_canonical_horizon", {}) or {}
    )
    if bool(horizon_config.get("expand_from_source_predictions", False)):
        predictions_path = _meta_output_dir(config) / "meta_auxiliary_predictions.csv"
        prediction_dates = {
            row.get("rebalance_date")
            for row in _read_csv(predictions_path)
            if row.get("rebalance_date")
        }
        if prediction_dates:
            return prediction_dates
    return {
        row.get("rebalance_date")
        for row in meta_rows
        if row.get("split") == "holdout" and row.get("rebalance_date")
    }


def _diagnostic_baseline_rows(return_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_name = {
        row.get("candidate_name"): row
        for row in return_audit.get("candidates", [])
        if isinstance(row, dict)
    }
    for source_name, baseline_name in (
        ("champion_baseline", "champion_full_exposure_diagnostic"),
        ("always_full_exposure", "always_full_exposure"),
    ):
        source = by_name.get(source_name, {})
        rows.append({
            "baseline_name": baseline_name,
            "source_candidate_name": source_name,
            "semantic_type": "diagnostic_full_allocation_exposure",
            "available": bool(source),
            "target_exposure": 1.0,
            "total_return": source.get("total_return"),
            "continuous_total_return": None,
            "max_drawdown": source.get("max_drawdown"),
            "turnover": source.get("turnover"),
            "costs": source.get("costs"),
            "cost_turnover_status": "allocation overlay exposure turnover only",
            "is_exact_champion_replay": False,
            **RESEARCH_METADATA,
        })
    return rows


def _top_date_report(
    return_audit: dict[str, Any],
    exact_replay: dict[str, Any],
    expanded_rows: list[dict[str, str]],
) -> dict[str, Any]:
    expanded_by_date = _expanded_rows_by_date(expanded_rows)
    output = {}
    for name in (
        "champion_baseline",
        "return_only_allocation",
        "selected_bayesian_optimizer_diagnostic_policy",
    ):
        candidate = _return_audit_candidate(return_audit, name)
        output[name] = {
            "top_20": _attach_expanded_symbols(
                candidate.get("top_20_contributing_rebalance_dates", []),
                expanded_by_date,
            ),
            "worst_20": _attach_expanded_symbols(
                candidate.get("worst_20_contributing_rebalance_dates", []),
                expanded_by_date,
            ),
        }
    output["exact_champion_replay"] = {
        "top_20": exact_replay.get("period_grid_summary", {}).get(
            "top_20_rebalance_dates",
            [],
        ),
        "worst_20": exact_replay.get("period_grid_summary", {}).get(
            "worst_20_rebalance_dates",
            [],
        ),
    }
    output["late_2025_early_2026_dominance"] = _late_period_dominance(return_audit)
    return output


def _attach_expanded_symbols(
    records: list[dict[str, Any]],
    expanded_by_date: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        date = str(record.get("date") or record.get("rebalance_date") or "")
        variants = sorted(
            expanded_by_date.get(date, []),
            key=lambda row: float(row.get("champion_return_next_period", 0.0) or 0.0),
            reverse=True,
        )
        output.append({
            **record,
            "expanded_variant_count": len(variants),
            "top_expanded_variants": [
                {
                    "variant_id": row.get("variant_id"),
                    "selected_symbols": row.get("selected_symbols"),
                    "exposure_target": _number(row.get("exposure_target")),
                    "champion_return_next_period": _number(
                        row.get("champion_return_next_period")
                    ),
                }
                for row in variants[:5]
            ],
        })
    return output


def _stooq_adjustment_audit(
    config: dict[str, Any],
    exact_replay: dict[str, Any],
) -> dict[str, Any]:
    parquet_dir = config.get("ml", {}).get(
        "stooq_parquet_dir",
        "data/processed/stooq_parquet",
    )
    anomaly_rows = [
        anomaly
        for row in exact_replay.get("period_rows", [])
        for anomaly in row.get("symbol_return_anomalies", [])
    ]
    return {
        "data_path": parquet_dir,
        "data_source": "local Stooq parquet research data",
        "price_column_used": "close",
        "adjusted_status": (
            "unknown_from_repo_metadata; code reads Stooq close/Close columns and "
            "does not persist split/dividend adjustment metadata"
        ),
        "top_symbol_anomaly_count": len(anomaly_rows),
        "top_symbol_anomalies": anomaly_rows[:50],
    }


def _v2_vs_exact(
    return_audit: dict[str, Any],
    exact_replay: dict[str, Any],
) -> dict[str, Any]:
    exact_total = exact_replay.get("period_grid_summary", {}).get("total_return")
    continuous_total = exact_replay.get("continuous_equity_summary", {}).get(
        "total_return"
    )
    comparisons = {}
    for name in (
        "return_only_allocation",
        "selected_bayesian_optimizer_diagnostic_policy",
        "meta_ensemble_allocation",
        "binary_exposure_overlay",
    ):
        candidate = _return_audit_candidate(return_audit, name)
        candidate_return = candidate.get("reported_total_return") or candidate.get(
            "total_return"
        )
        comparisons[name] = {
            "candidate_return": candidate_return,
            "beats_exact_period_grid": (
                candidate_return is not None
                and exact_total is not None
                and float(candidate_return) > float(exact_total)
            ),
            "return_delta_vs_exact_period_grid": (
                float(candidate_return) - float(exact_total)
                if candidate_return is not None and exact_total is not None
                else None
            ),
            "beats_exact_continuous_equity": (
                candidate_return is not None
                and continuous_total is not None
                and float(candidate_return) > float(continuous_total)
            ),
        }
    return comparisons


def _red_flags(
    exact_replay: dict[str, Any],
    return_audit: dict[str, Any],
) -> list[str]:
    flags = [
        "current_champion_baseline_is_diagnostic_not_exact_replay",
        "current_allocation_baseline_compounds_overlapping_forward_periods",
    ]
    if not exact_replay.get("available"):
        flags.append("exact_champion_replay_unavailable")
    if _return_audit_candidate(return_audit, "champion_baseline").get("total_return"):
        flags.append("old_champion_baseline_name_is_misleading")
    if exact_replay.get("stooq_adjusted_status") == "unknown":
        flags.append("stooq_adjustment_status_unknown")
    return sorted(set(flags))


def _late_period_dominance(return_audit: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name in ("champion_baseline", "return_only_allocation"):
        candidate = _return_audit_candidate(return_audit, name)
        top = candidate.get("top_20_contributing_rebalance_dates", [])
        if not top:
            output[name] = None
            continue
        late_count = sum(
            "2025-08-01" <= str(row.get("date", "")) <= "2026-02-28"
            for row in top
        )
        output[name] = {
            "top_20_dates_in_2025_08_to_2026_02": late_count,
            "share": late_count / len(top),
        }
    return output


def _expanded_rows_by_date(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        output.setdefault(row.get("rebalance_date", ""), []).append(row)
    return output


def _return_audit_candidate(
    return_audit: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    return next(
        (
            row for row in return_audit.get("candidates", [])
            if row.get("candidate_name") == name
        ),
        {},
    )
