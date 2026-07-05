from __future__ import annotations

from typing import Any

from core.research.ml.independent_period_expansion_audit_math import (
    _compound,
    _max_drawdown,
    _number,
    _top_positive_share,
)
from core.research.ml.independent_period_expansion_audit_types import RESEARCH_METADATA


def _setting_metrics(
    *,
    candidate_name: str,
    setting: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    suspicious_dates: set[str],
    adjusted_closes_by_symbol: dict[str, dict[str, float]],
    minimum_independent_periods: int,
    adjusted_alignment: dict[str, Any],
    benchmark_relative_validation: dict[str, Any],
) -> dict[str, Any]:
    returns = [_number(row.get("net_return")) or 0.0 for row in selected_rows]
    total_return = _compound(returns)
    anomaly_rows = [
        row for row in selected_rows
        if str(row.get("rebalance_date")) not in suspicious_dates
    ]
    anomaly_adjusted_return = _compound(
        [_number(row.get("net_return")) or 0.0 for row in anomaly_rows]
    )
    spy_return = _benchmark_return(selected_rows, adjusted_closes_by_symbol, "SPY")
    excess = None if total_return is None or spy_return is None else total_return - spy_return
    top_5_share = _top_positive_share(returns, 5)
    drawdown = _max_drawdown(returns)
    overlap_risk = "none" if not skipped_rows else "controlled_by_filter"
    failed_gates = []
    if len(selected_rows) < minimum_independent_periods:
        failed_gates.append("minimum_adjusted_independent_periods")
    if not bool(adjusted_alignment.get("aligned_correctly", False)):
        failed_gates.append("adjusted_replay_alignment")
    if not bool(coverage.get("adjusted_full_symbol_coverage", False)):
        failed_gates.append("adjusted_replay_full_symbol_coverage")
    if excess is None or excess <= 0.0:
        failed_gates.append("positive_excess_vs_spy")
    if top_5_share is not None and top_5_share > 0.50:
        failed_gates.append("top_5_date_concentration")
    return {
        "candidate": candidate_name,
        "setting": setting["name"],
        "description": setting["description"],
        "spacing": setting["spacing"],
        "minimum_gap_days": setting["minimum_gap_days"],
        "enforce_non_overlap": setting["enforce_non_overlap"],
        "leakage_safe": True,
        "overlap_risk": overlap_risk,
        "independent_period_count": len(selected_rows),
        "overlap_skipped_period_count": len(skipped_rows),
        "adjusted_coverage_ratio": coverage.get("adjusted_coverage_ratio"),
        "valid_adjusted_period_count": coverage.get("valid_adjusted_period_count"),
        "invalid_adjusted_period_count": coverage.get("invalid_adjusted_period_count"),
        "empty_selection_with_positive_exposure_count": coverage.get(
            "empty_selection_with_positive_exposure_count",
            0,
        ),
        "empty_selection_resolution": coverage.get(
            "empty_selection_resolution",
            "unchanged",
        ),
        "canonical_return": total_return,
        "anomaly_adjusted_return": anomaly_adjusted_return,
        "max_drawdown": drawdown,
        "top_5_positive_return_share": top_5_share,
        "benchmark_return": spy_return,
        "benchmark_excess_return": excess,
        "promotion_gate_status": "blocked" if failed_gates else "pass",
        "failed_gates": sorted(set(failed_gates)),
        "selected_rebalance_dates": [
            str(row.get("rebalance_date")) for row in selected_rows
        ],
        "skipped_overlap_dates": [
            str(row.get("rebalance_date")) for row in skipped_rows
        ],
        "source_promotion_gates_preserved": _source_gates_preserved(
            benchmark_relative_validation,
            candidate_name,
        ),
        **RESEARCH_METADATA,
    }
def _benchmark_return(
    rows: list[dict[str, Any]],
    closes: dict[str, dict[str, float]],
    symbol: str,
) -> float | None:
    symbol_closes = closes.get(symbol, {})
    returns = []
    for row in rows:
        start = symbol_closes.get(str(row.get("rebalance_date")))
        end = symbol_closes.get(str(row.get("outcome_end_date")))
        if start is None or end is None or start <= 0:
            return None
        returns.append((end / start) - 1.0)
    return _compound(returns)
def _safest_expansion(
    rows: list[dict[str, Any]],
    minimum_independent_periods: int,
) -> dict[str, Any]:
    safe_rows = [row for row in rows if row["leakage_safe"]]
    if not safe_rows:
        return {"setting": None, "reason": "no_leakage_safe_setting"}
    best = max(
        safe_rows,
        key=lambda row: (
            int(row["independent_period_count"]),
            float(row.get("benchmark_excess_return") or -999.0),
        ),
    )
    reason = (
        "best_available_but_still_below_minimum"
        if int(best["independent_period_count"]) < minimum_independent_periods
        else "meets_minimum_independent_periods"
    )
    return {
        "setting": best["setting"],
        "independent_period_count": best["independent_period_count"],
        "reason": reason,
    }
def _red_flags(rows: list[dict[str, Any]], minimum: int) -> list[str]:
    flags = []
    if not any(int(row["independent_period_count"]) >= minimum for row in rows):
        flags.append("no_expansion_setting_reaches_minimum_independent_periods")
    if any(row["promotion_gate_status"] == "blocked" for row in rows):
        flags.append("promotion_gates_remain_blocked")
    return flags
def _source_gates_preserved(
    benchmark_relative_validation: dict[str, Any],
    candidate_name: str,
) -> bool:
    candidates = benchmark_relative_validation.get("candidates", []) or []
    row = next(
        (
            item for item in candidates
            if isinstance(item, dict) and item.get("candidate_name") == candidate_name
        ),
        {},
    )
    return bool(row.get("promotion_candidate_status") == "blocked")
