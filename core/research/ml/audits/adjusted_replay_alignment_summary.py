from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_comparison import RESEARCH_METADATA
from core.research.ml.audits.adjusted_replay_alignment_math import _count, _max_abs
from core.research.ml.audits.adjusted_replay_alignment_types import REPORT_CANDIDATES


def _candidate_summary(
    candidate: str,
    rows: list[dict[str, Any]],
    replay_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "row_count": len(rows),
        "adjusted_coverage_ratio": replay_row.get("adjusted_coverage_ratio"),
        "adjusted_full_symbol_coverage": replay_row.get(
            "adjusted_full_symbol_coverage",
        ),
        "missing_adjusted_symbols": replay_row.get("missing_adjusted_symbols", []),
        "empty_selection_with_positive_exposure_count": int(
            replay_row.get("empty_selection_with_positive_exposure_count") or 0
        ),
        "empty_selection_with_positive_exposure_dates": replay_row.get(
            "empty_selection_with_positive_exposure_dates",
            replay_row.get("affected_dates", []),
        ),
        "empty_selection_resolution": replay_row.get(
            "empty_selection_resolution",
            "unchanged",
        ),
        "invalid_adjusted_period_count": int(
            replay_row.get("invalid_adjusted_period_count") or 0
        ),
        "valid_adjusted_period_count": int(
            replay_row.get("valid_adjusted_period_count") or 0
        ),
        "valid_adjusted_independent_period_count": int(
            replay_row.get("valid_adjusted_independent_period_count") or 0
        ),
        "fail_closed_reason": replay_row.get("fail_closed_reason"),
        "missing_adjusted_price_row_count": _count(rows, "missing_adjusted_prices"),
        "date_misalignment_row_count": _count(rows, "date_misalignment"),
        "symbol_mismatch_row_count": _count(rows, "symbol_mismatch"),
        "exposure_mismatch_row_count": _count(rows, "exposure_mismatch"),
        "label_window_mismatch_row_count": _count(rows, "label_window_mismatch"),
        "non_overlap_mismatch_row_count": _count(rows, "non_overlap_mismatch"),
        "large_return_delta_row_count": _count(rows, "return_delta_above_threshold"),
        "large_candidate_net_return_delta_row_count": _count(
            rows,
            "candidate_net_return_delta_above_threshold",
        ),
        "adjustment_ratio_jump_row_count": _count(rows, "adjustment_ratio_jump"),
        "unexplained_adjusted_delta_row_count": _count(
            rows,
            "unexplained_adjusted_delta",
        ),
        "max_abs_return_delta": _max_abs(rows, "return_delta"),
        "max_abs_candidate_net_return_delta": _max_abs(
            rows,
            "candidate_net_return_delta",
        ),
        **RESEARCH_METADATA,
    }


def _alignment_summary(
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "same_rebalance_dates": not any(row["date_misalignment"] for row in rows),
        "same_selected_symbols": not any(row["symbol_mismatch"] for row in rows),
        "same_exposure_path": not any(row["exposure_mismatch"] for row in rows),
        "same_label_windows": not any(row["label_window_mismatch"] for row in rows),
        "same_non_overlap_periods": not any(row["non_overlap_mismatch"] for row in rows),
        "adjusted_coverage_complete": not any(
            int(summary.get("invalid_adjusted_period_count") or 0) > 0
            or summary.get("adjusted_full_symbol_coverage") is False
            for summary in summaries.values()
        ),
        "no_missing_adjusted_prices": not any(
            row["missing_adjusted_prices"] for row in rows
        ),
        "no_unexplained_large_return_deltas": not any(
            row["unexplained_adjusted_delta"] for row in rows
        ),
    }
    aligned = all(checks.values())
    return {
        "aligned_correctly": aligned,
        "checks": checks,
        "explanation_verdict": _explanation_verdict(checks, rows),
        "candidate_count": len(summaries),
        "row_count": len(rows),
        "missing_adjusted_price_row_count": _count(rows, "missing_adjusted_prices"),
        "date_misalignment_row_count": _count(rows, "date_misalignment"),
        "symbol_mismatch_row_count": _count(rows, "symbol_mismatch"),
        "exposure_mismatch_row_count": _count(rows, "exposure_mismatch"),
        "label_window_mismatch_row_count": _count(rows, "label_window_mismatch"),
        "non_overlap_mismatch_row_count": _count(rows, "non_overlap_mismatch"),
        "invalid_adjusted_period_count": sum(
            int(summary.get("invalid_adjusted_period_count") or 0)
            for summary in summaries.values()
        ),
        "valid_adjusted_period_count": sum(
            int(summary.get("valid_adjusted_period_count") or 0)
            for summary in summaries.values()
        ),
        "valid_adjusted_independent_period_count": sum(
            int(summary.get("valid_adjusted_independent_period_count") or 0)
            for summary in summaries.values()
        ),
        "large_return_delta_row_count": _count(rows, "return_delta_above_threshold"),
        "large_candidate_net_return_delta_row_count": _count(
            rows,
            "candidate_net_return_delta_above_threshold",
        ),
        "adjustment_ratio_jump_row_count": _count(rows, "adjustment_ratio_jump"),
        "unexplained_adjusted_delta_row_count": _count(
            rows,
            "unexplained_adjusted_delta",
        ),
        **RESEARCH_METADATA,
    }


def _explanation_verdict(
    checks: dict[str, bool],
    rows: list[dict[str, Any]],
) -> str:
    if not checks["no_missing_adjusted_prices"]:
        return "not_aligned_missing_adjusted_prices"
    if not checks["adjusted_coverage_complete"]:
        return "not_aligned_adjusted_coverage_failure"
    structural = [
        "same_rebalance_dates",
        "same_selected_symbols",
        "same_exposure_path",
        "same_label_windows",
        "same_non_overlap_periods",
    ]
    if not all(checks[name] for name in structural):
        return "not_aligned_replay_path_mismatch"
    if not checks["no_unexplained_large_return_deltas"]:
        return "not_aligned_unexplained_return_delta"
    if any(row["return_delta_above_threshold"] for row in rows):
        return "aligned_large_deltas_explained_by_adjustment_ratios"
    return "aligned_no_material_adjusted_delta"


def _red_flags(alignment: dict[str, Any]) -> list[str]:
    checks = alignment.get("checks", {})
    flags = []
    if not checks.get("same_rebalance_dates", False):
        flags.append("date_misalignment")
    if not checks.get("same_selected_symbols", False):
        flags.append("symbol_mismatch")
    if not checks.get("same_exposure_path", False):
        flags.append("exposure_mismatch")
    if not checks.get("same_label_windows", False):
        flags.append("label_window_mismatch")
    if not checks.get("same_non_overlap_periods", False):
        flags.append("non_overlap_mismatch")
    if not checks.get("adjusted_coverage_complete", False):
        flags.append("adjusted_coverage_failure")
    if not checks.get("no_missing_adjusted_prices", False):
        flags.append("missing_adjusted_prices")
    if not checks.get("no_unexplained_large_return_deltas", False):
        flags.append("unexplained_adjusted_return_delta")
    return flags
