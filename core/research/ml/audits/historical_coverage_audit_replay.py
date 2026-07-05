from __future__ import annotations

from typing import Any

from core.research.ml.audits.historical_coverage_audit_math import _date


def _canonical_summary(canonical_replay: dict[str, Any]) -> dict[str, Any]:
    candidates = canonical_replay.get("candidates", {}) or {}
    exact = candidates.get("exact_champion_replay", {}) or {}
    optimizer = candidates.get("selected_bayesian_optimizer_diagnostic_policy", {}) or {}
    all_rows = [
        row for candidate in (exact, optimizer)
        for row in candidate.get("rows", []) or []
        if isinstance(row, dict) and row.get("rebalance_date")
    ]
    return {
        "earliest_canonical_replay_date": min(
            (str(row.get("rebalance_date")) for row in all_rows),
            default=None,
        ),
        "latest_canonical_replay_date": max(
            (str(row.get("rebalance_date")) for row in all_rows),
            default=None,
        ),
        "rebalance_date_count": len({
            str(row.get("rebalance_date")) for row in all_rows
        }),
        "raw_independent_periods_exact": int(
            (exact.get("canonical_continuous_equity", {}) or {}).get("row_count")
            or 0
        ),
        "raw_independent_periods_optimizer": int(
            (optimizer.get("canonical_continuous_equity", {}) or {}).get("row_count")
            or 0
        ),
        "diagnostic_periods_exact": int(
            (exact.get("diagnostic_period_grid", {}) or {}).get("row_count") or 0
        ),
        "diagnostic_periods_optimizer": int(
            (optimizer.get("diagnostic_period_grid", {}) or {}).get("row_count") or 0
        ),
    }
def _adjusted_replay_summary(adjusted_price_replay: dict[str, Any]) -> dict[str, Any]:
    candidates = adjusted_price_replay.get("candidates", {}) or {}
    return {
        name: {
            "valid_adjusted_independent_period_count": int(
                (row or {}).get("valid_adjusted_independent_period_count") or 0
            ),
            "valid_adjusted_period_count": int(
                (row or {}).get("valid_adjusted_period_count") or 0
            ),
            "invalid_adjusted_period_count": int(
                (row or {}).get("invalid_adjusted_period_count") or 0
            ),
            "fail_closed_reason": (row or {}).get("fail_closed_reason"),
        }
        for name, row in candidates.items()
        if isinstance(row, dict)
    }
def _possible_non_overlap_windows(canonical_replay: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, candidate in (canonical_replay.get("candidates", {}) or {}).items():
        rows = [
            row for row in candidate.get("rows", []) or []
            if isinstance(row, dict) and not row.get("exclusion_reason")
        ]
        output[name] = _non_overlap_count(rows)
    return output
def _non_overlap_count(rows: list[dict[str, Any]]) -> int:
    previous_end = None
    count = 0
    for row in sorted(rows, key=lambda item: str(item.get("rebalance_date", ""))):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None:
            continue
        if previous_end is None or start >= previous_end:
            count += 1
            previous_end = end
    return count
def _median_label_window_days(canonical_replay: dict[str, Any]) -> int:
    windows = []
    for candidate in (canonical_replay.get("candidates", {}) or {}).values():
        for row in candidate.get("rows", []) or []:
            start = _date(row.get("rebalance_date"))
            end = _date(row.get("outcome_end_date"))
            if start and end and end >= start:
                windows.append((end - start).days)
    if not windows:
        return 60
    windows = sorted(windows)
    return int(windows[len(windows) // 2])
