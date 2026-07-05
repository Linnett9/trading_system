from __future__ import annotations

from typing import Any

from core.research.ml.independent_period_expansion_audit_math import _number
from core.research.ml.independent_period_expansion_audit_types import (
    REPORT_CANDIDATES,
    RESEARCH_METADATA,
)


def _candidate_adjusted_rows(
    adjusted_price_replay: dict[str, Any],
    candidate_name: str,
) -> list[dict[str, Any]]:
    return [
        row for row in (
            adjusted_price_replay.get("adjusted_canonical_replay", {})
            .get("candidates", {})
            .get(candidate_name, {})
            .get("rows", [])
            or []
        )
        if isinstance(row, dict)
    ]
def _candidate_summary(
    adjusted_price_replay: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    row = adjusted_price_replay.get("candidates", {}).get(candidate_name, {})
    return row if isinstance(row, dict) else {}
def _candidate_coverage(
    adjusted_price_replay: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    coverage = _candidate_summary(adjusted_price_replay, candidate_name).get(
        "coverage",
        {},
    )
    return coverage if isinstance(coverage, dict) else {}
def _no_selected_symbol_rows(
    *,
    adjusted_price_replay: dict[str, Any],
    canonical_replay: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for candidate_name in REPORT_CANDIDATES:
        raw_by_date = {
            str(row.get("rebalance_date")): row
            for row in (
                canonical_replay.get("candidates", {})
                .get(candidate_name, {})
                .get("rows", [])
                or []
            )
            if isinstance(row, dict)
        }
        for period in (
            _candidate_summary(adjusted_price_replay, candidate_name)
            .get("coverage", {})
            .get("periods", [])
            or []
        ):
            if period.get("fail_closed_reason") not in {
                "no_selected_symbols",
                "empty_selection_with_positive_exposure",
            }:
                continue
            date = str(period.get("rebalance_date"))
            raw = raw_by_date.get(date, {})
            exposure = _number(raw.get("exposure"))
            selected_symbols = raw.get("selected_symbols", []) or []
            expected_no_position = (
                len(selected_symbols) == 0
                and (exposure is None or abs(exposure) <= 1e-12)
            )
            positive_exposure_without_symbols = (
                bool(period.get("empty_selection_with_positive_exposure"))
                or (
                    len(selected_symbols) == 0
                    and exposure is not None
                    and exposure > 1e-12
                )
            )
            output.append({
                "candidate": candidate_name,
                "rebalance_date": date,
                "outcome_end_date": period.get("outcome_end_date"),
                "selected_symbols": [],
                "selected_symbol_count": 0,
                "exposure": exposure,
                "included_in_raw_canonical": bool(raw.get("included_in_canonical")),
                "why_no_symbols": (
                    "source replay row has zero exposure and empty target weights"
                    if expected_no_position
                    else "source replay row has positive exposure but no selected symbols"
                    if positive_exposure_without_symbols
                    else "source replay row has no selected symbols"
                ),
                "expected_no_position": expected_no_position,
                "replay_bug_suspected": not expected_no_position,
                **RESEARCH_METADATA,
            })
    return output
def _no_selected_symbol_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, int] = {}
    bug_rows = 0
    for row in rows:
        by_candidate[row["candidate"]] = by_candidate.get(row["candidate"], 0) + 1
        if row.get("replay_bug_suspected"):
            bug_rows += 1
    return {
        "row_count": len(rows),
        "by_candidate": by_candidate,
        "replay_bug_suspected_count": bug_rows,
        "verdict": (
            "expected_no_position_periods"
            if rows and bug_rows == 0
            else "review_required"
            if bug_rows
            else "none"
        ),
    }
