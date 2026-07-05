from __future__ import annotations

from datetime import timedelta
from typing import Any

from core.research.ml.independent_period_expansion_audit_math import _date


def _select_periods(
    rows: list[dict[str, Any]],
    setting: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [row for row in rows if row.get("exclusion_reason") is None]
    if setting["spacing"] == "strict_non_overlap":
        candidates = [row for row in eligible if row.get("included_in_canonical")]
    else:
        candidates = _first_periods_by_bucket(eligible, setting["spacing"])
    if not setting["enforce_non_overlap"]:
        return candidates, []
    selected = []
    skipped = []
    previous_end = None
    gap = timedelta(days=int(setting["minimum_gap_days"]))
    for row in sorted(candidates, key=lambda item: str(item.get("rebalance_date"))):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None or end is None:
            skipped.append(row)
            continue
        if previous_end is None or start >= previous_end + gap:
            selected.append(row)
            previous_end = end
        else:
            skipped.append(row)
    return selected, skipped
def _first_periods_by_bucket(
    rows: list[dict[str, Any]],
    spacing: str,
) -> list[dict[str, Any]]:
    output = {}
    for row in sorted(rows, key=lambda item: str(item.get("rebalance_date"))):
        start = _date(row.get("rebalance_date"))
        if start is None:
            continue
        if spacing == "monthly":
            key = start.strftime("%Y-%m")
        elif spacing == "quarterly":
            key = f"{start.year}-Q{((start.month - 1) // 3) + 1}"
        elif spacing == "all_valid_min_gap":
            key = str(row.get("rebalance_date"))
        else:
            key = start.strftime("%Y-%m")
        output.setdefault(key, row)
    return list(output.values())
