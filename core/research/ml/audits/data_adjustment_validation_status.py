from __future__ import annotations

from typing import Any

from core.research.ml.audits.data_adjustment_validation_detection import detect_split_like_jumps
from core.research.ml.audits.data_adjustment_validation_price_rows import _normalized_price_rows
from core.research.ml.audits.data_adjustment_validation_types import RESEARCH_METADATA
from core.research.ml.audits.data_adjustment_validation_utils import _number, _numbers_close


def _symbol_adjustment_report(
    symbol: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalized_price_rows(rows)
    suspicious = detect_split_like_jumps(
        symbol,
        normalized,
        suspicious_daily_return_abs=config["suspicious_daily_return_abs"],
        impossible_daily_return_abs=config["impossible_daily_return_abs"],
        split_ratio_tolerance=config["split_ratio_tolerance"],
    )
    raw_adjusted = _raw_adjusted_comparison(normalized)
    adjusted_status = _symbol_adjusted_status(normalized, suspicious, raw_adjusted)
    return {
        "symbol": symbol.upper(),
        "row_count": len(normalized),
        "first_date": normalized[0]["date"] if normalized else None,
        "last_date": normalized[-1]["date"] if normalized else None,
        "columns_present": sorted({
            column
            for row in rows
            for column in row
        }),
        "adjusted_status": adjusted_status,
        "raw_adjusted_comparison": raw_adjusted,
        "suspicious_row_count": len(suspicious),
        "suspicious_rows": suspicious,
        **RESEARCH_METADATA,
    }
def _raw_adjusted_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        row for row in rows
        if _number(row.get("raw_close")) is not None
        and _number(row.get("adjusted_close")) is not None
    ]
    if not comparable:
        return {
            "available": False,
            "reason": "raw and adjusted close columns were not both present",
            "row_count": 0,
        }
    close_matches_raw = 0
    close_matches_adjusted = 0
    raw_adjusted_differ = 0
    for row in comparable:
        close = _number(row.get("close"))
        raw = _number(row.get("raw_close"))
        adjusted = _number(row.get("adjusted_close"))
        if raw is None or adjusted is None:
            continue
        if not _numbers_close(raw, adjusted):
            raw_adjusted_differ += 1
        if close is not None and _numbers_close(close, raw):
            close_matches_raw += 1
        if close is not None and _numbers_close(close, adjusted):
            close_matches_adjusted += 1
    return {
        "available": True,
        "row_count": len(comparable),
        "raw_adjusted_differ_count": raw_adjusted_differ,
        "close_matches_raw_count": close_matches_raw,
        "close_matches_adjusted_count": close_matches_adjusted,
    }
def _symbol_adjusted_status(
    rows: list[dict[str, Any]],
    suspicious_rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    if not rows:
        return "unknown_missing_data"
    if comparison.get("available"):
        if int(comparison.get("raw_adjusted_differ_count") or 0) == 0:
            return "raw_adjusted_identical"
        raw_matches = int(comparison.get("close_matches_raw_count") or 0)
        adjusted_matches = int(comparison.get("close_matches_adjusted_count") or 0)
        if adjusted_matches > raw_matches:
            return "known_adjusted"
        if raw_matches > adjusted_matches:
            return "known_unadjusted"
        return "unknown_close_column_mismatch"
    if any(row.get("event_type") == "split_like_jump" for row in suspicious_rows):
        return "appears_unadjusted"
    return "unknown_no_adjusted_column"
def _overall_adjusted_status(symbol_reports: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("adjusted_status")) for row in symbol_reports}
    if not statuses:
        return "unknown_no_symbols"
    if statuses & {"known_unadjusted", "appears_unadjusted"}:
        return "appears_unadjusted"
    if statuses <= {"known_adjusted", "raw_adjusted_identical"}:
        return "known_adjusted"
    if "known_adjusted" in statuses and not any(status.startswith("unknown") for status in statuses):
        return "appears_adjusted"
    return "unknown"
def _adjusted_status_acceptable(
    status: str,
    config: dict[str, Any],
) -> bool:
    if status in config["acceptable_adjusted_price_statuses"]:
        return True
    return bool(config["allow_unknown_adjusted_price_status"]) and status.startswith(
        "unknown"
    )
