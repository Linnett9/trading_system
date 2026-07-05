from __future__ import annotations

from typing import Any

from core.research.ml.audits.data_adjustment_validation_types import RESEARCH_METADATA
from core.research.ml.audits.data_adjustment_validation_utils import _date, _number


def _period_anomaly_rows(
    champion_audit: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(config["large_symbol_period_return_abs"])
    rows = []
    for period in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []:
        if not isinstance(period, dict):
            continue
        for anomaly in period.get("symbol_return_anomalies", []) or []:
            symbol_return = _number(anomaly.get("return"))
            if symbol_return is None or abs(symbol_return) < threshold:
                continue
            rows.append({
                "symbol": str(anomaly.get("symbol", "")).upper(),
                "rebalance_date": period.get("rebalance_date"),
                "outcome_end_date": period.get("outcome_end_date"),
                "start_date": anomaly.get("start_date") or period.get("rebalance_date"),
                "end_date": anomaly.get("end_date") or period.get("outcome_end_date"),
                "start_close": _number(anomaly.get("start_close")),
                "end_close": _number(anomaly.get("end_close")),
                "period_return": symbol_return,
                "event_type": "large_symbol_period_return",
                "severity": "suspicious",
                **RESEARCH_METADATA,
            })
    return rows
def _candidate_suspicious_dependencies(
    canonical_replay: dict[str, Any],
    *,
    suspicious_rows: list[dict[str, Any]],
    period_anomalies: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for name, candidate in canonical_replay.get("candidates", {}).items():
        rows = [
            row for row in candidate.get("rows", []) or []
            if isinstance(row, dict)
        ]
        dependencies = []
        for row in rows:
            dependencies.extend(_daily_row_dependencies(row, suspicious_rows))
            dependencies.extend(_period_anomaly_dependencies(row, period_anomalies))
        unique = _unique_dependency_rows(dependencies)
        output[str(name)] = {
            "candidate_name": str(name),
            "suspicious_dependency_count": len(unique),
            "suspicious_rebalance_dates": sorted({
                str(row.get("rebalance_date"))
                for row in unique
                if row.get("rebalance_date")
            }),
            "suspicious_symbols": sorted({
                str(row.get("symbol"))
                for row in unique
                if row.get("symbol")
            }),
            "dependencies": unique[:100],
            **RESEARCH_METADATA,
        }
    return output
def _daily_row_dependencies(
    period: dict[str, Any],
    suspicious_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = _date(period.get("rebalance_date"))
    end = _date(period.get("outcome_end_date")) or start
    symbols = {str(symbol).upper() for symbol in period.get("selected_symbols", [])}
    if start is None or end is None or not symbols:
        return []
    output = []
    for event in suspicious_rows:
        event_date = _date(event.get("date"))
        symbol = str(event.get("symbol", "")).upper()
        if symbol not in symbols or event_date is None:
            continue
        if start <= event_date <= end:
            output.append({
                "dependency_type": "daily_price_event",
                "rebalance_date": period.get("rebalance_date"),
                "outcome_end_date": period.get("outcome_end_date"),
                "symbol": symbol,
                "event_date": event.get("date"),
                "event_type": event.get("event_type"),
                "daily_return": event.get("daily_return"),
            })
    return output
def _period_anomaly_dependencies(
    period: dict[str, Any],
    period_anomalies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    period_date = str(period.get("rebalance_date", ""))
    symbols = {str(symbol).upper() for symbol in period.get("selected_symbols", [])}
    output = []
    for anomaly in period_anomalies:
        symbol = str(anomaly.get("symbol", "")).upper()
        if anomaly.get("rebalance_date") == period_date and symbol in symbols:
            output.append({
                "dependency_type": "large_symbol_period_return",
                "rebalance_date": period_date,
                "outcome_end_date": period.get("outcome_end_date"),
                "symbol": symbol,
                "event_date": anomaly.get("end_date"),
                "event_type": anomaly.get("event_type"),
                "period_return": anomaly.get("period_return"),
            })
    return output
def _unique_dependency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (
            row.get("dependency_type"),
            row.get("rebalance_date"),
            row.get("symbol"),
            row.get("event_date"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output
def _suspicious_rebalance_dates_from_daily_rows(
    canonical_replay: dict[str, Any],
    suspicious_rows: list[dict[str, Any]],
) -> set[str]:
    dates = set()
    for candidate in canonical_replay.get("candidates", {}).values():
        for row in candidate.get("rows", []) or []:
            if _daily_row_dependencies(row, suspicious_rows):
                dates.add(str(row.get("rebalance_date")))
    return dates
