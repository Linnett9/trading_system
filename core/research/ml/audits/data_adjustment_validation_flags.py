from __future__ import annotations

from typing import Any


def _adjustment_red_flags(
    adjusted_status: str,
    acceptable: bool,
    suspicious_rows: list[dict[str, Any]],
    period_anomalies: list[dict[str, Any]],
    dependencies: dict[str, dict[str, Any]],
) -> list[str]:
    flags = []
    if not acceptable:
        flags.append("adjusted_price_status_not_acceptable")
    if adjusted_status.startswith("unknown"):
        flags.append("adjusted_price_status_unknown")
    if suspicious_rows:
        flags.append("suspicious_daily_price_rows_present")
    if period_anomalies:
        flags.append("large_symbol_period_anomalies_present")
    if any(
        int(row.get("suspicious_dependency_count") or 0) > 0
        for row in dependencies.values()
    ):
        flags.append("candidate_depends_on_suspicious_rows")
    return sorted(set(flags))
def _symbols_to_audit(
    canonical_replay: dict[str, Any],
    champion_audit: dict[str, Any],
    audit_config: dict[str, Any],
) -> list[str]:
    symbols = {str(symbol).upper() for symbol in audit_config["inspect_symbols"]}
    for candidate in canonical_replay.get("candidates", {}).values():
        for row in candidate.get("rows", []) or []:
            symbols.update(
                str(symbol).upper()
                for symbol in row.get("selected_symbols", []) or []
            )
    for row in champion_audit.get("exact_champion_replay", {}).get("period_rows", []) or []:
        symbols.update(
            str(symbol).upper()
            for symbol in row.get("selected_symbols", []) or []
        )
        for anomaly in row.get("symbol_return_anomalies", []) or []:
            if anomaly.get("symbol"):
                symbols.add(str(anomaly["symbol"]).upper())
    return sorted(symbol for symbol in symbols if symbol)
