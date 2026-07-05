from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.research.ml.audits.data_adjustment_validation_types import (
    NOTICE,
    RESEARCH_METADATA,
)
from core.research.ml.audits.data_adjustment_validation_utils import _fmt


def _write_adjustment_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "symbol",
        "date",
        "previous_date",
        "event_type",
        "severity",
        "daily_return",
        "price_ratio",
        "split_like_factor",
        "adjusted_status",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in payload.get("symbols", []):
            rows = report.get("suspicious_rows", []) or [{
                "symbol": report.get("symbol"),
                "adjusted_status": report.get("adjusted_status"),
            }]
            for row in rows:
                writer.writerow({
                    **{name: row.get(name) for name in fieldnames},
                    "adjusted_status": report.get("adjusted_status"),
                    **RESEARCH_METADATA,
                })


def _write_clean_replay_csv(path: Path, payload: dict[str, Any]) -> None:
    fieldnames = [
        "candidate_name",
        "available",
        "raw_canonical_return",
        "clean_canonical_return",
        "return_delta_clean_vs_raw",
        "raw_benchmark_relative_pass",
        "clean_benchmark_relative_pass",
        "clean_data_return_positive",
        "clean_data_benchmark_relative",
        "clean_data_verdict",
        "excluded_period_count",
        "remaining_period_count",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("candidates", {}).values():
            writer.writerow({name: row.get(name) for name in fieldnames})


def _adjustment_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Data Adjustment Audit",
        "",
        NOTICE,
        "",
        f"Adjusted price status: {payload.get('adjusted_price_status')}",
        f"Suspicious rows: {payload.get('suspicious_row_count', 0)}",
        f"Suspicious rebalance dates: {len(payload.get('suspicious_rebalance_dates', []))}",
        "",
        "|symbol|status|rows|suspicious rows|first date|last date|",
        "|---|---|---:|---:|---|---|",
    ]
    for row in payload.get("symbols", []):
        lines.append(
            "|{symbol}|{status}|{rows}|{suspicious}|{first}|{last}|".format(
                symbol=row.get("symbol"),
                status=row.get("adjusted_status"),
                rows=row.get("row_count"),
                suspicious=row.get("suspicious_row_count"),
                first=row.get("first_date") or "",
                last=row.get("last_date") or "",
            )
        )
    lines.extend(["", "## Red Flags", ""])
    lines.extend(f"- {flag}" for flag in payload.get("red_flags", []))
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _clean_replay_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Clean Data Replay",
        "",
        NOTICE,
        "",
        f"Excluded rebalance dates: {payload.get('excluded_rebalance_date_count', 0)}",
        "",
        "|candidate|raw return|clean return|delta|clean benchmark-relative|verdict|",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload.get("candidates", {}).values():
        lines.append(
            "|{name}|{raw}|{clean}|{delta}|{relative}|{verdict}|".format(
                name=row.get("candidate_name"),
                raw=_fmt(row.get("raw_canonical_return")),
                clean=_fmt(row.get("clean_canonical_return")),
                delta=_fmt(row.get("return_delta_clean_vs_raw")),
                relative=row.get("clean_benchmark_relative_pass"),
                verdict=row.get("clean_data_verdict"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _independent_period_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Independent Period Validation",
        "",
        NOTICE,
        "",
        "Independent canonical periods: "
        f"{payload.get('independent_canonical_period_count', 0)}",
        "Minimum required: "
        f"{payload.get('minimum_independent_periods', 0)}",
        f"Gate passed: {payload.get('gate', {}).get('passed', False)}",
        "",
        "|candidate|independent periods|diagnostic rows|period-grid only rows|passes|",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload.get("candidate_periods", {}).values():
        lines.append(
            "|{name}|{count}|{diagnostic}|{extra}|{passes}|".format(
                name=row.get("candidate_name"),
                count=row.get("independent_period_count"),
                diagnostic=row.get("diagnostic_period_grid_count"),
                extra=row.get("period_grid_only_rows"),
                passes=row.get("passes_minimum"),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)
