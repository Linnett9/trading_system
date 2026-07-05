from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.research.ml.independent_period_expansion_audit_math import _number


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate",
        "setting",
        "spacing",
        "minimum_gap_days",
        "leakage_safe",
        "overlap_risk",
        "independent_period_count",
        "overlap_skipped_period_count",
        "adjusted_coverage_ratio",
        "valid_adjusted_period_count",
        "invalid_adjusted_period_count",
        "canonical_return",
        "anomaly_adjusted_return",
        "max_drawdown",
        "top_5_positive_return_share",
        "benchmark_return",
        "benchmark_excess_return",
        "promotion_gate_status",
        "failed_gates",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: ";".join(row.get(name, []))
                if isinstance(row.get(name), list)
                else row.get(name)
                for name in fieldnames
            })
def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Independent Adjusted Period Expansion Audit",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        f"Minimum independent periods: {payload.get('minimum_independent_periods')}",
        f"Promotion thresholds changed: {payload.get('promotion_thresholds_changed')}",
        "",
        "## No Selected Symbols",
        "",
        (
            f"Rows: {payload.get('no_selected_symbol_summary', {}).get('row_count')} | "
            f"Verdict: {payload.get('no_selected_symbol_summary', {}).get('verdict')}"
        ),
        "",
        "## Expansion Settings",
        "",
        "|candidate|setting|periods|coverage|return|anomaly-adjusted|drawdown|excess vs SPY|status|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "|{candidate}|{setting}|{periods}|{coverage}|{ret}|{clean}|{dd}|{excess}|{status}|".format(
                candidate=row.get("candidate"),
                setting=row.get("setting"),
                periods=row.get("independent_period_count"),
                coverage=_fmt(row.get("adjusted_coverage_ratio")),
                ret=_fmt(row.get("canonical_return")),
                clean=_fmt(row.get("anomaly_adjusted_return")),
                dd=_fmt(row.get("max_drawdown")),
                excess=_fmt(row.get("benchmark_excess_return")),
                status=row.get("promotion_gate_status"),
            )
        )
    return "\n".join(lines) + "\n"
def _fmt(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.4f}"
