from __future__ import annotations

import csv
import json
from typing import Any

from core.research.ml.audits.benchmark_relative_validation_types import COST_STRESS_BPS
from core.research.ml.audits.benchmark_relative_validation_math import _fmt


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_name", "available", "canonical_non_overlap_return",
        "anomaly_adjusted_return", "anomaly_dependency_ratio", "max_drawdown",
        "sharpe", "sortino", "turnover",
        *[f"cost_stressed_return_{bps}bps" for bps in COST_STRESS_BPS],
        "top_1_date_profit_share", "top_5_date_profit_share",
        "top_1_symbol_profit_share", "excess_return_vs_spy",
        "excess_return_vs_qqq", "excess_return_vs_equal_weight",
        "benchmark_relative_pass", "tradability_validation_pass",
        "promotion_candidate_status", "failed_gates", "research_only",
        "trading_impact", "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({
            **{name: row.get(name) for name in fieldnames},
            "failed_gates": json.dumps(row.get("failed_gates", [])),
        } for row in rows)

def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark-Relative and Tradability Validation", "",
        "Research only. Trading impact: none. Production validated: false.", "",
        "|candidate|canonical return|anomaly-adjusted|drawdown|Sharpe|turnover|top 5 dates|excess SPY|excess QQQ|excess equal-weight|status|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("candidates", []):
        lines.append(
            f"|{row['candidate_name']}|{_fmt(row.get('canonical_non_overlap_return'))}|"
            f"{_fmt(row.get('anomaly_adjusted_return'))}|{_fmt(row.get('max_drawdown'))}|"
            f"{_fmt(row.get('sharpe'))}|{_fmt(row.get('turnover'))}|"
            f"{_fmt(row.get('top_5_date_profit_share'))}|{_fmt(row.get('excess_return_vs_spy'))}|"
            f"{_fmt(row.get('excess_return_vs_qqq'))}|{_fmt(row.get('excess_return_vs_equal_weight'))}|"
            f"{row.get('promotion_candidate_status')}|"
        )
    return "\n".join(lines) + "\n"

def _promotion_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Promotion Readiness", "",
        "Research only. No automatic promotion or trading impact.", "",
        f"Any candidate passes all gates: {payload.get('any_candidate_passes', False)}", "",
    ]
    for row in payload.get("candidates", []):
        lines.extend([
            f"## {row['candidate_name']}", "",
            f"Status: {row.get('promotion_candidate_status')}",
            f"Failed gates: {', '.join(row.get('failed_gates', [])) or 'none'}", "",
        ])
    return "\n".join(lines) + "\n"
