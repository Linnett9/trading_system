from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.research.ml.audits.champion_baseline_audit_math import _fmt


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "baseline_name",
        "source_candidate_name",
        "semantic_type",
        "available",
        "target_exposure",
        "total_return",
        "continuous_total_return",
        "max_drawdown",
        "turnover",
        "costs",
        "cost_turnover_status",
        "is_exact_champion_replay",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Champion Baseline Audit",
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
        "## Semantics",
        "",
        f"- Current champion baseline exact replay: {payload['baseline_semantics']['current_champion_baseline_is_exact_champion_replay']}",
        f"- Why equal to always full: {payload['baseline_semantics']['why_champion_baseline_equals_always_full_exposure']}",
        "",
        "## Baselines",
        "",
        "|baseline|semantic type|available|target exposure|total return|continuous return|drawdown|turnover|costs|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["baseline_rows"]:
        lines.append(
            "|{name}|{kind}|{available}|{target}|{total}|{continuous}|"
            "{drawdown}|{turnover}|{costs}|".format(
                name=row.get("baseline_name"),
                kind=row.get("semantic_type"),
                available=row.get("available"),
                target=_fmt(row.get("target_exposure")),
                total=_fmt(row.get("total_return")),
                continuous=_fmt(row.get("continuous_total_return")),
                drawdown=_fmt(row.get("max_drawdown")),
                turnover=_fmt(row.get("turnover")),
                costs=_fmt(row.get("costs")),
            )
        )
    exact = payload["exact_champion_replay"]
    lines.extend([
        "",
        "## Exact Replay",
        "",
        f"- Available: {exact.get('available')}",
        f"- Reason: {exact.get('availability_reason') or 'ok'}",
        f"- Period-grid return: {_fmt(exact.get('period_grid_summary', {}).get('total_return'))}",
        f"- Continuous equity return: {_fmt(exact.get('continuous_equity_summary', {}).get('total_return'))}",
        "",
        "## Red Flags",
        "",
    ])
    lines.extend(f"- {flag}" for flag in payload.get("red_flags", []))
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
        "",
    ])
    return "\n".join(lines)
