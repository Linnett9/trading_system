from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.research.ml.allocation.allocation_optimizer_types import OPTIMIZER_NOTICE, OptimizerPaths


def write_optimizer_reports(
    output_dir: Path,
    report: dict[str, Any],
) -> OptimizerPaths:
    paths = OptimizerPaths(
        candidates_csv=output_dir / "allocation_optimizer_candidates.csv",
        results_json=output_dir / "allocation_optimizer_results.json",
        report_markdown=output_dir / "allocation_optimizer_report.md",
        selected_exposure_path_csv=output_dir / "selected_optimizer_exposure_path.csv",
        selected_exposure_path_json=output_dir / "selected_optimizer_exposure_path.json",
    )
    payload = {
        "mode": "allocation_optimizer_research_only",
        **report,
        "optimizer_notice": OPTIMIZER_NOTICE,
        "automatic_promotion": False,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    paths.results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    candidates = list(payload.get("candidates", []))
    rows = [_csv_row(row) for row in candidates]
    fieldnames = list(rows[0]) if rows else [
        "candidate_id",
        "objective_rank",
        "outcome_rank",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with paths.candidates_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_selected_exposure_path(paths, payload)
    _write_markdown(paths.report_markdown, payload)
    return paths

def _write_selected_exposure_path(
    paths: OptimizerPaths,
    payload: dict[str, Any],
) -> None:
    rows = list(payload.get("selected_optimizer_exposure_path", []))
    path_payload = {
        "mode": "selected_optimizer_exposure_path_research_only",
        "objective_mode": payload.get("objective_mode"),
        "sampler_requested": payload.get("sampler_requested"),
        "sampler_used": payload.get("sampler_used"),
        "selected_policy": payload.get("selected_policy", {}),
        "row_count": len(rows),
        "rows": rows,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    paths.selected_exposure_path_json.write_text(
        json.dumps(path_payload, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "rebalance_date",
        "outcome_end_date",
        "source_row_count",
        "period_return",
        "exposure",
        "score",
        "predicted_forward_return",
        "predicted_future_drawdown",
        "predicted_future_volatility",
        "turnover",
        "transaction_cost_bps",
        "cost",
        "net_return",
        "equity",
        "drawdown",
        "selected_symbols",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with paths.selected_exposure_path_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                **{name: row.get(name) for name in fieldnames},
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
            }
            for row in rows
        )

def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Allocation Optimizer",
        "",
        OPTIMIZER_NOTICE,
        "",
        f"Sampler requested: {payload.get('sampler_requested', 'unknown')}",
        f"Sampler used: {payload.get('sampler_used', 'unknown')}",
        f"Optuna available: {payload.get('optuna_available', False)}",
        f"Fallback reason: {payload.get('fallback_reason') or 'none'}",
        f"Objective mode: {payload.get('objective_mode', 'diagnostic_period_grid_return')}",
        f"Candidates: {payload.get('candidate_count', 0)}",
        "",
    ]
    selected = payload.get("selected_policy")
    if selected:
        lines.extend([
            "## Selected Diagnostic Policy",
            "",
            f"Candidate: {selected.get('candidate_id')}",
            f"Selection objective: {selected.get('objective')}",
            f"Selected by robustness objective: {selected.get('selected_by_robustness_objective', False)}",
            f"Selected params: {selected.get('selected_params')}",
            f"Holdout return: {selected.get('frozen_holdout_metrics', {}).get('total_return')}",
            f"Holdout max drawdown: {selected.get('frozen_holdout_metrics', {}).get('max_drawdown')}",
            "",
        ])
    if payload.get("skip_reason"):
        lines.extend([f"Skipped: {payload['skip_reason']}", ""])
    lines.append(
        "Research only. Trading impact: none. Production validated: false."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, value in row.items():
        output[name] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    output.update({
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    })
    return output
