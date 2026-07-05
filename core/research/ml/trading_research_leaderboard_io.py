from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.research.ml.trading_research_leaderboard_math import _format
from core.research.ml.trading_research_leaderboard_types import NOTICE


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "entity_name",
        "entity_type",
        "source",
        "selection_role",
        "total_return",
        "diagnostic_period_grid_return",
        "canonical_continuous_return",
        "canonical_non_overlap_return",
        "canonical_tradable_total_return",
        "paper_tradable_equity_return",
        "anomaly_adjusted_return",
        "anomaly_adjusted_canonical_return",
        "anomaly_dependency_ratio",
        "robustness_adjusted_score",
        "selected_by_robustness_objective",
        "optimizer_objective_mode",
        "benchmark_relative_pass",
        "tradability_validation_pass",
        "promotion_candidate_status",
        "profit_concentration_ratio",
        "turnover_after_hysteresis",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "turnover",
        "estimated_transaction_costs",
        "balanced_accuracy",
        "walk_forward_balanced_accuracy",
        "brier_score",
        "expected_calibration_error",
        "classification_metrics_role",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row.get(name) for name in fieldnames} for row in rows
        )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trading Research Leaderboard",
        "",
        "Canonical continuous returns determine rank when available. Old period-grid returns are diagnostic only. Classification metrics are diagnostics only.",
        "",
        "|rank|candidate|type|canonical return|diagnostic period-grid return|anomaly-adjusted return|max drawdown|Sharpe|turnover|costs|promotion status|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            "|{rank}|{entity_name}|{entity_type}|{canonical}|{diagnostic}|"
            "{anomaly}|{max_drawdown}|{sharpe}|{turnover}|{costs}|{status}|".format(
                rank=row["rank"],
                entity_name=row["entity_name"],
                entity_type=row["entity_type"],
                canonical=_format(row.get("canonical_continuous_return")),
                diagnostic=_format(row.get("diagnostic_period_grid_return")),
                anomaly=_format(row.get("anomaly_adjusted_return")),
                max_drawdown=_format(row.get("max_drawdown")),
                sharpe=_format(row.get("sharpe")),
                turnover=_format(row.get("turnover")),
                costs=_format(row.get("estimated_transaction_costs")),
                status=row.get("promotion_candidate_status") or "",
            )
        )

    lines.extend([
        "",
        "## Classification Diagnostics",
        "",
        "|model|role|balanced accuracy|walk-forward balanced accuracy|Brier|ECE|",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in payload["classification_diagnostics"]:
        lines.append(
            "|{model}|{role}|{balanced}|{walk_forward}|{brier}|{ece}|".format(
                model=row.get("model") or "",
                role=row.get("selection_role") or "",
                balanced=_format(row.get("holdout_balanced_accuracy")),
                walk_forward=_format(
                    row.get("walk_forward_balanced_accuracy")
                ),
                brier=_format(row.get("brier_score")),
                ece=_format(row.get("expected_calibration_error")),
            )
        )

    lines.extend([
        "",
        "## Meta Auxiliary Forecast Diagnostics",
        "",
        "|target|available|MAE|RMSE|Pearson|Spearman|directional accuracy|",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for target, metrics in payload["meta_auxiliary_forecast_metrics"].items():
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "|{target}|{available}|{mae}|{rmse}|{pearson}|{spearman}|{directional}|".format(
                target=target,
                available=metrics.get("available", False),
                mae=_format(metrics.get("mae")),
                rmse=_format(metrics.get("rmse")),
                pearson=_format(metrics.get("pearson_correlation")),
                spearman=_format(metrics.get("spearman_correlation")),
                directional=_format(metrics.get("directional_accuracy")),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)
