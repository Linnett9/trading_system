from __future__ import annotations

from core.research.dual_momentum.scoring import (
    production_constraint_gap_details,
)
from core.research.dual_momentum.reporting_format import format_percent


def production_gap_label(result):
    details = production_constraint_gap_details(result)
    if not details:
        return "ready"

    labels = []
    for item in details:
        label = item["label"]
        gap = item["gap"]
        if item["key"] in {"turnover", "drawdown", "return", "cost", "excess"}:
            labels.append(f"{label} +{format_percent(gap)}")
        elif item["key"] == "sharpe":
            labels.append(f"{label} +{gap:.2f}")
        else:
            labels.append(label)

    return ", ".join(labels)


def walk_forward_readiness_label(summary):
    if (
        summary["worst_excess_return"] >= 0
        and summary["consistency"] >= (2 / 3)
        and summary["average_drawdown"] <= 0.18
        and summary["average_turnover"] <= 6.0
        and summary.get("rejected_fold_count", 0) == 0
    ):
        return "production-ready"

    if (
        summary["average_excess_return"] > 0
        and summary["consistency"] >= 0.5
        and summary["average_drawdown"] <= 0.20
        and summary["average_turnover"] <= 10.0
        and summary.get("rejected_fold_count", 0) <= 1
    ):
        return "paper-ready"

    return "research-only"


def champion_delta_label(result, champion):
    if champion is None:
        return "n/a"

    return_delta = result.result.total_return - champion.result.total_return
    drawdown_delta = (
        result.result.max_drawdown - champion.result.max_drawdown
    )
    turnover_delta = (
        result.annualized_turnover_percent
        - champion.annualized_turnover_percent
    )

    return (
        f"ret {format_percent(return_delta)} "
        f"dd {format_percent(drawdown_delta)} "
        f"turn {format_percent(turnover_delta)}"
    )
