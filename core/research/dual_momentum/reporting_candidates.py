import csv
from pathlib import Path

from core.research.dual_momentum.scoring import (
    classify_dual_momentum_result,
    dual_momentum_quality_score,
    paper_safe_dual_momentum_score,
)


def save_dual_momentum_filtered_walk_forward_candidates(
    folds,
    report_dir,
    filename="dual_momentum_walk_forward_candidates.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    fieldnames = [
        "fold",
        "train_start",
        "train_end",
        "rank",
        "name",
        "selector_mode",
        "filter_passed",
        "filter_fallback",
        "filter_reasons",
        "eligible_for_production_selector",
        "return",
        "excess_vs_benchmark",
        "excess_vs_equal_weight",
        "sharpe",
        "max_drawdown",
        "annualized_turnover_percent",
        "cost_drag_percent",
        "quality_score",
        "paper_safe_score",
        "tag",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for fold_index, item in enumerate(folds, start=1):
            fold = item["fold"]
            for rank, result in enumerate(item["training_results"], start=1):
                writer.writerow({
                    "fold": fold_index,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "rank": rank,
                    "name": result.config.get("experiment_name", "n/a"),
                    "selector_mode": getattr(
                        result,
                        "walk_forward_selector_mode",
                        "",
                    ),
                    "filter_passed": getattr(
                        result,
                        "walk_forward_filter_passed",
                        "",
                    ),
                    "filter_fallback": getattr(
                        result,
                        "walk_forward_filter_fallback",
                        "",
                    ),
                    "filter_reasons": ",".join(
                        getattr(
                            result,
                            "walk_forward_filter_reasons",
                            [],
                        )
                    ),
                    "eligible_for_production_selector": result.config.get(
                        "eligible_for_production_selector",
                        False,
                    ),
                    "return": result.result.total_return,
                    "excess_vs_benchmark": result.excess_return,
                    "excess_vs_equal_weight": result.excess_vs_equal_weight,
                    "sharpe": result.result.sharpe,
                    "max_drawdown": result.result.max_drawdown,
                    "annualized_turnover_percent": (
                        result.annualized_turnover_percent
                    ),
                    "cost_drag_percent": result.cost_drag_percent,
                    "quality_score": dual_momentum_quality_score(result),
                    "paper_safe_score": paper_safe_dual_momentum_score(result),
                    "tag": classify_dual_momentum_result(result),
                })

    return path
