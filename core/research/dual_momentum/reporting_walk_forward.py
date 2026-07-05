import json
from pathlib import Path

from core.research.dual_momentum.scoring import (
    dual_momentum_quality_score,
    dual_momentum_walk_forward_summary,
    paper_safe_dual_momentum_score,
)


def save_dual_momentum_walk_forward(
    results,
    report_dir,
    filename="dual_momentum_walk_forward.json",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    payload = {
        "summary": dual_momentum_walk_forward_summary(results),
        "folds": [],
    }

    for item in results:
        result = item["result"]
        training_result = item.get("training_result")

        bull_capture = (
            result.result.total_return / result.benchmark_return
            if result.benchmark_return > 0
            else None
        )

        payload["folds"].append({
            "fold": item["fold"],
            "selected_config": (
                training_result.config
                if training_result is not None
                else result.config
            ),
            "selected_name": (
                training_result.config.get("experiment_name")
                if training_result is not None
                else result.config.get("experiment_name")
            ),
            "train_return": (
                training_result.result.total_return
                if training_result is not None
                else None
            ),
            "train_quality_score": (
                dual_momentum_quality_score(training_result)
                if training_result is not None
                else None
            ),
            "train_paper_safe_score": (
                paper_safe_dual_momentum_score(training_result)
                if training_result is not None
                else None
            ),
            "return": result.result.total_return,
            "benchmark_return": result.benchmark_return,
            "equal_weight_return": result.equal_weight_return,
            "excess_return": result.excess_return,
            "excess_vs_equal_weight": result.excess_vs_equal_weight,
            "bull_capture_ratio": bull_capture,
            "sharpe": result.result.sharpe,
            "max_drawdown": result.result.max_drawdown,
            "cagr": result.cagr,
            "calmar": result.calmar,
            "annualized_turnover_percent": (
                result.annualized_turnover_percent
            ),
            "cost_drag_percent": result.cost_drag_percent,
            "closed_trades": result.result.closed_trades,
            "open_trades": result.result.open_trades,
        })

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return path
