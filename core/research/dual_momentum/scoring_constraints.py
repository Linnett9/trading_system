from core.research.dual_momentum.scoring_helpers import _annual_return


def production_constraint_failures(result):
    return [
        item["label"]
        for item in production_constraint_gap_details(result)
    ]
def production_constraint_gap_details(result):
    gaps = []

    if result.result.total_return < 1.10:
        gaps.append({
            "key": "return",
            "label": "return<110%",
            "gap": 1.10 - result.result.total_return,
        })

    if result.result.max_drawdown > 0.18:
        gaps.append({
            "key": "drawdown",
            "label": "drawdown>18%",
            "gap": result.result.max_drawdown - 0.18,
        })

    if result.result.sharpe < 1.0:
        gaps.append({
            "key": "sharpe",
            "label": "sharpe<1.0",
            "gap": 1.0 - result.result.sharpe,
        })

    if result.annualized_turnover_percent > 6:
        gaps.append({
            "key": "turnover",
            "label": "turnover>600%",
            "gap": result.annualized_turnover_percent - 6,
        })

    if result.cost_drag_percent > 0.015:
        gaps.append({
            "key": "cost",
            "label": "cost>1.5%",
            "gap": result.cost_drag_percent - 0.015,
        })

    if (
        result.excess_return <= 0.15
        and result.excess_vs_equal_weight <= 0.10
    ):
        gaps.append({
            "key": "excess",
            "label": "excess too low",
            "gap": min(
                0.15 - result.excess_return,
                0.10 - result.excess_vs_equal_weight,
            ),
        })

    annual_returns = getattr(result, "annual_returns", {}) or {}

    if 2025 in annual_returns and _annual_return(result, 2025) <= 0:
        gaps.append({
            "key": "2025",
            "label": "2025<=0",
            "gap": abs(_annual_return(result, 2025)),
        })

    if 2026 in annual_returns and _annual_return(result, 2026) <= 0:
        gaps.append({
            "key": "2026",
            "label": "2026<=0",
            "gap": abs(_annual_return(result, 2026)),
        })

    return gaps
def production_gap_score(result):
    score = 0

    for item in production_constraint_gap_details(result):
        key = item["key"]
        gap = item["gap"]

        if key == "turnover":
            score += gap * 0.20
        elif key == "drawdown":
            score += gap * 12.0
        elif key == "return":
            score += gap * 2.0
        elif key == "sharpe":
            score += gap * 0.8
        elif key == "cost":
            score += gap * 8.0
        elif key == "excess":
            score += gap * 1.5
        else:
            score += gap

    return score
def fold_constraint_gap_details(result):
    gaps = []

    if result.result.total_return <= 0:
        gaps.append({
            "key": "return",
            "label": "return<=0",
            "gap": abs(result.result.total_return),
        })

    if result.result.max_drawdown > 0.18:
        gaps.append({
            "key": "drawdown",
            "label": "drawdown>18%",
            "gap": result.result.max_drawdown - 0.18,
        })

    if result.result.sharpe < 1.0:
        gaps.append({
            "key": "sharpe",
            "label": "sharpe<1.0",
            "gap": 1.0 - result.result.sharpe,
        })

    if result.annualized_turnover_percent > 6.0:
        gaps.append({
            "key": "turnover",
            "label": "turnover>600%",
            "gap": result.annualized_turnover_percent - 6.0,
        })

    if result.excess_return <= 0:
        gaps.append({
            "key": "benchmark",
            "label": "excess<=0",
            "gap": abs(result.excess_return),
        })

    if result.excess_vs_equal_weight <= 0:
        gaps.append({
            "key": "equal-weight",
            "label": "equal-weight<=0",
            "gap": abs(result.excess_vs_equal_weight),
        })

    return gaps
def fold_gap_label(result):
    return " ".join(item["label"] for item in fold_constraint_gap_details(result))
