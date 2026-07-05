from core.research.dual_momentum.scoring_classification import classify_dual_momentum_result
from core.research.dual_momentum.scoring_constraints import production_gap_score
from core.research.dual_momentum.scoring_helpers import (
    _annual_consistency_penalty,
    _annual_return,
    _annual_return_values,
    _bull_capture,
    _negative_year_count,
)


def risk_regime_score(result):
    bull_capture = _bull_capture(result)
    annual_turnover = result.annualized_turnover_percent
    max_drawdown = result.result.max_drawdown

    turnover_penalty = (
        max(0, annual_turnover - 5.0) * 0.06
        + max(0, annual_turnover - 7.0) * 0.08
        + max(0, annual_turnover - 9.0) * 0.12
    )

    capture_penalty = max(
        0,
        0.60 - bull_capture,
    ) * 0.10

    drawdown_penalty = 0

    if max_drawdown > 0.18:
        drawdown_penalty += 0.15

    if max_drawdown > 0.20:
        drawdown_penalty += 0.20

    if max_drawdown > 0.25:
        drawdown_penalty += 0.35

    recent_penalty = 0

    if _annual_return(result, 2025) < 0:
        recent_penalty += 0.20

    if _annual_return(result, 2026) < 0:
        recent_penalty += 0.30

    if _annual_return(result, 2026) < -0.05:
        recent_penalty += 0.25

    return (
        result.excess_return * 0.35
        + result.excess_vs_equal_weight * 0.25
        + result.result.sharpe * 0.15
        + bull_capture * 0.05
        - max_drawdown * 0.30
        - annual_turnover * 0.02
        - result.cost_drag_percent * 0.10
        - turnover_penalty
        - capture_penalty
        - drawdown_penalty
        - recent_penalty
    )
def dual_momentum_quality_score(result):
    hard_penalty = 0

    if result.excess_vs_equal_weight <= 0:
        hard_penalty -= 1.0

    if result.excess_return <= 0:
        hard_penalty -= 0.50

    if result.result.max_drawdown > 0.20:
        hard_penalty -= 0.30

    if result.result.max_drawdown > 0.25:
        hard_penalty -= 0.75

    if result.annualized_turnover_percent > 5:
        hard_penalty -= 0.20

    if result.annualized_turnover_percent > 7:
        hard_penalty -= 0.35

    bull_capture = _bull_capture(result)

    if result.benchmark_return > 0 and bull_capture < 0.60:
        hard_penalty -= 0.35

    annual_values = _annual_return_values(result)
    negative_years = _negative_year_count(result)
    consistency_penalty = _annual_consistency_penalty(result)

    recent_penalty = 0

    if _annual_return(result, 2025) < 0:
        recent_penalty += 0.20

    if _annual_return(result, 2026) < 0:
        recent_penalty += 0.35

    if _annual_return(result, 2026) < -0.05:
        recent_penalty += 0.30

    recent_bonus = 0

    if _annual_return(result, 2025) > 0:
        recent_bonus += 0.05

    if _annual_return(result, 2026) > 0:
        recent_bonus += 0.10

    return (
        result.excess_vs_equal_weight * 0.35
        + result.excess_return * 0.25
        + result.result.sharpe * 0.15
        + result.calmar * 0.15
        + bull_capture * 0.08
        - result.result.max_drawdown * 0.20
        - consistency_penalty * 0.15
        - negative_years * 0.05
        - result.annualized_turnover_percent * 0.03
        - result.cost_drag_percent * 0.10
        - recent_penalty
        + recent_bonus
        + hard_penalty
    )
def paper_safe_dual_momentum_score(result):
    bull_capture = _bull_capture(result)

    total_return = result.result.total_return
    sharpe = result.result.sharpe
    max_drawdown = result.result.max_drawdown
    excess_return = result.excess_return
    excess_equal_weight = result.excess_vs_equal_weight
    calmar = result.calmar
    annual_turnover = result.annualized_turnover_percent
    cost_drag = result.cost_drag_percent

    recent_2025 = _annual_return(result, 2025)
    recent_2026 = _annual_return(result, 2026)

    score = (
        total_return * 0.15
        + excess_return * 0.25
        + excess_equal_weight * 0.10
        + sharpe * 0.25
        + calmar * 0.15
        + bull_capture * 0.05
        - max_drawdown * 0.80
        - annual_turnover * 0.04
        - cost_drag * 0.25
    )

    score -= max(0, annual_turnover - 4.0) * 0.10
    score -= max(0, annual_turnover - 5.0) * 0.12
    score -= max(0, annual_turnover - 7.0) * 0.16
    score -= max(0, annual_turnover - 9.0) * 0.20

    if excess_return <= 0:
        score -= 0.50

    if excess_equal_weight <= 0:
        score -= 0.35

    if sharpe < 1.0:
        score -= 0.20

    if max_drawdown > 0.18:
        score -= 0.20

    if max_drawdown > 0.20:
        score -= 0.35

    if max_drawdown > 0.25:
        score -= 0.75

    if annual_turnover > 5.00:
        score -= 0.35

    if annual_turnover > 7.00:
        score -= 0.90

    if annual_turnover > 9.00:
        score -= 0.90

    if annual_turnover > 10.00:
        score -= 0.75

    if result.benchmark_return > 0 and bull_capture < 0.60:
        score -= 0.35

    if result.benchmark_return > 0 and bull_capture < 0.40:
        score -= 0.50

    if recent_2025 < 0:
        score -= 0.25

    if recent_2026 < 0:
        score -= 0.40

    if recent_2026 < -0.05:
        score -= 0.40

    if recent_2025 > 0:
        score += 0.05

    if recent_2026 > 0:
        score += 0.10

    return score
def walk_forward_selection_score(result):
    bull_capture = _bull_capture(result)
    annual_turnover = result.annualized_turnover_percent
    max_drawdown = result.result.max_drawdown
    tag = classify_dual_momentum_result(result)

    score = (
        result.excess_return * 0.30
        + result.excess_vs_equal_weight * 0.20
        + result.result.sharpe * 0.25
        + result.calmar * 0.15
        + bull_capture * 0.05
        - max_drawdown * 0.85
        - result.cost_drag_percent * 0.20
        - production_gap_score(result) * 0.15
    )

    score -= max(0, annual_turnover - 5.0) * 0.06
    score -= max(0, annual_turnover - 6.0) * 0.08
    score -= max(0, annual_turnover - 7.0) * 0.10
    score -= max(0, annual_turnover - 8.0) * 0.12
    score -= max(0, annual_turnover - 9.0) * 0.16

    if max_drawdown > 0.18:
        score -= 0.45

    if max_drawdown > 0.20:
        score -= 1.00

    if tag.startswith("rejected"):
        score -= 1.25
    elif tag == "research candidate":
        score -= 0.35
    elif tag == "production candidate":
        score += 0.35
    elif tag == "near-production candidate":
        score += 0.20
    elif tag == "paper candidate":
        score += 0.10

    if result.excess_return <= 0:
        score -= 0.50

    if result.excess_vs_equal_weight <= 0:
        score -= 0.35

    if _annual_return(result, 2025) < 0:
        score -= 0.20

    if _annual_return(result, 2026) < 0:
        score -= 0.35

    return score
