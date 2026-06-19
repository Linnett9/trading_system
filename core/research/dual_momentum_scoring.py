def _bull_capture(result):
    if result.benchmark_return > 0:
        return result.result.total_return / result.benchmark_return

    return 0


def _annual_return(result, year):
    return (getattr(result, "annual_returns", {}) or {}).get(year, 0)


def _annual_return_values(result):
    return list((getattr(result, "annual_returns", {}) or {}).values())


def _annual_consistency_penalty(result):
    annual_values = _annual_return_values(result)

    if not annual_values:
        return 0

    annual_mean = sum(annual_values) / len(annual_values)

    annual_variance = (
        sum(
            (value - annual_mean) ** 2
            for value in annual_values
        )
        / len(annual_values)
    )

    return annual_variance ** 0.5


def _negative_year_count(result):
    return sum(
        1
        for value in _annual_return_values(result)
        if value < 0
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


def classify_walk_forward_fold_result(result):
    if result.result.total_return <= 0:
        return "rejected: return"

    if result.result.max_drawdown > 0.18:
        return "rejected: drawdown"

    if result.annualized_turnover_percent > 6.0:
        return "rejected: turnover"

    if result.result.sharpe < 1.0:
        return "rejected: sharpe"

    if result.excess_return <= 0:
        return "rejected: benchmark"

    if result.excess_vs_equal_weight <= 0:
        return "rejected: equal-weight"

    return "fold pass"


def _paper_candidate_floor(result):
    return (
        result.result.total_return >= 1.10
        and result.result.max_drawdown <= 0.20
        and result.result.sharpe >= 1.0
        and result.annualized_turnover_percent <= 10
    )


def classify_dual_momentum_result(result):
    if result.result.max_drawdown > 0.25:
        return "rejected: drawdown"

    if result.result.max_drawdown > 0.20:
        return "rejected: drawdown"

    if _annual_return(result, 2026) < -0.05:
        return "rejected: bad 2026"

    if result.annualized_turnover_percent > 10:
        return "rejected: turnover"

    if result.excess_return <= 0:
        return "rejected: benchmark"

    if result.excess_vs_equal_weight <= 0:
        return "rejected: equal-weight"

    production_failures = production_constraint_failures(result)
    if not production_failures:
        return "production candidate"

    if _paper_candidate_floor(result) and len(production_failures) == 1:
        return "near-production candidate"

    if _paper_candidate_floor(result):
        return "paper candidate"

    return "research candidate"


def dual_momentum_walk_forward_summary(results):
    if not results:
        return {
            "average_excess_return": 0,
            "worst_excess_return": 0,
            "average_excess_vs_equal_weight": 0,
            "average_drawdown": 0,
            "average_turnover": 0,
            "average_bull_capture": 0,
            "worst_bull_capture": 0,
            "rejected_fold_count": 0,
            "accepted_fold_count": 0,
            "consistency": 0,
            "dispersion": 0,
            "score": 0,
        }

    excess_returns = [
        item["result"].excess_return
        for item in results
    ]

    equal_weight_excess = [
        item["result"].excess_vs_equal_weight
        for item in results
    ]

    drawdowns = [
        item["result"].result.max_drawdown
        for item in results
    ]

    turnovers = [
        item["result"].annualized_turnover_percent
        for item in results
    ]

    bull_capture_values = [
        item["result"].result.total_return / item["result"].benchmark_return
        for item in results
        if item["result"].benchmark_return > 0
    ]
    fold_tags = [
        classify_walk_forward_fold_result(item["result"])
        for item in results
    ]
    rejected_fold_count = sum(
        1
        for tag in fold_tags
        if tag.startswith("rejected")
    )
    accepted_fold_count = len(fold_tags) - rejected_fold_count

    avg_excess = sum(excess_returns) / len(excess_returns)

    avg_equal_weight_excess = (
        sum(equal_weight_excess) / len(equal_weight_excess)
    )

    avg_drawdown = sum(drawdowns) / len(drawdowns)
    avg_turnover = sum(turnovers) / len(turnovers)

    avg_bull_capture = (
        sum(bull_capture_values) / len(bull_capture_values)
        if bull_capture_values
        else 0
    )

    worst_bull_capture = (
        min(bull_capture_values)
        if bull_capture_values
        else 0
    )

    worst_excess = min(excess_returns)

    consistency = (
        sum(1 for value in excess_returns if value > 0)
        / len(excess_returns)
    )

    dispersion = (
        sum((value - avg_excess) ** 2 for value in excess_returns)
        / len(excess_returns)
    ) ** 0.5

    turnover_penalty = max(0, avg_turnover - 5.0) * 0.05
    high_turnover_penalty = max(0, avg_turnover - 7.0) * 0.05

    capture_shortfall = max(0, 0.60 - avg_bull_capture)
    worst_capture_shortfall = max(0, 0.40 - worst_bull_capture)

    worst_fold_penalty = 0

    if worst_excess < 0:
        worst_fold_penalty += abs(worst_excess) * 0.75

    if worst_excess < -0.10:
        worst_fold_penalty += 0.40

    if worst_excess < -0.15:
        worst_fold_penalty += 0.35

    rejected_fold_penalty = rejected_fold_count * 0.35

    drawdown_penalty = 0

    if avg_drawdown > 0.18:
        drawdown_penalty += 0.10

    if avg_drawdown > 0.20:
        drawdown_penalty += 0.20

    score = (
        avg_excess * 0.30
        + worst_excess * 0.45
        + avg_equal_weight_excess * 0.10
        + consistency * 0.15
        + avg_bull_capture * 0.08
        - avg_drawdown * 0.20
        - dispersion * 0.10
        - turnover_penalty
        - high_turnover_penalty
        - capture_shortfall * 0.10
        - worst_capture_shortfall * 0.10
        - worst_fold_penalty
        - rejected_fold_penalty
        - drawdown_penalty
    )

    return {
        "average_excess_return": avg_excess,
        "worst_excess_return": worst_excess,
        "average_excess_vs_equal_weight": avg_equal_weight_excess,
        "average_drawdown": avg_drawdown,
        "average_turnover": avg_turnover,
        "average_bull_capture": avg_bull_capture,
        "worst_bull_capture": worst_bull_capture,
        "rejected_fold_count": rejected_fold_count,
        "accepted_fold_count": accepted_fold_count,
        "consistency": consistency,
        "dispersion": dispersion,
        "score": score,
    }
