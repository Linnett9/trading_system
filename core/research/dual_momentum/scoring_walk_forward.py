from core.research.dual_momentum.scoring_classification import classify_walk_forward_fold_result


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
