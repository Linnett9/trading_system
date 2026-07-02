from __future__ import annotations

from core.research.dual_momentum.reporting_format import format_percent


def dual_momentum_result_explanation(result):
    parts = []

    if result.excess_return > 0:
        parts.append(
            f"beat SPY by {format_percent(result.excess_return)}"
        )
    else:
        parts.append(
            f"trailed SPY by {format_percent(abs(result.excess_return))}"
        )

    if result.excess_vs_equal_weight > 0:
        parts.append(
            "beat equal-weight by "
            f"{format_percent(result.excess_vs_equal_weight)}"
        )
    else:
        parts.append(
            "trailed equal-weight by "
            f"{format_percent(abs(result.excess_vs_equal_weight))}"
        )

    if result.result.max_drawdown <= 0.20:
        parts.append("drawdown is within the 20% target")
    else:
        parts.append("drawdown is above the 20% target")

    bull_capture = (
        result.result.total_return / result.benchmark_return
        if result.benchmark_return > 0
        else 0
    )

    if result.benchmark_return > 0:
        parts.append(f"bull capture is {format_percent(bull_capture)}")

    if result.annualized_turnover_percent > 7:
        parts.append("turnover is high")
    else:
        parts.append("turnover is controlled")

    return "; ".join(parts) + "."


def dual_momentum_walk_forward_explanation(summary):
    if (
        summary["average_excess_return"] > 0
        and summary["consistency"] >= 0.66
    ):
        verdict = "robust so far"
    elif summary["average_excess_return"] > 0:
        verdict = "promising but inconsistent"
    else:
        verdict = "not robust yet"

    return (
        f"{verdict}; average excess is "
        f"{format_percent(summary['average_excess_return'])}, worst fold is "
        f"{format_percent(summary['worst_excess_return'])}, consistency is "
        f"{format_percent(summary['consistency'])}, and bull capture is "
        f"{format_percent(summary['average_bull_capture'])}."
    )
