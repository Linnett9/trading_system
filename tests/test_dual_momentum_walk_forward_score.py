from types import SimpleNamespace

from main import dual_momentum_walk_forward_summary


def make_fold(
    excess,
    equal_weight_excess,
    drawdown,
    total_return=None,
    benchmark_return=0.10,
    turnover=3.0,
):
    return {
        "result": SimpleNamespace(
            excess_return=excess,
            excess_vs_equal_weight=equal_weight_excess,
            benchmark_return=benchmark_return,
            annualized_turnover_percent=turnover,
            result=SimpleNamespace(
                max_drawdown=drawdown,
                total_return=(
                    total_return
                    if total_return is not None
                    else benchmark_return + excess
                ),
            ),
        )
    }


def test_walk_forward_summary_penalizes_bad_worst_fold():
    steady = [
        make_fold(0.02, 0.01, 0.10),
        make_fold(0.02, 0.01, 0.10),
        make_fold(0.02, 0.01, 0.10),
    ]
    fragile = [
        make_fold(0.12, 0.08, 0.10),
        make_fold(0.12, 0.08, 0.10),
        make_fold(-0.20, -0.10, 0.25),
    ]

    steady_summary = dual_momentum_walk_forward_summary(steady)
    fragile_summary = dual_momentum_walk_forward_summary(fragile)

    assert steady_summary["consistency"] == 1
    assert fragile_summary["worst_excess_return"] < 0
    assert steady_summary["score"] > fragile_summary["score"]


def test_walk_forward_summary_penalizes_low_bull_capture_and_turnover():
    efficient = [
        make_fold(0.02, 0.01, 0.10, total_return=0.12, turnover=3.0),
        make_fold(0.01, 0.01, 0.10, total_return=0.11, turnover=3.0),
    ]
    low_capture_high_turnover = [
        make_fold(0.02, 0.01, 0.10, total_return=0.05, turnover=10.0),
        make_fold(0.01, 0.01, 0.10, total_return=0.04, turnover=10.0),
    ]

    efficient_summary = dual_momentum_walk_forward_summary(efficient)
    weak_summary = dual_momentum_walk_forward_summary(low_capture_high_turnover)

    assert efficient_summary["average_bull_capture"] > 1
    assert weak_summary["average_bull_capture"] < 0.60
    assert efficient_summary["score"] > weak_summary["score"]
