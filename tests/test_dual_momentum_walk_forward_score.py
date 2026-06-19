from types import SimpleNamespace

from core.research.dual_momentum_experiments import (
    walk_forward_candidate_hard_filter,
)
from core.research.dual_momentum_scoring import (
    classify_walk_forward_fold_result,
    fold_gap_label,
)
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


def test_classify_walk_forward_fold_result_does_not_require_full_period_return():
    fold_result = SimpleNamespace(
        result=SimpleNamespace(
            max_drawdown=0.12,
            sharpe=1.20,
            total_return=0.45,
        ),
        annualized_turnover_percent=4.0,
        excess_return=0.15,
        excess_vs_equal_weight=0.08,
    )

    assert classify_walk_forward_fold_result(fold_result) == "fold pass"
    assert fold_gap_label(fold_result) == ""


def test_walk_forward_candidate_hard_filter_rejects_fragile_config():
    candidate_result = SimpleNamespace(
        result=SimpleNamespace(max_drawdown=0.20, sharpe=0.95),
        annualized_turnover_percent=7.5,
        excess_return=0.10,
        excess_vs_equal_weight=0.05,
        config={"max_position_weight": 0.30},
        annual_returns={2024: 0.10, 2025: -0.05, 2026: -0.02},
    )
    assert not walk_forward_candidate_hard_filter(
        candidate_result,
        {
            "walk_forward_max_in_sample_drawdown": 0.18,
            "walk_forward_max_in_sample_turnover": 6.0,
            "walk_forward_min_in_sample_sharpe": 1.0,
            "walk_forward_max_negative_years": 1,
            "walk_forward_max_position_weight": 0.28,
        },
    )
