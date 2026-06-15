import pytest

from core.research.performance_metrics import (
    cagr,
    calmar_ratio,
    composite_score,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    trade_count_quality,
)


def test_total_return_calculation():
    assert total_return(100, 125) == 0.25


def test_cagr_calculation():
    assert cagr(100, 121, 365.25) == pytest.approx(0.21)


def test_sharpe_ratio_positive_for_positive_returns():
    assert sharpe_ratio([0.01, 0.02, -0.005, 0.015]) > 0


def test_max_drawdown_calculation():
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)


def test_profit_factor_calculation():
    assert profit_factor([10, -5, 15, -5]) == 2.5


def test_calmar_ratio_calculation():
    assert calmar_ratio(0.20, 0.10) == 2


def test_trade_count_quality_caps_at_one():
    assert trade_count_quality(30, 20) == 1
    assert trade_count_quality(10, 20) == 0.5


def test_composite_score_rewards_quality_inputs():
    weak = composite_score(
        excess_return=-0.1,
        sharpe=0,
        max_drawdown_value=0.2,
        profit_factor_value=0.5,
        closed_trades=1,
        target_trades=20,
    )
    strong = composite_score(
        excess_return=0.1,
        sharpe=1.5,
        max_drawdown_value=0.05,
        profit_factor_value=2,
        closed_trades=25,
        target_trades=20,
    )

    assert strong > weak
