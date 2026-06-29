from datetime import datetime, timedelta

from core.entities.candle import Candle
from core.research.relative_strength.portfolio import (
    RelativeStrengthPortfolioBacktester,
)


def make_candles(symbol, prices):
    start = datetime(2025, 1, 1)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + timedelta(days=index),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000,
        )
        for index, price in enumerate(prices)
    ]


def test_relative_strength_selects_strong_symbol():
    tester = RelativeStrengthPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        sma_period=3,
        target_exposure=1.0,
        benchmark_symbol="SPY",
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 20, 19, 18, 17, 16, 15]),
        "SPY": make_candles("SPY", [10, 10, 10, 10, 11, 11, 11]),
    })

    assert result.selections
    assert result.selections[0].symbols == ["AAPL"]
    assert result.result.final_equity > result.result.starting_equity
    assert result.result.open_trades == 1
    assert result.result.capital_utilization.average_exposure_percent > 0
    assert result.equal_weight_return > 0
    assert result.excess_vs_equal_weight == (
        result.result.total_return - result.equal_weight_return
    )
    assert result.turnover_percent > 0
    assert result.rebalance_count == len(result.selections)


def test_relative_strength_goes_to_cash_when_no_symbol_passes_trend_filter():
    tester = RelativeStrengthPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        sma_period=3,
        target_exposure=1.0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 9, 8, 7, 6, 5, 4]),
        "MSFT": make_candles("MSFT", [20, 19, 18, 17, 16, 15, 14]),
    })

    assert result.selections[0].symbols == []
    assert result.result.final_equity == result.result.starting_equity
    assert result.result.open_trades == 0


def test_relative_strength_applies_transaction_costs():
    free_tester = RelativeStrengthPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        sma_period=3,
        target_exposure=1.0,
        transaction_cost_bps=0,
    )
    cost_tester = RelativeStrengthPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        sma_period=3,
        target_exposure=1.0,
        transaction_cost_bps=10,
    )
    candles = {
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 20, 19, 18, 17, 16, 15]),
    }

    free_result = free_tester.run(candles)
    cost_result = cost_tester.run(candles)

    assert cost_result.estimated_cost > 0
    assert cost_result.result.final_equity < free_result.result.final_equity
