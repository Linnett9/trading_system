from datetime import datetime, timedelta

from core.entities.candle import Candle
from core.research.multi_strategy_portfolio import (
    MultiStrategyPortfolioBacktester,
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


def test_multi_strategy_combines_portfolio_sleeves():
    prices = [10 + index for index in range(20)]
    candles = {
        "AAPL": make_candles("AAPL", prices),
        "MSFT": make_candles("MSFT", [20 + index for index in range(20)]),
        "SPY": make_candles("SPY", prices),
    }
    tester = MultiStrategyPortfolioBacktester(
        starting_equity=500,
        sleeves=[
            {
                "name": "dual_momentum",
                "weight": 0.70,
                "parameters": {
                    "top_n": 2,
                    "momentum_periods": [2],
                    "regime_sma_period": 3,
                    "asset_sma_period": 3,
                    "transaction_cost_bps": 0,
                },
            },
            {
                "name": "relative_strength",
                "weight": 0.30,
                "parameters": {
                    "top_n": 2,
                    "momentum_periods": [2],
                    "sma_period": 3,
                    "transaction_cost_bps": 0,
                },
            },
        ],
    )

    result = tester.run(candles)

    assert result.sleeves
    assert len(result.sleeves) == 2
    assert result.result.final_equity > result.result.starting_equity
    assert result.result.closed_trades >= 0
    assert result.config["sleeves"][0]["weight"] == 0.70
    assert result.diagnostics["annual"]
    assert "regime_label" in result.diagnostics["annual"]["2025"]


def test_multi_strategy_walk_forward_slice_rescales_starting_equity():
    prices = [10 + index for index in range(30)]
    candles = {
        "AAPL": make_candles("AAPL", prices),
        "SPY": make_candles("SPY", prices),
    }
    tester = MultiStrategyPortfolioBacktester(
        starting_equity=500,
        warmup_days=10,
        sleeves=[
            {
                "name": "dual_momentum",
                "weight": 1.0,
                "parameters": {
                    "top_n": 1,
                    "momentum_periods": [2],
                    "regime_sma_period": 3,
                    "asset_sma_period": 3,
                    "transaction_cost_bps": 0,
                },
            },
        ],
    )
    start_at = datetime(2025, 1, 16)
    end_at = datetime(2025, 1, 25)

    result = tester.run(candles, start_at=start_at, end_at=end_at)

    assert result.result.equity_curve[0].timestamp >= start_at
    assert result.result.equity_curve[-1].timestamp <= end_at
    assert result.result.starting_equity == 500
    assert result.result.equity_curve[0].equity == 500
