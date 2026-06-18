from datetime import datetime, timedelta, timezone

from core.entities.candle import Candle
from core.research.dual_momentum_portfolio import (
    DualMomentumPortfolioBacktester,
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


def make_aware_candles(symbol, prices):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
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


def test_dual_momentum_selects_top_positive_momentum_assets():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2, 4],
        regime_sma_period=3,
        asset_sma_period=3,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 23, 24, 25, 26]),
        "TSLA": make_candles("TSLA", [30, 29, 28, 27, 26, 25, 24]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    assert result.selections
    assert result.selections[0].risk_on
    assert result.selections[0].symbols == ["AAPL", "MSFT"]
    assert result.result.open_trades == 2
    assert result.result.final_equity > result.result.starting_equity
    assert result.turnover_percent > 0


def test_dual_momentum_can_hold_all_positive_momentum_assets():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        selection_mode="all_positive",
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 23, 24, 25, 26]),
        "TSLA": make_candles("TSLA", [30, 29, 28, 27, 26, 25, 24]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    assert result.selections[0].symbols == ["AAPL", "MSFT"]
    assert result.result.open_trades == 2
    assert result.config["selection_mode"] == "all_positive"


def test_dual_momentum_all_positive_can_filter_weak_scores_and_cap_assets():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        selection_mode="all_positive",
        min_selection_score=0.10,
        max_selected_assets=2,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 16, 20, 25]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 24, 28, 32, 36]),
        "NVDA": make_candles("NVDA", [30, 31, 32, 33, 36, 39, 43]),
        "TSLA": make_candles("TSLA", [40, 40, 40, 40, 40, 40, 41]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    assert len(result.selections[0].symbols) == 2
    assert "TSLA" not in result.selections[0].symbols
    assert result.config["min_selection_score"] == 0.10
    assert result.config["max_selected_assets"] == 2


def test_dual_momentum_inverse_volatility_weights_lower_volatility_more():
    tester = DualMomentumPortfolioBacktester(
        weighting="inverse_volatility",
        max_position_weight=0.60,
        weight_volatility_lookback=4,
    )
    candles = {
        "LOW": make_candles("LOW", [10, 10.1, 10.2, 10.3, 10.4]),
        "HIGH": make_candles("HIGH", [10, 13, 9, 14, 8]),
        "SPY": make_candles("SPY", [10, 10.1, 10.2, 10.3, 10.4]),
    }
    prices_by_symbol = tester._prices_by_symbol(candles)
    timestamp = sorted(prices_by_symbol["LOW"])[-1]

    weights = tester._target_weights(
        selected=["LOW", "HIGH"],
        timestamp=timestamp,
        prices_by_symbol=prices_by_symbol,
    )

    assert weights["LOW"] > weights["HIGH"]
    assert weights["LOW"] <= 0.60


def test_dual_momentum_moves_to_cash_when_spy_below_sma():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2, 4],
        regime_sma_period=3,
        asset_sma_period=3,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 23, 24, 25, 26]),
        "SPY": make_candles("SPY", [16, 15, 14, 13, 12, 11, 10]),
    })

    assert result.selections[0].risk_on is False
    assert result.selections[0].symbols == []
    assert result.result.open_trades == 0
    assert result.result.final_equity == result.result.starting_equity


def test_dual_momentum_can_rotate_to_defensive_assets_when_risk_off():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        risk_off_symbols=["BIL", "GLD"],
        risk_off_top_n=1,
        risk_off_momentum_periods=[2],
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "BIL": make_candles("BIL", [20, 21, 22, 23, 24, 25, 26]),
        "GLD": make_candles("GLD", [30, 29, 28, 27, 26, 25, 24]),
        "SPY": make_candles("SPY", [16, 15, 14, 13, 12, 11, 10]),
    })

    assert result.selections[0].risk_on is False
    assert result.selections[0].symbols == ["BIL"]
    assert result.result.open_trades == 1


def test_dual_momentum_fast_reentry_can_reenter_before_slow_regime():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=5,
        asset_sma_period=3,
        fast_reentry_enabled=True,
        fast_reentry_sma_period=2,
        mixed_risk_exposure=0.50,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "SPY": make_candles("SPY", [20, 19, 18, 17, 16, 17, 18]),
    })

    assert result.selections[0].risk_on is False
    assert result.selections[0].symbols == ["AAPL"]
    assert 0 < result.result.capital_utilization.average_exposure_percent < 1


def test_dual_momentum_fast_reentry_can_use_secondary_symbol():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=5,
        asset_sma_period=3,
        fast_reentry_enabled=True,
        fast_reentry_symbols=["SPY", "QQQ"],
        fast_reentry_sma_period=2,
        mixed_risk_exposure=0.50,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "QQQ": make_candles("QQQ", [10, 9, 8, 7, 8, 9, 10]),
        "SPY": make_candles("SPY", [20, 19, 18, 17, 16, 15, 14]),
    })

    assert result.selections[0].regime_label == "fast-reentry"
    assert result.selections[0].symbols


def test_dual_momentum_scaled_risk_off_keeps_partial_risk_exposure():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=5,
        asset_sma_period=3,
        risk_regime_mode="scaled",
        risk_off_risk_exposure=0.25,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "SPY": make_candles("SPY", [20, 19, 18, 17, 16, 17, 18]),
    })

    assert result.selections[0].risk_on is False
    assert result.selections[0].symbols == ["AAPL"]
    assert result.result.capital_utilization.average_exposure_percent <= 0.30


def test_dual_momentum_can_add_benchmark_fallback_when_few_assets_qualify():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        fallback_symbols=["SPY", "QQQ"],
        fallback_allocation=0.25,
        fallback_min_risk_assets=2,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 19, 18, 17, 16, 15, 14]),
        "QQQ": make_candles("QQQ", [30, 31, 32, 33, 34, 35, 36]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    selection = result.selections[0]

    assert selection.regime_label == "risk-on"
    assert selection.exposure_target == 1.0
    assert len(selection.fallback_symbols) == 1
    assert selection.fallback_symbols[0] in {"SPY", "QQQ"}
    assert selection.symbols == ["AAPL", selection.fallback_symbols[0]]


def test_dual_momentum_can_add_benchmark_sleeve_during_risk_on():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        benchmark_sleeve_symbols=["SPY", "QQQ"],
        benchmark_sleeve_allocation=0.20,
        benchmark_sleeve_momentum_periods=[2],
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "QQQ": make_candles("QQQ", [30, 31, 32, 33, 34, 35, 36]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    weights = result.selections[0].target_weights

    assert result.selections[0].symbols == ["AAPL", "SPY"]
    assert round(weights["AAPL"], 2) == 0.80
    assert round(weights["SPY"], 2) == 0.20


def test_dual_momentum_decay_exit_sells_weakening_position_between_rebalances():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        decay_exit_enabled=True,
        decay_momentum_period=2,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 12, 14, 16, 15, 14, 13]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    assert result.result.closed_trades == 1
    assert result.result.open_trades == 0


def test_dual_momentum_chop_filter_reduces_risk_on_exposure():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        chop_filter_enabled=True,
        chop_lookback=2,
        min_chop_momentum=0.10,
        chop_risk_exposure=0.40,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "SPY": make_candles("SPY", [10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6]),
    })

    assert result.selections[0].regime_label == "chop-filter"
    assert result.selections[0].chop_filter_active
    assert result.selections[0].exposure_target == 0.40


def test_dual_momentum_quality_filter_requires_clean_short_term_momentum():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        quality_filter_enabled=True,
        quality_momentum_period=2,
        quality_sma_period=3,
        quality_require_momentum_improving=True,
        transaction_cost_bps=0,
    )

    candles = {
        "AAPL": make_candles("AAPL", [10, 10.5, 11, 12, 14, 17, 21]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 23, 22, 21, 20]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    }
    prices_by_symbol = tester._prices_by_symbol(candles)
    timestamp = sorted(prices_by_symbol["AAPL"])[-1]

    ranked = tester._rank_symbols(timestamp, prices_by_symbol)

    assert ranked == [("AAPL", ranked[0][1])]
    assert tester.quality_filter_enabled


def test_dual_momentum_leadership_filter_requires_outperformance():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        leadership_filter_enabled=True,
        leadership_symbol="SPY",
        leadership_momentum_periods=[2],
        transaction_cost_bps=0,
    )

    candles = {
        "AAPL": make_candles("AAPL", [10, 11, 12, 14, 18, 23, 29]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 23, 24, 25, 26]),
        "SPY": make_candles("SPY", [10, 11, 12, 14, 16, 18, 20]),
    }
    prices_by_symbol = tester._prices_by_symbol(candles)
    timestamp = sorted(prices_by_symbol["AAPL"])[-1]

    ranked = tester._rank_symbols(timestamp, prices_by_symbol)

    assert ranked == [("AAPL", ranked[0][1])]


def test_dual_momentum_enhanced_ranking_rewards_relative_strength():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        ranking_score_mode="enhanced",
        enhanced_momentum_periods=[2, 4],
        enhanced_momentum_weights=[0.40, 0.60],
        relative_strength_symbol="SPY",
        relative_strength_periods=[2],
        relative_strength_weight=0.50,
        volatility_penalty_weight=0.0,
        transaction_cost_bps=0,
    )

    candles = {
        "AAPL": make_candles("AAPL", [10, 11, 12, 14, 17, 21, 26]),
        "MSFT": make_candles("MSFT", [20, 21, 22, 23, 24, 25, 26]),
        "SPY": make_candles("SPY", [10, 10.5, 11, 11.5, 12, 12.5, 13]),
    }
    prices_by_symbol = tester._prices_by_symbol(candles)
    timestamp = sorted(prices_by_symbol["AAPL"])[-1]

    ranked = tester._rank_symbols(timestamp, prices_by_symbol)

    assert ranked[0][0] == "AAPL"
    assert ranked[0][1] > ranked[1][1]


def test_dual_momentum_cooldown_blocks_bad_monthly_loser_reentry():
    aapl_prices = []
    msft_prices = []
    spy_prices = []

    for index in range(70):
        if index <= 30:
            aapl_prices.append(10 + min(index, 2))
        elif index < 58:
            aapl_prices.append(8)
        else:
            aapl_prices.append(8 + (index - 57) * 2)

        msft_prices.append(10 + index * 0.1)
        spy_prices.append(10 + index * 0.1)

    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[1],
        regime_sma_period=2,
        asset_sma_period=2,
        cooldown_enabled=True,
        cooldown_periods=2,
        cooldown_loss_threshold=-0.01,
        transaction_cost_bps=0,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", aapl_prices),
        "MSFT": make_candles("MSFT", msft_prices),
        "SPY": make_candles("SPY", spy_prices),
    })

    assert result.selections[0].symbols == ["AAPL"]
    assert result.selections[1].symbols == ["MSFT"]
    assert "AAPL" in result.selections[2].cooldown_symbols
    assert result.selections[2].symbols == ["MSFT"]


def test_dual_momentum_reports_period_returns_and_drawdowns():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        transaction_cost_bps=5,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 13, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 19, 18, 17, 16, 15, 14, 13]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16, 17]),
    })

    assert result.annual_returns
    assert result.monthly_returns
    assert result.cagr > 0
    assert result.calmar >= 0
    assert result.annualized_turnover_percent > 0
    assert result.turnover_per_rebalance_percent > 0
    assert result.cost_drag_percent > 0
    assert "max_drawdown" in result.drawdown_statistics
    assert result.estimated_cost > 0


def test_dual_momentum_asset_trend_filter_excludes_weak_asset():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 23, 19, 18, 17, 18, 19]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    assert result.selections[0].symbols == ["AAPL"]


def test_dual_momentum_breadth_filter_blocks_risk_when_market_is_narrow():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=2,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
        min_breadth_percent=0.75,
    )

    result = tester.run({
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "MSFT": make_candles("MSFT", [20, 19, 18, 17, 16, 15, 14]),
        "TSLA": make_candles("TSLA", [30, 29, 28, 27, 26, 25, 24]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    })

    assert result.selections[0].risk_on is False
    assert result.selections[0].symbols == []


def test_dual_momentum_volatility_target_reduces_exposure():
    tester = DualMomentumPortfolioBacktester(
        target_exposure=1.0,
        target_volatility=0.10,
        volatility_lookback=3,
    )

    exposure = tester._target_exposure_for_rebalance([0.10, -0.10, 0.10])

    assert 0 < exposure < 1.0


def test_dual_momentum_can_run_on_test_slice_with_prior_warmup():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
    )
    candles = {
        "AAPL": make_candles("AAPL", [10, 11, 12, 13, 14, 15, 16, 17]),
        "SPY": make_candles("SPY", [10, 11, 12, 13, 14, 15, 16, 17]),
    }
    start_at = datetime(2025, 1, 6)
    end_at = datetime(2025, 1, 8)

    result = tester.run(candles, start_at=start_at, end_at=end_at)

    assert result.result.equity_curve[0].timestamp >= start_at
    assert result.result.equity_curve[-1].timestamp <= end_at


def test_dual_momentum_slice_accepts_timezone_aware_candles():
    tester = DualMomentumPortfolioBacktester(
        starting_equity=500,
        top_n=1,
        momentum_periods=[2],
        regime_sma_period=3,
        asset_sma_period=3,
    )
    candles = {
        "AAPL": make_aware_candles("AAPL", [10, 11, 12, 13, 14, 15, 16]),
        "SPY": make_aware_candles("SPY", [10, 11, 12, 13, 14, 15, 16]),
    }

    result = tester.run(
        candles,
        start_at=datetime(2025, 1, 5),
        end_at=datetime(2025, 1, 7),
    )

    assert len(result.result.equity_curve) == 3
