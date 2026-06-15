from datetime import datetime, timedelta, timezone

import pytest

from core.entities.backtest_result import BacktestResult
from core.entities.benchmark_result import BenchmarkResult
from core.entities.candle import Candle
from core.entities.optimization_result import OptimizationResult
from core.entities.walk_forward_result import (
    WalkForwardFoldResult,
    WalkForwardResult,
)
from core.entities.strategy_comparison_result import StrategyComparisonResult
from core.research.market_regime_analyzer import MarketRegimeAnalyzer
from core.research.parameter_optimizer import (
    ParameterOptimizer,
    expand_grid,
    get_metric,
    parameter_overrides,
    valid_parameters,
)
from core.research.strategy_comparison import StrategyComparison
from core.research.strategy_factory import build_strategy
from core.research.walk_forward import (
    WalkForwardTester,
    benchmark_metrics,
    buy_and_hold_return,
    candles_between,
    evaluate_fold,
)
from strategies.ema_crossover_strategy import EMACrossoverStrategy
from strategies.bollinger_mean_reversion_strategy import (
    BollingerMeanReversionStrategy,
)
from strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
from strategies.ema_rsi_pullback_strategy import EMARSIPullbackStrategy
from strategies.ema_rsi_filter_strategy import EMARSIFilterStrategy
from strategies.ensemble_vote_strategy import EnsembleVoteStrategy
from strategies.rsi_mean_reversion_strategy import RSIMeanReversionStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from strategies.trend_pullback_strategy import TrendPullbackStrategy


def make_research_config():
    return {
        "backtest": {
            "timeframe": "1Day",
            "starting_equity": 10_000,
            "warmup_bars": 20,
        },
        "strategy": {
            "name": "ema_crossover",
            "ema_fast_period": 5,
            "ema_slow_period": 10,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        },
        "risk": {
            "manager": "atr",
            "max_risk_per_trade": 0.0025,
            "max_exposure": 0.02,
            "atr_multiplier": 2.0,
            "atr_stop_multiplier": 100,
            "atr_take_profit_multiplier": 100,
        },
        "execution": {
            "spread_bps": 0,
            "slippage_bps": 0,
            "seed": 42,
        },
        "research": {
            "min_closed_trades": 0,
            "optimization_metric": "final_equity",
            "walk_forward_folds": [
                {
                    "train_start": "2021-01-01",
                    "train_end": "2021-03-31",
                    "test_start": "2021-04-01",
                    "test_end": "2021-05-31",
                },
            ],
            "strategy_comparison": [
                {
                    "name": "ema_crossover",
                    "parameter_grid": {
                        "ema_fast_period": [5],
                        "ema_slow_period": [20],
                    },
                },
                {
                    "name": "rsi_mean_reversion",
                    "parameter_grid": {
                        "rsi_oversold": [35],
                        "rsi_exit_level": [50],
                    },
                },
            ],
        },
    }


def make_research_candles(count):
    start = datetime(2021, 1, 1)

    return [
        Candle(
            symbol="AAPL",
            timestamp=start + timedelta(days=index),
            open=100 + index * 0.5,
            high=101 + index * 0.5,
            low=99 + index * 0.5,
            close=100 + index * 0.5,
            volume=1_000,
        )
        for index in range(count)
    ]


def make_aware_research_candles(count):
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)

    return [
        Candle(
            symbol="AAPL",
            timestamp=start + timedelta(days=index),
            open=100 + index * 0.5,
            high=101 + index * 0.5,
            low=99 + index * 0.5,
            close=100 + index * 0.5,
            volume=1_000,
        )
        for index in range(count)
    ]


def test_strategy_factory_builds_ema_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "ema_crossover",
            "ema_fast_period": 20,
            "ema_slow_period": 50,
        },
    )

    assert isinstance(strategy, EMACrossoverStrategy)
    assert strategy.fast_period == 20
    assert strategy.slow_period == 50


def test_strategy_factory_builds_rsi_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "rsi",
            "rsi_period": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
        },
    )

    assert isinstance(strategy, RSIStrategy)
    assert strategy.oversold == 25
    assert strategy.overbought == 75


def test_strategy_factory_builds_donchian_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "donchian_breakout",
            "donchian_lookback": 55,
        },
    )

    assert isinstance(strategy, DonchianBreakoutStrategy)
    assert strategy.lookback_period == 55


def test_strategy_factory_builds_volatility_filtered_donchian_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "donchian_with_volatility_filter",
            "donchian_lookback": 10,
        },
    )

    assert isinstance(strategy, DonchianBreakoutStrategy)
    assert strategy.use_volatility_filter


def test_strategy_factory_builds_volume_filtered_donchian_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "donchian_breakout",
            "use_volume_filter": True,
            "min_relative_volume": 1.2,
        },
    )

    assert strategy.use_volume_filter
    assert strategy.min_relative_volume == 1.2


def test_strategy_factory_builds_rsi_mean_reversion_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "rsi_mean_reversion",
            "rsi_oversold": 35,
            "rsi_exit_level": 55,
        },
    )

    assert isinstance(strategy, RSIMeanReversionStrategy)
    assert strategy.oversold == 35
    assert strategy.exit_level == 55


def test_strategy_factory_builds_sideways_rsi_mean_reversion_strategy():
    strategy = build_strategy(
        "AAPL",
        {"name": "rsi_sideways_mean_reversion"},
    )

    assert isinstance(strategy, RSIMeanReversionStrategy)
    assert strategy.require_sideways_regime


def test_strategy_factory_builds_ema_rsi_filter_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "ema_rsi_filter",
            "ema_fast_period": 20,
            "ema_slow_period": 50,
            "rsi_entry": 55,
        },
    )

    assert isinstance(strategy, EMARSIFilterStrategy)
    assert strategy.rsi_entry == 55


def test_strategy_factory_builds_ema_rsi_pullback_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "ema_rsi_pullback",
            "ema_fast_period": 20,
            "ema_slow_period": 200,
            "rsi_pullback": 45,
        },
    )

    assert isinstance(strategy, EMARSIPullbackStrategy)
    assert strategy.rsi_pullback == 45


def test_strategy_factory_builds_ema_rsi_pullback_with_volume_filter():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "ema_rsi_pullback",
            "min_relative_volume": 1.1,
        },
    )

    assert strategy.min_relative_volume == 1.1


def test_strategy_factory_builds_bollinger_mean_reversion_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "bollinger_mean_reversion",
            "rsi_entry": 35,
            "require_sideways_regime": True,
        },
    )

    assert isinstance(strategy, BollingerMeanReversionStrategy)
    assert strategy.rsi_entry == 35
    assert strategy.require_sideways_regime


def test_strategy_factory_builds_bollinger_with_bandwidth_filters():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "bollinger_mean_reversion",
            "min_bandwidth": 0.03,
            "max_bandwidth": 0.20,
        },
    )

    assert strategy.min_bandwidth == 0.03
    assert strategy.max_bandwidth == 0.20


def test_strategy_factory_builds_trend_pullback_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "trend_pullback",
            "pullback_fast_period": 20,
            "pullback_tolerance": 0.04,
        },
    )

    assert isinstance(strategy, TrendPullbackStrategy)
    assert strategy.fast_period == 20
    assert strategy.pullback_tolerance == 0.04


def test_strategy_factory_builds_trend_pullback_with_adx_filter():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "trend_pullback",
            "min_adx": 20,
            "min_relative_volume": 1.0,
        },
    )

    assert strategy.min_adx == 20
    assert strategy.min_relative_volume == 1.0


def test_strategy_factory_builds_buy_and_hold_strategy():
    strategy = build_strategy(
        "AAPL",
        {"name": "buy_and_hold"},
    )

    assert isinstance(strategy, BuyAndHoldStrategy)


def test_strategy_factory_builds_ensemble_vote_strategy():
    strategy = build_strategy(
        "AAPL",
        {
            "name": "ensemble_vote",
            "ensemble_min_buy_votes": 3,
            "rsi_entry": 55,
            "use_breakout_vote": False,
        },
    )

    assert isinstance(strategy, EnsembleVoteStrategy)
    assert strategy.min_buy_votes == 3
    assert strategy.rsi_entry == 55
    assert not strategy.use_breakout_vote


def test_expand_grid_creates_all_parameter_combinations():
    grid = {
        "ema_fast_period": [5, 10],
        "ema_slow_period": [20, 30],
        "atr_stop_multiplier": [1, 2],
    }

    assert len(expand_grid(grid)) == 8


def test_invalid_ema_parameter_combinations_are_rejected():
    assert valid_parameters({
        "ema_fast_period": 20,
        "ema_slow_period": 50,
    })
    assert not valid_parameters({
        "ema_fast_period": 50,
        "ema_slow_period": 20,
    })


def test_invalid_take_profit_parameter_is_rejected():
    assert not valid_parameters({
        "atr_stop_multiplier": 2,
        "atr_take_profit_multiplier": 1,
    })
    assert valid_parameters({
        "atr_stop_multiplier": 2,
        "atr_take_profit_multiplier": None,
    })


def test_parameter_overrides_routes_position_sizing_parameters():
    overrides = parameter_overrides({
        "ema_fast_period": 20,
        "target_exposure": 0.30,
        "sizing_mode": "fixed_fractional",
        "position_max_exposure": 0.40,
    })

    assert overrides["strategy"]["ema_fast_period"] == 20
    assert overrides["position_sizing"]["target_exposure"] == 0.30
    assert overrides["position_sizing"]["mode"] == "fixed_fractional"
    assert overrides["position_sizing"]["max_exposure"] == 0.40


def test_parameter_overrides_lifts_exposure_cap_to_target():
    overrides = parameter_overrides({
        "target_exposure": 0.60,
    })

    assert overrides["position_sizing"]["target_exposure"] == 0.60
    assert overrides["position_sizing"]["max_exposure"] == 0.60


def test_excess_return_metric_uses_benchmark_return():
    result = BacktestResult(
        starting_equity=500,
        final_equity=550,
        total_return=0.10,
        max_drawdown=0.02,
        sharpe=1,
        closed_trades=5,
        open_trades=0,
        equity_curve=[],
        profit_factor=1.5,
    )

    assert get_metric(
        result,
        "excess_return",
        benchmark_return_value=0.04,
    ) == pytest.approx(0.06)


def test_market_regime_analyzer_classifies_bull_and_high_volatility():
    regime = MarketRegimeAnalyzer().classify(
        close=120,
        sma_200=100,
        previous_sma_200=99,
        volatility=0.04,
        volatility_average=0.02,
    )

    assert regime.market_regime == "bull"
    assert regime.volatility_regime == "high"


def test_parameter_optimizer_ranks_results():
    optimizer = ParameterOptimizer(
        config=make_research_config(),
        metric_name="final_equity",
    )

    results = optimizer.run(
        candles=make_research_candles(140),
        symbol="AAPL",
        grid={
            "ema_fast_period": [5, 10],
            "ema_slow_period": [20],
            "atr_stop_multiplier": [1, 2],
        },
    )

    assert len(results) == 4
    assert results[0].metric_value >= results[-1].metric_value
    assert "ema_fast_period" in results[0].parameters


def test_parameter_optimizer_filters_low_trade_results():
    optimizer = ParameterOptimizer(
        config=make_research_config(),
        metric_name="final_equity",
        min_closed_trades=999,
    )

    results = optimizer.run(
        candles=make_research_candles(140),
        symbol="AAPL",
        grid={
            "ema_fast_period": [5],
            "ema_slow_period": [20],
            "atr_stop_multiplier": [1],
        },
    )

    assert results == []


def test_walk_forward_runs_each_fold_with_best_training_parameters():
    config = make_research_config()
    tester = WalkForwardTester(
        config=config,
        metric_name="final_equity",
    )

    result = tester.run(
        candles=make_research_candles(220),
        symbol="AAPL",
        folds=[
            {
                "train_start": "2021-01-01",
                "train_end": "2021-03-31",
                "test_start": "2021-04-01",
                "test_end": "2021-05-31",
            },
            {
                "train_start": "2021-02-01",
                "train_end": "2021-04-30",
                "test_start": "2021-05-01",
                "test_end": "2021-06-30",
            },
        ],
        grid={
            "ema_fast_period": [5],
            "ema_slow_period": [20],
            "atr_stop_multiplier": [1, 2],
        },
    )

    assert result.symbol == "AAPL"
    assert result.timeframe == "1Day"
    assert len(result.folds) == 2
    assert result.average_test_return >= 0
    assert result.folds[0].benchmark_return > 0
    assert (
        result.folds[0].excess_return
        == result.folds[0].test_result.total_return
        - result.folds[0].benchmark_return
    )
    fold_report = result.folds[0].to_dict()
    assert "test_max_drawdown" in fold_report
    assert "passed" in fold_report
    assert "test_trade_analysis" in fold_report


def test_walk_forward_accepts_timezone_aware_candle_timestamps():
    candles = make_aware_research_candles(10)

    filtered = candles_between(
        candles,
        datetime(2021, 1, 2),
        datetime(2021, 1, 4),
    )

    assert len(filtered) == 3


def test_buy_and_hold_return_uses_first_and_last_close():
    candles = make_research_candles(3)

    assert buy_and_hold_return(candles) == (101 / 100) - 1


def test_benchmark_metrics_include_sharpe_and_drawdown():
    candles = make_research_candles(10)
    benchmark = benchmark_metrics(candles)

    assert benchmark.total_return > 0
    assert benchmark.sharpe > 0
    assert benchmark.max_drawdown == 0


def test_quality_gates_reject_low_quality_fold():
    result = BacktestResult(
        starting_equity=500,
        final_equity=500,
        total_return=0,
        max_drawdown=0.01,
        sharpe=0.2,
        closed_trades=2,
        open_trades=0,
        equity_curve=[],
        profit_factor=0.8,
    )
    passed, reason = evaluate_fold(
        test_result=result,
        benchmark=BenchmarkResult(total_return=0.03, sharpe=0.5),
        excess_return=-0.03,
        research_config={
            "max_drawdown": 0.20,
            "min_time_in_market": 0,
            "max_time_in_market": 1,
            "min_sharpe": 0,
            "require_sharpe_edge": True,
            "require_positive_excess": True,
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
        },
    )

    assert not passed
    assert "closed_trades below minimum" in reason
    assert "profit_factor below minimum" in reason
    assert "excess_return is not positive" in reason


def test_buy_and_hold_quality_gate_skips_trade_count_and_profit_factor():
    result = BacktestResult(
        starting_equity=500,
        final_equity=550,
        total_return=0.10,
        max_drawdown=0.01,
        sharpe=1.2,
        closed_trades=1,
        open_trades=0,
        equity_curve=[],
        profit_factor=0,
    )
    passed, reason = evaluate_fold(
        test_result=result,
        benchmark=BenchmarkResult(total_return=0.03, sharpe=0.5),
        excess_return=0.07,
        research_config={
            "max_drawdown": 0.20,
            "min_time_in_market": 0,
            "max_time_in_market": 1,
            "min_sharpe": 0,
            "require_sharpe_edge": True,
            "require_positive_excess": True,
            "min_closed_trades": 20,
            "min_profit_factor": 1.1,
        },
        strategy_name="buy_and_hold",
    )

    assert passed
    assert reason == ""


def test_walk_forward_applies_quality_gates_to_fold():
    config = make_research_config()
    config["research"].update({
        "min_closed_trades": 20,
        "min_profit_factor": 1.1,
        "require_positive_excess": True,
    })
    tester = WalkForwardTester(
        config=config,
        metric_name="final_equity",
    )
    result = tester.run(
        candles=make_research_candles(120),
        symbol="AAPL",
        folds=[
            {
                "train_start": "2021-01-01",
                "train_end": "2021-02-28",
                "test_start": "2021-03-01",
                "test_end": "2021-04-15",
            },
        ],
        grid={
            "ema_fast_period": [5],
            "ema_slow_period": [20],
            "atr_stop_multiplier": [1],
        },
    )

    if result.folds:
        assert not result.folds[0].passed
        assert result.folds[0].failure_reason


def test_walk_forward_result_can_be_saved(tmp_path):
    config = make_research_config()
    tester = WalkForwardTester(
        config=config,
        metric_name="final_equity",
    )

    result = tester.run(
        candles=make_research_candles(120),
        symbol="AAPL",
        folds=[
            {
                "train_start": "2021-01-01",
                "train_end": "2021-02-28",
                "test_start": "2021-03-01",
                "test_end": "2021-04-15",
            },
        ],
        grid={
            "ema_fast_period": [5],
            "ema_slow_period": [20],
            "atr_stop_multiplier": [1],
        },
    )

    path = result.save_json(report_dir=str(tmp_path))

    assert path.exists()
    assert "average_excess_return" in path.read_text(encoding="utf-8")
    assert "passed" in path.read_text(encoding="utf-8")
    assert "test_trade_analysis" in path.read_text(encoding="utf-8")


def test_strategy_comparison_ranks_multiple_strategies():
    config = make_research_config()
    comparison = StrategyComparison(
        config=config,
        candles_by_symbol={"AAPL": make_research_candles(140)},
    )

    results = comparison.run()
    table = comparison.to_table(results)

    assert len(results) == 2
    assert "ema_crossover" in table
    assert "rsi_mean_reversion" in table


def make_comparison_result(
    strategy_name,
    symbol,
    excess_return,
    sharpe,
    trades,
    passed,
):
    test_result = BacktestResult(
        starting_equity=500,
        final_equity=500 * (1 + excess_return + 0.02),
        total_return=excess_return + 0.02,
        max_drawdown=0.01,
        sharpe=sharpe,
        closed_trades=trades,
        open_trades=0,
        equity_curve=[],
        profit_factor=1.5,
    )
    optimization = OptimizationResult(
        parameters={},
        metric_name="composite",
        metric_value=sharpe,
        result=test_result,
    )
    fold = WalkForwardFoldResult(
        train_start=datetime(2021, 1, 1),
        train_end=datetime(2021, 12, 31),
        test_start=datetime(2022, 1, 1),
        test_end=datetime(2022, 12, 31),
        best_training_result=optimization,
        test_result=test_result,
        benchmark_return=0.02,
        benchmark=BenchmarkResult(total_return=0.02, sharpe=0.5),
        excess_return=excess_return,
        excess_return_per_unit_risk=excess_return / 0.01,
        passed=passed,
    )

    return StrategyComparisonResult(
        strategy_name=strategy_name,
        symbol=symbol,
        walk_forward_result=WalkForwardResult(
            symbol=symbol,
            timeframe="1Day",
            folds=[fold],
        ),
    )


def test_strategy_comparison_ranking_prefers_positive_excess():
    comparison = StrategyComparison(
        config=make_research_config(),
        candles_by_symbol={},
    )
    negative = make_comparison_result(
        "rsi_mean_reversion",
        "TSLA",
        excess_return=-0.20,
        sharpe=2.5,
        trades=2,
        passed=False,
    )
    positive = make_comparison_result(
        "ema_crossover",
        "AAPL",
        excess_return=0.01,
        sharpe=0.6,
        trades=20,
        passed=True,
    )

    ranked = comparison.rank([negative, positive])

    assert ranked[0] == positive
