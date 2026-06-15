from core.entities.backtest_result import BacktestResult
from core.entities.benchmark_result import BenchmarkResult
from core.entities.walk_forward_result import (
    WalkForwardFoldResult,
    WalkForwardResult,
)
from core.entities.optimization_result import OptimizationResult
from core.research.experiment_reporter import ExperimentReporter
from datetime import datetime


def make_backtest_result(
    total_return,
    sharpe,
    max_drawdown,
    closed_trades,
    profit_factor,
):
    return BacktestResult(
        starting_equity=500,
        final_equity=500 * (1 + total_return),
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        closed_trades=closed_trades,
        open_trades=0,
        equity_curve=[],
        profit_factor=profit_factor,
    )


def make_walk_forward_result(symbol, sharpe, max_drawdown, trades, passed):
    test_result = make_backtest_result(
        total_return=0.05,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        closed_trades=trades,
        profit_factor=1.5,
    )
    optimization_result = OptimizationResult(
        parameters={"ema_fast_period": 20},
        metric_name="sharpe",
        metric_value=sharpe,
        result=test_result,
    )

    fold = WalkForwardFoldResult(
        train_start=datetime(2021, 1, 1),
        train_end=datetime(2021, 12, 31),
        test_start=datetime(2022, 1, 1),
        test_end=datetime(2022, 12, 31),
        best_training_result=optimization_result,
        test_result=test_result,
        benchmark_return=0.03,
        benchmark=BenchmarkResult(
            total_return=0.03,
            max_drawdown=0.01,
            sharpe=0.4,
        ),
        excess_return=0.02,
        excess_return_per_unit_risk=1.0,
        passed=passed,
    )

    return WalkForwardResult(
        symbol=symbol,
        timeframe="1Day",
        folds=[fold],
    )


def test_experiment_reporter_ranks_by_quality_metrics():
    reporter = ExperimentReporter()
    reporter.add_walk_forward_result(
        make_walk_forward_result("AAPL", sharpe=0.5, max_drawdown=0.02, trades=20, passed=True)
    )
    reporter.add_walk_forward_result(
        make_walk_forward_result("MSFT", sharpe=1.0, max_drawdown=0.05, trades=10, passed=True)
    )

    ranked = reporter.ranked()

    assert ranked[0].symbol == "MSFT"
    assert ranked[0].walk_forward_sharpe == 1.0


def test_experiment_reporter_table_includes_passed_folds():
    reporter = ExperimentReporter()
    reporter.add_walk_forward_result(
        make_walk_forward_result("AAPL", sharpe=0.5, max_drawdown=0.02, trades=20, passed=True)
    )

    table = reporter.to_table()

    assert "AAPL" in table
    assert "1/1" in table
