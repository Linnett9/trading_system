from application.services.market_data_loader import load_candles
from application.reporting.walk_forward_reporter import (
    print_walk_forward_summary,
)
from core.research.backtest_runner import run_backtest
from core.research.parameter_optimizer import ParameterOptimizer
from core.research.strategy_comparison import StrategyComparison
from core.research.walk_forward import WalkForwardTester


def run_base_backtests(config, feed):
    for symbol in config["backtest"]["symbols"]:
        candles = load_candles(symbol, config, feed)

        result = run_backtest(
            candles=candles,
            symbol=symbol,
            config=config,
        )

        report_path = result.save_json(
            symbol=symbol,
            timeframe=config["backtest"]["timeframe"],
            report_dir=config["reports"]["backtest_dir"],
        )

        print_result(symbol, result, report_path)


def run_optimization(config, feed):
    research_config = config["research"]

    optimizer = ParameterOptimizer(
        config=config,
        metric_name=research_config["optimization_metric"],
        min_closed_trades=research_config.get(
            "optimizer_min_closed_trades",
            research_config.get("min_closed_trades", 0),
        ),
    )

    for symbol in config["backtest"]["symbols"]:
        candles = load_candles(symbol, config, feed)

        results = optimizer.run(
            candles=candles,
            symbol=symbol,
            grid=research_config["parameter_grid"],
        )

        if not results:
            print(f"{symbol} | no valid parameter combinations")
            continue

        best = results[0]

        print(
            f"{symbol} | best_{best.metric_name}={best.metric_value:.4f} | "
            f"params={best.parameters}"
        )


def run_walk_forward(config, feed, show_details=False):
    research_config = config["research"]

    tester = WalkForwardTester(
        config=config,
        metric_name=research_config["optimization_metric"],
    )

    results = []

    for symbol in config["backtest"]["symbols"]:
        candles = load_candles(symbol, config, feed)

        result = tester.run(
            candles=candles,
            symbol=symbol,
            folds=research_config["walk_forward_folds"],
            grid=research_config["parameter_grid"],
        )

        report_path = result.save_json(
            report_dir=config["reports"]["walk_forward_dir"],
        )

        results.append((symbol, result, report_path))

    print_walk_forward_summary(results, show_details=show_details)


def run_strategy_comparison(config, feed, show_all=False):
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in config["backtest"]["symbols"]
    }

    comparison = StrategyComparison(
        config=config,
        candles_by_symbol=candles_by_symbol,
    )

    results = comparison.run()

    report_path = comparison.save_csv(
        results,
        report_dir=config["reports"]["summary_dir"],
    )

    print("\nSTRATEGY COMPARISON")

    limit = None if show_all else config["research"].get("report_top_n", 10)

    print(comparison.to_table(results, limit=limit))

    if limit and len(results) > limit:
        print(
            f"\nShowing top {limit} of {len(results)} results. "
            "Use --all-results to print everything."
        )

    print(f"\nSaved summary: {report_path}")


def print_result(symbol, result, report_path):
    print(
        f"{symbol} | "
        f"return={result.total_return * 100:.2f}% | "
        f"max_dd={result.max_drawdown * 100:.2f}% | "
        f"sharpe={result.sharpe:.2f} | "
        f"closed={result.closed_trades} | "
        f"report={report_path}"
    )