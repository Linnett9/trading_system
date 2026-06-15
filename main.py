import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

from config.config_loader import load_config
from core.entities.candle import Candle
from core.research.backtest_runner import run_backtest
from core.research.experiment_reporter import ExperimentReporter
from core.research.parameter_optimizer import ParameterOptimizer
from core.research.strategy_comparison import StrategyComparison
from core.research.walk_forward import WalkForwardTester
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed


def load_candles(symbol, config, feed):
    backtest_config = config["backtest"]
    end = datetime.utcnow()
    start = end - timedelta(days=365 * backtest_config["years"])
    cache_config = config.get("cache", {})
    cache_enabled = cache_config.get("enabled", False)
    cache_path = data_cache_path(symbol, backtest_config, cache_config)

    if cache_enabled and cache_path.exists():
        return read_candle_cache(cache_path)

    candles = feed.get_historical_bars(
        symbol=symbol,
        timeframe=backtest_config["timeframe"],
        start=start,
        end=end,
    )

    if cache_enabled:
        write_candle_cache(cache_path, candles)

    return candles


def data_cache_path(symbol, backtest_config, cache_config):
    directory = Path(cache_config.get("data_dir", "cache/data"))
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{symbol}_{backtest_config['timeframe']}_"
        f"{backtest_config['years']}y.json"
    )
    return directory / filename


def read_candle_cache(path):
    payload = json.loads(path.read_text(encoding="utf-8"))

    return [
        Candle(
            symbol=item["symbol"],
            timestamp=datetime.fromisoformat(item["timestamp"]),
            open=item["open"],
            high=item["high"],
            low=item["low"],
            close=item["close"],
            volume=item["volume"],
        )
        for item in payload
    ]


def write_candle_cache(path, candles):
    payload = [
        {
            "symbol": candle.symbol,
            "timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")



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


def print_walk_forward_folds(symbol, result):
    for index, fold in enumerate(result.folds, start=1):
        diagnostics = fold.test_result.signal_diagnostics
        utilization = fold.test_result.capital_utilization
        print(
            f"  {symbol} fold {index} | "
            f"train_sharpe={fold.best_training_result.result.sharpe:.2f} | "
            f"test_sharpe={fold.test_result.sharpe:.2f} | "
            f"benchmark_sharpe={fold.benchmark.sharpe:.2f} | "
            f"test_return={fold.test_result.total_return * 100:.2f}% | "
            f"test_dd={fold.test_result.max_drawdown * 100:.2f}% | "
            f"benchmark={fold.benchmark_return * 100:.2f}% | "
            f"excess={fold.excess_return * 100:.2f}% | "
            f"trades={fold.test_result.closed_trades} | "
            f"time_in={fold.test_result.trade_analysis.time_in_market_percent * 100:.2f}% | "
            f"exposure={utilization.average_exposure_percent * 100:.2f}% | "
            f"cash={utilization.average_cash_percent * 100:.2f}% | "
            f"avg_hold={fold.test_result.trade_analysis.average_trade_duration_days:.1f}d | "
            f"signals=B{diagnostics.buy_signals}/S{diagnostics.sell_signals}/H{diagnostics.hold_signals} | "
            f"blocks=dup{diagnostics.duplicate_buy_skips}/flat{diagnostics.flat_sell_skips}/risk{diagnostics.risk_blocked_signals} | "
            f"exits=stop{diagnostics.stop_loss_exits}/tp{diagnostics.take_profit_exits} | "
            f"passed={fold.passed} | "
            f"reason={fold.failure_reason or 'passed'} | "
            f"params={fold.best_training_result.parameters}"
        )


def print_walk_forward_summary(results, show_details=False):
    print("\nWALK-FORWARD SUMMARY")
    print(
        "Symbol | Folds | Test Ret | Benchmark | Excess | Sharpe | "
        "Bench Sh | Trades | Report"
    )
    print("-" * 104)

    for symbol, result, report_path in results:
        print(
            f"{symbol:<6} | "
            f"{len(result.folds):>5} | "
            f"{format_percent(result.average_test_return):>8} | "
            f"{format_percent(result.average_benchmark_return):>9} | "
            f"{format_percent(result.average_excess_return):>7} | "
            f"{result.average_test_sharpe:>6.2f} | "
            f"{result.average_benchmark_sharpe:>8.2f} | "
            f"{total_closed_trades(result):>6} | "
            f"{report_path}"
        )

        if show_details:
            print_walk_forward_folds(symbol, result)

    print("-" * 104)
    print_portfolio_walk_forward_summary(results)


def print_portfolio_walk_forward_summary(results):
    if not results:
        print("No walk-forward results.")
        return

    count = len(results)
    avg_test_return = sum(
        result.average_test_return
        for _, result, _ in results
    ) / count
    avg_benchmark_return = sum(
        result.average_benchmark_return
        for _, result, _ in results
    ) / count
    avg_excess_return = sum(
        result.average_excess_return
        for _, result, _ in results
    ) / count
    avg_sharpe = sum(
        result.average_test_sharpe
        for _, result, _ in results
    ) / count
    trades = sum(
        total_closed_trades(result)
        for _, result, _ in results
    )

    print(
        "Equal-weight average | "
        f"test={format_percent(avg_test_return)} | "
        f"benchmark={format_percent(avg_benchmark_return)} | "
        f"excess={format_percent(avg_excess_return)} | "
        f"sharpe={avg_sharpe:.2f} | "
        f"trades={trades}"
    )
    print()
    print_experiment_ranking(results)


def print_experiment_ranking(results):
    reporter = ExperimentReporter()

    for _, result, _ in results:
        reporter.add_walk_forward_result(result)

    print("EXPERIMENT RANKING")
    print(reporter.to_table())


def total_closed_trades(result):
    return sum(
        fold.test_result.closed_trades
        for fold in result.folds
    )


def format_percent(value):
    return f"{value * 100:.2f}%"


def print_result(symbol, result, report_path):
    print(
        f"{symbol} | "
        f"return={result.total_return * 100:.2f}% | "
        f"max_dd={result.max_drawdown * 100:.2f}% | "
        f"sharpe={result.sharpe:.2f} | "
        f"closed={result.closed_trades} | "
        f"report={report_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["backtest", "optimize", "walk-forward", "compare-strategies"],
        default="backtest",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show fold-level walk-forward details.",
    )
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="Print all strategy comparison rows instead of top-N.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    feed = AlpacaDataFeed(
        api_key=config["alpaca"]["api_key"],
        secret_key=config["alpaca"]["secret_key"],
    )

    if args.mode == "optimize":
        run_optimization(config, feed)
        return

    if args.mode == "walk-forward":
        run_walk_forward(config, feed, show_details=args.details)
        return

    if args.mode == "compare-strategies":
        run_strategy_comparison(config, feed, show_all=args.all_results)
        return

    run_base_backtests(config, feed)


if __name__ == "__main__":
    main()
