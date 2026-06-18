from core.research.experiment_reporter import ExperimentReporter


def format_percent(value):
    return f"{value * 100:.2f}%"


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