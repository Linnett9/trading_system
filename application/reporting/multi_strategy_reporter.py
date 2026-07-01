from core.research.multi_strategy.experiments import (
    multi_strategy_quality_score,
)


def format_percent(value):
    return f"{value * 100:.2f}%"


def print_multi_strategy_result(result, report_path):
    backtest = result.result
    utilization = backtest.capital_utilization
    analysis = backtest.trade_analysis

    print("\nMULTI-STRATEGY PORTFOLIO")
    print(
        f"return={format_percent(backtest.total_return)} | "
        f"benchmark={format_percent(result.benchmark_return)} | "
        f"excess={format_percent(result.excess_return)} | "
        f"equal_weight={format_percent(result.equal_weight_return)} | "
        f"ex_eq={format_percent(result.excess_vs_equal_weight)} | "
        f"sharpe={backtest.sharpe:.2f} | "
        f"max_dd={format_percent(backtest.max_drawdown)} | "
        f"closed={backtest.closed_trades} | "
        f"open={backtest.open_trades}"
    )
    print(
        f"time_in={format_percent(analysis.time_in_market_percent)} | "
        f"exposure={format_percent(utilization.average_exposure_percent)} | "
        f"cash={format_percent(utilization.average_cash_percent)} | "
        f"profit_factor={backtest.profit_factor:.2f}"
    )

    print("Sleeves:")
    for sleeve in result.sleeves:
        print(
            f"  {sleeve.name:<18} | "
            f"weight={format_percent(sleeve.weight)} | "
            f"return={format_percent(sleeve.result.total_return)} | "
            f"sharpe={sleeve.result.sharpe:.2f} | "
            f"dd={format_percent(sleeve.result.max_drawdown)}"
        )

    print("Annual diagnosis:")
    for year, diagnosis in result.diagnostics.get("annual", {}).items():
        print(
            f"  {year} | "
            f"bot={format_percent(diagnosis['bot_return'])} | "
            f"spy={format_percent(diagnosis['benchmark_return'])} | "
            f"ex_spy={format_percent(diagnosis['excess_vs_benchmark'])} | "
            f"ex_eq={format_percent(diagnosis['excess_vs_equal_weight'])} | "
            f"{diagnosis['regime_label']}"
        )

    print(f"Saved summary: {report_path}")


def print_multi_strategy_walk_forward(results, report_path):
    print("\nMULTI-STRATEGY WALK-FORWARD")
    print(
        "Fold | Test | Weights | Return | SPY | Ex SPY | Ex EqWt | "
        "Sharpe | DD | Trades"
    )
    print("-" * 128)

    for index, item in enumerate(results, start=1):
        fold = item["fold"]
        result = item["result"]
        weights = format_sleeve_weights(result.sleeves)

        print(
            f"{index:>4} | "
            f"{fold['test_start']}..{fold['test_end']} | "
            f"{weights:<28} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{result.result.closed_trades:>6}"
        )

    if results:
        avg_excess = (
            sum(item["result"].excess_return for item in results)
            / len(results)
        )
        avg_excess_equal_weight = (
            sum(item["result"].excess_vs_equal_weight for item in results)
            / len(results)
        )
        avg_drawdown = (
            sum(item["result"].result.max_drawdown for item in results)
            / len(results)
        )

        print("-" * 128)
        print(
            "Average | "
            f"excess_spy={format_percent(avg_excess)} | "
            f"excess_eq={format_percent(avg_excess_equal_weight)} | "
            f"drawdown={format_percent(avg_drawdown)}"
        )

    print(f"\nSaved walk-forward: {report_path}")


def print_multi_strategy_experiments(results, report_path):
    print("\nMULTI-STRATEGY EXPERIMENTS")
    print(
        "Weights | Return | SPY | Ex SPY | Ex EqWt | Sharpe | DD | "
        "Trades | Score"
    )
    print("-" * 100)

    for result in results[:10]:
        weights = ", ".join(
            f"{sleeve.name}:{format_percent(sleeve.weight)}"
            for sleeve in result.sleeves
        )

        print(
            f"{weights:<38} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{result.result.closed_trades:>6} | "
            f"{multi_strategy_quality_score(result):>5.2f}"
        )

    print(f"\nSaved experiments: {report_path}")


def format_sleeve_weights(sleeves):
    return ", ".join(
        f"{sleeve.name}:{int(round(sleeve.weight * 100))}%"
        for sleeve in sleeves
    )