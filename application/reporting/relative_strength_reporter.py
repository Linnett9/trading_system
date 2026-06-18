def format_percent(value):
    return f"{value * 100:.2f}%"


def print_relative_strength_experiments(results, report_path):
    print("\nRELATIVE STRENGTH EXPERIMENTS")
    print(
        "TopN | Mom | Rebal | Return | SPY | EqWt | Ex SPY | "
        "Ex EqWt | Sharpe | DD | Turnover | Cost"
    )
    print("-" * 118)

    for result in results[:10]:
        print(
            f"{result.config['top_n']:>4} | "
            f"{'/'.join(str(value) for value in result.config['momentum_periods']):<7} | "
            f"{result.config['rebalance_frequency']:<7} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.equal_weight_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{format_percent(result.turnover_percent):>8} | "
            f"{result.estimated_cost:>6.2f}"
        )

    print(f"\nSaved experiments: {report_path}")


def print_relative_strength_result(result, report_path):
    backtest = result.result
    utilization = backtest.capital_utilization
    analysis = backtest.trade_analysis

    print("\nRELATIVE STRENGTH PORTFOLIO")
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
        f"profit_factor={backtest.profit_factor:.2f} | "
        f"turnover={format_percent(result.turnover_percent)} | "
        f"cost={result.estimated_cost:.2f}"
    )

    print("Recent selections:")

    for selection in result.selections[-5:]:
        names = ", ".join(selection.symbols) if selection.symbols else "cash"
        print(f"  {selection.timestamp.date()} | {names}")

    print(f"Saved summary: {report_path}")