from core.research.dual_momentum_scoring import (
    risk_regime_score,
    dual_momentum_quality_score,
    dual_momentum_walk_forward_summary,
)


def format_percent(value):
    return f"{value * 100:.2f}%"


def print_dual_momentum_diagnosis(diagnosis, report_path):
    print("\nDUAL MOMENTUM DIAGNOSIS")
    print(
        "Year | Return | Avg Target | RiskOn | Fast | Partial | Cash | "
        "Top Symbols | Worst Months"
    )
    print("-" * 118)

    for year, year_item in sorted(diagnosis["annual"].items()):
        worst_months = ", ".join(
            f"{month['month']} {format_percent(month['bot_return'])}"
            for month in year_item["worst_months"]
        )

        print(
            f"{year} | "
            f"{format_percent(year_item['bot_return']):>7} | "
            f"{format_percent(year_item['average_exposure_target']):>10} | "
            f"{year_item['risk_on_months']:>6} | "
            f"{year_item['fast_reentry_months']:>4} | "
            f"{year_item['partial_risk_months']:>7} | "
            f"{year_item['cash_months']:>4} | "
            f"{', '.join(year_item['top_selected_symbols']):<24} | "
            f"{worst_months}"
        )

        top = ", ".join(
            f"{contributor['symbol']} "
            f"{format_percent(contributor['contribution'])}"
            for contributor in year_item["top_contributors"][:3]
        )
        worst = ", ".join(
            f"{contributor['symbol']} "
            f"{format_percent(contributor['contribution'])}"
            for contributor in year_item["worst_contributors"][:3]
        )

        print(
            f"     contributors | best: {top or 'n/a'} | "
            f"worst: {worst or 'n/a'}"
        )

        missed = ", ".join(
            f"{winner['symbol']} "
            f"{format_percent(winner['average_return'])}"
            for winner in year_item["missed_winners"][:3]
        )

        print(f"     missed winners | {missed or 'n/a'}")

    print(f"\nSaved diagnosis: {report_path}")


def print_dual_momentum_risk_regime_experiments(results, report_path):
    years = sorted({
        year
        for item in results
        for year in item["result"].annual_returns
    })
    year_headers = " | ".join(str(year) for year in years)

    print("\nDUAL MOMENTUM RISK REGIME PERFORMANCE")
    print(
        f"Config | Return | SPY | Ex SPY | EqWt | Ex EqWt | Sharpe | DD | "
        f"Turn | Score | {year_headers}"
    )
    print("-" * (116 + len(years) * 9))

    for item in results:
        result = item["result"]
        year_values = " | ".join(
            f"{format_percent(result.annual_returns.get(year, 0)):>7}"
            for year in years
        )

        print(
            f"{item['name']:<26} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.equal_weight_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{format_percent(result.annualized_turnover_percent):>7} | "
            f"{risk_regime_score(result):>5.2f} | "
            f"{year_values}"
        )

    print("\nDUAL MOMENTUM RISK REGIME CONFIG")
    print(
        "Regime Config | Mix | Off | Fallback | Chop | ChopExp | "
        "Quality | Cooldown | Decay | Lead | Bench | Rank"
    )
    print("-" * 140)

    for item in results:
        result = item["result"]

        print(
            f"{item['name']:<30} | "
            f"{format_percent(result.config['mixed_risk_exposure']):>5} | "
            f"{format_percent(result.config['risk_off_risk_exposure']):>5} | "
            f"{format_percent(result.config['fallback_allocation']):>8} | "
            f"{str(result.config['chop_filter_enabled']):>5} | "
            f"{format_percent(result.config['chop_risk_exposure']):>7} | "
            f"{str(result.config['quality_filter_enabled']):>7} | "
            f"{str(result.config['cooldown_enabled']):>8} | "
            f"{str(result.config['decay_exit_enabled']):>5} | "
            f"{str(result.config['leadership_filter_enabled']):>5} | "
            f"{format_percent(result.config['benchmark_sleeve_allocation']):>5} | "
            f"{result.config['ranking_score_mode']}"
        )

    if results:
        print(
            "\nInterpretation: "
            f"{dual_momentum_result_explanation(results[0]['result'])} "
            "Use this table to compare full-period configs, then confirm the "
            "winner with walk-forward."
        )

    print(f"\nSaved risk-regime experiments: {report_path}")


def print_dual_momentum_walk_forward(results, report_path):
    summary = dual_momentum_walk_forward_summary(results)

    print("\nDUAL MOMENTUM WALK-FORWARD")
    print(
        "Fold | Test | Mode | Weight | TopN | Mom | VolTgt | DDGuard | "
        "Return | SPY | Ex SPY | Ex EqWt | BullCap | Sharpe | DD | Calmar"
    )
    print("-" * 158)

    for index, item in enumerate(results, start=1):
        fold = item["fold"]
        result = item["result"]
        selected_config = (
            item["training_result"].config
            if item.get("training_result") is not None
            else result.config
        )
        bull_capture = (
            result.result.total_return / result.benchmark_return
            if result.benchmark_return > 0
            else 0
        )

        print(
            f"{index:>4} | "
            f"{fold['test_start']}..{fold['test_end']} | "
            f"{selected_config.get('selection_mode', 'ranked'):<12} | "
            f"{selected_config.get('weighting', 'equal'):<18} | "
            f"{selected_config['top_n']:>4} | "
            f"{'/'.join(str(value) for value in selected_config['momentum_periods']):<7} | "
            f"{str(selected_config['target_volatility']):<6} | "
            f"{str(selected_config['max_drawdown_guard']):<7} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.benchmark_return):>7} | "
            f"{format_percent(result.excess_return):>7} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{format_percent(bull_capture):>7} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{result.calmar:>6.2f}"
        )

    if results:
        print("-" * 158)
        print(
            "Average | "
            f"excess_spy={format_percent(summary['average_excess_return'])} | "
            f"excess_eq={format_percent(summary['average_excess_vs_equal_weight'])} | "
            f"drawdown={format_percent(summary['average_drawdown'])} | "
            f"worst_excess={format_percent(summary['worst_excess_return'])} | "
            f"consistency={format_percent(summary['consistency'])} | "
            f"bull_capture={format_percent(summary['average_bull_capture'])} | "
            f"worst_capture={format_percent(summary['worst_bull_capture'])} | "
            f"turnover={format_percent(summary['average_turnover'])} | "
            f"score={summary['score']:.2f}"
        )
        print(
            f"Interpretation: "
            f"{dual_momentum_walk_forward_explanation(summary)}"
        )

    print(f"\nSaved walk-forward: {report_path}")


def dual_momentum_result_explanation(result):
    parts = []

    if result.excess_return > 0:
        parts.append(
            f"beat SPY by {format_percent(result.excess_return)}"
        )
    else:
        parts.append(
            f"trailed SPY by {format_percent(abs(result.excess_return))}"
        )

    if result.excess_vs_equal_weight > 0:
        parts.append(
            "beat equal-weight by "
            f"{format_percent(result.excess_vs_equal_weight)}"
        )
    else:
        parts.append(
            "trailed equal-weight by "
            f"{format_percent(abs(result.excess_vs_equal_weight))}"
        )

    if result.result.max_drawdown <= 0.20:
        parts.append("drawdown is within the 20% target")
    else:
        parts.append("drawdown is above the 20% target")

    bull_capture = (
        result.result.total_return / result.benchmark_return
        if result.benchmark_return > 0
        else 0
    )

    if result.benchmark_return > 0:
        parts.append(f"bull capture is {format_percent(bull_capture)}")

    if result.annualized_turnover_percent > 7:
        parts.append("turnover is high")
    else:
        parts.append("turnover is controlled")

    return "; ".join(parts) + "."


def dual_momentum_walk_forward_explanation(summary):
    if (
        summary["average_excess_return"] > 0
        and summary["consistency"] >= 0.66
    ):
        verdict = "robust so far"
    elif summary["average_excess_return"] > 0:
        verdict = "promising but inconsistent"
    else:
        verdict = "not robust yet"

    return (
        f"{verdict}; average excess is "
        f"{format_percent(summary['average_excess_return'])}, worst fold is "
        f"{format_percent(summary['worst_excess_return'])}, consistency is "
        f"{format_percent(summary['consistency'])}, and bull capture is "
        f"{format_percent(summary['average_bull_capture'])}."
    )


def print_dual_momentum_experiments(results, report_path):
    print("\nDUAL MOMENTUM EXPERIMENTS")
    print(
        "Mode | Rank | Weight | MaxW | Mix | Off | Fallback | Decay | Chop | "
        "ChExp | Lead | Bench | Kill | TopN | Mom | Rebal | Asset SMA | "
        "Breadth | VolTgt | DD Guard | Return | CAGR | Calmar | Ex EqWt | "
        "Sharpe | DD | AnnTurn | Score"
    )
    print("-" * 258)

    for result in results[:10]:
        print(
            f"{result.config['selection_mode']:<12} | "
            f"{result.config['ranking_score_mode']:<16} | "
            f"{result.config['weighting']:<18} | "
            f"{format_percent(result.config['max_position_weight'] or 0):<6} | "
            f"{format_percent(result.config['mixed_risk_exposure']):<5} | "
            f"{format_percent(result.config['risk_off_risk_exposure']):<5} | "
            f"{format_percent(result.config['fallback_allocation']):<8} | "
            f"{str(result.config['decay_exit_enabled']):<5} | "
            f"{str(result.config['chop_filter_enabled']):<5} | "
            f"{format_percent(result.config['chop_risk_exposure']):<5} | "
            f"{str(result.config['leadership_filter_enabled']):<5} | "
            f"{format_percent(result.config['benchmark_sleeve_allocation']):<5} | "
            f"{str(result.config['strict_drawdown_kill_switch']):<5} | "
            f"{result.config['top_n']:>4} | "
            f"{'/'.join(str(value) for value in result.config['momentum_periods']):<7} | "
            f"{result.config['rebalance_frequency']:<7} | "
            f"{str(result.config['use_asset_trend_filter']):<9} | "
            f"{format_percent(result.config['min_breadth_percent']):<7} | "
            f"{str(result.config['target_volatility']):<6} | "
            f"{str(result.config['max_drawdown_guard']):<8} | "
            f"{format_percent(result.result.total_return):>7} | "
            f"{format_percent(result.cagr):>6} | "
            f"{result.calmar:>6.2f} | "
            f"{format_percent(result.excess_vs_equal_weight):>8} | "
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{format_percent(result.annualized_turnover_percent):>7} | "
            f"{dual_momentum_quality_score(result):>5.2f}"
        )

    print(f"\nSaved experiments: {report_path}")


def print_dual_momentum_result(result, report_path):
    backtest = result.result
    utilization = backtest.capital_utilization
    analysis = backtest.trade_analysis
    drawdown = result.drawdown_statistics

    print("\nDUAL MOMENTUM PORTFOLIO")
    print(
        f"return={format_percent(backtest.total_return)} | "
        f"benchmark={format_percent(result.benchmark_return)} | "
        f"excess={format_percent(result.excess_return)} | "
        f"equal_weight={format_percent(result.equal_weight_return)} | "
        f"ex_eq={format_percent(result.excess_vs_equal_weight)} | "
        f"cagr={format_percent(result.cagr)} | "
        f"calmar={result.calmar:.2f} | "
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
        f"ann_turn={format_percent(result.annualized_turnover_percent)} | "
        f"turn/rebal={format_percent(result.turnover_per_rebalance_percent)} | "
        f"cost={result.estimated_cost:.2f}"
    )

    print(
        "drawdown | "
        f"avg={format_percent(drawdown['average_drawdown'])} | "
        f"current={format_percent(drawdown['current_drawdown'])} | "
        f"longest={drawdown['longest_drawdown_days']}d"
    )

    print(f"Interpretation: {dual_momentum_result_explanation(result)}")
    print("Recent selections:")

    for selection in result.selections[-5:]:
        names = ", ".join(selection.symbols) if selection.symbols else "cash"
        regime = "risk-on" if selection.risk_on else "risk-off"
        print(f"  {selection.timestamp.date()} | {regime} | {names}")

    print(f"Saved summary: {report_path}")