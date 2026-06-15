from core.entities.experiment_result import ExperimentResult


class ExperimentReporter:

    def __init__(self):
        self.results = []

    def add_backtest_result(self, symbol, result):
        self.results.append(
            ExperimentResult(
                symbol=symbol,
                source="backtest",
                return_value=result.total_return,
                walk_forward_sharpe=result.sharpe,
                max_drawdown=result.max_drawdown,
                closed_trades=result.closed_trades,
                profit_factor=result.profit_factor,
                time_in_market_percent=(
                    result.trade_analysis.time_in_market_percent
                ),
                average_trade_duration_days=(
                    result.trade_analysis.average_trade_duration_days
                ),
                average_exposure_percent=(
                    result.capital_utilization.average_exposure_percent
                ),
                average_cash_percent=(
                    result.capital_utilization.average_cash_percent
                ),
                average_position_value=(
                    result.capital_utilization.average_position_value
                ),
            )
        )

    def add_optimization_results(self, symbol, results):
        if not results:
            return

        best = results[0]
        result = best.result

        self.results.append(
            ExperimentResult(
                symbol=symbol,
                source="optimization",
                return_value=result.total_return,
                walk_forward_sharpe=result.sharpe,
                max_drawdown=result.max_drawdown,
                closed_trades=result.closed_trades,
                profit_factor=result.profit_factor,
                time_in_market_percent=(
                    result.trade_analysis.time_in_market_percent
                ),
                average_trade_duration_days=(
                    result.trade_analysis.average_trade_duration_days
                ),
                average_exposure_percent=(
                    result.capital_utilization.average_exposure_percent
                ),
                average_cash_percent=(
                    result.capital_utilization.average_cash_percent
                ),
                average_position_value=(
                    result.capital_utilization.average_position_value
                ),
            )
        )

    def add_walk_forward_result(self, result):
        folds = result.folds

        if folds:
            max_drawdown = sum(
                fold.test_result.max_drawdown
                for fold in folds
            ) / len(folds)
            closed_trades = sum(
                fold.test_result.closed_trades
                for fold in folds
            )
            profit_factor = sum(
                fold.test_result.profit_factor
                for fold in folds
            ) / len(folds)
            time_in_market_percent = sum(
                fold.test_result.trade_analysis.time_in_market_percent
                for fold in folds
            ) / len(folds)
            average_trade_duration_days = sum(
                fold.test_result.trade_analysis.average_trade_duration_days
                for fold in folds
            ) / len(folds)
            average_exposure_percent = sum(
                fold.test_result.capital_utilization.average_exposure_percent
                for fold in folds
            ) / len(folds)
            average_cash_percent = sum(
                fold.test_result.capital_utilization.average_cash_percent
                for fold in folds
            ) / len(folds)
            average_position_value = sum(
                fold.test_result.capital_utilization.average_position_value
                for fold in folds
            ) / len(folds)
            benchmark_sharpe = sum(
                fold.benchmark.sharpe
                for fold in folds
            ) / len(folds)
            benchmark_max_drawdown = sum(
                fold.benchmark.max_drawdown
                for fold in folds
            ) / len(folds)
            excess_return_per_unit_risk = sum(
                fold.excess_return_per_unit_risk
                for fold in folds
            ) / len(folds)
            passed_folds = sum(1 for fold in folds if fold.passed)
        else:
            max_drawdown = 0
            closed_trades = 0
            profit_factor = 0
            time_in_market_percent = 0
            average_trade_duration_days = 0
            average_exposure_percent = 0
            average_cash_percent = 1
            average_position_value = 0
            benchmark_sharpe = 0
            benchmark_max_drawdown = 0
            excess_return_per_unit_risk = 0
            passed_folds = 0

        self.results.append(
            ExperimentResult(
                symbol=result.symbol,
                source="walk_forward",
                return_value=result.average_test_return,
                walk_forward_sharpe=result.average_test_sharpe,
                max_drawdown=max_drawdown,
                closed_trades=closed_trades,
                profit_factor=profit_factor,
                time_in_market_percent=time_in_market_percent,
                average_trade_duration_days=average_trade_duration_days,
                average_exposure_percent=average_exposure_percent,
                average_cash_percent=average_cash_percent,
                average_position_value=average_position_value,
                benchmark_sharpe=benchmark_sharpe,
                benchmark_max_drawdown=benchmark_max_drawdown,
                excess_return_per_unit_risk=excess_return_per_unit_risk,
                passed_folds=passed_folds,
                total_folds=len(folds),
            )
        )

    def ranked(self):
        return sorted(
            self.results,
            key=lambda result: (
                result.walk_forward_sharpe,
                -result.max_drawdown,
                result.closed_trades,
                result.return_value,
                result.profit_factor,
                result.average_exposure_percent,
            ),
            reverse=True,
        )

    def to_table(self):
        rows = [
            "Symbol | Source | Return | Sharpe | Max DD | Trades | "
            "Profit Factor | Time In | Exposure | Cash | Pos $ | "
            "Bench Sh | Bench DD | Ex/Risk | Passed"
        ]
        rows.append("-" * 155)

        for result in self.ranked():
            rows.append(
                f"{result.symbol:<6} | "
                f"{result.source:<12} | "
                f"{result.return_value * 100:>6.2f}% | "
                f"{result.walk_forward_sharpe:>6.2f} | "
                f"{result.max_drawdown * 100:>6.2f}% | "
                f"{result.closed_trades:>6} | "
                f"{result.profit_factor:>13.2f} | "
                f"{result.time_in_market_percent * 100:>7.2f}% | "
                f"{result.average_exposure_percent * 100:>8.2f}% | "
                f"{result.average_cash_percent * 100:>5.1f}% | "
                f"{result.average_position_value:>6.2f} | "
                f"{result.benchmark_sharpe:>8.2f} | "
                f"{result.benchmark_max_drawdown * 100:>8.2f}% | "
                f"{result.excess_return_per_unit_risk:>7.2f} | "
                f"{result.passed_folds}/{result.total_folds}"
            )

        return "\n".join(rows)
