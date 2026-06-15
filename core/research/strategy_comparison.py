from copy import deepcopy
import csv
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.entities.strategy_comparison_result import StrategyComparisonResult
from core.research.walk_forward import WalkForwardTester


def strategy_grid(strategy_config: dict) -> dict:
    grid = strategy_config.get("parameter_grid")

    if grid:
        return grid

    return {
        key: [value]
        for key, value in strategy_config.items()
        if key != "name"
    }


def run_strategy_comparison_task(task) -> StrategyComparisonResult:
    config, strategy_config, symbol, candles = task
    research_config = config["research"]
    strategy_name = strategy_config["name"]
    comparison_config = deepcopy(config)
    comparison_config["strategy"].update(strategy_config)

    tester = WalkForwardTester(
        config=comparison_config,
        metric_name=research_config["optimization_metric"],
    )
    walk_forward_result = tester.run(
        candles=candles,
        symbol=symbol,
        folds=research_config["walk_forward_folds"],
        grid=strategy_grid(strategy_config),
    )

    return StrategyComparisonResult(
        strategy_name=strategy_name,
        symbol=symbol,
        walk_forward_result=walk_forward_result,
    )


class StrategyComparison:

    def __init__(
        self,
        config: dict,
        candles_by_symbol: dict,
    ):
        self.config = config
        self.candles_by_symbol = candles_by_symbol

    def run(self) -> list[StrategyComparisonResult]:
        research_config = self.config["research"]
        workers = research_config.get("parallel_workers", 1)
        parallel_mode = research_config.get("parallel_mode", "thread")
        tasks = []

        for strategy_config in research_config["strategy_comparison"]:
            for symbol, candles in self.candles_by_symbol.items():
                tasks.append((self.config, strategy_config, symbol, candles))

        if workers and workers > 1:
            executor_class = (
                ProcessPoolExecutor
                if parallel_mode == "process"
                else ThreadPoolExecutor
            )
            with executor_class(max_workers=workers) as executor:
                results = list(
                    executor.map(run_strategy_comparison_task, tasks)
                )
        else:
            results = [
                run_strategy_comparison_task(task)
                for task in tasks
            ]

        return self.rank(results)

    def rank(self, results):
        return sorted(
            results,
            key=lambda result: (
                result.qualified_score,
                result.composite_score,
                result.average_excess_return,
                result.average_test_sharpe,
                result.closed_trades,
                -result.average_max_drawdown,
                result.average_profit_factor,
                result.average_exposure_percent,
            ),
            reverse=True,
        )

    def to_table(self, results, limit: int | None = None):
        displayed_results = results[:limit] if limit else results
        rows = [
            "Strategy | Symbol | QScore | Score | Excess | Sharpe | Trades | Max DD | "
            "Profit Factor | Time In | Exposure | Cash | Pos $ | "
            "Bench Sh | Bench DD | Ex/Risk | B/S/H | Blocks | Exits | Passed"
        ]
        rows.append("-" * 215)

        for result in displayed_results:
            total_folds = len(result.walk_forward_result.folds)
            rows.append(
                f"{result.strategy_name:<18} | "
                f"{result.symbol:<6} | "
                f"{result.qualified_score:>6.2f} | "
                f"{result.composite_score:>5.2f} | "
                f"{result.average_excess_return * 100:>6.2f}% | "
                f"{result.average_test_sharpe:>6.2f} | "
                f"{result.closed_trades:>6} | "
                f"{result.average_max_drawdown * 100:>6.2f}% | "
                f"{result.average_profit_factor:>13.2f} | "
                f"{result.average_time_in_market_percent * 100:>7.2f}% | "
                f"{result.average_exposure_percent * 100:>8.2f}% | "
                f"{result.average_cash_percent * 100:>5.1f}% | "
                f"{result.average_position_value:>6.2f} | "
                f"{result.average_benchmark_sharpe:>8.2f} | "
                f"{result.average_benchmark_max_drawdown * 100:>8.2f}% | "
                f"{result.average_excess_return_per_unit_risk:>7.2f} | "
                f"{result.buy_signals}/{result.sell_signals}/{result.hold_signals} | "
                f"{result.duplicate_buy_skips}/{result.flat_sell_skips}/{result.risk_blocked_signals} | "
                f"{result.stop_loss_exits}/{result.take_profit_exits} | "
                f"{result.passed_folds}/{total_folds}"
            )

        return "\n".join(rows)

    def save_csv(
        self,
        results,
        report_dir: str = "reports/summary",
        filename: str = "latest_experiment.csv",
    ) -> Path:
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "strategy",
                    "symbol",
                    "qualified_score",
                    "composite_score",
                    "average_excess_return",
                    "average_test_sharpe",
                    "closed_trades",
                    "average_max_drawdown",
                    "average_profit_factor",
                    "average_time_in_market_percent",
                    "average_exposure_percent",
                    "average_cash_percent",
                    "average_position_value",
                    "average_benchmark_sharpe",
                    "average_benchmark_max_drawdown",
                    "average_excess_return_per_unit_risk",
                    "buy_signals",
                    "sell_signals",
                    "hold_signals",
                    "duplicate_buy_skips",
                    "flat_sell_skips",
                    "risk_blocked_signals",
                    "stop_loss_exits",
                    "take_profit_exits",
                    "passed_folds",
                    "total_folds",
                ],
            )
            writer.writeheader()

            for result in results:
                writer.writerow({
                    "strategy": result.strategy_name,
                    "symbol": result.symbol,
                    "qualified_score": result.qualified_score,
                    "composite_score": result.composite_score,
                    "average_excess_return": (
                        result.average_excess_return
                    ),
                    "average_test_sharpe": result.average_test_sharpe,
                    "closed_trades": result.closed_trades,
                    "average_max_drawdown": result.average_max_drawdown,
                    "average_profit_factor": result.average_profit_factor,
                    "average_time_in_market_percent": (
                        result.average_time_in_market_percent
                    ),
                    "average_exposure_percent": (
                        result.average_exposure_percent
                    ),
                    "average_cash_percent": result.average_cash_percent,
                    "average_position_value": (
                        result.average_position_value
                    ),
                    "average_benchmark_sharpe": (
                        result.average_benchmark_sharpe
                    ),
                    "average_benchmark_max_drawdown": (
                        result.average_benchmark_max_drawdown
                    ),
                    "average_excess_return_per_unit_risk": (
                        result.average_excess_return_per_unit_risk
                    ),
                    "buy_signals": result.buy_signals,
                    "sell_signals": result.sell_signals,
                    "hold_signals": result.hold_signals,
                    "duplicate_buy_skips": result.duplicate_buy_skips,
                    "flat_sell_skips": result.flat_sell_skips,
                    "risk_blocked_signals": result.risk_blocked_signals,
                    "stop_loss_exits": result.stop_loss_exits,
                    "take_profit_exits": result.take_profit_exits,
                    "passed_folds": result.passed_folds,
                    "total_folds": len(result.walk_forward_result.folds),
                })

        return path
