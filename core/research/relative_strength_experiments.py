import csv
from itertools import product
from pathlib import Path

from core.research.relative_strength_portfolio import (
    RelativeStrengthPortfolioBacktester,
)


def run_relative_strength_experiments(
    config,
    relative_config,
    candles_by_symbol,
):
    grid = relative_config.get("experiment_grid", {})

    top_values = grid.get(
        "top_n",
        [relative_config.get("top_n", 2)],
    )
    rebalance_values = grid.get(
        "rebalance_frequency",
        [relative_config.get("rebalance_frequency", "monthly")],
    )
    momentum_values = grid.get(
        "momentum_periods",
        [relative_config.get("momentum_periods", [63, 126])],
    )

    results = []

    for top_n, rebalance, momentum_periods in product(
        top_values,
        rebalance_values,
        momentum_values,
    ):
        tester = RelativeStrengthPortfolioBacktester(
            starting_equity=config["backtest"]["starting_equity"],
            top_n=top_n,
            momentum_periods=momentum_periods,
            sma_period=relative_config.get("sma_period", 200),
            rebalance_frequency=rebalance,
            target_exposure=relative_config.get("target_exposure", 1.0),
            benchmark_symbol=relative_config.get("benchmark_symbol", "SPY"),
            transaction_cost_bps=relative_config.get(
                "transaction_cost_bps",
                0,
            ),
        )

        results.append(tester.run(candles_by_symbol))

    return sorted(
        results,
        key=lambda result: (
            result.excess_vs_equal_weight,
            result.excess_return,
            result.result.sharpe,
            -result.result.max_drawdown,
        ),
        reverse=True,
    )


def save_relative_strength_experiments(
    results,
    report_dir,
    filename="relative_strength_experiments.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "top_n",
                "momentum_periods",
                "rebalance_frequency",
                "return",
                "benchmark_return",
                "equal_weight_return",
                "excess_vs_benchmark",
                "excess_vs_equal_weight",
                "sharpe",
                "max_drawdown",
                "turnover_percent",
                "rebalance_count",
                "estimated_cost",
                "closed_trades",
                "open_trades",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                "top_n": result.config["top_n"],
                "momentum_periods": result.config["momentum_periods"],
                "rebalance_frequency": result.config["rebalance_frequency"],
                "return": result.result.total_return,
                "benchmark_return": result.benchmark_return,
                "equal_weight_return": result.equal_weight_return,
                "excess_vs_benchmark": result.excess_return,
                "excess_vs_equal_weight": result.excess_vs_equal_weight,
                "sharpe": result.result.sharpe,
                "max_drawdown": result.result.max_drawdown,
                "turnover_percent": result.turnover_percent,
                "rebalance_count": result.rebalance_count,
                "estimated_cost": result.estimated_cost,
                "closed_trades": result.result.closed_trades,
                "open_trades": result.result.open_trades,
            })

    return path