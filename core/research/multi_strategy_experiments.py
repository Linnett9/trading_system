import csv
import json
from copy import deepcopy
from pathlib import Path

from core.research.multi_strategy_portfolio import (
    MultiStrategyPortfolioBacktester,
)


def run_multi_strategy_fold_optimization(
    config,
    multi_config,
    candles_by_symbol,
    start_at,
    end_at,
):
    results = []

    for candidate_config in multi_strategy_candidate_configs(multi_config):
        tester = build_multi_strategy_tester(config, candidate_config)
        results.append(
            tester.run(
                candles_by_symbol,
                start_at=start_at,
                end_at=end_at,
            )
        )

    return sorted(
        results,
        key=lambda result: (
            multi_strategy_quality_score(result),
            result.excess_return,
            -result.result.max_drawdown,
        ),
        reverse=True,
    )


def run_multi_strategy_experiments(config, multi_config, candles_by_symbol):
    results = []

    for candidate_config in multi_strategy_candidate_configs(multi_config):
        tester = build_multi_strategy_tester(config, candidate_config)
        results.append(tester.run(candles_by_symbol))

    return sorted(
        results,
        key=lambda result: (
            multi_strategy_quality_score(result),
            result.excess_return,
            result.result.sharpe,
            -result.result.max_drawdown,
        ),
        reverse=True,
    )


def multi_strategy_candidate_configs(multi_config):
    grid = multi_config.get("experiment_grid", {})

    sleeve_weights = grid.get(
        "sleeve_weights",
        [
            [
                sleeve.get("weight", 0)
                for sleeve in multi_config.get("sleeves", [])
            ]
        ],
    )

    for weights in sleeve_weights:
        candidate = deepcopy(multi_config)
        sleeves = deepcopy(candidate.get("sleeves", []))

        for index, weight in enumerate(weights):
            if index < len(sleeves):
                sleeves[index]["weight"] = weight

        candidate["sleeves"] = sleeves

        yield candidate


def multi_strategy_quality_score(result):
    penalty = 0

    if result.excess_return <= 0:
        penalty -= 1

    if result.result.max_drawdown > 0.25:
        penalty -= 0.50

    return (
        result.excess_return * 0.35
        + result.excess_vs_equal_weight * 0.25
        + result.result.sharpe * 0.20
        - result.result.max_drawdown * 0.30
        + penalty
    )


def multi_strategy_walk_forward_summary(results):
    if not results:
        return {
            "average_excess_return": 0,
            "worst_excess_return": 0,
            "average_excess_vs_equal_weight": 0,
            "average_drawdown": 0,
            "consistency": 0,
            "dispersion": 0,
            "score": 0,
        }

    excess_returns = [
        item["result"].excess_return
        for item in results
    ]
    equal_weight_excess = [
        item["result"].excess_vs_equal_weight
        for item in results
    ]
    drawdowns = [
        item["result"].result.max_drawdown
        for item in results
    ]

    avg_excess = sum(excess_returns) / len(excess_returns)
    worst_excess = min(excess_returns)
    avg_equal_weight_excess = (
        sum(equal_weight_excess) / len(equal_weight_excess)
    )
    avg_drawdown = sum(drawdowns) / len(drawdowns)

    consistency = (
        sum(1 for value in excess_returns if value > 0)
        / len(excess_returns)
    )

    dispersion = (
        sum((value - avg_excess) ** 2 for value in excess_returns)
        / len(excess_returns)
    ) ** 0.5

    score = (
        avg_excess * 0.40
        + worst_excess * 0.30
        + avg_equal_weight_excess * 0.15
        + consistency * 0.15
        - avg_drawdown * 0.25
        - dispersion * 0.10
    )

    return {
        "average_excess_return": avg_excess,
        "worst_excess_return": worst_excess,
        "average_excess_vs_equal_weight": avg_equal_weight_excess,
        "average_drawdown": avg_drawdown,
        "consistency": consistency,
        "dispersion": dispersion,
        "score": score,
    }


def save_multi_strategy_experiments(
    results,
    report_dir,
    filename="multi_strategy_experiments.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sleeve_weights",
                "return",
                "benchmark_return",
                "equal_weight_return",
                "excess_vs_benchmark",
                "excess_vs_equal_weight",
                "sharpe",
                "max_drawdown",
                "closed_trades",
                "diagnostics",
                "quality_score",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                "sleeve_weights": [
                    {
                        "name": sleeve.name,
                        "weight": sleeve.weight,
                    }
                    for sleeve in result.sleeves
                ],
                "return": result.result.total_return,
                "benchmark_return": result.benchmark_return,
                "equal_weight_return": result.equal_weight_return,
                "excess_vs_benchmark": result.excess_return,
                "excess_vs_equal_weight": result.excess_vs_equal_weight,
                "sharpe": result.result.sharpe,
                "max_drawdown": result.result.max_drawdown,
                "closed_trades": result.result.closed_trades,
                "diagnostics": result.diagnostics,
                "quality_score": multi_strategy_quality_score(result),
            })

    return path


def build_multi_strategy_tester(config, multi_config):
    return MultiStrategyPortfolioBacktester(
        starting_equity=config["backtest"]["starting_equity"],
        sleeves=multi_config.get("sleeves", []),
        benchmark_symbol=multi_config.get("benchmark_symbol", "SPY"),
        warmup_days=multi_config.get("warmup_days", 500),
    )


def save_multi_strategy_walk_forward(
    results,
    report_dir,
    filename="multi_strategy_walk_forward.json",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename

    payload = {
        "summary": multi_strategy_walk_forward_summary(results),
        "folds": [],
    }

    for item in results:
        result = item["result"]
        training_result = item.get("training_result")

        payload["folds"].append({
            "fold": item["fold"],
            "return": result.result.total_return,
            "benchmark_return": result.benchmark_return,
            "equal_weight_return": result.equal_weight_return,
            "excess_return": result.excess_return,
            "excess_vs_equal_weight": result.excess_vs_equal_weight,
            "sharpe": result.result.sharpe,
            "max_drawdown": result.result.max_drawdown,
            "closed_trades": result.result.closed_trades,
            "open_trades": result.result.open_trades,
            "selected_config": (
                training_result.config
                if training_result is not None
                else result.config
            ),
            "train_score": (
                multi_strategy_quality_score(training_result)
                if training_result is not None
                else None
            ),
            "diagnostics": result.diagnostics,
            "sleeves": [
                {
                    "name": sleeve.name,
                    "weight": sleeve.weight,
                    "return": sleeve.result.total_return,
                    "sharpe": sleeve.result.sharpe,
                    "max_drawdown": sleeve.result.max_drawdown,
                }
                for sleeve in result.sleeves
            ],
        })

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return path