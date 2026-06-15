import argparse
import csv
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import product
import json
from pathlib import Path

from config.config_loader import load_config
from core.entities.candle import Candle
from core.research.backtest_runner import run_backtest
from core.research.dual_momentum_portfolio import (
    DualMomentumPortfolioBacktester,
)
from core.research.experiment_reporter import ExperimentReporter
from core.research.multi_strategy_portfolio import (
    MultiStrategyPortfolioBacktester,
)
from core.research.parameter_optimizer import ParameterOptimizer
from core.research.relative_strength_portfolio import (
    RelativeStrengthPortfolioBacktester,
)
from core.research.strategy_comparison import StrategyComparison
from core.research.walk_forward import WalkForwardTester


def load_candles(symbol, config, feed):
    backtest_config = config["backtest"]
    end = datetime.utcnow()
    start = end - timedelta(days=365 * backtest_config["years"])
    cache_config = config.get("cache", {})
    cache_enabled = cache_config.get("enabled", False)
    cache_path = data_cache_path(symbol, backtest_config, cache_config)

    if cache_enabled and cache_path and cache_path.exists():
        cached_candles = read_candle_cache(cache_path)
        if cached_candles is not None:
            return cached_candles

    candles = feed.get_historical_bars(
        symbol=symbol,
        timeframe=backtest_config["timeframe"],
        start=start,
        end=end,
    )

    if cache_enabled and cache_path:
        write_candle_cache(cache_path, candles)

    return candles


def data_cache_path(symbol, backtest_config, cache_config):
    directory = Path(cache_config.get("data_dir", "cache/data"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    filename = (
        f"{symbol}_{backtest_config['timeframe']}_"
        f"{backtest_config['years']}y.json"
    )
    return directory / filename


def read_candle_cache(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

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
    try:
        path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        return



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


def run_relative_strength(config, feed, run_experiments=False):
    relative_config = config["research"].get("relative_strength", {})
    symbols = relative_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    if run_experiments:
        results = run_relative_strength_experiments(
            config=config,
            relative_config=relative_config,
            candles_by_symbol=candles_by_symbol,
        )
        report_path = save_relative_strength_experiments(
            results,
            report_dir=config["reports"]["summary_dir"],
        )
        print_relative_strength_experiments(results, report_path)
        return

    tester = RelativeStrengthPortfolioBacktester(
        starting_equity=config["backtest"]["starting_equity"],
        top_n=relative_config.get("top_n", 2),
        momentum_periods=relative_config.get("momentum_periods", [63, 126]),
        sma_period=relative_config.get("sma_period", 200),
        rebalance_frequency=relative_config.get(
            "rebalance_frequency",
            "monthly",
        ),
        target_exposure=relative_config.get("target_exposure", 1.0),
        benchmark_symbol=relative_config.get("benchmark_symbol", "SPY"),
        transaction_cost_bps=relative_config.get("transaction_cost_bps", 0),
    )
    result = tester.run(candles_by_symbol)
    report_path = result.save_json(
        report_dir=config["reports"]["summary_dir"],
    )
    print_relative_strength_result(result, report_path)


def run_dual_momentum(config, feed, run_experiments=False):
    dual_config = config["research"].get("dual_momentum", {})
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    if run_experiments:
        results = run_dual_momentum_experiments(
            config=config,
            dual_config=dual_config,
            candles_by_symbol=candles_by_symbol,
        )
        report_path = save_dual_momentum_experiments(
            results,
            report_dir=config["reports"]["summary_dir"],
        )
        print_dual_momentum_experiments(results, report_path)
        return

    tester = build_dual_momentum_tester(config, dual_config)
    result = tester.run(candles_by_symbol)
    report_path = result.save_json(
        report_dir=config["reports"]["summary_dir"],
    )
    print_dual_momentum_result(result, report_path)


def run_dual_momentum_risk_regime_experiments(config, feed):
    dual_config = config["research"].get("dual_momentum", {})
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }
    results = []

    for candidate in dual_momentum_risk_regime_configs(dual_config):
        tester = build_dual_momentum_tester(config, candidate["config"])
        results.append({
            "name": candidate["name"],
            "result": tester.run(candles_by_symbol),
        })

    results = sorted(
        results,
        key=lambda item: (
            risk_regime_score(item["result"]),
            item["result"].excess_return,
            -item["result"].result.max_drawdown,
        ),
        reverse=True,
    )
    report_path = save_dual_momentum_risk_regime_experiments(
        results,
        report_dir=config["reports"]["summary_dir"],
    )
    print_dual_momentum_risk_regime_experiments(results, report_path)


def dual_momentum_risk_regime_configs(dual_config):
    grid = dual_config.get("risk_regime_experiments", [])

    if not grid:
        grid = [
            {
                "name": "baseline_inverse_vol",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "defensive_assets",
                "overrides": {
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "cash_risk_off",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "scaled_exposure",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": False,
                },
            },
            {
                "name": "fast_reentry",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "binary",
                    "fast_reentry_enabled": True,
                },
            },
            {
                "name": "scaled_plus_fast_reentry",
                "overrides": {
                    "risk_off_symbols": [],
                    "risk_regime_mode": "scaled",
                    "mixed_risk_exposure": 0.50,
                    "risk_off_risk_exposure": 0.25,
                    "fast_reentry_enabled": True,
                },
            },
        ]

    for item in grid:
        candidate = deepcopy(dual_config)
        candidate.update(item.get("overrides", {}))
        yield {
            "name": item["name"],
            "config": candidate,
        }


def risk_regime_score(result):
    return (
        result.excess_return * 0.35
        + result.excess_vs_equal_weight * 0.25
        + result.result.sharpe * 0.20
        - result.result.max_drawdown * 0.30
        - result.annualized_turnover_percent * 0.02
    )


def run_dual_momentum_walk_forward(config, feed):
    dual_config = config["research"].get("dual_momentum", {})
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }
    folds = dual_config.get(
        "walk_forward_folds",
        config["research"].get("walk_forward_folds", []),
    )
    results = []

    for fold in folds:
        training_results = run_dual_momentum_fold_optimization(
            config=config,
            dual_config=dual_config,
            candles_by_symbol=candles_by_symbol,
            start_at=parse_config_date(fold["train_start"]),
            end_at=parse_config_date(fold["train_end"]),
        )
        best_training = training_results[0] if training_results else None
        selected_config = (
            best_training.config
            if best_training is not None
            else dual_config
        )
        tester = build_dual_momentum_tester(config, selected_config)
        test_result = tester.run(
            candles_by_symbol,
            start_at=parse_config_date(fold["test_start"]),
            end_at=parse_config_date(fold["test_end"]),
        )
        results.append({
            "fold": fold,
            "training_result": best_training,
            "result": test_result,
        })

    report_path = save_dual_momentum_walk_forward(
        results,
        report_dir=config["reports"]["summary_dir"],
    )
    print_dual_momentum_walk_forward(results, report_path)


def run_multi_strategy(config, feed, run_experiments=False):
    multi_config = config["research"].get("multi_strategy", {})
    symbols = multi_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    if run_experiments:
        results = run_multi_strategy_experiments(
            config=config,
            multi_config=multi_config,
            candles_by_symbol=candles_by_symbol,
        )
        report_path = save_multi_strategy_experiments(
            results,
            report_dir=config["reports"]["summary_dir"],
        )
        print_multi_strategy_experiments(results, report_path)
        return

    tester = build_multi_strategy_tester(config, multi_config)
    result = tester.run(candles_by_symbol)
    report_path = result.save_json(
        report_dir=config["reports"]["summary_dir"],
    )
    print_multi_strategy_result(result, report_path)


def run_multi_strategy_experiments_mode(config, feed):
    multi_config = config["research"].get("multi_strategy", {})
    symbols = multi_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }
    results = run_multi_strategy_experiments(
        config=config,
        multi_config=multi_config,
        candles_by_symbol=candles_by_symbol,
    )
    report_path = save_multi_strategy_experiments(
        results,
        report_dir=config["reports"]["summary_dir"],
    )
    print_multi_strategy_experiments(results, report_path)


def run_multi_strategy_walk_forward(config, feed):
    multi_config = config["research"].get("multi_strategy", {})
    symbols = multi_config.get("symbols", config["backtest"]["symbols"])
    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }
    folds = multi_config.get(
        "walk_forward_folds",
        config["research"].get("walk_forward_folds", []),
    )
    results = []

    for fold in folds:
        training_results = run_multi_strategy_fold_optimization(
            config=config,
            multi_config=multi_config,
            candles_by_symbol=candles_by_symbol,
            start_at=parse_config_date(fold["train_start"]),
            end_at=parse_config_date(fold["train_end"]),
        )
        best_training = training_results[0] if training_results else None
        selected_config = (
            best_training.config
            if best_training is not None
            else multi_config
        )
        tester = build_multi_strategy_tester(config, selected_config)
        result = tester.run(
            candles_by_symbol,
            start_at=parse_config_date(fold["test_start"]),
            end_at=parse_config_date(fold["test_end"]),
        )
        results.append({
            "fold": fold,
            "training_result": best_training,
            "result": result,
        })

    report_path = save_multi_strategy_walk_forward(
        results,
        report_dir=config["reports"]["summary_dir"],
    )
    print_multi_strategy_walk_forward(results, report_path)


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
    payload = []

    for item in results:
        result = item["result"]
        payload.append({
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
                item["training_result"].config
                if item.get("training_result") is not None
                else result.config
            ),
            "train_score": (
                multi_strategy_quality_score(item["training_result"])
                if item.get("training_result") is not None
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


def run_dual_momentum_fold_optimization(
    config,
    dual_config,
    candles_by_symbol,
    start_at,
    end_at,
):
    results = []

    for candidate_config in dual_momentum_candidate_configs(dual_config):
        tester = build_dual_momentum_tester(config, candidate_config)
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
            dual_momentum_quality_score(result),
            result.calmar,
            result.excess_vs_equal_weight,
            -result.result.max_drawdown,
        ),
        reverse=True,
    )


def dual_momentum_candidate_configs(dual_config):
    grid = dual_config.get("experiment_grid", {})
    top_values = grid.get("top_n", [dual_config.get("top_n", 3)])
    rebalance_values = grid.get(
        "rebalance_frequency",
        [dual_config.get("rebalance_frequency", "monthly")],
    )
    momentum_values = grid.get(
        "momentum_periods",
        [dual_config.get("momentum_periods", [126, 252])],
    )
    asset_filter_values = grid.get(
        "use_asset_trend_filter",
        [dual_config.get("use_asset_trend_filter", True)],
    )
    volatility_values = grid.get(
        "target_volatility",
        [dual_config.get("target_volatility")],
    )
    drawdown_values = grid.get(
        "max_drawdown_guard",
        [dual_config.get("max_drawdown_guard")],
    )
    breadth_values = grid.get(
        "min_breadth_percent",
        [dual_config.get("min_breadth_percent", 0)],
    )
    selection_mode_values = grid.get(
        "selection_mode",
        [dual_config.get("selection_mode", "ranked")],
    )
    weighting_values = grid.get(
        "weighting",
        [dual_config.get("weighting", "equal")],
    )
    max_position_weight_values = grid.get(
        "max_position_weight",
        [dual_config.get("max_position_weight")],
    )
    strict_kill_switch_values = grid.get(
        "strict_drawdown_kill_switch",
        [dual_config.get("strict_drawdown_kill_switch", False)],
    )

    for (
        top_n,
        rebalance,
        momentum_periods,
        use_asset_filter,
        target_volatility,
        max_drawdown_guard,
        min_breadth_percent,
        selection_mode,
        weighting,
        max_position_weight,
        strict_drawdown_kill_switch,
    ) in product(
        top_values,
        rebalance_values,
        momentum_values,
        asset_filter_values,
        volatility_values,
        drawdown_values,
        breadth_values,
        selection_mode_values,
        weighting_values,
        max_position_weight_values,
        strict_kill_switch_values,
    ):
        candidate = deepcopy(dual_config)
        candidate.update({
            "top_n": top_n,
            "momentum_periods": momentum_periods,
            "rebalance_frequency": rebalance,
            "use_asset_trend_filter": use_asset_filter,
            "target_volatility": target_volatility,
            "max_drawdown_guard": max_drawdown_guard,
            "min_breadth_percent": min_breadth_percent,
            "selection_mode": selection_mode,
            "weighting": weighting,
            "max_position_weight": max_position_weight,
            "strict_drawdown_kill_switch": strict_drawdown_kill_switch,
        })
        yield candidate


def build_dual_momentum_tester(config, dual_config):
    return DualMomentumPortfolioBacktester(
        starting_equity=config["backtest"]["starting_equity"],
        top_n=dual_config.get("top_n", 3),
        momentum_periods=dual_config.get("momentum_periods", [126, 252]),
        regime_symbol=dual_config.get("regime_symbol", "SPY"),
        regime_sma_period=dual_config.get("regime_sma_period", 200),
        rebalance_frequency=dual_config.get(
            "rebalance_frequency",
            "monthly",
        ),
        target_exposure=dual_config.get("target_exposure", 1.0),
        benchmark_symbol=dual_config.get("benchmark_symbol", "SPY"),
        transaction_cost_bps=dual_config.get("transaction_cost_bps", 2.0),
        use_asset_trend_filter=dual_config.get(
            "use_asset_trend_filter",
            True,
        ),
        asset_sma_period=dual_config.get("asset_sma_period", 200),
        target_volatility=dual_config.get("target_volatility"),
        volatility_lookback=dual_config.get("volatility_lookback", 63),
        max_drawdown_guard=dual_config.get("max_drawdown_guard"),
        drawdown_guard_cooldown=dual_config.get(
            "drawdown_guard_cooldown",
            1,
        ),
        min_breadth_percent=dual_config.get("min_breadth_percent", 0),
        selection_mode=dual_config.get("selection_mode", "ranked"),
        weighting=dual_config.get("weighting", "equal"),
        max_position_weight=dual_config.get("max_position_weight"),
        weight_volatility_lookback=dual_config.get(
            "weight_volatility_lookback",
            dual_config.get("volatility_lookback", 63),
        ),
        strict_drawdown_kill_switch=dual_config.get(
            "strict_drawdown_kill_switch",
            False,
        ),
        risk_off_symbols=dual_config.get("risk_off_symbols", []),
        risk_off_top_n=dual_config.get("risk_off_top_n", 1),
        risk_off_momentum_periods=dual_config.get(
            "risk_off_momentum_periods",
        ),
        risk_regime_mode=dual_config.get("risk_regime_mode", "binary"),
        mixed_risk_exposure=dual_config.get("mixed_risk_exposure", 0.50),
        risk_off_risk_exposure=dual_config.get("risk_off_risk_exposure", 0),
        fast_reentry_enabled=dual_config.get("fast_reentry_enabled", False),
        fast_reentry_sma_period=dual_config.get(
            "fast_reentry_sma_period",
            100,
        ),
        fast_reentry_momentum_period=dual_config.get(
            "fast_reentry_momentum_period",
            63,
        ),
        fast_reentry_breadth_percent=dual_config.get(
            "fast_reentry_breadth_percent",
            0.60,
        ),
    )


def parse_config_date(value):
    return datetime.fromisoformat(value)


def save_dual_momentum_walk_forward(
    results,
    report_dir,
    filename="dual_momentum_walk_forward.json",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    payload = []

    for item in results:
        result = item["result"]
        training_result = item.get("training_result")
        payload.append({
            "fold": item["fold"],
            "selected_config": (
                training_result.config
                if training_result is not None
                else result.config
            ),
            "train_return": (
                training_result.result.total_return
                if training_result is not None
                else None
            ),
            "train_score": (
                dual_momentum_quality_score(training_result)
                if training_result is not None
                else None
            ),
            "return": result.result.total_return,
            "benchmark_return": result.benchmark_return,
            "equal_weight_return": result.equal_weight_return,
            "excess_return": result.excess_return,
            "excess_vs_equal_weight": result.excess_vs_equal_weight,
            "sharpe": result.result.sharpe,
            "max_drawdown": result.result.max_drawdown,
            "cagr": result.cagr,
            "calmar": result.calmar,
            "annualized_turnover_percent": (
                result.annualized_turnover_percent
            ),
            "cost_drag_percent": result.cost_drag_percent,
            "closed_trades": result.result.closed_trades,
            "open_trades": result.result.open_trades,
        })

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_dual_momentum_risk_regime_experiments(
    results,
    report_dir,
    filename="dual_momentum_risk_regimes.csv",
):
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename

    years = sorted({
        year
        for item in results
        for year in item["result"].annual_returns
    })

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "return",
                "benchmark_return",
                "equal_weight_return",
                "excess_vs_benchmark",
                "excess_vs_equal_weight",
                "sharpe",
                "max_drawdown",
                "cagr",
                "calmar",
                "annualized_turnover_percent",
                "score",
            ]
            + [str(year) for year in years],
        )
        writer.writeheader()

        for item in results:
            result = item["result"]
            row = {
                "name": item["name"],
                "return": result.result.total_return,
                "benchmark_return": result.benchmark_return,
                "equal_weight_return": result.equal_weight_return,
                "excess_vs_benchmark": result.excess_return,
                "excess_vs_equal_weight": result.excess_vs_equal_weight,
                "sharpe": result.result.sharpe,
                "max_drawdown": result.result.max_drawdown,
                "cagr": result.cagr,
                "calmar": result.calmar,
                "annualized_turnover_percent": (
                    result.annualized_turnover_percent
                ),
                "score": risk_regime_score(result),
            }
            row.update({
                str(year): result.annual_returns.get(year, 0)
                for year in years
            })
            writer.writerow(row)

    return path


def print_dual_momentum_risk_regime_experiments(results, report_path):
    years = sorted({
        year
        for item in results
        for year in item["result"].annual_returns
    })
    year_headers = " | ".join(str(year) for year in years)

    print("\nDUAL MOMENTUM RISK REGIME EXPERIMENTS")
    print(
        "Config | Return | SPY | Ex SPY | EqWt | Ex EqWt | Sharpe | DD | "
        f"Turn | Score | {year_headers}"
    )
    print("-" * (112 + len(years) * 9))

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

    print(f"\nSaved risk-regime experiments: {report_path}")


def print_dual_momentum_walk_forward(results, report_path):
    print("\nDUAL MOMENTUM WALK-FORWARD")
    print(
        "Fold | Test | Mode | Weight | TopN | Mom | VolTgt | DDGuard | "
        "Return | SPY | Ex SPY | Ex EqWt | Sharpe | DD | Calmar"
    )
    print("-" * 146)

    for index, item in enumerate(results, start=1):
        fold = item["fold"]
        result = item["result"]
        selected_config = (
            item["training_result"].config
            if item.get("training_result") is not None
            else result.config
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
            f"{result.result.sharpe:>6.2f} | "
            f"{format_percent(result.result.max_drawdown):>6} | "
            f"{result.calmar:>6.2f}"
        )

    if results:
        avg_excess = sum(
            item["result"].excess_return for item in results
        ) / len(results)
        avg_excess_equal_weight = sum(
            item["result"].excess_vs_equal_weight for item in results
        ) / len(results)
        avg_drawdown = sum(
            item["result"].result.max_drawdown for item in results
        ) / len(results)
        print("-" * 146)
        print(
            "Average | "
            f"excess_spy={format_percent(avg_excess)} | "
            f"excess_eq={format_percent(avg_excess_equal_weight)} | "
            f"drawdown={format_percent(avg_drawdown)}"
        )

    print(f"\nSaved walk-forward: {report_path}")


def run_dual_momentum_experiments(config, dual_config, candles_by_symbol):
    results = []

    for candidate_config in dual_momentum_candidate_configs(dual_config):
        tester = build_dual_momentum_tester(config, candidate_config)
        results.append(tester.run(candles_by_symbol))

    return sorted(
        results,
        key=lambda result: (
            dual_momentum_quality_score(result),
            result.calmar,
            result.excess_vs_equal_weight,
            -result.result.max_drawdown,
        ),
        reverse=True,
    )


def dual_momentum_quality_score(result):
    hard_penalty = 0

    if result.excess_vs_equal_weight <= 0:
        hard_penalty -= 1.0

    if result.result.max_drawdown > 0.25:
        hard_penalty -= 0.75

    if result.annualized_turnover_percent > 15:
        hard_penalty -= 0.25

    return (
        result.excess_vs_equal_weight * 0.35
        + result.excess_return * 0.25
        + result.result.sharpe * 0.15
        + result.calmar * 0.15
        - result.result.max_drawdown * 0.20
        - result.annualized_turnover_percent * 0.04
        - result.cost_drag_percent * 0.10
        + hard_penalty
    )


def save_dual_momentum_experiments(
    results,
    report_dir,
    filename="dual_momentum_experiments.csv",
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
                "selection_mode",
                "weighting",
                "max_position_weight",
                "strict_drawdown_kill_switch",
                "use_asset_trend_filter",
                "min_breadth_percent",
                "target_volatility",
                "max_drawdown_guard",
                "return",
                "cagr",
                "calmar",
                "benchmark_return",
                "equal_weight_return",
                "excess_vs_benchmark",
                "excess_vs_equal_weight",
                "sharpe",
                "max_drawdown",
                "turnover_percent",
                "annualized_turnover_percent",
                "turnover_per_rebalance_percent",
                "rebalance_count",
                "estimated_cost",
                "cost_drag_percent",
                "closed_trades",
                "open_trades",
                "quality_score",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                "top_n": result.config["top_n"],
                "momentum_periods": result.config["momentum_periods"],
                "rebalance_frequency": result.config["rebalance_frequency"],
                "selection_mode": result.config["selection_mode"],
                "weighting": result.config["weighting"],
                "max_position_weight": result.config["max_position_weight"],
                "strict_drawdown_kill_switch": (
                    result.config["strict_drawdown_kill_switch"]
                ),
                "use_asset_trend_filter": (
                    result.config["use_asset_trend_filter"]
                ),
                "min_breadth_percent": (
                    result.config["min_breadth_percent"]
                ),
                "target_volatility": result.config["target_volatility"],
                "max_drawdown_guard": result.config["max_drawdown_guard"],
                "return": result.result.total_return,
                "cagr": result.cagr,
                "calmar": result.calmar,
                "benchmark_return": result.benchmark_return,
                "equal_weight_return": result.equal_weight_return,
                "excess_vs_benchmark": result.excess_return,
                "excess_vs_equal_weight": result.excess_vs_equal_weight,
                "sharpe": result.result.sharpe,
                "max_drawdown": result.result.max_drawdown,
                "turnover_percent": result.turnover_percent,
                "annualized_turnover_percent": (
                    result.annualized_turnover_percent
                ),
                "turnover_per_rebalance_percent": (
                    result.turnover_per_rebalance_percent
                ),
                "rebalance_count": result.rebalance_count,
                "estimated_cost": result.estimated_cost,
                "cost_drag_percent": result.cost_drag_percent,
                "closed_trades": result.result.closed_trades,
                "open_trades": result.result.open_trades,
                "quality_score": dual_momentum_quality_score(result),
            })

    return path


def print_dual_momentum_experiments(results, report_path):
    print("\nDUAL MOMENTUM EXPERIMENTS")
    print(
        "Mode | Weight | MaxW | Kill | TopN | Mom | Rebal | Asset SMA | "
        "Breadth | VolTgt | DD Guard | Return | CAGR | Calmar | Ex EqWt | "
        "Sharpe | DD | AnnTurn | Score"
    )
    print("-" * 194)

    for result in results[:10]:
        print(
            f"{result.config['selection_mode']:<12} | "
            f"{result.config['weighting']:<18} | "
            f"{format_percent(result.config['max_position_weight'] or 0):<6} | "
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
    print("Recent selections:")
    for selection in result.selections[-5:]:
        names = ", ".join(selection.symbols) if selection.symbols else "cash"
        regime = "risk-on" if selection.risk_on else "risk-off"
        print(f"  {selection.timestamp.date()} | {regime} | {names}")
    print(f"Saved summary: {report_path}")


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
        avg_excess = sum(
            item["result"].excess_return for item in results
        ) / len(results)
        avg_excess_equal_weight = sum(
            item["result"].excess_vs_equal_weight for item in results
        ) / len(results)
        avg_drawdown = sum(
            item["result"].result.max_drawdown for item in results
        ) / len(results)
        print("-" * 128)
        print(
            "Average | "
            f"excess_spy={format_percent(avg_excess)} | "
            f"excess_eq={format_percent(avg_excess_equal_weight)} | "
            f"drawdown={format_percent(avg_drawdown)}"
        )

    print(f"\nSaved walk-forward: {report_path}")


def format_sleeve_weights(sleeves):
    return ", ".join(
        f"{sleeve.name}:{int(round(sleeve.weight * 100))}%"
        for sleeve in sleeves
    )


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


def run_relative_strength_experiments(config, relative_config, candles_by_symbol):
    grid = relative_config.get("experiment_grid", {})
    top_values = grid.get("top_n", [relative_config.get("top_n", 2)])
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


def apply_runtime_overrides(config, args):
    config = deepcopy(config)

    if args.fast:
        apply_fast_mode(config)

    if args.symbols:
        config["backtest"]["symbols"] = args.symbols
        config["research"].setdefault("relative_strength", {})["symbols"] = (
            args.symbols
        )
        config["research"].setdefault("dual_momentum", {})["symbols"] = (
            args.symbols
        )
        config["research"].setdefault("multi_strategy", {})["symbols"] = (
            args.symbols
        )

    if args.universe == "etf":
        etf_symbols = (
            config["research"]
            .get("dual_momentum", {})
            .get("etf_symbols", [])
        )
        if etf_symbols:
            config["backtest"]["symbols"] = etf_symbols
            config["research"].setdefault("dual_momentum", {})["symbols"] = (
                etf_symbols
            )
            config["research"].setdefault("multi_strategy", {})["symbols"] = (
                etf_symbols
            )

    if args.years is not None:
        config["backtest"]["years"] = args.years

    if args.strategies:
        config["research"]["strategy_comparison"] = [
            strategy_config
            for strategy_config in config["research"]["strategy_comparison"]
            if strategy_config["name"] in set(args.strategies)
        ]

    if args.grid_values is not None:
        limit_strategy_grids(
            config["research"]["strategy_comparison"],
            args.grid_values,
        )
        config["research"]["parameter_grid"] = limited_grid(
            config["research"]["parameter_grid"],
            args.grid_values,
        )

    return config


def apply_fast_mode(config):
    research_config = config["research"]
    fast_config = research_config.get("fast_mode", {})

    config["backtest"]["symbols"] = fast_config.get(
        "symbols",
        config["backtest"]["symbols"][:2],
    )
    config["research"].setdefault("relative_strength", {})["symbols"] = (
        fast_config.get(
            "relative_strength_symbols",
            config["backtest"]["symbols"],
        )
    )
    config["research"].setdefault("dual_momentum", {})["symbols"] = (
        fast_config.get(
            "dual_momentum_symbols",
            config["backtest"]["symbols"],
        )
    )
    config["research"].setdefault("multi_strategy", {})["symbols"] = (
        fast_config.get(
            "multi_strategy_symbols",
            config["backtest"]["symbols"],
        )
    )
    config["backtest"]["years"] = fast_config.get(
        "years",
        min(config["backtest"].get("years", 5), 2),
    )
    config["backtest"]["warmup_bars"] = min(
        config["backtest"].get("warmup_bars", 200),
        fast_config.get("warmup_bars", 120),
    )

    for key in (
        "stage_one_max_combinations",
        "two_stage_top_n",
        "min_closed_trades",
        "optimizer_min_closed_trades",
    ):
        if key in fast_config:
            research_config[key] = fast_config[key]

    if "walk_forward_folds" in fast_config:
        research_config["walk_forward_folds"] = (
            fast_config["walk_forward_folds"]
        )

    strategy_names = set(fast_config.get("strategies", []))
    if strategy_names:
        research_config["strategy_comparison"] = [
            strategy_config
            for strategy_config in research_config["strategy_comparison"]
            if strategy_config["name"] in strategy_names
        ]

    max_grid_values = fast_config.get("max_grid_values_per_parameter")
    if max_grid_values:
        limit_strategy_grids(
            research_config["strategy_comparison"],
            max_grid_values,
        )
        research_config["parameter_grid"] = limited_grid(
            research_config["parameter_grid"],
            max_grid_values,
        )


def limit_strategy_grids(strategy_configs, max_values):
    for strategy_config in strategy_configs:
        if "parameter_grid" in strategy_config:
            strategy_config["parameter_grid"] = limited_grid(
                strategy_config["parameter_grid"],
                max_values,
            )


def limited_grid(grid, max_values):
    return {
        key: values[:max_values] if isinstance(values, list) else values
        for key, values in grid.items()
    }


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
        choices=[
            "backtest",
            "optimize",
            "walk-forward",
            "compare-strategies",
            "relative-strength",
            "dual-momentum",
            "dual-momentum-walk-forward",
            "dual-momentum-risk-regimes",
            "multi-strategy",
            "multi-strategy-walk-forward",
        ],
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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the reduced research config for quick iteration.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override backtest symbols for this run.",
    )
    parser.add_argument(
        "--years",
        type=int,
        help="Override backtest history length for this run.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Only compare the named strategies for this run.",
    )
    parser.add_argument(
        "--grid-values",
        type=int,
        help="Limit each grid parameter to the first N configured values.",
    )
    parser.add_argument(
        "--universe",
        choices=["default", "etf"],
        default="default",
        help="Use a configured research universe preset.",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run the selected research mode's experiment grid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = apply_runtime_overrides(load_config(), args)

    from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed

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

    if args.mode == "relative-strength":
        run_relative_strength(
            config,
            feed,
            run_experiments=args.experiments,
        )
        return

    if args.mode == "dual-momentum":
        run_dual_momentum(
            config,
            feed,
            run_experiments=args.experiments,
        )
        return

    if args.mode == "dual-momentum-walk-forward":
        run_dual_momentum_walk_forward(config, feed)
        return

    if args.mode == "dual-momentum-risk-regimes":
        run_dual_momentum_risk_regime_experiments(config, feed)
        return

    if args.mode == "multi-strategy":
        run_multi_strategy(
            config,
            feed,
            run_experiments=args.experiments,
        )
        return

    if args.mode == "multi-strategy-walk-forward":
        run_multi_strategy_walk_forward(config, feed)
        return

    run_base_backtests(config, feed)


if __name__ == "__main__":
    main()
