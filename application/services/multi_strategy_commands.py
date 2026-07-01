from application.services.market_data_loader import load_candles
from application.reporting.multi_strategy_reporter import (
    print_multi_strategy_result,
    print_multi_strategy_walk_forward,
    print_multi_strategy_experiments,
)
from core.research.dual_momentum.experiments import parse_config_date
from core.research.multi_strategy.experiments import (
    run_multi_strategy_fold_optimization,
    run_multi_strategy_experiments,
    save_multi_strategy_experiments,
    build_multi_strategy_tester,
    save_multi_strategy_walk_forward,
)


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