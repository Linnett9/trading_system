from application.services.market_data_loader import load_candles
from application.reporting.relative_strength_reporter import (
    print_relative_strength_experiments,
    print_relative_strength_result,
)
from core.research.relative_strength_experiments import (
    run_relative_strength_experiments,
    save_relative_strength_experiments,
)
from core.research.relative_strength_portfolio import (
    RelativeStrengthPortfolioBacktester,
)


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