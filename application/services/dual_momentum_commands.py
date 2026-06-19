from application.services.market_data_loader import load_candles
from application.services.dual_momentum_config import active_dual_momentum_config
from application.reporting.dual_momentum_reporter import (
    print_dual_momentum_diagnosis,
    print_dual_momentum_risk_regime_experiments,
    print_dual_momentum_walk_forward,
    print_dual_momentum_experiments,
    print_dual_momentum_result,
)
from core.research.dual_momentum_factory import build_dual_momentum_tester
from core.research.dual_momentum_scoring import (
    risk_regime_score,
    paper_safe_dual_momentum_score,
)
from core.research.dual_momentum_diagnostics import (
    dual_momentum_diagnosis,
    save_dual_momentum_diagnosis,
)
from core.research.dual_momentum_experiments import (
    dual_momentum_risk_regime_configs,
    run_dual_momentum_experiments,
    run_dual_momentum_fold_optimization,
    save_dual_momentum_experiments,
    save_dual_momentum_filtered_walk_forward_candidates,
    save_dual_momentum_risk_regime_experiments,
    save_dual_momentum_walk_forward,
    parse_config_date,
)


def run_dual_momentum(config, feed, run_experiments=False):
    dual_config = active_dual_momentum_config(
        config,
        use_frozen_champion=not run_experiments,
    )
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
            paper_safe_dual_momentum_score(item["result"]),
            risk_regime_score(item["result"]),
            item["result"].result.sharpe,
            item["result"].calmar,
            -item["result"].result.max_drawdown,
            -item["result"].annualized_turnover_percent,
        ),
        reverse=True,
    )

    report_path = save_dual_momentum_risk_regime_experiments(
        results,
        report_dir=config["reports"]["summary_dir"],
    )

    print_dual_momentum_risk_regime_experiments(results, report_path)


def run_dual_momentum_diagnosis(config, feed):
    dual_config = active_dual_momentum_config(config)
    symbols = dual_config.get("symbols", config["backtest"]["symbols"])

    candles_by_symbol = {
        symbol: load_candles(symbol, config, feed)
        for symbol in symbols
    }

    tester = build_dual_momentum_tester(config, dual_config)
    result = tester.run(candles_by_symbol)

    diagnosis = dual_momentum_diagnosis(result, candles_by_symbol)
    report_path = save_dual_momentum_diagnosis(
        diagnosis,
        report_dir=config["reports"]["summary_dir"],
    )

    print_dual_momentum_diagnosis(diagnosis, report_path)


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
            "training_results": training_results,
            "result": test_result,
        })

    candidates_report_path = save_dual_momentum_filtered_walk_forward_candidates(
        results,
        report_dir=config["reports"]["summary_dir"],
    )
    report_path = save_dual_momentum_walk_forward(
        results,
        report_dir=config["reports"]["summary_dir"],
    )

    print(f"Saved walk-forward candidates: {candidates_report_path}")
    print_dual_momentum_walk_forward(results, report_path)
