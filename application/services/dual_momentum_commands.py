from datetime import datetime, time, timedelta, timezone

from application.services.market_data_loader import load_candles
from application.services.dual_momentum_config import active_dual_momentum_config
from application.reporting.dual_momentum_reporter import (
    print_dual_momentum_diagnosis,
    print_dual_momentum_risk_regime_experiments,
    print_dual_momentum_walk_forward,
    print_dual_momentum_experiments,
    print_dual_momentum_result,
)
from core.research.dual_momentum.factory import build_dual_momentum_tester
from core.research.dual_momentum.stock_ml_comparison import (
    stock_ml_comparison_artifact_symbols,
    stock_ml_comparison_artifact_dates,
    stock_ml_comparison_config,
    write_stock_ml_dual_momentum_comparison,
)
from core.research.dual_momentum.scoring import (
    risk_regime_score,
    paper_safe_dual_momentum_score,
)
from core.research.dual_momentum.diagnostics import (
    dual_momentum_diagnosis,
    save_dual_momentum_diagnosis,
)
from core.research.dual_momentum.experiments import (
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


def run_stock_ml_dual_momentum_comparison(
    config,
    feed,
    strategy_names=None,
):
    dual_config = active_dual_momentum_config(
        config,
        use_frozen_champion=False,
    )
    comparison_config = stock_ml_comparison_config(config)
    symbols = stock_ml_comparison_requested_symbols(
        config,
        dual_config,
        comparison_config,
    )
    if feed is None:
        candles_by_symbol = load_stock_ml_comparison_local_candles(
            config,
            dual_config,
            symbols,
            allow_missing=(
                comparison_config["mode"] == "broad_research_universe"
            ),
        )
    else:
        candles_by_symbol = {
            symbol: load_candles(symbol, config, feed)
            for symbol in symbols
        }
    result = write_stock_ml_dual_momentum_comparison(
        config=config,
        dual_config=dual_config,
        candles_by_symbol=candles_by_symbol,
        strategy_names=strategy_names,
    )
    print("\nSTOCK ML DUAL-MOMENTUM SCORE COMPARISON")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"CSV: {result.csv_path}")
    print(f"Markdown: {result.markdown_path}")


def load_stock_ml_comparison_local_candles(
    config,
    dual_config,
    symbols=None,
    allow_missing=False,
):
    """Load local daily candles for the feedless stock-score comparison."""
    comparison_config = stock_ml_comparison_config(config)
    oos_dates = stock_ml_comparison_artifact_dates(config)
    if not oos_dates:
        raise ValueError(
            "Stock ML dual-momentum comparison artifact has no OOS dates: "
            f"{comparison_config['artifact_path']}"
        )
    start_at, end_at = stock_ml_comparison_local_date_window(
        config,
        dual_config,
        oos_dates,
    )
    local_feed = build_stock_ml_comparison_local_feed(config)
    timeframe = comparison_config.get(
        "timeframe",
        config.get("backtest", {}).get("timeframe", "1Day"),
    )
    requested_symbols = symbols or dual_config.get(
        "symbols",
        config.get("backtest", {}).get("symbols", []),
    )
    candles_by_symbol = {}

    for symbol in requested_symbols:
        try:
            candles = local_feed.get_historical_bars(
                symbol,
                timeframe,
                start_at,
                end_at,
            )
        except FileNotFoundError as exc:
            if allow_missing:
                continue
            raise FileNotFoundError(
                "Missing local historical data for stock ML "
                f"dual-momentum comparison: symbol={symbol}, "
                f"timeframe={timeframe}, provider="
                f"{_local_history_provider(config)}. {exc}"
            ) from exc
        if not candles:
            if allow_missing:
                continue
            raise FileNotFoundError(
                "No local candles loaded for stock ML dual-momentum "
                f"comparison: symbol={symbol}, timeframe={timeframe}, "
                f"start={start_at.isoformat()}, end={end_at.isoformat()}"
            )
        candles_by_symbol[symbol] = candles

    return candles_by_symbol


def stock_ml_comparison_requested_symbols(
    config,
    dual_config,
    comparison_config=None,
):
    resolved = comparison_config or stock_ml_comparison_config(config)
    if resolved["mode"] == "broad_research_universe":
        return sorted(stock_ml_comparison_artifact_symbols(config))
    return dual_config.get(
        "symbols",
        config.get("backtest", {}).get("symbols", []),
    )


def build_stock_ml_comparison_local_feed(config):
    provider = _local_history_provider(config)
    ml_config = config.get("ml", {})
    comparison_config = stock_ml_comparison_config(config)

    if provider == "stooq_parquet":
        from infrastructure.data.stooq_parquet_data_feed import (
            StooqParquetDataFeed,
        )

        return StooqParquetDataFeed(
            data_dir=ml_config.get(
                "stooq_parquet_dir",
                config.get("backtest", {}).get(
                    "data_dir",
                    "data/processed/stooq_parquet",
                ),
            )
        )

    if provider == "market_parquet":
        from infrastructure.data.market_parquet import MarketParquetDataFeed

        return MarketParquetDataFeed(
            data_root=comparison_config.get(
                "market_parquet_dir",
                ml_config.get(
                    "market_parquet_dir",
                    config.get("backtest", {}).get(
                        "data_dir",
                        "data/processed",
                    ),
                ),
            )
        )

    raise RuntimeError(
        "ml-dual-momentum-stock-score-comparison supports local research "
        "providers only: stooq_parquet, market_parquet. "
        f"Configured provider: {provider}"
    )


def stock_ml_comparison_local_date_window(config, dual_config, oos_dates):
    lookback_bars = _stock_ml_comparison_required_lookback_bars(dual_config)
    warmup_days = int(
        config.get("ml", {})
        .get("stock_ml_dual_momentum_comparison", {})
        .get("warmup_calendar_days", lookback_bars * 3 + 30)
    )
    first_oos = min(oos_dates)
    last_oos = max(oos_dates)
    start = datetime.combine(first_oos - timedelta(days=warmup_days), time.min)
    end = datetime.combine(last_oos, time.max)

    if _local_history_provider(config) == "market_parquet":
        start = start.replace(tzinfo=timezone.utc)
        end = end.replace(tzinfo=timezone.utc)

    return start, end


def _stock_ml_comparison_required_lookback_bars(dual_config):
    leadership_periods = dual_config.get("leadership_momentum_periods") or [
        21,
        63,
    ]
    values = [
        dual_config.get("regime_sma_period", 200),
        *(dual_config.get("momentum_periods") or [126, 252]),
        *(dual_config.get("risk_off_momentum_periods") or []),
        *(dual_config.get("fallback_momentum_periods") or []),
        *(dual_config.get("enhanced_momentum_periods") or []),
        *(dual_config.get("relative_strength_periods") or []),
        dual_config.get("ranking_volatility_lookback", 63),
        dual_config.get("volatility_lookback", 63),
        *leadership_periods,
        dual_config.get("relative_strength_filter_period", 63),
    ]
    if dual_config.get("use_asset_trend_filter", True):
        values.append(dual_config.get("asset_sma_period", 200))
    if dual_config.get("quality_filter_enabled", False):
        values.extend([
            dual_config.get("quality_momentum_period", 21),
            dual_config.get("quality_sma_period", 50),
        ])
    return max(int(value or 0) for value in values)


def _local_history_provider(config):
    comparison_config = (
        config.get("ml", {}).get("stock_ml_dual_momentum_comparison", {})
    )
    return (
        comparison_config.get("local_data_provider")
        or config.get("ml", {}).get("historical_data_provider")
        or config.get("backtest", {}).get("provider")
        or "market_parquet"
    ).lower()


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
