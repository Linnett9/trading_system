from copy import deepcopy


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

    if args.universe == "stocks":
        stock_symbols = (
            config["research"]
            .get("dual_momentum", {})
            .get("stock_symbols", [])
        )

        if stock_symbols:
            config["backtest"]["symbols"] = stock_symbols
            config["research"].setdefault("relative_strength", {})[
                "symbols"
            ] = stock_symbols
            config["research"].setdefault("dual_momentum", {})["symbols"] = (
                stock_symbols
            )
            config["research"].setdefault("multi_strategy", {})["symbols"] = (
                stock_symbols
            )

    if args.universe == "stooq_test":
        stooq_test_symbols = config.get("research", {}).get(
            "stooq_test_symbols", []
        )
        if stooq_test_symbols:
            config["backtest"]["symbols"] = stooq_test_symbols
            config["research"].setdefault("relative_strength", {})["symbols"] = (
                stooq_test_symbols
            )
            config["research"].setdefault("dual_momentum", {})["symbols"] = (
                stooq_test_symbols
            )
            config["research"].setdefault("multi_strategy", {})["symbols"] = (
                stooq_test_symbols
            )

    if args.universe == "stooq_30":
        research_config = config.setdefault("research", {})
        import_symbols = research_config.get("stooq_30_symbols", [])
        investable_symbols = research_config.get(
            "stooq_30_investable_symbols", import_symbols
        )
        if import_symbols and investable_symbols:
            research_config["stooq_import_symbols"] = import_symbols
            config["backtest"]["symbols"] = investable_symbols
            research_config.setdefault("relative_strength", {})["symbols"] = (
                investable_symbols
            )
            research_config.setdefault("dual_momentum", {})["symbols"] = (
                investable_symbols
            )
            research_config.setdefault("multi_strategy", {})["symbols"] = (
                investable_symbols
            )

    if args.universe == "all":
        dual_config = config["research"].get("dual_momentum", {})
        combined_symbols = unique_symbols(
            dual_config.get("stock_symbols", [])
            + dual_config.get("etf_symbols", [])
        )

        if combined_symbols:
            config["backtest"]["symbols"] = combined_symbols
            config["research"].setdefault("relative_strength", {})[
                "symbols"
            ] = combined_symbols
            config["research"].setdefault("dual_momentum", {})["symbols"] = (
                combined_symbols
            )
            config["research"].setdefault("multi_strategy", {})["symbols"] = (
                combined_symbols
            )

    if args.years is not None:
        config["backtest"]["years"] = args.years
    elif (
        getattr(args, "mode", None) in {
            "ml-research",
            "ml-smoke-test",
            "ml-expanded-rebalance-dataset",
            "champion-robustness",
        }
        and not args.fast
        and config.get("ml", {}).get("research_years")
    ):
        config["backtest"]["years"] = int(config["ml"]["research_years"])

    if getattr(args, "mode", None) in {
        "ml-research",
        "ml-smoke-test",
        "ml-expanded-rebalance-dataset",
        "champion-robustness",
        "paper-dry-run",
    }:
        historical_provider = config.get("ml", {}).get("historical_data_provider")
        if historical_provider:
            config["backtest"]["provider"] = historical_provider
        if historical_provider == "stooq_csv":
            config["backtest"]["data_dir"] = config.get("ml", {}).get(
                "stooq_csv_dir", "data/raw/stooq"
            )
        if historical_provider == "stooq_parquet":
            config["backtest"]["data_dir"] = config.get("ml", {}).get(
                "stooq_parquet_dir", "data/processed/stooq_parquet"
            )

    if getattr(args, "mode", None) == "paper-trial":
        config["backtest"]["provider"] = "alpaca"
        config.setdefault("paper_trading", {})["enabled"] = False
        config["paper_trading"]["submit_orders"] = False
        config["paper_trading"]["inspect_broker_state"] = False

    if getattr(args, "mode", None) == "ml-smoke-test":
        ml_config = config.setdefault("ml", {})
        ml_config["minimum_history_years"] = int(
            ml_config.get("smoke_test_minimum_history_years", 5)
        )
        ml_config["allow_short_history_for_smoke_test"] = True
        ml_config["research_label"] = "SMOKE_TEST_NOT_PRODUCTION_VALIDATED"
        ml_config["output_dir"] = str(
            ml_config.get("smoke_test_output_dir", "reports/ml/smoke_test")
        )
        config.setdefault("cache", {})["ml_dir"] = str(
            ml_config.get("smoke_test_cache_dir", "cache/ml/smoke_test")
        )

    if getattr(args, "selector_mode", None):
        config["research"].setdefault("dual_momentum", {})[
            "walk_forward_selector_mode"
        ] = args.selector_mode

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


def unique_symbols(symbols):
    unique = []
    seen = set()

    for symbol in symbols:
        if symbol in seen:
            continue

        seen.add(symbol)
        unique.append(symbol)

    return unique


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
