import argparse

from application.services.runtime_overrides import apply_runtime_overrides
from application.services.research_profiles import apply_research_profile
from application.services.paper_commands import (
    run_paper_trade,
    run_paper_fill,
    run_paper_status,
    run_paper_report,
    run_paper_repair,
    run_paper_reset,
    run_paper_run,
    run_paper_trading,
    run_paper_dry_run,
    run_paper_weekly_summary,
    run_paper_promotion_checklist,
)
from application.services.dual_momentum_commands import (
    run_dual_momentum,
    run_dual_momentum_risk_regime_experiments,
    run_dual_momentum_diagnosis,
    run_dual_momentum_walk_forward,
)
from application.services.multi_strategy_commands import (
    run_multi_strategy,
    run_multi_strategy_walk_forward,
)
from application.services.relative_strength_commands import (
    run_relative_strength,
)
from application.services.research_commands import (
    run_data_audit,
    run_base_backtests,
    run_optimization,
    run_walk_forward,
    run_strategy_comparison,
)
from application.services.ml_commands import (
    run_ml_build_universes,
    run_ml_data_inventory,
    run_ml_expanded_rebalance_dataset,
    run_ml_research_batch,
    run_ml_run_inventory,
    run_ml_meta_ensemble,
    run_ml_research,
    run_ml_validate_artifacts,
)
from application.services.stooq_bulk_commands import run_stooq_bulk_import
from application.services.champion_robustness_commands import run_champion_robustness
from config.config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trading system command line interface"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "backtest",
            "optimize",
            "walk-forward",
            "compare-strategies",
            "data-audit",
            "relative-strength",
            "dual-momentum",
            "dual-momentum-walk-forward",
            "dual-momentum-risk-regimes",
            "dual-momentum-diagnosis",
            "paper-trade",
            "paper-fill",
            "paper-status",
            "paper-report",
            "paper-trading",
            "paper-dry-run",
            "paper-trial",
            "paper-weekly-summary",
            "paper-promotion-checklist",
            "paper-run",
            "paper-repair",
            "paper-reset",
            "multi-strategy",
            "multi-strategy-walk-forward",
            "ml-research",
            "ml-research-batch",
            "ml-run-inventory",
            "ml-validate-artifacts",
            "ml-smoke-test",
            "ml-data-inventory",
            "ml-build-universes",
            "ml-expanded-rebalance-dataset",
            "ml-meta-ensemble",
            "import-stooq-bulk",
            "champion-robustness",
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
        "--top",
        type=int,
        choices=[500, 1000],
        help="Import the top N raw Stooq symbols by available row count.",
    )
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="Import every raw Stooq symbol after asset/min-row filters.",
    )
    parser.add_argument(
        "--asset-class",
        choices=["stocks", "etfs", "all"],
        default="all",
        help="Filter raw Stooq imports by asset class.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        help="Filter raw Stooq imports to symbols with at least this many rows.",
    )
    parser.add_argument(
        "--exclude-warrants-units-rights",
        action="store_true",
        help="Exclude likely warrants, units, and rights from raw Stooq imports.",
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
        choices=["default", "etf", "stocks", "all", "stooq_test", "stooq_30"],
        default="default",
        help="Use a configured research universe preset.",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run the selected research mode's experiment grid.",
    )
    parser.add_argument(
        "--selector-mode",
        choices=["research", "paper", "production"],
        help=(
            "Override the dual-momentum walk-forward selector mode. "
            "Defaults to the configured mode."
        ),
    )
    parser.add_argument(
        "--decision-file",
        help="Paper decision JSON to fill. Defaults to the latest decision.",
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Confirm destructive paper state reset.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file to load. Can point at a frozen paper config.",
    )
    parser.add_argument(
        "--profile",
        choices=["development", "benchmark"],
        help="Apply a research profile for isolated ML cache/report outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview paper-trading orders without filling/submitting.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit/fill the approved paper-trading decision.",
    )
    parser.add_argument(
        "--confirm-fill",
        action="store_true",
        help="Confirm direct paper-fill command. Prefer paper-trading --submit.",
    )

    return parser.parse_args()


def build_feed(config):
    provider = config["backtest"].get("provider", "alpaca").lower()
    if provider == "stooq":
        from infrastructure.data.stooq_data_feed import StooqDataFeed

        return StooqDataFeed()
    if provider == "stooq_csv":
        from infrastructure.data.stooq_csv_data_feed import StooqCsvDataFeed

        return StooqCsvDataFeed(
            data_dir=config["backtest"].get("data_dir", "data/raw/stooq")
        )
    if provider == "stooq_parquet":
        from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed

        return StooqParquetDataFeed(
            data_dir=config["backtest"].get(
                "data_dir", "data/processed/stooq_parquet"
            )
        )
    if provider != "alpaca":
        raise ValueError(f"Unsupported historical data provider: {provider}")

    alpaca_config = config.get("alpaca", {})
    if not alpaca_config.get("api_key") or not alpaca_config.get("secret_key"):
        raise RuntimeError(
            "Alpaca data provider requires ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "environment variables."
        )

    from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed

    return AlpacaDataFeed(
        api_key=alpaca_config["api_key"],
        secret_key=alpaca_config["secret_key"],
        data_feed=config["backtest"].get("data_feed", "iex"),
        adjustment=config["backtest"].get("data_adjustment", "all"),
        historical_bar_limit=int(
            config["backtest"].get("historical_bar_limit", 10_000)
        ),
    )


def dispatch(args, config, feed):
    if args.mode == "import-stooq-bulk":
        run_stooq_bulk_import(
            config,
            symbols=args.symbols,
            top=args.top,
            all_raw=args.all_raw,
            asset_class=args.asset_class,
            min_rows=args.min_rows,
            exclude_warrants_units_rights=args.exclude_warrants_units_rights,
        )
        return
    if args.mode == "ml-data-inventory":
        run_ml_data_inventory(config)
        return
    if args.mode == "ml-build-universes":
        run_ml_build_universes(config)
        return
    if args.mode == "ml-run-inventory":
        run_ml_run_inventory(config)
        return
    if args.mode == "ml-validate-artifacts":
        run_ml_validate_artifacts(config)
        return
    if args.mode == "ml-meta-ensemble":
        run_ml_meta_ensemble(config)
        return
    if args.mode == "champion-robustness":
        run_champion_robustness(config, feed)
        return

    if args.mode == "optimize":
        run_optimization(config, feed)
        return

    if args.mode == "walk-forward":
        run_walk_forward(config, feed, show_details=args.details)
        return

    if args.mode == "compare-strategies":
        run_strategy_comparison(config, feed, show_all=args.all_results)
        return

    if args.mode == "data-audit":
        run_data_audit(config, feed)
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

    if args.mode == "dual-momentum-diagnosis":
        run_dual_momentum_diagnosis(config, feed)
        return

    if args.mode == "paper-trade":
        run_paper_trade(config, feed)
        return

    if args.mode == "paper-fill":
        run_paper_fill(
            config,
            decision_file=args.decision_file,
            confirm_fill=args.confirm_fill,
        )
        return

    if args.mode == "paper-status":
        run_paper_status(config)
        return

    if args.mode == "paper-report":
        run_paper_report(config, feed)
        return

    if args.mode == "paper-trading":
        run_paper_trading(
            config,
            feed,
            dry_run=args.dry_run or not args.submit,
            submit=args.submit,
        )
        return
    if args.mode == "paper-dry-run":
        run_paper_dry_run(config, feed)
        return
    if args.mode == "paper-trial":
        run_paper_dry_run(config, feed)
        return

    if args.mode == "paper-weekly-summary":
        run_paper_weekly_summary(config)
        return

    if args.mode == "paper-promotion-checklist":
        run_paper_promotion_checklist(config)
        return

    if args.mode == "paper-run":
        run_paper_run(config, feed)
        return

    if args.mode == "paper-repair":
        run_paper_repair(config, feed)
        return

    if args.mode == "paper-reset":
        run_paper_reset(config, confirm_reset=args.confirm_reset)
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

    if args.mode in {"ml-research", "ml-smoke-test"}:
        run_ml_research(config, feed)
        return

    if args.mode == "ml-research-batch":
        run_ml_research_batch(config)
        return

    if args.mode == "ml-expanded-rebalance-dataset":
        run_ml_expanded_rebalance_dataset(config, feed)
        return

    run_base_backtests(config, feed)


def run_cli():
    args = parse_args()
    loaded_config = apply_research_profile(
        load_config(args.config, overlay_project_config=True),
        getattr(args, "profile", None),
    )
    config = apply_runtime_overrides(
        loaded_config,
        args,
    )
    feedless_modes = {
        "import-stooq-bulk",
        "ml-data-inventory",
        "ml-build-universes",
        "ml-run-inventory",
        "ml-validate-artifacts",
        "ml-meta-ensemble",
        "ml-research-batch",
    }
    feed = None if args.mode in feedless_modes else build_feed(config)

    dispatch(args, config, feed)
