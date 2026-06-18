import argparse

from application.services.runtime_overrides import apply_runtime_overrides
from application.services.paper_commands import (
    run_paper_trade,
    run_paper_fill,
    run_paper_status,
    run_paper_report,
    run_paper_repair,
    run_paper_reset,
    run_paper_run,
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
    run_base_backtests,
    run_optimization,
    run_walk_forward,
    run_strategy_comparison,
)
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
            "relative-strength",
            "dual-momentum",
            "dual-momentum-walk-forward",
            "dual-momentum-risk-regimes",
            "dual-momentum-diagnosis",
            "paper-trade",
            "paper-fill",
            "paper-status",
            "paper-report",
            "paper-run",
            "paper-repair",
            "paper-reset",
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
    parser.add_argument(
        "--decision-file",
        help="Paper decision JSON to fill. Defaults to the latest decision.",
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Confirm destructive paper state reset.",
    )

    return parser.parse_args()


def build_feed(config):
    from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed

    return AlpacaDataFeed(
        api_key=config["alpaca"]["api_key"],
        secret_key=config["alpaca"]["secret_key"],
    )


def dispatch(args, config, feed):
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

    if args.mode == "dual-momentum-diagnosis":
        run_dual_momentum_diagnosis(config, feed)
        return

    if args.mode == "paper-trade":
        run_paper_trade(config, feed)
        return

    if args.mode == "paper-fill":
        run_paper_fill(config, decision_file=args.decision_file)
        return

    if args.mode == "paper-status":
        run_paper_status(config)
        return

    if args.mode == "paper-report":
        run_paper_report(config, feed)
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

    run_base_backtests(config, feed)


def run_cli():
    args = parse_args()
    config = apply_runtime_overrides(load_config(), args)
    feed = build_feed(config)

    dispatch(args, config, feed)