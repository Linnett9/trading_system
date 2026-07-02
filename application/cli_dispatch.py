from importlib import import_module
from types import ModuleType


def _commands(module_name: str) -> ModuleType:
    """Load a command service only after its CLI mode has been selected."""
    return import_module(f"application.services.{module_name}")


def dispatch(args, config, feed):
    if args.mode == "import-stooq-bulk":
        _commands("stooq_bulk_commands").run_stooq_bulk_import(
            config,
            symbols=args.symbols,
            top=args.top,
            all_raw=args.all_raw,
            asset_class=args.asset_class,
            min_rows=args.min_rows,
            exclude_warrants_units_rights=args.exclude_warrants_units_rights,
        )
        return
    if args.mode == "ml-refresh-adjusted-prices":
        _commands("adjusted_price_commands").run_refresh_adjusted_prices(
            config,
            symbols=args.symbols,
            auto_discover_replay_symbols=args.auto_discover_replay_symbols,
        )
        return
    if args.mode == "ml-data-inventory":
        _commands("ml_commands").run_ml_data_inventory(config)
        return
    if args.mode == "ml-build-universes":
        _commands("ml_commands").run_ml_build_universes(config)
        return
    if args.mode == "ml-run-inventory":
        _commands("ml_commands").run_ml_run_inventory(config)
        return
    if args.mode == "ml-clean-incomplete-runs":
        _commands("ml_commands").run_ml_clean_incomplete_runs(config)
        return
    if args.mode == "ml-model-contract-audit":
        _commands("ml_commands").run_ml_model_contract_audit(config)
        return
    if args.mode == "ml-validate-artifacts":
        _commands("ml_commands").run_ml_validate_artifacts(config)
        return
    if args.mode == "ml-meta-ensemble":
        _commands("ml_commands").run_ml_meta_ensemble(config)
        return
    if args.mode in {"ml-return-mechanics-audit", "ml-benchmark-return-audit"}:
        _commands("ml_commands").run_ml_return_mechanics_audit(config)
        return
    if args.mode == "ml-stock-level-alpha-benchmark":
        _commands("ml_commands").run_ml_stock_level_alpha_benchmark(config)
        return
    if args.mode == "ml-stock-level-target-comparison":
        _commands("ml_commands").run_ml_stock_level_target_comparison(config)
        return
    if args.mode == "ml-stock-level-portfolio-replay":
        _commands("ml_commands").run_ml_stock_level_portfolio_replay(config)
        return
    if args.mode == "ml-stock-level-portfolio-policy-sweep":
        _commands("ml_commands").run_ml_stock_level_portfolio_policy_sweep(config)
        return
    if args.mode == "ml-stock-alpha-experiment-report":
        _commands("ml_commands").run_ml_stock_alpha_experiment_report(config)
        return
    if args.mode == "ml-stock-alpha-candidate-report":
        _commands("ml_commands").run_ml_stock_alpha_candidate_report(config)
        return
    if args.mode == "ml-stock-alpha-deep-diagnostics":
        _commands("ml_commands").run_ml_stock_alpha_deep_diagnostics(config)
        return
    if args.mode == "ml-stock-alpha-ensemble":
        _commands("ml_commands").run_ml_stock_alpha_ensemble(config)
        return
    if args.mode == "ml-stock-alpha-ensemble-portfolio-sweep":
        _commands("ml_commands").run_ml_stock_alpha_ensemble_portfolio_sweep(config)
        return
    if args.mode == "ml-stock-alpha-experiment-preflight":
        _commands("ml_commands").run_ml_stock_alpha_experiment_preflight(config)
        return
    if args.mode == "ml-stock-alpha-news-features":
        _commands("ml_commands").run_ml_stock_alpha_news_features(config)
        return
    if args.mode == "ml-stock-alpha-news-contract-ingest":
        _commands("ml_commands").run_ml_stock_alpha_news_contract_ingest(config)
        return
    if args.mode == "ml-stock-alpha-news-coverage-audit":
        _commands("ml_commands").run_ml_stock_alpha_news_coverage_audit(config)
        return
    if args.mode == "ml-stock-alpha-news-provider-audit":
        _commands("ml_commands").run_ml_stock_alpha_news_provider_audit(config)
        return
    if args.mode == "ml-stock-alpha-news-pipeline-preflight":
        _commands("ml_commands").run_ml_stock_alpha_news_pipeline_preflight(config)
        return
    if args.mode == "ml-stock-alpha-news-readiness-preflight":
        _commands("ml_commands").run_ml_stock_alpha_news_readiness_preflight(config)
        return
    if args.mode == "ml-stock-alpha-dev-smoke":
        _commands("ml_commands").run_ml_stock_alpha_dev_smoke(config)
        return
    if args.mode == "ml-stock-alpha-parallelism-audit":
        _commands("ml_commands").run_ml_stock_alpha_parallelism_audit(config)
        return
    if args.mode == "ml-stock-alpha-run-status":
        _commands("ml_commands").run_ml_stock_alpha_run_status(config)
        return
    if args.mode == "ml-stock-level-feature-attribution":
        _commands("ml_commands").run_ml_stock_level_feature_attribution(config)
        return
    if args.mode == "ml-stock-level-alpha-features":
        _commands("ml_commands").run_ml_stock_level_alpha_features(config)
        return
    if args.mode == "ml-overnight-stock-alpha":
        _commands("ml_commands").run_ml_overnight_stock_alpha(config)
        return
    if args.mode == "champion-robustness":
        _commands("champion_robustness_commands").run_champion_robustness(config, feed)
        return

    if args.mode == "optimize":
        _commands("research_commands").run_optimization(config, feed)
        return

    if args.mode == "walk-forward":
        _commands("research_commands").run_walk_forward(config, feed, show_details=args.details)
        return

    if args.mode == "compare-strategies":
        _commands("research_commands").run_strategy_comparison(config, feed, show_all=args.all_results)
        return

    if args.mode == "data-audit":
        _commands("research_commands").run_data_audit(config, feed)
        return

    if args.mode == "relative-strength":
        _commands("relative_strength_commands").run_relative_strength(
            config,
            feed,
            run_experiments=args.experiments,
        )
        return

    if args.mode == "dual-momentum":
        _commands("dual_momentum_commands").run_dual_momentum(
            config,
            feed,
            run_experiments=args.experiments,
        )
        return

    if args.mode == "dual-momentum-walk-forward":
        _commands("dual_momentum_commands").run_dual_momentum_walk_forward(config, feed)
        return

    if args.mode == "dual-momentum-risk-regimes":
        _commands("dual_momentum_commands").run_dual_momentum_risk_regime_experiments(config, feed)
        return

    if args.mode == "dual-momentum-diagnosis":
        _commands("dual_momentum_commands").run_dual_momentum_diagnosis(config, feed)
        return

    if args.mode == "paper-trade":
        _commands("paper_commands").run_paper_trade(config, feed)
        return

    if args.mode == "paper-fill":
        _commands("paper_commands").run_paper_fill(
            config,
            decision_file=args.decision_file,
            confirm_fill=args.confirm_fill,
        )
        return

    if args.mode == "paper-status":
        _commands("paper_commands").run_paper_status(config)
        return

    if args.mode == "paper-report":
        _commands("paper_commands").run_paper_report(config, feed)
        return

    if args.mode == "paper-trading":
        _commands("paper_commands").run_paper_trading(
            config,
            feed,
            dry_run=args.dry_run or not args.submit,
            submit=args.submit,
        )
        return
    if args.mode == "paper-dry-run":
        _commands("paper_commands").run_paper_dry_run(config, feed)
        return
    if args.mode == "paper-trial":
        _commands("paper_commands").run_paper_dry_run(config, feed)
        return

    if args.mode == "paper-weekly-summary":
        _commands("paper_commands").run_paper_weekly_summary(config)
        return

    if args.mode == "paper-promotion-checklist":
        _commands("paper_commands").run_paper_promotion_checklist(config)
        return

    if args.mode == "paper-run":
        _commands("paper_commands").run_paper_run(config, feed)
        return

    if args.mode == "paper-repair":
        _commands("paper_commands").run_paper_repair(config, feed)
        return

    if args.mode == "paper-reset":
        _commands("paper_commands").run_paper_reset(config, confirm_reset=args.confirm_reset)
        return

    if args.mode == "multi-strategy":
        _commands("multi_strategy_commands").run_multi_strategy(
            config,
            feed,
            run_experiments=args.experiments,
        )
        return

    if args.mode == "multi-strategy-walk-forward":
        _commands("multi_strategy_commands").run_multi_strategy_walk_forward(config, feed)
        return

    if args.mode in {"ml-research", "ml-smoke-test"}:
        _commands("ml_commands").run_ml_research(config, feed)
        return

    if args.mode == "ml-research-batch":
        _commands("ml_commands").run_ml_research_batch(config)
        return

    if args.mode == "ml-expanded-rebalance-dataset":
        _commands("ml_commands").run_ml_expanded_rebalance_dataset(config, feed)
        return

    _commands("research_commands").run_base_backtests(config, feed)
