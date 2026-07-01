from application.services.adjusted_price_commands import run_refresh_adjusted_prices
from application.services.champion_robustness_commands import run_champion_robustness
from application.services.dual_momentum_commands import (
    run_dual_momentum,
    run_dual_momentum_diagnosis,
    run_dual_momentum_risk_regime_experiments,
    run_dual_momentum_walk_forward,
)
from application.services.ml_commands import (
    run_ml_build_universes,
    run_ml_clean_incomplete_runs,
    run_ml_data_inventory,
    run_ml_expanded_rebalance_dataset,
    run_ml_meta_ensemble,
    run_ml_model_contract_audit,
    run_ml_overnight_stock_alpha,
    run_ml_research,
    run_ml_research_batch,
    run_ml_return_mechanics_audit,
    run_ml_run_inventory,
    run_ml_stock_alpha_candidate_report,
    run_ml_stock_alpha_deep_diagnostics,
    run_ml_stock_alpha_dev_smoke,
    run_ml_stock_alpha_ensemble,
    run_ml_stock_alpha_ensemble_portfolio_sweep,
    run_ml_stock_alpha_experiment_preflight,
    run_ml_stock_alpha_experiment_report,
    run_ml_stock_alpha_news_contract_ingest,
    run_ml_stock_alpha_news_coverage_audit,
    run_ml_stock_alpha_news_features,
    run_ml_stock_alpha_news_provider_audit,
    run_ml_stock_alpha_news_readiness_preflight,
    run_ml_stock_alpha_parallelism_audit,
    run_ml_stock_alpha_run_status,
    run_ml_stock_level_alpha_benchmark,
    run_ml_stock_level_alpha_features,
    run_ml_stock_level_feature_attribution,
    run_ml_stock_level_portfolio_policy_sweep,
    run_ml_stock_level_portfolio_replay,
    run_ml_stock_level_target_comparison,
    run_ml_validate_artifacts,
)
from application.services.multi_strategy_commands import (
    run_multi_strategy,
    run_multi_strategy_walk_forward,
)
from application.services.paper_commands import (
    run_paper_dry_run,
    run_paper_fill,
    run_paper_promotion_checklist,
    run_paper_repair,
    run_paper_report,
    run_paper_reset,
    run_paper_run,
    run_paper_status,
    run_paper_trade,
    run_paper_trading,
    run_paper_weekly_summary,
)
from application.services.relative_strength_commands import run_relative_strength
from application.services.research_commands import (
    run_base_backtests,
    run_data_audit,
    run_optimization,
    run_strategy_comparison,
    run_walk_forward,
)
from application.services.stooq_bulk_commands import run_stooq_bulk_import


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
    if args.mode == "ml-refresh-adjusted-prices":
        run_refresh_adjusted_prices(
            config,
            symbols=args.symbols,
            auto_discover_replay_symbols=args.auto_discover_replay_symbols,
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
    if args.mode == "ml-clean-incomplete-runs":
        run_ml_clean_incomplete_runs(config)
        return
    if args.mode == "ml-model-contract-audit":
        run_ml_model_contract_audit(config)
        return
    if args.mode == "ml-validate-artifacts":
        run_ml_validate_artifacts(config)
        return
    if args.mode == "ml-meta-ensemble":
        run_ml_meta_ensemble(config)
        return
    if args.mode in {"ml-return-mechanics-audit", "ml-benchmark-return-audit"}:
        run_ml_return_mechanics_audit(config)
        return
    if args.mode == "ml-stock-level-alpha-benchmark":
        run_ml_stock_level_alpha_benchmark(config)
        return
    if args.mode == "ml-stock-level-target-comparison":
        run_ml_stock_level_target_comparison(config)
        return
    if args.mode == "ml-stock-level-portfolio-replay":
        run_ml_stock_level_portfolio_replay(config)
        return
    if args.mode == "ml-stock-level-portfolio-policy-sweep":
        run_ml_stock_level_portfolio_policy_sweep(config)
        return
    if args.mode == "ml-stock-alpha-experiment-report":
        run_ml_stock_alpha_experiment_report(config)
        return
    if args.mode == "ml-stock-alpha-candidate-report":
        run_ml_stock_alpha_candidate_report(config)
        return
    if args.mode == "ml-stock-alpha-deep-diagnostics":
        run_ml_stock_alpha_deep_diagnostics(config)
        return
    if args.mode == "ml-stock-alpha-ensemble":
        run_ml_stock_alpha_ensemble(config)
        return
    if args.mode == "ml-stock-alpha-ensemble-portfolio-sweep":
        run_ml_stock_alpha_ensemble_portfolio_sweep(config)
        return
    if args.mode == "ml-stock-alpha-experiment-preflight":
        run_ml_stock_alpha_experiment_preflight(config)
        return
    if args.mode == "ml-stock-alpha-news-features":
        run_ml_stock_alpha_news_features(config)
        return
    if args.mode == "ml-stock-alpha-news-contract-ingest":
        run_ml_stock_alpha_news_contract_ingest(config)
        return
    if args.mode == "ml-stock-alpha-news-coverage-audit":
        run_ml_stock_alpha_news_coverage_audit(config)
        return
    if args.mode == "ml-stock-alpha-news-provider-audit":
        run_ml_stock_alpha_news_provider_audit(config)
        return
    if args.mode == "ml-stock-alpha-news-readiness-preflight":
        run_ml_stock_alpha_news_readiness_preflight(config)
        return
    if args.mode == "ml-stock-alpha-dev-smoke":
        run_ml_stock_alpha_dev_smoke(config)
        return
    if args.mode == "ml-stock-alpha-parallelism-audit":
        run_ml_stock_alpha_parallelism_audit(config)
        return
    if args.mode == "ml-stock-alpha-run-status":
        run_ml_stock_alpha_run_status(config)
        return
    if args.mode == "ml-stock-level-feature-attribution":
        run_ml_stock_level_feature_attribution(config)
        return
    if args.mode == "ml-stock-level-alpha-features":
        run_ml_stock_level_alpha_features(config)
        return
    if args.mode == "ml-overnight-stock-alpha":
        run_ml_overnight_stock_alpha(config)
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
