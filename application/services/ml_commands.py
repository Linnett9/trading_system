from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "MLResearchBatchItem": ("application.services.ml_commands_types", "MLResearchBatchItem"),
    "MLResearchBatchResult": ("application.services.ml_commands_types", "MLResearchBatchResult"),
    "_batch_worker_config": ("application.services.ml_commands_batch", "_batch_worker_config"),
    "_build_research_feed": ("application.services.ml_commands_batch", "_build_research_feed"),
    "_expanded_rebalance_dataset_path": ("application.services.ml_commands_batch", "_expanded_rebalance_dataset_path"),
    "_run_ml_research_batch_worker": ("application.services.ml_commands_batch", "_run_ml_research_batch_worker"),
    "run_ml_research_batch": ("application.services.ml_commands_batch", "run_ml_research_batch"),
    "run_ml_online_intraday_benchmark": ("application.services.ml_commands_online", "run_ml_online_intraday_benchmark"),
    "validate_ml_research_batch_config": ("application.services.ml_commands_batch", "validate_ml_research_batch_config"),
    "run_ml_expanded_rebalance_dataset": ("application.services.ml_commands_research", "run_ml_expanded_rebalance_dataset"),
    "run_ml_research": ("application.services.ml_commands_research", "run_ml_research"),
    "run_ml_build_universes": ("application.services.ml_commands_inventory", "run_ml_build_universes"),
    "run_ml_data_inventory": ("application.services.ml_commands_inventory", "run_ml_data_inventory"),
    "run_ml_meta_ensemble": ("application.services.ml_commands_meta", "run_ml_meta_ensemble"),
    "run_ml_model_contract_audit": ("application.services.ml_commands_audits", "run_ml_model_contract_audit"),
    "run_ml_return_mechanics_audit": ("application.services.ml_commands_audits", "run_ml_return_mechanics_audit"),
    "run_ml_overnight_stock_alpha": ("application.services.ml_commands_stock", "run_ml_overnight_stock_alpha"),
    "run_ml_stock_alpha_candidate_report": ("application.services.ml_commands_stock", "run_ml_stock_alpha_candidate_report"),
    "run_ml_stock_alpha_deep_diagnostics": ("application.services.ml_commands_stock", "run_ml_stock_alpha_deep_diagnostics"),
    "run_ml_stock_alpha_dev_smoke": ("application.services.ml_commands_stock", "run_ml_stock_alpha_dev_smoke"),
    "run_ml_stock_alpha_ensemble": ("application.services.ml_commands_stock", "run_ml_stock_alpha_ensemble"),
    "run_ml_stock_alpha_ensemble_portfolio_sweep": ("application.services.ml_commands_stock", "run_ml_stock_alpha_ensemble_portfolio_sweep"),
    "run_ml_stock_alpha_experiment_preflight": ("application.services.ml_commands_stock", "run_ml_stock_alpha_experiment_preflight"),
    "run_ml_stock_alpha_experiment_report": ("application.services.ml_commands_stock", "run_ml_stock_alpha_experiment_report"),
    "run_ml_stock_alpha_news_collect_free_sources": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_collect_free_sources"),
    "run_ml_stock_alpha_news_collection_plan": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_collection_plan"),
    "run_ml_stock_alpha_news_historical_backfill": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_historical_backfill"),
    "run_ml_stock_alpha_news_daily_confirmation": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_daily_confirmation"),
    "run_ml_stock_alpha_news_contract_ingest": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_contract_ingest"),
    "run_ml_stock_alpha_news_coverage_audit": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_coverage_audit"),
    "run_ml_stock_alpha_news_feature_diagnostics": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_feature_diagnostics"),
    "run_ml_stock_alpha_news_features": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_features"),
    "run_ml_stock_alpha_news_pipeline_inspect": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_pipeline_inspect"),
    "run_ml_stock_alpha_news_pipeline_preflight": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_pipeline_preflight"),
    "run_ml_stock_alpha_news_provider_audit": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_provider_audit"),
    "run_ml_stock_alpha_news_provider_sample_check": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_provider_sample_check"),
    "run_ml_stock_alpha_news_readiness_preflight": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_readiness_preflight"),
    "run_ml_stock_alpha_news_source_diagnostics": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_source_diagnostics"),
    "run_ml_stock_alpha_news_source_setup_check": ("application.services.ml_commands_stock", "run_ml_stock_alpha_news_source_setup_check"),
    "run_ml_stock_alpha_parallelism_audit": ("application.services.ml_commands_stock", "run_ml_stock_alpha_parallelism_audit"),
    "run_ml_stock_alpha_run_status": ("application.services.ml_commands_stock", "run_ml_stock_alpha_run_status"),
    "run_ml_stock_level_alpha_benchmark": ("application.services.ml_commands_stock", "run_ml_stock_level_alpha_benchmark"),
    "run_ml_stock_selector_bounded": ("application.services.ml_commands_stock", "run_ml_stock_selector_bounded"),
    "run_ml_stock_selector_final_fit": ("application.services.ml_commands_stock", "run_ml_stock_selector_final_fit"),
    "run_ml_selector_exposure_comparison": ("application.services.ml_commands_stock", "run_ml_selector_exposure_comparison"),
    "run_ml_selector_portfolio_promotion": ("application.services.ml_commands_stock", "run_ml_selector_portfolio_promotion"),
    "run_ml_selector_target_tournament": ("application.services.ml_commands_stock", "run_ml_selector_target_tournament"),
    "run_ml_selector_cost_aware_policy_evaluation": ("application.services.ml_commands_stock", "run_ml_selector_cost_aware_policy_evaluation"),
    "run_ml_selector_confidence_ensemble": ("application.services.ml_commands_stock", "run_ml_selector_confidence_ensemble"),
    "run_ml_selector_feature_ablation": ("application.services.ml_commands_stock", "run_ml_selector_feature_ablation"),
    "run_ml_selector_universe_integrity_audit": ("application.services.ml_commands_stock", "run_ml_selector_universe_integrity_audit"),
    "run_ml_stock_fundamentals_preflight": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_preflight"),
    "run_ml_stock_fundamentals_collect": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_collect"),
    "run_ml_stock_fundamentals_normalize": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_normalize"),
    "run_ml_stock_fundamentals_audit": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_audit"),
    "run_ml_stock_fundamentals_snapshots": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_snapshots"),
    "run_ml_stock_fundamentals_enrich": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_enrich"),
    "run_ml_stock_fundamentals_pipeline": ("application.services.ml_commands_stock", "run_ml_stock_fundamentals_pipeline"),
    "run_ml_stock_level_alpha_features": ("application.services.ml_commands_stock", "run_ml_stock_level_alpha_features"),
    "run_ml_stock_level_feature_attribution": ("application.services.ml_commands_stock", "run_ml_stock_level_feature_attribution"),
    "run_ml_stock_level_portfolio_policy_sweep": ("application.services.ml_commands_stock", "run_ml_stock_level_portfolio_policy_sweep"),
    "run_ml_stock_level_portfolio_replay": ("application.services.ml_commands_stock", "run_ml_stock_level_portfolio_replay"),
    "run_ml_stock_selector_rebalance_dataset": ("application.services.ml_commands_stock", "run_ml_stock_selector_rebalance_dataset"),
    "run_ml_stock_level_target_comparison": ("application.services.ml_commands_stock", "run_ml_stock_level_target_comparison"),
    "_artifact_child_dirs": ("application.services.ml_commands_artifacts", "_artifact_child_dirs"),
    "_artifact_source_dirs": ("application.services.ml_commands_artifacts", "_artifact_source_dirs"),
    "_is_incomplete_run_dir": ("application.services.ml_commands_artifacts", "_is_incomplete_run_dir"),
    "_is_valid_source_artifact_dir": ("application.services.ml_commands_artifacts", "_is_valid_source_artifact_dir"),
    "_leaderboard_report_dir": ("application.services.ml_commands_artifacts", "_leaderboard_report_dir"),
    "_meta_ensemble_output_dir": ("application.services.ml_commands_artifacts", "_meta_ensemble_output_dir"),
    "_refresh_trading_research_leaderboard": ("application.services.ml_commands_artifacts", "_refresh_trading_research_leaderboard"),
    "_update_source_leaderboard": ("application.services.ml_commands_artifacts", "_update_source_leaderboard"),
    "_valid_source_leaderboard_dirs": ("application.services.ml_commands_artifacts", "_valid_source_leaderboard_dirs"),
    "incomplete_ml_run_dirs": ("application.services.ml_commands_artifacts", "incomplete_ml_run_dirs"),
    "run_ml_clean_incomplete_runs": ("application.services.ml_commands_artifacts", "run_ml_clean_incomplete_runs"),
    "run_ml_run_inventory": ("application.services.ml_commands_artifacts", "run_ml_run_inventory"),
    "run_ml_validate_artifacts": ("application.services.ml_commands_artifacts", "run_ml_validate_artifacts"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
