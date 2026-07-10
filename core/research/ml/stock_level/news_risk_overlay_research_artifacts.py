from __future__ import annotations

from typing import Any, Callable, Mapping

from core.research.ml.stock_level.news_risk_overlay_research_paths import NewsRiskResearchPaths


def write_news_risk_research_artifacts(
    *,
    paths: NewsRiskResearchPaths,
    labeled: list[Mapping[str, Any]],
    dataset_max_rows: int,
    decision_rows: list[dict[str, Any]],
    shadow_max_rows: int,
    coverage: Mapping[str, Any],
    leakage: Mapping[str, Any],
    audit_detail_max_rows: int,
    metrics: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    replay: Mapping[str, Any],
    score_direction_audit: Mapping[str, Any],
    score_decile_rows: list[dict[str, Any]],
    decile_join_audit: Mapping[str, Any],
    decile_reconciliation: Mapping[str, Any],
    score_direction_report: Mapping[str, Any],
    replay_action_attribution: Mapping[str, Any],
    event_category_analysis: Mapping[str, Any],
    contrarian_report: Mapping[str, Any],
    price_stabilisation: Mapping[str, Any],
    resilience_analysis: Mapping[str, Any],
    extreme_archive_rows: list[dict[str, Any]],
    extreme_memory_report: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
    validation: Mapping[str, Any],
    oos_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    write_csv: Callable[..., None],
    write_json: Callable[..., None],
    limited_rows: Callable[[list[Mapping[str, Any]], int], list[Mapping[str, Any]]],
    limited_audit_details: Callable[[Mapping[str, Any], int], dict[str, Any]],
    accounting_definitions: Callable[[], dict[str, Any]],
    accounting_audit: Callable[[Mapping[str, Any]], dict[str, Any]],
    score_direction_markdown: Callable[[Mapping[str, Any]], str],
    append_experiment_registry_entry: Callable[..., None],
    research_artifact_manifest: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    artifact_validation_report: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    news_validation_workflow_map: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    validation_dependency_graph: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    validation_readiness_dashboard: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    artifact_lineage_report: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    news_validation_gap_analysis: Callable[[NewsRiskResearchPaths], dict[str, Any]],
    markdown: Callable[..., str],
) -> None:
    write_csv(paths.dataset_csv_path, limited_rows(labeled, dataset_max_rows))
    write_csv(paths.shadow_csv_path, limited_rows(decision_rows, shadow_max_rows))
    write_json(paths.coverage_json_path, coverage)
    write_json(paths.leakage_json_path, limited_audit_details(leakage, audit_detail_max_rows))
    write_json(paths.metrics_json_path, metrics)
    write_json(paths.portfolio_json_path, portfolio)
    write_json(paths.accounting_json_path, accounting_definitions())
    write_json(paths.accounting_audit_json_path, accounting_audit(portfolio))
    write_csv(paths.equity_curve_csv_path, portfolio["equity_curve"])
    write_csv(paths.drawdown_curve_csv_path, portfolio["drawdown_curve"])
    write_csv(paths.trade_ledger_csv_path, replay["trade_ledger"])
    write_csv(paths.daily_equity_price_only_csv_path, replay["daily_equity"]["price_only"])
    write_csv(paths.daily_equity_news_cash_csv_path, replay["daily_equity"]["news_cash"])
    write_csv(paths.daily_equity_news_replacement_csv_path, replay["daily_equity"]["news_replacement"])
    write_csv(paths.daily_equity_news_reduced_size_csv_path, replay["daily_equity"]["news_reduced_size"])
    write_json(paths.open_trade_portfolio_json_path, replay["portfolio_comparison"])
    write_json(paths.replay_risk_metrics_json_path, replay["risk_metrics"])
    write_json(paths.action_attribution_json_path, replay["action_attribution"])
    write_json(paths.score_direction_audit_json_path, score_direction_audit)
    write_csv(paths.news_score_deciles_csv_path, score_decile_rows)
    write_csv(paths.corrected_news_score_deciles_csv_path, score_decile_rows)
    write_json(paths.decile_join_audit_json_path, decile_join_audit)
    write_json(paths.decile_trade_reconciliation_json_path, decile_reconciliation)
    write_json(paths.news_score_direction_report_json_path, score_direction_report)
    paths.news_score_direction_summary_md_path.write_text(
        score_direction_markdown(score_direction_report),
        encoding="utf-8",
    )
    write_json(paths.replay_action_attribution_json_path, replay_action_attribution)
    write_json(paths.event_category_analysis_json_path, event_category_analysis)
    write_json(paths.contrarian_strategy_comparison_json_path, contrarian_report)
    write_csv(paths.contrarian_trade_ledger_csv_path, replay.get("contrarian_trade_ledger", []))
    write_json(paths.price_stabilisation_comparison_json_path, price_stabilisation)
    write_json(paths.resilience_filter_analysis_json_path, resilience_analysis)
    write_csv(paths.extreme_event_archive_csv_path, extreme_archive_rows)
    write_json(paths.extreme_event_memory_report_json_path, extreme_memory_report)
    write_json(paths.cost_scenario_comparison_json_path, cost_scenarios)
    write_json(paths.chronological_split_manifest_json_path, validation["chronological_split_manifest"])
    append_experiment_registry_entry(
        paths.experiment_registry_jsonl_path,
        rows=oos_rows,
        replay=replay,
        validation=validation,
        coverage=coverage,
        event_category_analysis=event_category_analysis,
        decile_reconciliation=decile_reconciliation,
        config=config,
    )
    write_csv(paths.contrarian_grid_results_csv_path, validation["contrarian_grid_results"])
    write_json(paths.contrarian_grid_selection_json_path, validation["contrarian_grid_selection"])
    write_csv(paths.contrarian_fold_results_csv_path, validation["contrarian_fold_results"])
    write_json(paths.contrarian_parameter_stability_json_path, validation["contrarian_parameter_stability"])
    write_json(paths.contrarian_frozen_config_json_path, validation["contrarian_frozen_config"])
    write_json(paths.contrarian_holdout_report_json_path, validation["contrarian_holdout_report"])
    write_csv(paths.contrarian_holdout_trade_ledger_csv_path, validation["contrarian_holdout_trade_ledger"])
    write_csv(paths.contrarian_holdout_equity_csv_path, validation["contrarian_holdout_equity"])
    paths.contrarian_holdout_comparison_md_path.write_text(
        validation["contrarian_holdout_comparison_md"],
        encoding="utf-8",
    )
    write_csv(paths.contrarian_walk_forward_folds_csv_path, validation["contrarian_walk_forward_folds"])
    write_json(paths.contrarian_walk_forward_summary_json_path, validation["contrarian_walk_forward_summary"])
    write_json(paths.contrarian_chronological_validation_plan_json_path, validation["contrarian_chronological_validation_plan"])
    write_csv(paths.contrarian_chronological_periods_csv_path, validation["contrarian_chronological_periods"])
    write_json(paths.contrarian_walk_forward_validation_report_json_path, validation["contrarian_walk_forward_validation_report"])
    write_json(paths.contrarian_placebo_permutation_report_json_path, validation["contrarian_placebo_permutation_report"])
    write_csv(paths.contrarian_placebo_permutation_results_csv_path, validation["contrarian_placebo_permutation_results"])
    write_json(paths.contrarian_matched_control_report_json_path, validation["contrarian_matched_control_report"])
    write_csv(paths.contrarian_matched_control_results_csv_path, validation["contrarian_matched_control_results"])
    write_json(paths.contrarian_profit_concentration_report_json_path, validation["contrarian_profit_concentration_report"])
    write_csv(paths.contrarian_trade_fragility_by_symbol_csv_path, validation["contrarian_trade_fragility_by_symbol"])
    write_csv(paths.contrarian_trade_fragility_by_year_csv_path, validation["contrarian_trade_fragility_by_year"])
    write_csv(paths.contrarian_top_trade_removal_csv_path, validation["contrarian_top_trade_removal"])
    write_json(paths.contrarian_year_regime_report_json_path, validation["contrarian_year_regime_report"])
    write_csv(paths.contrarian_year_regime_results_csv_path, validation["contrarian_year_regime_results"])
    write_csv(paths.contrarian_year_regime_examples_csv_path, validation["contrarian_year_regime_examples"])
    write_json(paths.contrarian_symbol_year_ablation_report_json_path, validation["contrarian_symbol_year_ablation_report"])
    write_csv(paths.contrarian_without_top_symbols_csv_path, validation["contrarian_without_top_symbols"])
    write_csv(paths.contrarian_without_top_years_csv_path, validation["contrarian_without_top_years"])
    write_json(paths.contrarian_cost_slippage_robustness_report_json_path, validation["contrarian_cost_slippage_robustness_report"])
    write_csv(paths.contrarian_cost_slippage_robustness_csv_path, validation["contrarian_cost_slippage_robustness"])
    write_json(paths.contrarian_data_validity_audit_json_path, validation["contrarian_data_validity_audit"])
    write_json(paths.intraday_5min_expansion_plan_json_path, validation["intraday_5min_expansion_plan"])
    write_csv(paths.contrarian_placebo_results_csv_path, validation["contrarian_placebo_results"])
    write_json(paths.contrarian_placebo_summary_json_path, validation["contrarian_placebo_summary"])
    write_json(paths.contrarian_matched_controls_json_path, validation["contrarian_matched_controls"])
    write_csv(paths.contrarian_contribution_by_year_csv_path, validation["contrarian_contribution_by_year"])
    write_csv(paths.contrarian_contribution_by_symbol_csv_path, validation["contrarian_contribution_by_symbol"])
    write_json(paths.contrarian_concentration_report_json_path, validation["contrarian_concentration_report"])
    write_json(paths.universe_survivorship_audit_json_path, validation["universe_survivorship_audit"])
    write_csv(paths.universe_membership_by_date_csv_path, validation["universe_membership_by_date"])
    write_json(paths.corporate_action_audit_json_path, validation["corporate_action_audit"])
    write_json(paths.missing_news_bias_report_json_path, validation["missing_news_bias_report"])
    write_csv(paths.covered_vs_uncovered_candidates_csv_path, validation["covered_vs_uncovered_candidates"])
    write_json(paths.text_model_readiness_json_path, validation["text_model_readiness"])
    write_json(paths.news_transformer_readiness_json_path, validation["news_transformer_readiness"])
    write_json(paths.news_transformer_training_plan_json_path, validation["news_transformer_training_plan"])
    write_json(paths.catastrophic_news_audit_json_path, validation["catastrophic_news_audit"])
    write_csv(paths.catastrophic_news_candidates_csv_path, validation["catastrophic_news_candidates"])
    write_json(paths.catastrophic_news_veto_report_json_path, validation["catastrophic_news_veto_report"])
    write_json(paths.catastrophic_veto_candidate_attribution_json_path, validation["catastrophic_veto_candidate_attribution"])
    write_csv(paths.catastrophic_veto_trade_attribution_csv_path, validation["catastrophic_veto_trade_attribution"])
    write_json(paths.catastrophic_veto_strategy_comparison_json_path, validation["catastrophic_veto_strategy_comparison"])
    write_json(paths.catastrophic_veto_policy_json_path, validation["catastrophic_veto_policy"])
    write_json(paths.catastrophic_veto_filtered_strategy_report_json_path, validation["catastrophic_veto_filtered_strategy_report"])
    write_csv(paths.catastrophic_veto_removed_trades_csv_path, validation["catastrophic_veto_removed_trades"])
    write_csv(paths.catastrophic_veto_removed_symbols_csv_path, validation["catastrophic_veto_removed_symbols"])
    write_json(paths.catastrophic_veto_full_replay_report_json_path, validation["catastrophic_veto_full_replay_report"])
    write_csv(
        paths.catastrophic_veto_full_replay_trade_ledger_csv_path,
        validation["catastrophic_veto_full_replay_trade_ledger"],
        empty_fields=("trade_id", "candidate_id", "symbol", "strategy_variant", "entry_date", "exit_date"),
    )
    write_csv(
        paths.catastrophic_veto_full_replay_equity_csv_path,
        validation["catastrophic_veto_full_replay_equity"],
        empty_fields=("date", "strategy_variant", "total_equity", "daily_return"),
    )
    write_csv(paths.catastrophic_veto_filtered_candidates_csv_path, validation["catastrophic_veto_filtered_candidates"])
    write_csv(paths.catastrophic_veto_blocked_candidates_csv_path, validation["catastrophic_veto_blocked_candidates"])
    write_json(paths.catastrophic_veto_replay_seam_report_json_path, validation["catastrophic_veto_replay_seam_report"])
    write_json(paths.catastrophic_veto_bounceback_report_json_path, validation["catastrophic_veto_bounceback_report"])
    write_csv(paths.catastrophic_veto_bounceback_by_category_csv_path, validation["catastrophic_veto_bounceback_by_category"])
    write_csv(paths.catastrophic_veto_bounceback_examples_csv_path, validation["catastrophic_veto_bounceback_examples"])
    write_json(paths.catastrophic_veto_extreme_only_policy_proposal_json_path, validation["catastrophic_veto_extreme_only_policy_proposal"])
    write_json(paths.catastrophic_veto_policy_variant_comparison_json_path, validation["catastrophic_veto_policy_variant_comparison"])
    write_csv(paths.catastrophic_veto_policy_variant_counts_csv_path, validation["catastrophic_veto_policy_variant_counts"])
    write_csv(paths.catastrophic_veto_policy_variant_metrics_csv_path, validation["catastrophic_veto_policy_variant_metrics"])
    write_csv(paths.catastrophic_veto_policy_variant_removed_trades_csv_path, validation["catastrophic_veto_policy_variant_removed_trades"])
    write_csv(paths.catastrophic_veto_policy_variant_bounceback_csv_path, validation["catastrophic_veto_policy_variant_bounceback"])
    write_json(paths.catastrophic_veto_policy_frontier_report_json_path, validation["catastrophic_veto_policy_frontier_report"])
    write_csv(paths.catastrophic_veto_policy_frontier_csv_path, validation["catastrophic_veto_policy_frontier"])
    write_csv(paths.catastrophic_veto_policy_variant_examples_csv_path, validation["catastrophic_veto_policy_variant_examples"])
    write_json(paths.catastrophic_veto_loser_bounceback_casebook_json_path, validation["catastrophic_veto_loser_bounceback_casebook"])
    write_csv(paths.catastrophic_veto_loser_bounceback_cases_csv_path, validation["catastrophic_veto_loser_bounceback_cases"])
    write_csv(paths.catastrophic_veto_loser_bounceback_feature_diff_csv_path, validation["catastrophic_veto_loser_bounceback_feature_diff"])
    write_csv(paths.catastrophic_veto_loser_bounceback_keyword_diff_csv_path, validation["catastrophic_veto_loser_bounceback_keyword_diff"])
    write_json(paths.catastrophic_veto_taxonomy_improvement_plan_json_path, validation["catastrophic_veto_taxonomy_improvement_plan"])
    write_json(paths.catastrophic_veto_parked_status_json_path, validation["catastrophic_veto_parked_status"])
    write_json(paths.catastrophic_news_evidence_quality_report_json_path, validation["catastrophic_news_evidence_quality_report"])
    write_csv(paths.catastrophic_news_evidence_quality_by_field_csv_path, validation["catastrophic_news_evidence_quality_by_field"])
    write_csv(paths.catastrophic_news_evidence_quality_by_symbol_csv_path, validation["catastrophic_news_evidence_quality_by_symbol"])
    write_json(paths.catastrophic_veto_policy_mode_comparison_json_path, validation["catastrophic_veto_policy_mode_comparison"])
    write_csv(paths.catastrophic_veto_policy_mode_counts_csv_path, validation["catastrophic_veto_policy_mode_counts"])
    write_json(paths.news_evidence_lineage_report_json_path, validation["news_evidence_lineage_report"])
    write_csv(paths.news_evidence_lineage_by_stage_csv_path, validation["news_evidence_lineage_by_stage"])
    write_csv(paths.news_evidence_missing_field_examples_csv_path, validation["news_evidence_missing_field_examples"])
    write_json(paths.news_evidence_readiness_report_json_path, validation["news_evidence_readiness_report"])
    write_json(paths.news_event_taxonomy_report_json_path, validation["news_event_taxonomy_report"])
    write_csv(paths.news_event_taxonomy_counts_csv_path, validation["news_event_taxonomy_counts"])
    write_csv(paths.news_event_taxonomy_examples_csv_path, validation["news_event_taxonomy_examples"])
    write_json(paths.news_duplicate_grouping_report_json_path, validation["news_duplicate_grouping_report"])
    write_csv(paths.news_duplicate_grouping_examples_csv_path, validation["news_duplicate_grouping_examples"])
    write_json(paths.news_point_in_time_text_safety_report_json_path, validation["news_point_in_time_text_safety_report"])
    write_csv(paths.news_point_in_time_text_safety_examples_csv_path, validation["news_point_in_time_text_safety_examples"])
    write_json(paths.news_text_keyword_baseline_report_json_path, validation["news_text_keyword_baseline_report"])
    write_csv(paths.news_text_keyword_baseline_scores_csv_path, validation["news_text_keyword_baseline_scores"])
    write_json(paths.validation_stage_placeholders_json_path, validation["validation_stage_placeholders"])
    write_json(paths.walk_forward_validation_report_json_path, validation["walk_forward_validation_report"])
    write_csv(paths.walk_forward_fold_results_csv_path, validation["walk_forward_fold_results"])
    write_json(paths.placebo_permutation_report_json_path, validation["placebo_permutation_report"])
    write_csv(paths.placebo_permutation_results_csv_path, validation["placebo_permutation_results"])
    write_json(paths.exposure_matched_controls_json_path, validation["exposure_matched_controls"])
    write_json(paths.trade_count_matched_controls_json_path, validation["trade_count_matched_controls"])
    write_json(paths.concentration_fragility_report_json_path, validation["concentration_fragility_report"])
    write_json(paths.replay_assumptions_json_path, replay["replay_assumptions"])
    write_json(paths.replay_data_audit_json_path, replay["replay_data_audit"])
    write_json(paths.manifest_json_path, manifest)
    write_json(paths.artifact_manifest_json_path, research_artifact_manifest(paths))
    write_json(paths.artifact_validation_report_json_path, artifact_validation_report(paths))
    write_json(paths.news_validation_workflow_map_json_path, news_validation_workflow_map(paths))
    write_json(paths.validation_dependency_graph_json_path, validation_dependency_graph(paths))
    write_json(paths.validation_readiness_dashboard_json_path, validation_readiness_dashboard(paths))
    write_json(paths.artifact_lineage_report_json_path, artifact_lineage_report(paths))
    write_json(paths.news_validation_gap_analysis_json_path, news_validation_gap_analysis(paths))
    paths.markdown_path.write_text(
        markdown(
            manifest,
            coverage,
            metrics,
            portfolio,
            replay,
            score_direction_report,
            contrarian_report,
            cost_scenarios,
        ),
        encoding="utf-8",
    )
