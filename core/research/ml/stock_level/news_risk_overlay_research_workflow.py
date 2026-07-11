from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.stock_level.news_risk_overlay_research_inspection import (
    _read_json_if_available,
)
from core.research.ml.stock_level.news_risk_overlay_research_paths import NewsRiskResearchPaths
from core.research.ml.stock_level.news_risk_overlay_research_workflow_gaps import (
    _news_validation_gap_analysis,
)
from core.research.ml.stock_level.news_risk_overlay_research_workflow_placeholders import (
    _validation_stage_placeholders,
)


def _research_artifact_manifest(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    artifacts = {
        name: str(value)
        for name, value in paths.__dict__.items()
        if name.endswith("_path") and isinstance(value, Path)
    }
    return {
        "schema_name": "stock_alpha_news_risk_overlay_artifact_manifest",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _artifact_validation_report(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    required_names = (
        "chronological_split_manifest_json_path",
        "experiment_registry_jsonl_path",
        "contrarian_grid_selection_json_path",
        "contrarian_frozen_config_json_path",
        "contrarian_holdout_report_json_path",
        "contrarian_chronological_validation_plan_json_path",
        "contrarian_chronological_periods_csv_path",
        "contrarian_walk_forward_validation_report_json_path",
        "contrarian_placebo_permutation_report_json_path",
        "contrarian_placebo_permutation_results_csv_path",
        "contrarian_matched_control_report_json_path",
        "contrarian_matched_control_results_csv_path",
        "contrarian_profit_concentration_report_json_path",
        "contrarian_trade_fragility_by_symbol_csv_path",
        "contrarian_trade_fragility_by_year_csv_path",
        "contrarian_top_trade_removal_csv_path",
        "contrarian_year_regime_report_json_path",
        "contrarian_year_regime_results_csv_path",
        "contrarian_year_regime_examples_csv_path",
        "contrarian_symbol_year_ablation_report_json_path",
        "contrarian_without_top_symbols_csv_path",
        "contrarian_without_top_years_csv_path",
        "contrarian_cost_slippage_robustness_report_json_path",
        "contrarian_cost_slippage_robustness_csv_path",
        "contrarian_data_validity_audit_json_path",
        "intraday_5min_expansion_plan_json_path",
        "text_model_readiness_json_path",
        "validation_stage_placeholders_json_path",
        "decile_join_audit_json_path",
        "decile_trade_reconciliation_json_path",
        "corrected_news_score_deciles_csv_path",
        "news_validation_workflow_map_json_path",
        "validation_dependency_graph_json_path",
        "validation_readiness_dashboard_json_path",
        "artifact_lineage_report_json_path",
        "news_validation_gap_analysis_json_path",
        "news_transformer_readiness_json_path",
        "news_transformer_training_plan_json_path",
        "catastrophic_news_audit_json_path",
        "catastrophic_news_candidates_csv_path",
        "catastrophic_news_veto_report_json_path",
        "catastrophic_veto_candidate_attribution_json_path",
        "catastrophic_veto_trade_attribution_csv_path",
        "catastrophic_veto_strategy_comparison_json_path",
        "catastrophic_veto_policy_json_path",
        "catastrophic_veto_filtered_strategy_report_json_path",
        "catastrophic_veto_removed_trades_csv_path",
        "catastrophic_veto_removed_symbols_csv_path",
        "catastrophic_veto_full_replay_report_json_path",
        "catastrophic_veto_full_replay_trade_ledger_csv_path",
        "catastrophic_veto_full_replay_equity_csv_path",
        "catastrophic_veto_filtered_candidates_csv_path",
        "catastrophic_veto_blocked_candidates_csv_path",
        "catastrophic_veto_replay_seam_report_json_path",
        "catastrophic_veto_bounceback_report_json_path",
        "catastrophic_veto_bounceback_by_category_csv_path",
        "catastrophic_veto_bounceback_examples_csv_path",
        "catastrophic_veto_extreme_only_policy_proposal_json_path",
        "catastrophic_veto_policy_variant_comparison_json_path",
        "catastrophic_veto_policy_variant_counts_csv_path",
        "catastrophic_veto_policy_variant_metrics_csv_path",
        "catastrophic_veto_policy_variant_removed_trades_csv_path",
        "catastrophic_veto_policy_variant_bounceback_csv_path",
        "catastrophic_veto_policy_frontier_report_json_path",
        "catastrophic_veto_policy_frontier_csv_path",
        "catastrophic_veto_policy_variant_examples_csv_path",
        "catastrophic_veto_loser_bounceback_casebook_json_path",
        "catastrophic_veto_loser_bounceback_cases_csv_path",
        "catastrophic_veto_loser_bounceback_feature_diff_csv_path",
        "catastrophic_veto_loser_bounceback_keyword_diff_csv_path",
        "catastrophic_veto_taxonomy_improvement_plan_json_path",
        "catastrophic_veto_parked_status_json_path",
        "catastrophic_news_evidence_quality_report_json_path",
        "catastrophic_news_evidence_quality_by_field_csv_path",
        "catastrophic_news_evidence_quality_by_symbol_csv_path",
        "catastrophic_veto_policy_mode_comparison_json_path",
        "catastrophic_veto_policy_mode_counts_csv_path",
        "news_evidence_lineage_report_json_path",
        "news_evidence_lineage_by_stage_csv_path",
        "news_evidence_missing_field_examples_csv_path",
        "news_evidence_readiness_report_json_path",
        "news_event_taxonomy_report_json_path",
        "news_event_taxonomy_counts_csv_path",
        "news_event_taxonomy_examples_csv_path",
        "news_duplicate_grouping_report_json_path",
        "news_duplicate_grouping_examples_csv_path",
        "news_point_in_time_text_safety_report_json_path",
        "news_point_in_time_text_safety_examples_csv_path",
        "news_text_keyword_baseline_report_json_path",
        "news_text_keyword_baseline_scores_csv_path",
        "walk_forward_validation_report_json_path",
        "walk_forward_fold_results_csv_path",
        "placebo_permutation_report_json_path",
        "placebo_permutation_results_csv_path",
        "exposure_matched_controls_json_path",
        "trade_count_matched_controls_json_path",
        "concentration_fragility_report_json_path",
    )
    artifact_status = []
    missing = []
    for name in required_names:
        path = getattr(paths, name)
        exists = path.exists()
        if not exists:
            missing.append(name)
        artifact_status.append(
            {
                "artifact_key": name,
                "path": str(path),
                "exists": exists,
                "status": "PRESENT" if exists else "MISSING",
            }
        )
    return {
        "schema_name": "stock_alpha_news_risk_overlay_artifact_validation_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "PASSED" if not missing else "FAILED",
        "status_scope": "ARTIFACT_PRESENCE_ONLY",
        "artifact_presence_status": "PASSED" if not missing else "FAILED",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "is_final_validation": False,
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "required_artifact_count": len(required_names),
        "missing_artifact_count": len(missing),
        "missing_artifacts": missing,
        "artifacts": artifact_status,
    }


def _news_validation_workflow_map(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    evidence_status = _read_json_if_available(
        paths.catastrophic_news_evidence_quality_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    news_evidence_status = _read_json_if_available(
        paths.news_evidence_readiness_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    event_taxonomy_status = _read_json_if_available(
        paths.news_event_taxonomy_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    duplicate_grouping_status = _read_json_if_available(
        paths.news_duplicate_grouping_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    text_safety_status = _read_json_if_available(
        paths.news_point_in_time_text_safety_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    keyword_baseline_status = _read_json_if_available(
        paths.news_text_keyword_baseline_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    bounceback_status = _read_json_if_available(
        paths.catastrophic_veto_bounceback_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    extreme_policy_status = _read_json_if_available(
        paths.catastrophic_veto_extreme_only_policy_proposal_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_variant_status = _read_json_if_available(
        paths.catastrophic_veto_policy_variant_comparison_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_frontier_status = _read_json_if_available(
        paths.catastrophic_veto_policy_frontier_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    casebook_status = _read_json_if_available(
        paths.catastrophic_veto_loser_bounceback_casebook_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    taxonomy_plan_status = _read_json_if_available(
        paths.catastrophic_veto_taxonomy_improvement_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    parked_veto_status = _read_json_if_available(
        paths.catastrophic_veto_parked_status_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    contrarian_profit_status = _read_json_if_available(
        paths.contrarian_profit_concentration_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    year_regime_status = _read_json_if_available(
        paths.contrarian_year_regime_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    symbol_year_ablation_status = _read_json_if_available(
        paths.contrarian_symbol_year_ablation_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    contrarian_data_validity_status = _read_json_if_available(
        paths.contrarian_data_validity_audit_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    intraday_5min_status = _read_json_if_available(
        paths.intraday_5min_expansion_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    casebook_status = _read_json_if_available(
        paths.catastrophic_veto_loser_bounceback_casebook_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    taxonomy_plan_status = _read_json_if_available(
        paths.catastrophic_veto_taxonomy_improvement_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    bounceback_status = _read_json_if_available(
        paths.catastrophic_veto_bounceback_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    extreme_policy_status = _read_json_if_available(
        paths.catastrophic_veto_extreme_only_policy_proposal_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    event_taxonomy_status = _read_json_if_available(
        paths.news_event_taxonomy_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    duplicate_grouping_status = _read_json_if_available(
        paths.news_duplicate_grouping_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    text_safety_status = _read_json_if_available(
        paths.news_point_in_time_text_safety_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    keyword_baseline_status = _read_json_if_available(
        paths.news_text_keyword_baseline_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    node_specs = [
        ("price_model_candidates", "Price model candidates", "IMPLEMENTED", True, [paths.dataset_csv_path], True, True, []),
        ("news_feature_join", "Point-in-time news feature join", "IMPLEMENTED", True, [paths.leakage_json_path], True, True, ["price_model_candidates"]),
        ("coverage_report", "News coverage report", "PRESENT", True, [paths.coverage_json_path], True, True, ["news_feature_join"]),
        ("score_direction_audit", "Score direction audit", "PRESENT", True, [paths.score_direction_audit_json_path, paths.news_score_direction_report_json_path], True, True, ["news_feature_join"]),
        ("strategy_variants", "Research strategy variants", "IMPLEMENTED", True, [paths.contrarian_strategy_comparison_json_path], True, True, ["score_direction_audit"]),
        ("portfolio_replay", "Open-position research replay", "IMPLEMENTED", True, [paths.open_trade_portfolio_json_path, paths.replay_risk_metrics_json_path], True, True, ["strategy_variants"]),
        ("cost_sensitivity", "Cost sensitivity scenarios", "PRESENT", True, [paths.cost_scenario_comparison_json_path], True, True, ["portfolio_replay"]),
        ("decile_reconciliation", "Decile candidate-to-trade reconciliation", "PASSED", True, [paths.decile_join_audit_json_path, paths.decile_trade_reconciliation_json_path], True, True, ["portfolio_replay"]),
        ("chronological_split", "Chronological split manifest", "PSEUDO_HOLDOUT", True, [paths.chronological_split_manifest_json_path], True, True, ["decile_reconciliation"]),
        ("experiment_registry", "Append-only experiment registry", "PRESENT", True, [paths.experiment_registry_jsonl_path], True, True, ["chronological_split"]),
        ("grid_selection", "Contrarian grid selection scaffold", "SCAFFOLD", True, [paths.contrarian_grid_selection_json_path], True, True, ["experiment_registry"]),
        ("frozen_config", "Frozen contrarian configuration", "PRESENT", True, [paths.contrarian_frozen_config_json_path], True, True, ["grid_selection"]),
        ("pseudo_holdout_report", "Pseudo-holdout report", "PSEUDO_HOLDOUT", True, [paths.contrarian_holdout_report_json_path], True, True, ["frozen_config"]),
        ("contrarian_chronological_validation", "Main contrarian chronological validation plan", "IN_PROGRESS", True, [paths.contrarian_chronological_validation_plan_json_path, paths.contrarian_chronological_periods_csv_path], True, True, ["pseudo_holdout_report"]),
        ("contrarian_profit_concentration", "Main contrarian profit concentration", contrarian_profit_status, True, [paths.contrarian_profit_concentration_report_json_path, paths.contrarian_trade_fragility_by_symbol_csv_path, paths.contrarian_trade_fragility_by_year_csv_path, paths.contrarian_top_trade_removal_csv_path], True, True, ["portfolio_replay"]),
        ("contrarian_year_regime", "Main contrarian year/regime robustness", year_regime_status, True, [paths.contrarian_year_regime_report_json_path, paths.contrarian_year_regime_results_csv_path, paths.contrarian_year_regime_examples_csv_path], True, True, ["portfolio_replay"]),
        ("contrarian_symbol_year_ablation", "Main contrarian symbol/year ablations", symbol_year_ablation_status, True, [paths.contrarian_symbol_year_ablation_report_json_path, paths.contrarian_without_top_symbols_csv_path, paths.contrarian_without_top_years_csv_path], True, True, ["contrarian_profit_concentration"]),
        ("contrarian_data_validity", "Main contrarian data-validity audit", contrarian_data_validity_status, True, [paths.contrarian_data_validity_audit_json_path], True, True, ["portfolio_replay"]),
        ("intraday_5min_expansion_plan", "Future Dell PC intraday expansion plan", intraday_5min_status, True, [paths.intraday_5min_expansion_plan_json_path], True, False, ["contrarian_data_validity"]),
        ("validation_stage_placeholders", "Validation stage placeholders", "PRESENT", True, [paths.validation_stage_placeholders_json_path], True, True, ["pseudo_holdout_report"]),
        ("artifact_manifest", "Artifact manifest", "PRESENT", True, [paths.artifact_manifest_json_path], True, False, ["validation_stage_placeholders"]),
        ("artifact_validation_report", "Artifact validation report", "PASSED", True, [paths.artifact_validation_report_json_path], True, False, ["artifact_manifest"]),
        ("text_model_readiness", "Text model readiness report", "NOT_READY", True, [paths.text_model_readiness_json_path], True, True, ["validation_stage_placeholders"]),
        ("future_walk_forward", "Future walk-forward robustness", "NOT_IMPLEMENTED", False, [], False, True, ["frozen_config"]),
        ("future_placebo", "Future placebo and permutation testing", "NOT_IMPLEMENTED", False, [], False, True, ["future_walk_forward"]),
        ("future_matched_controls", "Future matched controls", "NOT_IMPLEMENTED", False, [], False, True, ["future_placebo"]),
        ("walk_forward_validation", "Walk-forward validation report", "NOT_IMPLEMENTED", True, [paths.walk_forward_validation_report_json_path, paths.walk_forward_fold_results_csv_path], True, True, ["frozen_config"]),
        ("placebo_permutation_validation", "Placebo/permutation validation report", "UNAVAILABLE_INPUT", True, [paths.placebo_permutation_report_json_path, paths.placebo_permutation_results_csv_path], True, True, ["frozen_config"]),
        ("matched_control_reports", "Matched-control reports", "NOT_IMPLEMENTED", True, [paths.exposure_matched_controls_json_path, paths.trade_count_matched_controls_json_path], True, True, ["frozen_config"]),
        ("concentration_fragility", "Concentration and fragility report", "SCAFFOLD", True, [paths.concentration_fragility_report_json_path], True, True, ["frozen_config"]),
        ("future_concentration", "Future concentration analysis", "NOT_IMPLEMENTED", False, [], False, True, ["future_matched_controls"]),
        ("future_survivorship_audit", "Future survivorship audit", "NOT_IMPLEMENTED", False, [], False, True, ["future_concentration"]),
        ("future_corporate_action_audit", "Future corporate-action audit", "NOT_IMPLEMENTED", False, [], False, True, ["future_survivorship_audit"]),
        ("future_missing_news_bias", "Future missing-news bias analysis", "NOT_IMPLEMENTED", False, [], False, True, ["future_corporate_action_audit"]),
        ("future_text_models", "Future text models", "NOT_READY", False, [], False, True, ["text_model_readiness", "future_missing_news_bias"]),
        ("news_transformer_scaffold", "Disabled news transformer scaffold", "NOT_READY", True, [paths.news_transformer_readiness_json_path, paths.news_transformer_training_plan_json_path], True, True, ["text_model_readiness"]),
        ("catastrophic_news_veto", "Research-only catastrophic-news veto audit", "PASSED_WITH_WARNINGS", True, [paths.catastrophic_news_audit_json_path, paths.catastrophic_news_candidates_csv_path, paths.catastrophic_news_veto_report_json_path], True, True, ["text_model_readiness"]),
        ("catastrophic_veto_strategy_comparison", "Research-only catastrophic-veto strategy comparison", "ATTRIBUTION_ONLY", True, [paths.catastrophic_veto_candidate_attribution_json_path, paths.catastrophic_veto_trade_attribution_csv_path, paths.catastrophic_veto_strategy_comparison_json_path, paths.catastrophic_veto_policy_json_path], True, True, ["catastrophic_news_veto"]),
        ("catastrophic_veto_filtered_scenario", "Research-only catastrophic-veto replay-impact simulation", "APPROXIMATE_LEDGER_SIMULATION", True, [paths.catastrophic_veto_filtered_strategy_report_json_path, paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_removed_symbols_csv_path], True, True, ["catastrophic_veto_strategy_comparison"]),
        ("catastrophic_veto_full_replay", "Research-only catastrophic-veto full replay variant", "FULL_REPLAY_COMPUTED" if full_replay_computed else "FULL_REPLAY_NOT_AVAILABLE", True, [paths.catastrophic_veto_full_replay_report_json_path, paths.catastrophic_veto_full_replay_trade_ledger_csv_path, paths.catastrophic_veto_full_replay_equity_csv_path, paths.catastrophic_veto_filtered_candidates_csv_path, paths.catastrophic_veto_blocked_candidates_csv_path], True, True, ["catastrophic_veto_filtered_scenario"]),
        ("catastrophic_veto_replay_seam", "Research-only filtered replay input seam", "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "ADAPTER_ADDED_NOT_EXECUTED", True, [paths.catastrophic_veto_replay_seam_report_json_path], True, True, ["catastrophic_veto_full_replay"]),
        ("catastrophic_news_evidence_quality", "Catastrophic-news evidence quality", evidence_status, True, [paths.catastrophic_news_evidence_quality_report_json_path, paths.catastrophic_news_evidence_quality_by_field_csv_path, paths.catastrophic_news_evidence_quality_by_symbol_csv_path], True, True, ["catastrophic_news_veto"]),
        ("catastrophic_veto_policy_modes", "Research-only catastrophic-veto policy modes", "COUNT_ONLY_RESEARCH", True, [paths.catastrophic_veto_policy_mode_comparison_json_path, paths.catastrophic_veto_policy_mode_counts_csv_path], True, True, ["catastrophic_news_evidence_quality"]),
        ("news_evidence_lineage", "News evidence contract and lineage audit", news_evidence_status, True, [paths.news_evidence_lineage_report_json_path, paths.news_evidence_lineage_by_stage_csv_path, paths.news_evidence_missing_field_examples_csv_path, paths.news_evidence_readiness_report_json_path], True, True, ["news_feature_join"]),
        ("event_taxonomy_research", "Deterministic headline event taxonomy", event_taxonomy_status, True, [paths.news_event_taxonomy_report_json_path, paths.news_event_taxonomy_counts_csv_path, paths.news_event_taxonomy_examples_csv_path], True, False, ["news_evidence_lineage"]),
        ("duplicate_grouping_heuristic", "Heuristic duplicate grouping", duplicate_grouping_status, True, [paths.news_duplicate_grouping_report_json_path, paths.news_duplicate_grouping_examples_csv_path], True, False, ["news_evidence_lineage"]),
        ("point_in_time_text_safety", "Point-in-time text safety audit", text_safety_status, True, [paths.news_point_in_time_text_safety_report_json_path, paths.news_point_in_time_text_safety_examples_csv_path], True, False, ["news_evidence_lineage"]),
        ("keyword_text_baseline", "Deterministic keyword text baseline", keyword_baseline_status, True, [paths.news_text_keyword_baseline_report_json_path, paths.news_text_keyword_baseline_scores_csv_path], True, False, ["point_in_time_text_safety"]),
        ("catastrophic_veto_bounceback", "Research-only blocked-trade bounce-back attribution", bounceback_status, True, [paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_bounceback_by_category_csv_path, paths.catastrophic_veto_bounceback_examples_csv_path], True, False, ["catastrophic_veto_full_replay", "event_taxonomy_research"]),
        ("extreme_only_policy_proposal", "Extreme-distress-only policy proposal", extreme_policy_status, True, [paths.catastrophic_veto_extreme_only_policy_proposal_json_path], True, False, ["catastrophic_veto_bounceback"]),
        ("catastrophic_policy_variants", "Research-only catastrophic-veto policy variants", policy_variant_status, True, [paths.catastrophic_veto_policy_variant_comparison_json_path, paths.catastrophic_veto_policy_variant_counts_csv_path, paths.catastrophic_veto_policy_variant_metrics_csv_path, paths.catastrophic_veto_policy_variant_removed_trades_csv_path, paths.catastrophic_veto_policy_variant_bounceback_csv_path, paths.catastrophic_veto_policy_variant_examples_csv_path], True, False, ["catastrophic_veto_bounceback", "event_taxonomy_research"]),
        ("catastrophic_policy_frontier", "Research-only catastrophic-veto policy frontier", policy_frontier_status, True, [paths.catastrophic_veto_policy_frontier_report_json_path, paths.catastrophic_veto_policy_frontier_csv_path], True, False, ["catastrophic_policy_variants"]),
        ("loser_bounceback_casebook", "Research-only loser-vs-bounceback casebook", casebook_status, True, [paths.catastrophic_veto_loser_bounceback_casebook_json_path, paths.catastrophic_veto_loser_bounceback_cases_csv_path, paths.catastrophic_veto_loser_bounceback_feature_diff_csv_path, paths.catastrophic_veto_loser_bounceback_keyword_diff_csv_path], True, False, ["catastrophic_veto_bounceback"]),
        ("taxonomy_improvement_plan", "Research-only catastrophic taxonomy improvement plan", taxonomy_plan_status, True, [paths.catastrophic_veto_taxonomy_improvement_plan_json_path], True, False, ["loser_bounceback_casebook"]),
        ("catastrophic_veto_parked", "Catastrophic veto parked diagnostic status", parked_veto_status, True, [paths.catastrophic_veto_parked_status_json_path], True, False, ["catastrophic_policy_frontier", "loser_bounceback_casebook"]),
    ]
    downstream: dict[str, list[str]] = {node_id: [] for node_id, *_ in node_specs}
    for node_id, _name, _status, _implemented, _paths, _current, _final, upstream in node_specs:
        for dependency in upstream:
            downstream.setdefault(dependency, []).append(node_id)
    nodes = []
    for node_id, name, status, implemented, artifact_paths, required_current, required_final, upstream in node_specs:
        blocks_final = required_final and status in {"PSEUDO_HOLDOUT", "SCAFFOLD", "NOT_IMPLEMENTED", "NOT_READY", "BLOCKED", "FAILED", "UNAVAILABLE_INPUT", "INSUFFICIENT", "INSUFFICIENT_FOR_STRICT_VETO", "PARTIAL_EVIDENCE"}
        nodes.append(
            {
                "node_id": node_id,
                "name": name,
                "status": status,
                "implemented": implemented,
                "artifact_paths": [str(path) for path in artifact_paths],
                "required_for_current_stage": required_current,
                "required_for_final_validation": required_final,
                "blocks_final_validation": blocks_final,
                "contains_blocking_stages": node_id == "validation_stage_placeholders",
                "blocking_stage_count": (
                    sum(
                        1
                        for stage in _validation_stage_placeholders().values()
                        if isinstance(stage, Mapping) and stage.get("blocks_final_validation")
                    )
                    if node_id == "validation_stage_placeholders"
                    else 0
                ),
                "upstream_dependencies": upstream,
                "downstream_dependencies": downstream.get(node_id, []),
                "warnings": _workflow_node_warnings(status, node_id=node_id),
            }
        )
    edges = [
        {"from": dependency, "to": node["node_id"]}
        for node in nodes
        for dependency in node["upstream_dependencies"]
    ]
    return {
        "schema_name": "stock_alpha_news_validation_workflow_map",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "PSEUDO_HOLDOUT_WORKFLOW_INCOMPLETE",
        "workflow_name": "stock_alpha_news_risk_overlay_validation_spine",
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "nodes": nodes,
        "edges": edges,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "warnings": [
            "Workflow map is descriptive and does not validate the strategy.",
            "Pseudo-holdout and unimplemented validation stages block final validation.",
        ],
    }


def _workflow_node_warnings(status: str, *, node_id: str = "") -> list[str]:
    if node_id == "validation_stage_placeholders":
        return ["Validation stage placeholder container includes incomplete gates that block final validation."]
    if status == "PSEUDO_HOLDOUT":
        return ["Previously inspected period; not an untouched holdout."]
    if status == "SCAFFOLD":
        return ["Scaffold artifact only; full validation logic is not complete."]
    if status == "NOT_IMPLEMENTED":
        return ["Stage is not implemented and blocks final validation."]
    if status == "NOT_READY":
        return ["Stage is not ready and blocks final validation."]
    if status == "UNAVAILABLE_INPUT":
        return ["Stage cannot be computed from available inputs and blocks final validation."]
    return []


def _validation_dependency_graph(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    evidence_status = _read_json_if_available(
        paths.catastrophic_news_evidence_quality_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    news_evidence_status = _read_json_if_available(
        paths.news_evidence_readiness_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    event_taxonomy_status = _read_json_if_available(
        paths.news_event_taxonomy_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    duplicate_grouping_status = _read_json_if_available(
        paths.news_duplicate_grouping_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    text_safety_status = _read_json_if_available(
        paths.news_point_in_time_text_safety_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    keyword_baseline_status = _read_json_if_available(
        paths.news_text_keyword_baseline_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    bounceback_status = _read_json_if_available(
        paths.catastrophic_veto_bounceback_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    extreme_policy_status = _read_json_if_available(
        paths.catastrophic_veto_extreme_only_policy_proposal_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_variant_status = _read_json_if_available(
        paths.catastrophic_veto_policy_variant_comparison_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    policy_frontier_status = _read_json_if_available(
        paths.catastrophic_veto_policy_frontier_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    casebook_status = _read_json_if_available(
        paths.catastrophic_veto_loser_bounceback_casebook_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    taxonomy_plan_status = _read_json_if_available(
        paths.catastrophic_veto_taxonomy_improvement_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    parked_veto_status = _read_json_if_available(
        paths.catastrophic_veto_parked_status_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    profit_concentration_status = _read_json_if_available(
        paths.contrarian_profit_concentration_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    year_regime_status = _read_json_if_available(
        paths.contrarian_year_regime_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    symbol_year_ablation_status = _read_json_if_available(
        paths.contrarian_symbol_year_ablation_report_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    data_validity_status = _read_json_if_available(
        paths.contrarian_data_validity_audit_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    intraday_5min_status = _read_json_if_available(
        paths.intraday_5min_expansion_plan_json_path
    ).get("status", "UNAVAILABLE_INPUT")
    gate_specs = [
        ("decile_reconciliation", "PASSED", True, True, paths.decile_trade_reconciliation_json_path, "", "continue to chronological validation"),
        ("chronological_split", "PSEUDO_HOLDOUT", True, False, paths.chronological_split_manifest_json_path, "final period is pseudo-holdout", "obtain untouched prospective data"),
        ("experiment_registry", "PRESENT", True, True, paths.experiment_registry_jsonl_path, "", "preserve append-only registry entries"),
        ("frozen_config", "PRESENT", True, True, paths.contrarian_frozen_config_json_path, "", "use frozen config for future validation"),
        ("pseudo_holdout_gate", "BLOCKED", True, False, paths.contrarian_holdout_report_json_path, "pseudo-holdout cannot pass final validation", "evaluate only on genuinely untouched data"),
        ("contrarian_chronological_validation", "IN_PROGRESS", True, False, paths.contrarian_chronological_validation_plan_json_path, "chronological validation is in progress and pseudo-holdout is not final validation", "collect untouched future final holdout"),
        ("contrarian_profit_concentration", profit_concentration_status, True, profit_concentration_status == "IMPLEMENTED", paths.contrarian_profit_concentration_report_json_path, "profit concentration is diagnostic and does not pass final validation", "review top-trade/symbol/year fragility"),
        ("contrarian_year_regime", year_regime_status, True, year_regime_status == "AVAILABLE", paths.contrarian_year_regime_report_json_path, "year/regime robustness is ledger-level and not final validation", "review negative and partial years"),
        ("contrarian_symbol_year_ablation", symbol_year_ablation_status, True, symbol_year_ablation_status == "AVAILABLE", paths.contrarian_symbol_year_ablation_report_json_path, "symbol/year ablations are ledger-level approximations", "review concentration sensitivity before final validation"),
        ("contrarian_data_validity", data_validity_status, True, False, paths.contrarian_data_validity_audit_json_path, "data-validity audits are incomplete", "implement survivorship/corporate-action/missing-data audits"),
        ("intraday_5min_expansion_plan", intraday_5min_status, True, False, paths.intraday_5min_expansion_plan_json_path, "intraday expansion is planning-only", "confirm Dell PC intraday data paths and run small subset later"),
        ("walk_forward", "NOT_IMPLEMENTED", True, False, paths.walk_forward_validation_report_json_path, "walk-forward robustness is scaffolded but fold-level replay metrics are not implemented", "implement expanding/rolling chronological walk-forward"),
        ("placebo_permutation", "UNAVAILABLE_INPUT", True, False, paths.placebo_permutation_report_json_path, "placebo/permutation report is scaffolded without computable placebo metrics", "implement deterministic placebo tests"),
        ("exposure_matched_controls", "NOT_IMPLEMENTED", True, False, paths.exposure_matched_controls_json_path, "exposure-matched controls are not implemented", "implement exposure-matched controls"),
        ("trade_count_matched_controls", "NOT_IMPLEMENTED", True, False, paths.trade_count_matched_controls_json_path, "trade-count-matched controls are not implemented", "implement trade-count-matched controls"),
        ("concentration_analysis", "SCAFFOLD", True, False, paths.concentration_fragility_report_json_path, "concentration analysis is a limited scaffold, not full fragility validation", "implement top-trade and top-symbol fragility analysis"),
        ("survivorship_audit", "NOT_IMPLEMENTED", False, False, None, "survivorship audit is not implemented", "audit point-in-time universe membership"),
        ("corporate_action_audit", "NOT_IMPLEMENTED", False, False, None, "corporate-action audit is not implemented", "validate split/dividend adjustment semantics"),
        ("missing_news_bias", "NOT_IMPLEMENTED", False, False, None, "missing-news bias analysis is not complete", "complete covered versus uncovered candidate analysis"),
        ("transaction_cost_validation", "NOT_IMPLEMENTED", False, False, None, "realistic transaction-cost validation is not implemented", "validate cost assumptions and path-dependent replay"),
        ("text_model_readiness", "NOT_READY", True, False, paths.text_model_readiness_json_path, "FinBERT/text modelling is not ready", "complete taxonomy, timestamp, and duplicate-handling gates first"),
        ("news_transformer_scaffold", "NOT_READY", True, False, paths.news_transformer_readiness_json_path, "disabled transformer scaffold is not ready for training or inference", "complete readiness gates before enabling transformer research"),
        ("catastrophic_news_veto", "PASSED_WITH_WARNINGS", True, False, paths.catastrophic_news_audit_json_path, "catastrophic-news veto is research-only and not enforced in replay or strategy", "validate veto taxonomy and point-in-time availability before final validation"),
        ("catastrophic_veto_strategy_comparison", "ATTRIBUTION_ONLY", True, False, paths.catastrophic_veto_strategy_comparison_json_path, "catastrophic-veto replay impact is approximate and not full replay", "compute a full research-only filtered replay before treating veto impact as evaluated"),
        ("catastrophic_veto_filtered_scenario", "APPROXIMATE_LEDGER_SIMULATION", True, False, paths.catastrophic_veto_filtered_strategy_report_json_path, "catastrophic-veto impact is approximate ledger simulation, not full replay", "compute full research-only filtered replay before treating veto impact as validated"),
        ("catastrophic_veto_full_replay", "FULL_REPLAY_COMPUTED" if full_replay_computed else "FULL_REPLAY_NOT_AVAILABLE", True, full_replay_computed, paths.catastrophic_veto_full_replay_report_json_path, "" if full_replay_computed else "replay output does not contain the catastrophic-veto filtered variant", "retain as a separate research-only scenario" if full_replay_computed else "inspect the optional variant replay output"),
        ("catastrophic_veto_replay_seam", "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "ADAPTER_ADDED_NOT_EXECUTED", True, full_replay_computed, paths.catastrophic_veto_replay_seam_report_json_path, "" if full_replay_computed else "opt-in replay adapter is present but not executed as a full replay", "preserve the research-only boundary" if full_replay_computed else "execute and validate separate research-only filtered replay"),
        ("catastrophic_news_evidence_quality", evidence_status, True, False, paths.catastrophic_news_evidence_quality_report_json_path, "catastrophic-news evidence quality insufficient for strict live-style filtering", "improve point-in-time text and availability evidence before any execution use"),
        ("news_evidence_lineage", news_evidence_status, True, False, paths.news_evidence_readiness_report_json_path, "news evidence contract is incomplete across ingestion, feature, and candidate stages", "preserve text, availability, source, category, duplicate, and candidate linkage fields"),
        ("event_taxonomy_research", event_taxonomy_status, True, event_taxonomy_status in {"RESEARCH_RULES_READY"}, paths.news_event_taxonomy_report_json_path, "headline event taxonomy is research-only", "review deterministic event taxonomy coverage"),
        ("duplicate_grouping_heuristic", duplicate_grouping_status, True, duplicate_grouping_status == "HEURISTIC_ONLY", paths.news_duplicate_grouping_report_json_path, "duplicate grouping is heuristic-only and not production-grade", "replace with provider-grade duplicate identifiers before text modelling"),
        ("point_in_time_text_safety", text_safety_status, True, text_safety_status == "PARTIAL_POINT_IN_TIME_SAFE", paths.news_point_in_time_text_safety_report_json_path, "point-in-time text safety is partial", "increase availability timestamp coverage"),
        ("keyword_text_baseline", keyword_baseline_status, True, keyword_baseline_status == "RESEARCH_ONLY", paths.news_text_keyword_baseline_report_json_path, "keyword baseline is research-only and unused by strategy", "keep as diagnostic until validation gates pass"),
        ("catastrophic_veto_bounceback", bounceback_status, True, False, paths.catastrophic_veto_bounceback_report_json_path, "bounce-back attribution is research-only and not a validation gate", "review removed-trade winners/losers before narrowing policy"),
        ("extreme_only_policy_proposal", extreme_policy_status, True, False, paths.catastrophic_veto_extreme_only_policy_proposal_json_path, "extreme-only policy is proposed but not replayed", "run a future separate research-only replay before interpreting policy impact"),
        ("catastrophic_policy_variants", policy_variant_status, True, False, paths.catastrophic_veto_policy_variant_comparison_json_path, "policy variants are research-only diagnostics and not validation gates", "review policy frontier and examples before any future policy narrowing"),
        ("catastrophic_policy_frontier", policy_frontier_status, True, False, paths.catastrophic_veto_policy_frontier_report_json_path, "policy frontier is a diagnostic ranking, not model selection", "treat frontier output as hypothesis triage only"),
        ("loser_bounceback_casebook", casebook_status, True, False, paths.catastrophic_veto_loser_bounceback_casebook_json_path, "casebook is observational research only", "inspect loser-vs-bounceback differences before changing taxonomy"),
        ("taxonomy_improvement_plan", taxonomy_plan_status, True, False, paths.catastrophic_veto_taxonomy_improvement_plan_json_path, "taxonomy improvement plan is proposal-only", "review proposed deterministic rule changes before implementation"),
        ("catastrophic_veto_parked", parked_veto_status, True, False, paths.catastrophic_veto_parked_status_json_path, "catastrophic veto is parked diagnostic-only", "focus validation on news_contrarian_rerank"),
    ]
    gates = []
    for name, status, implemented, passed, artifact, reason, action in gate_specs:
        gates.append(
            {
                "gate_name": name,
                "status": status,
                "implemented": implemented,
                "passed": passed,
                "blocks_final_validation": not passed,
                "required_artifact": str(artifact) if artifact is not None else None,
                "failure_reason": reason,
                "next_required_action": action,
            }
        )
    blocked_by = [gate["gate_name"] for gate in gates if gate["blocks_final_validation"]]
    return {
        "schema_name": "stock_alpha_news_validation_dependency_graph",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "gates": gates,
        "all_required_gates_complete": all(gate["implemented"] for gate in gates),
        "all_required_gates_passed": all(gate["passed"] for gate in gates),
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "is_final_validation": False,
        "blocked_by": blocked_by,
        "warnings": [
            "Artifact presence alone does not pass final validation.",
            "Pseudo-holdout cannot pass final validation.",
            "Text-model readiness cannot pass while FinBERT/text readiness is NOT_READY.",
        ],
    }


def _validation_readiness_dashboard(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    dependency_graph = _validation_dependency_graph(paths)
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    news_evidence = _read_json_if_available(paths.news_evidence_readiness_report_json_path)
    event_taxonomy = _read_json_if_available(paths.news_event_taxonomy_report_json_path)
    duplicate_grouping = _read_json_if_available(paths.news_duplicate_grouping_report_json_path)
    text_safety = _read_json_if_available(paths.news_point_in_time_text_safety_report_json_path)
    keyword_baseline = _read_json_if_available(paths.news_text_keyword_baseline_report_json_path)
    bounceback = _read_json_if_available(paths.catastrophic_veto_bounceback_report_json_path)
    extreme_policy = _read_json_if_available(paths.catastrophic_veto_extreme_only_policy_proposal_json_path)
    policy_variants = _read_json_if_available(paths.catastrophic_veto_policy_variant_comparison_json_path)
    policy_frontier = _read_json_if_available(paths.catastrophic_veto_policy_frontier_report_json_path)
    casebook = _read_json_if_available(paths.catastrophic_veto_loser_bounceback_casebook_json_path)
    taxonomy_plan = _read_json_if_available(paths.catastrophic_veto_taxonomy_improvement_plan_json_path)
    parked_veto = _read_json_if_available(paths.catastrophic_veto_parked_status_json_path)
    walk_forward = _read_json_if_available(paths.contrarian_walk_forward_validation_report_json_path)
    placebo = _read_json_if_available(paths.contrarian_placebo_permutation_report_json_path)
    matched_controls = _read_json_if_available(paths.contrarian_matched_control_report_json_path)
    profit_concentration = _read_json_if_available(paths.contrarian_profit_concentration_report_json_path)
    year_regime = _read_json_if_available(paths.contrarian_year_regime_report_json_path)
    symbol_year_ablation = _read_json_if_available(paths.contrarian_symbol_year_ablation_report_json_path)
    data_validity = _read_json_if_available(paths.contrarian_data_validity_audit_json_path)
    intraday_5min = _read_json_if_available(paths.intraday_5min_expansion_plan_json_path)
    gates = list(dependency_graph.get("gates", []) or [])
    implemented_stage_count = sum(bool(gate.get("implemented")) for gate in gates)
    blocking_stage_count = sum(bool(gate.get("blocks_final_validation")) for gate in gates)
    not_implemented_stage_count = sum(str(gate.get("status")) == "NOT_IMPLEMENTED" for gate in gates)
    not_ready_stage_count = sum(str(gate.get("status")) == "NOT_READY" for gate in gates)
    passed_stage_count = sum(bool(gate.get("passed")) for gate in gates)
    failed_stage_count = sum(str(gate.get("status")) == "FAILED" for gate in gates)
    return {
        "schema_name": "stock_alpha_news_validation_readiness_dashboard",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "overall_status": "DEVELOPMENT_ONLY",
        "research_only": True,
        "production_signal": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "validation_label": "PSEUDO_HOLDOUT",
        "holdout_type": "PSEUDO_HOLDOUT",
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "implemented_stage_count": implemented_stage_count,
        "blocking_stage_count": blocking_stage_count,
        "not_implemented_stage_count": not_implemented_stage_count,
        "not_ready_stage_count": not_ready_stage_count,
        "passed_stage_count": passed_stage_count,
        "failed_stage_count": failed_stage_count,
        "news_evidence_readiness": news_evidence.get("status", "UNAVAILABLE_INPUT"),
        "strict_veto_ready": bool(news_evidence.get("strict_veto_ready", False)),
        "confirmed_only_veto_ready": bool(news_evidence.get("confirmed_only_veto_ready", False)),
        "event_taxonomy_status": event_taxonomy.get("status", "UNAVAILABLE_INPUT"),
        "event_taxonomy_research_ready": bool(event_taxonomy.get("event_taxonomy_research_ready", False)),
        "duplicate_grouping_status": duplicate_grouping.get("status", "UNAVAILABLE_INPUT"),
        "duplicate_grouping_heuristic_ready": bool(duplicate_grouping.get("duplicate_grouping_heuristic_ready", False)),
        "point_in_time_text_safety_status": text_safety.get("status", "UNAVAILABLE_INPUT"),
        "point_in_time_text_safety_ready": bool(text_safety.get("point_in_time_text_safety_ready", False)),
        "keyword_baseline_status": keyword_baseline.get("status", "UNAVAILABLE_INPUT"),
        "keyword_baseline_ready": bool(keyword_baseline.get("keyword_baseline_ready", False)),
        "catastrophic_veto_bounceback_status": bounceback.get("status", "UNAVAILABLE_INPUT"),
        "extreme_only_policy_proposal_status": extreme_policy.get("status", "UNAVAILABLE_INPUT"),
        "catastrophic_policy_variant_status": policy_variants.get("status", "UNAVAILABLE_INPUT"),
        "catastrophic_policy_frontier_status": policy_frontier.get("status", "UNAVAILABLE_INPUT"),
        "best_balanced_catastrophic_policy": policy_frontier.get("best_balanced_policy", "UNAVAILABLE_INPUT"),
        "loser_bounceback_casebook_status": casebook.get("status", "UNAVAILABLE_INPUT"),
        "taxonomy_improvement_plan_status": taxonomy_plan.get("status", "UNAVAILABLE_INPUT"),
        "catastrophic_veto": parked_veto.get("status", "UNAVAILABLE_INPUT"),
        "contrarian_validation": "IN_PROGRESS",
        "walk_forward": walk_forward.get("status", "UNAVAILABLE_INPUT"),
        "placebo": placebo.get("status", "UNAVAILABLE_INPUT"),
        "matched_controls": matched_controls.get("status", "UNAVAILABLE_INPUT"),
        "profit_concentration": profit_concentration.get("status", "UNAVAILABLE_INPUT"),
        "year_regime": year_regime.get("status", "UNAVAILABLE_INPUT"),
        "symbol_year_ablation": symbol_year_ablation.get("status", "UNAVAILABLE_INPUT"),
        "data_validity": data_validity.get("status", "UNAVAILABLE_INPUT"),
        "intraday_5min": intraday_5min.get("status", "UNAVAILABLE_INPUT"),
        "text_model_ready": False,
        "transformer_ready": False,
        "top_blockers": [item for item in [
            "walk-forward not implemented",
            "placebo/permutation not implemented",
            "matched controls not implemented",
            "review year/regime robustness",
            "review symbol/year ablations",
            "survivorship audit not implemented",
            "corporate-action audit not implemented",
            "missing-news bias not implemented",
            "events still uncategorized",
            "catastrophic-news evidence quality insufficient for strict live-style filtering",
            "news evidence contract incomplete across ingestion and candidate stages",
            None if full_replay_computed else "catastrophic-veto full replay not computed",
            "text model readiness not ready",
            "pseudo-holdout is not genuine holdout",
        ] if item is not None],
        "safe_next_steps": [item for item in [
            "implement walk-forward validation",
            "implement placebo/permutation checks",
            "implement matched controls",
            "implement concentration/fragility analysis",
            "implement survivorship/corporate-action/missing-news audits",
            "build structured event taxonomy",
            None if full_replay_computed else "compute full research-only catastrophic-veto filtered replay",
        ] if item is not None],
        "unsafe_next_steps": [
            "FinBERT",
            "BERT",
            "transformer training",
            "paper trading",
            "live trading",
            "new providers",
            "news-originated entries",
        ],
        "finbert_readiness": "NOT_READY",
        "warnings": [
            "Dashboard is a readiness rollup, not strategy validation.",
            "Development-only pseudo-holdout results remain blocked from final validation.",
            "Paper/live trading and text-model training remain disabled.",
        ],
    }


def _artifact_lineage_report(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    relationships = [
        (paths.decile_join_audit_json_path, paths.validation_readiness_dashboard_json_path, "feeds_readiness_status", True),
        (paths.decile_trade_reconciliation_json_path, paths.validation_readiness_dashboard_json_path, "feeds_readiness_status", True),
        (paths.chronological_split_manifest_json_path, paths.contrarian_grid_selection_json_path, "defines_selection_periods", True),
        (paths.chronological_split_manifest_json_path, paths.contrarian_holdout_report_json_path, "defines_holdout_period", True),
        (paths.contrarian_grid_selection_json_path, paths.contrarian_frozen_config_json_path, "freezes_selected_configuration", True),
        (paths.contrarian_frozen_config_json_path, paths.contrarian_holdout_report_json_path, "drives_holdout_evaluation", True),
        (paths.validation_stage_placeholders_json_path, paths.validation_dependency_graph_json_path, "declares_future_validation_blockers", True),
        (paths.text_model_readiness_json_path, paths.validation_dependency_graph_json_path, "declares_text_model_blockers", True),
        (paths.news_transformer_readiness_json_path, paths.validation_dependency_graph_json_path, "declares_transformer_scaffold_blockers", True),
        (paths.news_transformer_training_plan_json_path, paths.validation_readiness_dashboard_json_path, "documents_disabled_transformer_plan", True),
        (paths.catastrophic_news_audit_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_research_only_catastrophic_news_audit", True),
        (paths.catastrophic_news_veto_report_json_path, paths.validation_dependency_graph_json_path, "declares_research_only_catastrophic_news_veto_blocker", True),
        (paths.catastrophic_veto_policy_json_path, paths.catastrophic_veto_candidate_attribution_json_path, "defines_research_only_veto_policy", True),
        (paths.catastrophic_news_audit_json_path, paths.catastrophic_veto_candidate_attribution_json_path, "feeds_candidate_veto_attribution", True),
        (paths.catastrophic_veto_candidate_attribution_json_path, paths.catastrophic_veto_strategy_comparison_json_path, "feeds_attribution_only_strategy_comparison", True),
        (paths.catastrophic_veto_strategy_comparison_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_catastrophic_veto_replay_impact_status", True),
        (paths.catastrophic_veto_trade_attribution_csv_path, paths.catastrophic_veto_removed_trades_csv_path, "identifies_research_only_removed_trades", True),
        (paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_removed_symbols_csv_path, "rolls_removed_trades_up_by_symbol", True),
        (paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_filtered_strategy_report_json_path, "feeds_approximate_filtered_strategy_report", True),
        (paths.catastrophic_veto_filtered_strategy_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_approximate_veto_simulation", True),
        (paths.catastrophic_veto_filtered_candidates_csv_path, paths.catastrophic_veto_full_replay_report_json_path, "documents_full_replay_candidate_filter_input", True),
        (paths.catastrophic_veto_blocked_candidates_csv_path, paths.catastrophic_veto_full_replay_report_json_path, "documents_candidates_excluded_from_full_replay_variant", True),
        (paths.catastrophic_veto_replay_seam_report_json_path, paths.catastrophic_veto_full_replay_report_json_path, "documents_filtered_replay_input_seam", True),
        (paths.catastrophic_veto_replay_seam_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_replay_seam_status", True),
        (paths.catastrophic_veto_full_replay_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_full_replay_availability", True),
        (paths.catastrophic_news_evidence_quality_report_json_path, paths.validation_dependency_graph_json_path, "declares_catastrophic_evidence_quality_blocker", True),
        (paths.catastrophic_news_evidence_quality_report_json_path, paths.catastrophic_veto_policy_mode_comparison_json_path, "feeds_research_policy_mode_comparison", True),
        (paths.catastrophic_veto_policy_mode_comparison_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_research_only_policy_modes", True),
        (paths.news_evidence_lineage_report_json_path, paths.news_evidence_readiness_report_json_path, "feeds_news_evidence_readiness", True),
        (paths.news_evidence_readiness_report_json_path, paths.validation_dependency_graph_json_path, "declares_news_evidence_readiness_blocker", True),
        (paths.news_evidence_readiness_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_news_evidence_readiness", True),
        (paths.news_evidence_readiness_report_json_path, paths.news_event_taxonomy_report_json_path, "gates_research_taxonomy_inputs", True),
        (paths.news_event_taxonomy_report_json_path, paths.news_event_taxonomy_counts_csv_path, "summarizes_research_taxonomy_counts", True),
        (paths.news_event_taxonomy_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_event_taxonomy_research_status", True),
        (paths.news_evidence_readiness_report_json_path, paths.news_duplicate_grouping_report_json_path, "gates_heuristic_duplicate_grouping_inputs", True),
        (paths.news_duplicate_grouping_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_duplicate_grouping_status", True),
        (paths.news_evidence_readiness_report_json_path, paths.news_point_in_time_text_safety_report_json_path, "gates_text_safety_inputs", True),
        (paths.news_point_in_time_text_safety_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_point_in_time_text_safety", True),
        (paths.news_point_in_time_text_safety_report_json_path, paths.news_text_keyword_baseline_report_json_path, "documents_keyword_baseline_input_safety", True),
        (paths.news_text_keyword_baseline_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_keyword_baseline_status", True),
        (paths.catastrophic_veto_removed_trades_csv_path, paths.catastrophic_veto_bounceback_report_json_path, "feeds_removed_trade_bounceback_attribution", True),
        (paths.news_event_taxonomy_report_json_path, paths.catastrophic_veto_bounceback_report_json_path, "supports_reversible_vs_extreme_grouping", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_extreme_only_policy_proposal_json_path, "motivates_future_narrow_policy", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_bounceback_status", True),
        (paths.catastrophic_veto_extreme_only_policy_proposal_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_extreme_only_policy_proposal", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_policy_variant_comparison_json_path, "feeds_policy_variant_bounceback_tradeoff", True),
        (paths.news_event_taxonomy_report_json_path, paths.catastrophic_veto_policy_variant_comparison_json_path, "feeds_policy_variant_taxonomy", True),
        (paths.catastrophic_veto_policy_variant_comparison_json_path, paths.catastrophic_veto_policy_frontier_report_json_path, "feeds_policy_frontier_ranking", True),
        (paths.catastrophic_veto_policy_variant_comparison_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_policy_variant_status", True),
        (paths.catastrophic_veto_policy_frontier_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_policy_frontier_status", True),
        (paths.catastrophic_veto_bounceback_report_json_path, paths.catastrophic_veto_loser_bounceback_casebook_json_path, "feeds_loser_bounceback_casebook", True),
        (paths.catastrophic_veto_loser_bounceback_casebook_json_path, paths.catastrophic_veto_taxonomy_improvement_plan_json_path, "motivates_taxonomy_improvement_plan", True),
        (paths.catastrophic_veto_loser_bounceback_casebook_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_casebook_status", True),
        (paths.catastrophic_veto_taxonomy_improvement_plan_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_taxonomy_plan_status", True),
        (paths.catastrophic_veto_parked_status_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_parked_catastrophic_veto_status", True),
        (paths.contrarian_chronological_validation_plan_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_chronology_gate", True),
        (paths.contrarian_chronological_periods_csv_path, paths.validation_readiness_dashboard_json_path, "summarizes_main_contrarian_chronology_periods", True),
        (paths.contrarian_walk_forward_validation_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_walk_forward_gate", True),
        (paths.contrarian_placebo_permutation_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_placebo_gate", True),
        (paths.contrarian_matched_control_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_matched_control_gate", True),
        (paths.contrarian_profit_concentration_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_profit_concentration_gate", True),
        (paths.contrarian_year_regime_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_year_regime_gate", True),
        (paths.contrarian_symbol_year_ablation_report_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_ablation_gate", True),
        (paths.contrarian_data_validity_audit_json_path, paths.validation_dependency_graph_json_path, "declares_main_contrarian_data_validity_gate", True),
        (paths.intraday_5min_expansion_plan_json_path, paths.validation_readiness_dashboard_json_path, "documents_future_intraday_expansion_plan", True),
        (paths.artifact_validation_report_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_artifact_presence", True),
        (paths.experiment_registry_jsonl_path, paths.validation_readiness_dashboard_json_path, "records_experiment_history", True),
        (paths.validation_dependency_graph_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_gate_status", True),
        (paths.news_validation_workflow_map_json_path, paths.validation_readiness_dashboard_json_path, "summarizes_workflow_status", True),
        (paths.chronological_split_manifest_json_path, paths.walk_forward_validation_report_json_path, "defines_walk_forward_periods", True),
        (paths.contrarian_frozen_config_json_path, paths.walk_forward_validation_report_json_path, "freezes_walk_forward_configuration", True),
        (paths.contrarian_frozen_config_json_path, paths.placebo_permutation_report_json_path, "freezes_placebo_configuration", True),
        (paths.walk_forward_validation_report_json_path, paths.validation_dependency_graph_json_path, "feeds_walk_forward_gate", True),
        (paths.placebo_permutation_report_json_path, paths.validation_dependency_graph_json_path, "feeds_placebo_gate", True),
        (paths.exposure_matched_controls_json_path, paths.validation_dependency_graph_json_path, "feeds_exposure_control_gate", True),
        (paths.trade_count_matched_controls_json_path, paths.validation_dependency_graph_json_path, "feeds_trade_count_control_gate", True),
        (paths.concentration_fragility_report_json_path, paths.validation_dependency_graph_json_path, "feeds_concentration_gate", True),
    ]
    lineage = [
        {
            "source_artifact": source.name,
            "target_artifact": target.name,
            "relationship_type": relationship_type,
            "required": required,
            "status": "DECLARED",
            "warning_if_missing": f"{source.name} missing would make {target.name} incomplete.",
        }
        for source, target, relationship_type, required in relationships
    ]
    return {
        "schema_name": "stock_alpha_news_artifact_lineage_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "DECLARED",
        "lineage": lineage,
        "warnings": [
            "Lineage report documents artifact dependencies only.",
            "Declared lineage does not imply final validation.",
        ],
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
