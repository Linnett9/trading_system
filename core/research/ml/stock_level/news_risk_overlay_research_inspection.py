from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def _build_executive_summary(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifacts = _news_risk_artifact_paths(output_dir)
    status = [_artifact_status(name, path) for name, path in artifacts.items()]
    risk = _read_json_if_available(artifacts["risk_metrics"])
    coverage = _read_json_if_available(artifacts["coverage_report"])
    direction = _read_json_if_available(artifacts["news_score_direction_report"])
    cost = _read_json_if_available(artifacts["cost_scenario_comparison"])
    event = _read_json_if_available(artifacts["event_category_analysis"])
    extreme = _read_json_if_available(artifacts["extreme_event_memory_report"])
    comparison = _read_json_if_available(artifacts["portfolio_comparison"])
    replay_audit = _read_json_if_available(artifacts["replay_data_audit"])
    holdout = _read_json_if_available(artifacts["contrarian_holdout_report"])
    grid_selection = _read_json_if_available(artifacts["contrarian_grid_selection"])
    walk_forward = _read_json_if_available(artifacts["contrarian_walk_forward_summary"])
    walk_forward_validation = _read_json_if_available(artifacts["contrarian_walk_forward_validation_report"])
    placebo_permutation = _read_json_if_available(artifacts["contrarian_placebo_permutation_report"])
    matched_control = _read_json_if_available(artifacts["contrarian_matched_control_report"])
    profit_concentration = _read_json_if_available(artifacts["contrarian_profit_concentration_report"])
    year_regime = _read_json_if_available(artifacts["contrarian_year_regime_report"])
    symbol_year_ablation = _read_json_if_available(artifacts["contrarian_symbol_year_ablation_report"])
    data_validity = _read_json_if_available(artifacts["contrarian_data_validity_audit"])
    intraday_5min = _read_json_if_available(artifacts["intraday_5min_expansion_plan"])
    placebo = _read_json_if_available(artifacts["contrarian_placebo_summary"])
    matched = _read_json_if_available(artifacts["contrarian_matched_controls"])
    concentration = _read_json_if_available(artifacts["contrarian_concentration_report"])
    universe = _read_json_if_available(artifacts["universe_survivorship_audit"])
    corporate_actions = _read_json_if_available(artifacts["corporate_action_audit"])
    catastrophic_veto_attribution = _read_json_if_available(artifacts["catastrophic_veto_candidate_attribution"])
    catastrophic_veto_comparison = _read_json_if_available(artifacts["catastrophic_veto_strategy_comparison"])
    catastrophic_veto_filtered = _read_json_if_available(artifacts["catastrophic_veto_filtered_strategy_report"])
    catastrophic_veto_full_replay = _read_json_if_available(artifacts["catastrophic_veto_full_replay_report"])
    catastrophic_evidence_quality = _read_json_if_available(artifacts["catastrophic_news_evidence_quality_report"])
    news_evidence_readiness = _read_json_if_available(artifacts["news_evidence_readiness_report"])
    event_taxonomy = _read_json_if_available(artifacts["news_event_taxonomy_report"])
    duplicate_grouping = _read_json_if_available(artifacts["news_duplicate_grouping_report"])
    text_safety = _read_json_if_available(artifacts["news_point_in_time_text_safety_report"])
    keyword_baseline = _read_json_if_available(artifacts["news_text_keyword_baseline_report"])
    bounceback = _read_json_if_available(artifacts["catastrophic_veto_bounceback_report"])
    extreme_policy = _read_json_if_available(artifacts["catastrophic_veto_extreme_only_policy_proposal"])
    policy_frontier = _read_json_if_available(artifacts["catastrophic_veto_policy_frontier_report"])
    casebook = _read_json_if_available(artifacts["catastrophic_veto_loser_bounceback_casebook"])
    taxonomy_plan = _read_json_if_available(artifacts["catastrophic_veto_taxonomy_improvement_plan"])
    parked_veto = _read_json_if_available(artifacts["catastrophic_veto_parked_status"])
    deciles = _read_csv_if_available(artifacts["news_score_deciles"])
    strategy_rows = _strategy_summary_rows(risk)
    cost_rows = _cost_robustness_rows(cost)
    diagnostics = _diagnostics_summary(
        coverage=coverage,
        direction=direction,
        event=event,
        extreme=extreme,
        risk=risk,
        cost_rows=cost_rows,
        comparison=comparison,
        holdout=holdout,
        grid_selection=grid_selection,
        walk_forward=walk_forward,
        placebo=placebo,
        matched=matched,
        concentration=concentration,
        universe=universe,
        corporate_actions=corporate_actions,
    )
    warnings = _executive_warnings(
        artifact_status=status,
        deciles=deciles,
        diagnostics=diagnostics,
        cost_rows=cost_rows,
        replay_audit=replay_audit,
        risk=risk,
    )
    return {
        "output_dir": str(output_dir),
        "strategy_comparison": strategy_rows,
        "cost_robustness": cost_rows,
        "diagnostics": diagnostics,
        "catastrophic_veto": _catastrophic_veto_summary(
            catastrophic_veto_attribution,
            catastrophic_veto_comparison,
            catastrophic_veto_filtered,
            catastrophic_veto_full_replay,
            catastrophic_evidence_quality,
        ),
        "news_evidence": news_evidence_readiness,
        "event_taxonomy": event_taxonomy,
        "duplicate_grouping": duplicate_grouping,
        "text_safety": text_safety,
        "keyword_baseline": keyword_baseline,
        "catastrophic_veto_bounceback": bounceback,
        "extreme_only_policy_proposal": extreme_policy,
        "catastrophic_policy_frontier": policy_frontier,
        "loser_bounceback_casebook": casebook,
        "taxonomy_improvement_plan": taxonomy_plan,
        "catastrophic_veto_parked": parked_veto,
        "contrarian_validation": {
            "status": "IN_PROGRESS",
            "walk_forward": walk_forward_validation.get("status", "UNAVAILABLE_INPUT"),
            "placebo": placebo_permutation.get("status", "UNAVAILABLE_INPUT"),
            "matched_controls": matched_control.get("status", "UNAVAILABLE_INPUT"),
            "profit_concentration": profit_concentration.get("status", "UNAVAILABLE_INPUT"),
            "year_regime": year_regime.get("status", "UNAVAILABLE_INPUT"),
            "symbol_year_ablation": symbol_year_ablation.get("status", "UNAVAILABLE_INPUT"),
            "data_validity": data_validity.get("status", "UNAVAILABLE_INPUT"),
            "intraday_5min": intraday_5min.get("status", "UNAVAILABLE_INPUT"),
        },
        "winners": _winner_summary(strategy_rows, cost_rows),
        "warnings": warnings,
        "paper_orders_enabled": bool(comparison.get("paper_orders_enabled", False)),
        "live_orders_enabled": bool(comparison.get("live_orders_enabled", False)),
    }, status


def _catastrophic_veto_summary_lines(output_dir: Path) -> list[str]:
    paths = _news_risk_artifact_paths(output_dir)
    attribution = _read_optional_json(paths["catastrophic_veto_candidate_attribution"])
    comparison = _read_optional_json(paths["catastrophic_veto_strategy_comparison"])
    filtered = _read_optional_json(paths["catastrophic_veto_filtered_strategy_report"])
    full_replay = _read_optional_json(paths["catastrophic_veto_full_replay_report"])
    evidence_quality = _read_optional_json(paths["catastrophic_news_evidence_quality_report"])
    blocked_candidates = full_replay.get(
        "strict_policy_blocked_candidate_count",
        full_replay.get("blocked_candidate_count", attribution.get("blocked_candidate_count", "UNAVAILABLE_INPUT")),
    )
    manual_review_candidates = full_replay.get(
        "manual_review_candidate_count",
        attribution.get("manual_review_candidate_count", "UNAVAILABLE_INPUT"),
    )
    blocked_trades = full_replay.get(
        "removed_trade_count",
        attribution.get("blocked_executed_trade_count", "NOT_COMPUTED"),
    )
    replay_status = (
        full_replay.get("replay_impact_status")
        or filtered.get("replay_impact_status")
        or comparison.get("replay_impact_status")
        or "NOT_COMPUTED"
    )
    approximate_status = (
        filtered.get("replay_impact_status")
        or comparison.get("replay_impact_status")
        or "NOT_COMPUTED"
    )
    delta_metrics = dict(full_replay.get("delta_metrics", {}) or {})
    removed_count = full_replay.get("removed_trade_count", "UNAVAILABLE_INPUT")
    replacement_count = full_replay.get("replacement_trade_count", "UNAVAILABLE_INPUT")
    return [
        f"catastrophic news veto: RESEARCH_ONLY / NOT_CURRENT_STRATEGY ({replay_status})",
        f"catastrophic veto full replay: {replay_status}",
        f"catastrophic evidence quality: {evidence_quality.get('status', 'UNAVAILABLE_INPUT')}",
        f"catastrophic usable strict-veto candidates: {evidence_quality.get('usable_for_strict_veto_count', 'UNAVAILABLE_INPUT')}",
        "catastrophic policy modes: STRICT_SAFETY / CONFIRMED_ONLY_RESEARCH / MANUAL_REVIEW_RESEARCH",
        f"catastrophic veto scenario: {replay_status}",
        f"catastrophic veto approximate simulation: {approximate_status}",
        f"catastrophic veto candidates before/after: {full_replay.get('candidate_count_before_veto', 'UNAVAILABLE_INPUT')}/{full_replay.get('candidate_count_after_veto', 'UNAVAILABLE_INPUT')}",
        f"catastrophic blocked candidates: {blocked_candidates}",
        f"catastrophic veto blocked trades: {blocked_trades}",
        f"catastrophic veto trades removed/replaced: {removed_count}/{replacement_count}",
        "catastrophic veto paper/live allowed: False / False",
        f"catastrophic veto delta return: {delta_metrics.get('delta_return', 'UNAVAILABLE_INPUT')}",
        f"manual review candidates: {manual_review_candidates}",
    ]


def _catastrophic_veto_summary(
    attribution: Mapping[str, Any],
    comparison: Mapping[str, Any],
    filtered_report: Mapping[str, Any],
    full_replay_report: Mapping[str, Any],
    evidence_quality_report: Mapping[str, Any],
) -> dict[str, Any]:
    full_delta = dict(full_replay_report.get("delta_metrics", {}) or {})
    return {
        "replay_impact_status": full_replay_report.get(
            "replay_impact_status",
            filtered_report.get(
                "replay_impact_status",
                comparison.get("replay_impact_status", "NOT_COMPUTED"),
            ),
        ),
        "approximate_replay_impact_status": filtered_report.get("replay_impact_status", "NOT_COMPUTED"),
        "candidate_count_before_veto": full_replay_report.get("candidate_count_before_veto", "UNAVAILABLE_INPUT"),
        "candidate_count_after_veto": full_replay_report.get("candidate_count_after_veto", "UNAVAILABLE_INPUT"),
        "blocked_candidate_count": full_replay_report.get("strict_policy_blocked_candidate_count", full_replay_report.get("blocked_candidate_count", attribution.get("blocked_candidate_count", "UNAVAILABLE_INPUT"))),
        "blocked_trade_count": full_replay_report.get("removed_trade_count", attribution.get("blocked_executed_trade_count", "NOT_COMPUTED")),
        "manual_review_candidate_count": full_replay_report.get("manual_review_candidate_count", attribution.get("manual_review_candidate_count", "UNAVAILABLE_INPUT")),
        "delta_return": full_delta.get("delta_return", "UNAVAILABLE_INPUT"),
        "evidence_quality_status": evidence_quality_report.get("status", "UNAVAILABLE_INPUT"),
        "usable_for_strict_veto_count": evidence_quality_report.get("usable_for_strict_veto_count", "UNAVAILABLE_INPUT"),
        "policy_modes": "STRICT_SAFETY / CONFIRMED_ONLY_RESEARCH / MANUAL_REVIEW_RESEARCH",
    }


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _news_risk_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "coverage_report": output_dir / "coverage_report.json",
        "risk_metrics": output_dir / "risk_metrics.json",
        "portfolio_comparison": output_dir / "portfolio_comparison.json",
        "news_score_deciles": output_dir / "news_score_deciles.csv",
        "news_score_direction_report": output_dir / "news_score_direction_report.json",
        "cost_scenario_comparison": output_dir / "cost_scenario_comparison.json",
        "event_category_analysis": output_dir / "event_category_analysis.json",
        "extreme_event_memory_report": output_dir / "extreme_event_memory_report.json",
        "replay_data_audit": output_dir / "replay_data_audit.json",
        "corrected_news_score_deciles": output_dir / "corrected_news_score_deciles.csv",
        "decile_join_audit": output_dir / "decile_join_audit.json",
        "decile_trade_reconciliation": output_dir / "decile_trade_reconciliation.json",
        "chronological_split_manifest": output_dir / "chronological_split_manifest.json",
        "experiment_registry": output_dir / "experiment_registry.jsonl",
        "contrarian_grid_results": output_dir / "contrarian_grid_results.csv",
        "contrarian_grid_selection": output_dir / "contrarian_grid_selection.json",
        "contrarian_fold_results": output_dir / "contrarian_fold_results.csv",
        "contrarian_parameter_stability": output_dir / "contrarian_parameter_stability.json",
        "contrarian_frozen_config": output_dir / "contrarian_frozen_config.json",
        "contrarian_holdout_report": output_dir / "contrarian_holdout_report.json",
        "contrarian_holdout_trade_ledger": output_dir / "contrarian_holdout_trade_ledger.csv",
        "contrarian_holdout_equity": output_dir / "contrarian_holdout_equity.csv",
        "contrarian_holdout_comparison": output_dir / "contrarian_holdout_comparison.md",
        "contrarian_walk_forward_folds": output_dir / "contrarian_walk_forward_folds.csv",
        "contrarian_walk_forward_summary": output_dir / "contrarian_walk_forward_summary.json",
        "contrarian_chronological_validation_plan": output_dir / "contrarian_chronological_validation_plan.json",
        "contrarian_chronological_periods": output_dir / "contrarian_chronological_periods.csv",
        "contrarian_walk_forward_validation_report": output_dir / "contrarian_walk_forward_validation_report.json",
        "contrarian_placebo_permutation_report": output_dir / "contrarian_placebo_permutation_report.json",
        "contrarian_placebo_permutation_results": output_dir / "contrarian_placebo_permutation_results.csv",
        "contrarian_matched_control_report": output_dir / "contrarian_matched_control_report.json",
        "contrarian_matched_control_results": output_dir / "contrarian_matched_control_results.csv",
        "contrarian_profit_concentration_report": output_dir / "contrarian_profit_concentration_report.json",
        "contrarian_trade_fragility_by_symbol": output_dir / "contrarian_trade_fragility_by_symbol.csv",
        "contrarian_trade_fragility_by_year": output_dir / "contrarian_trade_fragility_by_year.csv",
        "contrarian_top_trade_removal": output_dir / "contrarian_top_trade_removal.csv",
        "contrarian_year_regime_report": output_dir / "contrarian_year_regime_report.json",
        "contrarian_year_regime_results": output_dir / "contrarian_year_regime_results.csv",
        "contrarian_year_regime_examples": output_dir / "contrarian_year_regime_examples.csv",
        "contrarian_symbol_year_ablation_report": output_dir / "contrarian_symbol_year_ablation_report.json",
        "contrarian_without_top_symbols": output_dir / "contrarian_without_top_symbols.csv",
        "contrarian_without_top_years": output_dir / "contrarian_without_top_years.csv",
        "contrarian_cost_slippage_robustness_report": output_dir / "contrarian_cost_slippage_robustness_report.json",
        "contrarian_cost_slippage_robustness": output_dir / "contrarian_cost_slippage_robustness.csv",
        "contrarian_data_validity_audit": output_dir / "contrarian_data_validity_audit.json",
        "intraday_5min_expansion_plan": output_dir / "intraday_5min_expansion_plan.json",
        "contrarian_placebo_results": output_dir / "contrarian_placebo_results.csv",
        "contrarian_placebo_summary": output_dir / "contrarian_placebo_summary.json",
        "contrarian_matched_controls": output_dir / "contrarian_matched_controls.json",
        "contrarian_contribution_by_year": output_dir / "contrarian_contribution_by_year.csv",
        "contrarian_contribution_by_symbol": output_dir / "contrarian_contribution_by_symbol.csv",
        "contrarian_concentration_report": output_dir / "contrarian_concentration_report.json",
        "universe_survivorship_audit": output_dir / "universe_survivorship_audit.json",
        "universe_membership_by_date": output_dir / "universe_membership_by_date.csv",
        "corporate_action_audit": output_dir / "corporate_action_audit.json",
        "missing_news_bias_report": output_dir / "missing_news_bias_report.json",
        "covered_vs_uncovered_candidates": output_dir / "covered_vs_uncovered_candidates.csv",
        "text_model_readiness": output_dir / "text_model_readiness.json",
        "news_transformer_readiness": output_dir / "news_transformer_readiness.json",
        "news_transformer_training_plan": output_dir / "news_transformer_training_plan.json",
        "catastrophic_news_audit": output_dir / "catastrophic_news_audit.json",
        "catastrophic_news_candidates": output_dir / "catastrophic_news_candidates.csv",
        "catastrophic_news_veto_report": output_dir / "catastrophic_news_veto_report.json",
        "catastrophic_veto_candidate_attribution": output_dir / "catastrophic_veto_candidate_attribution.json",
        "catastrophic_veto_trade_attribution": output_dir / "catastrophic_veto_trade_attribution.csv",
        "catastrophic_veto_strategy_comparison": output_dir / "catastrophic_veto_strategy_comparison.json",
        "catastrophic_veto_policy": output_dir / "catastrophic_veto_policy.json",
        "catastrophic_veto_filtered_strategy_report": output_dir / "catastrophic_veto_filtered_strategy_report.json",
        "catastrophic_veto_removed_trades": output_dir / "catastrophic_veto_removed_trades.csv",
        "catastrophic_veto_removed_symbols": output_dir / "catastrophic_veto_removed_symbols.csv",
        "catastrophic_veto_full_replay_report": output_dir / "catastrophic_veto_full_replay_report.json",
        "catastrophic_veto_full_replay_trade_ledger": output_dir / "catastrophic_veto_full_replay_trade_ledger.csv",
        "catastrophic_veto_full_replay_equity": output_dir / "catastrophic_veto_full_replay_equity.csv",
        "catastrophic_veto_parked_status": output_dir / "catastrophic_veto_parked_status.json",
        "catastrophic_veto_loser_bounceback_casebook": output_dir / "catastrophic_veto_loser_bounceback_casebook.json",
        "catastrophic_veto_taxonomy_improvement_plan": output_dir / "catastrophic_veto_taxonomy_improvement_plan.json",
        "catastrophic_veto_filtered_candidates": output_dir / "catastrophic_veto_filtered_candidates.csv",
        "catastrophic_veto_blocked_candidates": output_dir / "catastrophic_veto_blocked_candidates.csv",
        "catastrophic_veto_replay_seam_report": output_dir / "catastrophic_veto_replay_seam_report.json",
        "catastrophic_veto_bounceback_report": output_dir / "catastrophic_veto_bounceback_report.json",
        "catastrophic_veto_bounceback_by_category": output_dir / "catastrophic_veto_bounceback_by_category.csv",
        "catastrophic_veto_bounceback_examples": output_dir / "catastrophic_veto_bounceback_examples.csv",
        "catastrophic_veto_extreme_only_policy_proposal": output_dir / "catastrophic_veto_extreme_only_policy_proposal.json",
        "catastrophic_veto_policy_variant_comparison": output_dir / "catastrophic_veto_policy_variant_comparison.json",
        "catastrophic_veto_policy_variant_counts": output_dir / "catastrophic_veto_policy_variant_counts.csv",
        "catastrophic_veto_policy_variant_metrics": output_dir / "catastrophic_veto_policy_variant_metrics.csv",
        "catastrophic_veto_policy_variant_removed_trades": output_dir / "catastrophic_veto_policy_variant_removed_trades.csv",
        "catastrophic_veto_policy_variant_bounceback": output_dir / "catastrophic_veto_policy_variant_bounceback.csv",
        "catastrophic_veto_policy_frontier_report": output_dir / "catastrophic_veto_policy_frontier_report.json",
        "catastrophic_veto_policy_frontier": output_dir / "catastrophic_veto_policy_frontier.csv",
        "catastrophic_veto_policy_variant_examples": output_dir / "catastrophic_veto_policy_variant_examples.csv",
        "catastrophic_veto_loser_bounceback_casebook": output_dir / "catastrophic_veto_loser_bounceback_casebook.json",
        "catastrophic_veto_loser_bounceback_cases": output_dir / "catastrophic_veto_loser_bounceback_cases.csv",
        "catastrophic_veto_loser_bounceback_feature_diff": output_dir / "catastrophic_veto_loser_bounceback_feature_diff.csv",
        "catastrophic_veto_loser_bounceback_keyword_diff": output_dir / "catastrophic_veto_loser_bounceback_keyword_diff.csv",
        "catastrophic_veto_taxonomy_improvement_plan": output_dir / "catastrophic_veto_taxonomy_improvement_plan.json",
        "catastrophic_news_evidence_quality_report": output_dir / "catastrophic_news_evidence_quality_report.json",
        "catastrophic_news_evidence_quality_by_field": output_dir / "catastrophic_news_evidence_quality_by_field.csv",
        "catastrophic_news_evidence_quality_by_symbol": output_dir / "catastrophic_news_evidence_quality_by_symbol.csv",
        "catastrophic_veto_policy_mode_comparison": output_dir / "catastrophic_veto_policy_mode_comparison.json",
        "catastrophic_veto_policy_mode_counts": output_dir / "catastrophic_veto_policy_mode_counts.csv",
        "news_evidence_lineage_report": output_dir / "news_evidence_lineage_report.json",
        "news_evidence_lineage_by_stage": output_dir / "news_evidence_lineage_by_stage.csv",
        "news_evidence_missing_field_examples": output_dir / "news_evidence_missing_field_examples.csv",
        "news_evidence_readiness_report": output_dir / "news_evidence_readiness_report.json",
        "news_event_taxonomy_report": output_dir / "news_event_taxonomy_report.json",
        "news_event_taxonomy_counts": output_dir / "news_event_taxonomy_counts.csv",
        "news_event_taxonomy_examples": output_dir / "news_event_taxonomy_examples.csv",
        "news_duplicate_grouping_report": output_dir / "news_duplicate_grouping_report.json",
        "news_duplicate_grouping_examples": output_dir / "news_duplicate_grouping_examples.csv",
        "news_point_in_time_text_safety_report": output_dir / "news_point_in_time_text_safety_report.json",
        "news_point_in_time_text_safety_examples": output_dir / "news_point_in_time_text_safety_examples.csv",
        "news_text_keyword_baseline_report": output_dir / "news_text_keyword_baseline_report.json",
        "news_text_keyword_baseline_scores": output_dir / "news_text_keyword_baseline_scores.csv",
        "walk_forward_validation_report": output_dir / "walk_forward_validation_report.json",
        "walk_forward_fold_results": output_dir / "walk_forward_fold_results.csv",
        "placebo_permutation_report": output_dir / "placebo_permutation_report.json",
        "placebo_permutation_results": output_dir / "placebo_permutation_results.csv",
        "exposure_matched_controls": output_dir / "exposure_matched_controls.json",
        "trade_count_matched_controls": output_dir / "trade_count_matched_controls.json",
        "concentration_fragility_report": output_dir / "concentration_fragility_report.json",
        "validation_stage_placeholders": output_dir / "validation_stage_placeholders.json",
        "artifact_manifest": output_dir / "artifact_manifest.json",
        "artifact_validation_report": output_dir / "artifact_validation_report.json",
        "news_validation_workflow_map": output_dir / "news_validation_workflow_map.json",
        "validation_dependency_graph": output_dir / "validation_dependency_graph.json",
        "validation_readiness_dashboard": output_dir / "validation_readiness_dashboard.json",
        "artifact_lineage_report": output_dir / "artifact_lineage_report.json",
        "news_validation_gap_analysis": output_dir / "news_validation_gap_analysis.json",
        "parallel_execution_report": output_dir / "parallel_execution_report.json",
        "README": output_dir / "README.md",
    }


def _artifact_status(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        if name == "parallel_execution_report":
            return {
                "name": name,
                "path": str(path),
                "status": "NOT_ENABLED",
                "bytes": 0,
                "reason": "parallel execution report is only required after a research run with reporting enabled",
            }
        return {"name": name, "path": str(path), "status": "MISSING", "bytes": 0}
    stat = path.stat()
    size = stat.st_size
    base = {"name": name, "path": str(path), "bytes": size, "modified_timestamp": stat.st_mtime}
    if size <= 0:
        return {**base, "status": "EMPTY_PLACEHOLDER"}
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {**base, "status": "FAILED_VALIDATION", "reason": str(exc)}
        if not payload:
            return {**base, "status": "EMPTY_VALID"}
    if path.suffix == ".csv":
        rows = _read_csv_if_available(path)
        if not rows:
            return {**base, "status": "EMPTY_VALID"}
        return {**base, "status": "COMPLETE", "row_count": len(rows)}
    return {**base, "status": "COMPLETE"}


def _read_json_if_available(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_if_available(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        return _read_csv(path)
    except (OSError, csv.Error):
        return []


def _strategy_summary_rows(risk: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in sorted(dict(risk).items()):
        if not isinstance(metrics, Mapping) or not metrics:
            continue
        rows.append(
            {
                "strategy": name,
                "ending_wealth": _metric(metrics, "ending_equity", "wealth_multiple"),
                "total_return": _metric(metrics, "total_return_percent"),
                "cagr": _metric(metrics, "cagr", "CAGR"),
                "maximum_drawdown": _metric(metrics, "maximum_drawdown"),
                "sharpe_ratio": _metric(metrics, "sharpe_ratio", "Sharpe_ratio"),
                "calmar_ratio": _metric(metrics, "calmar_ratio", "Calmar_ratio"),
                "cvar": _metric(metrics, "cvar_5pct", "CVaR_5pct", "expected_shortfall_CVaR_5pct"),
                "trade_count": int(_metric(metrics, "number_of_trades") or 0),
                "total_costs": _metric(metrics, "total_costs"),
            }
        )
    return rows


def _cost_robustness_rows(cost: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = dict(cost.get("scenarios", {}) or {})
    rows = []
    for key, payload in sorted(scenarios.items(), key=lambda item: _number(str(item[1].get("round_trip_bps", 0))) or 0.0):
        variants = dict(payload.get("variants", {}) or {})
        price = dict(variants.get("price_only", {}) or {})
        contrarian = dict(variants.get("news_contrarian_rerank", variants.get("news_inverted_gate", {})) or {})
        price_wealth = _metric(price, "ending_equity", "wealth_multiple")
        contrarian_wealth = _metric(contrarian, "ending_equity", "wealth_multiple")
        rows.append(
            {
                "round_trip_bps": _metric(payload, "round_trip_bps"),
                "price_only_ending_wealth": price_wealth,
                "contrarian_ending_wealth": contrarian_wealth,
                "difference": (
                    contrarian_wealth - price_wealth
                    if contrarian_wealth is not None and price_wealth is not None
                    else None
                ),
                "price_only_maximum_drawdown": _metric(price, "maximum_drawdown"),
                "contrarian_maximum_drawdown": _metric(contrarian, "maximum_drawdown"),
                "scenario": key,
            }
        )
    return rows


def _diagnostics_summary(
    *,
    coverage: Mapping[str, Any],
    direction: Mapping[str, Any],
    event: Mapping[str, Any],
    extreme: Mapping[str, Any],
    risk: Mapping[str, Any],
    cost_rows: list[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    holdout: Mapping[str, Any],
    grid_selection: Mapping[str, Any],
    walk_forward: Mapping[str, Any],
    placebo: Mapping[str, Any],
    matched: Mapping[str, Any],
    concentration: Mapping[str, Any],
    universe: Mapping[str, Any],
    corporate_actions: Mapping[str, Any],
) -> dict[str, Any]:
    event_counts = _event_counts(event)
    categorized = sum(count for category, count in event_counts.items() if category != "general_negative_sentiment_or_uncategorized")
    uncategorized = event_counts.get("general_negative_sentiment_or_uncategorized", 0)
    total_events = categorized + uncategorized
    price = dict(risk.get("price_only", {}) or {})
    contrarian = dict(risk.get("news_contrarian_rerank", {}) or {})
    price_wealth = _metric(price, "ending_equity", "wealth_multiple")
    contrarian_wealth = _metric(contrarian, "ending_equity", "wealth_multiple")
    return {
        "news_coverage_percentage": float(coverage.get("row_coverage_ratio", 0.0)) * 100.0,
        "categorized_event_percentage": categorized / max(total_events, 1) * 100.0,
        "uncategorized_event_percentage": uncategorized / max(total_events, 1) * 100.0,
        "extreme_event_count": int(extreme.get("point_in_time_archive_rows", 0) or 0),
        "score_direction_conclusion": _score_direction_conclusion(direction),
        "contrarian_reranking_beat_price_only": (
            contrarian_wealth > price_wealth
            if contrarian_wealth is not None and price_wealth is not None
            else None
        ),
        "superior_after_5_10_20_bps": _superior_after_costs(cost_rows, (5.0, 10.0, 20.0)),
        "untouched_holdout_used": False,
        "untouched_holdout_status": holdout.get("holdout_status", "unavailable_in_current_artifacts"),
        "validation_label": holdout.get("validation_label", "UNVALIDATED"),
        "selected_configuration_id": grid_selection.get("selected_configuration_id"),
        "grid_validation_status": grid_selection.get("validation_status"),
        "holdout_excess_return": holdout.get("excess_return_over_price_only"),
        "walk_forward_positive_fold_proportion": walk_forward.get("positive_excess_return_fold_proportion"),
        "placebo_empirical_p_value": placebo.get("empirical_p_value"),
        "matched_control_advantage": matched.get("advantage_after_matching"),
        "concentration_warning": concentration.get("concentration_warning"),
        "survivorship_status": universe.get("survivorship_bias_risk"),
        "corporate_action_status": corporate_actions.get("validation_status"),
        "paper_orders_enabled": bool(comparison.get("paper_orders_enabled", False)),
        "live_orders_enabled": bool(comparison.get("live_orders_enabled", False)),
    }


def _event_counts(event: Mapping[str, Any]) -> dict[str, int]:
    counts = {}
    for key, payload in event.items():
        if key in {"policy_defaults", "category_source"} or not isinstance(payload, Mapping):
            continue
        counts[key] = int(payload.get("count", 0) or 0)
    return counts


def _score_direction_conclusion(direction: Mapping[str, Any]) -> str:
    answers = dict(direction.get("answers", {}) or {})
    if answers.get("relationship_supports_inversion"):
        return "supports diagnostic inversion"
    if answers.get("higher_score_predicts_lower_return") or answers.get("higher_score_predicts_deeper_temporary_drawdown"):
        return "higher score aligns with downside risk"
    if answers.get("higher_score_predicts_greater_movement_both_directions"):
        return "higher score appears volatility-like"
    return "inconclusive"


def _superior_after_costs(rows: list[Mapping[str, Any]], bps_values: Iterable[float]) -> dict[str, bool | None]:
    by_bps = {float(row.get("round_trip_bps", 0.0) or 0.0): row for row in rows}
    output = {}
    for value in bps_values:
        row = by_bps.get(float(value))
        output[f"{value:g}_bps"] = (row.get("difference") > 0) if row and row.get("difference") is not None else None
    return output


def _executive_warnings(
    *,
    artifact_status: list[Mapping[str, Any]],
    deciles: list[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    cost_rows: list[Mapping[str, Any]],
    replay_audit: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> list[str]:
    warnings = []
    mtimes = [
        float(row["modified_timestamp"])
        for row in artifact_status
        if row.get("status") == "COMPLETE" and row.get("modified_timestamp") is not None
    ]
    if mtimes and max(mtimes) - min(mtimes) > 24 * 60 * 60:
        warnings.append("stale artifacts: output modification times span more than one day")
    if _decile_values_repeated(deciles, "average_forward_return"):
        warnings.append("repeated or identical metrics across score deciles")
    if _decile_values_repeated(deciles, "executed_trade_count"):
        warnings.append("identical executed-trade counts across multiple deciles")
    if float(diagnostics.get("uncategorized_event_percentage", 0.0) or 0.0) >= 80.0:
        warnings.append("mostly uncategorized events")
    if int(diagnostics.get("extreme_event_count", 0) or 0) < 30:
        warnings.append("insufficient extreme-event sample size")
    if "not explicit" in str(replay_audit.get("adjusted_status", "")).lower():
        warnings.append("missing corporate-action adjustment information")
    zero_cost = next((row for row in cost_rows if float(row.get("round_trip_bps", 0.0) or 0.0) == 0.0), None)
    positive_zero = zero_cost and (zero_cost.get("contrarian_ending_wealth") or 0.0) > (zero_cost.get("price_only_ending_wealth") or 0.0)
    positive_realistic = any(
        (row.get("difference") or 0.0) > 0 and float(row.get("round_trip_bps", 0.0) or 0.0) > 0.0
        for row in cost_rows
    )
    if positive_zero and not positive_realistic:
        warnings.append("zero-cost-only profitability")
    bad_artifacts = [row for row in artifact_status if row.get("status") not in {"COMPLETE", "EMPTY_VALID"}]
    if bad_artifacts:
        warnings.append("failed or missing output validation artifacts are present")
    if any(row.get("status") == "EMPTY_PLACEHOLDER" for row in artifact_status):
        warnings.append("empty placeholder report files are present")
    if diagnostics.get("untouched_holdout_used") is False:
        warnings.append("in-sample or post-hypothesis evaluation: untouched holdout unavailable")
    if diagnostics.get("validation_label") in {"UNVALIDATED", "PSEUDO_HOLDOUT", "DEVELOPMENT_ONLY"}:
        warnings.append(f"contrarian validation remains {diagnostics.get('validation_label')}")
    if diagnostics.get("corporate_action_status") == "BLOCKED":
        warnings.append("missing corporate-action adjustment information")
    if diagnostics.get("concentration_warning"):
        warnings.append("fragility warning: result concentration is high")
    if not risk:
        warnings.append("risk metrics unavailable")
    return warnings


def _decile_values_repeated(rows: list[Mapping[str, Any]], field: str) -> bool:
    values = [str(row.get(field, "")) for row in rows if str(row.get("candidate_count", "0")) not in {"0", ""}]
    return len(values) >= 3 and len(set(values)) == 1


def _winner_summary(
    strategy_rows: list[Mapping[str, Any]],
    cost_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "best_absolute_return": _best_strategy(strategy_rows, "ending_wealth", higher=True),
        "best_risk_adjusted_result": _best_strategy(strategy_rows, "sharpe_ratio", higher=True),
        "lowest_maximum_drawdown": _best_strategy(strategy_rows, "maximum_drawdown", higher=True),
        "best_result_after_realistic_costs": _best_after_costs(cost_rows),
    }


def _best_strategy(rows: list[Mapping[str, Any]], field: str, *, higher: bool) -> str | None:
    usable = [row for row in rows if row.get(field) is not None]
    if not usable:
        return None
    return str(max(usable, key=lambda row: row[field])["strategy"] if higher else min(usable, key=lambda row: row[field])["strategy"])


def _best_after_costs(rows: list[Mapping[str, Any]]) -> str:
    realistic = [row for row in rows if float(row.get("round_trip_bps", 0.0) or 0.0) in {5.0, 10.0, 20.0}]
    if not realistic:
        return "unavailable"
    wins = sum((row.get("difference") or 0.0) > 0 for row in realistic)
    return "contrarian" if wins == len(realistic) else "price_only_or_mixed"


def _summary_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "STOCK-ALPHA NEWS RISK OVERLAY SUMMARY",
        f"Output: {summary.get('output_dir', '')}",
        "",
        "Strategy comparison:",
        _table(
            ["strategy", "wealth", "return", "cagr", "max_dd", "sharpe", "calmar", "cvar", "trades", "costs"],
            [
                [
                    row["strategy"],
                    _fmt(row.get("ending_wealth")),
                    _fmt_pct(row.get("total_return")),
                    _fmt_pct_decimal(row.get("cagr")),
                    _fmt_pct_decimal(row.get("maximum_drawdown")),
                    _fmt(row.get("sharpe_ratio")),
                    _fmt(row.get("calmar_ratio")),
                    _fmt_pct_decimal(row.get("cvar")),
                    str(row.get("trade_count", 0)),
                    _fmt(row.get("total_costs")),
                ]
                for row in summary.get("strategy_comparison", [])
            ],
        ),
        "",
        "Cost robustness:",
        _table(
            ["bps", "price_w", "contra_w", "diff", "price_dd", "contra_dd"],
            [
                [
                    _fmt(row.get("round_trip_bps"), digits=0),
                    _fmt(row.get("price_only_ending_wealth")),
                    _fmt(row.get("contrarian_ending_wealth")),
                    _fmt(row.get("difference")),
                    _fmt_pct_decimal(row.get("price_only_maximum_drawdown")),
                    _fmt_pct_decimal(row.get("contrarian_maximum_drawdown")),
                ]
                for row in summary.get("cost_robustness", [])
            ],
        ),
    ]
    diagnostics = dict(summary.get("diagnostics", {}) or {})
    winners = dict(summary.get("winners", {}) or {})
    catastrophic_veto = dict(summary.get("catastrophic_veto", {}) or {})
    news_evidence = dict(summary.get("news_evidence", {}) or {})
    event_taxonomy = dict(summary.get("event_taxonomy", {}) or {})
    duplicate_grouping = dict(summary.get("duplicate_grouping", {}) or {})
    text_safety = dict(summary.get("text_safety", {}) or {})
    keyword_baseline = dict(summary.get("keyword_baseline", {}) or {})
    bounceback = dict(summary.get("catastrophic_veto_bounceback", {}) or {})
    extreme_policy = dict(summary.get("extreme_only_policy_proposal", {}) or {})
    policy_frontier = dict(summary.get("catastrophic_policy_frontier", {}) or {})
    casebook = dict(summary.get("loser_bounceback_casebook", {}) or {})
    taxonomy_plan = dict(summary.get("taxonomy_improvement_plan", {}) or {})
    contrarian_validation = dict(summary.get("contrarian_validation", {}) or {})
    parked_veto = dict(summary.get("catastrophic_veto_parked", {}) or {})
    lines.extend(
        [
            "",
            "Diagnostics:",
            f"- news coverage: {_fmt_pct(diagnostics.get('news_coverage_percentage'))}",
            f"- categorized/uncategorized events: {_fmt_pct(diagnostics.get('categorized_event_percentage'))} / {_fmt_pct(diagnostics.get('uncategorized_event_percentage'))}",
            f"- extreme events: {diagnostics.get('extreme_event_count', 0)}",
            f"- score direction: {diagnostics.get('score_direction_conclusion', 'unavailable')}",
            f"- contrarian beat price-only: {diagnostics.get('contrarian_reranking_beat_price_only')}",
            f"- superior after 5/10/20 bps: {diagnostics.get('superior_after_5_10_20_bps')}",
            f"- holdout/status: {diagnostics.get('untouched_holdout_used')} / {diagnostics.get('untouched_holdout_status')}",
            f"- validation label: {diagnostics.get('validation_label') or 'PSEUDO_HOLDOUT'}",
            f"- walk-forward positive folds: {_fmt_pct_decimal(diagnostics.get('walk_forward_positive_fold_proportion'))}",
            f"- placebo p-value: {_fmt(diagnostics.get('placebo_empirical_p_value'))}",
            f"- paper/live trading enabled: {diagnostics.get('paper_orders_enabled')} / {diagnostics.get('live_orders_enabled')}",
            f"- contrarian validation: {contrarian_validation.get('status', 'IN_PROGRESS')} | year/regime: {contrarian_validation.get('year_regime', 'UNAVAILABLE_INPUT')} | data validity: {contrarian_validation.get('data_validity', 'UNAVAILABLE_INPUT')} | intraday 5min: {contrarian_validation.get('intraday_5min', 'UNAVAILABLE_INPUT')} | catastrophic veto: {parked_veto.get('status', 'UNAVAILABLE_INPUT')}",
            f"- news evidence readiness: {news_evidence.get('status', 'UNAVAILABLE_INPUT')} | news evidence text coverage: {news_evidence.get('has_any_text_count', 0)} / {news_evidence.get('candidate_count', 0)} | news evidence availability timestamps: {news_evidence.get('has_availability_timestamp_count', 0)} / {news_evidence.get('candidate_count', 0)}",
            f"- event taxonomy / duplicate grouping / text safety / keyword baseline: {event_taxonomy.get('status', news_evidence.get('event_taxonomy_status', 'UNAVAILABLE_INPUT'))} / {duplicate_grouping.get('status', news_evidence.get('duplicate_grouping_status', 'UNAVAILABLE_INPUT'))} / {text_safety.get('status', news_evidence.get('point_in_time_text_safety_status', 'UNAVAILABLE_INPUT'))} / {keyword_baseline.get('status', news_evidence.get('keyword_baseline_status', 'UNAVAILABLE_INPUT'))} | veto bounceback/extreme-only: {bounceback.get('status', 'UNAVAILABLE_INPUT')} / {extreme_policy.get('status', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic policy frontier: {policy_frontier.get('best_balanced_policy', 'UNAVAILABLE_INPUT')} | strict veto: {dict(bounceback.get('veto_breadth_diagnostic', {}) or {}).get('strict_veto_breadth_status', 'UNAVAILABLE_INPUT')} | loser/bounceback casebook: {casebook.get('status', 'UNAVAILABLE_INPUT')} | taxonomy improvement plan: {taxonomy_plan.get('status', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic news veto: RESEARCH_ONLY / NOT_CURRENT_STRATEGY ({catastrophic_veto.get('replay_impact_status', 'NOT_COMPUTED')}) | catastrophic evidence quality: {catastrophic_veto.get('evidence_quality_status', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic veto scenario: {catastrophic_veto.get('replay_impact_status', 'NOT_COMPUTED')} | catastrophic veto approximate simulation: {catastrophic_veto.get('approximate_replay_impact_status', 'NOT_COMPUTED')} | policy modes: {catastrophic_veto.get('policy_modes')}",
            f"- catastrophic veto candidates before/after: {catastrophic_veto.get('candidate_count_before_veto', 'UNAVAILABLE_INPUT')}/{catastrophic_veto.get('candidate_count_after_veto', 'UNAVAILABLE_INPUT')} | catastrophic usable strict-veto candidates: {catastrophic_veto.get('usable_for_strict_veto_count', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic blocked candidates: {catastrophic_veto.get('blocked_candidate_count', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic veto blocked trades: {catastrophic_veto.get('blocked_trade_count', 'UNAVAILABLE_INPUT')}",
            f"- catastrophic veto delta return: {catastrophic_veto.get('delta_return', 'UNAVAILABLE_INPUT')}",
            f"- manual review candidates: {catastrophic_veto.get('manual_review_candidate_count', 'UNAVAILABLE_INPUT')}",
            "",
            "Winners:",
            f"- best absolute return: {winners.get('best_absolute_return')}",
            f"- best risk-adjusted: {winners.get('best_risk_adjusted_result')}",
            f"- lowest max drawdown: {winners.get('lowest_maximum_drawdown')}",
            f"- best after realistic costs: {winners.get('best_result_after_realistic_costs')}",
        ]
    )
    warnings = list(summary.get("warnings", []) or [])
    if warnings:
        lines.extend(["", "WARNINGS:"])
        lines.extend(f"- {warning}" for warning in warnings[:3])
    return lines


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(unavailable)"
    widths = [
        max(len(str(value)) for value in [header, *[row[index] for row in rows]])
        for index, header in enumerate(headers)
    ]
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _metric(payload: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.1f}%"


def _fmt_pct_decimal(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number * 100.0:.1f}%"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
