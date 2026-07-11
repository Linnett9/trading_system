from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.research.ml.stock_level.news_transformer import (
    build_news_transformer_readiness_report,
    build_news_transformer_training_plan,
)
from core.research.ml.stock_level.news_risk_overlay_research_audit import (
    _contribution_reports,
    _corporate_action_audit,
    _holdout_markdown,
    _holdout_rows,
    _matched_controls,
    _missing_news_bias,
    _parameter_stability,
    _placebo_reports,
    _text_model_readiness,
    _universe_membership,
    _universe_survivorship_audit,
    _walk_forward_reports,
)
from core.research.ml.stock_level.news_risk_overlay_research_catastrophic import (
    _catastrophic_news_artifacts,
    _catastrophic_news_evidence_quality_artifacts,
    _catastrophic_policy_variant_artifacts,
    _catastrophic_veto_bounceback_artifacts,
    _catastrophic_veto_loser_bounceback_casebook_artifacts,
    _catastrophic_veto_replay_seam_report,
    _catastrophic_veto_strategy_artifacts,
    _news_duplicate_grouping_artifacts,
    _news_event_taxonomy_artifacts,
    _news_point_in_time_text_safety_artifacts,
    _news_text_keyword_baseline_artifacts,
)
from core.research.ml.stock_level.news_risk_overlay_research_workflow import (
    _validation_stage_placeholders,
)
from core.research.ml.stock_level.news_risk_overlay_research_evidence import (
    _catastrophic_veto_parked_status,
    _chronological_periods,
    _concentration_fragility_artifact,
    _matched_control_artifact,
    _not_ready_text_model_report,
    _placebo_permutation_artifacts,
    _walk_forward_validation_artifacts,
)
from core.research.ml.stock_level.news_risk_overlay_research_robustness import (
    _contrarian_chronological_validation_plan,
    _contrarian_cost_slippage_robustness,
    _contrarian_data_validity_audit,
    _contrarian_matched_control_report,
    _contrarian_placebo_permutation_report,
    _contrarian_profit_concentration_artifacts,
    _contrarian_symbol_year_ablation_artifacts,
    _contrarian_year_regime_artifacts,
    _intraday_5min_expansion_plan,
)
from core.research.ml.stock_level.news_risk_overlay_research_selection import (
    _contrarian_grid_reports,
    _frozen_contrarian_config,
    _holdout_report,
    _selection_cost_metrics_from_scenarios,
)

def _contrarian_validation_stage_reports(
    *,
    rows: list[Mapping[str, Any]],
    replay: Mapping[str, Any],
    price_score_column: str,
    config: Mapping[str, Any],
    cost_scenarios: Mapping[str, Any],
    data_audit: Mapping[str, Any],
) -> dict[str, Any]:
    periods = _chronological_periods(rows, replay.get("trade_ledger", []))
    selection_config = {
        **dict(config),
        "stock_alpha_news_risk_overlay_contrarian_grid_cost_metrics": _selection_cost_metrics_from_scenarios(
            config,
            cost_scenarios,
        ),
    }
    grid_rows, fold_rows, selection = _contrarian_grid_reports(rows, replay, price_score_column, periods, selection_config)
    frozen = _frozen_contrarian_config(selection, config, periods)
    holdout = _holdout_report(replay, periods, frozen, cost_scenarios)
    walk_forward_folds, walk_forward_summary = _walk_forward_reports(replay, periods)
    placebo_rows, placebo_summary = _placebo_reports(replay, config)
    contribution_year, contribution_symbol, concentration = _contribution_reports(replay)
    missing_news_report, covered_vs_uncovered = _missing_news_bias(rows, price_score_column)
    walk_forward_validation_report, walk_forward_fold_rows = _walk_forward_validation_artifacts(
        replay,
        periods,
        selection,
        frozen,
        config,
    )
    placebo_permutation_report, placebo_permutation_rows = _placebo_permutation_artifacts(
        rows,
        replay,
        config,
    )
    catastrophic_news_audit, catastrophic_news_candidates, catastrophic_news_veto_report = (
        _catastrophic_news_artifacts(rows)
    )
    (
        catastrophic_veto_candidate_attribution,
        catastrophic_veto_trade_attribution,
        catastrophic_veto_strategy_comparison,
        catastrophic_veto_policy,
        catastrophic_veto_filtered_strategy_report,
        catastrophic_veto_removed_trades,
        catastrophic_veto_removed_symbols,
        catastrophic_veto_full_replay_report,
        catastrophic_veto_full_replay_trade_ledger,
        catastrophic_veto_full_replay_equity,
        catastrophic_veto_filtered_candidates,
        catastrophic_veto_blocked_candidates,
    ) = _catastrophic_veto_strategy_artifacts(rows, replay)
    (
        catastrophic_news_evidence_quality_report,
        catastrophic_news_evidence_quality_by_field,
        catastrophic_news_evidence_quality_by_symbol,
        catastrophic_veto_policy_mode_comparison,
        catastrophic_veto_policy_mode_counts,
    ) = _catastrophic_news_evidence_quality_artifacts(
        rows,
        replay,
        catastrophic_veto_full_replay_report,
    )
    (
        news_event_taxonomy_report,
        news_event_taxonomy_counts,
        news_event_taxonomy_examples,
    ) = _news_event_taxonomy_artifacts(rows)
    (
        news_duplicate_grouping_report,
        news_duplicate_grouping_examples,
    ) = _news_duplicate_grouping_artifacts(rows)
    (
        news_point_in_time_text_safety_report,
        news_point_in_time_text_safety_examples,
    ) = _news_point_in_time_text_safety_artifacts(rows)
    (
        news_text_keyword_baseline_report,
        news_text_keyword_baseline_scores,
    ) = _news_text_keyword_baseline_artifacts(rows)
    (
        catastrophic_veto_bounceback_report,
        catastrophic_veto_bounceback_by_category,
        catastrophic_veto_bounceback_examples,
        catastrophic_veto_extreme_only_policy_proposal,
    ) = _catastrophic_veto_bounceback_artifacts(
        rows,
        replay,
        catastrophic_veto_removed_trades,
        catastrophic_veto_blocked_candidates,
        catastrophic_veto_full_replay_report,
        catastrophic_veto_policy_mode_counts,
    )
    (
        catastrophic_veto_policy_variant_comparison,
        catastrophic_veto_policy_variant_counts,
        catastrophic_veto_policy_variant_metrics,
        catastrophic_veto_policy_variant_removed_trades,
        catastrophic_veto_policy_variant_bounceback,
        catastrophic_veto_policy_frontier_report,
        catastrophic_veto_policy_frontier,
        catastrophic_veto_policy_variant_examples,
    ) = _catastrophic_policy_variant_artifacts(
        rows,
        replay,
        catastrophic_veto_bounceback_report,
    )
    (
        catastrophic_veto_loser_bounceback_casebook,
        catastrophic_veto_loser_bounceback_cases,
        catastrophic_veto_loser_bounceback_feature_diff,
        catastrophic_veto_loser_bounceback_keyword_diff,
        catastrophic_veto_taxonomy_improvement_plan,
    ) = _catastrophic_veto_loser_bounceback_casebook_artifacts(
        rows,
        catastrophic_veto_removed_trades,
    )
    contrarian_chronological_validation_plan, contrarian_chronological_periods = _contrarian_chronological_validation_plan(
        rows,
        replay,
        periods,
    )
    contrarian_placebo_permutation_report, contrarian_placebo_permutation_results = _contrarian_placebo_permutation_report(config)
    contrarian_matched_control_report, contrarian_matched_control_results = _contrarian_matched_control_report()
    (
        contrarian_profit_concentration_report,
        contrarian_trade_fragility_by_symbol,
        contrarian_trade_fragility_by_year,
        contrarian_top_trade_removal,
    ) = _contrarian_profit_concentration_artifacts(replay)
    (
        contrarian_year_regime_report,
        contrarian_year_regime_results,
        contrarian_year_regime_examples,
    ) = _contrarian_year_regime_artifacts(replay)
    (
        contrarian_symbol_year_ablation_report,
        contrarian_without_top_symbols,
        contrarian_without_top_years,
    ) = _contrarian_symbol_year_ablation_artifacts(replay)
    (
        contrarian_cost_slippage_robustness_report,
        contrarian_cost_slippage_robustness,
    ) = _contrarian_cost_slippage_robustness(cost_scenarios)
    contrarian_data_validity_audit = _contrarian_data_validity_audit(
        data_audit,
        missing_news_report,
    )
    intraday_5min_expansion_plan = _intraday_5min_expansion_plan(config)
    catastrophic_veto_parked_status = _catastrophic_veto_parked_status(
        catastrophic_veto_full_replay_report,
        catastrophic_veto_policy_frontier_report,
        catastrophic_veto_loser_bounceback_casebook,
    )
    return {
        "chronological_split_manifest": periods,
        "contrarian_grid_results": grid_rows,
        "contrarian_grid_selection": selection,
        "contrarian_fold_results": fold_rows,
        "contrarian_parameter_stability": _parameter_stability(grid_rows, selection),
        "contrarian_frozen_config": frozen,
        "contrarian_holdout_report": holdout,
        "contrarian_holdout_trade_ledger": _holdout_rows(replay.get("trade_ledger", []), periods, "news_contrarian_rerank"),
        "contrarian_holdout_equity": _holdout_rows(replay.get("daily_equity", {}).get("news_contrarian_rerank", []), periods, "news_contrarian_rerank"),
        "contrarian_holdout_comparison_md": _holdout_markdown(holdout),
        "contrarian_walk_forward_folds": walk_forward_folds,
        "contrarian_walk_forward_summary": walk_forward_summary,
        "contrarian_chronological_validation_plan": contrarian_chronological_validation_plan,
        "contrarian_chronological_periods": contrarian_chronological_periods,
        "contrarian_walk_forward_validation_report": walk_forward_validation_report,
        "contrarian_placebo_permutation_report": contrarian_placebo_permutation_report,
        "contrarian_placebo_permutation_results": contrarian_placebo_permutation_results,
        "contrarian_matched_control_report": contrarian_matched_control_report,
        "contrarian_matched_control_results": contrarian_matched_control_results,
        "contrarian_profit_concentration_report": contrarian_profit_concentration_report,
        "contrarian_trade_fragility_by_symbol": contrarian_trade_fragility_by_symbol,
        "contrarian_trade_fragility_by_year": contrarian_trade_fragility_by_year,
        "contrarian_top_trade_removal": contrarian_top_trade_removal,
        "contrarian_year_regime_report": contrarian_year_regime_report,
        "contrarian_year_regime_results": contrarian_year_regime_results,
        "contrarian_year_regime_examples": contrarian_year_regime_examples,
        "contrarian_symbol_year_ablation_report": contrarian_symbol_year_ablation_report,
        "contrarian_without_top_symbols": contrarian_without_top_symbols,
        "contrarian_without_top_years": contrarian_without_top_years,
        "contrarian_cost_slippage_robustness_report": contrarian_cost_slippage_robustness_report,
        "contrarian_cost_slippage_robustness": contrarian_cost_slippage_robustness,
        "contrarian_data_validity_audit": contrarian_data_validity_audit,
        "intraday_5min_expansion_plan": intraday_5min_expansion_plan,
        "contrarian_placebo_results": placebo_rows,
        "contrarian_placebo_summary": placebo_summary,
        "contrarian_matched_controls": _matched_controls(replay),
        "contrarian_contribution_by_year": contribution_year,
        "contrarian_contribution_by_symbol": contribution_symbol,
        "contrarian_concentration_report": concentration,
        "universe_survivorship_audit": _universe_survivorship_audit(rows, config),
        "universe_membership_by_date": _universe_membership(rows),
        "corporate_action_audit": _corporate_action_audit(data_audit),
        "missing_news_bias_report": missing_news_report,
        "covered_vs_uncovered_candidates": covered_vs_uncovered,
        "text_model_readiness": _not_ready_text_model_report(_text_model_readiness(rows)),
        "news_transformer_readiness": build_news_transformer_readiness_report(rows),
        "news_transformer_training_plan": build_news_transformer_training_plan(),
        "catastrophic_news_audit": catastrophic_news_audit,
        "catastrophic_news_candidates": catastrophic_news_candidates,
        "catastrophic_news_veto_report": catastrophic_news_veto_report,
        "catastrophic_veto_candidate_attribution": catastrophic_veto_candidate_attribution,
        "catastrophic_veto_trade_attribution": catastrophic_veto_trade_attribution,
        "catastrophic_veto_strategy_comparison": catastrophic_veto_strategy_comparison,
        "catastrophic_veto_policy": catastrophic_veto_policy,
        "catastrophic_veto_filtered_strategy_report": catastrophic_veto_filtered_strategy_report,
        "catastrophic_veto_removed_trades": catastrophic_veto_removed_trades,
        "catastrophic_veto_removed_symbols": catastrophic_veto_removed_symbols,
        "catastrophic_veto_full_replay_report": catastrophic_veto_full_replay_report,
        "catastrophic_veto_full_replay_trade_ledger": catastrophic_veto_full_replay_trade_ledger,
        "catastrophic_veto_full_replay_equity": catastrophic_veto_full_replay_equity,
        "catastrophic_veto_filtered_candidates": catastrophic_veto_filtered_candidates,
        "catastrophic_veto_blocked_candidates": catastrophic_veto_blocked_candidates,
        "catastrophic_veto_replay_seam_report": _catastrophic_veto_replay_seam_report(
            full_replay_computed=bool(catastrophic_veto_full_replay_report.get("full_replay_computed")),
        ),
        "catastrophic_news_evidence_quality_report": catastrophic_news_evidence_quality_report,
        "catastrophic_news_evidence_quality_by_field": catastrophic_news_evidence_quality_by_field,
        "catastrophic_news_evidence_quality_by_symbol": catastrophic_news_evidence_quality_by_symbol,
        "catastrophic_veto_policy_mode_comparison": catastrophic_veto_policy_mode_comparison,
        "catastrophic_veto_policy_mode_counts": catastrophic_veto_policy_mode_counts,
        "news_event_taxonomy_report": news_event_taxonomy_report,
        "news_event_taxonomy_counts": news_event_taxonomy_counts,
        "news_event_taxonomy_examples": news_event_taxonomy_examples,
        "news_duplicate_grouping_report": news_duplicate_grouping_report,
        "news_duplicate_grouping_examples": news_duplicate_grouping_examples,
        "news_point_in_time_text_safety_report": news_point_in_time_text_safety_report,
        "news_point_in_time_text_safety_examples": news_point_in_time_text_safety_examples,
        "news_text_keyword_baseline_report": news_text_keyword_baseline_report,
        "news_text_keyword_baseline_scores": news_text_keyword_baseline_scores,
        "catastrophic_veto_bounceback_report": catastrophic_veto_bounceback_report,
        "catastrophic_veto_bounceback_by_category": catastrophic_veto_bounceback_by_category,
        "catastrophic_veto_bounceback_examples": catastrophic_veto_bounceback_examples,
        "catastrophic_veto_extreme_only_policy_proposal": catastrophic_veto_extreme_only_policy_proposal,
        "catastrophic_veto_policy_variant_comparison": catastrophic_veto_policy_variant_comparison,
        "catastrophic_veto_policy_variant_counts": catastrophic_veto_policy_variant_counts,
        "catastrophic_veto_policy_variant_metrics": catastrophic_veto_policy_variant_metrics,
        "catastrophic_veto_policy_variant_removed_trades": catastrophic_veto_policy_variant_removed_trades,
        "catastrophic_veto_policy_variant_bounceback": catastrophic_veto_policy_variant_bounceback,
        "catastrophic_veto_policy_frontier_report": catastrophic_veto_policy_frontier_report,
        "catastrophic_veto_policy_frontier": catastrophic_veto_policy_frontier,
        "catastrophic_veto_policy_variant_examples": catastrophic_veto_policy_variant_examples,
        "catastrophic_veto_loser_bounceback_casebook": catastrophic_veto_loser_bounceback_casebook,
        "catastrophic_veto_loser_bounceback_cases": catastrophic_veto_loser_bounceback_cases,
        "catastrophic_veto_loser_bounceback_feature_diff": catastrophic_veto_loser_bounceback_feature_diff,
        "catastrophic_veto_loser_bounceback_keyword_diff": catastrophic_veto_loser_bounceback_keyword_diff,
        "catastrophic_veto_taxonomy_improvement_plan": catastrophic_veto_taxonomy_improvement_plan,
        "catastrophic_veto_parked_status": catastrophic_veto_parked_status,
        "validation_stage_placeholders": _validation_stage_placeholders(
            full_replay_computed=bool(catastrophic_veto_full_replay_report.get("full_replay_computed")),
        ),
        "walk_forward_validation_report": walk_forward_validation_report,
        "walk_forward_fold_results": walk_forward_fold_rows,
        "placebo_permutation_report": placebo_permutation_report,
        "placebo_permutation_results": placebo_permutation_rows,
        "exposure_matched_controls": _matched_control_artifact("exposure_matched_controls"),
        "trade_count_matched_controls": _matched_control_artifact("trade_count_matched_controls"),
        "concentration_fragility_report": _concentration_fragility_artifact(replay),
    }


__all__ = [name for name in globals() if not name.startswith("__")]
