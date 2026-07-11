from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.research.ml.stock_level.news_risk_overlay_research_inspection import (
    _read_json_if_available,
)
from core.research.ml.stock_level.news_risk_overlay_research_paths import NewsRiskResearchPaths


def _news_validation_gap_analysis(paths: NewsRiskResearchPaths) -> dict[str, Any]:
    full_replay_computed = bool(
        _read_json_if_available(paths.catastrophic_veto_full_replay_report_json_path).get("full_replay_computed")
    )
    event_taxonomy_ready = bool(
        _read_json_if_available(paths.news_event_taxonomy_report_json_path).get("event_taxonomy_research_ready")
    )
    duplicate_heuristic_ready = bool(
        _read_json_if_available(paths.news_duplicate_grouping_report_json_path).get("duplicate_grouping_heuristic_ready")
    )
    text_safety_ready = bool(
        _read_json_if_available(paths.news_point_in_time_text_safety_report_json_path).get("point_in_time_text_safety_ready")
    )
    keyword_baseline_ready = bool(
        _read_json_if_available(paths.news_text_keyword_baseline_report_json_path).get("keyword_baseline_ready")
    )
    bounceback_available = bool(
        _read_json_if_available(paths.catastrophic_veto_bounceback_report_json_path).get("status") == "AVAILABLE"
    )
    gaps = [
        {
            "gap_id": "contrarian_validation_in_progress",
            "area": "validation_spine",
            "severity": "critical",
            "description": "Main news_contrarian_rerank validation is in progress; chronological, walk-forward, placebo, matched-control, concentration, and data-validity gates are not final.",
            "blocks_final_validation": True,
            "recommended_action": "complete the main contrarian validation spine before revisiting execution readiness",
        },
        {
            "gap_id": "catastrophic_veto_parked_diagnostic_only",
            "area": "catastrophic_veto_policy",
            "severity": "low",
            "description": "Catastrophic-veto work is parked as diagnostic-only and is not used by the current strategy.",
            "blocks_final_validation": False,
            "recommended_action": "keep catastrophic-veto artifacts observational while validating news_contrarian_rerank",
        },
        {
            "gap_id": "year_regime_review_required",
            "area": "regime_robustness",
            "severity": "medium",
            "description": "Year/regime report is ledger-level and must be reviewed for negative or partial years before final validation.",
            "blocks_final_validation": True,
            "recommended_action": "review annual robustness, especially negative-year and partial-year behavior",
        },
        {
            "gap_id": "symbol_year_ablation_review_required",
            "area": "fragility",
            "severity": "medium",
            "description": "Symbol/year ablations are ledger-level approximations and do not recompute portfolio compounding.",
            "blocks_final_validation": True,
            "recommended_action": "review without-top-symbol/year sensitivity and implement full replay ablations if needed",
        },
        {
            "gap_id": "intraday_5min_planning_only",
            "area": "future_data_expansion",
            "severity": "low",
            "description": "Intraday 5-minute expansion is a Dell PC planning artifact only.",
            "blocks_final_validation": False,
            "recommended_action": "confirm local 5min/15min data paths and run a small subset later",
        },
        {
            "gap_id": "walk_forward_not_implemented",
            "area": "robustness",
            "severity": "critical",
            "description": "Walk-forward artifact exists, but fold-level replay metrics are not implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement walk-forward validation",
        },
        {
            "gap_id": "placebo_permutation_not_implemented",
            "area": "statistical_controls",
            "severity": "critical",
            "description": "Placebo/permutation artifact exists, but checks are UNAVAILABLE_INPUT until placebo replay/statistics are implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement placebo/permutation checks",
        },
        {
            "gap_id": "matched_controls_not_implemented",
            "area": "controls",
            "severity": "critical",
            "description": "Exposure- and trade-count-matched controls have not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement matched controls",
        },
        {
            "gap_id": "concentration_analysis_not_implemented",
            "area": "fragility",
            "severity": "high",
            "description": "Contribution and concentration analysis has not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement concentration/fragility analysis",
        },
        {
            "gap_id": "survivorship_audit_not_implemented",
            "area": "data_integrity",
            "severity": "critical",
            "description": "Point-in-time universe and survivorship audit has not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement survivorship audit",
        },
        {
            "gap_id": "corporate_action_audit_not_implemented",
            "area": "data_integrity",
            "severity": "critical",
            "description": "Corporate-action adjustment validation has not been implemented.",
            "blocks_final_validation": True,
            "recommended_action": "implement corporate-action audit",
        },
        {
            "gap_id": "missing_news_bias_not_implemented",
            "area": "coverage_bias",
            "severity": "high",
            "description": "Missing-news bias analysis is not complete.",
            "blocks_final_validation": True,
            "recommended_action": "implement missing-news bias analysis",
        },
        {
            "gap_id": "events_uncategorized",
            "area": "event_taxonomy",
            "severity": "medium" if event_taxonomy_ready else "high",
            "description": "Production event_category remains unavailable; deterministic headline taxonomy is research-only.",
            "blocks_final_validation": True,
            "recommended_action": "review deterministic event taxonomy coverage and add production-grade event_category upstream",
        },
        {
            "gap_id": "duplicate_grouping_not_production_grade",
            "area": "text_models",
            "severity": "medium",
            "description": "Duplicate grouping exists only as a deterministic heuristic.",
            "blocks_final_validation": True,
            "recommended_action": "add provider-grade duplicate_group_id/source lineage before text-model readiness",
        },
        {
            "gap_id": "point_in_time_text_safety_partial",
            "area": "event_evidence",
            "severity": "medium",
            "description": "Point-in-time text safety is partial and depends on availability timestamp coverage.",
            "blocks_final_validation": True,
            "recommended_action": "increase availability timestamp coverage and audit unsafe examples",
        },
        {
            "gap_id": "keyword_baseline_research_only",
            "area": "text_baseline",
            "severity": "low",
            "description": "Keyword baseline is deterministic research-only output and is not used by strategy ranking.",
            "blocks_final_validation": True,
            "recommended_action": "keep keyword baseline observational until validation gates are complete",
        },
        {
            "gap_id": "catastrophic_policy_frontier_research_only",
            "area": "catastrophic_veto_policy",
            "severity": "medium",
            "description": "Catastrophic policy frontier is diagnostic hypothesis triage, not final validation or model selection.",
            "blocks_final_validation": True,
            "recommended_action": "review policy examples and run future validation gates before interpreting any narrowed policy as usable",
        },
        {
            "gap_id": "loser_bounceback_casebook_research_only",
            "area": "catastrophic_veto_policy",
            "severity": "medium",
            "description": "Loser-vs-bounceback casebook is observational and only proposes taxonomy improvements.",
            "blocks_final_validation": True,
            "recommended_action": "review casebook differences before implementing deterministic taxonomy changes",
        },
        {
            "gap_id": "catastrophic_news_veto_not_validated",
            "area": "event_taxonomy",
            "severity": "critical",
            "description": "Catastrophic-news veto audit is research-only and not validated for replay or strategy enforcement.",
            "blocks_final_validation": True,
            "recommended_action": "validate taxonomy coverage, point-in-time availability, and veto impact before final validation",
        },
        {
            "gap_id": "catastrophic_news_evidence_quality_insufficient",
            "area": "event_evidence",
            "severity": "critical",
            "description": "Catastrophic-news evidence quality is insufficient for strict live-style filtering.",
            "blocks_final_validation": True,
            "recommended_action": "improve point-in-time text and availability evidence before any execution use",
        },
        {
            "gap_id": "news_evidence_contract_incomplete",
            "area": "evidence_lineage",
            "severity": "critical",
            "description": "News text, availability, source/category, duplicate, or candidate linkage evidence is incomplete across pipeline stages.",
            "blocks_final_validation": True,
            "recommended_action": "apply the field mapping fixes documented by news_evidence_lineage_report.json",
        },
        {
            "gap_id": "catastrophic_veto_full_replay_not_computed",
            "area": "strategy_validation",
            "severity": "critical",
            "description": "Catastrophic-veto filtered scenario is approximate ledger simulation, not full replay.",
            "blocks_final_validation": True,
            "recommended_action": "compute a full separate research-only filtered replay variant before interpreting validated veto impact",
        },
        {
            "gap_id": "catastrophic_veto_full_replay_not_available",
            "area": "strategy_validation",
            "severity": "critical",
            "description": "The optional research variant is absent from replay metrics, equity, or variant metadata output.",
            "blocks_final_validation": True,
            "recommended_action": "execute the separate candidate-filtered replay variant through the seam without changing replay mechanics or base strategy results",
        },
        {
            "gap_id": "text_model_readiness_not_ready",
            "area": "text_models",
            "severity": "medium",
            "description": "Text-model readiness is NOT_READY; FinBERT/BERT/transformer training is deferred.",
            "blocks_final_validation": True,
            "recommended_action": "complete taxonomy, timestamp, and duplicate-handling checks before text models",
        },
        {
            "gap_id": "pseudo_holdout_not_genuine",
            "area": "holdout",
            "severity": "critical",
            "description": "Current final period is pseudo-holdout, not a demonstrably untouched holdout.",
            "blocks_final_validation": True,
            "recommended_action": "collect or wait for genuinely untouched prospective data",
        },
    ]
    if full_replay_computed:
        gaps = [
            gap
            for gap in gaps
            if gap["gap_id"] not in {
                "catastrophic_veto_full_replay_not_computed",
                "catastrophic_veto_full_replay_not_available",
            }
        ]
    if duplicate_heuristic_ready:
        gaps = [gap for gap in gaps if gap["gap_id"] != "duplicate_grouping_not_production_grade"] + [
            {
                "gap_id": "duplicate_grouping_not_production_grade",
                "area": "text_models",
                "severity": "medium",
                "description": "Heuristic duplicate grouping is available, but production duplicate_group_id remains unavailable.",
                "blocks_final_validation": True,
                "recommended_action": "replace heuristic grouping with provider-grade duplicate/source identifiers before text models",
            }
        ]
    if text_safety_ready:
        gaps = [gap for gap in gaps if gap["gap_id"] != "point_in_time_text_safety_partial"] + [
            {
                "gap_id": "point_in_time_text_safety_partial",
                "area": "event_evidence",
                "severity": "medium",
                "description": "Point-in-time text safety audit is present, but coverage is still partial.",
                "blocks_final_validation": True,
                "recommended_action": "expand availability timestamp coverage before text-model readiness",
            }
        ]
    if keyword_baseline_ready:
        gaps = [gap for gap in gaps if gap["gap_id"] != "keyword_baseline_research_only"] + [
            {
                "gap_id": "keyword_baseline_research_only",
                "area": "text_baseline",
                "severity": "low",
                "description": "Keyword baseline is available as a deterministic research-only scaffold.",
                "blocks_final_validation": True,
                "recommended_action": "do not feed keyword scores into strategy ranking until validation gates pass",
            }
        ]
    critical_gaps = [gap["gap_id"] for gap in gaps if gap["severity"] == "critical"]
    return {
        "schema_name": "stock_alpha_news_validation_gap_analysis",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "stock-alpha-news-contrarian-validation-v1",
        "status": "OPEN_GAPS_BLOCK_FINAL_VALIDATION",
        "gaps": gaps,
        "critical_gaps": critical_gaps,
        "next_recommended_implementation_order": [
            "complete main news_contrarian_rerank chronological validation spine",
            "implement walk-forward validation",
            "implement placebo/permutation checks",
            "implement matched controls",
            "implement concentration/fragility analysis",
            "implement survivorship/corporate-action/missing-news audits",
            "build structured event taxonomy",
        ],
        "finbert_blockers": [
            "validation spine not complete",
            "events still uncategorized",
            "text timestamps not proven point-in-time",
            "duplicate/syndication handling not proven",
            "FinBERT deferred",
        ],
        "paper_live_blockers": [
            "final validation status is NOT_FINAL_VALIDATION",
            "validation_passed is false",
            "pseudo-holdout is not a genuine holdout",
            "walk-forward/placebo/matched-control gates are incomplete",
        ],
        "warnings": [
            "Gap analysis is descriptive and does not validate the strategy.",
            "Unsafe next steps remain blocked.",
        ],
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "research_only": True,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
