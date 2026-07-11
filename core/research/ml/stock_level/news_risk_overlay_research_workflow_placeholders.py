from __future__ import annotations

from typing import Any


def _validation_stage_placeholders(*, full_replay_computed: bool = False) -> dict[str, Any]:
    def stage(name: str, status: str, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
        implemented = status not in {"NOT_IMPLEMENTED", "NOT_READY", "NOT_ENABLED"}
        return {
            "stage_name": name,
            "status": status,
            "implemented": implemented,
            "blocks_final_validation": not implemented,
            "metric_output_allowed": implemented,
            "reason": reason,
            "warnings": warnings or [],
        }

    return {
        "schema_name": "stock_alpha_news_risk_overlay_validation_stage_placeholders",
        "schema_version": 1,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "validation_passed": False,
        "is_final_validation": False,
        "walk_forward": stage("walk_forward", "NOT_IMPLEMENTED", "Walk-forward robustness has not been implemented for this validation spine."),
        "placebo_permutation": stage("placebo_permutation", "NOT_IMPLEMENTED", "Placebo and permutation tests have not been implemented."),
        "exposure_matched_controls": stage("exposure_matched_controls", "NOT_IMPLEMENTED", "Exposure-matched controls have not been implemented."),
        "trade_count_matched_controls": stage("trade_count_matched_controls", "NOT_IMPLEMENTED", "Trade-count-matched controls have not been implemented."),
        "matched_controls": stage("matched_controls", "NOT_IMPLEMENTED", "Legacy aggregate matched-control placeholder; use exposure/trade-count controls when implemented."),
        "concentration_analysis": stage("concentration_analysis", "NOT_IMPLEMENTED", "Contribution and fragility analysis has not been implemented."),
        "year_regime_robustness": {
            **stage("year_regime_robustness", "AVAILABLE", "Ledger-level year/regime robustness report is available."),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "symbol_year_ablation": {
            **stage("symbol_year_ablation", "AVAILABLE", "Ledger-level symbol/year ablations are available."),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "survivorship_audit": stage("survivorship_audit", "NOT_IMPLEMENTED", "Point-in-time universe and survivorship audit has not been implemented."),
        "corporate_action_audit": stage("corporate_action_audit", "NOT_IMPLEMENTED", "Corporate-action validation has not been implemented."),
        "missing_news_bias": stage("missing_news_bias", "NOT_IMPLEMENTED", "Missing-news bias analysis has not been completed."),
        "transaction_cost_validation": stage("transaction_cost_validation", "NOT_IMPLEMENTED", "Realistic transaction-cost validation has not been implemented."),
        "intraday_5min_expansion_plan": {
            **stage("intraday_5min_expansion_plan", "PLANNING_ONLY", "Future Dell PC intraday 5-minute expansion is planning-only."),
            "implemented": True,
            "blocks_final_validation": False,
            "metric_output_allowed": False,
        },
        "catastrophic_news_evidence_quality": {
            **stage(
                "catastrophic_news_evidence_quality",
                "INSUFFICIENT_FOR_STRICT_VETO",
                "Catastrophic-news evidence quality is insufficient for strict live-style filtering.",
                ["research-only", "missing text/availability evidence", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "news_evidence_lineage": {
            **stage(
                "news_evidence_lineage",
                "INSUFFICIENT",
                "News evidence contract and lineage are incomplete for strict veto and text-model readiness.",
                ["observational audit only", "paper/live disabled", "text models disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "catastrophic_veto_strategy_comparison": {
            **stage(
                "catastrophic_veto_strategy_comparison",
                "FULL_REPLAY_COMPUTED" if full_replay_computed else "APPROXIMATE_LEDGER_SIMULATION",
                "Separate research-only full replay is computed." if full_replay_computed else "Catastrophic-veto policy, attribution, and ledger-level simulation are present, but full filtered replay is not computed.",
                ["research-only", "not enforced in current strategy", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": full_replay_computed,
            "replay_impact_status": "FULL_REPLAY_COMPUTED" if full_replay_computed else "APPROXIMATE_LEDGER_SIMULATION",
            "full_replay_status": "FULL_REPLAY_COMPUTED" if full_replay_computed else "FULL_REPLAY_NOT_AVAILABLE",
            "safe_replay_insertion_point": "RESEARCH_STRATEGY_VARIANT_INPUT_SEAM",
            "safe_filtered_variant_seam_status": "REPLAY_ADAPTER_EXECUTED" if full_replay_computed else "REPLAY_ADAPTER_AVAILABLE_OPT_IN_ONLY",
            "full_replay_blocker": "" if full_replay_computed else "integrated replay helper accepts optional extra research-only variants, but replay output does not contain the catastrophic-veto variant",
            "veto_strategy": "news_contrarian_rerank_catastrophic_veto",
        },
        "event_taxonomy_research": {
            **stage(
                "event_taxonomy_research",
                "RESEARCH_RULES_READY",
                "Deterministic headline taxonomy is available for research diagnostics only.",
                ["not production event_category", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "duplicate_grouping_heuristic": {
            **stage(
                "duplicate_grouping_heuristic",
                "HEURISTIC_ONLY",
                "Duplicate grouping is deterministic and heuristic, not provider-grade duplicate_group_id.",
                ["heuristic-only", "text-model readiness remains blocked"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "point_in_time_text_safety": {
            **stage(
                "point_in_time_text_safety",
                "PARTIAL_POINT_IN_TIME_SAFE",
                "Text safety audit is available but remains limited by availability timestamp coverage.",
                ["publication-only timestamps are not availability evidence"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "keyword_text_baseline": {
            **stage(
                "keyword_text_baseline",
                "RESEARCH_ONLY",
                "Deterministic keyword scores are emitted for diagnostics and are not used in strategy ranking.",
                ["no model training", "not used in replay"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "catastrophic_veto_bounceback": {
            **stage(
                "catastrophic_veto_bounceback",
                "RESEARCH_ONLY",
                "Removed-trade bounce-back attribution is observational and does not alter replay mechanics.",
                ["not used in strategy", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "extreme_only_policy_proposal": {
            **stage(
                "extreme_only_policy_proposal",
                "PROPOSED_NOT_REPLAYED",
                "Extreme-distress-only policy is a proposal and requires a future separate research-only replay.",
                ["proposal only", "not validated", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": False,
        },
        "catastrophic_policy_frontier": {
            **stage(
                "catastrophic_policy_frontier",
                "RESEARCH_ONLY_DIAGNOSTIC",
                "Policy frontier ranks catastrophic-veto variants for hypothesis triage only.",
                ["not final validation", "not model selection", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "loser_bounceback_casebook": {
            **stage(
                "loser_bounceback_casebook",
                "RESEARCH_ONLY_DIAGNOSTIC",
                "Loser-vs-bounceback casebook compares removed-trade losers and bounceback winners without changing replay mechanics.",
                ["observational only", "taxonomy proposal only", "paper/live disabled"],
            ),
            "implemented": True,
            "blocks_final_validation": True,
            "metric_output_allowed": True,
        },
        "text_model_readiness": {
            **stage(
                "text_model_readiness",
                "NOT_READY",
                "Text modelling is deferred until the numerical validation spine, taxonomy, timestamp, and duplicate-handling checks are complete.",
                [
                    "validation spine not complete",
                    "events still uncategorized",
                    "text timestamps not proven point-in-time",
                    "duplicate/syndication handling not proven",
                    "FinBERT deferred",
                ],
            ),
            "transformer_training_enabled": False,
            "bert_enabled": False,
            "finbert_enabled": False,
        },
    }
