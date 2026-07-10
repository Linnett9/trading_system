from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ResearchCandidateFilterSpec:
    filter_name: str
    enabled: bool = False
    reason: str = "opt-in research-only candidate filter"


@dataclass(frozen=True)
class ResearchStrategyVariantSpec:
    base_variant_name: str
    new_variant_name: str
    candidate_filter: ResearchCandidateFilterSpec | None = None
    filtered_candidate_rows: Sequence[Mapping[str, Any]] | None = None
    metadata: Mapping[str, Any] | None = None
    research_only: bool = True
    enabled_for_research: bool = True
    enabled_for_paper_trading: bool = False
    enabled_for_live_trading: bool = False


def build_research_strategy_variant_inputs(
    candidate_rows: Sequence[Mapping[str, Any]],
    variant_spec: ResearchStrategyVariantSpec,
) -> dict[str, Any]:
    source_rows = (
        variant_spec.filtered_candidate_rows
        if variant_spec.filtered_candidate_rows is not None
        else candidate_rows
    )
    copied_rows = [dict(row) for row in source_rows]
    filter_spec = variant_spec.candidate_filter
    filter_enabled = bool(filter_spec and filter_spec.enabled)
    return {
        "base_variant_name": variant_spec.base_variant_name,
        "new_variant_name": variant_spec.new_variant_name,
        "research_only": variant_spec.research_only,
        "filter_name": filter_spec.filter_name if filter_spec else "NONE",
        "filter_enabled": filter_enabled,
        "candidate_rows": copied_rows,
        "candidate_count": len(copied_rows),
        "metadata": dict(variant_spec.metadata or {}),
        "default_behavior_unchanged": not filter_enabled,
        "enabled_for_research": variant_spec.enabled_for_research,
        "paper_trading_enabled": variant_spec.enabled_for_paper_trading,
        "live_trading_enabled": variant_spec.enabled_for_live_trading,
    }


def build_news_risk_research_variants(
    oos_rows: list[dict[str, Any]],
    *,
    apply_catastrophic_veto_to_candidates: Callable[..., dict[str, Any]],
    apply_catastrophic_policy_variant_to_candidates: Callable[..., dict[str, Any]],
    catastrophic_policy_variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    catastrophic_veto_filter = apply_catastrophic_veto_to_candidates(oos_rows)
    catastrophic_veto_confirmed_only_filter = apply_catastrophic_veto_to_candidates(
        oos_rows,
        policy_mode="CONFIRMED_ONLY_RESEARCH",
    )
    catastrophic_veto_manual_review_filter = apply_catastrophic_veto_to_candidates(
        oos_rows,
        policy_mode="MANUAL_REVIEW_RESEARCH",
    )
    policy_variant_filters = {
        str(spec["policy_name"]): apply_catastrophic_policy_variant_to_candidates(oos_rows, str(spec["policy_name"]))
        for spec in catastrophic_policy_variants
    }
    catastrophic_veto_variant = ResearchStrategyVariantSpec(
        base_variant_name="news_contrarian_rerank",
        new_variant_name="news_contrarian_rerank_catastrophic_veto",
        candidate_filter=ResearchCandidateFilterSpec(
            filter_name="catastrophic_veto",
            enabled=True,
        ),
        filtered_candidate_rows=catastrophic_veto_filter["filtered_candidates"],
        metadata={
            "policy": "catastrophic_veto_v1",
            "candidate_count_before_veto": len(oos_rows),
            "candidate_count_after_veto": len(catastrophic_veto_filter["filtered_candidates"]),
            "blocked_candidate_count": len(catastrophic_veto_filter["blocked_candidates"]),
        },
        research_only=True,
        enabled_for_research=True,
        enabled_for_paper_trading=False,
        enabled_for_live_trading=False,
    )
    catastrophic_veto_confirmed_only_variant = ResearchStrategyVariantSpec(
        base_variant_name="news_contrarian_rerank",
        new_variant_name="news_contrarian_rerank_catastrophic_veto_confirmed_only",
        candidate_filter=ResearchCandidateFilterSpec(
            filter_name="catastrophic_veto_confirmed_only",
            enabled=True,
        ),
        filtered_candidate_rows=catastrophic_veto_confirmed_only_filter["filtered_candidates"],
        metadata={
            "policy": "catastrophic_veto_confirmed_only_research_v1",
            "policy_mode": "CONFIRMED_ONLY_RESEARCH",
            "candidate_count_before_veto": len(oos_rows),
            "candidate_count_after_veto": len(catastrophic_veto_confirmed_only_filter["filtered_candidates"]),
            "blocked_candidate_count": len(catastrophic_veto_confirmed_only_filter["blocked_candidates"]),
            "unknown_text_candidate_count": len(catastrophic_veto_confirmed_only_filter["unknown_text_candidates"]),
            "missing_availability_candidate_count": len(catastrophic_veto_confirmed_only_filter["missing_availability_candidates"]),
        },
        research_only=True,
        enabled_for_research=True,
        enabled_for_paper_trading=False,
        enabled_for_live_trading=False,
    )
    catastrophic_veto_manual_review_variant = ResearchStrategyVariantSpec(
        base_variant_name="news_contrarian_rerank",
        new_variant_name="news_contrarian_rerank_catastrophic_veto_manual_review",
        candidate_filter=ResearchCandidateFilterSpec(
            filter_name="catastrophic_veto_manual_review",
            enabled=True,
        ),
        filtered_candidate_rows=catastrophic_veto_manual_review_filter["filtered_candidates"],
        metadata={
            "policy": "catastrophic_veto_manual_review_research_v1",
            "policy_mode": "MANUAL_REVIEW_RESEARCH",
            "candidate_count_before_veto": len(oos_rows),
            "candidate_count_after_veto": len(catastrophic_veto_manual_review_filter["filtered_candidates"]),
            "blocked_candidate_count": len(catastrophic_veto_manual_review_filter["blocked_candidates"]),
            "unknown_text_candidate_count": len(catastrophic_veto_manual_review_filter["unknown_text_candidates"]),
            "missing_availability_candidate_count": len(catastrophic_veto_manual_review_filter["missing_availability_candidates"]),
        },
        research_only=True,
        enabled_for_research=True,
        enabled_for_paper_trading=False,
        enabled_for_live_trading=False,
    )
    policy_replay_variants = [
        ResearchStrategyVariantSpec(
            base_variant_name="news_contrarian_rerank",
            new_variant_name=str(filter_result["variant_name"]),
            candidate_filter=ResearchCandidateFilterSpec(
                filter_name=f"catastrophic_policy_variant_{policy_name.lower()}",
                enabled=True,
            ),
            filtered_candidate_rows=filter_result["filtered_candidates"],
            metadata={
                "policy": f"{policy_name.lower()}_research_v1",
                "policy_name": policy_name,
                "policy_stage": filter_result["policy_stage"],
                "candidate_count_before_veto": len(oos_rows),
                "candidate_count_after_veto": len(filter_result["filtered_candidates"]),
                "blocked_candidate_count": len(filter_result["blocked_candidates"]),
                "unknown_evidence_candidate_count": len(filter_result["unknown_candidates"]),
            },
            research_only=True,
            enabled_for_research=True,
            enabled_for_paper_trading=False,
            enabled_for_live_trading=False,
        )
        for policy_name, filter_result in policy_variant_filters.items()
        if filter_result["policy_stage"] == "FULL_REPLAY_RESEARCH"
    ]
    return {
        "catastrophic_veto_filter": catastrophic_veto_filter,
        "catastrophic_veto_confirmed_only_filter": catastrophic_veto_confirmed_only_filter,
        "catastrophic_veto_manual_review_filter": catastrophic_veto_manual_review_filter,
        "policy_variant_filters": policy_variant_filters,
        "extra_research_variants": [
            catastrophic_veto_variant,
            catastrophic_veto_confirmed_only_variant,
            catastrophic_veto_manual_review_variant,
            *policy_replay_variants,
        ],
    }
