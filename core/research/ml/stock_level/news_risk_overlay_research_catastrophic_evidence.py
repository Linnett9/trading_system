from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.news_sources import classify_catastrophic_news_rows
from core.research.ml.stock_level.news_risk_overlay_research_catastrophic_core import (
    apply_catastrophic_veto_to_candidates,
)
from core.research.ml.stock_level.news_risk_overlay_research_catastrophic_utils import (
    DISTRESSED_DILUTION_TERMS,
    EVENT_TAXONOMY_RULES,
    EXTREME_DISTRESS_TERMS,
    FRAUD_TERMS,
    KEYWORD_BASELINE_RULES,
    SEVERE_LOSS_AVOIDANCE_TERMS,
    SOFT_RISK_REDUCE_TERMS,
    _classify_event_taxonomy_from_headline,
    _headline_text,
    _mapping_first,
    _normalized_headline,
)


def _catastrophic_news_evidence_quality_artifacts(
    rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    full_replay_report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    classifications = classify_catastrophic_news_rows(rows)
    filter_result = apply_catastrophic_veto_to_candidates(rows, classifications)
    audit = dict(filter_result["filter_audit"])

    def present(row: Mapping[str, Any], *keys: str) -> bool:
        return any(row.get(key) is not None and str(row.get(key)).strip() for key in keys)

    field_specs = {
        "headline": ("headline_text", "headline", "title"),
        "summary": ("summary_text", "summary", "description"),
        "body": ("body_text", "body", "content", "article_body"),
        "publication_timestamp": ("publication_timestamp", "published_at", "timestamp"),
        "availability_timestamp": ("availability_timestamp", "available_at", "asof_timestamp"),
        "event_category": ("event_type", "event", "provider_category", "category"),
    }
    field_counts = {
        name: sum(present(row, *keys) for row in rows)
        for name, keys in field_specs.items()
    }
    has_any_text_count = sum(
        present(row, "headline_text", "headline", "title", "summary_text", "summary", "description", "body_text", "body", "content", "article_body", "event_type", "event", "provider_category", "category")
        for row in rows
    )
    point_in_time_safe_count = sum(bool(row.get("point_in_time_safe")) for row in classifications)
    usable_strict = sum(
        bool(row.get("point_in_time_safe"))
        and row.get("classification_method") != "UNAVAILABLE_INPUT"
        for row in classifications
    )
    usable_confirmed_only = sum(row.get("classification_method") != "UNAVAILABLE_INPUT" for row in classifications)
    status = "INSUFFICIENT_FOR_STRICT_VETO" if usable_strict == 0 else (
        "PARTIAL_EVIDENCE" if usable_strict < len(rows) else "SUFFICIENT_FOR_RESEARCH"
    )
    report = {
        "schema_name": "catastrophic_news_evidence_quality_report",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": status,
        "candidate_count": len(rows),
        "has_headline_count": field_counts["headline"],
        "has_summary_count": field_counts["summary"],
        "has_body_count": field_counts["body"],
        "has_any_text_count": has_any_text_count,
        "missing_text_count": len(rows) - has_any_text_count,
        "has_publication_timestamp_count": field_counts["publication_timestamp"],
        "has_availability_timestamp_count": field_counts["availability_timestamp"],
        "missing_availability_timestamp_count": len(rows) - field_counts["availability_timestamp"],
        "point_in_time_safe_count": point_in_time_safe_count,
        "point_in_time_unsafe_count": len(rows) - point_in_time_safe_count,
        "has_event_category_count": field_counts["event_category"],
        "uncategorized_event_count": len(rows) - field_counts["event_category"],
        "confirmed_catastrophic_candidate_count": audit["confirmed_catastrophic_candidate_count"],
        "manual_review_candidate_count": audit["manual_review_candidate_count"],
        "unknown_candidate_count": audit["unknown_candidate_count"],
        "strict_policy_blocked_candidate_count": audit["strict_policy_blocked_candidate_count"],
        "usable_for_strict_veto_count": usable_strict,
        "usable_for_confirmed_only_veto_count": usable_confirmed_only,
        "warnings": [
            "Publication timestamps alone are not point-in-time availability evidence.",
            "Unknown or missing evidence remains blocked by STRICT_SAFETY.",
            "Evidence-quality reporting is observational and does not alter replay inputs.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    by_field = [
        {
            "field": name,
            "available_count": count,
            "missing_count": len(rows) - count,
            "availability_ratio": count / max(len(rows), 1),
        }
        for name, count in field_counts.items()
    ]

    by_symbol_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_symbol_groups.setdefault(str(row.get("symbol", row.get("ticker", "UNKNOWN"))), []).append(row)
    by_symbol = []
    for symbol, symbol_rows in sorted(by_symbol_groups.items()):
        symbol_classifications = classify_catastrophic_news_rows(symbol_rows)
        symbol_filter = apply_catastrophic_veto_to_candidates(symbol_rows, symbol_classifications)["filter_audit"]
        symbol_text = sum(
            present(row, "headline_text", "headline", "title", "summary_text", "summary", "description", "body_text", "body", "content", "article_body", "event_type", "event", "provider_category", "category")
            for row in symbol_rows
        )
        symbol_availability = sum(present(row, "availability_timestamp", "available_at", "asof_timestamp") for row in symbol_rows)
        by_symbol.append({
            "symbol": symbol,
            "candidate_count": len(symbol_rows),
            "has_any_text_count": symbol_text,
            "missing_text_count": len(symbol_rows) - symbol_text,
            "has_availability_timestamp_count": symbol_availability,
            "missing_availability_timestamp_count": len(symbol_rows) - symbol_availability,
            "confirmed_catastrophic_candidate_count": symbol_filter["confirmed_catastrophic_candidate_count"],
            "strict_policy_blocked_candidate_count": symbol_filter["strict_policy_blocked_candidate_count"],
        })

    base_trades = [
        trade for trade in replay.get("trade_ledger", [])
        if str(trade.get("strategy_variant", "")) == "news_contrarian_rerank"
    ]
    candidate_id_sets = {
        "STRICT_SAFETY": {str(row.get("candidate_id")) for row in filter_result["blocked_candidates"]},
        "CONFIRMED_ONLY_RESEARCH": {str(row.get("candidate_id")) for row in filter_result["confirmed_catastrophic_candidates"]},
        "MANUAL_REVIEW_RESEARCH": {
            str(row.get("candidate_id"))
            for row in [*filter_result["confirmed_catastrophic_candidates"], *filter_result["manual_review_candidates"]]
        },
    }
    counts = []
    mode_variant_names = {
        "STRICT_SAFETY": "news_contrarian_rerank_catastrophic_veto",
        "CONFIRMED_ONLY_RESEARCH": "news_contrarian_rerank_catastrophic_veto_confirmed_only",
        "MANUAL_REVIEW_RESEARCH": "news_contrarian_rerank_catastrophic_veto_manual_review",
    }
    risk_metrics = replay.get("risk_metrics", {})
    daily_equity = replay.get("daily_equity", {})
    extra_variant_metadata = replay.get("extra_research_variant_metadata", {})
    for mode in ("STRICT_SAFETY", "CONFIRMED_ONLY_RESEARCH", "MANUAL_REVIEW_RESEARCH"):
        blocked_ids = candidate_id_sets[mode]
        total_blocked = len(blocked_ids)
        strict_mode = mode == "STRICT_SAFETY"
        variant_name = mode_variant_names[mode]
        replayed = (
            isinstance(risk_metrics, Mapping)
            and variant_name in risk_metrics
            and isinstance(daily_equity, Mapping)
            and variant_name in daily_equity
            and isinstance(extra_variant_metadata, Mapping)
            and variant_name in extra_variant_metadata
        )
        counts.append({
            "policy_mode": mode,
            "strategy_variant": variant_name,
            "candidate_count_before_veto": len(rows),
            "candidate_count_after_veto": len(rows) - total_blocked,
            "confirmed_catastrophic_blocked_candidate_count": audit["confirmed_catastrophic_candidate_count"],
            "manual_review_blocked_candidate_count": audit["manual_review_candidate_count"] if mode == "MANUAL_REVIEW_RESEARCH" else (audit["manual_review_candidate_count"] if strict_mode else 0),
            "unknown_text_blocked_candidate_count": audit["unknown_text_candidate_count"] if strict_mode else 0,
            "missing_availability_blocked_candidate_count": audit["missing_availability_candidate_count"] if strict_mode else 0,
            "total_blocked_candidate_count": total_blocked,
            "estimated_removed_trade_count": sum(str(trade.get("candidate_id")) in blocked_ids for trade in base_trades),
            "full_replay_computed": bool(full_replay_report.get("full_replay_computed")) if strict_mode else replayed,
            "full_replay_status": full_replay_report.get("replay_impact_status", "FULL_REPLAY_NOT_AVAILABLE") if strict_mode else ("FULL_REPLAY_COMPUTED" if replayed else "COUNT_ONLY_NOT_REPLAYED"),
            "veto_metrics_status": full_replay_report.get("veto_metrics_status", "UNAVAILABLE_INPUT") if strict_mode else ("AVAILABLE" if replayed else "COUNT_ONLY_NOT_REPLAYED"),
            "paper_trading_allowed": False,
            "live_trading_allowed": False,
            "validation_passed": False,
            "final_validation_status": "NOT_FINAL_VALIDATION",
            "warnings": ["Research-only policy comparison; unknown evidence is not approved for paper/live use."],
        })
    comparison = {
        "schema_name": "catastrophic_veto_policy_mode_comparison",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "code_schema_version": "catastrophic_veto_research_v1",
        "status": "RESEARCH_ONLY_COUNT_COMPARISON",
        "active_full_replay_policy_mode": "STRICT_SAFETY",
        "policy_modes": counts,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "warnings": ["Policy-mode variants are research-only; unknown evidence is not approved for paper/live use."],
    }
    return report, by_field, by_symbol, comparison, counts

def _news_event_taxonomy_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    counts: dict[str, int] = {}
    examples_by_category: dict[str, list[dict[str, Any]]] = {}
    text_count = 0
    for row in rows:
        headline = _headline_text(row)
        if headline.strip():
            text_count += 1
        category, matched_terms = _classify_event_taxonomy_from_headline(headline)
        counts[category] = counts.get(category, 0) + 1
        example = {
            "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
            "decision_timestamp": _mapping_first(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp", "timestamp") or "UNKNOWN",
            "event_category_research": category,
            "matched_terms": "|".join(matched_terms),
            "headline_text": headline,
            "research_only": True,
        }
        examples_by_category.setdefault(category, [])
        if len(examples_by_category[category]) < 5:
            examples_by_category[category].append(example)
    count_rows = [
        {
            "event_category_research": category,
            "candidate_count": count,
            "coverage_ratio": count / max(len(rows), 1),
        }
        for category, count in sorted(counts.items())
    ]
    examples = [
        example
        for category in sorted(examples_by_category)
        for example in examples_by_category[category]
    ]
    categorized_count = len(rows) - counts.get("uncategorized", 0)
    report = {
        "schema_name": "news_event_taxonomy_report",
        "schema_version": 1,
        "status": "RESEARCH_RULES_READY" if text_count else "UNAVAILABLE_INPUT",
        "taxonomy_method": "DETERMINISTIC_HEADLINE_RULES",
        "candidate_count": len(rows),
        "has_headline_count": text_count,
        "categorized_count": categorized_count,
        "uncategorized_count": counts.get("uncategorized", 0),
        "event_taxonomy_research_ready": text_count > 0,
        "production_event_category_ready": False,
        "categories": [category for category, _terms in EVENT_TAXONOMY_RULES] + ["uncategorized"],
        "warnings": [
            "Research-only headline taxonomy; not a production event_category field.",
            "Catastrophic taxonomy remains the conservative blocking taxonomy.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, count_rows, examples

def _news_duplicate_grouping_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        symbol = (_mapping_first(row, "symbol", "ticker") or "UNKNOWN").upper()
        headline = _normalized_headline(row)
        availability = _mapping_first(row, "availability_timestamp", "available_at", "asof_timestamp", "news_feature_timestamp") or "UNKNOWN"
        provider = (_mapping_first(row, "provider") or "UNKNOWN").lower()
        date_key = availability[:10] if availability != "UNKNOWN" else "UNKNOWN"
        key = "|".join((symbol, headline or "NO_HEADLINE", date_key, provider))
        group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        groups.setdefault(group_id, []).append(row)
    duplicate_groups = {group_id: members for group_id, members in groups.items() if len(members) > 1}
    examples = []
    for group_id, members in sorted(duplicate_groups.items()):
        for row in members[:5]:
            examples.append({
                "duplicate_group_id_heuristic": group_id,
                "duplicate_group_method": "symbol_normalized_headline_availability_date_provider",
                "duplicate_group_size": len(members),
                "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
                "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
                "headline_text": _headline_text(row),
            })
    duplicate_candidate_count = sum(len(members) for members in duplicate_groups.values())
    report = {
        "schema_name": "news_duplicate_grouping_report",
        "schema_version": 1,
        "status": "HEURISTIC_ONLY",
        "duplicate_group_method": "symbol_normalized_headline_availability_date_provider",
        "candidate_count": len(rows),
        "duplicate_group_count": len(duplicate_groups),
        "singleton_count": sum(1 for members in groups.values() if len(members) == 1),
        "duplicate_candidate_count": duplicate_candidate_count,
        "duplicate_grouping_heuristic_ready": True,
        "production_duplicate_group_id_ready": False,
        "warnings": [
            "Research-only heuristic; not provider-grade deduplication.",
            "Does not mark production duplicate_group_id as fully available.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, examples

def _parse_optional_timestamp(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None

def _news_point_in_time_text_safety_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    has_text_count = 0
    has_availability_count = 0
    safe_text_count = 0
    unsafe_text_count = 0
    publication_only_count = 0
    availability_after_decision_count = 0
    missing_decision_count = 0
    missing_availability_count = 0
    for row in rows:
        has_text = bool(_headline_text(row).strip() or _mapping_first(row, "summary_text", "summary", "body_text", "body", "content"))
        availability_text = _mapping_first(row, "availability_timestamp", "available_at", "asof_timestamp")
        publication_text = _mapping_first(row, "publication_timestamp", "published_at", "published_at_utc", "timestamp")
        decision_text = _mapping_first(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp")
        availability = _parse_optional_timestamp(availability_text)
        decision = _parse_optional_timestamp(decision_text)
        if has_text:
            has_text_count += 1
        if availability is not None:
            has_availability_count += 1
        else:
            missing_availability_count += 1
        if decision is None:
            missing_decision_count += 1
        if publication_text and availability is None:
            publication_only_count += 1
        safe = has_text and availability is not None and decision is not None and availability <= decision
        after_decision = has_text and availability is not None and decision is not None and availability > decision
        if safe:
            safe_text_count += 1
        elif has_text:
            unsafe_text_count += 1
        if after_decision:
            availability_after_decision_count += 1
        if (after_decision or (has_text and not safe)) and len(examples) < 50:
            examples.append({
                "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
                "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
                "decision_timestamp": decision_text or "UNKNOWN",
                "availability_timestamp": availability_text or "UNKNOWN",
                "publication_timestamp": publication_text or "UNKNOWN",
                "has_text": has_text,
                "safe_text": safe,
                "reason": "AVAILABILITY_AFTER_DECISION" if after_decision else "MISSING_POINT_IN_TIME_EVIDENCE",
            })
    report = {
        "schema_name": "news_point_in_time_text_safety_report",
        "schema_version": 1,
        "status": "PARTIAL_POINT_IN_TIME_SAFE" if safe_text_count else "INSUFFICIENT",
        "candidate_count": len(rows),
        "has_text_count": has_text_count,
        "has_availability_timestamp_count": has_availability_count,
        "safe_text_count": safe_text_count,
        "unsafe_text_count": unsafe_text_count,
        "publication_only_count": publication_only_count,
        "availability_after_decision_count": availability_after_decision_count,
        "missing_decision_timestamp_count": missing_decision_count,
        "missing_availability_timestamp_count": missing_availability_count,
        "point_in_time_text_safety_ready": safe_text_count > 0 and availability_after_decision_count == 0,
        "warnings": [
            "Publication timestamp alone is not availability evidence.",
            "Rows without availability_timestamp are not point-in-time safe for text use.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, examples

def _news_text_keyword_baseline_artifacts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    scored_count = 0
    for row in rows:
        headline = _headline_text(row)
        normalized = headline.lower()
        scores = {
            score_name: sum(1 for term in terms if term in normalized)
            for score_name, terms in KEYWORD_BASELINE_RULES
        }
        if headline.strip():
            scored_count += 1
        score_rows.append({
            "candidate_id": _mapping_first(row, "candidate_id", "trade_id", "row_id") or "UNKNOWN",
            "symbol": _mapping_first(row, "symbol", "ticker") or "UNKNOWN",
            "decision_timestamp": _mapping_first(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp", "timestamp") or "UNKNOWN",
            "headline_text": headline,
            **scores,
            "keyword_baseline_total_score": sum(scores.values()),
            "used_in_strategy": False,
            "research_only": True,
        })
    report = {
        "schema_name": "news_text_keyword_baseline_report",
        "schema_version": 1,
        "status": "RESEARCH_ONLY" if scored_count else "UNAVAILABLE_INPUT",
        "candidate_count": len(rows),
        "scored_headline_count": scored_count,
        "score_columns": [name for name, _terms in KEYWORD_BASELINE_RULES],
        "keyword_baseline_ready": scored_count > 0,
        "used_in_strategy": False,
        "sklearn_required": False,
        "finbert_enabled": False,
        "transformer_enabled": False,
        "warnings": [
            "Deterministic keyword baseline only; not used in ranking or replay.",
        ],
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
    }
    return report, score_rows
