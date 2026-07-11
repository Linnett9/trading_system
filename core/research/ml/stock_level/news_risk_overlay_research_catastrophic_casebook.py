from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.news_risk_overlay_research_catastrophic_utils import (
    DISTRESSED_DILUTION_TERMS,
    EXTREME_DISTRESS_TERMS,
    FILING_FORM_PATTERNS,
    FRAUD_TERMS,
    KEYWORD_BASELINE_RULES,
    SEVERE_LOSS_AVOIDANCE_TERMS,
    SOFT_RISK_REDUCE_TERMS,
    _avg_keyword,
    _bounceback_label,
    _case_keyword_scores,
    _catastrophic_trade_return,
    _catastrophic_veto_removal_reason,
    _difference,
    _event_category_for_candidate,
    _feature_diff_row,
    _headline_text,
    _mapping_first,
    _number,
    _rate,
    _severity_group_for_candidate,
)


def _filing_forms_detected(text: str) -> list[str]:
    normalized = text.lower()
    return [name for name, pattern in FILING_FORM_PATTERNS if re.search(pattern, normalized)]

def _is_generic_filing_headline(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return False
    has_form = bool(_filing_forms_detected(normalized))
    generic_terms = ("filed by", "files form", "form 10-q", "10-q filed", "8-k filed", "filed form")
    risk_terms = ("late", "nt 10-q", "going concern", "default", "delisting", "fraud", "investigation", "bankruptcy", "restatement")
    return has_form and any(term in normalized for term in generic_terms) and not any(term in normalized for term in risk_terms)

def _case_value(row: Mapping[str, Any], *keys: str) -> Any:
    value = _mapping_first(row, *keys)
    return value if value is not None else "UNAVAILABLE_INPUT"

def _casebook_case(row: Mapping[str, Any], case_type: str, reason: str) -> dict[str, Any]:
    keyword_scores = _case_keyword_scores(row)
    return {
        "case_type": case_type,
        "trade_id": _case_value(row, "trade_id", "id", "row_id"),
        "candidate_id": _case_value(row, "candidate_id", "trade_id", "row_id"),
        "symbol": _case_value(row, "symbol", "ticker"),
        "decision_timestamp": _case_value(row, "decision_timestamp", "rebalance_timestamp", "feature_timestamp"),
        "entry_date": _case_value(row, "entry_date", "entry_timestamp", "open_date"),
        "exit_date": _case_value(row, "exit_date", "exit_timestamp", "close_date"),
        "headline_text": _headline_text(row) or "UNAVAILABLE_INPUT",
        "summary_text": _case_value(row, "summary_text", "summary", "body_text", "body", "content"),
        "provider": _case_value(row, "provider", "source", "news_provider"),
        "availability_timestamp": _case_value(row, "availability_timestamp", "available_at", "asof_timestamp"),
        "publication_timestamp": _case_value(row, "publication_timestamp", "published_at", "published_at_utc", "timestamp"),
        "event_category_research": _event_category_for_candidate(row),
        "severity_group": _severity_group_for_candidate(row),
        "keyword_scores": json.dumps(keyword_scores, sort_keys=True),
        "news_score": _case_value(row, "news_score", "price_plus_news_risk_probability", "news_risk_score"),
        "price_model_score": _case_value(row, "price_model_score", "price_score", "price_only_news_risk_probability"),
        "removed_trade_return": _catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else "UNAVAILABLE_INPUT",
        "bounceback_label": _bounceback_label(row),
        "max_adverse_excursion_if_available": _case_value(row, "maximum_adverse_excursion", "max_adverse_excursion"),
        "max_favourable_excursion_if_available": _case_value(row, "maximum_favourable_excursion", "max_favourable_excursion"),
        "reason_for_selection": reason,
        "headline_is_generic_filing": _is_generic_filing_headline(_headline_text(row)),
        "filing_forms_detected": "|".join(_filing_forms_detected(_headline_text(row))) or "UNAVAILABLE_INPUT",
    }

def _catastrophic_veto_loser_bounceback_casebook_artifacts(
    rows: Sequence[Mapping[str, Any]],
    removed_trade_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates_by_id = {
        str(row.get("candidate_id")): row
        for row in rows
        if row.get("candidate_id") not in {None, ""}
    }
    enriched = []
    for trade in removed_trade_rows:
        candidate = candidates_by_id.get(str(trade.get("candidate_id")), {})
        merged = {**dict(candidate), **dict(trade)}
        enriched.append(merged)
    severe_losers = sorted(
        [row for row in enriched if _bounceback_label(row) == "SEVERE_LOSS"],
        key=lambda row: (_catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else math.inf, str(row.get("trade_id", ""))),
    )[:25]
    strong_winners = sorted(
        [row for row in enriched if _bounceback_label(row) == "BOUNCED_BACK_STRONGLY"],
        key=lambda row: (-(_catastrophic_trade_return(row) if _catastrophic_trade_return(row) is not None else -math.inf), str(row.get("trade_id", ""))),
    )[:25]
    cases = [
        _casebook_case(row, "top_severe_loser", "lowest removed trade returns among strict-veto removed trades")
        for row in severe_losers
    ] + [
        _casebook_case(row, "top_strong_bounceback_winner", "highest positive removed trade returns among strict-veto removed trades")
        for row in strong_winners
    ]
    features: list[tuple[str, Callable[[Mapping[str, Any]], bool], str]] = [
        ("has_headline_text", lambda row: bool(_headline_text(row).strip()), "headline text coverage"),
        ("has_provider", lambda row: _case_value(row, "provider", "source", "news_provider") != "UNAVAILABLE_INPUT", "provider/source coverage"),
        ("has_availability_timestamp", lambda row: _case_value(row, "availability_timestamp", "available_at", "asof_timestamp") != "UNAVAILABLE_INPUT", "point-in-time availability evidence"),
        ("headline_is_generic_filing", lambda row: _is_generic_filing_headline(_headline_text(row)), "generic filing headline rate"),
        ("headline_mentions_10q", lambda row: "10-q" in _headline_text(row).lower() or "10q" in _headline_text(row).lower(), "routine 10-Q headline mention"),
        ("headline_mentions_8k", lambda row: "8-k" in _headline_text(row).lower() or "8k" in _headline_text(row).lower(), "8-K headline mention"),
        ("headline_mentions_nt_10q", lambda row: "nt 10-q" in _headline_text(row).lower() or "nt10q" in _headline_text(row).lower(), "late filing signal"),
        ("headline_mentions_going_concern", lambda row: "going concern" in _headline_text(row).lower(), "going-concern signal"),
        ("headline_mentions_default", lambda row: "default" in _headline_text(row).lower(), "default signal"),
        ("headline_mentions_delisting", lambda row: "delisting" in _headline_text(row).lower(), "delisting signal"),
        ("headline_mentions_suspension", lambda row: "suspension" in _headline_text(row).lower(), "trading-suspension signal"),
        ("headline_mentions_fraud", lambda row: "fraud" in _headline_text(row).lower(), "fraud signal"),
        ("headline_mentions_investigation", lambda row: "investigation" in _headline_text(row).lower(), "investigation signal"),
        ("headline_mentions_dilution", lambda row: "dilution" in _headline_text(row).lower() or "dilutive" in _headline_text(row).lower(), "dilution signal"),
        ("headline_mentions_offering", lambda row: "offering" in _headline_text(row).lower(), "offering signal"),
        ("headline_mentions_bankruptcy", lambda row: "bankruptcy" in _headline_text(row).lower(), "bankruptcy signal"),
    ]
    feature_diff = [
        _feature_diff_row(name, _rate(severe_losers, predicate), _rate(strong_winners, predicate), interpretation)
        for name, predicate, interpretation in features
    ]
    for name in ("event_category_research", "severity_group"):
        loser_value = Counter(_event_category_for_candidate(row) if name == "event_category_research" else _severity_group_for_candidate(row) for row in severe_losers).most_common(1)
        winner_value = Counter(_event_category_for_candidate(row) if name == "event_category_research" else _severity_group_for_candidate(row) for row in strong_winners).most_common(1)
        feature_diff.append(_feature_diff_row(
            name,
            loser_value[0][0] if loser_value else "UNAVAILABLE_INPUT",
            winner_value[0][0] if winner_value else "UNAVAILABLE_INPUT",
            "most common categorical value",
        ))
    keyword_diff = [
        _feature_diff_row(score_name, _avg_keyword(severe_losers, score_name), _avg_keyword(strong_winners, score_name), f"average {score_name}")
        for score_name, _terms in KEYWORD_BASELINE_RULES
    ]
    feature_diff.extend([
        _feature_diff_row("keyword_distress_score", _avg_keyword(severe_losers, "distress_score"), _avg_keyword(strong_winners, "distress_score"), "average distress keyword score"),
        _feature_diff_row("keyword_dilution_score", _avg_keyword(severe_losers, "dilution_score"), _avg_keyword(strong_winners, "dilution_score"), "average dilution keyword score"),
        _feature_diff_row("keyword_litigation_score", _avg_keyword(severe_losers, "litigation_score"), _avg_keyword(strong_winners, "litigation_score"), "average litigation keyword score"),
        _feature_diff_row("keyword_generic_negative_score", _avg_keyword(severe_losers, "generic_negative_score"), _avg_keyword(strong_winners, "generic_negative_score"), "average generic-negative keyword score"),
        _feature_diff_row("news_score_decile_if_available", "UNAVAILABLE_INPUT", "UNAVAILABLE_INPUT", "requires explicit decile field on case rows"),
        _feature_diff_row("price_model_score_bucket_if_available", "UNAVAILABLE_INPUT", "UNAVAILABLE_INPUT", "requires explicit bucket field on case rows"),
    ])
    generic_cases = [case for case in cases if case["headline_is_generic_filing"]]
    filing_forms = sorted({
        form
        for case in cases
        for form in str(case.get("filing_forms_detected", "")).split("|")
        if form and form != "UNAVAILABLE_INPUT"
    })
    generic_diagnostic = {
        "generic_filing_case_count": len(generic_cases),
        "generic_filing_severe_loser_count": sum(case["case_type"] == "top_severe_loser" for case in generic_cases),
        "generic_filing_strong_bounceback_count": sum(case["case_type"] == "top_strong_bounceback_winner" for case in generic_cases),
        "filing_forms_detected": filing_forms,
        "needs_filing_content_not_just_headline": bool(generic_cases),
        "recommended_next_step": "Treat routine 10-Q/8-K headlines as weak evidence unless filing body, summary, or late-filing terms add risk context.",
    }
    report = {
        "schema_name": "catastrophic_veto_loser_bounceback_casebook",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "AVAILABLE" if cases else "UNAVAILABLE_INPUT",
        "case_selection": "top severe losers and top strong bounce-back winners among strict-veto removed trades",
        "severe_loser_case_count": len(severe_losers),
        "strong_bounceback_case_count": len(strong_winners),
        "generic_filing_diagnostic": generic_diagnostic,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "warnings": ["Research-only casebook; no replay, ranking, sizing, or execution behavior changed."],
    }
    plan = {
        "schema_name": "catastrophic_veto_taxonomy_improvement_plan",
        "schema_version": 1,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PROPOSED",
        "proposal_only": True,
        "recommended_rule_improvements": [
            "routine 10-Q headline alone is not enough",
            "NT 10-Q / late filing may be riskier than normal 10-Q",
            "going concern text requires filing body or summary, not just generic headline",
            "distressed financing needs terms like emergency, rescue, going concern, liquidity",
            "ordinary capital raise should not equal distressed dilution",
        ],
        "generic_filing_diagnostic": generic_diagnostic,
        "validation_label": "PSEUDO_HOLDOUT",
        "validation_passed": False,
        "final_validation_status": "NOT_FINAL_VALIDATION",
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "finbert_readiness": "NOT_READY",
        "transformer_enabled": False,
        "warnings": ["Proposal artifact only; deterministic taxonomy rules were not changed by this plan."],
    }
    return report, cases, feature_diff, keyword_diff, plan
