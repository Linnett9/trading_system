from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


RETURN_COLUMNS = (
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "forward_return",
)

REVERSIBLE_EVENT_CATEGORIES = {
    "earnings_miss",
    "guidance_cut",
    "analyst_downgrade",
    "operational_issue",
    "macro_or_sector",
}

SERIOUS_AMBIGUOUS_EVENT_CATEGORIES = {
    "litigation_or_regulatory",
    "management_change",
    "capital_raise_or_dilution",
    "distressed_dilution",
}

EXTREME_DISTRESS_EVENT_CATEGORIES = {
    "catastrophic_or_distress",
}

EXTREME_DISTRESS_TERMS = (
    "bankruptcy",
    "chapter 11",
    "chapter 7",
    "insolven",
    "liquidat",
    "administration",
    "default",
    "going concern",
    "delisting",
    "suspension",
    "trading suspension",
    "suspended trading",
    "fraud",
    "accounting irregular",
    "material misstatement",
    "rescue financing",
)

FRAUD_EVENT_CATEGORIES = {"fraud_or_accounting"}

DISTRESSED_DILUTION_EVENT_CATEGORIES = {"distressed_dilution"}

EXTREME_DISTRESS_ONLY_TERMS = (
    "bankruptcy",
    "chapter 11",
    "chapter 7",
    "insolven",
    "liquidat",
    "administration",
    "default",
    "debt default",
    "going concern",
    "delisting",
    "trading suspension",
    "suspended trading",
    "winding-up petition",
    "rescue financing",
    "emergency rescue financing",
)

FRAUD_TERMS = (
    "fraud",
    "accounting irregular",
    "auditor resign",
    "qualified audit",
    "criminal probe",
    "enforcement action",
    "major enforcement",
    "material misstatement",
)

DISTRESSED_DILUTION_TERMS = (
    "deep discount raise",
    "emergency capital raise",
    "highly dilutive",
    "distressed dilution",
    "covenant breach",
    "rescue financing",
)

SEVERE_LOSS_AVOIDANCE_TERMS = (
    "bankruptcy",
    "default",
    "delisting",
    "suspension",
    "fraud",
    "investigation",
    "liquidation",
    "going concern",
    "covenant breach",
    "rescue financing",
)

SOFT_RISK_REDUCE_TERMS = (
    "major litigation",
    "regulatory investigation",
    "distressed dilution",
    "management crisis",
    "large guidance cut",
)

EVENT_TAXONOMY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("catastrophic_or_distress", ("bankruptcy", "chapter 11", "insolvency", "liquidation", "going concern", "delisting", "default", "administration", "winding-up petition", "trading suspension", "suspended trading")),
    ("fraud_or_accounting", ("fraud", "accounting irregular", "restatement", "misstatement", "auditor resign", "qualified audit", "criminal probe", "enforcement action")),
    ("distressed_dilution", ("deep discount raise", "emergency capital raise", "rescue financing", "distressed dilution", "highly dilutive", "covenant breach")),
    ("guidance_cut", ("cuts guidance", "lowers guidance", "withdraws guidance", "guidance cut", "reduces outlook", "large guidance cut")),
    ("earnings_miss", ("earnings miss", "misses estimates", "misses expectations", "profit warning", "revenue miss")),
    ("analyst_downgrade", ("downgrade", "downgrades", "price target cut", "rating cut")),
    ("capital_raise_or_dilution", ("stock offering", "share offering", "capital raise", "dilution", "dilutive", "secondary offering")),
    ("litigation_or_regulatory", ("lawsuit", "litigation", "class action", "regulatory", "investigation", "probe", "sec charges", "major enforcement")),
    ("management_change", ("ceo resigns", "cfo resigns", "management change", "steps down", "departure")),
    ("operational_issue", ("plant shutdown", "production halt", "recall", "supply disruption", "cyberattack", "outage")),
    ("macro_or_sector", ("sector", "industry", "tariff", "rates", "inflation", "macro", "commodity")),
)

KEYWORD_BASELINE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("distress_score", ("bankruptcy", "insolvency", "going concern", "default", "delisting", "liquidation")),
    ("earnings_negative_score", ("earnings miss", "misses estimates", "profit warning", "revenue miss")),
    ("guidance_negative_score", ("cuts guidance", "lowers guidance", "withdraws guidance", "reduces outlook")),
    ("litigation_score", ("lawsuit", "litigation", "class action", "regulatory", "investigation", "probe")),
    ("dilution_score", ("offering", "capital raise", "dilution", "dilutive", "secondary")),
    ("management_change_score", ("ceo resigns", "cfo resigns", "steps down", "departure")),
    ("generic_negative_score", ("falls", "drops", "warning", "cuts", "misses", "lower", "weak")),
)

FILING_FORM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("NT_10_Q", r"\bnt\s*-?\s*10\s*-?\s*q\b"),
    ("10_Q", r"\b10\s*-?\s*q\b"),
    ("8_K", r"\b8\s*-?\s*k\b"),
)

def _headline_text(row: Mapping[str, Any]) -> str:
    return _mapping_first(row, "headline_text", "headline", "title") or ""

def _normalized_headline(row: Mapping[str, Any]) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", _headline_text(row).lower())).strip()

def _classify_event_taxonomy_from_headline(headline: str) -> tuple[str, list[str]]:
    normalized = re.sub(r"\s+", " ", headline.lower()).strip()
    if not normalized:
        return "uncategorized", []
    for category, terms in EVENT_TAXONOMY_RULES:
        matched = [term for term in terms if term in normalized]
        if matched:
            return category, matched
    return "uncategorized", []

def _bounceback_label(row: Mapping[str, Any]) -> str:
    trade_return = _as_optional_float(
        _mapping_first(
            row,
            "net_return",
            "trade_return_net",
            "return",
            "total_return",
            "removed_trade_return",
            "actual_forward_return_10d",
        )
    )
    if trade_return is None:
        return "UNAVAILABLE_OUTCOME"
    if trade_return > 0.10:
        return "BOUNCED_BACK_STRONGLY"
    if trade_return > 0.0:
        return "BOUNCED_BACK_WEAKLY"
    if trade_return <= -0.10:
        return "SEVERE_LOSS"
    return "DID_NOT_BOUNCE"

def _catastrophic_trade_return(row: Mapping[str, Any]) -> float | None:
    return _as_optional_float(
        _mapping_first(
            row,
            "net_return",
            "trade_return_net",
            "return",
            "total_return",
            "removed_trade_return",
            "actual_forward_return_10d",
        )
    )

def _event_category_for_candidate(row: Mapping[str, Any]) -> str:
    category, _terms = _classify_event_taxonomy_from_headline(_headline_text(row))
    return category

def _severity_group_for_candidate(row: Mapping[str, Any]) -> str:
    headline = _headline_text(row).lower()
    has_text = bool(headline.strip())
    has_availability = bool(_mapping_first(row, "availability_timestamp", "available_at", "asof_timestamp"))
    category = _event_category_for_candidate(row)
    if not has_text or not has_availability or category == "uncategorized":
        return "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"
    matched_categories = str(row.get("matched_categories", row.get("catastrophic_veto_matched_categories", ""))).lower()
    if (
        category in DISTRESSED_DILUTION_EVENT_CATEGORIES
        or any(term in headline for term in DISTRESSED_DILUTION_TERMS)
        or "distressed_dilution" in matched_categories
    ):
        return "DISTRESS_OR_DILUTION"
    if (
        category in FRAUD_EVENT_CATEGORIES
        or any(term in headline for term in FRAUD_TERMS)
        or any(term in matched_categories for term in ("fraud", "accounting", "auditor", "enforcement"))
    ):
        return "EXTREME_DISTRESS_OR_FRAUD"
    if (
        category in EXTREME_DISTRESS_EVENT_CATEGORIES
        or any(term in headline for term in EXTREME_DISTRESS_ONLY_TERMS)
        or any(term in matched_categories for term in ("bankruptcy", "insolvency", "default", "delisting", "going_concern"))
    ):
        return "EXTREME_DISTRESS"
    if category in REVERSIBLE_EVENT_CATEGORIES:
        return "REVERSIBLE_BAD_NEWS"
    if category in SERIOUS_AMBIGUOUS_EVENT_CATEGORIES:
        return "SERIOUS_BUT_AMBIGUOUS"
    return "UNKNOWN_OR_INSUFFICIENT_EVIDENCE"

def _metric_delta(base: Mapping[str, Any], candidate: Mapping[str, Any], *names: str) -> float | str:
    base_value = _metric(base, *names)
    candidate_value = _metric(candidate, *names)
    if base_value is None or candidate_value is None:
        return "UNAVAILABLE_INPUT"
    return candidate_value - base_value

def _rate(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float | str:
    if not rows:
        return "UNAVAILABLE_INPUT"
    return sum(1 for row in rows if predicate(row)) / len(rows)

def _difference(left: Any, right: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return "UNAVAILABLE_INPUT"

def _feature_diff_row(feature: str, loser_value: Any, winner_value: Any, interpretation: str) -> dict[str, Any]:
    diff = _difference(loser_value, winner_value)
    direction = "UNAVAILABLE_INPUT"
    if isinstance(diff, (int, float)):
        direction = "HIGHER_IN_SEVERE_LOSERS" if diff > 0 else ("HIGHER_IN_STRONG_BOUNCEBACK" if diff < 0 else "NO_DIFFERENCE")
    return {
        "feature_name": feature,
        "severe_loser_value_or_rate": loser_value,
        "strong_bounceback_value_or_rate": winner_value,
        "difference": diff,
        "direction": direction,
        "interpretation": interpretation,
    }

def _case_keyword_scores(row: Mapping[str, Any]) -> dict[str, int]:
    headline = _headline_text(row).lower()
    return {
        score_name: int(row.get(score_name, 0) or 0) if _number(row.get(score_name)) is not None else sum(1 for term in terms if term in headline)
        for score_name, terms in KEYWORD_BASELINE_RULES
    }

def _avg_keyword(rows: Sequence[Mapping[str, Any]], score_name: str) -> float | str:
    if not rows:
        return "UNAVAILABLE_INPUT"
    return mean(_case_keyword_scores(row).get(score_name, 0) for row in rows)

def _catastrophic_veto_removal_reason(
    blocked: bool,
    manual_review: bool,
    unknown: bool,
    missing_availability: bool = False,
) -> str:
    if blocked:
        return "BLOCK_CONTRARIAN_ENTRY"
    if manual_review:
        return "BLOCK_UNTIL_REVIEWED"
    if unknown:
        return "DO_NOT_TREAT_AS_SAFE"
    if missing_availability:
        return "NOT_POINT_IN_TIME_SAFE"
    return "NOT_REMOVED"

def _catastrophic_veto_removed_symbol_rows(
    removed_trade_rows: Sequence[Mapping[str, Any]],
    blocked_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_counts: dict[str, int] = {}
    for candidate in blocked_candidates:
        symbol = str(candidate.get("symbol", "UNKNOWN"))
        candidate_counts[symbol] = candidate_counts.get(symbol, 0) + 1
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for row in removed_trade_rows:
        symbol = str(row.get("symbol", "UNKNOWN"))
        by_symbol.setdefault(symbol, []).append(row)
    output = []
    for symbol, rows in sorted(by_symbol.items()):
        pnl_values = [_as_optional_float(row.get("pnl")) for row in rows]
        return_values = [_as_optional_float(row.get("net_return")) for row in rows]
        available_pnl = [value for value in pnl_values if value is not None]
        available_returns = [value for value in return_values if value is not None]
        severities = [str(row.get("highest_severity", "UNKNOWN")) for row in rows]
        output.append(
            {
                "symbol": symbol,
                "blocked_trade_count": len(rows),
                "blocked_candidate_count": candidate_counts.get(symbol, 0),
                "matched_categories": "|".join(
                    sorted(
                        {
                            category
                            for row in rows
                            for category in str(row.get("matched_categories", "")).split("|")
                            if category
                        }
                    )
                ),
                "highest_severity": "CATASTROPHIC" if "CATASTROPHIC" in severities else (severities[0] if severities else "UNKNOWN"),
                "available_pnl_contribution": sum(available_pnl) if available_pnl else "UNAVAILABLE_INPUT",
                "available_return_contribution": sum(available_returns) if available_returns else "UNAVAILABLE_INPUT",
                "limitations": "ledger-level attribution only; portfolio path was not recomputed",
            }
        )
    return output

def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "UNAVAILABLE_INPUT", "UNKNOWN"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _mapping_first(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None

def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None

def _metric(payload: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    return None
