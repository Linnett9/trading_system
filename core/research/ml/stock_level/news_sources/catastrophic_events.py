"""Deterministic catastrophic-news taxonomy for research-only audits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


UNKNOWN = "UNKNOWN"
UNAVAILABLE_INPUT = "UNAVAILABLE_INPUT"


@dataclass(frozen=True)
class CatastrophicNewsCategory:
    category_id: str
    severity: str
    blocks_contrarian_entry: bool
    requires_manual_review: bool
    keywords: tuple[str, ...]
    regex_patterns: tuple[str, ...]
    example_phrases: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "severity": self.severity,
            "blocks_contrarian_entry": self.blocks_contrarian_entry,
            "requires_manual_review": self.requires_manual_review,
            "keywords": list(self.keywords),
            "regex_patterns": list(self.regex_patterns),
            "example_phrases": list(self.example_phrases),
            "warnings": list(self.warnings),
        }


CATASTROPHIC_NEWS_TAXONOMY: tuple[CatastrophicNewsCategory, ...] = (
    CatastrophicNewsCategory(
        "bankruptcy_or_administration",
        "CATASTROPHIC",
        True,
        True,
        ("bankruptcy", "administration", "chapter 11", "chapter 7"),
        (r"\bfiles?\s+for\s+bankruptcy\b", r"\benters?\s+administration\b"),
        ("files for bankruptcy", "enters administration"),
        ("Potential insolvency event; never treat as routine negative news.",),
    ),
    CatastrophicNewsCategory(
        "insolvency_or_liquidation",
        "CATASTROPHIC",
        True,
        True,
        ("insolvent", "insolvency", "liquidation", "liquidator"),
        (r"\binsolvenc(?:y|ies)\b", r"\bappoints?\s+liquidators?\b"),
        ("company is insolvent", "appoints liquidators"),
        ("Potential terminal-capital-structure event.",),
    ),
    CatastrophicNewsCategory(
        "going_concern_warning",
        "SEVERE",
        True,
        True,
        ("going concern", "substantial doubt"),
        (r"\bgoing\s+concern\b", r"\bsubstantial\s+doubt\b"),
        ("going concern warning", "substantial doubt about ability to continue"),
        ("May indicate existential funding risk.",),
    ),
    CatastrophicNewsCategory(
        "delisting_or_trading_suspension",
        "SEVERE",
        True,
        True,
        ("delisting", "trading suspension", "suspended trading"),
        (r"\bdelist(?:ing|ed)?\b", r"\btrading\s+(?:is\s+)?(?:suspended|suspension)\b"),
        ("receives delisting notice", "trading suspended"),
        ("Price history can be discontinuous or unavailable.",),
    ),
    CatastrophicNewsCategory(
        "auditor_resignation_or_qualified_audit",
        "SEVERE",
        True,
        True,
        ("auditor resignation", "qualified audit", "adverse opinion"),
        (r"\bauditor\s+resign(?:s|ation)\b", r"\bqualified\s+audit\b", r"\badverse\s+opinion\b"),
        ("auditor resigns", "qualified audit opinion"),
        ("Financial statements may be unreliable.",),
    ),
    CatastrophicNewsCategory(
        "fraud_or_accounting_irregularity",
        "CATASTROPHIC",
        True,
        True,
        ("fraud", "accounting irregularity", "misstatement", "restatement"),
        (r"\bfraud(?:ulent)?\b", r"\baccounting\s+irregularit(?:y|ies)\b", r"\bmaterial\s+misstatement\b"),
        ("fraud investigation", "accounting irregularities"),
        ("Possible non-linear downside and unreliable fundamentals.",),
    ),
    CatastrophicNewsCategory(
        "regulatory_enforcement_or_criminal_probe",
        "SEVERE",
        True,
        True,
        ("criminal probe", "regulatory enforcement", "sec investigation", "doj investigation"),
        (r"\bcriminal\s+probe\b", r"\bregulatory\s+enforcement\b", r"\b(?:sec|doj)\s+investigation\b"),
        ("SEC investigation", "criminal probe"),
        ("Regulatory outcomes can dominate alpha signals.",),
    ),
    CatastrophicNewsCategory(
        "debt_default_or_covenant_breach",
        "SEVERE",
        True,
        True,
        ("debt default", "defaulted", "covenant breach", "breached covenant"),
        (r"\bdebt\s+default\b", r"\bdefault(?:ed|s)?\s+on\s+debt\b", r"\bcovenant\s+breach\b"),
        ("defaults on debt", "covenant breach"),
        ("Credit distress may invalidate contrarian-entry assumptions.",),
    ),
    CatastrophicNewsCategory(
        "emergency_rescue_financing",
        "SEVERE",
        True,
        True,
        ("rescue financing", "emergency financing", "lifeline financing"),
        (r"\brescue\s+financ(?:e|ing)\b", r"\bemergency\s+financ(?:e|ing)\b", r"\blifeline\s+financ(?:e|ing)\b"),
        ("emergency financing", "rescue financing package"),
        ("May indicate near-term survival risk.",),
    ),
    CatastrophicNewsCategory(
        "distressed_dilution_or_deep_discount_raise",
        "SEVERE",
        True,
        True,
        ("deep discount", "distressed raise", "highly dilutive", "dilutive financing"),
        (r"\bdeep\s+discount(?:ed)?\s+(?:raise|offering|financing)\b", r"\bhighly\s+dilutive\b"),
        ("deep discount raise", "highly dilutive financing"),
        ("Share-count shock can overwhelm price-reversion assumptions.",),
    ),
    CatastrophicNewsCategory(
        "winding_up_petition",
        "CATASTROPHIC",
        True,
        True,
        ("winding up petition", "wind-up petition"),
        (r"\bwinding[-\s]up\s+petition\b", r"\bwind[-\s]up\s+petition\b"),
        ("winding-up petition filed",),
        ("Potential forced liquidation event.",),
    ),
    CatastrophicNewsCategory(
        "major_litigation_existential",
        "SEVERE",
        True,
        True,
        ("existential litigation", "mass tort", "major litigation", "class action"),
        (r"\bexistential\s+litigation\b", r"\bmass\s+tort\b", r"\bmajor\s+litigation\b"),
        ("major litigation threatens company", "mass tort exposure"),
        ("Legal exposure may dominate historical return patterns.",),
    ),
)


def catastrophic_news_taxonomy_report() -> dict[str, Any]:
    return {
        "schema": "catastrophic_news_taxonomy_v1",
        "status": "PRESENT",
        "category_count": len(CATASTROPHIC_NEWS_TAXONOMY),
        "categories": [category.as_dict() for category in CATASTROPHIC_NEWS_TAXONOMY],
    }


def classify_catastrophic_news_event(
    *,
    headline: str | None = None,
    summary: str | None = None,
    body: str | None = None,
    event_type: str | None = None,
    provider_category: str | None = None,
    source: str | None = None,
    publication_timestamp: str | None = None,
    availability_timestamp: str | None = None,
    symbol: str | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    text_parts = [headline, summary, body, event_type, provider_category]
    normalized_text = "\n".join(part.strip() for part in text_parts if isinstance(part, str) and part.strip()).lower()
    warnings: list[str] = []
    if not normalized_text:
        warnings.append("UNAVAILABLE_INPUT: no headline, summary, body, event type, or provider category text was available.")
        return {
            "candidate_id": candidate_id or UNKNOWN,
            "symbol": symbol or UNKNOWN,
            "matched": False,
            "matched_categories": [],
            "highest_severity": UNKNOWN,
            "blocks_contrarian_entry": False,
            "requires_manual_review": True,
            "matched_terms": [],
            "matched_patterns": [],
            "classification_method": UNAVAILABLE_INPUT,
            "availability_timestamp_present": bool(availability_timestamp),
            "point_in_time_safe": False,
            "warnings": warnings,
            "source": source or UNKNOWN,
            "publication_timestamp": publication_timestamp or UNKNOWN,
            "availability_timestamp": availability_timestamp or UNKNOWN,
        }

    matched_categories: list[str] = []
    matched_terms: list[str] = []
    matched_patterns: list[str] = []
    severities: list[str] = []
    blocks_entry = False
    manual_review = False

    for category in CATASTROPHIC_NEWS_TAXONOMY:
        category_matched = False
        for keyword in category.keywords:
            if keyword.lower() in normalized_text:
                matched_terms.append(keyword)
                category_matched = True
        for pattern in category.regex_patterns:
            if re.search(pattern, normalized_text, flags=re.IGNORECASE):
                matched_patterns.append(pattern)
                category_matched = True
        if category_matched:
            matched_categories.append(category.category_id)
            severities.append(category.severity)
            blocks_entry = blocks_entry or category.blocks_contrarian_entry
            manual_review = manual_review or category.requires_manual_review
            warnings.extend(category.warnings)

    if not availability_timestamp:
        warnings.append("UNAVAILABLE_INPUT: availability timestamp is required for point-in-time safety.")

    matched = bool(matched_categories)
    return {
        "candidate_id": candidate_id or UNKNOWN,
        "symbol": symbol or UNKNOWN,
        "matched": matched,
        "matched_categories": matched_categories,
        "highest_severity": _highest_severity(severities),
        "blocks_contrarian_entry": blocks_entry,
        "requires_manual_review": manual_review or not availability_timestamp,
        "matched_terms": sorted(set(matched_terms)),
        "matched_patterns": sorted(set(matched_patterns)),
        "classification_method": "DETERMINISTIC_TAXONOMY" if matched else "NO_MATCH_DETERMINISTIC_TAXONOMY",
        "availability_timestamp_present": bool(availability_timestamp),
        "point_in_time_safe": bool(availability_timestamp),
        "warnings": sorted(set(warnings)),
        "source": source or UNKNOWN,
        "publication_timestamp": publication_timestamp or UNKNOWN,
        "availability_timestamp": availability_timestamp or UNKNOWN,
    }


def classify_catastrophic_news_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        classify_catastrophic_news_event(
            headline=_first(row, "headline_text", "headline", "title"),
            summary=_first(row, "summary_text", "summary", "description"),
            body=_first(row, "body_text", "body", "content", "article_body"),
            event_type=_first(row, "event_type", "event"),
            provider_category=_first(row, "provider_category", "category"),
            source=_first(row, "source", "provider"),
            publication_timestamp=_first(row, "publication_timestamp", "published_at_utc", "published_at", "timestamp"),
            availability_timestamp=_first(row, "availability_timestamp", "available_at", "asof_timestamp"),
            symbol=_first(row, "symbol", "ticker"),
            candidate_id=_first(row, "candidate_id", "trade_id", "row_id"),
        )
        for row in rows
    ]


def _highest_severity(severities: Sequence[str]) -> str:
    if "CATASTROPHIC" in severities:
        return "CATASTROPHIC"
    if "SEVERE" in severities:
        return "SEVERE"
    return "NONE"


def _first(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None
