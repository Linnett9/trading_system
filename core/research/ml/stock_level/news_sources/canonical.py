"""Provider-independent contracts for derived stock-alpha news records.

``stock_alpha_news_contract.py`` owns the existing compatibility rows used by
feature generation. This module owns derived canonical records only: adapters
may read compatibility/provider rows, but canonicalization must preserve raw
provenance and must not rewrite provider artifacts in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from core.research.ml.stock_level.news_sources.normalization import (
    format_utc_timestamp,
    normalize_source_name,
    normalize_symbol,
    normalize_url,
    parse_utc_timestamp,
)


CANONICAL_NEWS_SCHEMA_VERSION = "stock_alpha_news.canonical.v1"
STORY_GROUPING_VERSION = "stock_alpha_news.story_grouping.v1"
RELEVANCE_AUDIT_SCHEMA_VERSION = "stock_alpha_news.relevance_audit.v1"
PROVIDER_READINESS_SCHEMA_VERSION = "stock_alpha_news.provider_readiness.v1"


class SourceType(str, Enum):
    """High-level source category for canonical news records."""

    NEWSWIRE = "NEWSWIRE"
    SEC_FILING = "SEC_FILING"
    COMPANY_RSS = "COMPANY_RSS"
    INVESTOR_RELATIONS = "INVESTOR_RELATIONS"
    FREE_NEWS = "FREE_NEWS"
    MARKET_DATA_PROVIDER = "MARKET_DATA_PROVIDER"
    UNKNOWN = "UNKNOWN"


class FieldAvailability(str, Enum):
    """Distinguishes values that are often collapsed accidentally."""

    PRESENT = "PRESENT"
    MISSING_FIELD = "MISSING_FIELD"
    UNAVAILABLE_FROM_PROVIDER = "UNAVAILABLE_FROM_PROVIDER"
    EMPTY_VALUE = "EMPTY_VALUE"
    ZERO_VALUE = "ZERO_VALUE"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class FieldState:
    """A value plus an explicit availability state."""

    value: Any
    availability: FieldAvailability


@dataclass(frozen=True)
class RawNewsProvenance:
    """Reference to immutable provider artifacts used to derive a record."""

    artifact_uri: str | None = None
    row_number: int | None = None
    partition_id: str | None = None
    raw_record_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalNewsRecord:
    """Derived story-to-symbol news record.

    Raw provider artifacts remain immutable. Timestamps are intentionally
    separate: collection time must never be substituted for publication or
    provider availability time.
    """

    schema_version: str
    canonical_story_id: str
    story_symbol_id: str
    provider: str
    provider_article_id: str | None
    provider_original_article_id: str | None
    provider_symbols: tuple[str, ...]
    symbol: str
    published_at_utc: str | None
    provider_available_at_utc: str | None
    updated_at_utc: str | None
    collected_at_utc: str | None
    headline: str | None
    summary: str | None = None
    body_or_full_text: str | None = None
    source: str | None = None
    source_type: SourceType = SourceType.UNKNOWN
    delivery_provider: str | None = None
    original_source: str | None = None
    publisher: str | None = None
    author: str | None = None
    provider_url: str | None = None
    normalized_provider_url: str | None = None
    language: str | None = None
    historical_availability_note: str | None = None
    duplicate_group_id: str | None = None
    relevance_status: str | None = None
    relevance_evidence: Mapping[str, Any] = field(default_factory=dict)
    heuristic_relevance_status: str | None = None
    human_reviewed_relevance_label: str | None = None
    model_predicted_relevance_label: str | None = None
    event_type: str | None = None
    provenance: RawNewsProvenance = field(default_factory=RawNewsProvenance)


def field_state(
    value: Any,
    *,
    field_present: bool = True,
    provider_available: bool = True,
    was_run: bool = True,
) -> FieldState:
    """Return an explicit availability state without rewriting the value."""

    if not was_run:
        return FieldState(value=None, availability=FieldAvailability.NOT_RUN)
    if not field_present:
        return FieldState(value=None, availability=FieldAvailability.MISSING_FIELD)
    if not provider_available:
        return FieldState(value=None, availability=FieldAvailability.UNAVAILABLE_FROM_PROVIDER)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return FieldState(value=value, availability=FieldAvailability.ZERO_VALUE)
    if value == "" or value == () or value == [] or value == {}:
        return FieldState(value=value, availability=FieldAvailability.EMPTY_VALUE)
    if value is None:
        return FieldState(value=None, availability=FieldAvailability.MISSING_FIELD)
    return FieldState(value=value, availability=FieldAvailability.PRESENT)


def canonical_from_compatibility_row(
    row: Mapping[str, Any],
    *,
    artifact_uri: str | None = None,
    row_number: int | None = None,
    partition_id: str | None = None,
) -> CanonicalNewsRecord:
    """Adapt an existing stock-alpha news compatibility row into canonical form.

    ``stock_alpha_news_contract.py`` remains the owner of the serialized
    feature-generation compatibility contract. This adapter creates a derived
    canonical record and preserves raw row references in provenance.
    """

    provider = str(row.get("provider") or row.get("delivery_provider") or row.get("source") or "").strip()
    provider_article_id = _blank_to_none(row.get("provider_article_id") or row.get("article_id"))
    provider_original_article_id = _blank_to_none(row.get("provider_original_article_id") or provider_article_id)
    symbol = normalize_symbol(str(row.get("symbol", ""))) or ""
    published_at = format_utc_timestamp(parse_utc_timestamp(_blank_to_none(row.get("published_at_utc"))))
    updated_at = format_utc_timestamp(parse_utc_timestamp(_blank_to_none(row.get("updated_at_utc"))))
    collected_at = format_utc_timestamp(parse_utc_timestamp(_blank_to_none(row.get("collected_at_utc") or row.get("ingested_at"))))
    provider_available_at = format_utc_timestamp(
        parse_utc_timestamp(_blank_to_none(row.get("provider_available_at_utc") or row.get("available_at_utc")))
    )
    provider_url = _blank_to_none(row.get("provider_url") or row.get("source_url"))
    canonical_story_id = _stable_text(
        provider,
        provider_original_article_id or provider_article_id or "",
        provider_url or "",
        published_at or "",
        str(row.get("headline") or ""),
    )
    story_symbol_id = _stable_text(canonical_story_id, symbol)
    source_type = _source_type(row.get("source_type"))
    return CanonicalNewsRecord(
        schema_version=CANONICAL_NEWS_SCHEMA_VERSION,
        canonical_story_id=canonical_story_id,
        story_symbol_id=story_symbol_id,
        provider=provider,
        provider_article_id=provider_article_id,
        provider_original_article_id=provider_original_article_id,
        provider_symbols=_provider_symbols(row.get("provider_symbols"), fallback_symbol=symbol),
        symbol=symbol,
        published_at_utc=published_at,
        provider_available_at_utc=provider_available_at,
        updated_at_utc=updated_at,
        collected_at_utc=collected_at,
        headline=_blank_to_none(row.get("headline") or row.get("headline_or_title")),
        summary=_blank_to_none(row.get("summary") or row.get("body_or_summary")),
        body_or_full_text=_blank_to_none(row.get("body_or_full_text")),
        source=normalize_source_name(_blank_to_none(row.get("source"))),
        source_type=source_type,
        delivery_provider=_blank_to_none(row.get("delivery_provider")),
        original_source=_blank_to_none(row.get("original_source") or row.get("raw_source")),
        publisher=normalize_source_name(_blank_to_none(row.get("publisher"))),
        author=_blank_to_none(row.get("author")),
        provider_url=provider_url,
        normalized_provider_url=normalize_url(provider_url),
        language=_blank_to_none(row.get("language")),
        historical_availability_note=_blank_to_none(row.get("historical_availability_note")),
        duplicate_group_id=_blank_to_none(row.get("duplicate_group_id")),
        relevance_status=_blank_to_none(row.get("relevance_status")),
        relevance_evidence={},
        heuristic_relevance_status=_blank_to_none(row.get("heuristic_relevance_status")),
        human_reviewed_relevance_label=_blank_to_none(row.get("human_reviewed_relevance_label")),
        model_predicted_relevance_label=_blank_to_none(row.get("model_predicted_relevance_label")),
        event_type=_blank_to_none(row.get("event_type")),
        provenance=RawNewsProvenance(
            artifact_uri=artifact_uri,
            row_number=row_number,
            partition_id=partition_id,
            raw_record_id=_blank_to_none(row.get("article_id")),
            extra={
                "compatibility_contract": "stock_alpha_news_contract.v1",
                "raw_provider_values": dict(row),
            },
        ),
    )


def _blank_to_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _provider_symbols(value: Any, *, fallback_symbol: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        raw_values = list(value or [])
    symbols = []
    seen = set()
    for raw in raw_values or [fallback_symbol]:
        symbol = normalize_symbol(str(raw)) or ""
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    if not symbols and fallback_symbol:
        symbols.append(fallback_symbol)
    return tuple(symbols)


def _source_type(value: Any) -> SourceType:
    text = str(value or "").strip().lower()
    if text in {"newswire", "editorial_news", "news"}:
        return SourceType.NEWSWIRE
    if text in {"sec_filing", "sec", "sec_edgar", "filing"}:
        return SourceType.SEC_FILING
    if text in {"company_rss", "press_release", "company_press_release_rss"}:
        return SourceType.COMPANY_RSS
    if text in {"investor_relations", "ir"}:
        return SourceType.INVESTOR_RELATIONS
    if text in {"free_news", "gdelt"}:
        return SourceType.FREE_NEWS
    if text in {"market_data_provider", "alpaca_benzinga", "alpha_vantage", "fmp"}:
        return SourceType.MARKET_DATA_PROVIDER
    return SourceType.UNKNOWN


def _stable_text(*parts: str) -> str:
    from hashlib import sha256

    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
