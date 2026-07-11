"""Deterministic symbol-relevance audit scaffolding for news records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from core.research.ml.stock_level.news_sources.canonical import (
    CanonicalNewsRecord,
    RELEVANCE_AUDIT_SCHEMA_VERSION,
)
from core.research.ml.stock_level.news_sources.normalization import normalize_symbol


class RelevanceStatus(str, Enum):
    DIRECT_COMPANY_NEWS = "DIRECT_COMPANY_NEWS"
    INDIRECT_COMPANY_OR_INDUSTRY_NEWS = "INDIRECT_COMPANY_OR_INDUSTRY_NEWS"
    MACRO_OR_MARKET_NEWS = "MACRO_OR_MARKET_NEWS"
    WEAK_PROVIDER_TAG = "WEAK_PROVIDER_TAG"
    IRRELEVANT_TAG = "IRRELEVANT_TAG"
    NOT_EVALUATED = "NOT_EVALUATED"
    UNAVAILABLE_INPUT = "UNAVAILABLE_INPUT"


@dataclass(frozen=True)
class RelevanceEvidence:
    """Observable evidence, separated from verified labels or model outputs."""

    ticker_in_headline: bool
    company_name_in_headline: bool
    ticker_in_summary: bool
    company_name_in_summary: bool
    provider_symbol_count: int
    is_single_symbol_story: bool
    provider_relevance_score: float | None = None
    headline_subject_differs_from_symbol: bool | None = None
    source_type: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelevanceAudit:
    schema_version: str
    story_symbol_id: str
    heuristic_status: RelevanceStatus
    evidence: RelevanceEvidence
    human_reviewed_label: RelevanceStatus | None = None
    model_predicted_label: RelevanceStatus | None = None
    model_score: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


def build_relevance_evidence(
    record: CanonicalNewsRecord,
    *,
    company_name_by_symbol: Mapping[str, str] | None = None,
    provider_relevance_score: float | None = None,
) -> RelevanceEvidence:
    """Build deterministic evidence without claiming ground-truth relevance."""

    symbol = normalize_symbol(record.symbol) or ""
    company_name = None
    if company_name_by_symbol:
        company_name = company_name_by_symbol.get(symbol) or company_name_by_symbol.get(record.symbol)
    headline = (record.headline or "").casefold()
    summary = (record.summary or "").casefold()
    ticker_token = symbol.casefold()
    company_token = company_name.casefold() if company_name else None
    provider_symbol_count = len(tuple(record.provider_symbols))
    return RelevanceEvidence(
        ticker_in_headline=_contains_token(headline, ticker_token),
        company_name_in_headline=bool(company_token and company_token in headline),
        ticker_in_summary=_contains_token(summary, ticker_token),
        company_name_in_summary=bool(company_token and company_token in summary),
        provider_symbol_count=provider_symbol_count,
        is_single_symbol_story=provider_symbol_count == 1,
        provider_relevance_score=provider_relevance_score,
        source_type=record.source_type.value,
    )


def heuristic_relevance_status(evidence: RelevanceEvidence) -> RelevanceStatus:
    """Return a conservative heuristic status for triage only."""

    if evidence.provider_symbol_count <= 0:
        return RelevanceStatus.UNAVAILABLE_INPUT
    if (
        evidence.ticker_in_headline
        or evidence.company_name_in_headline
        or evidence.ticker_in_summary
        or evidence.company_name_in_summary
    ):
        return RelevanceStatus.DIRECT_COMPANY_NEWS
    if evidence.provider_symbol_count > 20:
        return RelevanceStatus.MACRO_OR_MARKET_NEWS
    if evidence.provider_symbol_count > 5:
        return RelevanceStatus.WEAK_PROVIDER_TAG
    return RelevanceStatus.NOT_EVALUATED


def build_relevance_audit(
    record: CanonicalNewsRecord,
    *,
    company_name_by_symbol: Mapping[str, str] | None = None,
    provider_relevance_score: float | None = None,
) -> RelevanceAudit:
    evidence = build_relevance_evidence(
        record,
        company_name_by_symbol=company_name_by_symbol,
        provider_relevance_score=provider_relevance_score,
    )
    return RelevanceAudit(
        schema_version=RELEVANCE_AUDIT_SCHEMA_VERSION,
        story_symbol_id=record.story_symbol_id,
        heuristic_status=heuristic_relevance_status(evidence),
        evidence=evidence,
    )


def _contains_token(text: str, token: str) -> bool:
    if not text or not token:
        return False
    padded = f" {text.replace('.', ' ').replace(',', ' ')} "
    return f" {token.casefold()} " in padded
