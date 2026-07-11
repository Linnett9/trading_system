"""Provider-independent readiness and incremental coverage report contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from core.research.ml.stock_level.news_sources.canonical import PROVIDER_READINESS_SCHEMA_VERSION
from core.research.ml.stock_level.news_sources.normalization import normalize_symbol


class ReadinessState(str, Enum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NOT_READY = "NOT_READY"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    NOT_ENABLED = "NOT_ENABLED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNAVAILABLE_INPUT = "UNAVAILABLE_INPUT"
    NOT_RUN = "NOT_RUN"
    # Backwards-compatible aliases retained for callers from the corrective pass.
    UNAVAILABLE = "UNAVAILABLE"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class UniverseReconciliation:
    requested_symbol_count: int
    provider_symbol_count: int
    overlapping_symbol_count: int
    missing_from_provider: tuple[str, ...]
    provider_only_symbols: tuple[str, ...]
    duplicate_requested_symbol_count: int = 0
    duplicate_provider_symbol_count: int = 0
    schema_version: str = PROVIDER_READINESS_SCHEMA_VERSION
    requested_news_symbols: tuple[str, ...] = ()
    collected_news_symbols: tuple[str, ...] = ()
    price_model_symbols: tuple[str, ...] = ()
    collected_news_symbol_count: int = 0
    price_model_symbol_count: int = 0
    collected_price_intersection: tuple[str, ...] = ()
    collected_price_intersection_count: int = 0
    price_model_symbols_missing_from_collected_news: tuple[str, ...] = ()
    collected_news_symbols_absent_from_price_model: tuple[str, ...] = ()
    requested_news_symbols_not_collected: tuple[str, ...] = ()
    collected_news_symbols_not_requested: tuple[str, ...] = ()
    price_model_coverage_percentage: float | None = None
    collected_news_coverage_percentage: float | None = None
    requested_vs_collected_coverage_percentage: float | None = None
    collected_news_is_strict_subset_of_price_universe: bool | None = None
    requested_news_is_strict_subset_of_price_universe: bool | None = None
    duplicate_collected_symbol_count: int = 0
    duplicate_price_model_symbol_count: int = 0
    requested_input_state: ReadinessState = ReadinessState.READY
    collected_input_state: ReadinessState = ReadinessState.READY
    price_model_input_state: ReadinessState = ReadinessState.UNAVAILABLE_INPUT
    denominator_zero_policy: str = "coverage_percentage_is_none_when_denominator_is_zero"


@dataclass(frozen=True)
class ProviderCoverageSnapshot:
    provider: str
    symbols_requested: int
    symbols_with_records: int
    story_symbol_rows: int
    unique_provider_articles: int
    schema_version: str = PROVIDER_READINESS_SCHEMA_VERSION
    readiness_state: ReadinessState = ReadinessState.NOT_RUN
    requested_symbols: tuple[str, ...] = ()
    symbols_with_rows: tuple[str, ...] = ()
    requested_start_date: str | None = None
    requested_end_date: str | None = None
    earliest_published_at_utc: str | None = None
    latest_published_at_utc: str | None = None
    raw_row_count: int | None = None
    unique_provider_story_count: int | None = None
    canonical_story_count: int | None = None
    headline_coverage_ratio: float | None = None
    summary_coverage_ratio: float | None = None
    body_coverage_ratio: float | None = None
    timestamp_coverage_ratio: float | None = None
    provider_available_at_coverage_ratio: float | None = None
    missing_required_fields: tuple[str, ...] = ()
    future_timestamp_count: int | None = None
    out_of_window_count: int | None = None
    exact_duplicate_count: int | None = None
    likely_duplicate_group_count: int | None = None
    relevance_counts: Mapping[str, int] = field(default_factory=dict)
    single_symbol_count: int | None = None
    multi_symbol_count: int | None = None
    source_distribution: Mapping[str, int] = field(default_factory=dict)
    publisher_distribution: Mapping[str, int] = field(default_factory=dict)
    language_distribution: Mapping[str, int] = field(default_factory=dict)
    provider_error_status: ReadinessState = ReadinessState.NOT_RUN
    entitlement_status: ReadinessState = ReadinessState.NOT_RUN
    rate_limit_status: ReadinessState = ReadinessState.NOT_RUN
    safe_for_canonical_ingest: bool = False
    safe_for_feature_generation: bool = False
    safe_for_text_baseline: bool = False
    safe_for_finbert_inference: bool = False
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    availability_state: ReadinessState = ReadinessState.NOT_RUN
    provider_available_at_state: ReadinessState = ReadinessState.UNAVAILABLE_INPUT
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncrementalCoverageComparison:
    baseline_provider: str
    candidate_provider: str
    overlapping_symbols: int
    candidate_only_symbols: int
    baseline_only_symbols: int
    candidate_incremental_story_rows: int | None = None
    candidate_incremental_unique_articles: int | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderReadinessReport:
    schema_version: str
    generated_at_utc: str
    baseline_provider: str | None
    provider_snapshots: tuple[ProviderCoverageSnapshot, ...]
    incremental_comparisons: tuple[IncrementalCoverageComparison, ...] = ()
    universe_reconciliation: UniverseReconciliation | None = None
    tfidf_readiness: ReadinessState = ReadinessState.NOT_RUN
    finbert_readiness: ReadinessState = ReadinessState.NOT_RUN
    text_model_notes: tuple[str, ...] = ()
    readiness_flags: Mapping[str, bool] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def build_provider_readiness_report(
    *,
    generated_at_utc: str,
    provider_snapshots: tuple[ProviderCoverageSnapshot, ...],
    baseline_provider: str | None = None,
    incremental_comparisons: tuple[IncrementalCoverageComparison, ...] = (),
    universe_reconciliation: UniverseReconciliation | None = None,
    tfidf_readiness: ReadinessState = ReadinessState.NOT_RUN,
    finbert_readiness: ReadinessState = ReadinessState.NOT_RUN,
    text_model_notes: tuple[str, ...] = (),
    readiness_flags: Mapping[str, bool] | None = None,
    notes: tuple[str, ...] = (),
) -> ProviderReadinessReport:
    """Build a report shell that can compare current and future providers."""

    return ProviderReadinessReport(
        schema_version=PROVIDER_READINESS_SCHEMA_VERSION,
        generated_at_utc=generated_at_utc,
        baseline_provider=baseline_provider,
        provider_snapshots=provider_snapshots,
        incremental_comparisons=incremental_comparisons,
        universe_reconciliation=universe_reconciliation,
        tfidf_readiness=tfidf_readiness,
        finbert_readiness=finbert_readiness,
        text_model_notes=text_model_notes,
        readiness_flags=readiness_flags or {},
        notes=notes,
    )


def derive_readiness_state(
    *,
    enabled: bool = True,
    implemented: bool = True,
    input_available: bool = True,
    sufficient_data: bool = True,
    blockers: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> ReadinessState:
    """Derive a non-contradictory readiness state from passive report inputs."""

    if not implemented:
        return ReadinessState.NOT_IMPLEMENTED
    if not enabled:
        return ReadinessState.NOT_ENABLED
    if not input_available:
        return ReadinessState.UNAVAILABLE_INPUT
    if not sufficient_data:
        return ReadinessState.INSUFFICIENT_DATA
    if blockers:
        return ReadinessState.NOT_READY
    if warnings:
        return ReadinessState.READY_WITH_WARNINGS
    return ReadinessState.READY


def reconcile_universes(
    requested_symbols: tuple[str, ...] | list[str] | None,
    provider_symbols: tuple[str, ...] | list[str] | None,
    price_model_symbols: tuple[str, ...] | list[str] | None = None,
) -> UniverseReconciliation:
    """Compare runtime-supplied universes without reading or collecting data."""

    requested_state = ReadinessState.UNAVAILABLE_INPUT if requested_symbols is None else ReadinessState.READY
    collected_state = ReadinessState.UNAVAILABLE_INPUT if provider_symbols is None else ReadinessState.READY
    price_state = ReadinessState.UNAVAILABLE_INPUT if price_model_symbols is None else ReadinessState.READY
    requested_symbols = [] if requested_symbols is None else list(requested_symbols)
    provider_symbols = [] if provider_symbols is None else list(provider_symbols)
    price_model_symbols = [] if price_model_symbols is None else list(price_model_symbols)
    requested = _normalized_unique(requested_symbols)
    provider = _normalized_unique(provider_symbols)
    price = _normalized_unique(price_model_symbols)
    requested_all = [normalize_symbol(str(value)) or "" for value in requested_symbols if str(value).strip()]
    provider_all = [normalize_symbol(str(value)) or "" for value in provider_symbols if str(value).strip()]
    price_all = [normalize_symbol(str(value)) or "" for value in price_model_symbols if str(value).strip()]
    requested_set = set(requested)
    provider_set = set(provider)
    price_set = set(price)
    collected_price_intersection = tuple(sorted(provider_set & price_set))
    price_missing = tuple(sorted(price_set - provider_set))
    provider_only_vs_price = tuple(sorted(provider_set - price_set))
    requested_not_collected = tuple(sorted(requested_set - provider_set))
    collected_not_requested = tuple(sorted(provider_set - requested_set))
    return UniverseReconciliation(
        requested_symbol_count=len(requested),
        provider_symbol_count=len(provider),
        overlapping_symbol_count=len(requested_set & provider_set),
        missing_from_provider=tuple(sorted(requested_set - provider_set)),
        provider_only_symbols=tuple(sorted(provider_set - requested_set)),
        duplicate_requested_symbol_count=len(requested_all) - len(requested),
        duplicate_provider_symbol_count=len(provider_all) - len(provider),
        requested_news_symbols=tuple(sorted(requested)),
        collected_news_symbols=tuple(sorted(provider)),
        price_model_symbols=tuple(sorted(price)),
        collected_news_symbol_count=len(provider),
        price_model_symbol_count=len(price),
        collected_price_intersection=collected_price_intersection,
        collected_price_intersection_count=len(collected_price_intersection),
        price_model_symbols_missing_from_collected_news=price_missing,
        collected_news_symbols_absent_from_price_model=provider_only_vs_price,
        requested_news_symbols_not_collected=requested_not_collected,
        collected_news_symbols_not_requested=collected_not_requested,
        price_model_coverage_percentage=_percentage(len(collected_price_intersection), len(price)),
        collected_news_coverage_percentage=_percentage(len(collected_price_intersection), len(provider)),
        requested_vs_collected_coverage_percentage=_percentage(len(requested_set & provider_set), len(requested)),
        collected_news_is_strict_subset_of_price_universe=_strict_subset(provider_set, price_set, price_state),
        requested_news_is_strict_subset_of_price_universe=_strict_subset(requested_set, price_set, price_state),
        duplicate_collected_symbol_count=len(provider_all) - len(provider),
        duplicate_price_model_symbol_count=len(price_all) - len(price),
        requested_input_state=requested_state,
        collected_input_state=collected_state,
        price_model_input_state=price_state,
    )


def build_text_model_readiness_report(
    *,
    tfidf_ready: bool = False,
    finbert_ready: bool = False,
    tfidf_state: ReadinessState | None = None,
    finbert_state: ReadinessState | None = None,
    headline_coverage: float | None = None,
    summary_coverage: float | None = None,
    full_text_coverage: float | None = None,
    language_coverage: float | None = None,
    duplicate_group_coverage: float | None = None,
    relevance_label_coverage: float | None = None,
    event_label_coverage: float | None = None,
    chronological_split_ready: bool = False,
    point_in_time_timestamp_ready: bool = False,
    estimated_usable_headline_count: int | None = None,
    estimated_usable_summary_count: int | None = None,
    estimated_labeled_count: int | None = None,
    notes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Report-only text-model readiness; does not train or load models."""

    blockers: list[str] = []
    warnings: list[str] = []
    if not point_in_time_timestamp_ready:
        blockers.append("point_in_time_timestamp_readiness_missing")
    if estimated_usable_headline_count is None:
        blockers.append("usable_headline_count_not_evaluated")
    elif estimated_usable_headline_count == 0:
        blockers.append("no_usable_headlines")
    if estimated_labeled_count is None:
        warnings.append("label_stage_not_run")
    elif estimated_labeled_count == 0:
        blockers.append("no_reliable_labels")
    if relevance_label_coverage is None:
        warnings.append("relevance_label_coverage_not_run")
    elif relevance_label_coverage <= 0.0:
        blockers.append("relevance_labels_missing")
    if event_label_coverage is None:
        warnings.append("event_label_coverage_not_run")
    elif event_label_coverage <= 0.0:
        blockers.append("event_labels_missing")
    if not chronological_split_ready:
        blockers.append("chronological_split_not_ready")

    safe_for_tfidf = bool(point_in_time_timestamp_ready and (estimated_usable_headline_count or 0) > 0)
    safe_for_finbert_inference = bool(
        point_in_time_timestamp_ready
        and ((estimated_usable_headline_count or 0) > 0 or (estimated_usable_summary_count or 0) > 0)
    )
    safe_for_finbert_fine_tuning = bool(
        safe_for_finbert_inference
        and chronological_split_ready
        and (estimated_labeled_count or 0) > 0
        and (relevance_label_coverage or 0.0) > 0.0
        and (event_label_coverage or 0.0) > 0.0
    )
    tfidf = tfidf_state or (ReadinessState.READY if tfidf_ready or safe_for_tfidf else ReadinessState.NOT_READY)
    finbert = finbert_state or (ReadinessState.READY if finbert_ready or safe_for_finbert_inference else ReadinessState.NOT_READY)
    return {
        "schema_name": "stock_alpha_news_text_model_readiness",
        "schema_version": 1,
        "headline_coverage": headline_coverage,
        "summary_coverage": summary_coverage,
        "full_text_coverage": full_text_coverage,
        "language_coverage": language_coverage,
        "duplicate_group_coverage": duplicate_group_coverage,
        "relevance_label_coverage": relevance_label_coverage,
        "event_label_coverage": event_label_coverage,
        "chronological_split_ready": chronological_split_ready,
        "point_in_time_timestamp_ready": point_in_time_timestamp_ready,
        "estimated_usable_headline_count": estimated_usable_headline_count,
        "estimated_usable_summary_count": estimated_usable_summary_count,
        "estimated_labeled_count": estimated_labeled_count,
        "safe_for_tfidf_baseline": safe_for_tfidf,
        "safe_for_finbert_inference": safe_for_finbert_inference,
        "safe_for_finbert_fine_tuning": safe_for_finbert_fine_tuning,
        "tfidf_readiness": tfidf.value,
        "finbert_readiness": finbert.value,
        "bert_readiness": finbert.value,
        "tfidf_training_started": False,
        "finbert_training_started": False,
        "finbert_inference_started": False,
        "transformer_dependency_added": False,
        "research_only": True,
        "trading_impact": "none",
        "blockers": blockers,
        "warnings": warnings,
        "notes": list(notes),
    }


def _normalized_unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = normalize_symbol(str(value)) or ""
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return tuple(result)


def _percentage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _strict_subset(left: set[str], right: set[str], right_state: ReadinessState) -> bool | None:
    if right_state == ReadinessState.UNAVAILABLE_INPUT:
        return None
    return bool(left) and left < right
