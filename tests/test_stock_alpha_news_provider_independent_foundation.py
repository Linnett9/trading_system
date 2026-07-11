from __future__ import annotations

from core.research.ml.stock_level.news_sources.canonical import (
    CANONICAL_NEWS_SCHEMA_VERSION,
    CanonicalNewsRecord,
    SourceType,
    canonical_from_compatibility_row,
    field_state,
    FieldAvailability,
)
from core.research.ml.stock_level.news_sources.deduplication import group_news_records
from core.research.ml.stock_level.news_sources.normalization import (
    format_utc_timestamp,
    normalize_headline,
    normalize_language,
    normalize_source_name,
    normalize_symbol,
    normalize_url,
    normalize_url_pair,
    normalize_whitespace,
    parse_utc_timestamp,
    validate_utc_timestamp,
)
from core.research.ml.stock_level.news_sources.readiness import (
    ProviderCoverageSnapshot,
    ReadinessState,
    build_text_model_readiness_report,
    build_provider_readiness_report,
    derive_readiness_state,
    reconcile_universes,
)
from core.research.ml.stock_level.news_sources.registry import (
    news_source_planning_registry,
    reconcile_news_source_registry,
)
from core.research.ml.stock_level.news_sources.relevance import (
    RelevanceStatus,
    build_relevance_audit,
)


def test_normalization_helpers_preserve_unknowns_and_identity_boundaries() -> None:
    assert normalize_whitespace(" A\n\n headline\t ") == "A headline"
    assert normalize_source_name("benzinga") == "Benzinga"
    assert normalize_headline("  Apple   Rallies ") == "apple rallies"
    assert normalize_language("EN_us") == "en"
    assert normalize_symbol(" brk.b ") == "BRK.B"
    assert (
        normalize_url("HTTPS://Example.COM:443/path?utm_source=x&b=2&a=1#fragment")
        == "https://example.com/path?a=1&b=2"
    )
    assert parse_utc_timestamp(None) is None
    assert format_utc_timestamp(parse_utc_timestamp("2020-01-02T03:04:05-05:00")) == "2020-01-02T08:04:05Z"


def test_url_pair_preserves_original_and_derives_normalized_url() -> None:
    original = "HTTPS://Example.COM/path?utm_source=x&b=2&a=1#fragment"

    raw_url, normalized_url = normalize_url_pair(original)

    assert raw_url == original
    assert normalized_url == "https://example.com/path?a=1&b=2"


def test_timestamp_parser_rejects_naive_publication_times() -> None:
    try:
        parse_utc_timestamp("2020-01-02T03:04:05")
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:
        raise AssertionError("naive timestamp should be rejected")


def test_timestamp_validation_result_reports_invalid_without_substitution() -> None:
    result = validate_utc_timestamp("2020-01-02T03:04:05")

    assert result.valid is False
    assert result.parsed_at_utc is None
    assert "timezone" in result.reason


def test_field_state_distinguishes_missing_empty_and_zero() -> None:
    assert field_state(None, field_present=False).availability == FieldAvailability.MISSING_FIELD
    assert field_state(None, provider_available=False).availability == FieldAvailability.UNAVAILABLE_FROM_PROVIDER
    assert field_state("").availability == FieldAvailability.EMPTY_VALUE
    assert field_state(()).availability == FieldAvailability.EMPTY_VALUE
    assert field_state(0).availability == FieldAvailability.ZERO_VALUE
    assert field_state(False).availability == FieldAvailability.PRESENT
    assert field_state(None, was_run=False).availability == FieldAvailability.NOT_RUN


def test_compatibility_rows_adapt_to_derived_canonical_contract_without_availability_substitution() -> None:
    record = canonical_from_compatibility_row(
        {
            "article_id": "alpaca_benzinga:101:AAPL",
            "symbol": "aapl",
            "published_at_utc": "2024-01-02T19:30:00Z",
            "ingested_at": "2026-07-10T00:00:00Z",
            "source": "benzinga",
            "headline": "Apple headline",
            "body_or_summary": "summary",
            "provider": "alpaca_benzinga",
            "provider_article_id": "101",
            "provider_symbols": "AAPL,MSFT",
            "collected_at_utc": "2026-07-10T00:00:00Z",
        },
        artifact_uri="raw.csv",
        row_number=7,
    )

    assert record.schema_version == CANONICAL_NEWS_SCHEMA_VERSION
    assert record.symbol == "AAPL"
    assert record.provider_symbols == ("AAPL", "MSFT")
    assert record.provider_available_at_utc is None
    assert record.collected_at_utc == "2026-07-10T00:00:00Z"
    assert record.normalized_provider_url is None
    assert record.provenance.artifact_uri == "raw.csv"
    assert record.provenance.row_number == 7
    assert record.provenance.extra["raw_provider_values"]["ingested_at"] == "2026-07-10T00:00:00Z"


def test_canonical_conversion_keeps_sec_form_type_separate_from_event_type() -> None:
    record = canonical_from_compatibility_row(
        {
            "article_id": "sec_company_filings:0000320193-24-000001:AAPL",
            "symbol": "AAPL",
            "provider": "sec_company_filings",
            "provider_article_id": "0000320193-24-000001",
            "published_at_utc": "2024-01-02T19:30:00Z",
            "source_type": "sec_filing",
            "form_type": "8-K",
        }
    )

    assert record.source_type == SourceType.SEC_FILING
    assert record.event_type is None
    assert record.provenance.extra["raw_provider_values"]["form_type"] == "8-K"


def test_grouping_retains_cross_symbol_story_relationships() -> None:
    records = (
        _record("AAPL", provider_article_id="1"),
        _record("MSFT", provider_article_id="1"),
    )

    grouped = group_news_records(records, method="cross_symbol_story")

    assert len(grouped) == 2
    assert grouped[0].duplicate.canonical_duplicate_group_id == grouped[1].duplicate.canonical_duplicate_group_id
    assert grouped[0].duplicate.duplicate_group_size == 2
    assert grouped[0].duplicate.symbols == ("AAPL", "MSFT")
    assert grouped[0].duplicate.provider_article_ids == ("1",)


def test_likely_publication_grouping_keeps_symbols_separate() -> None:
    records = (
        _record("AAPL", provider_article_id="1"),
        _record("MSFT", provider_article_id="1"),
    )

    grouped = group_news_records(records, method="likely_publication")

    assert len({item.duplicate.canonical_duplicate_group_id for item in grouped}) == 2


def test_grouping_is_non_destructive_and_preserves_revision_metadata() -> None:
    records = (
        _record("AAPL", provider_article_id="1"),
        _record("AAPL", provider_article_id="2", updated_at_utc="2020-01-02T10:04:05Z"),
    )

    grouped = group_news_records(records, method="likely_publication")

    assert [item.record.provider_article_id for item in grouped] == ["1", "2"]
    assert grouped[0].duplicate.duplicate_group_size == 2
    assert grouped[0].duplicate.earliest_publication_timestamp == "2020-01-02T08:04:05Z"
    assert grouped[0].duplicate.latest_update_timestamp == "2020-01-02T10:04:05Z"
    assert grouped[0].duplicate.provider_article_ids == ("1", "2")
    assert grouped[0].duplicate.grouping_version.startswith("stock_alpha_news.story_grouping")


def test_relevance_audit_separates_evidence_from_labels() -> None:
    audit = build_relevance_audit(
        _record("AAPL", headline="Apple announces new capital return plan"),
        company_name_by_symbol={"AAPL": "Apple"},
    )

    assert audit.heuristic_status == RelevanceStatus.DIRECT_COMPANY_NEWS
    assert audit.evidence.company_name_in_headline is True
    assert audit.human_reviewed_label is None
    assert audit.model_predicted_label is None


def test_relevance_audit_uses_unavailable_input_for_missing_provider_symbols() -> None:
    record = _record("AAPL", provider_symbols=())

    audit = build_relevance_audit(record)

    assert audit.heuristic_status == RelevanceStatus.UNAVAILABLE_INPUT
    assert audit.evidence.provider_symbol_count == 0


def test_provider_readiness_report_contract() -> None:
    reconciliation = reconcile_universes(
        ["AAPL", "MSFT", "AAPL"],
        ["AAPL", "NVDA"],
        ["AAPL", "MSFT", "TSLA"],
    )
    report = build_provider_readiness_report(
        generated_at_utc="2026-07-10T00:00:00Z",
        baseline_provider="alpaca_benzinga",
        provider_snapshots=(
            ProviderCoverageSnapshot(
                provider="alpaca_benzinga",
                symbols_requested=200,
                symbols_with_records=200,
                story_symbol_rows=316664,
                unique_provider_articles=244701,
                availability_state=ReadinessState.READY,
                provider_available_at_state=ReadinessState.UNAVAILABLE_INPUT,
                safe_for_canonical_ingest=False,
                blockers=("provider_available_at_unavailable",),
            ),
        ),
        universe_reconciliation=reconciliation,
        tfidf_readiness=ReadinessState.NOT_RUN,
        finbert_readiness=ReadinessState.NOT_RUN,
        readiness_flags={"raw_artifacts_immutable": True},
    )

    assert report.baseline_provider == "alpaca_benzinga"
    assert report.provider_snapshots[0].unique_provider_articles == 244701
    assert report.provider_snapshots[0].provider_available_at_state == ReadinessState.UNAVAILABLE_INPUT
    assert report.provider_snapshots[0].safe_for_canonical_ingest is False
    assert report.provider_snapshots[0].blockers == ("provider_available_at_unavailable",)
    assert report.universe_reconciliation is not None
    assert report.universe_reconciliation.missing_from_provider == ("MSFT",)
    assert report.universe_reconciliation.provider_only_symbols == ("NVDA",)
    assert report.universe_reconciliation.price_model_symbols_missing_from_collected_news == ("MSFT", "TSLA")
    assert report.universe_reconciliation.collected_news_symbols_absent_from_price_model == ("NVDA",)
    assert report.universe_reconciliation.price_model_coverage_percentage == 1 / 3
    assert report.universe_reconciliation.requested_vs_collected_coverage_percentage == 0.5
    assert report.universe_reconciliation.duplicate_requested_symbol_count == 1
    assert report.tfidf_readiness == ReadinessState.NOT_RUN
    assert report.finbert_readiness == ReadinessState.NOT_RUN
    assert report.readiness_flags["raw_artifacts_immutable"] is True


def test_readiness_state_derivation_prevents_ready_with_blockers() -> None:
    assert derive_readiness_state(blockers=("missing_timestamp",)) == ReadinessState.NOT_READY
    assert derive_readiness_state(warnings=("low_summary_coverage",)) == ReadinessState.READY_WITH_WARNINGS
    assert derive_readiness_state(enabled=False) == ReadinessState.NOT_ENABLED
    assert derive_readiness_state(implemented=False) == ReadinessState.NOT_IMPLEMENTED
    assert derive_readiness_state(input_available=False) == ReadinessState.UNAVAILABLE_INPUT
    assert derive_readiness_state(sufficient_data=False) == ReadinessState.INSUFFICIENT_DATA


def test_universe_reconciliation_handles_equal_disjoint_empty_and_unavailable_inputs() -> None:
    equal = reconcile_universes(["MSFT", "AAPL"], ["AAPL", "MSFT"], ["AAPL", "MSFT"])
    disjoint = reconcile_universes(["AAPL"], ["MSFT"], ["TSLA"])
    empty = reconcile_universes([], [], [])
    unavailable = reconcile_universes(None, None, None)

    assert equal.requested_news_symbols == ("AAPL", "MSFT")
    assert equal.collected_news_is_strict_subset_of_price_universe is False
    assert disjoint.collected_price_intersection_count == 0
    assert disjoint.price_model_coverage_percentage == 0.0
    assert empty.price_model_coverage_percentage is None
    assert empty.denominator_zero_policy == "coverage_percentage_is_none_when_denominator_is_zero"
    assert unavailable.requested_input_state == ReadinessState.UNAVAILABLE_INPUT
    assert unavailable.collected_input_state == ReadinessState.UNAVAILABLE_INPUT
    assert unavailable.price_model_input_state == ReadinessState.UNAVAILABLE_INPUT


def test_universe_reconciliation_strict_subset_flags_are_dynamic() -> None:
    reconciliation = reconcile_universes(["AAPL"], ["AAPL"], ["AAPL", "MSFT"])

    assert reconciliation.collected_news_is_strict_subset_of_price_universe is True
    assert reconciliation.requested_news_is_strict_subset_of_price_universe is True
    assert reconciliation.price_model_symbols_missing_from_collected_news == ("MSFT",)


def test_text_model_readiness_report_is_report_only() -> None:
    report = build_text_model_readiness_report(
        headline_coverage=1.0,
        summary_coverage=0.25,
        full_text_coverage=0.0,
        language_coverage=1.0,
        duplicate_group_coverage=1.0,
        relevance_label_coverage=None,
        event_label_coverage=None,
        chronological_split_ready=False,
        point_in_time_timestamp_ready=True,
        estimated_usable_headline_count=10,
        estimated_usable_summary_count=2,
        estimated_labeled_count=None,
        notes=("coverage audit pending",),
    )

    assert report["schema_name"] == "stock_alpha_news_text_model_readiness"
    assert report["safe_for_tfidf_baseline"] is True
    assert report["safe_for_finbert_inference"] is True
    assert report["safe_for_finbert_fine_tuning"] is False
    assert report["tfidf_readiness"] == "READY"
    assert report["finbert_readiness"] == "READY"
    assert "label_stage_not_run" in report["warnings"]
    assert "chronological_split_not_ready" in report["blockers"]
    assert report["tfidf_training_started"] is False
    assert report["finbert_inference_started"] is False
    assert report["transformer_dependency_added"] is False


def test_registry_reconciliation_extends_existing_rss_registry_for_non_rss_planning() -> None:
    report = reconcile_news_source_registry(
        universe_symbols=("AAPL", "MSFT", "NVDA"),
        registry_classifications={
            "AAPL": "verified_rss_feed",
            "MSFT": "sec_company_filings_candidate",
            "NVDA": "free_source_candidate",
        },
    )

    assert report["registry_complete"] is True
    assert report["supports_non_rss_source_planning"] is True
    assert report["classification_counts"]["sec_company_filings_candidate"] == 1


def test_source_planning_registry_defaults_all_sources_disabled() -> None:
    registry = news_source_planning_registry()

    assert {"alpaca_benzinga", "sec_edgar", "company_ir_or_rss", "financial_modeling_prep", "alpha_vantage", "gdelt"} <= set(registry)
    assert all(plan["collection_enabled"] is False for plan in registry.values())
    assert all(plan["canonical_ingest_enabled"] is False for plan in registry.values())
    assert all(plan["feature_generation_enabled"] is False for plan in registry.values())
    assert registry["sec_edgar"]["authoritative"] is True
    assert registry["alpaca_benzinga"]["editorial"] is True


def _record(
    symbol: str,
    *,
    provider_article_id: str = "article-1",
    headline: str = "Apple and Microsoft move on AI headline",
    updated_at_utc: str = "2020-01-02T09:04:05Z",
    provider_symbols: tuple[str, ...] = ("AAPL", "MSFT"),
) -> CanonicalNewsRecord:
    return CanonicalNewsRecord(
        schema_version=CANONICAL_NEWS_SCHEMA_VERSION,
        canonical_story_id="story-1",
        story_symbol_id=f"story-1:{symbol}",
        provider="alpaca_benzinga",
        provider_article_id=provider_article_id,
        provider_original_article_id=provider_article_id,
        provider_symbols=provider_symbols,
        symbol=symbol,
        published_at_utc="2020-01-02T08:04:05Z",
        provider_available_at_utc=None,
        updated_at_utc=updated_at_utc,
        collected_at_utc="2026-07-10T00:00:00Z",
        headline=headline,
        source="benzinga",
        source_type=SourceType.NEWSWIRE,
        provider_url="https://example.com/story",
    )
