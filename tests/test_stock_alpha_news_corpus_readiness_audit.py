from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.stock_level.news_sources.corpus_readiness_audit import (
    CORPUS_READINESS_AUDIT_SCHEMA_VERSION,
    NEEDS_TEXT,
    NEEDS_TIMESTAMPS,
    READY_FOR_TINY_CORPUS_DRY_RUN,
    build_corpus_readiness_audit,
    write_corpus_readiness_audit,
)


def test_corpus_readiness_audit_ready_for_tiny_dry_run_and_writes_under_report_dir(tmp_path: Path) -> None:
    report_dir = tmp_path / "scratch" / "corpus-readiness"

    paths = write_corpus_readiness_audit([_row("AAPL", event_type="earnings", relevance_status="DIRECT")], report_dir)

    assert paths.audit_json_path.parent == report_dir
    assert paths.markdown_path.parent == report_dir
    assert paths.audit_json_path.name == "corpus_readiness_audit.json"
    assert paths.markdown_path.name == "corpus_readiness_audit.md"

    payload = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CORPUS_READINESS_AUDIT_SCHEMA_VERSION
    assert payload["audit_type"] == "sample_news_corpus_readiness_dry_run"
    assert payload["row_count"] == 1
    assert payload["usable_text_row_count"] == 1
    assert payload["headline_coverage"] == 1.0
    assert payload["summary_coverage"] == 1.0
    assert payload["body_coverage"] == 1.0
    assert payload["any_text_coverage"] == 1.0
    assert payload["publication_timestamp_coverage"] == 1.0
    assert payload["point_in_time_safe_timestamp_count"] == 1
    assert payload["symbol_coverage"] == 1.0
    assert payload["provider_coverage"] == 1.0
    assert payload["event_type_coverage"] == 1.0
    assert payload["relevance_label_coverage"] == 1.0
    assert payload["blockers"] == []
    assert payload["recommendation"] == READY_FOR_TINY_CORPUS_DRY_RUN
    assert payload["provider_collection_invoked"] is False
    assert payload["network_invoked"] is False
    assert payload["canonical_ingest_invoked"] is False
    assert payload["corpus_assembly_invoked"] is False
    assert payload["feature_generation_invoked"] is False
    assert payload["model_training_invoked"] is False
    assert payload["model_inference_invoked"] is False
    assert payload["trading_impact"] == "none"
    assert set(payload["output_files"]) == {"audit_json", "markdown"}
    assert "Recommendation: READY_FOR_TINY_CORPUS_DRY_RUN" in paths.markdown_path.read_text(encoding="utf-8")


def test_corpus_readiness_audit_without_text_reports_text_blocker() -> None:
    row = _row("AAPL") | {"headline": "", "summary": "", "body_or_summary": "", "body_or_full_text": ""}

    payload = build_corpus_readiness_audit([row])

    assert payload["usable_text_row_count"] == 0
    assert payload["any_text_coverage"] == 0.0
    assert "no_usable_text" in payload["blockers"]
    assert payload["recommendation"] == NEEDS_TEXT


def test_corpus_readiness_audit_without_publication_timestamp_reports_timestamp_blocker() -> None:
    row = _row("AAPL") | {"published_at_utc": ""}

    payload = build_corpus_readiness_audit([row])

    assert payload["publication_timestamp_coverage"] == 0.0
    assert payload["point_in_time_safe_timestamp_count"] == 0
    assert "publication_timestamps_missing" in payload["blockers"]
    assert payload["recommendation"] == NEEDS_TIMESTAMPS


def test_corpus_readiness_audit_missing_labels_warns_without_blocking_tiny_corpus() -> None:
    payload = build_corpus_readiness_audit([_row("AAPL")])

    assert payload["event_type_coverage"] == 0.0
    assert payload["relevance_label_coverage"] == 0.0
    assert "event_labels_missing_for_supervised_modeling" in payload["warnings"]
    assert "relevance_labels_missing_for_supervised_modeling" in payload["warnings"]
    assert payload["blockers"] == []
    assert payload["recommendation"] == READY_FOR_TINY_CORPUS_DRY_RUN


def test_corpus_readiness_audit_does_not_treat_sec_form_type_as_event_type() -> None:
    payload = build_corpus_readiness_audit(
        [
            _row("MSFT")
            | {
                "provider": "sec_company_filings",
                "source_type": "sec_filing",
                "form_type": "8-K",
                "event_type": "",
            }
        ]
    )

    assert payload["event_type_coverage"] == 0.0
    assert payload["row_diagnostics"][0]["has_event_type"] is False
    assert payload["recommendation"] == READY_FOR_TINY_CORPUS_DRY_RUN


def test_corpus_readiness_audit_accepts_canonical_rows_without_reconversion() -> None:
    canonical_row = {
        "schema_version": "stock_alpha_news.canonical.v1",
        "provider": "manual_provider",
        "symbol": "NVDA",
        "published_at_utc": "2024-01-02T19:30:00Z",
        "provider_available_at_utc": None,
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "headline": "NVDA headline",
        "summary": "",
        "body_or_full_text": "",
        "event_type": "",
    }

    payload = build_corpus_readiness_audit([canonical_row], rows_are_canonical=True)

    assert payload["row_count"] == 1
    assert payload["usable_text_row_count"] == 1
    assert payload["provider_coverage"] == 1.0
    assert payload["symbol_coverage"] == 1.0
    assert payload["recommendation"] == READY_FOR_TINY_CORPUS_DRY_RUN


def _row(symbol: str, *, event_type: str = "", relevance_status: str = "") -> dict[str, str]:
    return {
        "article_id": f"alpaca_benzinga:1:{symbol}",
        "provider": "alpaca_benzinga",
        "provider_article_id": "1",
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": "2024-01-02T19:30:00Z",
        "provider_available_at_utc": "2024-01-02T19:31:00Z",
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "source": "benzinga",
        "source_type": "newswire",
        "headline": f"{symbol} headline",
        "summary": "short summary",
        "body_or_summary": "short summary",
        "body_or_full_text": "full body",
        "language": "en",
        "duplicate_group_id": "duplicate-1",
        "event_type": event_type,
        "relevance_status": relevance_status,
    }
