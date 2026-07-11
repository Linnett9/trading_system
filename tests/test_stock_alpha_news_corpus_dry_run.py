from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.stock_level.news_sources.corpus_dry_run import (
    CORPUS_DRY_RUN_SCHEMA_VERSION,
    build_corpus_dry_run,
    write_corpus_dry_run,
)


def test_corpus_dry_run_writes_jsonl_manifest_and_summary_under_report_dir(tmp_path: Path) -> None:
    report_dir = tmp_path / "scratch" / "tiny-corpus"

    paths = write_corpus_dry_run([_row("AAPL", event_type="earnings", relevance_status="DIRECT")], report_dir)

    assert paths.rows_jsonl_path.parent == report_dir
    assert paths.manifest_json_path.parent == report_dir
    assert paths.summary_markdown_path.parent == report_dir
    assert paths.rows_jsonl_path.name == "corpus_rows.jsonl"
    assert paths.manifest_json_path.name == "corpus_manifest.json"
    assert paths.summary_markdown_path.name == "corpus_summary.md"

    manifest = json.loads(paths.manifest_json_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in paths.rows_jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert manifest["schema_version"] == CORPUS_DRY_RUN_SCHEMA_VERSION
    assert manifest["artifact_type"] == "sample_corpus_dry_run"
    assert manifest["input_row_count"] == 1
    assert manifest["corpus_row_count"] == 1
    assert manifest["skipped_row_count"] == 0
    assert manifest["skip_reasons"] == {}
    assert manifest["symbols"] == ["AAPL"]
    assert manifest["providers"] == ["alpaca_benzinga"]
    assert manifest["languages"] == ["en"]
    assert manifest["start_published_at_utc"] == "2024-01-02T19:30:00Z"
    assert manifest["end_published_at_utc"] == "2024-01-02T19:30:00Z"
    assert manifest["readiness_recommendation"] == "READY_FOR_TINY_CORPUS_DRY_RUN"
    assert set(manifest["output_files"]) == {"corpus_rows_jsonl", "manifest_json", "summary_markdown"}
    assert manifest["safety_flags"]["provider_collection_invoked"] is False
    assert manifest["safety_flags"]["network_invoked"] is False
    assert manifest["safety_flags"]["canonical_ingest_invoked"] is False
    assert manifest["safety_flags"]["historical_backfill_invoked"] is False
    assert manifest["safety_flags"]["corpus_assembly_invoked"] is False
    assert manifest["safety_flags"]["feature_generation_invoked"] is False
    assert manifest["safety_flags"]["model_training_invoked"] is False
    assert manifest["safety_flags"]["model_inference_invoked"] is False
    assert manifest["safety_flags"]["trading_impact"] == "none"
    assert rows == [
        {
            "available_at_utc": "2024-01-02T19:31:00Z",
            "body": "full body",
            "canonical_story_id": rows[0]["canonical_story_id"],
            "corpus_row_id": "corpus-row-000001",
            "event_type": "earnings",
            "headline": "AAPL headline",
            "language": "en",
            "provider": "alpaca_benzinga",
            "provider_article_id": "1",
            "published_at_utc": "2024-01-02T19:30:00Z",
            "relevance_label": "DIRECT",
            "source_type": "NEWSWIRE",
            "story_symbol_id": rows[0]["story_symbol_id"],
            "summary": "short summary",
            "symbol": "AAPL",
            "text_field_used": "body",
            "text_for_model": "full body",
        }
    ]
    assert "Corpus rows: 1" in paths.summary_markdown_path.read_text(encoding="utf-8")


def test_corpus_dry_run_skips_rows_with_clear_reasons() -> None:
    rows = [
        _row("AAPL") | {"headline": "", "summary": "", "body_or_summary": "", "body_or_full_text": ""},
        _row("MSFT") | {"published_at_utc": ""},
        _row("") | {"symbol": "", "provider_symbols": ""},
        _row("NVDA") | {"provider": "", "source": ""},
    ]

    manifest, corpus_rows = build_corpus_dry_run(rows)

    assert corpus_rows == []
    assert manifest["corpus_row_count"] == 0
    assert manifest["skipped_row_count"] == 4
    assert manifest["skip_reasons"] == {
        "missing_provider": 1,
        "missing_publication_timestamp": 1,
        "missing_symbol": 1,
        "missing_text": 1,
    }
    assert [row["reasons"] for row in manifest["skipped_rows"]] == [
        ["missing_text"],
        ["missing_publication_timestamp"],
        ["missing_symbol"],
        ["missing_provider"],
    ]


def test_corpus_dry_run_keeps_sec_form_type_separate_from_event_type() -> None:
    manifest, corpus_rows = build_corpus_dry_run(
        [
            _row("AAPL")
            | {
                "provider": "sec_company_filings",
                "source": "sec",
                "source_type": "sec_filing",
                "form_type": "8-K",
                "event_type": "",
            }
        ]
    )

    assert manifest["corpus_row_count"] == 1
    assert corpus_rows[0]["source_type"] == "SEC_FILING"
    assert corpus_rows[0]["event_type"] is None


def test_corpus_dry_run_output_order_is_deterministic() -> None:
    manifest, corpus_rows = build_corpus_dry_run(
        [
            _row("MSFT") | {"provider_article_id": "2", "published_at_utc": "2024-01-03T00:00:00Z"},
            _row("AAPL") | {"provider_article_id": "1", "published_at_utc": "2024-01-02T00:00:00Z"},
        ]
    )

    assert manifest["symbols"] == ["AAPL", "MSFT"]
    assert [row["symbol"] for row in corpus_rows] == ["AAPL", "MSFT"]
    assert [row["corpus_row_id"] for row in corpus_rows] == ["corpus-row-000001", "corpus-row-000002"]


def test_corpus_dry_run_accepts_canonical_rows_without_provider_config_or_keys(tmp_path: Path) -> None:
    canonical_row = {
        "canonical_story_id": "story-1",
        "story_symbol_id": "story-1:TSLA",
        "provider": "manual_provider",
        "provider_article_id": "manual-1",
        "symbol": "TSLA",
        "published_at_utc": "2024-01-02T19:30:00Z",
        "provider_available_at_utc": "",
        "collected_at_utc": "",
        "language": "",
        "headline": "TSLA headline",
        "summary": "",
        "body_or_full_text": "",
        "event_type": "",
        "relevance_status": "",
        "source_type": "FREE_NEWS",
    }

    paths = write_corpus_dry_run([canonical_row], tmp_path / "canonical", rows_are_canonical=True)
    rows = [json.loads(line) for line in paths.rows_jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["canonical_story_id"] == "story-1"
    assert rows[0]["text_field_used"] == "headline"
    assert rows[0]["text_for_model"] == "TSLA headline"
    assert rows[0]["available_at_utc"] is None
    assert rows[0]["event_type"] is None


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
        "event_type": event_type,
        "relevance_status": relevance_status,
    }
