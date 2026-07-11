from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.news_sources.corpus_composition_smoke import (
    CORPUS_COMPOSITION_SMOKE_SCHEMA_VERSION,
    write_corpus_composition_smoke_report,
)
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)


def test_good_rows_produce_expected_nested_outputs_under_report_dir(tmp_path: Path) -> None:
    report_dir = tmp_path / "composition"

    report, paths = write_corpus_composition_smoke_report(
        [_row("AAPL", provider_article_id="1"), _row("MSFT", provider_article_id="2")],
        report_dir,
        sample_size=2,
    )

    assert paths.report_json_path == report_dir / "composition_smoke_report.json"
    assert paths.summary_markdown_path == report_dir / "composition_smoke_summary.md"
    assert paths.sample_selection_dir == report_dir / "sample_selection"
    assert paths.readiness_dir == report_dir / "readiness"
    assert paths.corpus_dir == report_dir / "corpus"
    for path in (
        paths.report_json_path,
        paths.summary_markdown_path,
        paths.sample_selection_dir / "corpus_sample_rows.json",
        paths.sample_selection_dir / "corpus_sample_selection_audit.json",
        paths.sample_selection_dir / "corpus_sample_selection_summary.md",
        paths.readiness_dir / "corpus_readiness_audit.json",
        paths.readiness_dir / "corpus_readiness_audit.md",
        paths.corpus_dir / "corpus_rows.jsonl",
        paths.corpus_dir / "corpus_manifest.json",
        paths.corpus_dir / "corpus_summary.md",
    ):
        assert path.exists()
        path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))

    persisted_report = json.loads(paths.report_json_path.read_text(encoding="utf-8"))
    assert persisted_report == report
    assert report["schema_version"] == CORPUS_COMPOSITION_SMOKE_SCHEMA_VERSION
    assert report["artifact_type"] == "sample_corpus_composition_smoke"
    assert report["input_row_count"] == 2
    assert report["selected_row_count"] == 2
    assert report["corpus_row_count"] == 2
    assert report["skipped_row_count"] == 0
    assert report["sample_excluded_row_count"] == 0
    assert report["sample_skip_reasons"] == {}
    assert report["corpus_skip_reasons"] == {}
    assert report["sample_selection_recommendation"] == "READY_FOR_COMPOSITION_SMOKE"
    assert report["readiness_recommendation"] == "READY_FOR_TINY_CORPUS_DRY_RUN"
    assert set(report["sample_output_files"]) == {"sample_rows_json", "audit_json", "summary_markdown"}
    assert set(report["readiness_output_files"]) == {"audit_json", "markdown"}
    assert set(report["corpus_output_files"]) == {"corpus_rows_jsonl", "manifest_json", "summary_markdown"}
    assert "Corpus rows: 2" in paths.summary_markdown_path.read_text(encoding="utf-8")


def test_mixed_rows_keep_deterministic_excluded_and_skipped_counts(tmp_path: Path) -> None:
    report, paths = write_corpus_composition_smoke_report(
        [
            _row("MSFT", provider_article_id="2", published_at_utc="2024-01-02T00:00:00Z"),
            _row("AAPL", provider_article_id="1", published_at_utc="2024-01-01T00:00:00Z"),
            _row("TSLA", provider_article_id="3") | {"published_at_utc": ""},
            _row("NVDA", provider_article_id="4"),
        ],
        tmp_path / "composition",
        sample_size=2,
    )

    sample_audit = json.loads(
        (paths.sample_selection_dir / "corpus_sample_selection_audit.json").read_text(encoding="utf-8")
    )
    sample_rows = json.loads((paths.sample_selection_dir / "corpus_sample_rows.json").read_text(encoding="utf-8"))
    corpus_rows = [
        json.loads(line)
        for line in (paths.corpus_dir / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [row["symbol"] for row in sample_rows] == ["AAPL", "MSFT"]
    assert [row["symbol"] for row in corpus_rows] == ["AAPL", "MSFT"]
    assert report["selected_row_count"] == 2
    assert report["corpus_row_count"] == 2
    assert report["skipped_row_count"] == 0
    assert report["sample_excluded_row_count"] == 2
    assert report["sample_skip_reasons"] == {
        "missing_publication_timestamp": 1,
        "not_selected_sample_size_limit": 1,
    }
    assert sample_audit["excluded_rows"] == [
        {
            "row_number": 3,
            "provider": "alpaca_benzinga",
            "symbol": "TSLA",
            "provider_article_id": "3",
            "reasons": ["missing_publication_timestamp"],
        },
        {
            "row_number": 4,
            "provider": "alpaca_benzinga",
            "symbol": "NVDA",
            "provider_article_id": "4",
            "reasons": ["not_selected_sample_size_limit"],
        },
    ]


def test_protected_active_backfill_output_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="protected active backfill"):
        write_corpus_composition_smoke_report(
            [_row("AAPL")],
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "composition",
            sample_size=1,
        )


def test_sec_form_type_alone_does_not_become_economic_event_type(tmp_path: Path) -> None:
    report, paths = write_corpus_composition_smoke_report(
        [
            _row("AAPL")
            | {
                "provider": "sec_company_filings",
                "source": "sec",
                "source_type": "sec_filing",
                "form_type": "8-K",
                "event_type": "",
            }
        ],
        tmp_path / "composition",
        sample_size=1,
    )

    sample_rows = json.loads((paths.sample_selection_dir / "corpus_sample_rows.json").read_text(encoding="utf-8"))
    corpus_rows = [
        json.loads(line)
        for line in (paths.corpus_dir / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["selected_row_count"] == 1
    assert sample_rows[0]["source_type"] == "SEC_FILING"
    assert sample_rows[0]["event_type"] is None
    assert corpus_rows[0]["source_type"] == "SEC_FILING"
    assert corpus_rows[0]["event_type"] is None


def test_no_provider_config_network_backfill_features_models_or_brokers_are_needed(tmp_path: Path) -> None:
    report, _paths = write_corpus_composition_smoke_report(
        [_row("AAPL")],
        tmp_path / "composition",
        sample_size=1,
    )

    assert report["safety_flags"] == {
        "provider_collection_invoked": False,
        "network_invoked": False,
        "canonical_ingest_invoked": False,
        "historical_backfill_invoked": False,
        "corpus_assembly_invoked": False,
        "feature_generation_invoked": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "trading_impact": "none",
        "protected_active_backfill_path_rejected": True,
    }


def _row(
    symbol: str,
    *,
    provider_article_id: str = "1",
    published_at_utc: str = "2024-01-02T19:30:00Z",
) -> dict[str, str]:
    return {
        "article_id": f"alpaca_benzinga:{provider_article_id}:{symbol}",
        "provider": "alpaca_benzinga",
        "provider_article_id": provider_article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": published_at_utc,
        "provider_available_at_utc": "2024-01-02T19:31:00Z",
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "source": "benzinga",
        "source_type": "newswire",
        "headline": f"{symbol} headline",
        "summary": "short summary",
        "body_or_full_text": "full body",
        "language": "en",
        "event_type": "",
        "relevance_status": "",
    }
