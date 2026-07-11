from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.news_sources.corpus_dry_run import build_corpus_dry_run
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    CORPUS_SAMPLE_SELECTOR_SCHEMA_VERSION,
    PROTECTED_ACTIVE_BACKFILL_PATH,
    build_corpus_sample_selection,
    write_corpus_sample_selection,
)


def test_in_memory_rows_produce_deterministic_selected_sample_and_audit() -> None:
    rows = [
        _row("AAPL", provider_article_id="3", published_at_utc="2024-01-03T00:00:00Z"),
        _row("MSFT", provider_article_id="2", published_at_utc="2024-01-02T00:00:00Z"),
        _row("AAPL", provider_article_id="1", published_at_utc="2024-01-01T00:00:00Z"),
    ]

    audit, selected_rows = build_corpus_sample_selection(rows, sample_size=2)

    assert audit["schema_version"] == CORPUS_SAMPLE_SELECTOR_SCHEMA_VERSION
    assert audit["artifact_type"] == "corpus_sample_selection"
    assert audit["input_row_count"] == 3
    assert audit["eligible_row_count"] == 3
    assert audit["selected_row_count"] == 2
    assert audit["excluded_row_count"] == 1
    assert audit["sample_size"] == 2
    assert audit["selection_strategy"] == "deterministic_round_robin_by_symbol_provider_then_publication"
    assert audit["skip_reasons"] == {"not_selected_sample_size_limit": 1}
    assert audit["symbols"] == ["AAPL", "MSFT"]
    assert audit["providers"] == ["alpaca_benzinga"]
    assert audit["start_published_at_utc"] == "2024-01-01T00:00:00Z"
    assert audit["end_published_at_utc"] == "2024-01-02T00:00:00Z"
    assert audit["input_source_type"] == "in_memory"
    assert audit["input_path"] is None
    assert audit["safety_flags"]["provider_collection_invoked"] is False
    assert audit["safety_flags"]["network_invoked"] is False
    assert audit["safety_flags"]["historical_backfill_invoked"] is False
    assert audit["safety_flags"]["feature_generation_invoked"] is False
    assert audit["safety_flags"]["model_training_invoked"] is False
    assert audit["safety_flags"]["model_inference_invoked"] is False
    assert audit["safety_flags"]["trading_impact"] == "none"
    assert [row["sample_row_id"] for row in selected_rows] == [
        "corpus-sample-row-000001",
        "corpus-sample-row-000002",
    ]
    assert [row["symbol"] for row in selected_rows] == ["AAPL", "MSFT"]
    assert [row["published_at_utc"] for row in selected_rows] == [
        "2024-01-01T00:00:00Z",
        "2024-01-02T00:00:00Z",
    ]


def test_jsonl_fixture_can_be_loaded_explicitly_and_sampled(tmp_path: Path) -> None:
    input_path = tmp_path / "fixture.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in [_row("AAPL"), _row("MSFT")]) + "\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports" / "sample"

    paths = write_corpus_sample_selection(None, report_dir, input_path=input_path, sample_size=1)

    assert paths.sample_rows_json_path.parent == report_dir
    assert paths.audit_json_path.parent == report_dir
    assert paths.summary_markdown_path.parent == report_dir
    assert paths.sample_rows_json_path.name == "corpus_sample_rows.json"
    assert paths.audit_json_path.name == "corpus_sample_selection_audit.json"
    assert paths.summary_markdown_path.name == "corpus_sample_selection_summary.md"

    selected_rows = json.loads(paths.sample_rows_json_path.read_text(encoding="utf-8"))
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))

    assert len(selected_rows) == 1
    assert selected_rows[0]["symbol"] == "AAPL"
    assert audit["input_source_type"] == "jsonl_file"
    assert audit["input_path"] == str(input_path)
    assert audit["output_files"] == {
        "sample_rows_json": str(paths.sample_rows_json_path),
        "audit_json": str(paths.audit_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    assert "Selected rows: 1" in paths.summary_markdown_path.read_text(encoding="utf-8")


def test_missing_required_fields_are_excluded_with_clear_reasons() -> None:
    rows = [
        _row("AAPL") | {"headline": "", "summary": "", "body_or_full_text": ""},
        _row("MSFT") | {"published_at_utc": ""},
        _row("") | {"symbol": "", "provider_symbols": ""},
        _row("NVDA") | {"provider": "", "source": ""},
    ]

    audit, selected_rows = build_corpus_sample_selection(rows, sample_size=10)

    assert selected_rows == []
    assert audit["eligible_row_count"] == 0
    assert audit["selected_row_count"] == 0
    assert audit["excluded_row_count"] == 4
    assert audit["skip_reasons"] == {
        "missing_provider": 1,
        "missing_publication_timestamp": 1,
        "missing_symbol": 1,
        "missing_text": 1,
    }
    assert [row["reasons"] for row in audit["excluded_rows"]] == [
        ["missing_text"],
        ["missing_publication_timestamp"],
        ["missing_symbol"],
        ["missing_provider"],
    ]


def test_sample_size_is_enforced_and_outputs_remain_usable_for_dry_run(tmp_path: Path) -> None:
    paths = write_corpus_sample_selection(
        [_row("AAPL"), _row("MSFT"), _row("NVDA")],
        tmp_path / "sample",
        sample_size=2,
    )

    selected_rows = json.loads(paths.sample_rows_json_path.read_text(encoding="utf-8"))
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    dry_run_manifest, dry_run_rows = build_corpus_dry_run(selected_rows, rows_are_canonical=True)

    assert len(selected_rows) == 2
    assert audit["selected_row_count"] == 2
    assert audit["excluded_row_count"] == 1
    assert audit["skip_reasons"] == {"not_selected_sample_size_limit": 1}
    assert dry_run_manifest["corpus_row_count"] == 2
    assert [row["corpus_row_id"] for row in dry_run_rows] == [
        "corpus-row-000001",
        "corpus-row-000002",
    ]


def test_protected_active_backfill_path_is_rejected_for_input() -> None:
    with pytest.raises(ValueError, match="protected active backfill"):
        write_corpus_sample_selection(
            None,
            Path("/private/tmp/phase5-sample-output"),
            input_path=Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "fixture.jsonl",
        )


def test_protected_active_backfill_path_is_rejected_for_output() -> None:
    with pytest.raises(ValueError, match="protected active backfill"):
        write_corpus_sample_selection(
            [_row("AAPL")],
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "sample-output",
            sample_size=1,
        )


def test_no_provider_config_api_keys_network_backfill_features_or_models_are_needed(tmp_path: Path) -> None:
    paths = write_corpus_sample_selection([_row("AAPL")], tmp_path / "sample", sample_size=1)
    audit = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))

    assert audit["safety_flags"] == {
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


def test_sec_form_type_alone_does_not_become_economic_event_type() -> None:
    audit, selected_rows = build_corpus_sample_selection(
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

    assert audit["selected_row_count"] == 1
    assert selected_rows[0]["source_type"] == "SEC_FILING"
    assert selected_rows[0]["event_type"] is None


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
