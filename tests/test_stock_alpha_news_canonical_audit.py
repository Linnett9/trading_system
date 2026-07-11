from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.stock_level.news_sources.canonical import CANONICAL_NEWS_SCHEMA_VERSION
from core.research.ml.stock_level.news_sources.canonical_audit import (
    CANONICAL_CONVERSION_AUDIT_SCHEMA_VERSION,
    build_canonical_conversion_audit,
    write_canonical_conversion_audit,
)


def test_canonical_conversion_audit_writes_outputs_under_supplied_directory(tmp_path: Path) -> None:
    rows = [
        _compatibility_row(
            provider="alpaca_benzinga",
            symbol="aapl",
            provider_article_id="101",
            event_type="earnings",
        )
    ]
    report_dir = tmp_path / "scratch" / "canonical_audit"

    paths = write_canonical_conversion_audit(rows, report_dir, artifact_uri="manual-fixture.json")

    assert paths.audit_json_path.parent == report_dir
    assert paths.canonical_rows_json_path.parent == report_dir
    assert paths.markdown_path.parent == report_dir
    assert paths.audit_json_path.is_file()
    assert paths.canonical_rows_json_path.is_file()
    assert paths.markdown_path.is_file()

    payload = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    canonical_rows = json.loads(paths.canonical_rows_json_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == CANONICAL_CONVERSION_AUDIT_SCHEMA_VERSION
    assert payload["audit_type"] == "manual_compatibility_rows_to_canonical_dry_run"
    assert payload["row_count"] == 1
    assert payload["converted_row_count"] == 1
    assert payload["canonical_schema_version"] == CANONICAL_NEWS_SCHEMA_VERSION
    assert payload["providers"] == ["alpaca_benzinga"]
    assert payload["symbols"] == ["AAPL"]
    assert payload["event_type_count"] == 1
    assert payload["provider_collection_invoked"] is False
    assert payload["network_invoked"] is False
    assert payload["canonical_ingest_invoked"] is False
    assert payload["feature_generation_invoked"] is False
    assert payload["model_training_invoked"] is False
    assert payload["trading_impact"] == "none"
    assert set(payload["output_files"]) == {"audit_json", "canonical_rows_json", "markdown"}
    assert canonical_rows[0]["schema_version"] == CANONICAL_NEWS_SCHEMA_VERSION
    assert canonical_rows[0]["symbol"] == "AAPL"
    assert canonical_rows[0]["event_type"] == "earnings"
    assert canonical_rows[0]["provenance"]["artifact_uri"] == "manual-fixture.json"
    assert canonical_rows[0]["source_type"] == "NEWSWIRE"
    assert "Rows converted: 1" in paths.markdown_path.read_text(encoding="utf-8")


def test_canonical_conversion_audit_does_not_require_provider_config_or_keys(tmp_path: Path) -> None:
    rows = [_compatibility_row(provider="manual_fixture", symbol="MSFT", provider_article_id="manual-1")]

    paths = write_canonical_conversion_audit(rows, tmp_path / "reports")

    payload = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    assert payload["converted_row_count"] == 1
    assert payload["provider_collection_invoked"] is False
    assert payload["network_invoked"] is False


def test_canonical_conversion_audit_keeps_sec_form_type_separate_from_event_type(tmp_path: Path) -> None:
    rows = [
        _compatibility_row(
            provider="sec_company_filings",
            symbol="AAPL",
            provider_article_id="0000320193-24-000001",
            source_type="sec_filing",
            form_type="8-K",
            event_type="",
        )
    ]

    paths = write_canonical_conversion_audit(rows, tmp_path / "sec-audit")
    payload = json.loads(paths.audit_json_path.read_text(encoding="utf-8"))
    canonical_rows = json.loads(paths.canonical_rows_json_path.read_text(encoding="utf-8"))

    assert payload["event_type_count"] == 0
    assert payload["form_type_only_count"] == 1
    assert payload["row_diagnostics"][0]["form_type_only"] is True
    assert canonical_rows[0]["source_type"] == "SEC_FILING"
    assert canonical_rows[0]["event_type"] is None
    assert canonical_rows[0]["provenance"]["extra"]["raw_provider_values"]["form_type"] == "8-K"


def test_canonical_conversion_audit_reports_missing_fields_deterministically() -> None:
    payload, canonical_rows = build_canonical_conversion_audit(
        [
            {
                "provider": "manual_fixture",
                "headline": "Missing required audit fields",
            }
        ]
    )

    assert payload["row_count"] == 1
    assert payload["converted_row_count"] == 1
    assert payload["missing_publication_timestamp_count"] == 1
    assert payload["missing_provider_article_id_count"] == 1
    assert payload["missing_symbol_count"] == 1
    assert payload["row_diagnostics"] == [
        {
            "row_number": 1,
            "provider": "manual_fixture",
            "symbol": "",
            "provider_article_id": "",
            "missing_publication_timestamp": True,
            "missing_provider_article_id": True,
            "missing_symbol": True,
            "event_type": "",
            "form_type": "",
            "form_type_only": False,
            "converted": True,
        }
    ]
    assert canonical_rows[0]["published_at_utc"] is None
    assert canonical_rows[0]["provider_article_id"] is None
    assert canonical_rows[0]["symbol"] == ""


def _compatibility_row(
    *,
    provider: str,
    symbol: str,
    provider_article_id: str,
    source_type: str = "newswire",
    form_type: str = "",
    event_type: str = "",
) -> dict[str, str]:
    return {
        "article_id": f"{provider}:{provider_article_id}:{symbol}",
        "provider": provider,
        "provider_article_id": provider_article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": "2024-01-02T19:30:00Z",
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "source": provider,
        "source_type": source_type,
        "headline": f"{symbol} test headline",
        "body_or_summary": "summary",
        "form_type": form_type,
        "event_type": event_type,
    }
