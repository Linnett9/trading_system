from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    CANONICAL_CORPUS_AUDIT_JSON,
    CANONICAL_CORPUS_CSV,
    CANONICAL_CORPUS_MANIFEST_JSON,
    HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
    LEGACY_HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
    HistoricalCanonicalCorpusConfig,
    HistoricalCanonicalCorpusError,
    build_historical_canonical_rows,
    canonical_rows_logical_checksum,
    historical_assembly_row_to_compatibility,
    materialize_historical_canonical_corpus,
    materialize_historical_canonical_corpus_from_config,
    sha256_file,
    verify_canonical_corpus_inventory,
)


def test_historical_assembly_adapter_preserves_provider_and_timestamp_semantics() -> None:
    row = _assembly_row("AAPL", provider_article_id="benzinga-1") | {
        "provider_available_at_utc": "",
        "collected_at_utc": "2026-07-10T12:00:00Z",
    }

    compatibility = historical_assembly_row_to_compatibility(
        row,
        source_assembly_path="frozen/assembly.csv",
        source_assembly_checksum="abc123",
        source_row_number=7,
    )

    assert compatibility["provider"] == "alpaca_benzinga"
    assert compatibility["provider_article_id"] == "benzinga-1"
    assert compatibility["provider_original_article_id"] == "benzinga-1"
    assert compatibility["symbol"] == "AAPL"
    assert compatibility["provider_symbols"] == "AAPL"
    assert compatibility["published_at_utc"] == "2024-01-02T14:30:00Z"
    assert compatibility["provider_available_at_utc"] == ""
    assert compatibility["collected_at_utc"] == "2026-07-10T12:00:00Z"
    assert compatibility["raw_source_artifact"] == "frozen/assembly.csv"
    assert compatibility["raw_source_row_number"] == "7"
    assert compatibility["source_assembly_checksum"] == "abc123"


def test_build_historical_canonical_rows_is_lossless_for_article_symbol_rows(tmp_path: Path) -> None:
    rows = [
        _assembly_row("AAPL", provider_article_id="shared-article") | {"headline": "Shared headline"},
        _assembly_row("MSFT", provider_article_id="shared-article") | {"headline": "Shared headline"},
    ]

    canonical_rows, audit = build_historical_canonical_rows(
        rows,
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="abc123",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )

    assert audit["source_row_count"] == 2
    assert audit["canonical_row_count"] == 2
    assert audit["failed_row_count"] == 0
    assert audit["row_count_reconciled"] is True
    assert [row["symbol"] for row in canonical_rows] == ["AAPL", "MSFT"]
    assert len({row["canonical_story_id"] for row in canonical_rows}) == 1
    assert len({row["story_symbol_id"] for row in canonical_rows}) == 2
    assert canonical_rows[0]["duplicate_group_id"] == canonical_rows[1]["duplicate_group_id"]
    assert canonical_rows[0]["source_assembly_path"] == str(tmp_path / "assembly.csv")
    assert canonical_rows[0]["source_assembly_checksum"] == "abc123"
    assert json.loads(canonical_rows[0]["raw_provider_values"])["provider_article_id"] == "shared-article"


def test_missing_optional_text_is_preserved_with_field_state(tmp_path: Path) -> None:
    rows = [_assembly_row("AAPL", provider_article_id="missing-summary") | {"summary": "", "body_or_full_text": ""}]

    canonical_rows, audit = build_historical_canonical_rows(
        rows,
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="abc123",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )
    field_states = json.loads(canonical_rows[0]["missing_field_states"])

    assert audit["failed_row_count"] == 0
    assert canonical_rows[0]["headline"] == "AAPL headline"
    assert canonical_rows[0]["summary"] == ""
    assert canonical_rows[0]["body_or_full_text"] == ""
    assert field_states["headline"] == "present"
    assert field_states["summary"] == "missing_or_unavailable"
    assert field_states["body_or_full_text"] == "missing_or_unavailable"


def test_missing_required_identity_fails_closed(tmp_path: Path) -> None:
    rows = [_assembly_row("AAPL", provider_article_id="") | {"article_id": ""}]

    canonical_rows, audit = build_historical_canonical_rows(
        rows,
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="abc123",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )

    assert canonical_rows == []
    assert audit["failed_row_count"] == 1
    assert audit["row_count_reconciled"] is False
    assert audit["blocking_rows"][0]["error_message"] == "provider_article_id or article_id is required"


def test_materialize_validates_checksum_and_writes_separate_canonical_bundle(tmp_path: Path) -> None:
    source_csv = tmp_path / "frozen" / "assembly.csv"
    metadata_json = tmp_path / "frozen" / "assembly.json"
    output_dir = tmp_path / "derived" / "canonical"
    _write_csv(source_csv, [_assembly_row("AAPL", provider_article_id="a1"), _assembly_row("MSFT", provider_article_id="a2")])
    checksum = sha256_file(source_csv)
    metadata_json.write_text(json.dumps({"assembly_checksum": checksum, "assembled_article_symbol_rows": 2}), encoding="utf-8")

    manifest = materialize_historical_canonical_corpus(
        source_assembly_csv_path=source_csv,
        source_assembly_metadata_json_path=metadata_json,
        output_dir=output_dir,
        expected_source_checksum=checksum,
        write_enabled=True,
        ingested_at_utc="2026-07-10T00:00:00Z",
    )

    corpus_path = output_dir / CANONICAL_CORPUS_CSV
    manifest_path = output_dir / CANONICAL_CORPUS_MANIFEST_JSON
    audit_path = output_dir / CANONICAL_CORPUS_AUDIT_JSON
    rows = list(csv.DictReader(corpus_path.open("r", encoding="utf-8", newline="")))

    assert manifest["source_row_count"] == 2
    assert manifest["canonical_row_count"] == 2
    assert manifest["row_count_reconciled"] is True
    assert manifest["transformation_version"] == HISTORICAL_CANONICAL_TRANSFORMATION_VERSION
    assert manifest["features_generated"] is False
    assert manifest["model_training_invoked"] is False
    assert corpus_path.exists()
    assert manifest_path.exists()
    assert audit_path.exists()
    assert rows[0]["canonical_schema_version"]
    assert rows[0]["source_assembly_checksum"] == checksum
    assert rows[0]["conversion_status"] == "converted"
    assert str(output_dir) in manifest["output_files"]["canonical_corpus_csv"]
    assert manifest["canonical_corpus_checksum"] == sha256_file(corpus_path)
    assert manifest["canonical_rows_logical_checksum"]
    assert manifest["canonical_corpus_identity"]
    assert manifest["canonical_schema_checksum"]
    assert manifest["logical_manifest_checksum"]
    assert verify_canonical_corpus_inventory(
        manifest, corpus_path=corpus_path
    )["inventory_certified"] is True


def test_materialize_from_config_uses_explicit_write_enable(tmp_path: Path) -> None:
    source_csv = tmp_path / "frozen" / "assembly.csv"
    metadata_json = tmp_path / "frozen" / "assembly.json"
    output_dir = tmp_path / "derived"
    _write_csv(source_csv, [_assembly_row("AAPL", provider_article_id="a1")])
    checksum = sha256_file(source_csv)
    metadata_json.write_text(json.dumps({"assembly_checksum": checksum}), encoding="utf-8")

    manifest = materialize_historical_canonical_corpus_from_config(
        HistoricalCanonicalCorpusConfig(
            source_assembly_csv_path=source_csv,
            source_assembly_metadata_json_path=metadata_json,
            output_dir=output_dir,
            expected_source_checksum=checksum,
            write_enabled=True,
            ingested_at_utc="2026-07-10T00:00:00Z",
        )
    )

    assert manifest["canonical_row_count"] == 1
    assert (output_dir / CANONICAL_CORPUS_CSV).exists()


def test_materialize_refuses_checksum_mismatch_without_outputs(tmp_path: Path) -> None:
    source_csv = tmp_path / "frozen" / "assembly.csv"
    metadata_json = tmp_path / "frozen" / "assembly.json"
    output_dir = tmp_path / "derived"
    _write_csv(source_csv, [_assembly_row("AAPL", provider_article_id="a1")])
    metadata_json.write_text(json.dumps({"assembly_checksum": "not-the-real-checksum"}), encoding="utf-8")

    with pytest.raises(HistoricalCanonicalCorpusError, match="metadata checksum"):
        materialize_historical_canonical_corpus(
            source_assembly_csv_path=source_csv,
            source_assembly_metadata_json_path=metadata_json,
            output_dir=output_dir,
            expected_source_checksum=sha256_file(source_csv),
            write_enabled=True,
            ingested_at_utc="2026-07-10T00:00:00Z",
        )

    assert not (output_dir / CANONICAL_CORPUS_CSV).exists()


def test_materialize_is_disabled_by_default(tmp_path: Path) -> None:
    source_csv = tmp_path / "frozen" / "assembly.csv"
    metadata_json = tmp_path / "frozen" / "assembly.json"
    _write_csv(source_csv, [_assembly_row("AAPL", provider_article_id="a1")])
    metadata_json.write_text(json.dumps({"assembly_checksum": sha256_file(source_csv)}), encoding="utf-8")

    with pytest.raises(HistoricalCanonicalCorpusError, match="disabled by default"):
        materialize_historical_canonical_corpus(
            source_assembly_csv_path=source_csv,
            source_assembly_metadata_json_path=metadata_json,
            output_dir=tmp_path / "derived",
            expected_source_checksum=sha256_file(source_csv),
        )


def test_materialize_requires_deterministic_ingestion_timestamp(tmp_path):
    source_csv = tmp_path / "frozen" / "assembly.csv"
    metadata_json = tmp_path / "frozen" / "assembly.json"
    _write_csv(source_csv, [_assembly_row("AAPL", provider_article_id="a1")])
    checksum = sha256_file(source_csv)
    metadata_json.write_text(
        json.dumps({"assembly_checksum": checksum}), encoding="utf-8"
    )
    with pytest.raises(HistoricalCanonicalCorpusError, match="ingested_at_utc"):
        materialize_historical_canonical_corpus(
            source_assembly_csv_path=source_csv,
            source_assembly_metadata_json_path=metadata_json,
            output_dir=tmp_path / "derived",
            expected_source_checksum=checksum,
            write_enabled=True,
        )


def test_reordered_source_rows_have_same_logical_inventory_checksum(tmp_path):
    rows = [
        _assembly_row("MSFT", provider_article_id="a2"),
        _assembly_row("AAPL", provider_article_id="a1"),
    ]
    left, _ = build_historical_canonical_rows(
        rows,
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="SOURCE",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )
    right, _ = build_historical_canonical_rows(
        list(reversed(rows)),
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="SOURCE",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )
    assert canonical_rows_logical_checksum(left) == canonical_rows_logical_checksum(
        right
    )


@pytest.mark.parametrize(
    "change",
    [
        {"headline": "changed transformed headline"},
        {"provider_article_id": "changed-article-id", "article_id": "changed"},
        {"symbol": "TSLA", "provider_symbols": "TSLA"},
        {"published_at_utc": "2024-01-03T14:30:00Z"},
        {"collected_at_utc": "2026-07-11T12:00:00Z"},
    ],
)
def test_relevant_canonical_content_changes_logical_checksum(tmp_path, change):
    base = _assembly_row("AAPL", provider_article_id="a1")
    changed = {**base, **change}
    left, _ = build_historical_canonical_rows(
        [base],
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="SOURCE",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )
    right, _ = build_historical_canonical_rows(
        [changed],
        source_assembly_path=tmp_path / "assembly.csv",
        source_assembly_checksum="SOURCE",
        ingested_at_utc="2026-07-10T00:00:00Z",
    )
    assert canonical_rows_logical_checksum(left) != canonical_rows_logical_checksum(
        right
    )


def test_legacy_manifest_is_readable_but_not_inventory_certified():
    result = verify_canonical_corpus_inventory(
        {
            "schema_version": LEGACY_HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
            "canonical_row_count": 1,
            "source_assembly_checksum": "SOURCE",
        }
    )
    assert result["readable"] is True
    assert result["inventory_certified"] is False
    assert result["reasons"] == ["STRENGTHENED_INVENTORY_IDENTITY_MISSING"]


def test_compatible_skip_and_incompatible_existing_output_fail_closed(tmp_path):
    source_csv = tmp_path / "frozen" / "assembly.csv"
    metadata_json = tmp_path / "frozen" / "assembly.json"
    output_dir = tmp_path / "canonical"
    _write_csv(source_csv, [_assembly_row("AAPL", provider_article_id="a1")])
    checksum = sha256_file(source_csv)
    metadata_json.write_text(
        json.dumps({"assembly_checksum": checksum}), encoding="utf-8"
    )
    values = {
        "source_assembly_csv_path": source_csv,
        "source_assembly_metadata_json_path": metadata_json,
        "output_dir": output_dir,
        "expected_source_checksum": checksum,
        "write_enabled": True,
        "ingested_at_utc": "2026-07-10T00:00:00Z",
    }
    first = materialize_historical_canonical_corpus(**values)
    second = materialize_historical_canonical_corpus(**values)
    assert first["canonical_corpus_checksum"] == second["canonical_corpus_checksum"]
    assert (
        first["canonical_rows_logical_checksum"]
        == second["canonical_rows_logical_checksum"]
    )
    assert second["publication_result"] == "SKIPPED_COMPATIBLE"

    (output_dir / CANONICAL_CORPUS_CSV).write_text(
        "incompatible\n", encoding="utf-8"
    )
    with pytest.raises(
        HistoricalCanonicalCorpusError, match="incompatible existing"
    ):
        materialize_historical_canonical_corpus(**values)


def test_corpus_publisher_has_no_finbert_or_training_dependency():
    source = Path(
        "core/research/ml/stock_level/news_sources/historical_canonical_corpus.py"
    ).read_text(encoding="utf-8").lower()
    for forbidden in ("finbert", "transformers", "torch", "score_batch", "fit("):
        assert forbidden not in source


def _assembly_row(symbol: str, *, provider_article_id: str) -> dict[str, str]:
    return {
        "provider": "alpaca_benzinga",
        "article_id": f"alpaca_benzinga:{provider_article_id}:{symbol}",
        "provider_article_id": provider_article_id,
        "provider_original_article_id": provider_article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": "2024-01-02T14:30:00Z",
        "updated_at_utc": "",
        "collected_at_utc": "2026-07-10T12:00:00Z",
        "headline": f"{symbol} headline",
        "summary": f"{symbol} summary",
        "body_or_full_text": "",
        "source": "benzinga",
        "publisher": "Benzinga",
        "author": "Reporter",
        "provider_url": f"https://example.test/{provider_article_id}",
        "language": "en",
        "event_type": "",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
