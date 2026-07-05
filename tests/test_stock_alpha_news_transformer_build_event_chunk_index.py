import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_build_event_chunk_index import build_event_chunk_index


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event(
    event_id: str,
    *,
    event_key: str | None = None,
    symbol: str = "AAA",
    form_type: str = "8-K",
    accession: str = "0000000001-24-000001",
    url: str = "",
    event_ts: str = "2024-01-01T12:00:00Z",
    available_ts: str = "2024-01-01T12:00:00.000Z",
    document_id: str = "",
) -> dict[str, str]:
    key = event_key or f"sec_company_filings|{symbol}|https://www.sec.gov/Archives/edgar/data/1/000000000124000001/|{event_ts}"
    row = {
        "event_id": event_id,
        "event_key": key,
        "symbol": symbol,
        "provider": "sec_company_filings",
        "source_type": "sec_filing",
        "event_timestamp": event_ts,
        "available_at_timestamp": available_ts,
        "form_type": form_type,
        "title": f"{form_type} filed by {symbol}",
        "summary_or_text": "fixture",
        "url_or_accession": url or accession,
        "is_sec_filing": "true",
        "is_rss_item": "false",
        "event_year": event_ts[:4],
        "event_month": "1",
        "event_day_of_week": "1",
    }
    if document_id:
        row["document_id"] = document_id
    return row


def _doc(
    document_id: str,
    *,
    accession: str = "0000000001-24-000001",
    url: str = "https://www.sec.gov/doc-1.htm",
    symbol: str = "AAA",
    form_type: str = "8-K",
    event_keys=None,
    truncated: bool = False,
    text: str = "chunk text",
) -> tuple[dict, dict, dict]:
    source_hash = _sha(f"source:{document_id}")
    clean_hash = _sha(f"clean:{document_id}")
    chunk_hash = _sha(text)
    manifest = {
        "document_id": document_id,
        "accession_number": accession,
        "symbols": [symbol],
        "form_types": [form_type],
        "event_timestamp": "",
        "availability_timestamp": "",
        "source_cache_path": f"reports/cache/{document_id}.txt",
        "primary_document_url": url,
        "source_content_sha256": source_hash,
        "cleaned_content_sha256": clean_hash,
        "source_character_length": len(text),
        "cleaned_character_length": len(text),
        "cleaning_flags": [],
        "truncated_document": truncated,
        "total_candidate_chunks": 10 if truncated else 1,
        "retained_chunk_count": 3 if truncated else 1,
        "dropped_chunk_count": 7 if truncated else 0,
    }
    cache = {
        "document_id": document_id,
        "accession_number": accession,
        "primary_document_url": url,
        "content_sha256": source_hash,
        "event_keys": event_keys or [],
    }
    chunk = {
        "chunk_id": f"{document_id}:00000:{chunk_hash[:12]}",
        "document_id": document_id,
        "accession_number": accession,
        "symbols": [symbol],
        "form_types": [form_type],
        "event_timestamp": "",
        "availability_timestamp": "",
        "chunk_index": 0,
        "source_character_start": 0,
        "source_character_end": len(text),
        "cleaned_character_start": 0,
        "cleaned_character_end": len(text),
        "chunk_text": text,
        "chunk_character_length": len(text),
        "source_content_sha256": source_hash,
        "cleaned_content_sha256": clean_hash,
        "chunk_content_sha256": chunk_hash,
        "cleaning_flags": ["likely_navigation_or_boilerplate"] if "boilerplate" in text else [],
        "truncated_document": truncated,
        "total_candidate_chunks": 10 if truncated else 1,
        "retained_chunk_count": 3 if truncated else 1,
        "source_cache_path": f"reports/cache/{document_id}.txt",
    }
    return manifest, cache, chunk


def _write_fixture(
    tmp_path: Path,
    *,
    events: list[dict],
    docs: list[tuple[dict, dict, dict]],
    duplicate_rows: list[dict] | None = None,
    truncation_rows: list[dict] | None = None,
    boundary_validation: dict | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    root = tmp_path / "reports"
    event_csv = root / "events.csv"
    chunk_dir = root / "chunks"
    cache_manifest = root / "cache_manifest.json"
    audit_dir = root / "chunk_audit"
    output_dir = root / "index"
    chunk_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)

    fieldnames = sorted({key for row in events for key in row})
    with event_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
    manifests, caches, chunks = zip(*docs)
    (chunk_dir / "sec_primary_document_text_clean_manifest.json").write_text(json.dumps({"documents": list(manifests)}), encoding="utf-8")
    (chunk_dir / "sec_primary_document_text_chunks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in chunks), encoding="utf-8")
    (cache_manifest).write_text(json.dumps({"documents": list(caches)}), encoding="utf-8")
    (audit_dir / "sec_primary_document_text_duplicate_chunks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (duplicate_rows or [])),
        encoding="utf-8",
    )
    (audit_dir / "sec_primary_document_text_truncation_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (truncation_rows or [])),
        encoding="utf-8",
    )
    (audit_dir / "sec_primary_document_text_chunk_audit.json").write_text(
        json.dumps({"boundary_validation": boundary_validation or {}}),
        encoding="utf-8",
    )
    return event_csv, chunk_dir, cache_manifest, audit_dir, output_dir


def _run(tmp_path: Path, **kwargs) -> tuple[dict, Path]:
    event_csv, chunk_dir, cache_manifest, audit_dir, output_dir = _write_fixture(tmp_path, **kwargs)
    summary = build_event_chunk_index(
        event_features_csv=event_csv,
        chunk_dir=chunk_dir,
        cache_manifest_path=cache_manifest,
        chunk_audit_dir=audit_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
    )
    return summary, output_dir


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_one_event_to_one_document_accession_match(tmp_path: Path) -> None:
    doc = _doc("doc1")
    summary, output = _run(tmp_path, events=[_event("event1")], docs=[doc])

    assert summary["event_document_link_count"] == 1
    assert summary["event_chunk_link_count"] == 1
    assert _rows(output / "news_transformer_event_chunk_index.jsonl")[0]["event_id"] == "event1"


def test_one_event_to_multiple_documents_via_event_key(tmp_path: Path) -> None:
    key = "event-key"
    docs = [_doc("doc1", event_keys=[key]), _doc("doc2", accession="0000000002-24-000002", event_keys=[key])]
    summary, _ = _run(tmp_path, events=[_event("event1", event_key=key, accession="")], docs=docs)

    assert summary["events_mapping_to_multiple_documents"] == 1
    assert summary["event_document_link_count"] == 2


def test_multiple_events_to_one_document(tmp_path: Path) -> None:
    doc = _doc("doc1")
    events = [_event("event1"), _event("event2")]
    summary, _ = _run(tmp_path, events=events, docs=[doc])

    assert summary["documents_mapping_to_multiple_events"] == 1
    assert summary["event_chunk_link_count"] == 2


def test_url_based_match(tmp_path: Path) -> None:
    url = "https://www.sec.gov/doc-1.htm"
    summary, _ = _run(tmp_path, events=[_event("event1", accession="", url=url)], docs=[_doc("doc1", url=url)])

    assert summary["event_document_link_count"] == 1


def test_stable_document_id_match(tmp_path: Path) -> None:
    summary, _ = _run(tmp_path, events=[_event("event1", accession="", document_id="doc1")], docs=[_doc("doc1")])

    assert summary["event_document_link_count"] == 1


def test_ambiguous_join_is_quarantined(tmp_path: Path) -> None:
    key = "event-key"
    docs = [_doc("doc1", event_keys=[key]), _doc("doc2", accession="0000000001-24-000001")]
    summary, output = _run(tmp_path, events=[_event("event1", event_key=key)], docs=docs)

    assert summary["ambiguous_join_count"] == 1
    assert summary["event_document_link_count"] == 0
    assert _rows(output / "news_transformer_ambiguous_joins.jsonl")


def test_unmatched_event_and_document_reporting(tmp_path: Path) -> None:
    summary, output = _run(tmp_path, events=[_event("event1", accession="0000009999-24-000001")], docs=[_doc("doc1")])

    assert summary["unmatched_event_count"] == 1
    assert summary["unmatched_document_count"] == 1
    assert _rows(output / "news_transformer_unmatched_events.jsonl")
    assert _rows(output / "news_transformer_unmatched_documents.jsonl")


def test_timestamps_are_preserved_and_restored(tmp_path: Path) -> None:
    summary, output = _run(tmp_path, events=[_event("event1", available_ts="2024-01-02T09:30:00Z")], docs=[_doc("doc1")])
    row = _rows(output / "news_transformer_event_chunk_index.jsonl")[0]

    assert summary["model_eligible_event_count"] == 1
    assert row["event_timestamp"] == "2024-01-01T12:00:00Z"
    assert row["available_at_timestamp"] == "2024-01-02T09:30:00Z"
    assert row["model_eligible"] is True


def test_missing_availability_blocks_model_eligibility(tmp_path: Path) -> None:
    summary, output = _run(tmp_path, events=[_event("event1", available_ts="")], docs=[_doc("doc1")])
    row = _rows(output / "news_transformer_event_chunk_index.jsonl")[0]

    assert summary["model_eligible_chunk_count"] == 0
    assert row["model_eligible"] is False


def test_invalid_timestamp_order_blocks_readiness(tmp_path: Path) -> None:
    summary, _ = _run(tmp_path, events=[_event("event1", available_ts="2023-12-31T00:00:00Z")], docs=[_doc("doc1")])

    assert summary["readiness"]["status"] == "blocked"
    assert "timestamp_ordering_violations" in summary["readiness"]["blocking_reasons"]


def test_duplicate_and_boilerplate_metadata(tmp_path: Path) -> None:
    manifest, cache, chunk = _doc("doc1", text="boilerplate")
    duplicate = {
        "chunk_content_sha256": chunk["chunk_content_sha256"],
        "classification": "likely_boilerplate",
        "instance_count": 3,
    }
    summary, output = _run(tmp_path, events=[_event("event1")], docs=[(manifest, cache, chunk)], duplicate_rows=[duplicate])
    row = _rows(output / "news_transformer_event_chunk_index.jsonl")[0]

    assert summary["boilerplate_duplicate_chunk_count"] == 1
    assert row["duplicate_chunk_hash"] is True
    assert row["duplicate_group_size"] == 3
    assert row["likely_boilerplate_duplicate"] is True


def test_truncation_metadata_preserved(tmp_path: Path) -> None:
    manifest, cache, chunk = _doc("doc1", truncated=True)
    truncation = {
        "document_id": "doc1",
        "retained_opening_chunk_count": 1,
        "retained_middle_chunk_count": 1,
        "retained_ending_chunk_count": 1,
        "retained_cleaned_character_coverage": 0.5,
        "largest_uncovered_character_gap": 1000,
    }
    summary, output = _run(tmp_path, events=[_event("event1")], docs=[(manifest, cache, chunk)], truncation_rows=[truncation])
    row = _rows(output / "news_transformer_event_chunk_index.jsonl")[0]

    assert summary["events_linked_to_truncated_documents"] == 1
    assert row["truncated_document"] is True
    assert row["retained_middle_chunk_count"] == 1
    assert row["retained_character_coverage"] == 0.5


def test_hash_and_chunk_id_preserved(tmp_path: Path) -> None:
    manifest, cache, chunk = _doc("doc1")
    summary, output = _run(tmp_path, events=[_event("event1")], docs=[(manifest, cache, chunk)])
    row = _rows(output / "news_transformer_event_chunk_index.jsonl")[0]

    assert summary["event_chunk_link_count"] == 1
    assert row["chunk_id"] == chunk["chunk_id"]
    assert row["chunk_content_sha256"] == chunk["chunk_content_sha256"]
    assert row["source_content_sha256"] == chunk["source_content_sha256"]


def test_source_hash_mismatch_blocks_readiness(tmp_path: Path) -> None:
    manifest, cache, chunk = _doc("doc1")
    cache["content_sha256"] = "different"
    summary, _ = _run(tmp_path, events=[_event("event1")], docs=[(manifest, cache, chunk)])

    assert summary["readiness"]["status"] == "blocked"
    assert "source_hash_mismatches" in summary["readiness"]["blocking_reasons"]


def test_repeated_run_is_deterministic(tmp_path: Path) -> None:
    event_csv, chunk_dir, cache_manifest, audit_dir, output_dir = _write_fixture(
        tmp_path,
        events=[_event("event1")],
        docs=[_doc("doc1")],
    )
    first = build_event_chunk_index(
        event_features_csv=event_csv,
        chunk_dir=chunk_dir,
        cache_manifest_path=cache_manifest,
        chunk_audit_dir=audit_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
    )
    second = build_event_chunk_index(
        event_features_csv=event_csv,
        chunk_dir=chunk_dir,
        cache_manifest_path=cache_manifest,
        chunk_audit_dir=audit_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
    )

    assert first == second


def test_output_restricted_to_reports(tmp_path: Path) -> None:
    event_csv, chunk_dir, cache_manifest, audit_dir, _ = _write_fixture(
        tmp_path,
        events=[_event("event1")],
        docs=[_doc("doc1")],
    )

    with pytest.raises(ValueError, match="output_dir must be under reports"):
        build_event_chunk_index(
            event_features_csv=event_csv,
            chunk_dir=chunk_dir,
            cache_manifest_path=cache_manifest,
            chunk_audit_dir=audit_dir,
            output_dir=tmp_path / "outside",
            reports_root=tmp_path / "reports",
        )


def test_source_datasets_unchanged(tmp_path: Path) -> None:
    event_csv, chunk_dir, cache_manifest, audit_dir, output_dir = _write_fixture(
        tmp_path,
        events=[_event("event1")],
        docs=[_doc("doc1")],
    )
    before = {
        path: path.read_bytes()
        for path in (event_csv, chunk_dir / "sec_primary_document_text_chunks.jsonl", cache_manifest)
    }
    build_event_chunk_index(
        event_features_csv=event_csv,
        chunk_dir=chunk_dir,
        cache_manifest_path=cache_manifest,
        chunk_audit_dir=audit_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
    )

    assert {path: path.read_bytes() for path in before} == before


def test_no_network_model_or_tokenizer_imports() -> None:
    source = Path("scripts/stock_alpha_news_transformer_build_event_chunk_index.py").read_text(encoding="utf-8")

    forbidden = ("urllib", "requests", "socket", "torch", "sklearn", "transformers", "AutoTokenizer", "FinBERT", "broker", "paper_trading")
    assert all(token not in source for token in forbidden)
