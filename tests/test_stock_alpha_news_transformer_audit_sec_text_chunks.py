import hashlib
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_audit_sec_text_chunks import audit_sec_text_chunks


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk(
    *,
    document_id: str,
    chunk_index: int,
    text: str,
    start: int,
    end: int,
    cleaned_hash: str,
    source_hash: str,
    accession: str = "0000000001-24-000001",
    symbols=None,
    forms=None,
    chunk_id: str | None = None,
    event_timestamp: str = "2024-01-01T12:00:00Z",
    availability_timestamp: str = "",
):
    chunk_hash = _sha(text)
    return {
        "chunk_id": chunk_id or f"{document_id[:4]}:{chunk_index:05d}:{chunk_hash[:12]}",
        "document_id": document_id,
        "accession_number": accession,
        "symbols": symbols or ["AAA"],
        "form_types": forms or ["8-K"],
        "event_timestamp": event_timestamp,
        "availability_timestamp": availability_timestamp,
        "chunk_index": chunk_index,
        "source_character_start": start,
        "source_character_end": end,
        "cleaned_character_start": start,
        "cleaned_character_end": end,
        "chunk_text": text,
        "chunk_character_length": len(text),
        "source_content_sha256": source_hash,
        "cleaned_content_sha256": cleaned_hash,
        "chunk_content_sha256": chunk_hash,
        "cleaning_flags": [],
        "truncated_document": False,
        "total_candidate_chunks": 1,
        "retained_chunk_count": 1,
        "source_cache_path": f"reports/cache/{document_id}.txt",
    }


def _write_output(
    tmp_path: Path,
    *,
    chunks: list[dict],
    manifest_overrides: dict[str, dict] | None = None,
    params: dict | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "reports" / "chunks"
    root.mkdir(parents=True)
    docs = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        current = docs.setdefault(
            doc_id,
            {
                "document_id": doc_id,
                "accession_number": chunk["accession_number"],
                "symbols": chunk["symbols"],
                "form_types": chunk["form_types"],
                "event_timestamp": chunk["event_timestamp"],
                "availability_timestamp": chunk["availability_timestamp"],
                "source_cache_path": chunk["source_cache_path"],
                "primary_document_url": f"https://www.sec.gov/{doc_id}.htm",
                "source_content_sha256": chunk["source_content_sha256"],
                "cleaned_content_sha256": chunk["cleaned_content_sha256"],
                "source_character_length": max(chunk["source_character_end"], chunk["cleaned_character_end"]),
                "cleaned_character_length": max(chunk["cleaned_character_end"], chunk["source_character_end"]),
                "cleaning_flags": chunk["cleaning_flags"],
                "truncated_document": chunk["truncated_document"],
                "total_candidate_chunks": chunk["total_candidate_chunks"],
                "retained_chunk_count": chunk["retained_chunk_count"],
                "dropped_chunk_count": max(0, chunk["total_candidate_chunks"] - chunk["retained_chunk_count"]),
            },
        )
        current["cleaned_character_length"] = max(current["cleaned_character_length"], chunk["cleaned_character_end"])
        current["source_character_length"] = max(current["source_character_length"], chunk["source_character_end"])
    for doc_id, overrides in (manifest_overrides or {}).items():
        docs[doc_id].update(overrides)
    summary = {
        "chunking_parameters": {"chunk_size": 2_000, "chunk_overlap": 250, "max_chunks_per_document": 128, **(params or {})},
    }
    (root / "sec_primary_document_text_clean_chunk_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "sec_primary_document_text_clean_manifest.json").write_text(
        json.dumps({"documents": list(docs.values())}),
        encoding="utf-8",
    )
    (root / "sec_primary_document_text_clean_document_report.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in docs.values()),
        encoding="utf-8",
    )
    (root / "sec_primary_document_text_chunks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in chunks),
        encoding="utf-8",
    )
    return root, tmp_path / "reports" / "audit"


def _audit(tmp_path: Path, chunks: list[dict], manifest_overrides: dict[str, dict] | None = None) -> dict:
    root, output = _write_output(tmp_path, chunks=chunks, manifest_overrides=manifest_overrides)
    return audit_sec_text_chunks(chunk_dir=root, output_dir=output, reports_root=tmp_path / "reports")


def test_chunk_audit_accepts_valid_untruncated_chunks(tmp_path: Path) -> None:
    clean_hash = _sha("alpha" * 500)
    chunks = [
        _chunk(document_id="doc1", chunk_index=0, text="A" * 2_000, start=0, end=2_000, cleaned_hash=clean_hash, source_hash="source"),
        _chunk(document_id="doc1", chunk_index=1, text="B" * 500, start=1_750, end=2_250, cleaned_hash=clean_hash, source_hash="source"),
    ]
    audit = _audit(tmp_path, chunks)

    assert audit["full_build_readiness"]["status"] == "approved_with_warnings"
    assert audit["boundary_validation"]["invalid_offset_count"] == 0
    assert audit["boundary_validation"]["chunk_hash_mismatch_count"] == 0


def test_chunk_audit_accepts_valid_truncated_selection(tmp_path: Path) -> None:
    clean_hash = _sha("long-doc")
    chunks = [
        _chunk(document_id="doc1", chunk_index=0, text="A" * 2_000, start=0, end=2_000, cleaned_hash=clean_hash, source_hash="source"),
        _chunk(document_id="doc1", chunk_index=1, text="M" * 2_000, start=5_000, end=7_000, cleaned_hash=clean_hash, source_hash="source"),
        _chunk(document_id="doc1", chunk_index=2, text="Z" * 1_000, start=9_000, end=10_000, cleaned_hash=clean_hash, source_hash="source"),
    ]
    for chunk in chunks:
        chunk["truncated_document"] = True
        chunk["total_candidate_chunks"] = 10
        chunk["retained_chunk_count"] = 3
    audit = _audit(
        tmp_path,
        chunks,
        {"doc1": {"truncated_document": True, "total_candidate_chunks": 10, "retained_chunk_count": 3, "dropped_chunk_count": 7, "cleaned_character_length": 10_000}},
    )

    assert audit["truncation_summary"]["all_keep_opening_context"] is True
    assert audit["truncation_summary"]["all_keep_ending_context"] is True
    assert audit["truncation_summary"]["all_balanced"] is True


def test_chunk_audit_blocks_truncation_that_drops_ending_context(tmp_path: Path) -> None:
    clean_hash = _sha("bad-truncation")
    chunks = [
        _chunk(document_id="doc1", chunk_index=0, text="A" * 2_000, start=0, end=2_000, cleaned_hash=clean_hash, source_hash="source"),
        _chunk(document_id="doc1", chunk_index=1, text="B" * 2_000, start=1_750, end=3_750, cleaned_hash=clean_hash, source_hash="source"),
    ]
    for chunk in chunks:
        chunk["truncated_document"] = True
        chunk["total_candidate_chunks"] = 10
        chunk["retained_chunk_count"] = 2
    audit = _audit(
        tmp_path,
        chunks,
        {"doc1": {"truncated_document": True, "total_candidate_chunks": 10, "retained_chunk_count": 2, "dropped_chunk_count": 8, "cleaned_character_length": 10_000}},
    )

    assert "truncation_drops_ending_context" in audit["full_build_readiness"]["blocking_reasons"]


def test_chunk_audit_blocks_invalid_offsets(tmp_path: Path) -> None:
    chunk = _chunk(document_id="doc1", chunk_index=0, text="A", start=10, end=5, cleaned_hash=_sha("A"), source_hash="source")
    audit = _audit(tmp_path, [chunk])

    assert audit["boundary_validation"]["invalid_offset_count"] == 1
    assert "invalid_offsets" in audit["full_build_readiness"]["blocking_reasons"]


def test_chunk_audit_blocks_length_mismatch(tmp_path: Path) -> None:
    chunk = _chunk(document_id="doc1", chunk_index=0, text="A", start=0, end=1, cleaned_hash=_sha("A"), source_hash="source")
    chunk["chunk_character_length"] = 2
    audit = _audit(tmp_path, [chunk])

    assert audit["boundary_validation"]["chunk_length_mismatch_count"] == 1


def test_chunk_audit_blocks_hash_mismatch(tmp_path: Path) -> None:
    chunk = _chunk(document_id="doc1", chunk_index=0, text="A", start=0, end=1, cleaned_hash=_sha("A"), source_hash="source")
    chunk["chunk_content_sha256"] = "bad"
    audit = _audit(tmp_path, [chunk])

    assert audit["boundary_validation"]["chunk_hash_mismatch_count"] == 1
    assert "chunk_hash_mismatches" in audit["full_build_readiness"]["blocking_reasons"]


def test_chunk_audit_blocks_duplicate_chunk_ids(tmp_path: Path) -> None:
    chunks = [
        _chunk(document_id="doc1", chunk_index=0, text="A", start=0, end=1, cleaned_hash=_sha("A"), source_hash="source", chunk_id="dup"),
        _chunk(document_id="doc2", chunk_index=0, text="B", start=0, end=1, cleaned_hash=_sha("B"), source_hash="source2", chunk_id="dup"),
    ]
    audit = _audit(tmp_path, chunks)

    assert audit["boundary_validation"]["duplicate_chunk_id_count"] == 1
    assert "duplicate_chunk_ids" in audit["full_build_readiness"]["blocking_reasons"]


def test_chunk_audit_classifies_within_document_duplicate_hash(tmp_path: Path) -> None:
    text = "same boilerplate text"
    chunks = [
        _chunk(document_id="doc1", chunk_index=0, text=text, start=0, end=len(text), cleaned_hash=_sha(text), source_hash="source"),
        _chunk(document_id="doc1", chunk_index=1, text=text, start=100, end=100 + len(text), cleaned_hash=_sha(text), source_hash="source"),
    ]
    audit = _audit(tmp_path, chunks)

    assert audit["duplicate_classification_counts"]["within_same_document"] == 1


def test_chunk_audit_classifies_cross_document_exact_source_duplicate(tmp_path: Path) -> None:
    text = "same exact source chunk"
    chunks = [
        _chunk(document_id="doc1", chunk_index=0, text=text, start=0, end=len(text), cleaned_hash=_sha(text), source_hash="same-source"),
        _chunk(document_id="doc2", chunk_index=0, text=text, start=0, end=len(text), cleaned_hash=_sha(text), source_hash="same-source"),
    ]
    audit = _audit(tmp_path, chunks)

    assert audit["duplicate_classification_counts"]["from_exact_duplicate_source_documents"] == 1


def test_chunk_audit_reports_missing_provenance_as_warning(tmp_path: Path) -> None:
    chunk = _chunk(document_id="doc1", chunk_index=0, text="A", start=0, end=1, cleaned_hash=_sha("A"), source_hash="source", event_timestamp="")
    audit = _audit(tmp_path, [chunk])

    assert audit["provenance_audit"]["event_timestamp_missing"] == 1
    assert audit["provenance_audit"]["availability_timestamp_missing"] == 1
    assert "availability_timestamps_missing_at_chunk_layer" in audit["full_build_readiness"]["warnings"]


def test_chunk_audit_refuses_output_outside_reports(tmp_path: Path) -> None:
    root, _ = _write_output(
        tmp_path,
        chunks=[_chunk(document_id="doc1", chunk_index=0, text="A", start=0, end=1, cleaned_hash=_sha("A"), source_hash="source")],
    )

    with pytest.raises(ValueError, match="output_dir must be under reports"):
        audit_sec_text_chunks(chunk_dir=root, output_dir=tmp_path / "outside", reports_root=tmp_path / "reports")


def test_chunk_audit_repeated_run_is_deterministic(tmp_path: Path) -> None:
    root, output = _write_output(
        tmp_path,
        chunks=[_chunk(document_id="doc1", chunk_index=0, text="A", start=0, end=1, cleaned_hash=_sha("A"), source_hash="source")],
    )
    first = audit_sec_text_chunks(chunk_dir=root, output_dir=output, reports_root=tmp_path / "reports")
    second = audit_sec_text_chunks(chunk_dir=root, output_dir=output, reports_root=tmp_path / "reports")

    assert first == second


def test_chunk_audit_has_no_network_model_or_tokenizer_imports() -> None:
    source = Path("scripts/stock_alpha_news_transformer_audit_sec_text_chunks.py").read_text(encoding="utf-8")

    forbidden = ("urllib", "requests", "socket", "torch", "sklearn", "transformers", "AutoTokenizer", "FinBERT", "broker", "paper_trading")
    assert all(token not in source for token in forbidden)
