import hashlib
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_clean_and_chunk_sec_text import (
    CleanChunkConfig,
    clean_and_chunk_sec_text_cache,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_cache(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    cache_dir = tmp_path / "reports" / "cache"
    docs_dir = cache_dir / "documents"
    docs_dir.mkdir(parents=True)
    manifest_rows = []
    for index, row in enumerate(rows, 1):
        accession = row.get("accession_number", f"000000000{index}-24-00000{index}")
        text = row.get("text", "")
        path = docs_dir / f"{accession}_{index}.txt"
        path.write_text(text, encoding="utf-8")
        manifest_rows.append(
            {
                "document_id": row.get("document_id", f"{accession}|https://www.sec.gov/doc-{index}.htm"),
                "accession_number": accession,
                "cache_path": str(path),
                "content_sha256": _sha256_text(text),
                "text_length": len(text),
                "form_types": row.get("form_types", ["8-K"]),
                "symbols": row.get("symbols", ["AAA"]),
                "event_keys": row.get("event_keys", [f"event|AAA|url|2024-01-0{index}T12:00:00Z"]),
                "available_at_timestamp": row.get("available_at_timestamp", ""),
                "primary_document_url": f"https://www.sec.gov/doc-{index}.htm",
                "status": "cached",
            }
        )
    (cache_dir / "sec_primary_document_text_cache_manifest.json").write_text(
        json.dumps({"documents": manifest_rows}),
        encoding="utf-8",
    )
    return cache_dir, tmp_path / "reports" / "chunks"


def _quality_dir(tmp_path: Path, flags: dict[str, list[str]] | None = None) -> Path:
    path = tmp_path / "reports" / "quality"
    path.mkdir(parents=True)
    (path / "sec_primary_document_text_quality_audit.json").write_text(
        json.dumps(
            {
                "quality_thresholds": {
                    "suspiciously_short_chars": 100,
                    "very_large_document_chars": 900,
                    "extreme_document_chars": 2_000,
                },
                "recommended_max_character_limit": 900,
            }
        ),
        encoding="utf-8",
    )
    lines = [
        json.dumps({"document_id": document_id, "quality_flags": values})
        for document_id, values in (flags or {}).items()
    ]
    (path / "sec_primary_document_text_quality_flags.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return path


def _run(
    tmp_path: Path,
    rows: list[dict],
    *,
    config: CleanChunkConfig | None = None,
    quality_flags: dict[str, list[str]] | None = None,
    max_documents: int | None = None,
):
    cache_dir, output_dir = _write_cache(tmp_path, rows)
    quality_dir = _quality_dir(tmp_path, quality_flags)
    summary = clean_and_chunk_sec_text_cache(
        cache_dir=cache_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
        quality_audit_dir=quality_dir,
        max_documents=max_documents,
        config=config or CleanChunkConfig(),
    )
    chunks = [
        json.loads(line)
        for line in (output_dir / "sec_primary_document_text_chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest = json.loads((output_dir / "sec_primary_document_text_clean_manifest.json").read_text(encoding="utf-8"))
    return summary, chunks, manifest, cache_dir, output_dir


def test_cleaner_normal_cleaning_preserves_visible_text(tmp_path: Path) -> None:
    summary, chunks, manifest, _, _ = _run(tmp_path, [{"text": "Item 1.01 Normal filing text. " * 20}])

    assert summary["documents_cleaned"] == 1
    assert chunks[0]["chunk_text"].startswith("Item 1.01")
    assert manifest["documents"][0]["cleaned_character_length"] > 0


def test_cleaner_normalizes_whitespace(tmp_path: Path) -> None:
    _, chunks, _, _, _ = _run(tmp_path, [{"text": "Alpha\t\tBeta   \n\n\n   Gamma"}])

    assert chunks[0]["chunk_text"] == "Alpha Beta\n\nGamma"


def test_cleaner_removes_residual_html_tags(tmp_path: Path) -> None:
    _, chunks, _, _, _ = _run(tmp_path, [{"text": "<html><body><h1>Heading</h1><p>Visible text</p></body></html>"}])

    assert "<html>" not in chunks[0]["chunk_text"]
    assert "Heading" in chunks[0]["chunk_text"]
    assert "Visible text" in chunks[0]["chunk_text"]


def test_cleaner_removes_inline_xbrl_tags_but_keeps_value(tmp_path: Path) -> None:
    _, chunks, _, _, _ = _run(tmp_path, [{"text": "<ix:nonFraction name='Assets'>123</ix:nonFraction> Assets"}])

    assert "ix:nonFraction" not in chunks[0]["chunk_text"]
    assert "123" in chunks[0]["chunk_text"]


def test_cleaner_removes_script_and_style_content(tmp_path: Path) -> None:
    _, chunks, _, _, _ = _run(
        tmp_path,
        [{"text": "<style>.hidden{}</style><script>bad()</script><p>Useful filing text</p>"}],
    )

    assert "bad()" not in chunks[0]["chunk_text"]
    assert ".hidden" not in chunks[0]["chunk_text"]
    assert "Useful filing text" in chunks[0]["chunk_text"]


def test_cleaner_removes_null_and_control_characters(tmp_path: Path) -> None:
    _, chunks, _, _, _ = _run(tmp_path, [{"text": "Alpha\x00\x01Beta"}])

    assert chunks[0]["chunk_text"] == "AlphaBeta"


def test_cleaner_reduces_repeated_boilerplate_lines(tmp_path: Path) -> None:
    line = "United States Securities and Exchange Commission repeated header"
    _, chunks, manifest, _, _ = _run(tmp_path, [{"text": "\n".join([line] * 8 + ["Business text"])}])

    assert chunks[0]["chunk_text"].count(line) == 3
    assert "repeated_line_reduced" in manifest["documents"][0]["cleaning_flags"]


def test_cleaner_retains_useful_table_text(tmp_path: Path) -> None:
    text = "\n".join(["Revenue | 2024 | $100 | 10%", "Operating income | 2024 | $20 | 2%"])
    _, chunks, _, _, _ = _run(tmp_path, [{"text": text}])

    assert "Revenue" in chunks[0]["chunk_text"]
    assert "Operating income" in chunks[0]["chunk_text"]


def test_chunk_boundaries_and_overlap_are_deterministic(tmp_path: Path) -> None:
    text = "A" * 1_400
    config = CleanChunkConfig(chunk_size=500, chunk_overlap=75, max_chunks_per_document=20)
    first, chunks, _, _, output_dir = _run(tmp_path, [{"text": text}], config=config)
    second = clean_and_chunk_sec_text_cache(
        cache_dir=tmp_path / "reports" / "cache",
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
        quality_audit_dir=tmp_path / "reports" / "quality",
        config=config,
    )

    assert chunks[1]["cleaned_character_start"] == chunks[0]["cleaned_character_end"] - 75
    assert first == second


def test_max_chunk_policy_retains_beginning_and_ending(tmp_path: Path) -> None:
    text = "B" * 5_000
    config = CleanChunkConfig(chunk_size=500, chunk_overlap=50, max_chunks_per_document=4)
    summary, chunks, manifest, _, _ = _run(tmp_path, [{"text": text}], config=config)

    assert summary["documents_truncated"] == 1
    assert len(chunks) == 4
    assert chunks[0]["cleaned_character_start"] == 0
    assert chunks[-1]["cleaned_character_end"] == manifest["documents"][0]["cleaned_character_length"]


def test_hashes_and_provenance_are_preserved(tmp_path: Path) -> None:
    text = "Provenance filing text. " * 50
    _, chunks, manifest, _, _ = _run(
        tmp_path,
        [{"text": text, "symbols": ["XYZ"], "form_types": ["10-Q"], "available_at_timestamp": "2024-01-02T09:30:00Z"}],
    )

    assert chunks[0]["source_content_sha256"] == _sha256_text(text)
    assert chunks[0]["source_content_sha256"] == manifest["documents"][0]["source_content_sha256"]
    assert chunks[0]["cleaned_content_sha256"] == manifest["documents"][0]["cleaned_content_sha256"]
    assert chunks[0]["chunk_content_sha256"] == _sha256_text(chunks[0]["chunk_text"])
    assert chunks[0]["symbols"] == ["XYZ"]
    assert chunks[0]["form_types"] == ["10-Q"]
    assert chunks[0]["event_timestamp"] == "2024-01-01T12:00:00Z"
    assert chunks[0]["availability_timestamp"] == "2024-01-02T09:30:00Z"
    assert manifest["documents"][0]["availability_timestamp"] == "2024-01-02T09:30:00Z"


def test_source_document_is_unchanged(tmp_path: Path) -> None:
    text = "<html>Immutable text</html>"
    _, _, _, cache_dir, _ = _run(tmp_path, [{"text": text}])
    source = next((cache_dir / "documents").glob("*.txt"))

    assert source.read_text(encoding="utf-8") == text


def test_output_restricted_to_reports(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path, [{"text": "Normal text"}])

    with pytest.raises(ValueError, match="output_dir must be under reports"):
        clean_and_chunk_sec_text_cache(
            cache_dir=cache_dir,
            output_dir=tmp_path / "outside",
            reports_root=tmp_path / "reports",
        )


def test_max_documents_smoke_selection_includes_flagged_duplicate_pair(tmp_path: Path) -> None:
    rows = [
        {"document_id": "doc-a", "text": "Normal A" * 100},
        {"document_id": "doc-b", "text": "Normal B" * 100},
        {"document_id": "doc-c", "text": "Normal C" * 100},
    ]
    summary, chunks, _, _, _ = _run(
        tmp_path,
        rows,
        quality_flags={"doc-b": ["exact_duplicate_content"], "doc-c": ["exact_duplicate_content"]},
        max_documents=2,
    )

    assert summary["documents_processed"] == 2
    assert {chunk["document_id"] for chunk in chunks} == {"doc-b", "doc-c"}


def test_cleaner_has_no_network_tokenizer_or_training_imports() -> None:
    source = Path("scripts/stock_alpha_news_transformer_clean_and_chunk_sec_text.py").read_text(encoding="utf-8")

    forbidden = ("urllib", "requests", "socket", "torch", "sklearn", "transformers", "AutoTokenizer", "FinBERT", "broker", "paper_trading")
    assert all(token not in source for token in forbidden)
