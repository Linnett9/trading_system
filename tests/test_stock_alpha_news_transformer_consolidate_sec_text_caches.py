import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_consolidate_sec_text_caches import (
    CACHE_MANIFEST_FILENAME,
    CACHE_SUMMARY_FILENAME,
    consolidate_sec_text_caches,
)


def _cache(root: Path, name: str, rows: list[dict], files: dict[str, str]) -> Path:
    cache_dir = root / "reports" / name
    docs = cache_dir / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        (docs / filename).write_text(text, encoding="utf-8")
    manifest_rows = []
    for row in rows:
        item = dict(row)
        if item.get("cache_path_name"):
            item["cache_path"] = str(docs / item.pop("cache_path_name"))
        manifest_rows.append(item)
    (cache_dir / CACHE_MANIFEST_FILENAME).write_text(json.dumps({"documents": manifest_rows}), encoding="utf-8")
    (cache_dir / CACHE_SUMMARY_FILENAME).write_text(json.dumps({"cached_documents": len(files)}), encoding="utf-8")
    return cache_dir


def _row(document_id: str, status: str, filename: str = "doc.txt", text_length: int = 3) -> dict:
    accession, url = document_id.split("|", 1)
    return {
        "document_id": document_id,
        "accession_number": accession,
        "primary_document_url": url,
        "symbols": ["AAA"],
        "form_types": ["8-K"],
        "status": status,
        "cache_path_name": filename,
        "text_length": text_length,
    }


def _run(tmp_path: Path, primary: Path, retries: list[Path], output_name: str = "out") -> dict:
    return consolidate_sec_text_caches(
        primary_cache_dir=primary,
        retry_cache_dirs=retries,
        output_dir=tmp_path / "reports" / output_name,
        reports_root=tmp_path / "reports",
    )


def test_consolidates_original_and_retry_successes_with_summary_counts(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "one"})
    retry = _cache(tmp_path, "retry", [_row("b|https://www.sec.gov/b.htm", "cached", "b.txt")], {"b.txt": "two"})

    summary = _run(tmp_path, primary, [retry])
    manifest = json.loads((tmp_path / "reports" / "out" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["consolidated_success_count"] == 2
    assert summary["consolidated_failed_count"] == 0
    assert summary["output_document_file_count"] == 2
    assert [row["document_id"] for row in manifest["documents"]] == ["a|https://www.sec.gov/a.htm", "b|https://www.sec.gov/b.htm"]


def test_retry_success_replaces_primary_failure_and_keeps_provenance(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [_row("a|https://www.sec.gov/a.htm", "failed", "a.txt")], {})
    retry = _cache(tmp_path, "retry", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "two"})

    summary = _run(tmp_path, primary, [retry])
    manifest = json.loads((tmp_path / "reports" / "out" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["newly_recovered_document_count"] == 1
    assert summary["consolidated_success_count"] == 1
    assert manifest["documents"][0]["recovered_from_retry"] is True
    assert manifest["documents"][0]["prior_failure_sources"] == [str(primary)]


def test_identical_duplicate_document_files_are_deduplicated(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "same"})
    retry = _cache(tmp_path, "retry", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "same"})

    summary = _run(tmp_path, primary, [retry])

    assert summary["duplicate_document_count"] == 1
    assert summary["identical_duplicate_file_count"] >= 1
    assert summary["conflicting_document_count"] == 0


def test_conflicting_duplicate_success_content_blocks(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "one"})
    retry = _cache(tmp_path, "retry", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "two"})

    summary = _run(tmp_path, primary, [retry])

    assert summary["conflicting_document_count"] == 1
    assert summary["blocking_reasons"] == ["conflicting_document_content"]


def test_missing_success_file_blocks_and_orphan_file_is_reported(tmp_path: Path) -> None:
    primary = _cache(
        tmp_path,
        "primary",
        [_row("a|https://www.sec.gov/a.htm", "cached", "missing.txt")],
        {"orphan.txt": "orphan"},
    )
    summary = _run(tmp_path, primary, [])

    assert summary["missing_success_file_count"] == 1
    assert summary["orphan_document_file_count"] == 1
    assert "missing_success_files" in summary["blocking_reasons"]


def test_malformed_manifest_row_is_excluded(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [{"status": "cached"}], {})
    summary = _run(tmp_path, primary, [])

    assert summary["invalid_manifest_row_count"] == 1
    assert summary["output_manifest_row_count"] == 0


def test_output_restricted_to_reports(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [], {})

    with pytest.raises(ValueError, match="output_dir must be under reports"):
        consolidate_sec_text_caches(
            primary_cache_dir=primary,
            retry_cache_dirs=[],
            output_dir=tmp_path / "outside",
            reports_root=tmp_path / "reports",
        )


def test_no_network_imports_in_consolidation_script() -> None:
    source = Path("scripts/stock_alpha_news_transformer_consolidate_sec_text_caches.py").read_text(encoding="utf-8")

    assert "urllib" not in source
    assert "requests" not in source
    assert "urlopen" not in source


def test_repeated_consolidation_is_deterministic_and_sources_unchanged(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [_row("b|https://www.sec.gov/b.htm", "cached", "b.txt")], {"b.txt": "two"})
    retry = _cache(tmp_path, "retry", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "one"})
    before = (primary / CACHE_MANIFEST_FILENAME).read_text()

    first = _run(tmp_path, primary, [retry], "out")
    first_manifest = (tmp_path / "reports" / "out" / CACHE_MANIFEST_FILENAME).read_text()
    second = _run(tmp_path, primary, [retry], "out")
    second_manifest = (tmp_path / "reports" / "out" / CACHE_MANIFEST_FILENAME).read_text()

    assert first == second
    assert first_manifest == second_manifest
    assert (primary / CACHE_MANIFEST_FILENAME).read_text() == before


def test_multiple_retry_cache_directories_and_empty_retry_cache(tmp_path: Path) -> None:
    primary = _cache(tmp_path, "primary", [_row("a|https://www.sec.gov/a.htm", "cached", "a.txt")], {"a.txt": "one"})
    retry_one = _cache(tmp_path, "retry_one", [_row("b|https://www.sec.gov/b.htm", "cached", "b.txt")], {"b.txt": "two"})
    retry_two = _cache(tmp_path, "retry_two", [], {})

    summary = _run(tmp_path, primary, [retry_one, retry_two])

    assert summary["source_cache_directories"] == [str(primary), str(retry_one), str(retry_two)]
    assert summary["consolidated_success_count"] == 2
