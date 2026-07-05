import hashlib
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_audit_sec_text_quality import (
    QualityThresholds,
    audit_sec_primary_document_text_quality,
)


def _write_cache(tmp_path: Path, specs: list[dict]) -> tuple[Path, Path]:
    cache_dir = tmp_path / "reports" / "cache"
    documents_dir = cache_dir / "documents"
    documents_dir.mkdir(parents=True)
    rows = []
    for index, spec in enumerate(specs, 1):
        accession = spec.get("accession", f"000000000{index}-24-00000{index}")
        filename = f"{accession}_{index}.txt"
        path = documents_dir / filename
        if "bytes" in spec:
            path.write_bytes(spec["bytes"])
        elif "text" in spec:
            path.write_text(spec["text"], encoding="utf-8")
        cache_path = path if not spec.get("missing") else documents_dir / f"missing_{filename}"
        rows.append(
            {
                "document_id": spec.get("document_id", f"{accession}|https://www.sec.gov/doc-{index}.htm"),
                "accession_number": accession,
                "cache_path": str(cache_path),
                "content_sha256": _sha256(cache_path) if cache_path.exists() else "",
                "text_length": cache_path.stat().st_size if cache_path.exists() else 0,
                "form_types": spec.get("form_types", ["8-K"]),
                "symbols": spec.get("symbols", ["AAA"]),
                "primary_document_url": f"https://www.sec.gov/doc-{index}.htm",
                "status": "cached",
            }
        )
    (cache_dir / "sec_primary_document_text_cache_manifest.json").write_text(
        json.dumps({"documents": rows}),
        encoding="utf-8",
    )
    return cache_dir, tmp_path / "reports" / "audit"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(tmp_path: Path, specs: list[dict], thresholds: QualityThresholds | None = None) -> dict:
    cache_dir, output_dir = _write_cache(tmp_path, specs)
    return audit_sec_primary_document_text_quality(
        cache_dir=cache_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
        thresholds=thresholds or QualityThresholds(),
    )


def _flags(tmp_path: Path) -> list[dict]:
    path = tmp_path / "reports" / "audit" / "sec_primary_document_text_quality_flags.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_quality_audit_accepts_normal_readable_text(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"text": "Normal SEC filing text. " * 160}])

    assert audit["readable_document_count"] == 1
    assert audit["failure_counts_by_quality_flag"]["suspiciously_short"] == 0
    assert audit["blocking_reasons"] == []


def test_quality_audit_flags_missing_file(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"missing": True}])

    assert audit["missing_document_count"] == 1
    assert audit["blocking_reasons"] == ["missing_document_files"]
    assert _flags(tmp_path)[0]["quality_flags"] == ["missing_file"]


def test_quality_audit_flags_empty_file(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"text": ""}])

    assert audit["empty_document_count"] == 1
    assert "empty_document_text" in audit["blocking_reasons"]
    assert "empty_text" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_suspiciously_short_text(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"text": "tiny filing"}])

    assert audit["suspiciously_short_count"] == 1
    assert "suspiciously_short" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_very_large_text_without_blocking(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"text": "A" * 250_001}])

    assert audit["very_large_document_count"] == 1
    assert audit["blocking_reasons"] == []
    assert "very_large_document" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_html_residue(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"text": "<html><body><table>Residual</table></body></html>" + (" text" * 500)}])

    assert audit["html_residue_count"] == 1
    assert "html_residue" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_inline_xbrl_residue(tmp_path: Path) -> None:
    text = "<ix:nonFraction name='us-gaap:Assets'>1</ix:nonFraction>" + (" text" * 500)
    audit = _audit(tmp_path, [{"text": text}])

    assert audit["inline_xbrl_residue_count"] == 1
    assert "inline_xbrl_residue" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_encoding_replacement_characters(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"bytes": b"\xff" + (b"A" * 2_100)}])

    assert audit["encoding_replacement_character_count"] == 1
    assert "encoding_replacement_characters" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_null_bytes(tmp_path: Path) -> None:
    audit = _audit(tmp_path, [{"bytes": b"A" * 2_100 + b"\x00"}])

    assert audit["null_byte_count"] == 1
    assert "null_bytes" in _flags(tmp_path)[0]["quality_flags"]


def test_quality_audit_flags_exact_duplicate_content(tmp_path: Path) -> None:
    text = "Duplicated SEC filing text. " * 140
    audit = _audit(tmp_path, [{"text": text}, {"text": text}])

    assert audit["exact_duplicate_text_count"] == 1
    assert audit["duplicate_content_hash_count"] == 1
    assert all("exact_duplicate_content" in row["quality_flags"] for row in _flags(tmp_path))


def test_quality_audit_percentiles_are_deterministic(tmp_path: Path) -> None:
    thresholds = QualityThresholds(suspiciously_short_chars=1)
    first = _audit(
        tmp_path,
        [{"text": "A" * 10}, {"text": "B" * 20}, {"text": "C" * 30}],
        thresholds=thresholds,
    )
    second = audit_sec_primary_document_text_quality(
        cache_dir=tmp_path / "reports" / "cache",
        output_dir=tmp_path / "reports" / "audit",
        reports_root=tmp_path / "reports",
        thresholds=thresholds,
    )

    assert first["text_length_median"] == 20
    assert first == second


def test_quality_audit_refuses_output_outside_reports(tmp_path: Path) -> None:
    cache_dir, _ = _write_cache(tmp_path, [{"text": "Normal SEC filing text. " * 160}])

    with pytest.raises(ValueError, match="output_dir must be under reports"):
        audit_sec_primary_document_text_quality(
            cache_dir=cache_dir,
            output_dir=tmp_path / "outside",
            reports_root=tmp_path / "reports",
        )


def test_quality_audit_does_not_modify_source_text(tmp_path: Path) -> None:
    cache_dir, output_dir = _write_cache(tmp_path, [{"text": "Immutable SEC filing text. " * 160}])
    source = next((cache_dir / "documents").glob("*.txt"))
    before = source.read_bytes()

    audit_sec_primary_document_text_quality(
        cache_dir=cache_dir,
        output_dir=output_dir,
        reports_root=tmp_path / "reports",
    )

    assert source.read_bytes() == before


def test_quality_audit_has_no_network_or_training_imports() -> None:
    source = Path("scripts/stock_alpha_news_transformer_audit_sec_text_quality.py").read_text(encoding="utf-8")

    assert "urllib" not in source
    assert "requests" not in source
    assert "http.client" not in source
    assert "socket" not in source
    assert "torch" not in source
    assert "sklearn" not in source
    assert "broker" not in source
    assert "paper_trading" not in source
