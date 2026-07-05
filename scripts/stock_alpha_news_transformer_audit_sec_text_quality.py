from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_FILENAME = "sec_primary_document_text_cache_manifest.json"
AUDIT_FILENAME = "sec_primary_document_text_quality_audit.json"
FLAGS_FILENAME = "sec_primary_document_text_quality_flags.jsonl"
LENGTHS_FILENAME = "sec_primary_document_text_quality_lengths.csv"

QUALITY_FLAGS = (
    "missing_file",
    "unreadable_file",
    "empty_text",
    "suspiciously_short",
    "very_large_document",
    "extreme_document",
    "html_residue",
    "inline_xbrl_residue",
    "script_or_style_residue",
    "encoding_replacement_characters",
    "null_bytes",
    "high_non_alphanumeric_ratio",
    "high_whitespace_ratio",
    "likely_navigation_or_boilerplate",
    "exact_duplicate_content",
    "unknown_quality_issue",
)


@dataclass(frozen=True)
class QualityThresholds:
    suspiciously_short_chars: int = 2_000
    very_large_document_chars: int = 250_000
    extreme_document_chars: int = 1_000_000
    high_non_alphanumeric_ratio: float = 0.35
    high_whitespace_ratio: float = 0.45
    navigation_phrase_occurrences: int = 8
    repeated_boilerplate_occurrences: int = 12
    chunk_bytes: int = 1024 * 1024

    def payload(self) -> dict[str, Any]:
        return {
            "suspiciously_short_chars": self.suspiciously_short_chars,
            "very_large_document_chars": self.very_large_document_chars,
            "extreme_document_chars": self.extreme_document_chars,
            "high_non_alphanumeric_ratio": self.high_non_alphanumeric_ratio,
            "high_whitespace_ratio": self.high_whitespace_ratio,
            "navigation_phrase_occurrences": self.navigation_phrase_occurrences,
            "repeated_boilerplate_occurrences": self.repeated_boilerplate_occurrences,
            "chunk_bytes": self.chunk_bytes,
        }


def audit_sec_primary_document_text_quality(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    reports_root: str | Path = "reports",
    thresholds: QualityThresholds = QualityThresholds(),
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    rows = _read_manifest(cache_path / MANIFEST_FILENAME)
    document_file_count = _count_text_files(cache_path)
    duplicate_document_id_count = _duplicate_count(_document_id(row) for row in rows)

    records: list[dict[str, Any]] = []
    hash_counts: Counter[str] = Counter()
    lengths: list[int] = []
    counts_by_form_type: Counter[str] = Counter()
    counts_by_year: Counter[str] = Counter()
    counts_by_symbol: Counter[str] = Counter()

    for index, row in enumerate(rows):
        record = _scan_manifest_row(
            row,
            cache_dir=cache_path,
            thresholds=thresholds,
            row_index=index,
        )
        records.append(record)
        if record["readable"]:
            lengths.append(record["text_length"])
            if record["content_sha256"]:
                hash_counts[record["content_sha256"]] += 1
        counts_by_form_type.update(record["form_types"])
        counts_by_year.update([record["year"]] if record["year"] else [])
        counts_by_symbol.update(record["symbols"])

    duplicate_hashes = {value for value, count in hash_counts.items() if count > 1}
    for record in records:
        if record["content_sha256"] in duplicate_hashes:
            record["quality_flags"].append("exact_duplicate_content")

    failure_counts = Counter(flag for record in records for flag in record["quality_flags"])
    blocking_reasons = _blocking_reasons(records)
    audit = {
        "mode": "sec_primary_document_text_quality_audit_report_only",
        "research_only": True,
        "input_cache_dir": str(cache_path),
        "manifest_row_count": len(rows),
        "document_file_count": document_file_count,
        "readable_document_count": sum(1 for record in records if record["readable"]),
        "missing_document_count": failure_counts.get("missing_file", 0),
        "unreadable_document_count": failure_counts.get("unreadable_file", 0),
        "empty_document_count": failure_counts.get("empty_text", 0),
        "duplicate_document_id_count": duplicate_document_id_count,
        "duplicate_content_hash_count": sum(count - 1 for count in hash_counts.values() if count > 1),
        "duplicate_content_hash_group_count": len(duplicate_hashes),
        "exact_duplicate_text_count": sum(count - 1 for count in hash_counts.values() if count > 1),
        **_length_percentiles(lengths),
        "suspiciously_short_count": failure_counts.get("suspiciously_short", 0),
        "very_large_document_count": failure_counts.get("very_large_document", 0),
        "extreme_document_count": failure_counts.get("extreme_document", 0),
        "html_residue_count": failure_counts.get("html_residue", 0),
        "inline_xbrl_residue_count": failure_counts.get("inline_xbrl_residue", 0),
        "script_or_style_residue_count": failure_counts.get("script_or_style_residue", 0),
        "encoding_replacement_character_count": failure_counts.get("encoding_replacement_characters", 0),
        "null_byte_count": failure_counts.get("null_bytes", 0),
        "high_non_alphanumeric_ratio_count": failure_counts.get("high_non_alphanumeric_ratio", 0),
        "high_whitespace_ratio_count": failure_counts.get("high_whitespace_ratio", 0),
        "likely_navigation_or_boilerplate_count": failure_counts.get("likely_navigation_or_boilerplate", 0),
        "failure_counts_by_quality_flag": {flag: failure_counts.get(flag, 0) for flag in QUALITY_FLAGS},
        "counts_by_form_type": dict(sorted(counts_by_form_type.items())),
        "counts_by_year": dict(sorted(counts_by_year.items())),
        "counts_by_symbol": dict(sorted(counts_by_symbol.items())),
        "top_20_shortest_documents": _top_by_length(records, shortest=True),
        "top_20_longest_documents": _top_by_length(records, shortest=False),
        "quality_thresholds": thresholds.payload(),
        "blocking_reasons": blocking_reasons,
        "warnings": _warnings(failure_counts),
        "recommended_cleaning_strategy": _recommended_cleaning_strategy(),
        "recommended_max_character_limit": thresholds.very_large_document_chars,
        "recommended_tokenization_strategy": (
            "Chunk long filings deterministically after cleaning, preserve accession/document_id/"
            "content hashes, retain leading business/context sections and final signature/exhibit "
            "context, and use fixed-size token windows with stable overlap."
        ),
        "next_allowed_step": (
            "design_deterministic_sec_text_cleaner_report_only"
            if not blocking_reasons
            else "resolve_sec_primary_text_quality_blockers"
        ),
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }

    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / AUDIT_FILENAME, audit)
    _write_flags(output_path / FLAGS_FILENAME, records)
    _write_lengths(output_path / LENGTHS_FILENAME, records)
    return audit


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    return [dict(row) for row in documents if isinstance(row, Mapping)]


def _scan_manifest_row(
    row: Mapping[str, Any],
    *,
    cache_dir: Path,
    thresholds: QualityThresholds,
    row_index: int,
) -> dict[str, Any]:
    path = _document_path(row, cache_dir)
    base = _base_record(row, path, row_index)
    if not path.exists():
        base["quality_flags"].append("missing_file")
        return base
    try:
        stats = _scan_text_file(path, thresholds)
    except OSError as exc:
        base["quality_flags"].extend(["unreadable_file"])
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    base.update(stats)
    base["readable"] = True
    base["quality_flags"].extend(_quality_flags(stats, thresholds, row))
    expected_hash = str(row.get("content_sha256") or "")
    if expected_hash and expected_hash != stats["content_sha256"]:
        base["quality_flags"].append("unknown_quality_issue")
        base["hash_mismatch"] = True
    return base


def _scan_text_file(path: Path, thresholds: QualityThresholds) -> dict[str, Any]:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text_length = 0
    replacement_count = 0
    null_count = 0
    whitespace_count = 0
    non_alnum_count = 0
    html_residue = False
    inline_xbrl_residue = False
    script_or_style_residue = False
    navigation_occurrences = 0
    boilerplate_occurrences = 0
    tail = ""

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(thresholds.chunk_bytes), b""):
            digest.update(chunk)
            text_length += len(chunk)
            null_count += chunk.count(b"\x00")
            alnum_count = _ascii_alnum_count(chunk)
            whitespace = _ascii_whitespace_count(chunk)
            whitespace_count += whitespace
            non_alnum_count += max(0, len(chunk) - alnum_count - whitespace)
            text = decoder.decode(chunk)
            replacement_count += text.count("\ufffd")
            window = (tail + text).lower()
            html_residue = html_residue or bool(re.search(r"<\s*/?\s*(html|body|table)\b", window))
            inline_xbrl_residue = inline_xbrl_residue or bool(re.search(r"<\s*/?\s*ix:|xmlns:ix|inlinexbrl", window))
            script_or_style_residue = script_or_style_residue or bool(re.search(r"<\s*/?\s*(script|style)\b", window))
            navigation_occurrences += sum(window.count(phrase) for phrase in _navigation_phrases())
            boilerplate_occurrences += sum(window.count(phrase) for phrase in _boilerplate_phrases())
            tail = window[-512:]
    final = decoder.decode(b"", final=True)
    if final:
        replacement_count += final.count("\ufffd")
    return {
        "text_length": text_length,
        "content_sha256": digest.hexdigest(),
        "encoding_replacement_character_total": replacement_count,
        "null_byte_total": null_count,
        "non_alphanumeric_ratio": (non_alnum_count / text_length) if text_length else 0.0,
        "whitespace_ratio": (whitespace_count / text_length) if text_length else 0.0,
        "html_residue": html_residue,
        "inline_xbrl_residue": inline_xbrl_residue,
        "script_or_style_residue": script_or_style_residue,
        "navigation_or_boilerplate_occurrences": navigation_occurrences + boilerplate_occurrences,
    }


def _quality_flags(
    stats: Mapping[str, Any],
    thresholds: QualityThresholds,
    row: Mapping[str, Any],
) -> list[str]:
    del row
    flags: list[str] = []
    length = int(stats["text_length"])
    if length == 0:
        flags.append("empty_text")
    elif length < thresholds.suspiciously_short_chars:
        flags.append("suspiciously_short")
    if length > thresholds.very_large_document_chars:
        flags.append("very_large_document")
    if length > thresholds.extreme_document_chars:
        flags.append("extreme_document")
    if stats["html_residue"]:
        flags.append("html_residue")
    if stats["inline_xbrl_residue"]:
        flags.append("inline_xbrl_residue")
    if stats["script_or_style_residue"]:
        flags.append("script_or_style_residue")
    if int(stats["encoding_replacement_character_total"]) > 0:
        flags.append("encoding_replacement_characters")
    if int(stats["null_byte_total"]) > 0:
        flags.append("null_bytes")
    if float(stats["non_alphanumeric_ratio"]) > thresholds.high_non_alphanumeric_ratio:
        flags.append("high_non_alphanumeric_ratio")
    if float(stats["whitespace_ratio"]) > thresholds.high_whitespace_ratio:
        flags.append("high_whitespace_ratio")
    if int(stats["navigation_or_boilerplate_occurrences"]) >= thresholds.navigation_phrase_occurrences:
        flags.append("likely_navigation_or_boilerplate")
    return flags


def _base_record(row: Mapping[str, Any], path: Path, row_index: int) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "document_id": _document_id(row),
        "accession_number": str(row.get("accession_number") or ""),
        "symbols": _list_field(row.get("symbols")),
        "form_types": _list_field(row.get("form_types")),
        "year": _year(row),
        "cache_path": str(path),
        "manifest_text_length": row.get("text_length"),
        "manifest_content_sha256": row.get("content_sha256"),
        "primary_document_url": row.get("primary_document_url", ""),
        "source_cache": row.get("source_cache", ""),
        "source_manifest": row.get("source_manifest", ""),
        "readable": False,
        "text_length": 0,
        "content_sha256": "",
        "quality_flags": [],
    }


def _document_path(row: Mapping[str, Any], cache_dir: Path) -> Path:
    value = Path(str(row.get("cache_path") or ""))
    if value.is_absolute() or (value.parts and value.parts[0] == "reports"):
        return value
    candidate = cache_dir / "documents" / value.name
    if candidate.exists() or value.name:
        return candidate
    return cache_dir / "documents" / ""


def _document_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("document_id") or "").strip()
    if value:
        return value
    accession = str(row.get("accession_number") or "").strip()
    url = str(row.get("primary_document_url") or "").strip()
    return f"{accession}|{url}" if accession or url else ""


def _year(row: Mapping[str, Any]) -> str:
    accession = str(row.get("accession_number") or "")
    match = re.search(r"-(\d{2})-", accession)
    if match:
        return f"20{match.group(1)}"
    for event_key in _list_field(row.get("event_keys")):
        match = re.search(r"\b(20\d{2})-\d{2}-\d{2}", event_key)
        if match:
            return match.group(1)
    return ""


def _length_percentiles(lengths: Sequence[int]) -> dict[str, int]:
    ordered = sorted(lengths)
    return {
        "text_length_min": ordered[0] if ordered else 0,
        "text_length_p01": _percentile(ordered, 0.01),
        "text_length_p05": _percentile(ordered, 0.05),
        "text_length_median": _percentile(ordered, 0.50),
        "text_length_p95": _percentile(ordered, 0.95),
        "text_length_p99": _percentile(ordered, 0.99),
        "text_length_max": ordered[-1] if ordered else 0,
    }


def _percentile(ordered: Sequence[int], quantile: float) -> int:
    if not ordered:
        return 0
    if len(ordered) == 1:
        return int(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return int(round(value))


def _top_by_length(records: Sequence[Mapping[str, Any]], *, shortest: bool) -> list[dict[str, Any]]:
    readable = [record for record in records if record.get("readable")]
    ordered = sorted(
        readable,
        key=lambda record: (
            int(record.get("text_length", 0)),
            str(record.get("document_id", "")),
        ),
        reverse=not shortest,
    )
    return [
        {
            "document_id": record["document_id"],
            "accession_number": record["accession_number"],
            "symbols": record["symbols"],
            "form_types": record["form_types"],
            "text_length": record["text_length"],
            "quality_flags": record["quality_flags"],
        }
        for record in ordered[:20]
    ]


def _blocking_reasons(records: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons = []
    flags = Counter(flag for record in records for flag in record["quality_flags"])
    if flags.get("missing_file"):
        reasons.append("missing_document_files")
    if flags.get("unreadable_file"):
        reasons.append("unreadable_document_files")
    if flags.get("empty_text"):
        reasons.append("empty_document_text")
    if flags.get("unknown_quality_issue"):
        reasons.append("unknown_quality_issues")
    return reasons


def _warnings(failure_counts: Mapping[str, int]) -> list[str]:
    warnings = []
    for flag in (
        "suspiciously_short",
        "very_large_document",
        "extreme_document",
        "html_residue",
        "inline_xbrl_residue",
        "high_non_alphanumeric_ratio",
        "likely_navigation_or_boilerplate",
    ):
        count = int(failure_counts.get(flag, 0))
        if count:
            warnings.append(f"{flag}: {count}")
    return warnings


def _recommended_cleaning_strategy() -> list[str]:
    return [
        "Remove residual HTML/XML/inline-XBRL tags with a deterministic parser before tokenization.",
        "Normalize repeated whitespace after preserving paragraph and table boundaries.",
        "Detect repeated filing headers, navigation phrases, and SEC boilerplate; remove only repeated duplicates.",
        "Handle tables separately: preserve compact captions/key-value rows and drop pure layout residue.",
        "Use form-specific section selection for 10-K/10-Q/8-K/6-K while retaining provenance fields.",
        "For long filings, chunk after cleaning rather than marking the source unusable.",
        "Keep deterministic hashes for original text and cleaned text, plus source cache path and accession.",
    ]


def _write_flags(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        {
            "document_id": record["document_id"],
            "accession_number": record["accession_number"],
            "symbols": record["symbols"],
            "form_types": record["form_types"],
            "text_length": record["text_length"],
            "quality_flags": record["quality_flags"],
            "cache_path": record["cache_path"],
        }
        for record in records
        if record["quality_flags"]
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_lengths(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "document_id",
                "accession_number",
                "symbols",
                "form_types",
                "text_length",
                "quality_flags",
            ],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: str(item["document_id"])):
            writer.writerow(
                {
                    "document_id": record["document_id"],
                    "accession_number": record["accession_number"],
                    "symbols": "|".join(record["symbols"]),
                    "form_types": "|".join(record["form_types"]),
                    "text_length": record["text_length"],
                    "quality_flags": "|".join(record["quality_flags"]),
                }
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _count_text_files(cache_dir: Path) -> int:
    documents_dir = cache_dir / "documents"
    return len(list(documents_dir.glob("*.txt"))) if documents_dir.exists() else 0


def _duplicate_count(values: Sequence[str] | Any) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


def _list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    return [text] if text else []


def _navigation_phrases() -> tuple[str, ...]:
    return (
        "table of contents",
        "back to contents",
        "skip to main content",
        "document and entity information",
        "cover page interactive data",
    )


def _boilerplate_phrases() -> tuple[str, ...]:
    return (
        "united states securities and exchange commission",
        "indicate by check mark",
        "not applicable",
    )


def _ascii_alnum_count(chunk: bytes) -> int:
    return sum(chunk.count(value) for value in _ASCII_ALNUM_BYTES)


def _ascii_whitespace_count(chunk: bytes) -> int:
    return sum(chunk.count(value) for value in b" \t\r\n\f\v")


_ASCII_ALNUM_BYTES = bytes(
    list(range(ord("0"), ord("9") + 1))
    + list(range(ord("A"), ord("Z") + 1))
    + list(range(ord("a"), ord("z") + 1))
)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit consolidated SEC primary-document text quality offline.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args(argv)
    audit = audit_sec_primary_document_text_quality(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(f"total_documents={audit['manifest_row_count']}")
    print(f"missing={audit['missing_document_count']} unreadable={audit['unreadable_document_count']} empty={audit['empty_document_count']}")
    print(
        "length_percentiles="
        f"min:{audit['text_length_min']} p01:{audit['text_length_p01']} "
        f"p05:{audit['text_length_p05']} median:{audit['text_length_median']} "
        f"p95:{audit['text_length_p95']} p99:{audit['text_length_p99']} "
        f"max:{audit['text_length_max']}"
    )
    print(
        "quality_counts="
        f"short:{audit['suspiciously_short_count']} "
        f"very_large:{audit['very_large_document_count']} "
        f"extreme:{audit['extreme_document_count']} "
        f"html:{audit['html_residue_count']} "
        f"xbrl:{audit['inline_xbrl_residue_count']} "
        f"duplicates:{audit['exact_duplicate_text_count']}"
    )
    print(f"blocking_reasons={audit['blocking_reasons']}")
    print(f"next_allowed_step={audit['next_allowed_step']}")
    print(f"recommended_cleaning_strategy={audit['recommended_cleaning_strategy'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
