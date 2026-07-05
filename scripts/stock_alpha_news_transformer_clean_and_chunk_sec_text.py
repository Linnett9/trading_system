from __future__ import annotations

import argparse
import codecs
import hashlib
import html
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CACHE_MANIFEST_FILENAME = "sec_primary_document_text_cache_manifest.json"
QUALITY_AUDIT_FILENAME = "sec_primary_document_text_quality_audit.json"
QUALITY_FLAGS_FILENAME = "sec_primary_document_text_quality_flags.jsonl"
CLEAN_MANIFEST_FILENAME = "sec_primary_document_text_clean_manifest.json"
CHUNKS_FILENAME = "sec_primary_document_text_chunks.jsonl"
SUMMARY_FILENAME = "sec_primary_document_text_clean_chunk_summary.json"
DOCUMENT_REPORT_FILENAME = "sec_primary_document_text_clean_document_report.jsonl"


@dataclass(frozen=True)
class CleanChunkConfig:
    chunk_size: int = 2_000
    chunk_overlap: int = 250
    max_chunks_per_document: int = 128
    suspiciously_short_chars: int = 2_000
    very_large_document_chars: int = 250_000
    extreme_document_chars: int = 1_000_000
    max_repeated_line_occurrences: int = 3
    max_table_run_lines: int = 80
    table_head_lines_to_keep: int = 40
    table_tail_lines_to_keep: int = 10

    def validate(self) -> None:
        if self.chunk_size < 500:
            raise ValueError("chunk_size must be at least 500")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if self.max_chunks_per_document < 2:
            raise ValueError("max_chunks_per_document must be at least 2")

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_chunks_per_document": self.max_chunks_per_document,
            "suspiciously_short_chars": self.suspiciously_short_chars,
            "very_large_document_chars": self.very_large_document_chars,
            "extreme_document_chars": self.extreme_document_chars,
            "max_repeated_line_occurrences": self.max_repeated_line_occurrences,
            "max_table_run_lines": self.max_table_run_lines,
            "table_head_lines_to_keep": self.table_head_lines_to_keep,
            "table_tail_lines_to_keep": self.table_tail_lines_to_keep,
        }


def clean_and_chunk_sec_text_cache(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    reports_root: str | Path = "reports",
    quality_audit_dir: str | Path | None = None,
    max_documents: int | None = None,
    config: CleanChunkConfig = CleanChunkConfig(),
) -> dict[str, Any]:
    config.validate()
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")

    manifest_rows = _read_manifest(cache_path / CACHE_MANIFEST_FILENAME)
    audit = _read_quality_audit(Path(quality_audit_dir)) if quality_audit_dir else {}
    quality_flags = _read_quality_flags(Path(quality_audit_dir)) if quality_audit_dir else {}
    effective_config = _config_from_audit(config, audit)
    selected_rows = _select_rows(manifest_rows, quality_flags, max_documents=max_documents)

    clean_manifest_rows: list[dict[str, Any]] = []
    document_reports: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    length_counts: list[int] = []
    chunk_count_values: list[int] = []
    duplicate_chunk_hashes: Counter[str] = Counter()
    aggregate = Counter()

    for row in selected_rows:
        result = _process_document(
            row,
            cache_dir=cache_path,
            quality_flags=quality_flags.get(_document_id(row), []),
            config=effective_config,
        )
        clean_manifest_rows.append(result["manifest_row"])
        document_reports.append(result["document_report"])
        chunk_rows.extend(result["chunks"])
        length_counts.extend(chunk["chunk_character_length"] for chunk in result["chunks"])
        chunk_count_values.append(len(result["chunks"]))
        duplicate_chunk_hashes.update(chunk["chunk_content_sha256"] for chunk in result["chunks"])
        aggregate.update(result["counters"])

    duplicate_chunk_hash_count = sum(count - 1 for count in duplicate_chunk_hashes.values() if count > 1)
    blocking_reasons = []
    if aggregate["missing_file"] or aggregate["failed_document"]:
        blocking_reasons.append("missing_or_failed_source_documents")
    summary = {
        "mode": "sec_primary_document_text_clean_and_chunk_report_only",
        "research_only": True,
        "input_cache_dir": str(cache_path),
        "quality_audit_dir": str(quality_audit_dir or ""),
        "output_dir": str(output_path),
        "documents_requested": len(selected_rows),
        "documents_processed": len(selected_rows),
        "documents_cleaned": aggregate["cleaned_document"],
        "chunks_created": len(chunk_rows),
        "chunk_count_average": _average(chunk_count_values),
        "chunk_count_percentiles": _percentiles(chunk_count_values),
        "documents_truncated": aggregate["truncated_document"],
        "source_characters": aggregate["source_characters"],
        "cleaned_characters": aggregate["cleaned_characters"],
        "characters_removed": max(0, aggregate["source_characters"] - aggregate["cleaned_characters"]),
        "boilerplate_reductions": aggregate["boilerplate_reductions"],
        "table_heavy_documents": aggregate["table_heavy_document"],
        "chunk_length_distribution": _percentiles(length_counts),
        "duplicate_chunk_hash_count": duplicate_chunk_hash_count,
        "missing_document_count": aggregate["missing_file"],
        "failed_document_count": aggregate["failed_document"],
        "cleaning_flag_counts": dict(sorted(aggregate.items())),
        "chunking_parameters": effective_config.payload(),
        "blocking_reasons": blocking_reasons,
        "recommended_next_step": "review_smoke_chunks_before_full_build_report_only",
        "next_allowed_step": "review_sec_text_clean_chunk_smoke_report",
        "model_training_started": False,
        "transformer_training_started": False,
        "finbert_execution_started": False,
        "embedding_generation_started": False,
        "price_labels_attached": False,
        "trading_impact": "none",
    }

    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path / CLEAN_MANIFEST_FILENAME, {"documents": clean_manifest_rows})
    _atomic_jsonl(output_path / CHUNKS_FILENAME, chunk_rows)
    _atomic_json(output_path / SUMMARY_FILENAME, summary)
    _atomic_jsonl(output_path / DOCUMENT_REPORT_FILENAME, document_reports)
    return summary


def _process_document(
    row: Mapping[str, Any],
    *,
    cache_dir: Path,
    quality_flags: list[str],
    config: CleanChunkConfig,
) -> dict[str, Any]:
    document_id = _document_id(row)
    source_path = _document_path(row, cache_dir)
    counters = Counter()
    if not source_path.exists():
        counters["missing_file"] += 1
        return _missing_result(row, source_path, quality_flags, counters)

    source_text, source_sha256 = _read_source_text(source_path)
    source_length = len(source_text)
    cleaned, cleaning_flags, cleaning_stats = _clean_text(source_text, quality_flags, config)
    cleaned_sha256 = _sha256_text(cleaned)
    candidate_chunks = _candidate_chunks(cleaned, config)
    retained_chunks, dropped = _retain_chunks(candidate_chunks, config.max_chunks_per_document)
    truncated = dropped > 0
    chunks = [
        _chunk_row(
            source_row=row,
            chunk=chunk,
            chunk_index=index,
            retained_count=len(retained_chunks),
            total_candidate_chunks=len(candidate_chunks),
            source_length=source_length,
            source_sha256=source_sha256,
            cleaned_sha256=cleaned_sha256,
            cleaning_flags=cleaning_flags,
            truncated=truncated,
            source_path=source_path,
        )
        for index, chunk in enumerate(retained_chunks)
    ]
    counters["cleaned_document"] += 1
    counters["source_characters"] += source_length
    counters["cleaned_characters"] += len(cleaned)
    counters["boilerplate_reductions"] += cleaning_stats["boilerplate_reductions"]
    counters["table_heavy_document"] += int("table_run_reduced" in cleaning_flags)
    counters["truncated_document"] += int(truncated)
    for flag in cleaning_flags:
        counters[flag] += 1

    manifest_row = {
        "document_id": document_id,
        "accession_number": str(row.get("accession_number") or ""),
        "symbols": _list_field(row.get("symbols")),
        "form_types": _list_field(row.get("form_types")),
        "event_timestamp": _event_timestamp(row),
        "availability_timestamp": _availability_timestamp(row),
        "source_cache_path": str(source_path),
        "primary_document_url": row.get("primary_document_url", ""),
        "source_content_sha256": source_sha256,
        "cleaned_content_sha256": cleaned_sha256,
        "source_character_length": source_length,
        "cleaned_character_length": len(cleaned),
        "cleaning_flags": cleaning_flags,
        "truncated_document": truncated,
        "total_candidate_chunks": len(candidate_chunks),
        "retained_chunk_count": len(retained_chunks),
        "dropped_chunk_count": dropped,
    }
    document_report = {
        **manifest_row,
        "characters_removed": max(0, source_length - len(cleaned)),
        "boilerplate_reductions": cleaning_stats["boilerplate_reductions"],
        "table_rows_removed": cleaning_stats["table_rows_removed"],
        "source_document_unchanged": True,
    }
    return {
        "manifest_row": manifest_row,
        "document_report": document_report,
        "chunks": chunks,
        "counters": counters,
    }


def _clean_text(
    text: str,
    inherited_quality_flags: list[str],
    config: CleanChunkConfig,
) -> tuple[str, list[str], dict[str, int]]:
    flags = set(inherited_quality_flags)
    stats = Counter()
    original = text
    text = text.replace("\x00", "")
    if text != original:
        flags.add("null_bytes_removed")
    text = _remove_invalid_controls(text)
    without_script = re.sub(r"(?is)<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", " ", text)
    if without_script != text:
        flags.add("script_or_style_removed")
    text = without_script
    without_tags = re.sub(r"(?is)<[^>]+>", " ", text)
    if without_tags != text:
        flags.add("markup_tags_removed")
    text = html.unescape(without_tags)
    text = _normalize_whitespace(text)
    lines, repeated_reductions = _reduce_repeated_lines(text.splitlines(), config.max_repeated_line_occurrences)
    stats["boilerplate_reductions"] += repeated_reductions
    if repeated_reductions:
        flags.add("repeated_line_reduced")
    lines, table_removed = _reduce_table_runs(lines, config)
    stats["table_rows_removed"] += table_removed
    if table_removed:
        flags.add("table_run_reduced")
    cleaned = _normalize_whitespace("\n".join(lines))
    if len(cleaned) < config.suspiciously_short_chars:
        flags.add("suspiciously_short")
    if len(cleaned) > config.very_large_document_chars:
        flags.add("very_large_document")
    if len(cleaned) > config.extreme_document_chars:
        flags.add("extreme_document")
    if _looks_boilerplate_heavy(cleaned):
        flags.add("likely_navigation_or_boilerplate")
    return cleaned, sorted(flags), dict(stats)


def _candidate_chunks(cleaned: str, config: CleanChunkConfig) -> list[dict[str, Any]]:
    if not cleaned:
        return []
    chunks = []
    start = 0
    text_length = len(cleaned)
    while start < text_length:
        target_end = min(text_length, start + config.chunk_size)
        end = _boundary_end(cleaned, start, target_end, config.chunk_size)
        if end <= start:
            end = target_end
        chunks.append(
            {
                "cleaned_start": start,
                "cleaned_end": end,
                "text": cleaned[start:end].strip(),
            }
        )
        if end >= text_length:
            break
        start = max(start + 1, end - config.chunk_overlap)
    return [chunk for chunk in chunks if chunk["text"]]


def _retain_chunks(
    chunks: Sequence[Mapping[str, Any]],
    max_chunks: int,
) -> tuple[list[Mapping[str, Any]], int]:
    if len(chunks) <= max_chunks:
        return list(chunks), 0
    head_count = max(1, max_chunks // 3)
    tail_count = max(1, max_chunks // 3)
    middle_count = max_chunks - head_count - tail_count
    selected = set(range(head_count))
    selected.update(range(len(chunks) - tail_count, len(chunks)))
    if middle_count > 0:
        span_start = head_count
        span_end = len(chunks) - tail_count - 1
        if span_end >= span_start:
            if middle_count == 1:
                selected.add((span_start + span_end) // 2)
            else:
                step = (span_end - span_start) / max(1, middle_count - 1)
                for index in range(middle_count):
                    selected.add(round(span_start + index * step))
    retained = [chunks[index] for index in sorted(selected)[:max_chunks]]
    return retained, len(chunks) - len(retained)


def _chunk_row(
    *,
    source_row: Mapping[str, Any],
    chunk: Mapping[str, Any],
    chunk_index: int,
    retained_count: int,
    total_candidate_chunks: int,
    source_length: int,
    source_sha256: str,
    cleaned_sha256: str,
    cleaning_flags: list[str],
    truncated: bool,
    source_path: Path,
) -> dict[str, Any]:
    text = str(chunk["text"])
    chunk_hash = _sha256_text(text)
    cleaned_start = int(chunk["cleaned_start"])
    cleaned_end = int(chunk["cleaned_end"])
    document_id = _document_id(source_row)
    return {
        "chunk_id": f"{_stable_id(document_id)}:{chunk_index:05d}:{chunk_hash[:12]}",
        "document_id": document_id,
        "accession_number": str(source_row.get("accession_number") or ""),
        "symbols": _list_field(source_row.get("symbols")),
        "form_types": _list_field(source_row.get("form_types")),
        "event_timestamp": _event_timestamp(source_row),
        "availability_timestamp": _availability_timestamp(source_row),
        "chunk_index": chunk_index,
        "source_character_start": min(cleaned_start, source_length),
        "source_character_end": min(cleaned_end, source_length),
        "cleaned_character_start": cleaned_start,
        "cleaned_character_end": cleaned_end,
        "chunk_text": text,
        "chunk_character_length": len(text),
        "source_content_sha256": source_sha256,
        "cleaned_content_sha256": cleaned_sha256,
        "chunk_content_sha256": chunk_hash,
        "cleaning_flags": cleaning_flags,
        "truncated_document": truncated,
        "total_candidate_chunks": total_candidate_chunks,
        "retained_chunk_count": retained_count,
        "source_cache_path": str(source_path),
    }


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    return [dict(row) for row in documents if isinstance(row, Mapping)]


def _read_quality_audit(path: Path) -> dict[str, Any]:
    audit_path = path / QUALITY_AUDIT_FILENAME
    return json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}


def _read_quality_flags(path: Path) -> dict[str, list[str]]:
    flag_path = path / QUALITY_FLAGS_FILENAME
    flags: dict[str, list[str]] = {}
    if not flag_path.exists():
        return flags
    with flag_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            flags[str(row.get("document_id") or "")] = _list_field(row.get("quality_flags"))
    return flags


def _config_from_audit(config: CleanChunkConfig, audit: Mapping[str, Any]) -> CleanChunkConfig:
    thresholds = dict(audit.get("quality_thresholds", {}) or {})
    max_chars = int(audit.get("recommended_max_character_limit") or config.very_large_document_chars)
    return CleanChunkConfig(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        max_chunks_per_document=config.max_chunks_per_document,
        suspiciously_short_chars=int(thresholds.get("suspiciously_short_chars", config.suspiciously_short_chars)),
        very_large_document_chars=max_chars,
        extreme_document_chars=int(thresholds.get("extreme_document_chars", config.extreme_document_chars)),
        max_repeated_line_occurrences=config.max_repeated_line_occurrences,
        max_table_run_lines=config.max_table_run_lines,
        table_head_lines_to_keep=config.table_head_lines_to_keep,
        table_tail_lines_to_keep=config.table_tail_lines_to_keep,
    )


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    flags_by_document_id: Mapping[str, list[str]],
    *,
    max_documents: int | None,
) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: _document_id(row))
    by_id = {_document_id(row): row for row in ordered}
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    priority_flags = (
        "suspiciously_short",
        "very_large_document",
        "extreme_document",
        "likely_navigation_or_boilerplate",
        "inline_xbrl_residue",
        "exact_duplicate_content",
    )
    for flag in priority_flags:
        per_flag_limit = 2 if flag == "exact_duplicate_content" else 1
        added_for_flag = 0
        for document_id in sorted(
            doc_id for doc_id, flags in flags_by_document_id.items() if flag in flags
        ):
            if document_id in by_id and document_id not in selected_ids:
                selected.append(by_id[document_id])
                selected_ids.add(document_id)
                added_for_flag += 1
                if added_for_flag >= per_flag_limit:
                    break
    for row in ordered:
        if max_documents is not None and len(selected) >= max_documents:
            break
        document_id = _document_id(row)
        if document_id not in selected_ids:
            selected.append(row)
            selected_ids.add(document_id)
    return selected if max_documents is None else selected[:max_documents]


def _read_source_text(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    parts: list[str] = []
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            parts.append(decoder.decode(chunk))
    parts.append(decoder.decode(b"", final=True))
    return "".join(parts), digest.hexdigest()


def _remove_invalid_controls(text: str) -> str:
    return "".join(
        char
        for char in text
        if char in "\n\r\t" or ord(char) >= 32
    )


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _reduce_repeated_lines(lines: Sequence[str], max_occurrences: int) -> tuple[list[str], int]:
    counts: Counter[str] = Counter()
    kept: list[str] = []
    dropped = 0
    for line in lines:
        key = re.sub(r"\s+", " ", line.strip().lower())
        if len(key) >= 20:
            counts[key] += 1
            if counts[key] > max_occurrences:
                dropped += 1
                continue
        kept.append(line)
    return kept, dropped


def _reduce_table_runs(lines: Sequence[str], config: CleanChunkConfig) -> tuple[list[str], int]:
    result: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        if not _table_like_line(lines[index]):
            result.append(lines[index])
            index += 1
            continue
        start = index
        while index < len(lines) and _table_like_line(lines[index]):
            index += 1
        run = list(lines[start:index])
        if len(run) <= config.max_table_run_lines:
            result.extend(run)
            continue
        head = run[: config.table_head_lines_to_keep]
        tail = run[-config.table_tail_lines_to_keep :]
        result.extend(head)
        result.append("[table rows omitted deterministically]")
        result.extend(tail)
        removed += len(run) - len(head) - len(tail)
    return result, removed


def _table_like_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 20:
        return False
    digit_count = sum(char.isdigit() for char in stripped)
    separator_count = sum(stripped.count(char) for char in ("|", "$", "%", "\t"))
    return digit_count >= 8 or separator_count >= 3


def _looks_boilerplate_heavy(text: str) -> bool:
    lowered = text.lower()
    phrases = (
        "table of contents",
        "united states securities and exchange commission",
        "indicate by check mark",
        "document and entity information",
        "not applicable",
    )
    return sum(lowered.count(phrase) for phrase in phrases) >= 8


def _boundary_end(text: str, start: int, target_end: int, chunk_size: int) -> int:
    lower_bound = start + max(500, int(chunk_size * 0.65))
    if target_end >= len(text):
        return len(text)
    window = text[lower_bound:target_end]
    for separator in ("\n\n", "\n", ". "):
        index = window.rfind(separator)
        if index >= 0:
            return lower_bound + index + len(separator)
    return target_end


def _document_path(row: Mapping[str, Any], cache_dir: Path) -> Path:
    value = Path(str(row.get("cache_path") or ""))
    if value.is_absolute() or (value.parts and value.parts[0] == "reports"):
        return value
    return cache_dir / "documents" / value.name


def _missing_result(
    row: Mapping[str, Any],
    source_path: Path,
    quality_flags: list[str],
    counters: Counter[str],
) -> dict[str, Any]:
    document_id = _document_id(row)
    flags = sorted(set([*quality_flags, "missing_file"]))
    manifest_row = {
        "document_id": document_id,
        "accession_number": str(row.get("accession_number") or ""),
        "symbols": _list_field(row.get("symbols")),
        "form_types": _list_field(row.get("form_types")),
        "event_timestamp": _event_timestamp(row),
        "source_cache_path": str(source_path),
        "primary_document_url": row.get("primary_document_url", ""),
        "source_content_sha256": "",
        "cleaned_content_sha256": "",
        "source_character_length": 0,
        "cleaned_character_length": 0,
        "cleaning_flags": flags,
        "truncated_document": False,
        "total_candidate_chunks": 0,
        "retained_chunk_count": 0,
        "dropped_chunk_count": 0,
    }
    return {
        "manifest_row": manifest_row,
        "document_report": {**manifest_row, "characters_removed": 0, "source_document_unchanged": True},
        "chunks": [],
        "counters": counters,
    }


def _document_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("document_id") or "").strip()
    if value:
        return value
    accession = str(row.get("accession_number") or "").strip()
    url = str(row.get("primary_document_url") or "").strip()
    return f"{accession}|{url}" if accession or url else ""


def _event_timestamp(row: Mapping[str, Any]) -> str:
    for key in ("event_timestamp", "accepted_datetime", "published_at_utc"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    for event_key in _list_field(row.get("event_keys")):
        match = re.search(r"(20\d{2}-\d{2}-\d{2}T[^|]+Z)", event_key)
        if match:
            return match.group(1)
    return ""


def _availability_timestamp(row: Mapping[str, Any]) -> str:
    for key in ("availability_timestamp", "available_at_timestamp", "available_at_utc", "collected_at_utc"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    return [text] if text else []


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _average(values: Sequence[int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentiles(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "min": ordered[0] if ordered else 0,
        "p05": _percentile(ordered, 0.05),
        "median": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1] if ordered else 0,
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
    return int(round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def _is_under_reports(path: Path, reports_root: Path) -> bool:
    try:
        path.resolve().relative_to(reports_root.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean and chunk SEC primary-document text offline.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--quality-audit-dir", default="")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--max-chunks-per-document", type=int, default=128)
    args = parser.parse_args(argv)
    summary = clean_and_chunk_sec_text_cache(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
        quality_audit_dir=args.quality_audit_dir or None,
        max_documents=args.max_documents,
        config=CleanChunkConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks_per_document=args.max_chunks_per_document,
        ),
    )
    print(f"documents_processed={summary['documents_processed']}")
    print(f"documents_cleaned={summary['documents_cleaned']}")
    print(f"chunks_created={summary['chunks_created']}")
    print(f"documents_truncated={summary['documents_truncated']}")
    print(f"duplicate_chunk_hash_count={summary['duplicate_chunk_hash_count']}")
    print(f"blocking_reasons={summary['blocking_reasons']}")
    print(f"next_allowed_step={summary['next_allowed_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
