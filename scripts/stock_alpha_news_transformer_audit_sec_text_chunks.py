from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_FILENAME = "sec_primary_document_text_clean_chunk_summary.json"
MANIFEST_FILENAME = "sec_primary_document_text_clean_manifest.json"
CHUNKS_FILENAME = "sec_primary_document_text_chunks.jsonl"
DOCUMENT_REPORT_FILENAME = "sec_primary_document_text_clean_document_report.jsonl"
AUDIT_FILENAME = "sec_primary_document_text_chunk_audit.json"
DUPLICATES_FILENAME = "sec_primary_document_text_duplicate_chunks.jsonl"
TRUNCATION_FILENAME = "sec_primary_document_text_truncation_audit.jsonl"
READINESS_FILENAME = "sec_primary_document_text_full_build_readiness.json"


def audit_sec_text_chunks(
    *,
    chunk_dir: str | Path,
    output_dir: str | Path,
    reports_root: str | Path = "reports",
) -> dict[str, Any]:
    chunk_path = Path(chunk_dir)
    output_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    summary = _read_json(chunk_path / SUMMARY_FILENAME)
    manifest = _read_manifest(chunk_path / MANIFEST_FILENAME)
    document_reports = _read_jsonl(chunk_path / DOCUMENT_REPORT_FILENAME)
    chunks, validation = _read_and_validate_chunks(chunk_path / CHUNKS_FILENAME, manifest)
    duplicates = _duplicate_audit(chunks, manifest)
    truncation_rows = _truncation_audit(manifest, chunks, summary)
    provenance = _provenance_audit(chunks, manifest)
    readiness = _readiness(summary, validation, duplicates, truncation_rows, provenance)
    audit = {
        "mode": "sec_primary_document_text_chunk_audit_report_only",
        "research_only": True,
        "input_chunk_dir": str(chunk_path),
        "document_count": len(manifest),
        "document_report_count": len(document_reports),
        "total_chunks": len(chunks),
        "unique_chunk_hashes": duplicates["unique_chunk_hashes"],
        "duplicate_hash_groups": duplicates["duplicate_hash_groups"],
        "duplicate_chunk_instances": duplicates["duplicate_chunk_instances"],
        "duplicate_classification_counts": duplicates["classification_counts"],
        "within_document_duplicate_count": duplicates["within_document_duplicate_count"],
        "cross_document_duplicate_count": duplicates["cross_document_duplicate_count"],
        "duplicate_groups_with_different_symbols_or_forms": duplicates["groups_with_different_symbols_or_forms"],
        "top_duplicate_groups": duplicates["top_duplicate_groups"],
        "duplicate_handling_recommendation": duplicates["handling_recommendation"],
        "truncated_document_count": len(truncation_rows),
        "truncation_summary": _truncation_summary(truncation_rows),
        "boundary_validation": validation,
        "provenance_audit": provenance,
        "full_build_readiness": readiness,
        "chunking_parameters": dict(summary.get("chunking_parameters", {}) or {}),
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path / AUDIT_FILENAME, audit)
    _atomic_jsonl(output_path / DUPLICATES_FILENAME, duplicates["duplicate_groups"])
    _atomic_jsonl(output_path / TRUNCATION_FILENAME, truncation_rows)
    _atomic_json(output_path / READINESS_FILENAME, readiness)
    return audit


def _read_and_validate_chunks(path: Path, manifest: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_ids: Counter[str] = Counter()
    hash_mismatches = 0
    length_mismatches = 0
    invalid_offsets = 0
    over_limit_chunks = 0
    cleaned_hash_inconsistencies = 0
    source_hash_inconsistencies = 0
    missing_doc = 0
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_chunk_size = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("chunk_text", ""))
            row["_normalized_chunk_text_hash"] = hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()
            chunks.append(row)
            document_id = str(row.get("document_id", ""))
            chunks_by_doc[document_id].append(row)
            chunk_ids[str(row.get("chunk_id", ""))] += 1
            max_chunk_size = max(max_chunk_size, int(row.get("chunk_character_length") or 0))
            if int(row.get("chunk_character_length") or -1) != len(text):
                length_mismatches += 1
            if str(row.get("chunk_content_sha256", "")) != hashlib.sha256(text.encode("utf-8")).hexdigest():
                hash_mismatches += 1
            if int(row.get("chunk_character_length") or 0) > _chunk_size_limit(row):
                over_limit_chunks += 1
            if _offsets_invalid(row):
                invalid_offsets += 1
            doc = manifest.get(document_id)
            if not doc:
                missing_doc += 1
            else:
                if row.get("cleaned_content_sha256") != doc.get("cleaned_content_sha256"):
                    cleaned_hash_inconsistencies += 1
                if row.get("source_content_sha256") != doc.get("source_content_sha256"):
                    source_hash_inconsistencies += 1

    overlap = _overlap_audit(chunks_by_doc)
    return chunks, {
        "duplicate_chunk_id_count": sum(count - 1 for count in chunk_ids.values() if count > 1),
        "chunk_hash_mismatch_count": hash_mismatches,
        "chunk_length_mismatch_count": length_mismatches,
        "invalid_offset_count": invalid_offsets,
        "chunk_text_over_limit_count": over_limit_chunks,
        "cleaned_hash_inconsistency_count": cleaned_hash_inconsistencies,
        "source_hash_inconsistency_count": source_hash_inconsistencies,
        "chunks_missing_manifest_document_count": missing_doc,
        "max_chunk_character_length": max_chunk_size,
        **overlap,
    }


def _duplicate_audit(chunks: Sequence[Mapping[str, Any]], manifest: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        groups[str(chunk.get("chunk_content_sha256", ""))].append(chunk)
    duplicate_rows = []
    class_counts = Counter()
    within_count = 0
    cross_count = 0
    different_symbol_form = 0
    for hash_value, items in groups.items():
        if len(items) < 2:
            continue
        docs = {str(item.get("document_id", "")) for item in items}
        source_hashes = {str(manifest.get(doc, {}).get("source_content_sha256", "")) for doc in docs}
        symbols = {tuple(item.get("symbols") or []) for item in items}
        forms = {tuple(item.get("form_types") or []) for item in items}
        classification = _duplicate_classification(items, docs, source_hashes)
        duplicate_instances = len(items) - 1
        class_counts[classification] += duplicate_instances
        within_count += duplicate_instances if len(docs) == 1 else 0
        cross_count += duplicate_instances if len(docs) > 1 else 0
        different = len(symbols) > 1 or len(forms) > 1
        different_symbol_form += int(different)
        duplicate_rows.append(
            {
                "chunk_content_sha256": hash_value,
                "instance_count": len(items),
                "duplicate_instances": duplicate_instances,
                "classification": classification,
                "document_ids": sorted(docs),
                "symbols": sorted("|".join(value) for value in symbols),
                "form_types": sorted("|".join(value) for value in forms),
                "different_symbols_or_forms": different,
                "normalized_hashes": sorted({str(item.get("_normalized_chunk_text_hash", "")) for item in items}),
                "identical_after_normalization": len({str(item.get("_normalized_chunk_text_hash", "")) for item in items}) == 1,
                "sample_chunk_ids": [str(item.get("chunk_id", "")) for item in items[:5]],
            }
        )
    duplicate_rows.sort(key=lambda row: (-int(row["instance_count"]), str(row["chunk_content_sha256"])))
    duplicate_instances = sum(len(items) - 1 for items in groups.values() if len(items) > 1)
    return {
        "unique_chunk_hashes": len(groups),
        "duplicate_hash_groups": len(duplicate_rows),
        "duplicate_chunk_instances": duplicate_instances,
        "classification_counts": dict(sorted(class_counts.items())),
        "within_document_duplicate_count": within_count,
        "cross_document_duplicate_count": cross_count,
        "groups_with_different_symbols_or_forms": different_symbol_form,
        "top_duplicate_groups": duplicate_rows[:20],
        "duplicate_groups": duplicate_rows,
        "handling_recommendation": (
            "Retain chunks for auditability now; in modeling, down-weight boilerplate and exact duplicate "
            "source-document chunks rather than deleting them before provenance review."
        ),
    }


def _duplicate_classification(items: Sequence[Mapping[str, Any]], docs: set[str], source_hashes: set[str]) -> str:
    if len(docs) > 1 and len({value for value in source_hashes if value}) == 1:
        return "from_exact_duplicate_source_documents"
    if any("likely_navigation_or_boilerplate" in (item.get("cleaning_flags") or []) for item in items):
        return "likely_boilerplate"
    if len(docs) == 1 and _items_overlap_or_repeat(items):
        return "likely_overlap_artifact"
    if len(docs) == 1:
        return "within_same_document"
    if len(docs) > 1:
        return "across_different_documents"
    return "unknown_duplicate"


def _truncation_audit(
    manifest: Mapping[str, Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_doc[str(chunk.get("document_id", ""))].append(chunk)
    chunk_size = int((summary.get("chunking_parameters") or {}).get("chunk_size") or 2_000)
    rows = []
    for document_id, doc in sorted(manifest.items()):
        if not doc.get("truncated_document"):
            continue
        doc_chunks = sorted(by_doc.get(document_id, []), key=lambda row: int(row.get("cleaned_character_start") or 0))
        cleaned_len = int(doc.get("cleaned_character_length") or 0)
        opening = sum(1 for row in doc_chunks if int(row.get("cleaned_character_start") or 0) < chunk_size)
        ending = sum(1 for row in doc_chunks if int(row.get("cleaned_character_end") or 0) >= max(0, cleaned_len - chunk_size))
        middle = max(0, len(doc_chunks) - opening - ending)
        coverage, largest_gap = _coverage_and_gap(doc_chunks, cleaned_len)
        rows.append(
            {
                "document_id": document_id,
                "accession_number": doc.get("accession_number", ""),
                "symbols": doc.get("symbols", []),
                "form_types": doc.get("form_types", []),
                "cleaned_character_count": cleaned_len,
                "total_candidate_chunks": doc.get("total_candidate_chunks", 0),
                "retained_chunk_count": doc.get("retained_chunk_count", 0),
                "dropped_chunk_count": doc.get("dropped_chunk_count", 0),
                "retained_opening_chunk_count": opening,
                "retained_middle_chunk_count": middle,
                "retained_ending_chunk_count": ending,
                "retained_cleaned_character_coverage": coverage,
                "largest_uncovered_character_gap": largest_gap,
                "first_meaningful_section_retained": bool(doc_chunks and int(doc_chunks[0].get("cleaned_character_start") or 0) == 0),
                "final_meaningful_section_retained": bool(doc_chunks and int(doc_chunks[-1].get("cleaned_character_end") or 0) == cleaned_len),
                "retained_indices_deterministic_and_balanced": bool(opening and ending and middle),
                "silently_first_only": bool(doc_chunks and int(doc_chunks[-1].get("cleaned_character_end") or 0) < cleaned_len),
            }
        )
    return rows


def _overlap_audit(chunks_by_doc: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    gap_count = 0
    overlap_off_target = 0
    final_chunk_errors = 0
    noncontiguous_untruncated = 0
    selected_truncated_docs = 0
    for items in chunks_by_doc.values():
        ordered = sorted(items, key=lambda row: int(row.get("cleaned_character_start") or 0))
        if not ordered:
            continue
        truncated = bool(ordered[0].get("truncated_document"))
        if truncated:
            selected_truncated_docs += 1
        previous_end = None
        for row in ordered:
            start = int(row.get("cleaned_character_start") or 0)
            end = int(row.get("cleaned_character_end") or 0)
            if previous_end is not None:
                delta = start - previous_end
                if not truncated:
                    if delta > 0:
                        gap_count += 1
                        noncontiguous_untruncated += 1
                    if not (-350 <= delta <= -150) and end - start > 0:
                        overlap_off_target += 1
            previous_end = end
        doc_clean_len = max(int(row.get("cleaned_character_end") or 0) for row in ordered)
        if not truncated and int(ordered[-1].get("cleaned_character_end") or 0) != doc_clean_len:
            final_chunk_errors += 1
    return {
        "untruncated_gap_count": gap_count,
        "untruncated_noncontiguous_document_count": noncontiguous_untruncated,
        "overlap_outside_expected_range_count": overlap_off_target,
        "final_chunk_error_count": final_chunk_errors,
        "truncated_documents_with_explicit_selection_count": selected_truncated_docs,
    }


def _provenance_audit(chunks: Sequence[Mapping[str, Any]], manifest: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    chunk_counts = Counter()
    for chunk in chunks:
        chunk_counts["missing_document_id"] += int(not chunk.get("document_id"))
        chunk_counts["missing_accession"] += int(not chunk.get("accession_number"))
        chunk_counts["missing_symbol_or_form_type"] += int(not chunk.get("symbols") or not chunk.get("form_types"))
        chunk_counts["event_timestamp_present"] += int(bool(chunk.get("event_timestamp")))
        chunk_counts["event_timestamp_missing"] += int(not chunk.get("event_timestamp"))
        chunk_counts["availability_timestamp_present"] += int(bool(chunk.get("availability_timestamp")))
        chunk_counts["availability_timestamp_missing"] += int(not chunk.get("availability_timestamp"))
        chunk_counts["source_cache_path_missing"] += int(not chunk.get("source_cache_path"))
        chunk_counts["source_content_hash_missing"] += int(not chunk.get("source_content_sha256"))
        chunk_counts["cleaned_content_hash_missing"] += int(not chunk.get("cleaned_content_sha256"))
    multi_event_docs = sum(1 for doc in manifest.values() if len(doc.get("event_keys") or []) > 1)
    return {**dict(chunk_counts), "documents_with_multiple_events": multi_event_docs}


def _readiness(
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    duplicates: Mapping[str, Any],
    truncation_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = []
    if validation["invalid_offset_count"]:
        blockers.append("invalid_offsets")
    if validation["chunk_hash_mismatch_count"]:
        blockers.append("chunk_hash_mismatches")
    if validation["duplicate_chunk_id_count"]:
        blockers.append("duplicate_chunk_ids")
    if validation["cleaned_hash_inconsistency_count"] or validation["source_hash_inconsistency_count"]:
        blockers.append("document_hash_inconsistencies")
    if provenance.get("missing_document_id"):
        blockers.append("missing_stable_document_identity")
    if any(not row["final_meaningful_section_retained"] for row in truncation_rows):
        blockers.append("truncation_drops_ending_context")
    if validation["chunk_text_over_limit_count"]:
        blockers.append("chunk_text_exceeds_limit")
    warnings = []
    if duplicates["duplicate_chunk_instances"]:
        warnings.append("duplicate_chunks_require_downweight_or_review")
    if provenance.get("event_timestamp_missing"):
        warnings.append("event_timestamps_missing_at_chunk_layer")
    if provenance.get("availability_timestamp_missing"):
        warnings.append("availability_timestamps_missing_at_chunk_layer")
    if truncation_rows:
        warnings.append("large_documents_truncated_by_policy")
    if duplicates["classification_counts"].get("from_exact_duplicate_source_documents"):
        warnings.append("exact_duplicate_source_documents_present")
    params = dict(summary.get("chunking_parameters", {}) or {})
    status = "blocked" if blockers else ("approved_with_warnings" if warnings else "approved")
    full_output = "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/sec_primary_document_text_chunks_120mo_v1"
    full_command = (
        "/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python "
        "scripts/stock_alpha_news_transformer_clean_and_chunk_sec_text.py "
        "--cache-dir reports/ml/benchmark/regime_transformer_meta_ensemble_v1/sec_primary_document_text_cache_120mo_v1_consolidated "
        "--quality-audit-dir reports/ml/benchmark/regime_transformer_meta_ensemble_v1/sec_primary_document_text_cache_120mo_v1_quality_audit "
        f"--output-dir {full_output} --reports-root reports "
        f"--chunk-size {int(params.get('chunk_size') or 2000)} "
        f"--chunk-overlap {int(params.get('chunk_overlap') or 250)} "
        f"--max-chunks-per-document {int(params.get('max_chunks_per_document') or 128)}"
    )
    return {
        "status": status,
        "blocking_reasons": blockers,
        "warnings": warnings,
        "approved_chunk_size": int(params.get("chunk_size") or 2_000),
        "approved_overlap": int(params.get("chunk_overlap") or 250),
        "approved_max_chunks_per_document": int(params.get("max_chunks_per_document") or 128),
        "duplicate_handling_recommendation": duplicates["handling_recommendation"],
        "truncation_policy_recommendation": "Use the existing head/tail/deterministic-middle sampling; full build may proceed after warning review.",
        "event_join_requirement": "Availability timestamps are absent in this cache layer; restore event/availability timestamps from the validated event dataset before walk-forward splitting.",
        "recommended_next_step": "review_smoke_audit_then_run_prepared_full_build_command_report_only" if status != "blocked" else "resolve_chunk_readiness_blockers",
        "prepared_full_build_command": full_command if status != "blocked" else "",
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json(path).get("documents", [])
    return {str(row.get("document_id", "")): dict(row) for row in rows if isinstance(row, Mapping)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _offsets_invalid(row: Mapping[str, Any]) -> bool:
    try:
        source_start = int(row.get("source_character_start"))
        source_end = int(row.get("source_character_end"))
        clean_start = int(row.get("cleaned_character_start"))
        clean_end = int(row.get("cleaned_character_end"))
    except (TypeError, ValueError):
        return True
    return source_start < 0 or clean_start < 0 or source_end < source_start or clean_end < clean_start


def _chunk_size_limit(row: Mapping[str, Any]) -> int:
    del row
    return 2_000


def _items_overlap_or_repeat(items: Sequence[Mapping[str, Any]]) -> bool:
    ordered = sorted(items, key=lambda row: int(row.get("cleaned_character_start") or 0))
    for left, right in zip(ordered, ordered[1:]):
        if int(right.get("cleaned_character_start") or 0) < int(left.get("cleaned_character_end") or 0):
            return True
    return False


def _coverage_and_gap(chunks: Sequence[Mapping[str, Any]], cleaned_len: int) -> tuple[float, int]:
    if cleaned_len <= 0 or not chunks:
        return 0.0, cleaned_len
    intervals = sorted((int(row.get("cleaned_character_start") or 0), int(row.get("cleaned_character_end") or 0)) for row in chunks)
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    gaps = [merged[0][0], cleaned_len - merged[-1][1]]
    gaps.extend(right[0] - left[1] for left, right in zip(merged, merged[1:]))
    return round(covered / cleaned_len, 6), max(gaps or [0])


def _truncation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "documents_truncated": len(rows),
        "all_keep_opening_context": all(row["first_meaningful_section_retained"] for row in rows),
        "all_keep_ending_context": all(row["final_meaningful_section_retained"] for row in rows),
        "all_balanced": all(row["retained_indices_deterministic_and_balanced"] for row in rows),
        "silently_first_only_count": sum(1 for row in rows if row["silently_first_only"]),
        "largest_uncovered_character_gap_max": max((int(row["largest_uncovered_character_gap"]) for row in rows), default=0),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
    parser = argparse.ArgumentParser(description="Audit SEC text chunk output and full-build readiness offline.")
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args(argv)
    audit = audit_sec_text_chunks(
        chunk_dir=args.chunk_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    readiness = audit["full_build_readiness"]
    print(f"total_chunks={audit['total_chunks']}")
    print(f"unique_chunk_hashes={audit['unique_chunk_hashes']}")
    print(f"duplicate_classification_counts={audit['duplicate_classification_counts']}")
    print(f"truncated_documents={audit['truncated_document_count']}")
    print(f"validation_failures={audit['boundary_validation']}")
    print(f"provenance_gaps={audit['provenance_audit']}")
    print(f"readiness_status={readiness['status']}")
    print(f"blocking_reasons={readiness['blocking_reasons']}")
    print(f"warnings={readiness['warnings']}")
    print(
        "approved_parameters="
        f"chunk_size:{readiness['approved_chunk_size']} "
        f"overlap:{readiness['approved_overlap']} "
        f"max_chunks:{readiness['approved_max_chunks_per_document']}"
    )
    print(f"recommended_next_step={readiness['recommended_next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
