from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CHUNK_MANIFEST_FILENAME = "sec_primary_document_text_clean_manifest.json"
CHUNKS_FILENAME = "sec_primary_document_text_chunks.jsonl"
CHUNK_AUDIT_FILENAME = "sec_primary_document_text_chunk_audit.json"
DUPLICATE_CHUNKS_FILENAME = "sec_primary_document_text_duplicate_chunks.jsonl"
TRUNCATION_AUDIT_FILENAME = "sec_primary_document_text_truncation_audit.jsonl"
CACHE_MANIFEST_FILENAME = "sec_primary_document_text_cache_manifest.json"

EVENT_CHUNK_INDEX_FILENAME = "news_transformer_event_chunk_index.jsonl"
EVENT_DOCUMENT_INDEX_FILENAME = "news_transformer_event_document_index.jsonl"
UNMATCHED_EVENTS_FILENAME = "news_transformer_unmatched_events.jsonl"
UNMATCHED_DOCUMENTS_FILENAME = "news_transformer_unmatched_documents.jsonl"
AMBIGUOUS_JOINS_FILENAME = "news_transformer_ambiguous_joins.jsonl"
SUMMARY_FILENAME = "news_transformer_event_chunk_index_summary.json"
READINESS_FILENAME = "news_transformer_event_chunk_index_readiness.json"


def build_event_chunk_index(
    *,
    event_features_csv: str | Path,
    chunk_dir: str | Path,
    cache_manifest_path: str | Path,
    chunk_audit_dir: str | Path,
    output_dir: str | Path,
    reports_root: str | Path = "reports",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    if not _is_under_reports(output_path, Path(reports_root)):
        raise ValueError("output_dir must be under reports/")

    events = _read_csv(Path(event_features_csv))
    chunk_path = Path(chunk_dir)
    chunk_manifest = _read_chunk_manifest(chunk_path / CHUNK_MANIFEST_FILENAME)
    cache_manifest = _read_cache_manifest(Path(cache_manifest_path))
    chunks_by_document = _read_chunks_by_document(chunk_path / CHUNKS_FILENAME)
    duplicate_by_hash = _read_duplicate_metadata(Path(chunk_audit_dir) / DUPLICATE_CHUNKS_FILENAME)
    truncation_by_document = _read_jsonl_by_key(Path(chunk_audit_dir) / TRUNCATION_AUDIT_FILENAME, "document_id")
    chunk_audit = _read_json_if_exists(Path(chunk_audit_dir) / CHUNK_AUDIT_FILENAME)

    documents = _merge_document_metadata(chunk_manifest, cache_manifest)
    indexes = _document_indexes(documents)
    event_document_links, unmatched_events, ambiguous_joins = _join_events_to_documents(events, indexes)
    linked_document_ids = {row["document_id"] for row in event_document_links}
    unmatched_documents = [
        _document_sample(doc)
        for doc_id, doc in sorted(documents.items())
        if doc_id not in linked_document_ids
    ]
    event_chunk_rows = _event_chunk_rows(
        event_document_links,
        chunks_by_document,
        documents,
        duplicate_by_hash,
        truncation_by_document,
    )
    readiness = _readiness(
        events=events,
        documents=documents,
        chunks_by_document=chunks_by_document,
        event_document_links=event_document_links,
        event_chunk_rows=event_chunk_rows,
        unmatched_events=unmatched_events,
        unmatched_documents=unmatched_documents,
        ambiguous_joins=ambiguous_joins,
        duplicate_by_hash=duplicate_by_hash,
        truncation_by_document=truncation_by_document,
        chunk_audit=chunk_audit,
    )
    summary = {
        "mode": "news_transformer_event_chunk_index_report_only",
        "research_only": True,
        "event_count": len(events),
        "document_count": len(documents),
        "chunk_count": sum(len(rows) for rows in chunks_by_document.values()),
        "event_document_link_count": len(event_document_links),
        "event_chunk_link_count": len(event_chunk_rows),
        "model_eligible_event_count": len({row["event_id"] for row in event_chunk_rows if row["model_eligible"]}),
        "model_eligible_chunk_count": sum(1 for row in event_chunk_rows if row["model_eligible"]),
        "unmatched_event_count": len(unmatched_events),
        "unmatched_document_count": len(unmatched_documents),
        "ambiguous_join_count": len(ambiguous_joins),
        "documents_mapping_to_multiple_events": _documents_mapping_to_multiple_events(event_document_links),
        "events_mapping_to_multiple_documents": _events_mapping_to_multiple_documents(event_document_links),
        "duplicate_join_key_combinations": indexes["duplicate_join_key_combinations"],
        "duplicate_chunk_group_count": len(duplicate_by_hash),
        "boilerplate_duplicate_chunk_count": sum(1 for row in event_chunk_rows if row["likely_boilerplate_duplicate"]),
        "truncated_document_count": len(truncation_by_document),
        "events_linked_to_truncated_documents": len({row["event_id"] for row in event_chunk_rows if row["truncated_document"]}),
        "chunks_linked_to_truncated_documents": sum(1 for row in event_chunk_rows if row["truncated_document"]),
        "truncation_coverage_percentiles": _coverage_percentiles(truncation_by_document.values()),
        "readiness": readiness,
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }

    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(output_path / EVENT_CHUNK_INDEX_FILENAME, event_chunk_rows)
    _atomic_jsonl(output_path / EVENT_DOCUMENT_INDEX_FILENAME, event_document_links)
    _atomic_jsonl(output_path / UNMATCHED_EVENTS_FILENAME, unmatched_events)
    _atomic_jsonl(output_path / UNMATCHED_DOCUMENTS_FILENAME, unmatched_documents)
    _atomic_jsonl(output_path / AMBIGUOUS_JOINS_FILENAME, ambiguous_joins)
    _atomic_json(output_path / SUMMARY_FILENAME, summary)
    _atomic_json(output_path / READINESS_FILENAME, readiness)
    return summary


def _join_events_to_documents(
    events: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    links: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: (_event_id(row), _event_key(row))):
        candidates_by_key = _candidate_documents(event, indexes)
        nonempty_sets = [set(values) for values in candidates_by_key.values() if values]
        if not nonempty_sets:
            unmatched.append(_event_sample(event, "no_strong_document_match"))
            continue
        union = set().union(*nonempty_sets)
        if len(nonempty_sets) > 1 and any(values != union for values in nonempty_sets):
            ambiguous.append(
                {
                    **_event_sample(event, "conflicting_strong_join_keys"),
                    "candidate_documents_by_key": {
                        key: sorted(values)
                        for key, values in sorted(candidates_by_key.items())
                        if values
                    },
                }
            )
            continue
        for document_id in sorted(union):
            links.append(
                {
                    "event_id": _event_id(event),
                    "event_key": _event_key(event),
                    "document_id": document_id,
                    "accession_number": _event_accession(event),
                    "primary_document_url": _event_url(event),
                    "symbol": str(event.get("symbol", "")),
                    "form_type": str(event.get("form_type", "")),
                    "event_timestamp": str(event.get("event_timestamp", "")),
                    "available_at_timestamp": str(event.get("available_at_timestamp", "")),
                    "provider": str(event.get("provider", "")),
                    "source_type": str(event.get("source_type", "")),
                    "join_keys_used": sorted(key for key, values in candidates_by_key.items() if document_id in values),
                    "model_eligible_event": _event_model_eligible(event),
                    "timestamp_order_valid": _timestamp_order_valid(event),
                }
            )
    return links, unmatched, ambiguous


def _event_chunk_rows(
    links: Sequence[Mapping[str, Any]],
    chunks_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    documents: Mapping[str, Mapping[str, Any]],
    duplicate_by_hash: Mapping[str, Mapping[str, Any]],
    truncation_by_document: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for link in links:
        document_id = str(link["document_id"])
        doc = documents.get(document_id, {})
        truncation = dict(truncation_by_document.get(document_id, {}) or {})
        for chunk in chunks_by_document.get(document_id, []):
            duplicate = dict(duplicate_by_hash.get(str(chunk.get("chunk_content_sha256", "")), {}) or {})
            model_eligible = bool(link["model_eligible_event"] and link["timestamp_order_valid"])
            rows.append(
                {
                    "event_id": link["event_id"],
                    "event_key": link["event_key"],
                    "document_id": document_id,
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "accession_number": str(doc.get("accession_number") or link.get("accession_number") or ""),
                    "primary_document_url": str(doc.get("primary_document_url") or link.get("primary_document_url") or ""),
                    "symbol": str(link.get("symbol") or ""),
                    "form_type": str(link.get("form_type") or ""),
                    "event_timestamp": str(link.get("event_timestamp") or ""),
                    "available_at_timestamp": str(link.get("available_at_timestamp") or ""),
                    "provider": str(link.get("provider") or ""),
                    "source_type": str(link.get("source_type") or ""),
                    "chunk_index": int(chunk.get("chunk_index") or 0),
                    "chunk_character_length": int(chunk.get("chunk_character_length") or 0),
                    "source_content_sha256": str(chunk.get("source_content_sha256") or ""),
                    "cleaned_content_sha256": str(chunk.get("cleaned_content_sha256") or ""),
                    "chunk_content_sha256": str(chunk.get("chunk_content_sha256") or ""),
                    "truncated_document": bool(doc.get("truncated_document")),
                    "total_candidate_chunks": int(doc.get("total_candidate_chunks") or 0),
                    "retained_chunk_count": int(doc.get("retained_chunk_count") or 0),
                    "dropped_chunk_count": int(doc.get("dropped_chunk_count") or 0),
                    "retained_opening_chunk_count": int(truncation.get("retained_opening_chunk_count") or 0),
                    "retained_middle_chunk_count": int(truncation.get("retained_middle_chunk_count") or 0),
                    "retained_ending_chunk_count": int(truncation.get("retained_ending_chunk_count") or 0),
                    "retained_character_coverage": truncation.get("retained_cleaned_character_coverage", ""),
                    "largest_uncovered_character_gap": truncation.get("largest_uncovered_character_gap", ""),
                    "selection_policy": "head_tail_deterministic_middle" if doc.get("truncated_document") else "all_chunks_retained",
                    "duplicate_chunk_hash": bool(duplicate),
                    "duplicate_group_size": int(duplicate.get("instance_count") or 1),
                    "duplicate_scope": str(duplicate.get("classification") or ""),
                    "likely_boilerplate_duplicate": duplicate.get("classification") == "likely_boilerplate",
                    "source_cache_path": str(doc.get("source_cache_path") or chunk.get("source_cache_path") or ""),
                    "model_eligible": model_eligible,
                    "timestamp_order_valid": bool(link["timestamp_order_valid"]),
                }
            )
    return sorted(rows, key=lambda row: (row["event_id"], row["document_id"], row["chunk_index"], row["chunk_id"]))


def _candidate_documents(event: Mapping[str, Any], indexes: Mapping[str, Any]) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    event_key = _event_key(event)
    if event_key:
        candidates["event_key"] = set(indexes["by_event_key"].get(event_key, set()))
    doc_id = str(event.get("document_id") or "").strip()
    if doc_id:
        candidates["document_id"] = set(indexes["by_document_id"].get(doc_id, set()))
    url = _event_url(event)
    if url:
        candidates["primary_document_url"] = set(indexes["by_url"].get(_normalize_url(url), set()))
    accession = _event_accession(event)
    if accession:
        candidates["accession_number"] = set(indexes["by_accession"].get(accession, set()))
    return candidates


def _document_indexes(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_event_key: dict[str, set[str]] = defaultdict(set)
    by_document_id: dict[str, set[str]] = defaultdict(set)
    by_url: dict[str, set[str]] = defaultdict(set)
    by_accession: dict[str, set[str]] = defaultdict(set)
    duplicate_join_keys = Counter()
    for document_id, doc in documents.items():
        by_document_id[document_id].add(document_id)
        accession = str(doc.get("accession_number") or "")
        if accession:
            by_accession[accession].add(document_id)
        url = str(doc.get("primary_document_url") or "")
        if url:
            by_url[_normalize_url(url)].add(document_id)
        for event_key in _list_field(doc.get("event_keys")):
            by_event_key[event_key].add(document_id)
    for label, mapping in (("accession_number", by_accession), ("primary_document_url", by_url), ("event_key", by_event_key)):
        for value, doc_ids in mapping.items():
            if len(doc_ids) > 1:
                duplicate_join_keys[(label, value)] = len(doc_ids)
    return {
        "by_event_key": by_event_key,
        "by_document_id": by_document_id,
        "by_url": by_url,
        "by_accession": by_accession,
        "duplicate_join_key_combinations": [
            {"key_type": key[0], "key_value": key[1], "document_count": count}
            for key, count in sorted(duplicate_join_keys.items())
        ],
    }


def _merge_document_metadata(
    chunk_manifest: Mapping[str, Mapping[str, Any]],
    cache_manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {}
    for document_id, chunk_doc in sorted(chunk_manifest.items()):
        cache_doc = cache_manifest.get(document_id, {})
        merged[document_id] = {
            **dict(cache_doc),
            **dict(chunk_doc),
            "cache_content_sha256": cache_doc.get("content_sha256", ""),
            "event_keys": _list_field(cache_doc.get("event_keys")),
        }
    return merged


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_chunk_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("document_id", "")): dict(row) for row in payload.get("documents", [])}


def _read_cache_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("document_id", "")): dict(row) for row in payload.get("documents", []) if row.get("document_id")}


def _read_chunks_by_document(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row.get("document_id", ""))].append(row)
    for values in rows.values():
        values.sort(key=lambda row: (int(row.get("chunk_index") or 0), str(row.get("chunk_id", ""))))
    return rows


def _read_duplicate_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    return {str(row.get("chunk_content_sha256", "")): dict(row) for row in rows if row.get("chunk_content_sha256")}


def _read_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key, "")): dict(row) for row in _read_jsonl(path) if row.get(key)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _readiness(
    *,
    events: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    chunks_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    event_document_links: Sequence[Mapping[str, Any]],
    event_chunk_rows: Sequence[Mapping[str, Any]],
    unmatched_events: Sequence[Mapping[str, Any]],
    unmatched_documents: Sequence[Mapping[str, Any]],
    ambiguous_joins: Sequence[Mapping[str, Any]],
    duplicate_by_hash: Mapping[str, Mapping[str, Any]],
    truncation_by_document: Mapping[str, Mapping[str, Any]],
    chunk_audit: Mapping[str, Any],
) -> dict[str, Any]:
    del event_document_links
    missing_availability = sum(1 for row in event_chunk_rows if not row["available_at_timestamp"])
    leakage_violations = sum(1 for row in event_chunk_rows if not row["timestamp_order_valid"])
    validation = dict(chunk_audit.get("boundary_validation", {}) or {})
    source_hash_mismatch = sum(
        1
        for doc in documents.values()
        if doc.get("cache_content_sha256")
        and doc.get("source_content_sha256")
        and doc.get("cache_content_sha256") != doc.get("source_content_sha256")
    )
    blocking = []
    if sum(1 for row in event_chunk_rows if row["model_eligible"] and not row["available_at_timestamp"]):
        blocking.append("model_eligible_rows_missing_availability_timestamp")
    if int(validation.get("duplicate_chunk_id_count", 0)):
        blocking.append("duplicate_chunk_ids")
    if int(validation.get("chunk_hash_mismatch_count", 0)):
        blocking.append("chunk_hash_mismatches")
    if any(not row.get("event_id") or not row.get("document_id") for row in event_chunk_rows):
        blocking.append("missing_stable_event_or_document_identity")
    if leakage_violations:
        blocking.append("timestamp_ordering_violations")
    if source_hash_mismatch:
        blocking.append("source_hash_mismatches")
    warnings = []
    if duplicate_by_hash:
        warnings.append("duplicate_chunks_retained_with_metadata")
    if any(row.get("classification") == "likely_boilerplate" for row in duplicate_by_hash.values()):
        warnings.append("likely_boilerplate_duplicate_chunks_present")
    if any(row.get("classification") == "from_exact_duplicate_source_documents" for row in duplicate_by_hash.values()):
        warnings.append("exact_duplicate_source_documents_present")
    if truncation_by_document:
        warnings.append("truncated_documents_have_balanced_retention_metadata")
    if unmatched_documents:
        warnings.append("unmatched_documents_quarantined_from_model_index")
    if unmatched_events:
        warnings.append("unmatched_events_quarantined_from_model_index")
    if ambiguous_joins:
        warnings.append("ambiguous_event_document_joins_quarantined")
    status = "blocked" if blocking else ("approved_with_warnings" if warnings else "approved")
    return {
        "status": status,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "event_count": len(events),
        "document_count": len(documents),
        "chunk_count": sum(len(values) for values in chunks_by_document.values()),
        "event_document_link_count": len({(row["event_id"], row["document_id"]) for row in event_chunk_rows}),
        "event_chunk_link_count": len(event_chunk_rows),
        "model_eligible_event_count": len({row["event_id"] for row in event_chunk_rows if row["model_eligible"]}),
        "model_eligible_chunk_count": sum(1 for row in event_chunk_rows if row["model_eligible"]),
        "missing_availability_timestamp_count": missing_availability,
        "ambiguous_join_count": len(ambiguous_joins),
        "unmatched_event_count": len(unmatched_events),
        "unmatched_document_count": len(unmatched_documents),
        "duplicate_chunk_group_count": len(duplicate_by_hash),
        "boilerplate_duplicate_chunk_count": sum(1 for row in event_chunk_rows if row["likely_boilerplate_duplicate"]),
        "truncated_document_count": len(truncation_by_document),
        "leakage_violation_count": leakage_violations,
        "source_hash_mismatch_count": source_hash_mismatch,
        "recommended_next_step": "attach_price_labels_report_only_after_review" if status != "blocked" else "resolve_event_chunk_index_blockers",
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }


def _event_model_eligible(event: Mapping[str, Any]) -> bool:
    return bool(event.get("event_id") and event.get("event_key") and event.get("available_at_timestamp") and _timestamp_order_valid(event))


def _timestamp_order_valid(event: Mapping[str, Any]) -> bool:
    event_ts = _parse_ts(str(event.get("event_timestamp", "")))
    available_ts = _parse_ts(str(event.get("available_at_timestamp", "")))
    if event_ts is None or available_ts is None:
        return False
    return available_ts >= event_ts


def _parse_ts(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("event_id") or "").strip()


def _event_key(event: Mapping[str, Any]) -> str:
    return str(event.get("event_key") or "").strip()


def _event_accession(event: Mapping[str, Any]) -> str:
    for key in ("accession_number", "accession", "url_or_accession"):
        value = str(event.get(key) or "").strip()
        if _looks_like_accession(value):
            return value
    return ""


def _event_url(event: Mapping[str, Any]) -> str:
    for key in ("primary_document_url", "source_url", "url"):
        value = str(event.get(key) or "").strip()
        if value.startswith("http"):
            return value
    value = str(event.get("url_or_accession") or "").strip()
    if value.startswith("http"):
        return value
    parts = _event_key(event).split("|")
    return parts[2] if len(parts) >= 3 and parts[2].startswith("http") else ""


def _event_sample(event: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "event_id": _event_id(event),
        "event_key": _event_key(event),
        "symbol": str(event.get("symbol", "")),
        "form_type": str(event.get("form_type", "")),
        "accession_number": _event_accession(event),
        "primary_document_url": _event_url(event),
        "event_timestamp": str(event.get("event_timestamp", "")),
        "available_at_timestamp": str(event.get("available_at_timestamp", "")),
        "reason": reason,
    }


def _document_sample(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "document_id": str(doc.get("document_id", "")),
        "accession_number": str(doc.get("accession_number", "")),
        "primary_document_url": str(doc.get("primary_document_url", "")),
        "symbols": _list_field(doc.get("symbols")),
        "form_types": _list_field(doc.get("form_types")),
        "event_keys": _list_field(doc.get("event_keys")),
    }


def _looks_like_accession(value: str) -> bool:
    return bool(re.fullmatch(r"\d{10}-\d{2}-\d{6}", value))


def _normalize_url(value: str) -> str:
    return value.strip().rstrip("/")


def _list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    return [text] if text else []


def _documents_mapping_to_multiple_events(links: Sequence[Mapping[str, Any]]) -> int:
    mapping: dict[str, set[str]] = defaultdict(set)
    for link in links:
        mapping[str(link["document_id"])].add(str(link["event_id"]))
    return sum(1 for values in mapping.values() if len(values) > 1)


def _events_mapping_to_multiple_documents(links: Sequence[Mapping[str, Any]]) -> int:
    mapping: dict[str, set[str]] = defaultdict(set)
    for link in links:
        mapping[str(link["event_id"])].add(str(link["document_id"]))
    return sum(1 for values in mapping.values() if len(values) > 1)


def _coverage_percentiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = sorted(float(row.get("retained_cleaned_character_coverage") or 0.0) for row in rows)
    return {
        "min": values[0] if values else 0.0,
        "median": _percentile(values, 0.50),
        "p05": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 6)
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return round(values[lower] * (1.0 - weight) + values[upper] * weight, 6)


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
    parser = argparse.ArgumentParser(description="Build leakage-safe event/document/chunk index offline.")
    parser.add_argument("--event-features-csv", required=True)
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--cache-manifest-path", required=True)
    parser.add_argument("--chunk-audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", default="reports")
    args = parser.parse_args(argv)
    summary = build_event_chunk_index(
        event_features_csv=args.event_features_csv,
        chunk_dir=args.chunk_dir,
        cache_manifest_path=args.cache_manifest_path,
        chunk_audit_dir=args.chunk_audit_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    readiness = summary["readiness"]
    print(f"events={summary['event_count']}")
    print(f"documents={summary['document_count']}")
    print(f"chunks={summary['chunk_count']}")
    print(f"event_document_links={summary['event_document_link_count']}")
    print(f"event_chunk_links={summary['event_chunk_link_count']}")
    print(f"unmatched_events={summary['unmatched_event_count']} unmatched_documents={summary['unmatched_document_count']} ambiguous={summary['ambiguous_join_count']}")
    print(f"model_eligible_events={summary['model_eligible_event_count']} model_eligible_chunks={summary['model_eligible_chunk_count']}")
    print(f"duplicate_chunk_groups={summary['duplicate_chunk_group_count']} boilerplate_duplicate_chunks={summary['boilerplate_duplicate_chunk_count']}")
    print(f"truncated_documents={summary['truncated_document_count']}")
    print(f"leakage_violations={readiness['leakage_violation_count']}")
    print(f"readiness_status={readiness['status']}")
    print(f"recommended_next_step={readiness['recommended_next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
