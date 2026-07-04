from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.stock_alpha_news_transformer_cache_sec_primary_text import (
    CACHE_MANIFEST_FILENAME,
    CACHE_SUMMARY_FILENAME,
    ENRICHMENT_MANIFEST_FILENAME,
    _deduplicated_documents,
    _is_under_reports,
)


AUDIT_FILENAME = "sec_primary_document_text_cache_failure_audit.json"
RETRY_CANDIDATES_FILENAME = "sec_primary_document_text_cache_retry_candidates.jsonl"

RETRY_PRIORITIES = {
    "retryable_timeout": "high",
    "retryable_rate_limit": "high",
    "retryable_http_5xx": "high",
    "retryable_connection_error": "high",
    "http_403_or_access_denied": "medium",
    "unexpected_content_type": "low",
    "html_or_text_extraction_error": "low",
    "empty_or_unusable_document": "low",
    "http_404_or_missing_document": "do_not_retry",
    "invalid_or_missing_url": "do_not_retry",
    "malformed_failure_record": "unknown",
    "unknown_missing_from_cache": "unknown",
    "other": "unknown",
}


def audit_sec_primary_text_cache_failures(
    *,
    enrichment_plan_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    reports_root: str | Path,
) -> dict[str, Any]:
    cache_dir_path = Path(cache_dir)
    output_dir_path = Path(output_dir)
    reports_root_path = Path(reports_root)
    if not _is_under_reports(output_dir_path, reports_root_path):
        raise ValueError("output_dir must be under reports/")

    plan_documents = _read_plan_documents(Path(enrichment_plan_dir))
    manifest_documents = _read_manifest_documents(cache_dir_path)
    summary = _read_json(cache_dir_path / CACHE_SUMMARY_FILENAME)

    failures, observability_class = _failure_records(plan_documents, manifest_documents)
    duplicate_url_count = _duplicate_count([str(row.get("primary_document_url", "")) for row in failures])
    duplicate_document_id_count = _duplicate_count([_document_id(row) for row in failures])
    classified = [_classify_failure(row) for row in failures]

    retry_candidates = [
        _retry_candidate(row, category)
        for row, category in zip(failures, classified, strict=True)
        if RETRY_PRIORITIES.get(category) in {"high", "medium", "low", "unknown"}
    ]

    audit = _audit_payload(
        summary=summary,
        plan_documents=plan_documents,
        manifest_documents=manifest_documents,
        failures=failures,
        classified=classified,
        observability_class=observability_class,
        duplicate_failure_record_count=max(duplicate_url_count, duplicate_document_id_count),
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir_path / AUDIT_FILENAME, audit)
    if retry_candidates:
        _write_jsonl(output_dir_path / RETRY_CANDIDATES_FILENAME, retry_candidates)
    return audit


def _read_plan_documents(plan_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv(plan_dir / ENRICHMENT_MANIFEST_FILENAME)
    return [_document_request_to_record(document) for document in _deduplicated_documents(rows)]


def _read_manifest_documents(cache_dir: Path) -> list[dict[str, Any]]:
    path = cache_dir / CACHE_MANIFEST_FILENAME
    if not path.exists():
        return []
    payload = _read_json(path)
    documents = payload.get("documents", [])
    return [dict(row) for row in documents if isinstance(row, Mapping)]


def _failure_records(
    plan_documents: Sequence[Mapping[str, Any]],
    manifest_documents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    failed = [
        dict(row)
        for row in manifest_documents
        if row.get("status") in {"failed", "empty_text"} or ("status" not in row and row.get("primary_document_url"))
    ]
    if failed:
        has_reason = any(row.get("error") or row.get("error_type") for row in failed)
        return failed, "A" if has_reason else "B"
    if manifest_documents:
        cached_urls = {str(row.get("primary_document_url", "")) for row in manifest_documents if row.get("status") in {"cached", "skipped_existing"}}
        missing = [dict(row, status="missing_from_cache") for row in plan_documents if row.get("primary_document_url") not in cached_urls]
        return missing, "B" if missing else "A"
    return [], "C"


def _audit_payload(
    *,
    summary: Mapping[str, Any],
    plan_documents: Sequence[Mapping[str, Any]],
    manifest_documents: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    classified: Sequence[str],
    observability_class: str,
    duplicate_failure_record_count: int,
) -> dict[str, Any]:
    failure_counts = Counter(classified)
    retryable = sum(1 for category in classified if RETRY_PRIORITIES.get(category) in {"high", "medium", "low"})
    non_retryable = sum(1 for category in classified if RETRY_PRIORITIES.get(category) == "do_not_retry")
    unknown = sum(1 for category in classified if RETRY_PRIORITIES.get(category) == "unknown")
    cached_documents = sum(1 for row in manifest_documents if row.get("status") in {"cached", "skipped_existing"})
    requested_documents = int(summary.get("requested_documents") or len(plan_documents))
    failed_documents = int(summary.get("failed_documents") or len(failures))
    document_lengths = [
        int(row.get("text_length") or 0)
        for row in manifest_documents
        if row.get("status") in {"cached", "skipped_existing"} and int(row.get("text_length") or 0) > 0
    ]
    max_text_length = max(document_lengths) if document_lengths else int(summary.get("document_text_max_length") or 0)
    return {
        "mode": "sec_primary_document_text_cache_failure_audit_report_only",
        "research_only": True,
        "requested_documents": requested_documents,
        "cached_documents": int(summary.get("cached_documents") or cached_documents),
        "failed_documents": failed_documents,
        "failure_observability_class": observability_class,
        "failure_records_with_reason": sum(1 for row in failures if row.get("error") or row.get("error_type")),
        "failure_records_without_reason": sum(1 for row in failures if not (row.get("error") or row.get("error_type"))),
        "failure_counts_by_category": dict(sorted(failure_counts.items())),
        "failure_counts_by_http_status": _counts_by_http_status(failures),
        "failure_counts_by_exception_type": _count_field(failures, "error_type"),
        "failure_counts_by_form_type": _count_list_field(failures, "form_types"),
        "failure_counts_by_year": _counts_by_year(failures),
        "failure_counts_by_symbol": _count_list_field(failures, "symbols"),
        "failure_counts_by_sec_host": _counts_by_host(failures),
        "top_20_symbols_by_failure_count": _top(_count_list_field(failures, "symbols")),
        "top_20_form_types_by_failure_count": _top(_count_list_field(failures, "form_types")),
        "retryable_failure_count": retryable,
        "non_retryable_failure_count": non_retryable,
        "unknown_failure_count": unknown,
        "rate_limit_or_timeout_failure_count": sum(failure_counts.get(k, 0) for k in ("retryable_timeout", "retryable_rate_limit")),
        "missing_or_invalid_url_count": failure_counts.get("invalid_or_missing_url", 0),
        "http_403_count": failure_counts.get("http_403_or_access_denied", 0),
        "http_404_count": failure_counts.get("http_404_or_missing_document", 0),
        "http_429_count": _counts_by_http_status(failures).get("429", 0),
        "server_error_count": failure_counts.get("retryable_http_5xx", 0),
        "text_extraction_failure_count": failure_counts.get("html_or_text_extraction_error", 0),
        "unexpected_content_type_count": failure_counts.get("unexpected_content_type", 0),
        "duplicate_failure_record_count": duplicate_failure_record_count,
        "unique_failed_document_count": len({_document_id(row) for row in failures}),
        "sample_failures": [_sample_failure(row, category) for row, category in list(zip(failures, classified, strict=True))[:5]],
        "url_pattern_notes": _url_pattern_notes(failures),
        "ticker_accession_mapping_risk": _ticker_accession_mapping_risk(failures),
        "large_text_quality_risk": _large_text_quality_risk(max_text_length),
        "recommended_next_step": _recommended_next_step(observability_class, retryable, unknown),
        "blocking_reasons": ["document_fetch_failures"] if failures else [],
        "model_training_started": False,
        "transformer_training_started": False,
        "trading_impact": "none",
    }


def _classify_failure(record: Mapping[str, Any]) -> str:
    url = str(record.get("primary_document_url", "")).strip()
    error_type = str(record.get("error_type", "")).strip()
    error = str(record.get("error", "")).lower()
    status = _http_status(record)
    if not url:
        return "invalid_or_missing_url"
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"www.sec.gov", "sec.gov"}:
        return "invalid_or_missing_url"
    if record.get("status") == "missing_from_cache":
        return "unknown_missing_from_cache"
    if not record.get("status"):
        return "malformed_failure_record"
    if status in {408, 504} or "timed out" in error or "timeout" in error:
        return "retryable_timeout"
    if status == 429 or "rate limit" in error or "too many request" in error:
        return "retryable_rate_limit"
    if status and 500 <= status <= 599:
        return "retryable_http_5xx"
    if status == 403 or "access denied" in error or "forbidden" in error:
        return "http_403_or_access_denied"
    if status == 404 or "not found" in error:
        return "http_404_or_missing_document"
    if "content-type" in error or "unexpected content" in error:
        return "unexpected_content_type"
    if record.get("status") == "empty_text" or "shorter than" in error or "empty" in error:
        return "empty_or_unusable_document"
    if "htmlparser" in error_type.lower() or "parse" in error or "extraction" in error:
        return "html_or_text_extraction_error"
    if error_type in {"URLError", "ConnectionError", "RemoteDisconnected"}:
        return "retryable_connection_error"
    if not (record.get("error") or record.get("error_type")):
        return "unknown_missing_from_cache"
    return "other"


def _retry_candidate(record: Mapping[str, Any], category: str) -> dict[str, Any]:
    return {
        "document_id": _document_id(record),
        "document_url": record.get("primary_document_url"),
        "accession": record.get("accession_number"),
        "symbol": _first(record.get("symbols")),
        "form_type": _first(record.get("form_types")),
        "event_timestamp": None,
        "failure_category": category,
        "previous_http_status": _http_status(record),
        "previous_exception_type": record.get("error_type"),
        "retry_priority": RETRY_PRIORITIES.get(category, "unknown"),
    }


def _document_request_to_record(document: Any) -> dict[str, Any]:
    return {
        "accession_number": document.accession_number,
        "primary_document_url": document.primary_document_url,
        "event_keys": list(document.event_keys),
        "symbols": list(document.symbols),
        "form_types": list(document.form_types),
    }


def _document_id(record: Mapping[str, Any]) -> str:
    accession = str(record.get("accession_number", "")).strip()
    url = str(record.get("primary_document_url", "")).strip()
    return f"{accession}|{url}"


def _http_status(record: Mapping[str, Any]) -> int | None:
    for key in ("http_status", "status_code"):
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    text = f"{record.get('error_type', '')} {record.get('error', '')}"
    match = re.search(r"(?:HTTP Error|HTTP status|status)\\D+(\\d{3})", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _counts_by_http_status(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        status = _http_status(record)
        if status is not None:
            counts[str(status)] += 1
    return dict(sorted(counts.items()))


def _counts_by_year(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        candidates = [str(record.get("accession_number", "")), str(record.get("primary_document_url", ""))]
        year = next((f"20{m.group(1)}" for value in candidates for m in [re.search(r"-(\\d{2})-", value)] if m), "unknown")
        counts[year] += 1
    return dict(sorted(counts.items()))


def _counts_by_host(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        host = urlparse(str(record.get("primary_document_url", ""))).netloc.lower() or "unknown"
        counts[host] += 1
    return dict(sorted(counts.items()))


def _count_field(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "unknown") for row in records)
    return dict(sorted(counts.items()))


def _count_list_field(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in records:
        values = row.get(field) or ["unknown"]
        if not isinstance(values, list):
            values = [values]
        for value in values:
            counts[str(value or "unknown")] += 1
    return dict(sorted(counts.items()))


def _duplicate_count(values: Sequence[str]) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


def _top(counts: Mapping[str, int], limit: int = 20) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in Counter(counts).most_common(limit)]


def _sample_failure(record: Mapping[str, Any], category: str) -> dict[str, Any]:
    return {
        "document_id": _document_id(record),
        "url": _redact_url(str(record.get("primary_document_url", ""))),
        "accession": record.get("accession_number"),
        "symbols": record.get("symbols"),
        "form_types": record.get("form_types"),
        "failure_category": category,
        "error_type": record.get("error_type"),
        "error": record.get("error"),
    }


def _redact_url(url: str) -> str:
    parts = url.split("/")
    if len(parts) < 5:
        return url
    return "/".join(parts[:3] + parts[-2:])


def _url_pattern_notes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    archive_count = sum(1 for row in records if "/Archives/edgar/data/" in str(row.get("primary_document_url", "")))
    ixviewer_count = sum(1 for row in records if "ixviewer" in str(row.get("primary_document_url", "")).lower())
    return {
        "archives_edgar_data_count": archive_count,
        "ixviewer_url_count": ixviewer_count,
        "non_archives_url_count": len(records) - archive_count,
    }


def _ticker_accession_mapping_risk(records: Sequence[Mapping[str, Any]]) -> str:
    missing_accession = sum(1 for row in records if not str(row.get("accession_number", "")).strip())
    missing_symbols = sum(1 for row in records if not row.get("symbols"))
    if missing_accession or missing_symbols:
        return f"review_missing_metadata accession={missing_accession} symbols={missing_symbols}"
    return "no obvious missing symbol/accession metadata in failure records"


def _large_text_quality_risk(max_text_length: int) -> str:
    if max_text_length >= 1_000_000:
        return "review largest cached documents for inline XBRL, navigation text, exhibits, or full submission bundle extraction"
    return "no large-document text-quality risk flagged by max text length"


def _recommended_next_step(observability_class: str, retryable: int, unknown: int) -> str:
    if observability_class == "C":
        return "add failure observability before retrying"
    if retryable:
        return "run a 3-5 document smoke retry into a separate output directory"
    if unknown:
        return "inspect unknown failures before retrying"
    return "no retryable failures identified"


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit SEC primary-document cache failures without network access.")
    parser.add_argument("--enrichment-plan-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reports-root", required=True)
    args = parser.parse_args(argv)
    audit = audit_sec_primary_text_cache_failures(
        enrichment_plan_dir=args.enrichment_plan_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        reports_root=args.reports_root,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
