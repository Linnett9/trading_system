import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_cache_sec_primary_text import (
    CACHE_MANIFEST_FILENAME,
    CACHE_SUMMARY_FILENAME,
    ENRICHMENT_MANIFEST_FILENAME,
)
from scripts.stock_alpha_news_transformer_sec_cache_failure_audit import (
    AUDIT_FILENAME,
    RETRY_CANDIDATES_FILENAME,
    audit_sec_primary_text_cache_failures,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rows() -> list[dict[str, str]]:
    return [
        {
            "event_key": "event-1",
            "symbol": "AAA",
            "form_type": "8-K",
            "accession_number": "0000000001-24-000001",
            "primary_document_url": "https://www.sec.gov/Archives/edgar/data/1/000000000124000001/aaa-8k.htm",
        },
        {
            "event_key": "event-2",
            "symbol": "BBB",
            "form_type": "10-Q",
            "accession_number": "0000000002-24-000002",
            "primary_document_url": "https://www.sec.gov/Archives/edgar/data/2/000000000224000002/bbb-10q.htm",
        },
    ]


def _summary(**overrides) -> dict:
    payload = {
        "requested_documents": 2,
        "cached_documents": 1,
        "failed_documents": 1,
        "document_text_max_length": 2214671,
    }
    payload.update(overrides)
    return payload


def _write_fixture(tmp_path: Path, documents: list[dict], *, summary: dict | None = None) -> tuple[Path, Path, Path]:
    reports = tmp_path / "reports"
    plan = reports / "plan"
    cache = reports / "cache"
    output = reports / "audit"
    _write_csv(plan / ENRICHMENT_MANIFEST_FILENAME, _rows())
    _write_json(cache / CACHE_MANIFEST_FILENAME, {"documents": documents})
    _write_json(cache / CACHE_SUMMARY_FILENAME, summary or _summary(failed_documents=len([d for d in documents if d.get("status") == "failed"])))
    return plan, cache, output


def _audit(tmp_path: Path, documents: list[dict], *, summary: dict | None = None) -> dict:
    plan, cache, output = _write_fixture(tmp_path, documents, summary=summary)
    return audit_sec_primary_text_cache_failures(
        enrichment_plan_dir=plan,
        cache_dir=cache,
        output_dir=output,
        reports_root=tmp_path / "reports",
    )


def test_detailed_failure_records_are_classified_and_retry_candidates_exclude_successes(tmp_path: Path) -> None:
    failed = dict(_rows()[0], status="failed", error_type="URLError", error="timed out", text_length=0, symbols=["AAA"], form_types=["8-K"])
    cached = dict(_rows()[1], status="cached", text_length=300, symbols=["BBB"], form_types=["10-Q"])
    audit = _audit(tmp_path, [failed, cached])
    retry_path = tmp_path / "reports" / "audit" / RETRY_CANDIDATES_FILENAME
    retry_rows = [json.loads(line) for line in retry_path.read_text(encoding="utf-8").splitlines()]

    assert audit["failure_observability_class"] == "A"
    assert audit["failure_counts_by_category"] == {"retryable_timeout": 1}
    assert retry_rows == [
        {
            "accession": "0000000001-24-000001",
            "document_id": f"{failed['accession_number']}|{failed['primary_document_url']}",
            "document_url": failed["primary_document_url"],
            "event_timestamp": None,
            "failure_category": "retryable_timeout",
            "form_type": "8-K",
            "previous_exception_type": "URLError",
            "previous_http_status": None,
            "retry_priority": "high",
            "symbol": "AAA",
        }
    ]


def test_reconstructs_missing_documents_without_guessing_reason(tmp_path: Path) -> None:
    cached = dict(_rows()[0], status="cached", text_length=300, symbols=["AAA"], form_types=["8-K"])
    audit = _audit(tmp_path, [cached], summary=_summary(cached_documents=1, failed_documents=0))

    assert audit["failure_observability_class"] == "B"
    assert audit["failure_counts_by_category"] == {"unknown_missing_from_cache": 1}
    assert audit["unknown_failure_count"] == 1


def test_audit_has_no_network_dependency_and_rejects_output_outside_reports(tmp_path: Path) -> None:
    plan, cache, _output = _write_fixture(tmp_path, [])
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        audit_sec_primary_text_cache_failures(
            enrichment_plan_dir=plan,
            cache_dir=cache,
            output_dir=tmp_path / "outside",
            reports_root=tmp_path / "reports",
        )


def test_retryable_and_non_retryable_classification(tmp_path: Path) -> None:
    timeout = dict(_rows()[0], status="failed", error_type="TimeoutError", error="timed out", symbols=["AAA"], form_types=["8-K"])
    not_found = dict(_rows()[1], status="failed", error_type="HTTPError", error="HTTP Error 404: Not Found", symbols=["BBB"], form_types=["10-Q"])
    audit = _audit(tmp_path, [timeout, not_found], summary=_summary(cached_documents=0, failed_documents=2))

    assert audit["retryable_failure_count"] == 1
    assert audit["non_retryable_failure_count"] == 1
    assert audit["http_404_count"] == 1


def test_duplicate_failure_detection_and_unknown_reason(tmp_path: Path) -> None:
    one = dict(_rows()[0], status="failed", symbols=["AAA"], form_types=["8-K"])
    two = dict(_rows()[0], status="failed", symbols=["AAA"], form_types=["8-K"])
    audit = _audit(tmp_path, [one, two], summary=_summary(cached_documents=0, failed_documents=2))

    assert audit["duplicate_failure_record_count"] == 1
    assert audit["failure_records_without_reason"] == 2
    assert audit["failure_counts_by_category"] == {"unknown_missing_from_cache": 2}


def test_empty_and_malformed_failure_sets(tmp_path: Path) -> None:
    empty = _audit(tmp_path, [], summary=_summary(cached_documents=0, failed_documents=0))
    malformed = _audit(tmp_path, [{"primary_document_url": "https://www.sec.gov/Archives/edgar/data/x/y/z.htm"}], summary=_summary(cached_documents=0, failed_documents=1))

    assert empty["failure_observability_class"] == "C"
    assert empty["failed_documents"] == 0
    assert malformed["failure_counts_by_category"] == {"malformed_failure_record": 1}


def test_writes_only_expected_report_artifacts(tmp_path: Path) -> None:
    failed = dict(_rows()[0], status="failed", error_type="ConnectionError", error="connection reset", symbols=["AAA"], form_types=["8-K"])
    _audit(tmp_path, [failed])
    output = tmp_path / "reports" / "audit"

    assert sorted(path.name for path in output.iterdir()) == [AUDIT_FILENAME, RETRY_CANDIDATES_FILENAME]
    assert not (tmp_path / "data" / "news").exists()
