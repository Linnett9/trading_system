import csv
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from scripts.stock_alpha_news_transformer_cache_sec_primary_text import (
    CACHE_MANIFEST_FILENAME,
    CACHE_SUMMARY_FILENAME,
    ENRICHMENT_MANIFEST_FILENAME,
    ENRICHMENT_REPORT_FILENAME,
    cache_sec_primary_text_report_only,
    extract_sec_document_text,
    main,
)


LONG_HTML = """
<html>
  <head><title>ignore me</title><style>.x { color: red; }</style></head>
  <body>
    <script>hidden()</script>
    <h1>Item 1.01 Entry Into a Material Definitive Agreement</h1>
    <p>Alpha company entered into an official filing agreement with enough text
    to pass the minimum extraction threshold for this report-only cache fixture.
    The terms include ordinary business details, board approvals, dates,
    counterparties, risk references, and additional harmless language for length.</p>
  </body>
</html>
"""


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _enrichment_report(**overrides) -> dict:
    report = {
        "mode": "news_transformer_official_text_enrichment_plan_report_only",
        "research_only": True,
        "training_allowed": False,
        "model_training_started": False,
        "transformer_training_started": False,
        "next_allowed_step": "cache_official_sec_primary_document_text_report_only",
    }
    report.update(overrides)
    return report


def _manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "event_key": "event-1",
            "symbol": "AAA",
            "form_type": "8-K",
            "accession_number": "0000000001-24-000001",
            "primary_document_url": "https://www.sec.gov/Archives/edgar/data/1/000000000124000001/aaa-8k.htm",
        },
        {
            "event_key": "event-1-duplicate",
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


def _plan_dir(tmp_path: Path, *, report=None, rows=None) -> Path:
    plan_dir = tmp_path / "reports" / "plan"
    _write_json(plan_dir / ENRICHMENT_REPORT_FILENAME, report or _enrichment_report())
    _write_csv(plan_dir / ENRICHMENT_MANIFEST_FILENAME, rows or _manifest_rows())
    return plan_dir


def _run(
    tmp_path: Path,
    *,
    output_dir=None,
    fetch_text=None,
    max_documents=None,
    overwrite=False,
    input_manifest=None,
    enrichment_plan_dir=None,
):
    reports_root = tmp_path / "reports"
    return cache_sec_primary_text_report_only(
        enrichment_plan_dir=enrichment_plan_dir if enrichment_plan_dir is not None else _plan_dir(tmp_path),
        output_dir=output_dir or reports_root / "sec_cache",
        reports_root=reports_root,
        input_manifest=input_manifest,
        max_documents=max_documents,
        sleep_seconds=0.0,
        overwrite=overwrite,
        fetch_text=fetch_text or (lambda _url, _user_agent, _timeout: LONG_HTML),
    )


def _retry_manifest(path: Path) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "document_id": "0000000002-24-000002|https://www.sec.gov/Archives/edgar/data/2/000000000224000002/bbb-10q.htm",
                "document_url": "https://www.sec.gov/Archives/edgar/data/2/000000000224000002/bbb-10q.htm",
                "accession": "0000000002-24-000002",
                "symbol": "BBB",
                "form_type": "10-Q",
                "retry_priority": "high",
            },
            {
                "document_id": "0000000003-24-000003|https://www.sec.gov/Archives/edgar/data/3/000000000324000003/ccc-8k.htm",
                "document_url": "https://www.sec.gov/Archives/edgar/data/3/000000000324000003/ccc-8k.htm",
                "accession": "0000000003-24-000003",
                "symbol": "CCC",
                "form_type": "8-K",
                "retry_priority": "medium",
            },
        ],
    )


def test_sec_primary_text_cache_deduplicates_duplicate_urls(tmp_path: Path) -> None:
    summary = _run(tmp_path)
    manifest = json.loads((tmp_path / "reports" / "sec_cache" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["requested_documents"] == 2
    assert summary["unique_document_urls"] == 2
    assert summary["cached_documents"] == 2
    assert len(manifest["documents"]) == 2
    assert manifest["documents"][0]["event_keys"] == ["event-1", "event-1-duplicate"]


def test_sec_primary_text_cache_refuses_output_outside_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _run(tmp_path, output_dir=tmp_path / "outside")


def test_sec_primary_text_cache_skips_existing_unless_overwrite(tmp_path: Path) -> None:
    calls = {"count": 0}

    def fetch(_url: str, _user_agent: str, _timeout: int) -> str:
        calls["count"] += 1
        return LONG_HTML

    first = _run(tmp_path, fetch_text=fetch, max_documents=1)
    second = _run(tmp_path, fetch_text=fetch, max_documents=1)
    third = _run(tmp_path, fetch_text=fetch, max_documents=1, overwrite=True)

    assert first["cached_documents"] == 1
    assert second["skipped_existing_documents"] == 1
    assert second["attempted_documents"] == 0
    assert third["cached_documents"] == 1
    assert calls["count"] == 2


def test_sec_primary_text_cache_records_failed_fetches_separately(tmp_path: Path) -> None:
    def failing_fetch(_url: str, _user_agent: str, _timeout: int) -> str:
        raise URLError("timed out")

    summary = _run(tmp_path, fetch_text=failing_fetch, max_documents=1)
    manifest = json.loads((tmp_path / "reports" / "sec_cache" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["failed_documents"] == 1
    assert summary["cached_documents"] == 0
    assert summary["rate_limit_or_timeout_failures"] == 1
    assert manifest["documents"][0]["status"] == "failed"


def test_sec_primary_text_cache_extracts_text_from_tiny_html_fixture() -> None:
    text = extract_sec_document_text(LONG_HTML)

    assert "Item 1.01 Entry Into a Material Definitive Agreement" in text
    assert "hidden()" not in text
    assert ".x" not in text


def test_sec_primary_text_cache_does_not_write_to_data_news(tmp_path: Path) -> None:
    _run(tmp_path)

    assert not (tmp_path / "data" / "news").exists()
    assert (tmp_path / "reports" / "sec_cache" / CACHE_SUMMARY_FILENAME).exists()


def test_sec_primary_text_cache_does_not_import_model_or_live_paths() -> None:
    source = Path("scripts/stock_alpha_news_transformer_cache_sec_primary_text.py").read_text(encoding="utf-8")

    assert "torch" not in source
    assert "sklearn" not in source
    assert "broker" not in source
    assert "paper_trading" not in source
    assert "live" not in source


def test_sec_primary_text_cache_emits_clear_next_allowed_step(tmp_path: Path) -> None:
    summary = _run(tmp_path)

    assert summary["next_allowed_step"] == "build_enriched_official_text_dataset_report_only"


def test_sec_primary_text_cache_uses_exact_retry_manifest_subset(tmp_path: Path) -> None:
    urls: list[str] = []
    retry_manifest = _retry_manifest(tmp_path / "reports" / "retry.jsonl")

    summary = _run(tmp_path, input_manifest=retry_manifest, enrichment_plan_dir=None, fetch_text=lambda url, _ua, _timeout: urls.append(url) or LONG_HTML)
    manifest = json.loads((tmp_path / "reports" / "sec_cache" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["requested_documents"] == 2
    assert urls == [
        "https://www.sec.gov/Archives/edgar/data/2/000000000224000002/bbb-10q.htm",
        "https://www.sec.gov/Archives/edgar/data/3/000000000324000003/ccc-8k.htm",
    ]
    assert [row["symbols"] for row in manifest["documents"]] == [["BBB"], ["CCC"]]


def test_sec_primary_text_cache_retry_manifest_ignores_successes_outside_manifest(tmp_path: Path) -> None:
    retry_manifest = _retry_manifest(tmp_path / "reports" / "retry.jsonl")
    summary = _run(tmp_path, input_manifest=retry_manifest, max_documents=1, enrichment_plan_dir=None)
    manifest = json.loads((tmp_path / "reports" / "sec_cache" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["requested_documents"] == 1
    assert manifest["documents"][0]["accession_number"] == "0000000002-24-000002"
    assert all(row["accession_number"] != "0000000001-24-000001" for row in manifest["documents"])


def test_sec_primary_text_cache_retry_manifest_skips_existing_file(tmp_path: Path) -> None:
    retry_manifest = _retry_manifest(tmp_path / "reports" / "retry.jsonl")
    calls = {"count": 0}

    def fetch(_url: str, _user_agent: str, _timeout: int) -> str:
        calls["count"] += 1
        return LONG_HTML

    first = _run(tmp_path, input_manifest=retry_manifest, enrichment_plan_dir=None, fetch_text=fetch, max_documents=1)
    second = _run(tmp_path, input_manifest=retry_manifest, enrichment_plan_dir=None, fetch_text=fetch, max_documents=1)

    assert first["cached_documents"] == 1
    assert second["skipped_existing_documents"] == 1
    assert second["attempted_documents"] == 0
    assert calls["count"] == 1


def test_sec_primary_text_cache_rejects_malformed_retry_manifest_rows(tmp_path: Path) -> None:
    malformed = _write_jsonl(tmp_path / "reports" / "retry.jsonl", ["{not-json"])
    missing_url = _write_jsonl(tmp_path / "reports" / "missing-url.jsonl", [{"document_id": "stable", "retry_priority": "high"}])
    missing_identity = _write_jsonl(
        tmp_path / "reports" / "missing-identity.jsonl",
        [{"document_url": "https://www.sec.gov/Archives/edgar/data/4/0004/doc.htm", "retry_priority": "high"}],
    )

    with pytest.raises(ValueError, match="malformed retry manifest JSON"):
        _run(tmp_path, input_manifest=malformed, enrichment_plan_dir=None)
    with pytest.raises(ValueError, match="missing document_url"):
        _run(tmp_path, input_manifest=missing_url, enrichment_plan_dir=None)
    with pytest.raises(ValueError, match="missing stable document identity"):
        _run(tmp_path, input_manifest=missing_identity, enrichment_plan_dir=None)


def test_sec_primary_text_cache_retry_manifest_deduplicates_and_excludes_do_not_retry(tmp_path: Path) -> None:
    retry_manifest = _write_jsonl(
        tmp_path / "reports" / "retry.jsonl",
        [
            {
                "document_id": "same|https://www.sec.gov/Archives/edgar/data/5/0005/doc.htm",
                "document_url": "https://www.sec.gov/Archives/edgar/data/5/0005/doc.htm",
                "accession": "same",
                "retry_priority": "high",
            },
            {
                "document_id": "same|https://www.sec.gov/Archives/edgar/data/5/0005/doc.htm",
                "document_url": "https://www.sec.gov/Archives/edgar/data/5/0005/doc.htm",
                "accession": "same",
                "retry_priority": "high",
            },
            {
                "document_id": "skip|https://www.sec.gov/Archives/edgar/data/6/0006/doc.htm",
                "document_url": "https://www.sec.gov/Archives/edgar/data/6/0006/doc.htm",
                "accession": "skip",
                "retry_priority": "do_not_retry",
            },
        ],
    )

    summary = _run(tmp_path, input_manifest=retry_manifest, enrichment_plan_dir=None)
    manifest = json.loads((tmp_path / "reports" / "sec_cache" / CACHE_MANIFEST_FILENAME).read_text())

    assert summary["requested_documents"] == 1
    assert [row["accession_number"] for row in manifest["documents"]] == ["same"]


def test_sec_primary_text_cache_retry_manifest_can_write_to_separate_output_dir(tmp_path: Path) -> None:
    retry_manifest = _retry_manifest(tmp_path / "reports" / "retry.jsonl")
    output_dir = tmp_path / "reports" / "retry_smoke"
    summary = _run(tmp_path, input_manifest=retry_manifest, enrichment_plan_dir=None, output_dir=output_dir, max_documents=1)

    assert summary["cache_dir"] == str(output_dir / "documents")
    assert (output_dir / CACHE_SUMMARY_FILENAME).exists()
    assert not (tmp_path / "reports" / "sec_cache" / CACHE_SUMMARY_FILENAME).exists()


def test_sec_primary_text_cache_cli_accepts_retry_manifest_resume_flags_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retry_manifest = _retry_manifest(tmp_path / "reports" / "retry.jsonl")
    calls: list[str] = []

    def fetch(url: str, _user_agent: str, _timeout: int) -> str:
        calls.append(url)
        return LONG_HTML

    monkeypatch.setattr("scripts.stock_alpha_news_transformer_cache_sec_primary_text.standard_library_sec_text_get", fetch)

    exit_code = main(
        [
            "--input-manifest",
            str(retry_manifest),
            "--output-dir",
            str(tmp_path / "reports" / "retry_smoke"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--max-documents",
            "1",
            "--resume",
            "--skip-existing",
        ]
    )

    assert exit_code == 0
    assert calls == ["https://www.sec.gov/Archives/edgar/data/2/000000000224000002/bbb-10q.htm"]
