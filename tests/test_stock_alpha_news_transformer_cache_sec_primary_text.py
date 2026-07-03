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


def _run(tmp_path: Path, *, output_dir=None, fetch_text=None, max_documents=None, overwrite=False):
    reports_root = tmp_path / "reports"
    return cache_sec_primary_text_report_only(
        enrichment_plan_dir=_plan_dir(tmp_path),
        output_dir=output_dir or reports_root / "sec_cache",
        reports_root=reports_root,
        max_documents=max_documents,
        sleep_seconds=0.0,
        overwrite=overwrite,
        fetch_text=fetch_text or (lambda _url, _user_agent, _timeout: LONG_HTML),
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
