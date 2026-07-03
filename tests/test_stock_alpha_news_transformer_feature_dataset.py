import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_feature_dataset import (
    build_report_only_news_transformer_feature_dataset,
)


def _event(
    symbol: str = "AAPL",
    *,
    timestamp: str = "2024-01-02T13:00:00Z",
    title: str = "10-Q filed by AAPL",
    source_url: str = "https://www.sec.gov/aapl",
) -> dict:
    return {
        "accepted_datetime": timestamp,
        "accession_number": f"000-{symbol}",
        "form_type": "10-Q",
        "headline_or_title": title,
        "provider": "sec_company_filings",
        "published_at_utc": timestamp,
        "source_type": "sec_filing",
        "source_url": source_url,
        "symbol": symbol,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _build(tmp_path: Path, *, gate=None, events=None, preflight=None, output_dir=None):
    reports = tmp_path / "reports"
    event_path = _write_jsonl(
        reports / "selected" / "sec_company_filings_event_rows.jsonl",
        events if events is not None else [_event("AAPL"), _event("MSFT", timestamp="2024-01-03T13:00:00Z")],
    )
    contract_preflight = {
        "sec_event_rows_included": [str(event_path)],
        "rows_checked_by_provider": {"sec_company_filings": 2},
        "unresolved_provider_timeout_symbols": [],
    }
    if preflight:
        contract_preflight.update(preflight)
    return build_report_only_news_transformer_feature_dataset(
        feature_gate=gate or {"approved": True},
        coverage_audit={"sec_event_rows_included": [str(event_path)]},
        contract_preflight=contract_preflight,
        reports_root=reports,
        output_dir=output_dir or reports / "features",
    )


def test_builder_refuses_when_feature_gate_is_not_approved(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature-generation gate is not approved"):
        _build(tmp_path, gate={"approved": False})


def test_builder_emits_event_only_dataset_without_labels(tmp_path: Path) -> None:
    report = _build(tmp_path)
    dataset_path = Path(report["dataset_path"])

    rows = list(csv.DictReader(dataset_path.open(encoding="utf-8")))

    assert report["rows"] == 2
    assert report["labels_attached"] is False
    assert report["next_allowed_step"] == "attach_price_return_labels"
    assert {"event_id", "event_key", "symbol", "available_at_timestamp", "is_sec_filing"} <= set(rows[0])
    assert "future_return_1d" not in rows[0]


def test_duplicate_event_keys_and_future_timestamps_are_detected(tmp_path: Path) -> None:
    future = "2999-01-01T00:00:00Z"
    duplicate = _event("AAPL", timestamp=future, source_url="https://www.sec.gov/duplicate")
    report = _build(tmp_path, events=[duplicate, duplicate])

    assert report["input_duplicate_event_key_count"] == 1
    assert report["duplicate_event_key_count"] == 0
    assert report["future_timestamp_count"] == 1
    assert "overlapping selected SEC event rows were deduplicated by event_key" in report["warnings"]
    assert "future event timestamps detected" in report["blocking_reasons"]


def test_unresolved_timeout_artifacts_block_and_data_news_paths_are_excluded(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    valid = _write_jsonl(reports / "valid" / "sec_company_filings_event_rows.jsonl", [_event("AAPL")])
    data_news_path = tmp_path / "data" / "news" / "sec_company_filings_event_rows.jsonl"
    preflight = {
        "sec_event_rows_included": [str(valid), str(data_news_path)],
        "unresolved_provider_timeout_symbols": ["META"],
    }
    report = build_report_only_news_transformer_feature_dataset(
        feature_gate={"approved": True},
        coverage_audit={},
        contract_preflight=preflight,
        reports_root=reports,
        output_dir=reports / "features",
    )

    assert report["rows"] == 1
    assert "unresolved provider timeout artifacts remain" in report["blocking_reasons"]
    assert all("data/news" not in path for path in report["selected_sec_event_rows_included"])


def test_outputs_must_stay_under_reports_and_no_training_path_is_emitted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _build(tmp_path, output_dir=tmp_path / "outside")

    report = _build(tmp_path)
    rendered = json.dumps(report).lower()

    assert "/reports/" in report["dataset_path"] or report["dataset_path"].startswith(str(tmp_path / "reports"))
    assert "model_training" not in rendered
    assert "transformer_training" not in rendered
    assert "raw_write" not in rendered
