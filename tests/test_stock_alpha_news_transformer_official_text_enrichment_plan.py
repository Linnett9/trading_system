import csv
import json
from pathlib import Path

import pytest

from scripts.stock_alpha_news_transformer_official_text_enrichment_plan import (
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    build_official_text_enrichment_plan_report_only,
)


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


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _text_baseline_report(**overrides) -> dict:
    report = {
        "mode": "news_transformer_text_baseline_report_only",
        "research_only": True,
        "trading_impact": "none",
        "transformer_training_started": False,
        "leakage_violation_count": 0,
        "selected_model_average_test_metrics": {"balanced_accuracy": 0.5042, "log_loss": 0.7015},
        "metadata_baseline_average_test_metrics": {"balanced_accuracy": 0.5132},
        "text_vs_metadata_baseline": {
            "balanced_accuracy_delta": -0.009,
            "beats_metadata_on_balanced_accuracy": False,
        },
        "next_allowed_step": "review_news_transformer_text_baseline_report",
    }
    report.update(overrides)
    return report


def _labeled_rows() -> list[dict[str, str]]:
    return [
        {
            "event_key": "event-1",
            "provider": "sec_company_filings",
            "symbol": "AAA",
            "form_type": "8-K",
            "url_or_accession": "0000000001-24-000001",
            "available_at_timestamp": "2024-01-02T12:00:00Z",
            "label_date": "2024-01-02",
            "future_return_1d": "0.0100000000",
            "future_return_5d": "0.0200000000",
            "future_return_20d": "0.0300000000",
            "future_drawdown_20d": "-0.0100000000",
            "reduce_exposure_label": "false",
        },
        {
            "event_key": "event-2",
            "provider": "sec_company_filings",
            "symbol": "BBB",
            "form_type": "10-Q",
            "url_or_accession": "0000000002-24-000002",
            "available_at_timestamp": "2024-01-03T12:00:00Z",
            "label_date": "2024-01-03",
            "future_return_1d": "0.0100000000",
            "future_return_5d": "0.0200000000",
            "future_return_20d": "-0.0300000000",
            "future_drawdown_20d": "-0.0600000000",
            "reduce_exposure_label": "true",
        },
    ]


def _sec_rows(*, with_body_text: bool = False) -> list[dict[str, str]]:
    rows = [
        {
            "accession_number": "0000000001-24-000001",
            "form_type": "8-K",
            "primary_document_url": "https://www.sec.gov/Archives/edgar/data/1/000000000124000001/aaa-8k.htm",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/1/000000000124000001-index.htm",
        },
        {
            "accession_number": "0000000002-24-000002",
            "form_type": "10-Q",
            "primary_document_url": "https://www.sec.gov/Archives/edgar/data/2/000000000224000002/bbb-10q.htm",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/2/000000000224000002-index.htm",
        },
    ]
    if with_body_text:
        rows[0]["primary_document_text"] = "Official filing text cached locally."
    return rows


def _run(tmp_path: Path, *, baseline=None, sec_rows=None, output_dir=None):
    reports_root = tmp_path / "reports"
    labeled_path = _write_csv(reports_root / "labeled" / "events.csv", _labeled_rows())
    baseline_path = _write_json(reports_root / "text_baseline" / "report.json", baseline or _text_baseline_report())
    sec_root = reports_root / "sec_cache"
    _write_jsonl(sec_root / "sec_company_filings_event_rows.jsonl", sec_rows or _sec_rows())
    return build_official_text_enrichment_plan_report_only(
        labeled_dataset_path=labeled_path,
        text_baseline_report_path=baseline_path,
        sec_event_rows_root=sec_root,
        output_dir=output_dir or reports_root / "official_text_plan",
        reports_root=reports_root,
    )


def test_official_text_plan_refuses_unapproved_text_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="next_allowed_step"):
        _run(tmp_path, baseline=_text_baseline_report(next_allowed_step="train_transformer"))

    with pytest.raises(ValueError, match="text_did_not_beat_metadata"):
        _run(
            tmp_path,
            baseline=_text_baseline_report(
                text_vs_metadata_baseline={"beats_metadata_on_balanced_accuracy": True}
            ),
        )


def test_official_text_plan_maps_sec_accessions_and_blocks_without_cached_body_text(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert report["research_only"] is True
    assert report["trading_impact"] == "none"
    assert report["model_training_started"] is False
    assert report["rows_total"] == 2
    assert report["rows_with_required_labels"] == 2
    assert report["sec_rows_in_dataset"] == 2
    assert report["unique_dataset_accessions"] == 2
    assert report["matched_by_accession_count"] == 2
    assert report["matched_by_accession_rate"] == 1.0
    assert report["matched_rows_with_primary_document_url"] == 2
    assert report["local_body_text_available_count"] == 0
    assert report["blocking_reasons"] == ["official_sec_body_text_not_cached"]
    assert report["training_allowed"] is False
    assert report["next_allowed_step"] == "cache_official_sec_primary_document_text_report_only"


def test_official_text_plan_detects_existing_cached_body_text(tmp_path: Path) -> None:
    report = _run(tmp_path, sec_rows=_sec_rows(with_body_text=True))

    assert report["local_body_text_available_count"] == 1
    assert report["local_body_text_available_rate"] == 0.5
    assert report["blocking_reasons"] == []


def test_official_text_plan_output_stays_under_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_dir must be under reports"):
        _run(tmp_path, output_dir=tmp_path / "outside")

    _run(tmp_path)
    output_dir = tmp_path / "reports" / "official_text_plan"
    assert (output_dir / REPORT_FILENAME).exists()
    assert (output_dir / MANIFEST_FILENAME).exists()


def test_official_text_plan_does_not_import_training_or_trading_paths() -> None:
    source = Path("scripts/stock_alpha_news_transformer_official_text_enrichment_plan.py").read_text(encoding="utf-8")

    assert "torch" not in source
    assert "sklearn" not in source
    assert "broker" not in source
    assert "paper_trading" not in source
    assert "data/news" not in source
