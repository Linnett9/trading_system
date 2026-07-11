from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.stock_level.news_sources.manual_fixture_workflow import (
    write_manual_fixture_workflow_report,
)


FIXTURE_PATH = Path("tests/fixtures/stock_alpha_news/manual_fixture_workflow_tiny.jsonl")


def test_realistic_manual_fixture_runs_deterministic_workflow_under_tmp_path(tmp_path: Path) -> None:
    report_dir = tmp_path / "manual-fixture-report"

    report, paths = write_manual_fixture_workflow_report(FIXTURE_PATH, report_dir, sample_size=4)

    assert paths.workflow_report_json_path.parent == report_dir
    assert paths.workflow_summary_markdown_path.parent == report_dir
    assert paths.composition_dir == report_dir / "composition"
    assert paths.workflow_report_json_path.exists()
    assert paths.workflow_summary_markdown_path.exists()
    assert (paths.composition_dir / "composition_smoke_report.json").exists()
    assert (paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").exists()
    assert (paths.composition_dir / "sample_selection" / "corpus_sample_selection_audit.json").exists()
    assert (paths.composition_dir / "corpus" / "corpus_rows.jsonl").exists()

    persisted_report = json.loads(paths.workflow_report_json_path.read_text(encoding="utf-8"))
    composition_report = json.loads(
        (paths.composition_dir / "composition_smoke_report.json").read_text(encoding="utf-8")
    )
    sample_audit = json.loads(
        (paths.composition_dir / "sample_selection" / "corpus_sample_selection_audit.json").read_text(
            encoding="utf-8"
        )
    )
    sample_rows = json.loads(
        (paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(encoding="utf-8")
    )
    corpus_rows = [
        json.loads(line)
        for line in (paths.composition_dir / "corpus" / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert persisted_report == report
    assert report["fixture_path"] == str(FIXTURE_PATH)
    assert report["fixture_format"] == "jsonl"
    assert report["sample_size"] == 4
    assert report["input_row_count"] == 7
    assert report["selected_row_count"] == 4
    assert report["corpus_row_count"] == 4
    assert report["sample_excluded_row_count"] == 3
    assert report["skipped_row_count"] == 0
    assert report["composition_report_path"] == str(paths.composition_dir / "composition_smoke_report.json")
    assert report["safety_flags"]["provider_collection_invoked"] is False
    assert report["safety_flags"]["network_invoked"] is False
    assert report["safety_flags"]["historical_backfill_invoked"] is False
    assert report["safety_flags"]["feature_generation_invoked"] is False
    assert report["safety_flags"]["model_training_invoked"] is False
    assert report["safety_flags"]["model_inference_invoked"] is False
    assert report["safety_flags"]["trading_impact"] == "none"

    assert composition_report["sample_skip_reasons"] == {
        "missing_provider": 1,
        "missing_publication_timestamp": 1,
        "missing_symbol": 1,
        "missing_text": 1,
    }
    assert composition_report["corpus_skip_reasons"] == {}
    assert sample_audit["excluded_rows"] == [
        {
            "row_number": 5,
            "provider": "alpaca_benzinga",
            "symbol": "AMZN",
            "provider_article_id": "benzinga-missing-text",
            "reasons": ["missing_text"],
        },
        {
            "row_number": 6,
            "provider": "alpaca_benzinga",
            "symbol": "GOOGL",
            "provider_article_id": "benzinga-missing-time",
            "reasons": ["missing_publication_timestamp"],
        },
        {
            "row_number": 7,
            "provider": "",
            "symbol": "",
            "provider_article_id": "manual-missing-symbol-provider",
            "reasons": ["missing_symbol", "missing_provider"],
        },
    ]
    assert [row["symbol"] for row in sample_rows] == ["AAPL", "MSFT", "TSLA", "NVDA"]
    assert [row["symbol"] for row in corpus_rows] == ["AAPL", "MSFT", "TSLA", "NVDA"]

    sample_by_symbol = {row["symbol"]: row for row in sample_rows}
    corpus_by_symbol = {row["symbol"]: row for row in corpus_rows}
    assert sample_by_symbol["TSLA"]["source_type"] == "SEC_FILING"
    assert sample_by_symbol["TSLA"]["event_type"] is None
    assert corpus_by_symbol["TSLA"]["source_type"] == "SEC_FILING"
    assert corpus_by_symbol["TSLA"]["event_type"] is None
    assert sample_by_symbol["NVDA"]["event_type"] == "earnings"
    assert corpus_by_symbol["NVDA"]["event_type"] == "earnings"

    second_report, second_paths = write_manual_fixture_workflow_report(
        FIXTURE_PATH,
        tmp_path / "manual-fixture-report-second",
        sample_size=4,
    )
    second_sample_rows = json.loads(
        (second_paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(encoding="utf-8")
    )
    assert second_report["selected_row_count"] == report["selected_row_count"]
    assert second_report["corpus_row_count"] == report["corpus_row_count"]
    assert [row["symbol"] for row in second_sample_rows] == [row["symbol"] for row in sample_rows]
