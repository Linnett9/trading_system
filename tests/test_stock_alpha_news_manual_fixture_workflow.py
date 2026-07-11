from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.stock_level.news_sources.manual_fixture_workflow import (
    MANUAL_FIXTURE_WORKFLOW_SCHEMA_VERSION,
    write_manual_fixture_workflow_report,
)
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)


def test_tiny_jsonl_fixture_runs_end_to_end_and_writes_outputs(tmp_path: Path) -> None:
    fixture_path = tmp_path / "manual_fixture.jsonl"
    fixture_path.write_text(
        "\n".join(json.dumps(row) for row in [_row("AAPL"), _row("MSFT")]) + "\n",
        encoding="utf-8",
    )
    report_dir = tmp_path / "report"

    report, paths = write_manual_fixture_workflow_report(fixture_path, report_dir, sample_size=2)

    assert paths.workflow_report_json_path == report_dir / "manual_fixture_workflow_report.json"
    assert paths.workflow_summary_markdown_path == report_dir / "manual_fixture_workflow_summary.md"
    assert paths.composition_dir == report_dir / "composition"
    for path in (
        paths.workflow_report_json_path,
        paths.workflow_summary_markdown_path,
        paths.composition_dir / "composition_smoke_report.json",
        paths.composition_dir / "composition_smoke_summary.md",
        paths.composition_dir / "sample_selection" / "corpus_sample_rows.json",
        paths.composition_dir / "readiness" / "corpus_readiness_audit.json",
        paths.composition_dir / "corpus" / "corpus_rows.jsonl",
    ):
        assert path.exists()
        path.resolve(strict=False).relative_to(report_dir.resolve(strict=False))

    persisted_report = json.loads(paths.workflow_report_json_path.read_text(encoding="utf-8"))
    assert persisted_report == report
    assert report["schema_version"] == MANUAL_FIXTURE_WORKFLOW_SCHEMA_VERSION
    assert report["artifact_type"] == "manual_fixture_composition_smoke_workflow"
    assert report["fixture_path"] == str(fixture_path)
    assert report["fixture_format"] == "jsonl"
    assert report["sample_size"] == 2
    assert report["input_row_count"] == 2
    assert report["selected_row_count"] == 2
    assert report["corpus_row_count"] == 2
    assert report["skipped_row_count"] == 0
    assert report["composition_report_path"] == str(paths.composition_dir / "composition_smoke_report.json")
    assert set(report["output_files"]) == {
        "workflow_report_json",
        "workflow_summary_markdown",
        "composition_report_json",
        "composition_summary_markdown",
    }
    assert "Selected rows: 2" in paths.workflow_summary_markdown_path.read_text(encoding="utf-8")


def test_missing_fixture_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="input_path does not exist"):
        write_manual_fixture_workflow_report(
            tmp_path / "missing.jsonl",
            tmp_path / "report",
            sample_size=1,
        )


def test_directory_fixture_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit file"):
        write_manual_fixture_workflow_report(
            tmp_path,
            tmp_path / "report",
            sample_size=1,
        )


def test_protected_active_backfill_fixture_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected active backfill"):
        write_manual_fixture_workflow_report(
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "fixture.jsonl",
            tmp_path / "report",
            sample_size=1,
        )


def test_protected_active_backfill_output_path_is_rejected(tmp_path: Path) -> None:
    fixture_path = tmp_path / "manual_fixture.jsonl"
    fixture_path.write_text(json.dumps(_row("AAPL")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protected active backfill"):
        write_manual_fixture_workflow_report(
            fixture_path,
            Path(PROTECTED_ACTIVE_BACKFILL_PATH) / "manual-fixture-report",
            sample_size=1,
        )


def test_sample_size_is_enforced_through_composed_output(tmp_path: Path) -> None:
    fixture_path = tmp_path / "manual_fixture.jsonl"
    fixture_path.write_text(
        "\n".join(json.dumps(row) for row in [_row("AAPL"), _row("MSFT"), _row("NVDA")]) + "\n",
        encoding="utf-8",
    )

    report, paths = write_manual_fixture_workflow_report(fixture_path, tmp_path / "report", sample_size=2)
    composition_report = json.loads(
        (paths.composition_dir / "composition_smoke_report.json").read_text(encoding="utf-8")
    )
    sample_rows = json.loads(
        (paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(encoding="utf-8")
    )

    assert report["selected_row_count"] == 2
    assert report["corpus_row_count"] == 2
    assert composition_report["sample_excluded_row_count"] == 1
    assert composition_report["sample_skip_reasons"] == {"not_selected_sample_size_limit": 1}
    assert len(sample_rows) == 2


def test_sec_form_type_alone_does_not_become_economic_event_type(tmp_path: Path) -> None:
    fixture_path = tmp_path / "sec_fixture.jsonl"
    fixture_path.write_text(
        json.dumps(
            _row("AAPL")
            | {
                "provider": "sec_company_filings",
                "source": "sec",
                "source_type": "sec_filing",
                "form_type": "8-K",
                "event_type": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report, paths = write_manual_fixture_workflow_report(fixture_path, tmp_path / "report", sample_size=1)
    sample_rows = json.loads(
        (paths.composition_dir / "sample_selection" / "corpus_sample_rows.json").read_text(encoding="utf-8")
    )
    corpus_rows = [
        json.loads(line)
        for line in (paths.composition_dir / "corpus" / "corpus_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["selected_row_count"] == 1
    assert sample_rows[0]["source_type"] == "SEC_FILING"
    assert sample_rows[0]["event_type"] is None
    assert corpus_rows[0]["source_type"] == "SEC_FILING"
    assert corpus_rows[0]["event_type"] is None


def test_no_provider_config_network_backfill_features_models_or_brokers_are_needed(tmp_path: Path) -> None:
    fixture_path = tmp_path / "manual_fixture.jsonl"
    fixture_path.write_text(json.dumps(_row("AAPL")) + "\n", encoding="utf-8")

    report, _paths = write_manual_fixture_workflow_report(fixture_path, tmp_path / "report", sample_size=1)

    assert report["safety_flags"] == {
        "provider_collection_invoked": False,
        "network_invoked": False,
        "canonical_ingest_invoked": False,
        "historical_backfill_invoked": False,
        "corpus_assembly_invoked": False,
        "feature_generation_invoked": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "trading_impact": "none",
        "protected_active_backfill_path_rejected": True,
    }


def _row(
    symbol: str,
    *,
    provider_article_id: str = "1",
    published_at_utc: str = "2024-01-02T19:30:00Z",
) -> dict[str, str]:
    return {
        "article_id": f"alpaca_benzinga:{provider_article_id}:{symbol}",
        "provider": "alpaca_benzinga",
        "provider_article_id": provider_article_id,
        "provider_symbols": symbol,
        "symbol": symbol,
        "published_at_utc": published_at_utc,
        "provider_available_at_utc": "2024-01-02T19:31:00Z",
        "collected_at_utc": "2026-07-10T00:00:00Z",
        "source": "benzinga",
        "source_type": "newswire",
        "headline": f"{symbol} headline",
        "summary": "short summary",
        "body_or_full_text": "full body",
        "language": "en",
        "event_type": "",
        "relevance_status": "",
    }
