from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_stock_alpha_sec_event_row_coverage import build_sec_event_row_coverage_summary
from scripts.plan_stock_alpha_sec_window_reruns import build_sec_window_rerun_plan


def _write_report(
    reports_root: Path,
    family_name: str,
    payload: dict,
    event_rows: list[dict] | None = None,
) -> Path:
    report_dir = reports_root / family_name / "dev"
    report_dir.mkdir(parents=True)
    if event_rows is not None:
        event_rows_path = report_dir / "sec_company_filings_event_rows.jsonl"
        event_rows_path.write_text(
            "".join(json.dumps(row) + "\n" for row in event_rows),
            encoding="utf-8",
        )
        payload["sec_company_filings_event_rows_path"] = str(event_rows_path)
    report_path = report_dir / "stock_alpha_news_free_source_collect.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def _payload(
    symbol: str,
    *,
    attempted: bool = True,
    rows: int = 0,
    provider_failed: str | None = None,
    rate_limited: bool = False,
    output_path: str | None = None,
) -> dict:
    requested = [symbol]
    attempted_symbols = [symbol] if attempted else []
    provider_failures = {"sec_company_filings": provider_failed} if provider_failed else {}
    batch = {"sec_company_filings_attempted_symbols": attempted_symbols}
    if provider_failed and "TimeoutError" in provider_failed:
        batch["error_type"] = "TimeoutError"
    return {
        "only_symbols": requested,
        "provider_batch_diagnostics": [batch],
        "returned_symbols": [symbol] if rows else [],
        "rows_by_symbol": {symbol: rows},
        "provider_row_counts": {"sec_company_filings": rows},
        "providers_failed": provider_failures,
        "providers_rate_limited": ["sec_company_filings"] if rate_limited else [],
        "output_written": True,
        "output_path": output_path or "reports/ml/benchmark/sec.csv",
    }


def test_audit_summary_classifies_sec_event_row_coverage(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_001_dry_run",
        _payload("AAA", rows=1),
        event_rows=[{"symbol": "AAA", "source": "sec_company_filings"}],
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_002_dry_run",
        _payload("BBB", rows=2),
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_003_dry_run",
        _payload("CCC", provider_failed="TimeoutError: read timed out"),
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_004_dry_run",
        _payload("DDD", provider_failed="URLError: connection reset"),
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_005_dry_run",
        _payload("EEE", rate_limited=True),
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_006_dry_run",
        _payload("FFF", attempted=False),
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_007_dry_run",
        _payload("GGG", output_path="data/news/raw/sec.csv"),
    )

    summary = build_sec_event_row_coverage_summary(
        reports_root=reports_root,
        config_dir=config_dir,
        window_months=120,
        include_retries=True,
    )

    assert summary["classification_counts"] == {
        "missing_report": 0,
        "provider_failure": 1,
        "rate_limited": 1,
        "success_missing_event_rows": 1,
        "success_with_event_rows": 1,
        "success_zero_rows_clean": 0,
        "timeout_failure": 1,
        "unattempted_symbols": 1,
        "unsafe_output_path": 1,
    }
    assert summary["successful_reports_missing_event_rows"] == 1
    assert (
        "stock_alpha_news_collect_sec_company_filings_120mo_part_002_dry_run"
        in summary["reports_that_should_be_rerun"]
    )
    assert summary["timeout_symbols"] == ["CCC"]
    assert summary["provider_failure_symbols"] == ["DDD"]
    assert summary["rate_limited_symbols"] == ["EEE"]
    assert summary["unattempted_symbols"] == ["FFF"]
    assert summary["total_event_rows_found"] == 1
    assert summary["total_provider_rows_reported"] == 3
    assert summary["mismatch_count"] == 0
    assert summary["touches_data_news"] is True


def test_planner_emits_one_symbol_retry_commands_without_running_them(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_002_dry_run",
        _payload("BBB", rows=1),
    )
    _write_report(
        reports_root,
        "stock_alpha_news_collect_sec_company_filings_120mo_part_003_dry_run",
        _payload("CCC", provider_failed="TimeoutError: read timed out"),
    )
    summary = build_sec_event_row_coverage_summary(
        reports_root=reports_root,
        config_dir=config_dir,
        window_months=120,
        include_retries=True,
    )

    plan = build_sec_window_rerun_plan(summary)

    assert "# Run locally outside Codex." in plan
    assert "config.stock_alpha_news_collect_sec_company_filings_120mo_part_002_dry_run.yaml" in plan
    assert "120mo_retry_part_003_CCC" in plan
    assert 'main.py --mode ml-stock-alpha-news-collect-free-sources --config "' in plan
    assert not (tmp_path / "plan.sh").exists()
