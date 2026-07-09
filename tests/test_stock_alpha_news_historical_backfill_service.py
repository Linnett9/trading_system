from __future__ import annotations

import json
from pathlib import Path

import pytest

import application.services.ml_commands_stock as stock_commands
from core.research.ml.stock_level.stock_alpha_news_historical_backfill import (
    StockAlphaNewsHistoricalBackfillPaths,
    write_stock_alpha_news_historical_backfill,
)


@pytest.mark.parametrize("action", ["collect", "backfill"])
def test_historical_backfill_collection_actions_report_summary_json(monkeypatch, capsys, tmp_path, action):
    summary_path = tmp_path / "stock_alpha_news_historical_backfill_summary.json"
    assembly_path = tmp_path / "stock_alpha_news_historical_corpus_assembly.json"
    summary_path.write_text(
        json.dumps({
            "action": "collect",
            "status_counts": {"complete": 1},
            "processed_this_run": 1,
        }),
        encoding="utf-8",
    )
    result = _paths(tmp_path, summary_json_path=summary_path, assembly_json_path=assembly_path)
    called = {"count": 0}

    def fake_writer(config):
        called["count"] += 1
        return result

    monkeypatch.setattr(stock_commands, "write_stock_alpha_news_historical_backfill", fake_writer)

    stock_commands.run_ml_stock_alpha_news_historical_backfill(_config(tmp_path, action=action))

    output = capsys.readouterr().out
    assert called["count"] == 1
    assert f"JSON: {summary_path}" in output
    assert f"JSON: {assembly_path}" not in output


def test_historical_backfill_assembly_action_reports_assembly_json(monkeypatch, capsys, tmp_path):
    summary_path = tmp_path / "stock_alpha_news_historical_backfill_summary.json"
    assembly_path = tmp_path / "stock_alpha_news_historical_corpus_assembly.json"
    assembly_path.write_text(
        json.dumps({
            "action": "assemble",
            "row_count": 2,
            "incomplete_partition_count": 0,
        }),
        encoding="utf-8",
    )
    result = _paths(tmp_path, summary_json_path=summary_path, assembly_json_path=assembly_path)

    monkeypatch.setattr(stock_commands, "write_stock_alpha_news_historical_backfill", lambda config: result)

    stock_commands.run_ml_stock_alpha_news_historical_backfill(_config(tmp_path, action="assemble"))

    output = capsys.readouterr().out
    assert "row_count=2" in output
    assert f"JSON: {assembly_path}" in output
    assert f"JSON: {summary_path}" not in output


def test_historical_backfill_unknown_action_fails_clearly(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        stock_commands.run_ml_stock_alpha_news_historical_backfill(_config(tmp_path, action="publish"))

    assert exc_info.value.code == 1
    assert "blocking_issue=unsupported historical backfill action: publish" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("action", "expected_name"),
    [
        ("collect", "stock_alpha_news_historical_backfill_summary.json"),
        ("assemble", "stock_alpha_news_historical_corpus_assembly.json"),
    ],
)
def test_historical_backfill_missing_action_artifact_names_expected_path(
    monkeypatch,
    tmp_path,
    action,
    expected_name,
):
    result = _paths(tmp_path)
    monkeypatch.setattr(stock_commands, "write_stock_alpha_news_historical_backfill", lambda config: result)

    with pytest.raises(FileNotFoundError) as exc_info:
        stock_commands.run_ml_stock_alpha_news_historical_backfill(_config(tmp_path, action=action))

    message = str(exc_info.value)
    assert f"historical backfill action '{action}' expected artifact was not written" in message
    assert expected_name in message


def test_historical_backfill_writer_rejects_unknown_action_before_provider_collection(tmp_path):
    config = _config(tmp_path, action="publish")

    with pytest.raises(ValueError, match="unsupported historical backfill action: publish"):
        write_stock_alpha_news_historical_backfill(config, sources={"alpaca_benzinga": _FailingSource()})


def _config(tmp_path: Path, *, action: str) -> dict:
    return {
        "ml": {
            "stock_alpha_news_historical_backfill": {
                "action": action,
                "dry_run": False,
                "work_dir": str(tmp_path),
                "provider": "alpaca_benzinga",
                "start_date": "2016-01-01",
                "end_date": "2016-01-31",
                "symbols": ["AAPL"],
                "symbol_batch_size": 1,
                "provider_config": {"enabled": True},
            },
            "stock_alpha_news_enable_transformer": False,
        }
    }


def _paths(
    tmp_path: Path,
    *,
    summary_json_path: Path | None = None,
    assembly_json_path: Path | None = None,
) -> StockAlphaNewsHistoricalBackfillPaths:
    return StockAlphaNewsHistoricalBackfillPaths(
        manifest_path=tmp_path / "stock_alpha_news_historical_backfill_manifest.json",
        summary_json_path=summary_json_path or tmp_path / "stock_alpha_news_historical_backfill_summary.json",
        summary_markdown_path=tmp_path / "stock_alpha_news_historical_backfill_summary.md",
        assembly_csv_path=tmp_path / "stock_alpha_news_historical_corpus_assembly.csv",
        assembly_json_path=assembly_json_path or tmp_path / "stock_alpha_news_historical_corpus_assembly.json",
        assembly_markdown_path=tmp_path / "stock_alpha_news_historical_corpus_assembly.md",
    )


class _FailingSource:
    api_key_required = False

    def with_provider_config(self, provider_config):
        return self

    def collect(self, **kwargs):
        raise AssertionError("provider collection should not run for unknown action")
