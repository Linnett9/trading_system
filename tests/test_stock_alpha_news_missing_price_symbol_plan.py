from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

from scripts import stock_alpha_news_missing_price_symbol_plan as planner
from scripts.stock_alpha_news_missing_price_symbol_plan import (
    build_missing_price_symbol_plan,
    write_missing_price_symbol_plan,
)


def test_missing_symbol_detection(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [
            {"symbol": "AAA", "available_at_timestamp": "2024-01-01T10:00:00Z"},
            {"symbol": "BBB", "available_at_timestamp": "2024-01-02T10:00:00Z"},
        ],
    )
    adjusted = tmp_path / "adjusted"
    _write_csv(adjusted / "AAA.csv", [_price_row("AAA", "2024-01-01")])

    report = build_missing_price_symbol_plan(
        event_dataset_path=events,
        adjusted_price_dir=adjusted,
        output_path=tmp_path / "reports" / "plan.json",
        reports_root=tmp_path / "reports",
        project_root=tmp_path,
    )

    assert report["missing_price_symbol_count"] == 1
    assert report["missing_price_symbols"] == ["BBB"]


def test_covered_symbol_detection(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [{"symbol": "aaa", "available_at_timestamp": "2024-01-01T10:00:00Z"}],
    )
    adjusted = tmp_path / "adjusted"
    _write_csv(adjusted / "AAA.csv", [_price_row("AAA", "2024-01-01")])

    report = build_missing_price_symbol_plan(
        event_dataset_path=events,
        adjusted_price_dir=adjusted,
        output_path=tmp_path / "reports" / "plan.json",
        reports_root=tmp_path / "reports",
        project_root=tmp_path,
    )

    assert report["covered_price_symbol_count"] == 1
    assert report["covered_price_symbols"] == ["AAA"]


def test_date_range_summary(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [
            {"symbol": "AAA", "available_at_timestamp": "2024-02-03T10:00:00Z"},
            {"symbol": "AAA", "event_timestamp": "2024-01-03T10:00:00Z"},
        ],
    )
    adjusted = tmp_path / "adjusted"
    _write_csv(
        adjusted / "AAA.csv",
        [_price_row("AAA", "2020-01-02"), _price_row("AAA", "2024-02-05")],
    )

    report = build_missing_price_symbol_plan(
        event_dataset_path=events,
        adjusted_price_dir=adjusted,
        output_path=tmp_path / "reports" / "plan.json",
        reports_root=tmp_path / "reports",
        project_root=tmp_path,
    )

    assert report["existing_price_date_min"] == "2020-01-02"
    assert report["existing_price_date_max"] == "2024-02-05"
    assert report["required_history_start"] == "2024-01-03"
    assert report["required_history_end"] == "2024-02-03"


def test_no_api_or_download_calls() -> None:
    source = inspect.getsource(planner)

    assert "urlopen" not in source
    assert "requests" not in source
    assert "fetch_chart" not in source
    assert "download_rows" not in source


def test_output_stays_under_reports(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [{"symbol": "AAA", "available_at_timestamp": "2024-01-01T10:00:00Z"}],
    )

    with pytest.raises(ValueError, match="under reports"):
        build_missing_price_symbol_plan(
            event_dataset_path=events,
            adjusted_price_dir=tmp_path / "adjusted",
            output_path=tmp_path / "outside.json",
            reports_root=tmp_path / "reports",
            project_root=tmp_path,
        )


def test_report_includes_recommended_next_step(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [{"symbol": "AAA", "available_at_timestamp": "2024-01-01T10:00:00Z"}],
    )
    importer = tmp_path / "infrastructure" / "data" / "yahoo_adjusted_price_importer.py"
    importer.parent.mkdir(parents=True)
    importer.write_text("# importer marker\n", encoding="utf-8")

    report = build_missing_price_symbol_plan(
        event_dataset_path=events,
        adjusted_price_dir=tmp_path / "adjusted",
        output_path=tmp_path / "reports" / "plan.json",
        reports_root=tmp_path / "reports",
        project_root=tmp_path,
    )

    assert report["candidate_price_importers_found"] == [
        "infrastructure/data/yahoo_adjusted_price_importer.py"
    ]
    assert (
        report["recommended_next_step"]
        == "run_existing_adjusted_price_importer_for_missing_news_symbols_after_explicit_download_approval"
    )


def test_handles_empty_adjusted_price_directory(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [{"symbol": "AAA", "available_at_timestamp": "2024-01-01T10:00:00Z"}],
    )
    adjusted = tmp_path / "adjusted"
    adjusted.mkdir()

    report = build_missing_price_symbol_plan(
        event_dataset_path=events,
        adjusted_price_dir=adjusted,
        output_path=tmp_path / "reports" / "plan.json",
        reports_root=tmp_path / "reports",
        project_root=tmp_path,
    )

    assert report["covered_price_symbol_count"] == 0
    assert report["missing_price_symbols"] == ["AAA"]
    assert report["existing_price_date_min"] == ""
    assert report["adjusted_price_required_columns_present"] is False


def test_handles_one_symbol_csv_fixture(tmp_path: Path) -> None:
    events = _write_csv(
        tmp_path / "events.csv",
        [{"symbol": "AAA", "available_at_timestamp": "2024-01-01T10:00:00Z"}],
    )
    adjusted = tmp_path / "adjusted"
    _write_csv(adjusted / "custom_name.csv", [_price_row("AAA", "2024-01-01")])
    output = tmp_path / "reports" / "plan.json"

    report = write_missing_price_symbol_plan(
        event_dataset_path=events,
        adjusted_price_dir=adjusted,
        output_path=output,
        reports_root=tmp_path / "reports",
        project_root=tmp_path,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert report["event_symbol_count"] == 1
    assert saved["covered_price_symbols"] == ["AAA"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _price_row(symbol: str, date: str) -> dict[str, str]:
    return {
        "symbol": symbol,
        "date": date,
        "open": "10",
        "high": "11",
        "low": "9",
        "close": "10",
        "adj_close": "10",
        "volume": "100",
    }
