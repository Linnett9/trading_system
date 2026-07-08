from __future__ import annotations

import csv
import hashlib
import json
import socket
from pathlib import Path

import pytest

from scripts.stock_alpha_news_market_data_inventory import (
    audit_timeframe,
    build_market_data_inventory,
)


def test_daily_symbol_coverage(tmp_path: Path) -> None:
    events = _events(tmp_path, ["AAA", "BBB"])
    daily = tmp_path / "daily"
    _write_csv(daily / "AAA.csv", _rows("AAA"))
    _write_csv(daily / "BBB.csv", _rows("BBB"))

    report = build_market_data_inventory(
        event_features_path=events,
        output_dir=tmp_path / "reports" / "audit",
        candidate_paths=_candidates(daily=daily),
    )

    assert report["timeframes"]["daily"]["covered_news_symbol_count"] == 2


def test_hourly_symbol_coverage(tmp_path: Path) -> None:
    hourly = tmp_path / "processed" / "AAA" / "1h"
    _write_csv(hourly / "bars.csv", _rows("AAA", timestamp="2024-01-02T15:00:00+00:00"))

    report = audit_timeframe("hourly", [tmp_path / "processed"], ["AAA"])

    assert report["covered_news_symbols"] == ["AAA"]


def test_five_minute_symbol_coverage(tmp_path: Path) -> None:
    five = tmp_path / "processed" / "AAA" / "5m"
    _write_csv(five / "bars.csv", _rows("AAA", timestamp="2024-01-02T15:00:00+00:00"))

    report = audit_timeframe("5_minute", [tmp_path / "processed"], ["AAA"])

    assert report["covered_news_symbol_count"] == 1


def test_fifteen_minute_symbol_coverage(tmp_path: Path) -> None:
    fifteen = tmp_path / "processed" / "AAA" / "15m"
    _write_csv(fifteen / "bars.csv", _rows("AAA", timestamp="2024-01-02T15:00:00+00:00"))

    report = audit_timeframe("15_minute", [tmp_path / "processed"], ["AAA"])

    assert report["covered_news_symbols"] == ["AAA"]


def test_missing_symbol_detection(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(daily / "AAA.csv", _rows("AAA"))

    report = audit_timeframe("daily", [daily], ["AAA", "BBB"])

    assert report["missing_news_symbols"] == ["BBB"]


def test_date_range_detection(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(
        daily / "AAA.csv",
        _rows("AAA", timestamp="2024-01-02") + _rows("AAA", timestamp="2024-01-05"),
    )

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["date_min"] == "2024-01-02"
    assert report["date_max"] == "2024-01-05"


def test_schema_detection(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(daily / "AAA.csv", _rows("AAA"))

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["required_columns_present"] is True
    assert "close" in report["detected_columns"]


def test_duplicate_timestamp_detection(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    rows = _rows("AAA", timestamp="2024-01-02") + _rows("AAA", timestamp="2024-01-02")
    _write_csv(daily / "AAA.csv", rows)

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["timestamp_duplicates"] == 1
    assert report["symbol_timestamp_duplicates"] == 1


def test_null_and_invalid_ohlc_detection(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(
        daily / "AAA.csv",
        [{"symbol": "AAA", "date": "2024-01-02", "open": "", "high": "1", "low": "-1", "close": "0", "volume": "-5"}],
    )

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["null_ohlc_count"] == 1
    assert report["nonpositive_price_count"] == 1
    assert report["negative_volume_count"] == 1


def test_adjusted_price_detection(tmp_path: Path) -> None:
    daily = tmp_path / "adjusted_prices"
    _write_csv(daily / "AAA.csv", _rows("AAA", adjusted=True))

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["adjusted_price_detected"] is True


def test_empty_directory(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    daily.mkdir()

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["file_count"] == 0
    assert report["missing_news_symbols"] == ["AAA"]


def test_one_file_per_symbol_layout(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(daily / "AAA.csv", _rows("AAA"))

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["layout"] == "one_file_per_symbol"


def test_multi_symbol_csv_layout(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(daily / "prices.csv", _rows("AAA") + _rows("BBB"))

    report = audit_timeframe("daily", [daily], ["AAA", "BBB"])

    assert report["layout"] == "multi_symbol_file"
    assert report["covered_news_symbol_count"] == 2


def test_timezone_unknown_handling(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    _write_csv(daily / "AAA.csv", _rows("AAA", timestamp="2024-01-02"))

    report = audit_timeframe("daily", [daily], ["AAA"])

    assert report["timezone_detected"] == "naive"


def test_output_restricted_to_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reports"):
        build_market_data_inventory(
            event_features_path=_events(tmp_path, ["AAA"]),
            output_dir=tmp_path / "not_reports",
            candidate_paths=_candidates(),
        )


def test_no_network_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    build_market_data_inventory(
        event_features_path=_events(tmp_path, ["AAA"]),
        output_dir=tmp_path / "reports" / "audit",
        candidate_paths=_candidates(),
    )


def test_source_files_unchanged(tmp_path: Path) -> None:
    daily = tmp_path / "daily"
    source = daily / "AAA.csv"
    _write_csv(source, _rows("AAA"))
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    build_market_data_inventory(
        event_features_path=_events(tmp_path, ["AAA"]),
        output_dir=tmp_path / "reports" / "audit",
        candidate_paths=_candidates(daily=daily),
    )

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_deterministic_report_output(tmp_path: Path) -> None:
    events = _events(tmp_path, ["AAA"])
    daily = tmp_path / "daily"
    _write_csv(daily / "AAA.csv", _rows("AAA"))
    output = tmp_path / "reports" / "audit"

    build_market_data_inventory(
        event_features_path=events,
        output_dir=output,
        candidate_paths=_candidates(daily=daily),
    )
    first = json.loads((output / "news_transformer_market_data_inventory.json").read_text())
    build_market_data_inventory(
        event_features_path=events,
        output_dir=output,
        candidate_paths=_candidates(daily=daily),
    )
    second = json.loads((output / "news_transformer_market_data_inventory.json").read_text())

    assert first == second


def _events(tmp_path: Path, symbols: list[str]) -> Path:
    path = tmp_path / "events.csv"
    path.write_text(
        "event_id,symbol,available_at_timestamp,event_timestamp\n"
        + "".join(
            f"{index},{symbol},2024-01-02T12:00:00+00:00,2024-01-02T12:00:00+00:00\n"
            for index, symbol in enumerate(symbols, start=1)
        ),
        encoding="utf-8",
    )
    return path


def _rows(
    symbol: str,
    *,
    timestamp: str = "2024-01-02",
    adjusted: bool = False,
) -> list[dict[str, str]]:
    row = {
        "symbol": symbol,
        "date": timestamp,
        "open": "10",
        "high": "12",
        "low": "9",
        "close": "11",
        "volume": "100",
    }
    if adjusted:
        row["adj_close"] = "11"
    return [row]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidates(
    *,
    daily: Path | None = None,
    hourly: Path | None = None,
    five: Path | None = None,
    fifteen: Path | None = None,
) -> dict[str, list[Path]]:
    return {
        "daily": [daily] if daily else [],
        "hourly": [hourly] if hourly else [],
        "5_minute": [five] if five else [],
        "15_minute": [fifteen] if fifteen else [],
    }
