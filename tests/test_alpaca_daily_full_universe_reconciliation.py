from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from infrastructure.data.alpaca_daily_full_universe_reconciliation import (
    classify_reconciliation_row,
    read_alpaca_archive,
    run_full_universe_reconciliation,
)


def test_full_universe_audit_and_reconciliation_outputs_are_deterministic(tmp_path):
    alpaca_root, stooq_root, assets, aliases = _fixture_archives(tmp_path)
    report_root = tmp_path / "reports"

    result = run_full_universe_reconciliation(
        alpaca_archive_root=alpaca_root,
        stooq_root=stooq_root,
        asset_registry=assets,
        alias_registry=aliases,
        report_root=report_root,
    )
    second = run_full_universe_reconciliation(
        alpaca_archive_root=alpaca_root,
        stooq_root=stooq_root,
        asset_registry=assets,
        alias_registry=aliases,
        report_root=report_root,
    )

    assert result["audit"]["file_count"] == 4
    assert result["audit"]["row_count"] == 11
    assert result["audit"]["symbol_count"] == 4
    assert result["audit"]["missing_grid_rows"] == 1
    assert result["audit"]["duplicate_symbol_session_rows"] == 0
    assert result["audit"]["invalid_ohlc_rows"] == 0
    assert result["summary"]["overlapping_rows"] == 11
    assert result["summary"]["alpaca_only_rows"] == 0
    assert result["summary"]["stooq_only_rows"] == 1
    assert result["compatibility_decision"]["decision"] in {
        "FULL_UNIVERSE_COMPATIBILITY_ACCEPTABLE_WITH_CONTROLS",
        "FULL_UNIVERSE_REVIEW_REQUIRED",
    }
    assert result["summary"] == second["summary"]
    assert (report_root / "provider_boundary_policy.json").exists()
    missing = list(csv.DictReader((report_root / "alpaca_missing_sessions.csv").open(encoding="utf-8")))
    assert missing[0]["symbol"] == "BRK-B"
    assert missing[0]["missing_session"] == "2026-01-05"
    assert missing[0]["stooq_has_bar"] == "true"
    assert missing[0]["classification"] == "PROVIDER_OMISSION"


def test_dry_run_writes_nothing_and_source_archives_remain_unchanged(tmp_path):
    alpaca_root, stooq_root, assets, aliases = _fixture_archives(tmp_path)
    source_files = sorted(alpaca_root.glob("symbol=*/year=*/bars.parquet")) + sorted(stooq_root.glob("*.parquet"))
    before = {path: path.stat().st_mtime_ns for path in source_files}
    report_root = tmp_path / "reports"

    result = run_full_universe_reconciliation(
        alpaca_archive_root=alpaca_root,
        stooq_root=stooq_root,
        asset_registry=assets,
        alias_registry=aliases,
        report_root=report_root,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert not report_root.exists()
    assert {path: path.stat().st_mtime_ns for path in source_files} == before
    assert result["source_archives_modified"] is False
    assert result["canonical_market_data_modified"] is False


def test_reader_discovers_all_partitions_and_preserves_brk_aliases(tmp_path):
    alpaca_root, _stooq_root, _assets, _aliases = _fixture_archives(tmp_path)

    rows = read_alpaca_archive(alpaca_root)

    assert rows.file_count == 4
    by_symbol = {row["canonical_symbol"]: row for row in rows.rows}
    assert by_symbol["BRK-A"]["provider_symbol"] == "BRK.A"
    assert by_symbol["BRK-B"]["provider_symbol"] == "BRK.B"


def test_duplicate_and_invalid_ohlc_rows_are_detected(tmp_path):
    alpaca_root, stooq_root, assets, aliases = _fixture_archives(tmp_path, duplicate=True, invalid_ohlc=True)

    result = run_full_universe_reconciliation(
        alpaca_archive_root=alpaca_root,
        stooq_root=stooq_root,
        asset_registry=assets,
        alias_registry=aliases,
        report_root=tmp_path / "reports",
        dry_run=True,
    )

    assert result["audit"]["duplicate_symbol_session_rows"] == 1
    assert result["audit"]["invalid_ohlc_rows"] == 1
    assert result["audit"]["final_structural_validity"] is False


def test_reconciliation_math_and_classifications_have_no_blanks(tmp_path):
    alpaca_root, stooq_root, assets, aliases = _fixture_archives(tmp_path)

    result = run_full_universe_reconciliation(
        alpaca_archive_root=alpaca_root,
        stooq_root=stooq_root,
        asset_registry=assets,
        alias_registry=aliases,
        report_root=tmp_path / "reports",
        dry_run=True,
    )

    rows = result["summary"]["classification_totals"]
    assert sum(rows.values()) == result["summary"]["output_union_rows"]
    assert "" not in rows
    stats = {row["canonical_symbol"]: row for row in result["symbol_stats"]}
    assert stats["AAPL"]["overlap_row_count"] == 3
    assert float(stats["AAPL"]["maximum_absolute_return_difference"]) >= 0.0
    assert float(stats["AAPL"]["maximum_relative_volume_difference"]) > 0.0


def test_manual_classification_covers_large_adjustment_and_volume_cases():
    base = {
        "alpaca_present": "true",
        "stooq_present": "true",
        "close_rel_diff": 0.0,
        "return_abs_diff": 0.0,
        "volume_rel_diff": 0.0,
    }

    assert classify_reconciliation_row(base) == "MATCH"
    assert classify_reconciliation_row({**base, "volume_rel_diff": 0.5}) == "VOLUME_DEFINITION_DIFFERENCE"
    assert classify_reconciliation_row({**base, "close_rel_diff": 0.03}) == "POSSIBLE_ADJUSTMENT_DIFFERENCE"
    assert classify_reconciliation_row({**base, "close_rel_diff": 0.03, "return_abs_diff": 0.03}) == "POSSIBLE_CORPORATE_ACTION"
    assert classify_reconciliation_row({**base, "alpaca_present": "false"}) == "STOOQ_ONLY"
    assert classify_reconciliation_row({**base, "stooq_present": "false"}) == "ALPACA_ONLY"


def _fixture_archives(tmp_path: Path, *, duplicate: bool = False, invalid_ohlc: bool = False):
    alpaca_root = tmp_path / "alpaca"
    stooq_root = tmp_path / "stooq"
    assets = tmp_path / "assets.csv"
    aliases = tmp_path / "aliases.csv"
    _write_registry(assets, aliases)
    day1 = datetime(2026, 1, 2, 4, tzinfo=timezone.utc)
    day2 = datetime(2026, 1, 5, 4, tzinfo=timezone.utc)
    day3 = datetime(2026, 1, 6, 4, tzinfo=timezone.utc)
    _write_alpaca(
        alpaca_root,
        "AAPL",
        "AAPL",
        "asset_aapl",
        [_row("AAPL", "AAPL", "asset_aapl", day1, 100, 1000), _row("AAPL", "AAPL", "asset_aapl", day2, 101, 1100), _row("AAPL", "AAPL", "asset_aapl", day3, 102, 1200)],
    )
    _write_alpaca(
        alpaca_root,
        "SPY",
        "SPY",
        "asset_spy",
        [_row("SPY", "SPY", "asset_spy", day1, 400, 10000), _row("SPY", "SPY", "asset_spy", day2, 401, 11000), _row("SPY", "SPY", "asset_spy", day3, 402, 12000)],
    )
    _write_alpaca(
        alpaca_root,
        "BRK-A",
        "BRK.A",
        "asset_brka",
        [_row("BRK-A", "BRK.A", "asset_brka", day1, 500, 10), _row("BRK-A", "BRK.A", "asset_brka", day2, 501, 12), _row("BRK-A", "BRK.A", "asset_brka", day3, 502, 13)],
    )
    brkb_rows = [_row("BRK-B", "BRK.B", "asset_brkb", day1, 300, 20), _row("BRK-B", "BRK.B", "asset_brkb", day3, 302, 22)]
    if duplicate:
        brkb_rows.append(_row("BRK-B", "BRK.B", "asset_brkb", day1, 300, 20))
    if invalid_ohlc:
        brkb_rows[0]["high"] = 299.0
    _write_alpaca(alpaca_root, "BRK-B", "BRK.B", "asset_brkb", brkb_rows)
    _write_stooq(stooq_root / "AAPL.parquet", "AAPL", [(day1, 100, 800), (day2, 101.5, 900), (day3, 102, 950)])
    _write_stooq(stooq_root / "SPY.parquet", "SPY", [(day1, 400, 9000), (day2, 401, 9500), (day3, 402, 9600)])
    _write_stooq(stooq_root / "BRK-A.parquet", "BRK-A", [(day1, 500, 11), (day2, 501, 13), (day3, 502, 14)])
    _write_stooq(stooq_root / "BRK-B.parquet", "BRK-B", [(day1, 300, 19), (day2, 301, 21), (day3, 302, 23)])
    return alpaca_root, stooq_root, assets, aliases


def _row(symbol, provider_symbol, asset_id, timestamp, close, volume):
    return {
        "asset_id": asset_id,
        "canonical_symbol": symbol,
        "provider_symbol": provider_symbol,
        "session_date": timestamp.date().isoformat(),
        "timestamp_utc": timestamp,
        "open": float(close),
        "high": float(close) + 1.0,
        "low": float(close) - 1.0,
        "close": float(close),
        "volume": float(volume),
        "trade_count": 1,
        "vwap": float(close),
        "provider": "alpaca",
        "feed": "sip",
        "timeframe": "1Day",
        "adjustment_policy": "all",
        "request_chunk_id": "chunk",
        "dataset_version": "test",
    }


def _write_alpaca(root: Path, symbol: str, provider_symbol: str, asset_id: str, rows: list[dict]):
    target = root / f"symbol={symbol}" / "year=2026" / "bars.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target)


def _write_stooq(path: Path, symbol: str, rows: list[tuple[datetime, float, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "symbol": symbol,
                    "timestamp": timestamp.replace(tzinfo=None),
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": volume,
                    "source": "stooq_bulk",
                }
                for timestamp, close, volume in rows
            ]
        ),
        path,
    )


def _write_registry(assets: Path, aliases: Path):
    asset_fields = [
        "asset_id",
        "canonical_symbol",
        "security_name",
        "security_type",
        "share_class",
        "exchange",
        "currency",
        "country",
        "cik",
        "sector",
        "industry",
        "valid_from",
        "valid_to",
        "is_active",
        "collection_universe_514",
        "registry_version",
    ]
    alias_fields = ["asset_id", "provider", "provider_symbol", "valid_from", "valid_to", "is_primary", "mapping_reason", "source", "registry_version"]
    with assets.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=asset_fields, lineterminator="\n")
        writer.writeheader()
        for symbol, asset_id in [("AAPL", "asset_aapl"), ("SPY", "asset_spy"), ("BRK-A", "asset_brka"), ("BRK-B", "asset_brkb")]:
            writer.writerow(
                {
                    "asset_id": asset_id,
                    "canonical_symbol": symbol,
                    "security_type": "COMMON",
                    "currency": "USD",
                    "country": "US",
                    "valid_from": "1900-01-01",
                    "valid_to": "",
                    "is_active": "true",
                    "collection_universe_514": "true",
                    "registry_version": "test",
                }
            )
    with aliases.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=alias_fields, lineterminator="\n")
        writer.writeheader()
        for asset_id, provider_symbol in [("asset_aapl", "AAPL"), ("asset_spy", "SPY"), ("asset_brka", "BRK.A"), ("asset_brkb", "BRK.B")]:
            writer.writerow(
                {
                    "asset_id": asset_id,
                    "provider": "alpaca",
                    "provider_symbol": provider_symbol,
                    "valid_from": "1900-01-01",
                    "valid_to": "",
                    "is_primary": "true",
                    "mapping_reason": "test",
                    "source": "test",
                    "registry_version": "test",
                }
            )
