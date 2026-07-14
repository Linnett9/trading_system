from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from infrastructure.data.alpaca_daily_missing_session_repair import (
    MISSING_SESSION,
    inspect_missing_session_root_cause,
    merge_may27_archive,
    post_repair_archive_audit,
    validate_may27_rows,
)


def test_shared_missing_session_and_chunk_boundary_detection(tmp_path):
    raw_root = tmp_path / "raw"
    chunk = raw_root / "batch" / "20260427T000000Z_20260527T000000Z"
    chunk.mkdir(parents=True)
    (chunk / "manifest.json").write_text(
        json.dumps({"requested_start": "2026-04-27T00:00:00+00:00", "requested_end": "2026-05-27T00:00:00+00:00", "row_count": 2}),
        encoding="utf-8",
    )
    (chunk / "normalized_rows.json").write_text(
        json.dumps([{"timestamp": "2026-05-26T04:00:00+00:00"}, {"timestamp": "2026-05-23T04:00:00+00:00"}]),
        encoding="utf-8",
    )
    archive = tmp_path / "archive"
    _write_partition(archive, "AAPL", [_row("AAPL", "2026-05-26")])

    report = inspect_missing_session_root_cause(raw_root=raw_root, archive_root=archive, report_root=tmp_path / "reports", dry_run=True)

    assert report["failure_phase"] == "chunk start/end boundary semantics"
    assert report["may27_exists_in_raw"] is False
    assert report["raw_may27_rows"] == 0
    assert report["archive_may27_rows"] == 0


def test_validate_may27_rows_and_post_repair_count(tmp_path):
    archive = tmp_path / "repair"
    _write_partition(archive, "AAPL", [_row("AAPL", MISSING_SESSION)])
    _write_partition(archive, "SPY", [_row("SPY", MISSING_SESSION)])

    validation = validate_may27_rows(archive, expected_rows=2)

    assert validation["valid"] is True
    assert validation["row_count"] == 2
    assert validation["duplicate_symbol_session_rows"] == 0


def test_atomic_merge_allows_idempotent_duplicates_and_rejects_conflicts(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_partition(source, "AAPL", [_row("AAPL", MISSING_SESSION, close=101)])
    _write_partition(target, "AAPL", [_row("AAPL", "2026-05-26", close=100)])

    report = merge_may27_archive(source_archive_root=source, target_archive_root=target, report_root=tmp_path / "reports", expected_rows=1)
    audit = post_repair_archive_audit(target, report_root=tmp_path / "reports", dry_run=True)

    assert report["merged_row_count"] == 1
    assert audit["row_count"] == 2
    assert audit["may27_rows"] == 1
    second = merge_may27_archive(source_archive_root=source, target_archive_root=target, report_root=tmp_path / "reports", expected_rows=1)
    assert second["identical_duplicate_rows"] == 1
    _write_partition(source, "AAPL", [_row("AAPL", MISSING_SESSION, close=102)])
    with pytest.raises(ValueError, match="conflicting May 27"):
        merge_may27_archive(source_archive_root=source, target_archive_root=target, report_root=tmp_path / "reports", expected_rows=1)


def test_dry_run_writes_nothing(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    report_root = tmp_path / "reports"
    _write_partition(source, "AAPL", [_row("AAPL", MISSING_SESSION, close=101)])
    _write_partition(target, "AAPL", [_row("AAPL", "2026-05-26", close=100)])

    merge_may27_archive(source_archive_root=source, target_archive_root=target, report_root=report_root, expected_rows=1, dry_run=True)

    assert not report_root.exists()
    assert post_repair_archive_audit(target, report_root=report_root, dry_run=True)["row_count"] == 1


def _row(symbol: str, session: str, close: float = 100.0):
    return {
        "asset_id": f"asset_{symbol}",
        "canonical_symbol": symbol,
        "provider_symbol": symbol,
        "session_date": session,
        "timestamp_utc": datetime.fromisoformat(session).replace(tzinfo=timezone.utc),
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
        "trade_count": 10,
        "vwap": close,
        "provider": "alpaca",
        "feed": "sip",
        "timeframe": "1Day",
        "adjustment_policy": "all",
        "request_chunk_id": "repair",
        "dataset_version": "test",
    }


def _write_partition(root: Path, symbol: str, rows: list[dict]):
    target = root / f"symbol={symbol}" / "year=2026" / "bars.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), target)
