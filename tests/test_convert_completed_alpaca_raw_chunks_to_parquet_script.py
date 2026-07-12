from __future__ import annotations

import json
from pathlib import Path

from scripts.convert_completed_alpaca_raw_chunks_to_parquet import convert_one, find_candidates, run_conversions


def test_converts_completed_chunk_and_deletes_only_payloads(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "converted"
    assert result["source_row_count"] == 2
    assert result["parquet_row_count"] == 2
    assert not (chunk / "normalized_rows.json").exists()
    assert not (chunk / "provider_pages.json").exists()
    assert (chunk / "manifest.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["validation_result"] == "passed"
    assert Path(tombstone["parquet_path"]).exists()


def test_skips_tmp_in_progress_and_already_converted_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, suffix=".tmp")
    _write_chunk(raw_root, batch="MSFT", completion_state="in_progress")
    converted = _write_chunk(raw_root, batch="AAPL")
    candidate = find_candidates(raw_root, parquet_root)[0]
    assert convert_one(candidate, row_group_size=2)["status"] == "converted"

    candidates = find_candidates(raw_root, parquet_root)

    assert candidates == []
    assert (converted / "manifest.json").exists()


def test_failed_conversion_retains_payloads(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    (chunk / "normalized_rows.json").write_text("[{", encoding="utf-8")
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "failed"
    assert (chunk / "normalized_rows.json").exists()
    assert (chunk / "provider_pages.json").exists()
    assert not (chunk / "parquet_conversion.json").exists()


def test_parallel_runner_converts_distinct_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, batch="AAPL")
    _write_chunk(raw_root, batch="MSFT")
    candidates = find_candidates(raw_root, parquet_root)

    results = run_conversions(candidates, row_group_size=2, workers=2)

    assert sorted(result["status"] for result in results) == ["converted", "converted"]
    assert find_candidates(raw_root, parquet_root) == []


def _write_chunk(
    raw_root: Path,
    *,
    batch: str = "AAPL",
    suffix: str = "",
    completion_state: str = "completed",
) -> Path:
    chunk = raw_root / "sip" / "5m" / batch / f"20260102T143000Z_20260102T150000Z{suffix}"
    chunk.mkdir(parents=True, exist_ok=True)
    rows = [
        _row(batch, "2026-01-02 14:30:00+00:00"),
        _row(batch, "2026-01-02 14:35:00+00:00"),
    ]
    manifest = {
        "provider": "alpaca",
        "feed": "sip",
        "timeframe_requested": "5m",
        "native_timeframe": "5Min",
        "symbol_batch": [batch],
        "requested_start": "2026-01-02T14:30:00+00:00",
        "requested_end": "2026-01-02T15:00:00+00:00",
        "row_count": len(rows),
        "page_count": 1,
        "collection_timestamp": "2026-01-02T15:01:00+00:00",
        "adjustment_mode": "all",
        "session_policy": "regular_session_default",
        "normalizer_version": "historical_bar_provider_v1",
        "completion_state": completion_state,
    }
    (chunk / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (chunk / "normalized_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (chunk / "provider_pages.json").write_text(json.dumps([{"bars": {}}]), encoding="utf-8")
    return chunk


def _row(symbol: str, timestamp: str) -> dict:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 100.0,
        "trade_count": 1,
        "vwap": 10.25,
        "provider": "alpaca",
        "feed": "sip",
        "collection_timestamp": "2026-01-02T15:01:00+00:00",
        "requested_timeframe": "5m",
        "native_timeframe": "5Min",
        "adjustment_mode": "all",
        "extended_hours": False,
        "session_policy": "all_returned_bars_preserved",
        "session_type": "rth",
        "raw_chunk_identifier": f"chunk-{symbol}",
        "normalizer_version": "historical_bar_provider_v1",
    }
