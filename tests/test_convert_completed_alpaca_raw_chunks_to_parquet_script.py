from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from infrastructure.data.historical_bar_providers import CollectionManifest
from scripts.convert_completed_alpaca_raw_chunks_to_parquet import convert_one, find_candidates, run_conversions, scan_candidates


def test_converts_completed_chunk_and_preserves_payloads_by_default(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "converted"
    assert result["source_row_count"] == 2
    assert result["parquet_row_count"] == 2
    assert (chunk / "normalized_rows.json").exists()
    assert (chunk / "provider_pages.json").exists()
    assert (chunk / "manifest.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["validation_result"] == "passed"
    assert tombstone["json_payloads_preserved"] is True
    assert tombstone["source_bytes_deleted"] == 0
    assert Path(tombstone["parquet_path"]).exists()
    table = pq.read_table(tombstone["parquet_path"])
    assert "source_raw_chunk_path" in table.schema.names
    assert "conversion_timestamp" in table.schema.names


def test_delete_json_requires_explicit_dangerous_flag(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(raw_root)
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2, delete_json_after_validate=True)

    assert result["status"] == "converted"
    assert not (chunk / "normalized_rows.json").exists()
    assert not (chunk / "provider_pages.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["json_payloads_preserved"] is False
    assert tombstone["deleted_payload_files"]


def test_skips_tmp_in_progress_and_already_converted_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, suffix=".tmp")
    _write_chunk(raw_root, batch="MSFT", completion_state="in_progress")
    converted = _write_chunk(raw_root, batch="AAPL")
    candidate = find_candidates(raw_root, parquet_root)[0]
    assert convert_one(candidate, row_group_size=2)["status"] == "converted"

    candidates = find_candidates(raw_root, parquet_root)
    scan = scan_candidates(raw_root, parquet_root)

    assert candidates == []
    assert scan["skipped_existing_count"] == 1
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


def test_dry_run_candidate_selection_uses_completed_json_chunks_only(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    _write_chunk(raw_root, batch="BRK.A")
    _write_chunk(raw_root, batch="AAPL", completion_state="in_progress")
    converted = _write_chunk(raw_root, batch="MSFT")
    msft = [candidate for candidate in find_candidates(raw_root, parquet_root) if "MSFT" in candidate["source_path"]][0]
    assert convert_one(msft, row_group_size=2)["status"] == "converted"

    candidates = find_candidates(raw_root, parquet_root)

    assert [Path(candidate["source_path"]).parent.name for candidate in candidates] == ["BRK.A"]
    assert (converted / "parquet_conversion.json").exists()


def test_recovered_brk_chunk_conversion_records_canonical_and_provider_provenance(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    chunk = _write_chunk(
        raw_root,
        batch="BOOM-BP-BRK.A-BRK.B",
        canonical_symbols=["BOOM", "BP", "BRK-A", "BRK-B"],
        provider_symbol_map={"BRK-A": "BRK.A", "BRK-B": "BRK.B"},
        row_symbol="BRK-B",
        provider_symbol="BRK.B",
    )
    candidate = find_candidates(raw_root, parquet_root)[0]

    result = convert_one(candidate, row_group_size=2)

    assert result["status"] == "converted"
    assert (chunk / "normalized_rows.json").exists()
    tombstone = json.loads((chunk / "parquet_conversion.json").read_text(encoding="utf-8"))
    assert tombstone["canonical_symbol_batch"] == ["BOOM", "BP", "BRK-A", "BRK-B"]
    assert tombstone["provider_symbol_batch"] == ["BOOM", "BP", "BRK.A", "BRK.B"]
    assert tombstone["provider_symbol_map"] == {"BRK-A": "BRK.A", "BRK-B": "BRK.B"}
    table = pq.read_table(tombstone["parquet_path"])
    assert table.column("symbol").to_pylist() == ["BRK-B", "BRK-B"]
    assert table.column("provider_symbol").to_pylist() == ["BRK.B", "BRK.B"]


def test_collection_manifest_filter_excludes_pilot_chunks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    parquet_root = tmp_path / "parquet"
    production = _write_chunk(raw_root, batch="BRK.A", canonical_symbols=["BRK-A"])
    _write_chunk(raw_root, batch="SPY-AAPL-MSFT")
    manifest_path = tmp_path / "collection_manifest.json"
    manifest = CollectionManifest(manifest_path)
    manifest.update(
        "alpaca-sip-5m-BRK-A-20260102T143000Z-20260102T150000Z",
        "completed",
        {"rows": 2},
    )

    candidates = find_candidates(raw_root, parquet_root, collection_manifest_path=manifest_path)

    assert [Path(candidate["source_path"]) for candidate in candidates] == [production]


def _write_chunk(
    raw_root: Path,
    *,
    batch: str = "AAPL",
    suffix: str = "",
    completion_state: str = "completed",
    canonical_symbols: list[str] | None = None,
    provider_symbol_map: dict[str, str] | None = None,
    row_symbol: str | None = None,
    provider_symbol: str | None = None,
) -> Path:
    chunk = raw_root / "sip" / "5m" / batch / f"20260102T143000Z_20260102T150000Z{suffix}"
    chunk.mkdir(parents=True, exist_ok=True)
    row_symbol = row_symbol or batch
    rows = [
        _row(row_symbol, "2026-01-02 14:30:00+00:00", provider_symbol=provider_symbol),
        _row(row_symbol, "2026-01-02 14:35:00+00:00", provider_symbol=provider_symbol),
    ]
    manifest = {
        "provider": "alpaca",
        "feed": "sip",
        "timeframe_requested": "5m",
        "native_timeframe": "5Min",
        "symbol_batch": batch.split("-"),
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
    if canonical_symbols is not None:
        manifest["canonical_symbol_batch"] = canonical_symbols
    if provider_symbol_map is not None:
        manifest["provider_symbol_map"] = provider_symbol_map
    (chunk / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (chunk / "normalized_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (chunk / "provider_pages.json").write_text(json.dumps([{"bars": {}}]), encoding="utf-8")
    return chunk


def _row(symbol: str, timestamp: str, *, provider_symbol: str | None = None) -> dict:
    row = {
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
    if provider_symbol:
        row["provider_symbol"] = provider_symbol
    return row
