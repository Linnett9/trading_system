from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import infrastructure.data.alpaca_5m_symbol_year_finalizer as finalizer
from infrastructure.data.alpaca_5m_symbol_year_finalizer import (
    DATASET_VERSION,
    OUTPUT_SCHEMA,
    SourceChunk,
    build_intraday_summary_features,
    derive_regular_session_daily,
    deduplicate_rows,
    finalize_symbol_year_archive,
    normalize_output_row,
    production_preflight,
    select_production_chunks,
    validate_final_archive,
    validate_rows,
)


def test_production_chunks_included_and_pilot_excluded(tmp_path: Path) -> None:
    env = _env(tmp_path)
    production = _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    pilot = _write_chunk(env, "SPY", "20260102T143000Z", "20260102T150000Z", include_in_manifest=False)

    selection = select_production_chunks(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        collection_manifest_path=env["manifest"],
        symbols=set(),
        years=set(),
        max_chunks=0,
    )

    assert [chunk.raw_path for chunk in selection["chunks"]] == [production]
    assert selection["report"]["excluded_by_reason"]["not_in_production_collection_manifest"] == 1
    assert str(pilot) in selection["report"]["excluded_sample"][0]["source_path"]


def test_brk_symbol_mapping_and_output_provenance(tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunk_dir = _write_chunk(env, "BRK-B", "20260102T143000Z", "20260102T150000Z", provider_symbol="BRK.B")
    chunk = _source_chunk(env, chunk_dir)
    row = pq.read_table(chunk.parquet_path).to_pylist()[0]

    output = normalize_output_row(row, chunk=chunk, registry={"BRK-B": {"asset_id": "asset-brkb"}})

    assert output["canonical_symbol"] == "BRK-B"
    assert output["provider_symbol"] == "BRK.B"
    assert output["asset_id"] == "asset-brkb"


def test_unknown_symbol_quarantine_is_explicit(tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunk_dir = _write_chunk(env, "ZZZZ", "20260102T143000Z", "20260102T150000Z")
    chunk = _source_chunk(env, chunk_dir)
    row = pq.read_table(chunk.parquet_path).to_pylist()[0]

    with pytest.raises(ValueError, match="unknown canonical symbol"):
        normalize_output_row(row, chunk=chunk, registry={})


def test_timestamp_normalisation_and_session_classification(tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunk_dir = _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    chunk = _source_chunk(env, chunk_dir)
    row = pq.read_table(chunk.parquet_path).to_pylist()[0]
    row["session_type"] = None

    output = normalize_output_row(row, chunk=chunk, registry={"AAPL": {"asset_id": "asset-aapl"}})

    assert output["timestamp_utc"].tzinfo is not None
    assert output["session_type"] == "rth"
    assert output["timeframe"] == "5Min"
    assert output["feed"] == "SIP"


def test_exact_duplicate_removed_and_conflict_blocks() -> None:
    base = _output_row("AAPL", datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc), raw_chunk_id="a")
    exact = dict(base, raw_chunk_id="b")
    conflict = dict(base, close=99.0, raw_chunk_id="c")

    deduped, report = deduplicate_rows([base, exact])

    assert len(deduped) == 1
    assert report["exact_duplicates_removed"] == 1
    _, conflict_report = deduplicate_rows([base, conflict])
    assert conflict_report["conflicting_duplicate_count"] == 1


def test_ohlcv_validation() -> None:
    row = _output_row("AAPL", datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc))
    bad = dict(row, high=0.5)

    assert validate_rows([row]) == []
    assert validate_rows([bad])[0]["reason"] == "invalid_high"


def test_symbol_year_partition_routing_atomic_manifest_and_skip(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")

    result = finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        dry_run=False,
    )

    path = env["archive"] / "symbol=AAPL" / "year=2026" / "bars.parquet"
    manifest = json.loads((env["report"] / "partition_manifests" / "AAPL_2026.json").read_text())
    assert result["completed_partitions"] == 1
    assert path.exists()
    assert not path.with_suffix(".parquet.tmp").exists()
    assert manifest["output_row_count"] == 2
    second = finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        dry_run=False,
    )
    assert second["completed_partitions"] == 1


def test_retry_only_failed_processes_failed_partition(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    failure_root = env["report"] / "partition_failures"
    failure_root.mkdir(parents=True)
    (failure_root / "AAPL_2026.json").write_text(json.dumps({"canonical_symbol": "AAPL", "year": 2026}))

    result = finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        dry_run=False,
        retry_only_failed=True,
    )

    assert result["completed_partitions"] == 1


def test_manifest_row_count_reconciliation_and_no_raw_mutation(tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunk = _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    before = (chunk / "manifest.json").read_text()

    finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        dry_run=False,
    )

    manifest = json.loads((env["report"] / "partition_manifests" / "AAPL_2026.json").read_text())
    assert manifest["source_row_count"] == 2
    assert manifest["output_row_count"] == 2
    assert (chunk / "manifest.json").read_text() == before


def test_preflight_blocks_missing_conversion_and_reports_bounds(tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunk = _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    (env["parquet"] / chunk.relative_to(env["raw"]) / "bars.parquet").unlink()

    with pytest.raises(ValueError, match="missing_converted"):
        production_preflight(
            raw_root=env["raw"],
            parquet_root=env["parquet"],
            archive_root=env["archive"],
            report_root=env["report"],
            collection_manifest_path=env["manifest"],
            universe_path=env["universe"],
            symbols=["AAPL"],
            years=[2026],
            source_chunk_paths=[chunk],
        )


def test_progress_and_failure_payload_persist_immediately(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write_chunk(env, "ZZZZ", "20260102T143000Z", "20260102T150000Z")

    with pytest.raises(RuntimeError):
        finalize_symbol_year_archive(
            raw_root=env["raw"],
            parquet_root=env["parquet"],
            archive_root=env["archive"],
            report_root=env["report"],
            collection_manifest_path=env["manifest"],
            universe_path=env["universe"],
            symbols=["ZZZZ"],
            years=[2026],
            dry_run=False,
            fail_fast_same_signature_threshold=1,
        )

    failure = json.loads(next((env["report"] / "partition_failures").glob("*.json")).read_text())
    progress = json.loads((env["report"] / "progress_manifest.json").read_text())
    assert failure["failure_phase"] == "partition_finalisation"
    assert failure["normalised_failure_signature"]
    assert failure["source_chunk_identities"]
    assert progress["aborted_early"] is True


def test_conflicting_duplicate_quarantine_blocks_publication(tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunk = _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    parquet = env["parquet"] / chunk.relative_to(env["raw"]) / "bars.parquet"
    rows = pq.read_table(parquet).to_pylist()
    rows.append({**rows[0], "close": 9.9, "raw_chunk_identifier": "conflict"})
    pq.write_table(pa.Table.from_pylist(rows), parquet)

    with pytest.raises(RuntimeError):
        finalize_symbol_year_archive(
            raw_root=env["raw"],
            parquet_root=env["parquet"],
            archive_root=env["archive"],
            report_root=env["report"],
            collection_manifest_path=env["manifest"],
            universe_path=env["universe"],
            symbols=["AAPL"],
            years=[2026],
            dry_run=False,
            conflict_root=env["report"] / "conflicts",
        )

    assert not (env["archive"] / "symbol=AAPL" / "year=2026" / "bars.parquet").exists()
    conflict_files = list((env["report"] / "conflicts").glob("AAPL_2026_conflicts.json"))
    assert conflict_files


def test_final_archive_validation_reports_valid_smoke_partition(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        dry_run=False,
    )

    report = validate_final_archive(env["archive"], env["report"])

    assert report["valid"] is True
    assert report["partition_count"] == 1
    assert report["total_rows"] == 2


def test_daily_regular_session_aggregation_and_incomplete_flag() -> None:
    rows = [
        _output_row("AAPL", datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)),
        {**_output_row("AAPL", datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)), "close": 2.0, "high": 2.2, "volume": 50.0},
        {**_output_row("AAPL", datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)), "session_type": "pre_market"},
    ]

    daily = derive_regular_session_daily(rows, expected_regular_session_bar_count=78)

    assert len(daily) == 1
    assert daily[0]["open"] == 1.0
    assert daily[0]["close"] == 2.0
    assert daily[0]["regular_session_bar_count"] == 2
    assert daily[0]["session_completeness_flag"] == "incomplete"


def test_intraday_feature_formulas_and_pit_controls() -> None:
    stock = [
        _output_row("AAPL", datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)),
        {**_output_row("AAPL", datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)), "close": 2.0, "high": 2.0, "volume": 200.0},
    ]
    spy = [
        _output_row("SPY", datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)),
        {**_output_row("SPY", datetime(2026, 1, 2, 14, 35, tzinfo=timezone.utc)), "close": 1.6},
    ]

    features = build_intraday_summary_features(stock, spy_rows=spy)

    assert features[0]["intraday_return"] == pytest.approx(1.0)
    assert features[0]["stock_minus_spy_intraday_return"] == pytest.approx(0.4)
    assert features[0]["feature_available_timestamp"] == "2026-01-02T21:00:00+00:00"
    assert features[0]["same_session_preopen_safe"] is False


def test_smoke_configs_are_bounded_by_explicit_source_chunks() -> None:
    import yaml

    expected = {
        "config.alpaca_5m_symbol_year_finalizer_smoke_1.yaml": (["AAPL"], [2026], 1),
        "config.alpaca_5m_symbol_year_finalizer_smoke_3.yaml": (["AAPL", "SPY", "BRK-B"], [2026], 3),
        "config.alpaca_5m_symbol_year_finalizer_smoke_20.yaml": (
            ["AAPL", "SPY", "BRK-B", "BBW", "GE", "AMD", "TSLA", "WMT", "XOM", "V"],
            [2025, 2026],
            20,
        ),
    }
    for filename, (symbols, years, chunk_count) in expected.items():
        settings = yaml.safe_load((Path("config") / filename).read_text(encoding="utf-8"))["ml"]["alpaca_5m_symbol_year_finalizer"]

        assert settings["symbols"] == symbols
        assert settings["years"] == years
        assert settings["max_chunks"] == 0
        assert len(settings["source_chunk_paths"]) == chunk_count
        assert all(str(path).startswith("data/raw/alpaca/stock_bars/sip/5m/") for path in settings["source_chunk_paths"])


def test_multi_worker_matches_one_worker_outputs(tmp_path: Path) -> None:
    one = _env(tmp_path / "one")
    multi = _env(tmp_path / "multi")
    for env in (one, multi):
        _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
        _write_chunk(env, "SPY", "20260102T143000Z", "20260102T150000Z")

    one_result = finalize_symbol_year_archive(
        raw_root=one["raw"],
        parquet_root=one["parquet"],
        archive_root=one["archive"],
        report_root=one["report"],
        collection_manifest_path=one["manifest"],
        universe_path=one["universe"],
        symbols=["AAPL", "SPY"],
        years=[2026],
        workers=1,
        dry_run=False,
    )
    multi_result = finalize_symbol_year_archive(
        raw_root=multi["raw"],
        parquet_root=multi["parquet"],
        archive_root=multi["archive"],
        report_root=multi["report"],
        collection_manifest_path=multi["manifest"],
        universe_path=multi["universe"],
        symbols=["AAPL", "SPY"],
        years=[2026],
        workers=2,
        dry_run=False,
    )

    assert one_result["effective_workers"] == 1
    assert multi_result["effective_workers"] == 2
    assert multi_result["partition_processing_backend"] == "process_pool"
    for symbol in ("AAPL", "SPY"):
        one_path = one["archive"] / f"symbol={symbol}" / "year=2026" / "bars.parquet"
        multi_path = multi["archive"] / f"symbol={symbol}" / "year=2026" / "bars.parquet"
        assert pq.read_table(one_path).schema == pq.read_table(multi_path).schema
        assert pq.read_table(one_path).to_pylist() == pq.read_table(multi_path).to_pylist()
        one_manifest = json.loads((one["report"] / "partition_manifests" / f"{symbol}_2026.json").read_text())
        multi_manifest = json.loads((multi["report"] / "partition_manifests" / f"{symbol}_2026.json").read_text())
        assert one_manifest["output_file_hash"] == multi_manifest["output_file_hash"]


def test_multi_worker_skips_completed_partitions(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        workers=1,
        dry_run=False,
    )

    rerun = finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL"],
        years=[2026],
        workers=4,
        dry_run=False,
    )

    assert rerun["completed_partitions"] == 1
    assert rerun["partitions_dispatched"] == 0
    assert rerun["partition_keys_dispatched_unique"] == 0


def test_multi_worker_persists_failures_and_retry_only_failed(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    _write_chunk(env, "ZZZZ", "20260102T143000Z", "20260102T150000Z")

    with pytest.raises(RuntimeError):
        finalize_symbol_year_archive(
            raw_root=env["raw"],
            parquet_root=env["parquet"],
            archive_root=env["archive"],
            report_root=env["report"],
            collection_manifest_path=env["manifest"],
            universe_path=env["universe"],
            symbols=["AAPL", "ZZZZ"],
            years=[2026],
            workers=2,
            dry_run=False,
        )

    assert (env["archive"] / "symbol=AAPL" / "year=2026" / "bars.parquet").exists()
    failure = json.loads((env["report"] / "partition_failures" / "ZZZZ_2026.json").read_text())
    assert failure["normalised_failure_signature"].startswith("ValueError: unknown canonical symbol")
    env["universe"].write_text("AAPL\nSPY\nBRK-B\nZZZZ\n", encoding="utf-8")

    retry = finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL", "ZZZZ"],
        years=[2026],
        workers=2,
        retry_only_failed=True,
        dry_run=False,
    )

    assert retry["completed_partitions"] == 1
    assert retry["partitions_dispatched"] == 1
    assert (env["archive"] / "symbol=ZZZZ" / "year=2026" / "bars.parquet").exists()


def test_multi_worker_conflict_does_not_corrupt_other_partition(tmp_path: Path) -> None:
    env = _env(tmp_path)
    aapl = _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z")
    _write_chunk(env, "SPY", "20260102T143000Z", "20260102T150000Z")
    parquet = env["parquet"] / aapl.relative_to(env["raw"]) / "bars.parquet"
    rows = pq.read_table(parquet).to_pylist()
    rows.append({**rows[0], "close": 9.9, "raw_chunk_identifier": "conflict"})
    pq.write_table(pa.Table.from_pylist(rows), parquet)

    with pytest.raises(RuntimeError):
        finalize_symbol_year_archive(
            raw_root=env["raw"],
            parquet_root=env["parquet"],
            archive_root=env["archive"],
            report_root=env["report"],
            collection_manifest_path=env["manifest"],
            universe_path=env["universe"],
            symbols=["AAPL", "SPY"],
            years=[2026],
            workers=2,
            dry_run=False,
            conflict_root=env["report"] / "conflicts",
        )

    assert not (env["archive"] / "symbol=AAPL" / "year=2026" / "bars.parquet").exists()
    assert (env["archive"] / "symbol=SPY" / "year=2026" / "bars.parquet").exists()
    assert (env["report"] / "conflicts" / "AAPL_2026_conflicts.json").exists()
    assert not (env["archive"] / "symbol=AAPL" / "year=2026" / "bars.parquet.tmp").exists()


def test_multi_worker_source_discovery_occurs_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(tmp_path)
    chunks = [
        _write_chunk(env, "AAPL", "20260102T143000Z", "20260102T150000Z"),
        _write_chunk(env, "SPY", "20260102T143000Z", "20260102T150000Z"),
    ]
    calls = []

    def fake_raw_chunk_dirs(raw_root: Path):
        calls.append(raw_root)
        return list(chunks)

    monkeypatch.setattr(finalizer, "_raw_chunk_dirs", fake_raw_chunk_dirs)

    result = finalize_symbol_year_archive(
        raw_root=env["raw"],
        parquet_root=env["parquet"],
        archive_root=env["archive"],
        report_root=env["report"],
        collection_manifest_path=env["manifest"],
        universe_path=env["universe"],
        symbols=["AAPL", "SPY"],
        years=[2026],
        workers=2,
        dry_run=False,
    )

    assert len(calls) == 1
    assert result["source_discovery_passes"] == 1
    assert result["partitions_dispatched"] == 2


def _env(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = {
        "raw": tmp_path / "raw",
        "parquet": tmp_path / "parquet",
        "archive": tmp_path / "archive",
        "report": tmp_path / "report",
        "manifest": tmp_path / "collection_manifest.json",
        "universe": tmp_path / "universe.txt",
    }
    root["universe"].write_text("AAPL\nSPY\nBRK-B\n", encoding="utf-8")
    root["manifest"].write_text(json.dumps({"chunks": {}}), encoding="utf-8")
    return root


def _write_chunk(
    env: dict[str, Path],
    symbol: str,
    start: str,
    end: str,
    *,
    include_in_manifest: bool = True,
    provider_symbol: str | None = None,
) -> Path:
    chunk = env["raw"] / "sip" / "5m" / symbol / f"{start}_{end}"
    chunk.mkdir(parents=True, exist_ok=True)
    manifest = {
        "provider": "alpaca",
        "feed": "sip",
        "timeframe_requested": "5m",
        "native_timeframe": "5Min",
        "symbol_batch": [symbol],
        "canonical_symbol_batch": [symbol],
        "provider_symbol_map": {"BRK-B": "BRK.B"} if symbol == "BRK-B" else {},
        "requested_start": _iso(start),
        "requested_end": _iso(end),
        "row_count": 2,
        "adjustment_mode": "all",
        "completion_state": "completed",
    }
    (chunk / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        _chunk_row(symbol, _iso(start), provider_symbol=provider_symbol),
        _chunk_row(symbol, "2026-01-02T14:35:00+00:00", provider_symbol=provider_symbol),
    ]
    parquet = env["parquet"] / chunk.relative_to(env["raw"]) / "bars.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    if include_in_manifest:
        payload = json.loads(env["manifest"].read_text())
        from scripts.convert_completed_alpaca_raw_chunks_to_parquet import chunk_id_from_manifest

        payload["chunks"][chunk_id_from_manifest(manifest)] = {"status": "completed"}
        env["manifest"].write_text(json.dumps(payload), encoding="utf-8")
    return chunk


def _source_chunk(env: dict[str, Path], chunk: Path) -> SourceChunk:
    manifest = json.loads((chunk / "manifest.json").read_text())
    from scripts.convert_completed_alpaca_raw_chunks_to_parquet import chunk_id_from_manifest, parquet_path

    return SourceChunk(
        chunk_id=chunk_id_from_manifest(manifest),
        raw_path=chunk,
        parquet_path=parquet_path(env["parquet"], env["raw"], chunk),
        manifest=manifest,
        reason="test",
    )


def _chunk_row(symbol: str, timestamp: str, *, provider_symbol: str | None = None) -> dict[str, object]:
    return {
        "symbol": symbol,
        "provider_symbol": provider_symbol or symbol,
        "timestamp": datetime.fromisoformat(timestamp),
        "open": 1.0,
        "high": 2.0,
        "low": 1.0,
        "close": 1.5,
        "volume": 100.0,
        "trade_count": 10,
        "vwap": 1.4,
        "provider": "alpaca",
        "feed": "sip",
        "requested_timeframe": "5m",
        "native_timeframe": "5Min",
        "adjustment_mode": "all",
        "session_type": "rth",
        "raw_chunk_identifier": f"chunk-{symbol}",
    }


def _output_row(symbol: str, timestamp: datetime, *, raw_chunk_id: str = "chunk") -> dict[str, object]:
    return {
        "asset_id": f"asset-{symbol}",
        "canonical_symbol": symbol,
        "provider_symbol": symbol,
        "timestamp_utc": timestamp,
        "session_date": timestamp.date().isoformat(),
        "session_type": "rth",
        "open": 1.0,
        "high": 2.0,
        "low": 1.0,
        "close": 1.5,
        "volume": 100.0,
        "trade_count": 10,
        "vwap": 1.4,
        "provider": "alpaca",
        "feed": "SIP",
        "timeframe": "5Min",
        "adjustment_policy": "all",
        "raw_chunk_id": raw_chunk_id,
        "source_row_hash": "hash",
        "dataset_version": DATASET_VERSION,
    }


def _iso(value: str) -> str:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
