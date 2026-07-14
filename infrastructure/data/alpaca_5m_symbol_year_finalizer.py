from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import time
import tracemalloc
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from core.research.ml.reference.canonical_assets import (
    alpaca_provider_symbol,
    build_registry_from_universe,
    canonical_asset_id,
    normalize_symbol,
)
from infrastructure.data.market_sessions import session_type
from scripts.convert_completed_alpaca_raw_chunks_to_parquet import chunk_id_from_manifest, parquet_path


DEFAULT_RAW_ROOT = Path("data/raw/alpaca/stock_bars")
DEFAULT_PARQUET_ROOT = Path("data/processed/alpaca/stock_bars_parquet")
DEFAULT_ARCHIVE_ROOT = Path("data/processed/alpaca/symbol_bars/sip/5m")
DEFAULT_REPORT_ROOT = Path("reports/data_quality/alpaca_5m_symbol_year_finalisation")
DEFAULT_COLLECTION_MANIFEST = Path("reports/market_data/historical_bar_backfill/5m_sip_514_symbol_full/collection_manifest.json")
DATASET_VERSION = "alpaca_sip_5m_symbol_year_v1"

OUTPUT_SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("canonical_symbol", pa.string()),
        ("provider_symbol", pa.string()),
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("session_date", pa.string()),
        ("session_type", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("vwap", pa.float64()),
        ("provider", pa.string()),
        ("feed", pa.string()),
        ("timeframe", pa.string()),
        ("adjustment_policy", pa.string()),
        ("raw_chunk_id", pa.string()),
        ("source_row_hash", pa.string()),
        ("dataset_version", pa.string()),
    ]
)


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    raw_path: Path
    parquet_path: Path
    manifest: Mapping[str, Any]
    reason: str


def run_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict((config.get("ml", {}) or {}).get("alpaca_5m_symbol_year_finalizer", {}) or {})
    return finalize_symbol_year_archive(
        raw_root=Path(str(settings.get("raw_root", DEFAULT_RAW_ROOT))),
        parquet_root=Path(str(settings.get("parquet_root", DEFAULT_PARQUET_ROOT))),
        archive_root=Path(str(settings.get("archive_root", DEFAULT_ARCHIVE_ROOT))),
        report_root=Path(str(settings.get("report_root", DEFAULT_REPORT_ROOT))),
        collection_manifest_path=Path(str(settings.get("collection_manifest_path", DEFAULT_COLLECTION_MANIFEST))),
        universe_path=Path(str(settings.get("universe_path", "config/universes/alpaca_514_symbols.txt"))),
        symbols=[str(symbol).upper() for symbol in settings.get("symbols", [])],
        years=[int(year) for year in settings.get("years", [])],
        max_chunks=int(settings.get("max_chunks", 0) or 0),
        max_chunks_per_symbol_year=int(settings.get("max_chunks_per_symbol_year", 0) or 0),
        dry_run=bool(settings.get("dry_run", True)),
        retry_only_failed=bool(settings.get("retry_only_failed", False)),
        workers=int(settings.get("workers", 1) or 1),
        source_chunk_paths=[Path(str(path)) for path in settings.get("source_chunk_paths", [])],
        fail_fast_same_signature_threshold=int(settings.get("fail_fast_same_signature_threshold", 3) or 3),
        conflict_root=Path(str(settings.get("conflict_root", "reports/data_quality/alpaca_5m_symbol_year_conflicts"))),
        max_in_flight_tasks=int(settings.get("max_in_flight_tasks", 0) or 0),
    )


def finalize_symbol_year_archive(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    collection_manifest_path: Path = DEFAULT_COLLECTION_MANIFEST,
    universe_path: Path = Path("config/universes/alpaca_514_symbols.txt"),
    symbols: Sequence[str] = (),
    years: Sequence[int] = (),
    max_chunks: int = 0,
    max_chunks_per_symbol_year: int = 0,
    dry_run: bool = True,
    retry_only_failed: bool = False,
    workers: int = 1,
    source_chunk_paths: Sequence[Path] = (),
    fail_fast_same_signature_threshold: int = 3,
    conflict_root: Path = Path("reports/data_quality/alpaca_5m_symbol_year_conflicts"),
    max_in_flight_tasks: int = 0,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_in_flight_tasks < 0:
        raise ValueError("max_in_flight_tasks must be >= 0")
    report_root.mkdir(parents=True, exist_ok=True)
    selected_symbols = {normalize_symbol(symbol) for symbol in symbols if symbol}
    selected_years = {int(year) for year in years}
    registry = _registry(universe_path)
    selection = select_production_chunks(
        raw_root=raw_root,
        parquet_root=parquet_root,
        collection_manifest_path=collection_manifest_path,
        symbols=selected_symbols,
        years=selected_years,
        max_chunks=max_chunks,
        max_chunks_per_symbol_year=max_chunks_per_symbol_year,
        source_chunk_paths=source_chunk_paths,
    )
    _write_json(report_root / "source_chunk_plan.json", selection["report"])
    if dry_run:
        payload = {
            "mode": "alpaca_5m_symbol_year_finalizer",
            "dry_run": True,
            "archive_root": str(archive_root),
            "report_root": str(report_root),
            "candidate_chunks": selection["report"]["candidate_chunks"],
            "production_chunks": selection["report"]["production_chunks"],
            "excluded_chunks": selection["report"]["excluded_chunks"],
            "planned_partitions": _planned_partition_count(selection["chunks"]),
            "source_discovery_passes": 1,
            "protected_source_archives_modified": False,
        }
        _write_json(report_root / "finalisation_plan.json", payload)
        return payload

    archive_root.mkdir(parents=True, exist_ok=True)
    manifest_root = report_root / "partition_manifests"
    failure_root = report_root / "partition_failures"
    quarantine_root = conflict_root
    manifest_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    failed_before = _failed_partition_keys(failure_root) if retry_only_failed else None
    all_groups = _group_chunks_by_symbol_year(selection["chunks"], selected_symbols, selected_years)
    groups = {key: all_groups[key] for key in sorted(all_groups) if failed_before is None or key in failed_before}
    progress = _progress_payload(len(groups), started=time.perf_counter())
    _write_json(report_root / "progress_manifest.json", progress)
    completed = []
    failed = []
    started = time.perf_counter()
    failure_signatures: Counter[str] = Counter()
    rows_read = rows_written = exact_duplicates = conflicts = invalid_rows = 0
    tasks: list[tuple[tuple[str, int], list[SourceChunk]]] = []
    for key, chunks in sorted(groups.items()):
        symbol, year = key
        manifest_path = manifest_root / f"{symbol}_{year}.json"
        if failed_before is None and _completed_manifest_valid(manifest_path, archive_root):
            completed.append(_read_json(manifest_path))
            continue
        tasks.append((key, list(chunks)))
    effective_workers = min(workers, len(tasks)) if tasks else 0
    in_flight_limit = max_in_flight_tasks or max(1, effective_workers * 2)
    aborted_early = False
    abort_reason = ""
    current_partition = ""
    if effective_workers <= 1:
        for key, chunks in tasks:
            current_partition = f"{key[0]}_{key[1]}"
            _write_json(report_root / "progress_manifest.json", _progress_payload(
                len(groups), completed=len(completed), failed=len(failed), rows_read=rows_read,
                rows_written=rows_written, exact_duplicates=exact_duplicates, conflicts=conflicts,
                invalid_rows=invalid_rows, current_partition=current_partition, started=started,
                active_workers=1 if tasks else 0, configured_workers=workers, effective_workers=effective_workers,
            ))
            payload = _finalize_partition_worker(_worker_payload(key, chunks, archive_root, quarantine_root, registry))
            failure = _record_worker_payload(
                payload=payload,
                manifest_root=manifest_root,
                failure_root=failure_root,
                completed=completed,
                failed=failed,
            )
            if payload.get("ok"):
                rows_read, rows_written, exact_duplicates, conflicts, invalid_rows = _accumulate_partition_counts(
                    payload["result"], rows_read, rows_written, exact_duplicates, conflicts, invalid_rows
                )
            elif failure:
                failure_signatures[str(failure["normalised_failure_signature"])] += 1
                if failure_signatures[str(failure["normalised_failure_signature"])] >= fail_fast_same_signature_threshold:
                    aborted_early = True
                    abort_reason = f"repeated systemic failure: {failure['normalised_failure_signature']}"
                    break
    elif tasks:
        pending = iter(tasks)
        future_to_key: dict[concurrent.futures.Future[dict[str, Any]], tuple[str, int]] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            while len(future_to_key) < min(in_flight_limit, len(tasks)):
                key, chunks = next(pending)
                future_to_key[executor.submit(_finalize_partition_worker, _worker_payload(key, chunks, archive_root, quarantine_root, registry))] = key
            exhausted = len(future_to_key) == len(tasks)
            while future_to_key:
                _write_json(report_root / "progress_manifest.json", _progress_payload(
                    len(groups), completed=len(completed), failed=len(failed), rows_read=rows_read,
                    rows_written=rows_written, exact_duplicates=exact_duplicates, conflicts=conflicts,
                    invalid_rows=invalid_rows, current_partition=current_partition, started=started,
                    active_workers=len(future_to_key), configured_workers=workers, effective_workers=effective_workers,
                ))
                done, _pending_futures = concurrent.futures.wait(
                    future_to_key,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in sorted(done, key=lambda item: future_to_key[item]):
                    key = future_to_key.pop(future)
                    current_partition = f"{key[0]}_{key[1]}"
                    payload = _future_payload(future, key)
                    failure = _record_worker_payload(
                        payload=payload,
                        manifest_root=manifest_root,
                        failure_root=failure_root,
                        completed=completed,
                        failed=failed,
                    )
                    if payload.get("ok"):
                        rows_read, rows_written, exact_duplicates, conflicts, invalid_rows = _accumulate_partition_counts(
                            payload["result"], rows_read, rows_written, exact_duplicates, conflicts, invalid_rows
                        )
                    elif failure:
                        failure_signatures[str(failure["normalised_failure_signature"])] += 1
                        if failure_signatures[str(failure["normalised_failure_signature"])] >= fail_fast_same_signature_threshold:
                            aborted_early = True
                            abort_reason = f"repeated systemic failure: {failure['normalised_failure_signature']}"
                    if not aborted_early and not exhausted:
                        try:
                            next_key, next_chunks = next(pending)
                            future_to_key[executor.submit(
                                _finalize_partition_worker,
                                _worker_payload(next_key, next_chunks, archive_root, quarantine_root, registry),
                            )] = next_key
                        except StopIteration:
                            exhausted = True
                if aborted_early:
                    exhausted = True
    dataset_manifest = {
        "mode": "alpaca_5m_symbol_year_finalizer",
        "dry_run": False,
        "archive_root": str(archive_root),
        "report_root": str(report_root),
        "dataset_version": DATASET_VERSION,
        "partition_processing_status": "complete" if not failed else "failed",
        "completed_partitions": len(completed),
        "failed_partitions": len(failed),
        "row_count": sum(int(row.get("output_row_count", 0) or 0) for row in completed),
        "symbols": sorted({row.get("canonical_symbol") for row in completed}),
        "years": sorted({int(row.get("year")) for row in completed if row.get("year") is not None}),
        "elapsed_seconds": time.perf_counter() - started,
        "configured_workers": workers,
        "effective_workers": effective_workers,
        "max_in_flight_tasks": in_flight_limit,
        "source_discovery_passes": 1,
        "partitions_dispatched": len(tasks),
        "partition_keys_dispatched_unique": len({(key[0], key[1]) for key, _chunks in tasks}),
        "partition_processing_backend": "process_pool" if effective_workers > 1 else "inline",
        "rows_per_second": sum(int(row.get("output_row_count", 0) or 0) for row in completed) / max(0.001, time.perf_counter() - started),
        "partitions_per_second": len(completed) / max(0.001, time.perf_counter() - started),
        "approximate_peak_worker_memory_bytes": max((int(row.get("peak_python_memory_bytes") or 0) for row in completed), default=0),
        "retry_only_failed": retry_only_failed,
        "protected_source_archives_modified": False,
    }
    _write_json(report_root / "dataset_manifest.json", dataset_manifest)
    _write_json(report_root / "failed_partitions.json", {"failed_partition_count": len(failed), "failed_partitions": failed})
    _write_json(report_root / "progress_manifest.json", _progress_payload(
        len(groups), completed=len(completed), failed=len(failed), rows_read=rows_read,
        rows_written=rows_written, exact_duplicates=exact_duplicates, conflicts=conflicts,
        invalid_rows=invalid_rows, current_partition="", started=started,
        aborted_early=aborted_early,
        abort_reason=abort_reason,
        configured_workers=workers,
        effective_workers=effective_workers,
        active_workers=0,
    ))
    if failed:
        raise RuntimeError(f"5m symbol/year finalisation failed for {len(failed)} partitions")
    return dataset_manifest


def _worker_payload(
    key: tuple[str, int],
    chunks: Sequence[SourceChunk],
    archive_root: Path,
    quarantine_root: Path,
    registry: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    symbol, year = key
    return {
        "canonical_symbol": symbol,
        "year": year,
        "chunks": list(chunks),
        "archive_root": archive_root,
        "quarantine_root": quarantine_root,
        "registry": dict(registry),
    }


def _finalize_partition_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    symbol = str(payload["canonical_symbol"])
    year = int(payload["year"])
    chunks = list(payload["chunks"])
    started = time.perf_counter()
    tracemalloc.start()
    try:
        result = finalize_partition(
            canonical_symbol=symbol,
            year=year,
            chunks=chunks,
            archive_root=Path(payload["archive_root"]),
            quarantine_root=Path(payload["quarantine_root"]),
            registry=payload["registry"],
        )
        _current, peak = tracemalloc.get_traced_memory()
        result["worker_pid"] = os.getpid()
        result["peak_python_memory_bytes"] = int(peak)
        result["worker_elapsed_seconds"] = time.perf_counter() - started
        return {"ok": True, "result": result}
    except Exception as exc:
        _current, peak = tracemalloc.get_traced_memory()
        signature = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        return {
            "ok": False,
            "failure": {
                "canonical_symbol": symbol,
                "year": year,
                "failure_phase": "partition_finalisation",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "normalised_failure_signature": signature,
                "traceback": traceback.format_exc(),
                "source_chunk_identities": [chunk.chunk_id for chunk in chunks],
                "worker_pid": os.getpid(),
                "peak_python_memory_bytes": int(peak),
                "worker_elapsed_seconds": time.perf_counter() - started,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
    finally:
        tracemalloc.stop()


def _future_payload(future: concurrent.futures.Future[dict[str, Any]], key: tuple[str, int]) -> dict[str, Any]:
    try:
        return future.result()
    except Exception as exc:
        signature = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        return {
            "ok": False,
            "failure": {
                "canonical_symbol": key[0],
                "year": key[1],
                "failure_phase": "partition_worker_future",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "normalised_failure_signature": signature,
                "traceback": traceback.format_exc(),
                "source_chunk_identities": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }


def _record_worker_payload(
    *,
    payload: Mapping[str, Any],
    manifest_root: Path,
    failure_root: Path,
    completed: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if payload.get("ok"):
        result = dict(payload["result"])
        _write_json(manifest_root / f"{result['canonical_symbol']}_{result['year']}.json", result)
        completed.append(result)
        return None
    failure = dict(payload["failure"])
    _write_json(failure_root / f"{failure['canonical_symbol']}_{failure['year']}.json", failure)
    failed.append(failure)
    return failure


def _accumulate_partition_counts(
    result: Mapping[str, Any],
    rows_read: int,
    rows_written: int,
    exact_duplicates: int,
    conflicts: int,
    invalid_rows: int,
) -> tuple[int, int, int, int, int]:
    return (
        rows_read + int(result.get("source_row_count", 0) or 0),
        rows_written + int(result.get("output_row_count", 0) or 0),
        exact_duplicates + int(result.get("exact_duplicates_removed", 0) or 0),
        conflicts + int(result.get("conflicting_duplicate_count", 0) or 0),
        invalid_rows + int(result.get("invalid_row_count", 0) or 0),
    )


def select_production_chunks(
    *,
    raw_root: Path,
    parquet_root: Path,
    collection_manifest_path: Path,
    symbols: set[str],
    years: set[int],
    max_chunks: int,
    max_chunks_per_symbol_year: int = 0,
    source_chunk_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    manifest_payload = _read_json(collection_manifest_path)
    production_ids = set((manifest_payload.get("chunks") or {}).keys())
    chunks: list[SourceChunk] = []
    excluded: list[dict[str, Any]] = []
    candidate_count = 0
    per_symbol_year_counts: Counter[tuple[str, int]] = Counter()
    raw_dirs = [Path(path) for path in source_chunk_paths] if source_chunk_paths else list(_raw_chunk_dirs(raw_root))
    for chunk_dir in raw_dirs:
        candidate_count += 1
        manifest = _read_json(chunk_dir / "manifest.json")
        chunk_id = chunk_id_from_manifest(manifest)
        if chunk_id not in production_ids:
            excluded.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "reason": "not_in_production_collection_manifest"})
            continue
        if manifest.get("completion_state") != "completed":
            excluded.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "reason": "raw_manifest_not_completed"})
            continue
        if symbols and not (set(_canonical_batch(manifest)) & symbols):
            excluded.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "reason": "outside_requested_symbols"})
            continue
        if years and not _chunk_overlaps_years(manifest, years):
            excluded.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "reason": "outside_requested_years"})
            continue
        if max_chunks_per_symbol_year and symbols and years:
            matching_keys = [
                (symbol, year)
                for symbol in _canonical_batch(manifest)
                if symbol in symbols
                for year in _years_for_chunk(manifest)
                if year in years
            ]
            if not any(per_symbol_year_counts[key] < max_chunks_per_symbol_year for key in matching_keys):
                excluded.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "reason": "symbol_year_chunk_limit_reached"})
                continue
        path = parquet_path(parquet_root, raw_root, chunk_dir)
        if not path.exists():
            excluded.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "reason": "missing_converted_parquet", "expected_parquet": str(path)})
            continue
        chunks.append(SourceChunk(chunk_id=chunk_id, raw_path=chunk_dir, parquet_path=path, manifest=manifest, reason="production_completed_with_parquet"))
        for symbol in _canonical_batch(manifest):
            if symbols and symbol not in symbols:
                continue
            for year in _years_for_chunk(manifest):
                if years and year not in years:
                    continue
                per_symbol_year_counts[(symbol, year)] += 1
        if max_chunks and len(chunks) >= max_chunks:
            break
    duplicate_ids = [chunk_id for chunk_id, count in Counter(chunk.chunk_id for chunk in chunks).items() if count > 1]
    report = {
        "raw_root": str(raw_root),
        "parquet_root": str(parquet_root),
        "collection_manifest_path": str(collection_manifest_path),
        "candidate_chunks": candidate_count,
        "production_manifest_chunk_count": len(production_ids),
        "production_chunks": len(chunks),
        "excluded_chunks": len(excluded),
        "excluded_by_reason": dict(Counter(row["reason"] for row in excluded)),
        "missing_converted_chunks": sum(1 for row in excluded if row["reason"] == "missing_converted_parquet"),
        "duplicate_chunk_id_count": len(duplicate_ids),
        "duplicate_chunk_ids": duplicate_ids[:20],
        "max_chunks_per_symbol_year": max_chunks_per_symbol_year,
        "selected_chunks_per_symbol_year_min": min(per_symbol_year_counts.values(), default=0),
        "selected_chunks_per_symbol_year_max": max(per_symbol_year_counts.values(), default=0),
        "symbols_represented": sorted({symbol for chunk in chunks for symbol in _canonical_batch(chunk.manifest)}),
        "date_min": min((str(chunk.manifest.get("requested_start")) for chunk in chunks), default=None),
        "date_max": max((str(chunk.manifest.get("requested_end")) for chunk in chunks), default=None),
        "excluded_sample": excluded[:100],
        "selection_rule": "include iff raw manifest is completed, chunk id is completed in production collection manifest, converted Parquet exists, and optional symbol/year filters match",
    }
    return {"chunks": chunks, "report": report}


def finalize_partition(
    *,
    canonical_symbol: str,
    year: int,
    chunks: Sequence[SourceChunk],
    archive_root: Path,
    quarantine_root: Path,
    registry: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    source_row_count = 0
    for chunk in chunks:
        table = pq.ParquetFile(chunk.parquet_path).read()
        source_row_count += table.num_rows
        for row in _iter_table_rows(table):
            normalized = normalize_output_row(row, chunk=chunk, registry=registry)
            if normalized["canonical_symbol"] == canonical_symbol and normalized["timestamp_utc"].year == year:
                rows.append(normalized)
    deduped, duplicate_report = deduplicate_rows(rows)
    if duplicate_report["conflicting_duplicate_count"]:
        path = quarantine_root / f"{canonical_symbol}_{year}_conflicts.json"
        _write_json(path, duplicate_report)
        raise ValueError(f"conflicting duplicates for {canonical_symbol} {year}: {duplicate_report['conflicting_duplicate_count']}")
    invalid = validate_rows(deduped)
    if invalid:
        raise ValueError(f"invalid output rows for {canonical_symbol} {year}: {invalid[:3]}")
    target = archive_root / f"symbol={canonical_symbol}" / f"year={year}" / "bars.parquet"
    table = pa.Table.from_pylist(deduped, schema=OUTPUT_SCHEMA)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(target)
    timestamps = [row["timestamp_utc"] for row in deduped]
    manifest = {
        "canonical_symbol": canonical_symbol,
        "year": year,
        "source_chunk_count": len(chunks),
        "source_row_count": source_row_count,
        "output_row_count": len(deduped),
        "exact_duplicates_removed": duplicate_report["exact_duplicates_removed"],
        "conflicting_duplicate_count": duplicate_report["conflicting_duplicate_count"],
        "invalid_row_count": len(invalid),
        "minimum_timestamp": min((ts.isoformat() for ts in timestamps), default=None),
        "maximum_timestamp": max((ts.isoformat() for ts in timestamps), default=None),
        "session_counts": dict(Counter(row["session_type"] for row in deduped)),
        "schema_fingerprint": schema_fingerprint(OUTPUT_SCHEMA),
        "source_identity_hash": _hash_list([chunk.chunk_id for chunk in chunks]),
        "output_file_hash": _file_sha256(target),
        "path": str(target),
        "status": "COMPLETE",
        "dataset_version": DATASET_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return manifest


def normalize_output_row(row: Mapping[str, Any], *, chunk: SourceChunk, registry: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    canonical = normalize_symbol(row.get("symbol"))
    provider_symbol = normalize_symbol(row.get("provider_symbol")) or alpaca_provider_symbol(canonical)
    if provider_symbol in {"BRK.A", "BRK.B"}:
        canonical = provider_symbol.replace(".", "-")
    if canonical not in registry:
        raise ValueError(f"unknown canonical symbol: {canonical}")
    timestamp = _to_utc(row.get("timestamp"))
    open_ = _float(row.get("open"))
    high = _float(row.get("high"))
    low = _float(row.get("low"))
    close = _float(row.get("close"))
    values = {
        "asset_id": registry[canonical]["asset_id"],
        "canonical_symbol": canonical,
        "provider_symbol": provider_symbol,
        "timestamp_utc": timestamp,
        "session_date": timestamp.date().isoformat(),
        "session_type": str(row.get("session_type") or session_type(timestamp)),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": _float(row.get("volume")),
        "trade_count": _int_or_none(row.get("trade_count")),
        "vwap": _float_or_none(row.get("vwap")),
        "provider": str(row.get("provider") or "alpaca"),
        "feed": str(row.get("feed") or chunk.manifest.get("feed") or "sip").upper(),
        "timeframe": "5Min",
        "adjustment_policy": str(row.get("adjustment_mode") or chunk.manifest.get("adjustment_mode") or ""),
        "raw_chunk_id": str(row.get("raw_chunk_identifier") or chunk.chunk_id),
        "dataset_version": DATASET_VERSION,
    }
    values["source_row_hash"] = _source_row_hash(values)
    return values


def deduplicate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
    conflicts = []
    exact_duplicates = 0
    for row in sorted(rows, key=lambda item: (item["canonical_symbol"], item["timestamp_utc"], item["raw_chunk_id"])):
        key = (str(row["canonical_symbol"]), row["timestamp_utc"])
        current = dict(row)
        if key not in by_key:
            by_key[key] = current
            continue
        existing = by_key[key]
        market_fields = ("open", "high", "low", "close", "volume", "trade_count", "vwap")
        if all(existing.get(field) == current.get(field) for field in market_fields):
            exact_duplicates += 1
            existing["raw_chunk_id"] = "|".join(sorted(set(str(existing["raw_chunk_id"]).split("|") + [str(current["raw_chunk_id"])])))
            existing["source_row_hash"] = _source_row_hash(existing)
        else:
            conflicts.append({
                "canonical_symbol": key[0],
                "timestamp_utc": key[1].isoformat(),
                "left_raw_chunk_id": existing.get("raw_chunk_id"),
                "right_raw_chunk_id": current.get("raw_chunk_id"),
                "left_row": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in existing.items()},
                "right_row": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in current.items()},
            })
    return sorted(by_key.values(), key=lambda item: (item["canonical_symbol"], item["timestamp_utc"])), {
        "input_row_count": len(rows),
        "output_row_count": len(by_key),
        "exact_duplicates_removed": exact_duplicates,
        "conflicting_duplicate_count": len(conflicts),
        "conflicts": conflicts[:1000],
    }


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    invalid = []
    seen = set()
    for index, row in enumerate(rows):
        key = (row["canonical_symbol"], row["timestamp_utc"])
        if key in seen:
            invalid.append({"row_index": index, "reason": "duplicate_key", "key": str(key)})
        seen.add(key)
        if row["timestamp_utc"].tzinfo is None:
            invalid.append({"row_index": index, "reason": "timestamp_not_timezone_aware"})
        if row["high"] < max(row["open"], row["close"], row["low"]):
            invalid.append({"row_index": index, "reason": "invalid_high"})
        if row["low"] > min(row["open"], row["close"], row["high"]):
            invalid.append({"row_index": index, "reason": "invalid_low"})
        if row["volume"] is None or row["volume"] < 0:
            invalid.append({"row_index": index, "reason": "invalid_volume"})
        if row["trade_count"] is not None and row["trade_count"] < 0:
            invalid.append({"row_index": index, "reason": "invalid_trade_count"})
        if row["feed"].lower() != "sip":
            invalid.append({"row_index": index, "reason": "non_sip_feed"})
    return invalid


def production_preflight(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    collection_manifest_path: Path = DEFAULT_COLLECTION_MANIFEST,
    universe_path: Path = Path("config/universes/alpaca_514_symbols.txt"),
    symbols: Sequence[str] = (),
    years: Sequence[int] = (),
    max_chunks: int = 0,
    max_chunks_per_symbol_year: int = 0,
    source_chunk_paths: Sequence[Path] = (),
    workers: int = 1,
) -> dict[str, Any]:
    selected_symbols = {normalize_symbol(symbol) for symbol in symbols if symbol}
    selected_years = {int(year) for year in years}
    selection = select_production_chunks(
        raw_root=raw_root,
        parquet_root=parquet_root,
        collection_manifest_path=collection_manifest_path,
        symbols=selected_symbols,
        years=selected_years,
        max_chunks=max_chunks,
        max_chunks_per_symbol_year=max_chunks_per_symbol_year,
        source_chunk_paths=source_chunk_paths,
    )
    registry = _registry(universe_path)
    unresolved = sorted({symbol for chunk in selection["chunks"] for symbol in _canonical_batch(chunk.manifest) if symbol not in registry})
    schema_variants: Counter[str] = Counter()
    metadata_rows = 0
    missing_source_files = []
    for chunk in selection["chunks"]:
        if not chunk.parquet_path.exists():
            missing_source_files.append(str(chunk.parquet_path))
            continue
        pf = pq.ParquetFile(chunk.parquet_path)
        metadata_rows += pf.metadata.num_rows
        schema_variants[schema_fingerprint(pf.schema_arrow)] += 1
    groups = _group_chunks_by_symbol_year(selection["chunks"], selected_symbols, selected_years)
    completed_reusable = sum(
        1 for symbol, year in groups if _completed_manifest_valid(report_root / "partition_manifests" / f"{symbol}_{year}.json", archive_root)
    )
    failed = list((report_root / "partition_failures").glob("*.json")) if (report_root / "partition_failures").exists() else []
    free = shutil.disk_usage(archive_root.parent if archive_root.parent.exists() else Path(".")).free
    estimated_final = int(metadata_rows * 28) if metadata_rows else 0
    blocking = []
    if selection["report"]["missing_converted_chunks"]:
        blocking.append("missing_converted_production_chunks")
    if unresolved:
        blocking.append("unresolved_symbols")
    if "smoke" in str(archive_root).lower() and not source_chunk_paths:
        blocking.append("smoke_root_without_explicit_source_chunk_paths")
    if not groups:
        blocking.append("empty_partition_plan")
    if estimated_final and free < estimated_final * 2:
        blocking.append("unsafe_free_disk_estimate")
    payload = {
        "selected_production_chunk_count": len(selection["chunks"]),
        "selected_converted_file_count": sum(1 for chunk in selection["chunks"] if chunk.parquet_path.exists()),
        "selected_input_rows_from_metadata": metadata_rows,
        "planned_canonical_symbols": sorted({symbol for symbol, _year in groups}),
        "planned_symbol_year_partitions": len(groups),
        "planned_symbol_year_keys": [f"{symbol}:{year}" for symbol, year in sorted(groups)],
        "missing_source_files": missing_source_files,
        "unresolved_symbols": unresolved,
        "duplicate_chunk_identities": selection["report"]["duplicate_chunk_ids"],
        "overlapping_requested_chunk_intervals": _overlap_report(selection["chunks"]),
        "schema_variants": dict(schema_variants),
        "estimated_output_rows": metadata_rows,
        "estimated_temporary_disk_requirement_bytes": estimated_final,
        "estimated_final_disk_requirement_bytes": estimated_final,
        "free_disk_bytes": free,
        "worker_count": workers,
        "configured_workers": workers,
        "effective_workers": min(workers, len(groups)) if groups else 0,
        "source_discovery_passes": 1,
        "workers_rescan_sources": False,
        "output_root": str(archive_root),
        "report_root": str(report_root),
        "completed_partitions_already_reusable": completed_reusable,
        "failed_partitions": len(failed),
        "pending_partitions": max(0, len(groups) - completed_reusable),
        "blocking_issues": blocking,
        "valid": not blocking,
        "source_chunk_paths_explicit": bool(source_chunk_paths),
        "source_chunk_plan": selection["report"],
    }
    _write_json(report_root / "preflight.json", payload)
    (report_root / "preflight.md").write_text(_preflight_markdown(payload), encoding="utf-8")
    _write_json(Path("reports/data_quality/alpaca_5m_symbol_year_preflight.json"), payload)
    Path("reports/data_quality/alpaca_5m_symbol_year_preflight.md").write_text(_preflight_markdown(payload), encoding="utf-8")
    if blocking:
        raise ValueError(f"alpaca 5m preflight failed: {blocking}")
    return payload


def validate_final_archive(archive_root: Path, report_root: Path) -> dict[str, Any]:
    files = sorted(archive_root.glob("symbol=*/year=*/bars.parquet"))
    invalid = []
    total_rows = 0
    symbols = set()
    years = set()
    min_ts = None
    max_ts = None
    schema_fps = Counter()
    tmp_files = [str(path) for path in archive_root.rglob("*.tmp")] if archive_root.exists() else []
    for path in files:
        table = pq.ParquetFile(path).read()
        schema_fps[schema_fingerprint(table.schema)] += 1
        rows = list(_iter_table_rows(table))
        total_rows += len(rows)
        manifest = _read_json(report_root / "partition_manifests" / f"{path.parents[1].name.split('=',1)[1]}_{path.parent.name.split('=',1)[1]}.json")
        if manifest and int(manifest.get("output_row_count", -1)) != len(rows):
            invalid.append({"path": str(path), "reason": "manifest_row_count_mismatch"})
        if manifest and manifest.get("output_file_hash") != _file_sha256(path):
            invalid.append({"path": str(path), "reason": "manifest_file_hash_mismatch"})
        last_key = None
        seen = set()
        for index, row in enumerate(rows):
            symbols.add(row["canonical_symbol"])
            years.add(row["timestamp_utc"].year)
            key = (row["canonical_symbol"], row["timestamp_utc"])
            if key in seen:
                invalid.append({"path": str(path), "row_index": index, "reason": "duplicate_key"})
            seen.add(key)
            if last_key and key < last_key:
                invalid.append({"path": str(path), "row_index": index, "reason": "timestamp_order"})
            last_key = key
            if row["session_date"] != row["timestamp_utc"].date().isoformat():
                invalid.append({"path": str(path), "row_index": index, "reason": "session_date_mismatch"})
            invalid.extend({"path": str(path), **item} for item in validate_rows([row]))
            min_ts = min(row["timestamp_utc"].isoformat(), min_ts) if min_ts else row["timestamp_utc"].isoformat()
            max_ts = max(row["timestamp_utc"].isoformat(), max_ts) if max_ts else row["timestamp_utc"].isoformat()
            if not row.get("source_row_hash") or not row.get("dataset_version"):
                invalid.append({"path": str(path), "row_index": index, "reason": "missing_identity_field"})
    payload = {
        "partition_count": len(files),
        "symbol_count": len(symbols),
        "year_coverage": sorted(years),
        "total_rows": total_rows,
        "minimum_timestamp": min_ts,
        "maximum_timestamp": max_ts,
        "invalid_rows": len(invalid),
        "invalid_samples": invalid[:100],
        "schema_fingerprints": dict(schema_fps),
        "temporary_files_left_behind": tmp_files,
        "valid": not invalid and not tmp_files,
    }
    _write_json(report_root / "archive_validation.json", payload)
    (report_root / "archive_validation.md").write_text(f"# Alpaca 5m Archive Validation\n\nValid: {payload['valid']}\nRows: {total_rows}\nPartitions: {len(files)}\n", encoding="utf-8")
    return payload


def derive_regular_session_daily(rows: Sequence[Mapping[str, Any]], *, expected_regular_session_bar_count: int = 78) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("session_type") == "rth":
            grouped[(str(row["canonical_symbol"]), str(row["session_date"]))].append(row)
    output = []
    for (symbol, session_date), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["timestamp_utc"])
        volume = sum(float(row.get("volume") or 0.0) for row in ordered)
        vwap_num = sum(float(row.get("vwap") or 0.0) * float(row.get("volume") or 0.0) for row in ordered if row.get("vwap") is not None)
        output.append({
            "asset_id": ordered[0]["asset_id"],
            "canonical_symbol": symbol,
            "session_date": session_date,
            "open": float(ordered[0]["open"]),
            "high": max(float(row["high"]) for row in ordered),
            "low": min(float(row["low"]) for row in ordered),
            "close": float(ordered[-1]["close"]),
            "volume": volume,
            "trade_count": sum(int(row.get("trade_count") or 0) for row in ordered),
            "vwap": vwap_num / volume if volume else None,
            "first_bar_timestamp": ordered[0]["timestamp_utc"],
            "last_bar_timestamp": ordered[-1]["timestamp_utc"],
            "regular_session_bar_count": len(ordered),
            "expected_regular_session_bar_count": expected_regular_session_bar_count,
            "session_completeness_flag": "complete" if len(ordered) == expected_regular_session_bar_count else "incomplete",
            "source_5m_dataset_version": ordered[0].get("dataset_version"),
            "derived_dataset_version": "alpaca_5m_regular_session_daily_v1",
        })
    return output


INTRADAY_FEATURE_VERSION = "alpaca_5m_intraday_summary_features_v1"


def build_intraday_summary_features(rows: Sequence[Mapping[str, Any]], *, spy_rows: Sequence[Mapping[str, Any]] = (), feature_available_hour_utc: int = 21) -> list[dict[str, Any]]:
    daily = derive_regular_session_daily(rows)
    spy_daily = {row["session_date"]: row for row in derive_regular_session_daily(spy_rows)}
    features = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("session_type") == "rth":
            grouped[(row["canonical_symbol"], row["session_date"])].append(row)
    for day in daily:
        group = sorted(grouped[(day["canonical_symbol"], day["session_date"])], key=lambda row: row["timestamp_utc"])
        closes = [float(row["close"]) for row in group]
        returns = [(right / left - 1.0) for left, right in zip(closes, closes[1:]) if left]
        intraday_return = day["close"] / day["open"] - 1.0 if day["open"] else None
        spy = spy_daily.get(day["session_date"])
        spy_return = spy["close"] / spy["open"] - 1.0 if spy and spy["open"] else None
        volume = float(day["volume"] or 0.0)
        first6 = group[:6]
        last12 = group[-12:]
        feature_available = f"{day['session_date']}T{feature_available_hour_utc:02d}:00:00+00:00"
        features.append({
            "asset_id": day["asset_id"],
            "canonical_symbol": day["canonical_symbol"],
            "completed_session_date": day["session_date"],
            "feature_available_timestamp": feature_available,
            "feature_version": INTRADAY_FEATURE_VERSION,
            "intraday_return": intraday_return,
            "realised_volatility": _stddev(returns),
            "downside_realised_volatility": _stddev([value for value in returns if value < 0.0]),
            "high_low_range_percent": day["high"] / day["low"] - 1.0 if day["low"] else None,
            "maximum_intraday_drawdown": _max_drawdown(closes),
            "distance_from_session_high": day["close"] / day["high"] - 1.0 if day["high"] else None,
            "distance_from_session_low": day["close"] / day["low"] - 1.0 if day["low"] else None,
            "distance_from_session_vwap": day["close"] / day["vwap"] - 1.0 if day["vwap"] else None,
            "opening_30m_return": first6[-1]["close"] / first6[0]["open"] - 1.0 if len(first6) >= 2 and first6[0]["open"] else None,
            "morning_return": closes[min(len(closes)-1, 39)] / day["open"] - 1.0 if closes and day["open"] else None,
            "afternoon_return": day["close"] / closes[min(len(closes)-1, 39)] - 1.0 if closes and closes[min(len(closes)-1, 39)] else None,
            "last_hour_return": last12[-1]["close"] / last12[0]["open"] - 1.0 if len(last12) >= 2 and last12[0]["open"] else None,
            "intraday_reversal": -(intraday_return or 0.0),
            "opening_range_breakout": day["close"] / max(float(row["high"]) for row in first6) - 1.0 if first6 else None,
            "relative_volume": None,
            "opening_volume_share": sum(float(row.get("volume") or 0.0) for row in first6) / volume if volume else None,
            "closing_volume_share": sum(float(row.get("volume") or 0.0) for row in last12) / volume if volume else None,
            "average_trade_size": volume / day["trade_count"] if day["trade_count"] else None,
            "trade_count_intensity": day["trade_count"] / len(group) if group else None,
            "stock_minus_spy_intraday_return": intraday_return - spy_return if intraday_return is not None and spy_return is not None else None,
            "stock_minus_sector_intraday_return": None,
            "selector_join_contract": "asset_id + completed_session_date + feature_available_timestamp + selector row_id",
            "same_session_preopen_safe": False,
        })
    return features


def write_current_inventory(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    final_root: Path = DEFAULT_ARCHIVE_ROOT,
    collection_manifest_path: Path = DEFAULT_COLLECTION_MANIFEST,
    output_root: Path = Path("reports/data_quality"),
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    collection = _read_json(collection_manifest_path)
    production_ids = set((collection.get("chunks") or {}).keys())
    raw_manifests = list(_raw_chunk_dirs(raw_root))
    raw_counts = Counter()
    raw_symbols: set[str] = set()
    raw_min = None
    raw_max = None
    production_row_count = 0
    missing_conversions = []
    for chunk_dir in raw_manifests:
        manifest = _read_json(chunk_dir / "manifest.json")
        chunk_id = chunk_id_from_manifest(manifest)
        classification = "production" if chunk_id in production_ids else "unknown_or_pilot"
        raw_counts[classification] += 1
        raw_counts[f"{classification}_{manifest.get('completion_state', 'unknown')}"] += 1
        if classification == "production" and manifest.get("completion_state") == "completed":
            production_row_count += int(manifest.get("row_count", 0) or 0)
        raw_symbols.update(_canonical_batch(manifest))
        raw_min = _min_text(raw_min, manifest.get("actual_earliest_timestamp") or manifest.get("requested_start"))
        raw_max = _max_text(raw_max, manifest.get("actual_latest_timestamp") or manifest.get("requested_end"))
        if classification == "production" and not parquet_path(parquet_root, raw_root, chunk_dir).exists():
            missing_conversions.append({"chunk_id": chunk_id, "source_path": str(chunk_dir), "expected_parquet": str(parquet_path(parquet_root, raw_root, chunk_dir))})
    parquet_files = list(parquet_root.rglob("bars.parquet")) if parquet_root.exists() else []
    final_files = list(final_root.glob("symbol=*/year=*/bars.parquet")) if final_root.exists() else []
    coverage_rows = [{"canonical_symbol": symbol, "provider_symbol": alpaca_provider_symbol(symbol), "status": "CONVERTED_NOT_FINALISED"} for symbol in sorted(raw_symbols)]
    coverage_path = output_root / "alpaca_5m_symbol_coverage.parquet"
    _write_parquet_rows(coverage_path, coverage_rows, ["canonical_symbol", "provider_symbol", "status"])
    missing_path = output_root / "alpaca_5m_missing_or_invalid_chunks.parquet"
    _write_parquet_rows(missing_path, missing_conversions, ["chunk_id", "source_path", "expected_parquet"])
    report = {
        "raw": {
            "chunk_manifest_count": len(raw_manifests),
            "production_chunk_count": raw_counts["production"],
            "pilot_or_unknown_chunk_count": raw_counts["unknown_or_pilot"],
            "completed_production_chunk_count": raw_counts["production_completed"],
            "completed_production_manifest_row_count": production_row_count,
            "unique_symbols": len(raw_symbols),
            "actual_earliest_timestamp": raw_min,
            "actual_latest_timestamp": raw_max,
            "disk_size_bytes": "deferred_filesystem_scan",
        },
        "converted": {
            "bars_parquet_count": len(parquet_files),
            "completed_raw_chunks_missing_converted_parquet": len(missing_conversions),
            "disk_size_bytes": "deferred_filesystem_scan",
            "row_count": "deferred_metadata_scan",
        },
        "final_archive": {
            "exists": final_root.exists(),
            "bars_parquet_count": len(final_files),
            "status": "absent_not_finalised" if not final_root.exists() else "partial_or_unknown",
            "disk_size_bytes": 0 if not final_root.exists() else "deferred_filesystem_scan",
        },
        "production_selection_rule": "chunk id must be completed in 5m_sip_514_symbol_full collection manifest and raw manifest completion_state must be completed; converted Parquet must exist",
        "symbol_mapping": {"BRK.A": "BRK-A", "BRK.B": "BRK-B"},
        "recommended_next_action": "run bounded symbol/year finalisation smoke after alpha process is quiet",
        "protected_source_archives_modified": False,
    }
    _write_json(output_root / "alpaca_5m_current_inventory.json", report)
    (output_root / "alpaca_5m_current_inventory.md").write_text(_inventory_markdown(report), encoding="utf-8")
    return report


def _group_chunks_by_symbol_year(chunks: Sequence[SourceChunk], symbols: set[str], years: set[int]) -> dict[tuple[str, int], list[SourceChunk]]:
    groups: dict[tuple[str, int], list[SourceChunk]] = defaultdict(list)
    for chunk in chunks:
        chunk_symbols = set(_canonical_batch(chunk.manifest))
        chunk_years = _years_for_chunk(chunk.manifest)
        for symbol in sorted(chunk_symbols):
            if symbols and symbol not in symbols:
                continue
            for year in sorted(chunk_years):
                if years and year not in years:
                    continue
                groups[(symbol, year)].append(chunk)
    return groups


def _registry(universe_path: Path) -> dict[str, dict[str, str]]:
    assets, _aliases, _version = build_registry_from_universe(universe_path, provider_symbol_map={"BRK-A": "BRK.A", "BRK-B": "BRK.B"})
    return {asset.canonical_symbol: {"asset_id": asset.asset_id or canonical_asset_id(asset.canonical_symbol)} for asset in assets}


def _raw_chunk_dirs(raw_root: Path) -> Iterable[Path]:
    if not raw_root.exists():
        return []
    return (
        path.parent
        for path in raw_root.rglob("manifest.json")
        if not any(part.endswith(".tmp") or part == ".tmp" for part in path.parts)
    )


def _canonical_batch(manifest: Mapping[str, Any]) -> list[str]:
    return [normalize_symbol(symbol) for symbol in (manifest.get("canonical_symbol_batch") or manifest.get("symbol_batch") or [])]


def _years_for_chunk(manifest: Mapping[str, Any]) -> set[int]:
    years = set()
    for key in ("requested_start", "requested_end", "actual_earliest_timestamp", "actual_latest_timestamp"):
        value = manifest.get(key)
        if value:
            years.add(_to_utc(value).year)
    return years


def _chunk_overlaps_years(manifest: Mapping[str, Any], years: set[int]) -> bool:
    return bool(_years_for_chunk(manifest) & years)


def _iter_table_rows(table: pa.Table) -> Iterable[dict[str, Any]]:
    columns = {name: table[name] for name in table.column_names}
    for index in range(table.num_rows):
        yield {name: columns[name][index].as_py() for name in table.column_names}


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _float(value: Any) -> float:
    if value is None:
        raise ValueError("required numeric value missing")
    return float(value)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _source_row_hash(row: Mapping[str, Any]) -> str:
    payload = {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in row.items()
        if key != "source_row_hash"
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _hash_list(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256("|".join(f"{field.name}:{field.type}" for field in schema).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_manifest_valid(path: Path, archive_root: Path) -> bool:
    payload = _read_json(path)
    if payload.get("status") != "COMPLETE":
        return False
    output = Path(str(payload.get("path") or ""))
    return output.exists() and archive_root in output.parents


def _progress_payload(
    planned: int,
    *,
    completed: int = 0,
    failed: int = 0,
    rows_read: int = 0,
    rows_written: int = 0,
    exact_duplicates: int = 0,
    conflicts: int = 0,
    invalid_rows: int = 0,
    current_partition: str = "",
    started: float,
    active_workers: int = 0,
    configured_workers: int = 1,
    effective_workers: int = 1,
    aborted_early: bool = False,
    abort_reason: str = "",
) -> dict[str, Any]:
    elapsed = max(0.001, time.perf_counter() - started)
    return {
        "planned_partitions": planned,
        "completed_partitions": completed,
        "pending_partitions": max(0, planned - completed - failed),
        "failed_partitions": failed,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "exact_duplicates_removed": exact_duplicates,
        "conflicting_duplicates": conflicts,
        "invalid_rows": invalid_rows,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows_written / elapsed,
        "partitions_per_second": completed / elapsed,
        "configured_workers": configured_workers,
        "effective_workers": effective_workers,
        "active_workers": active_workers,
        "current_partition": current_partition,
        "aborted_early": aborted_early,
        "abort_reason": abort_reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _overlap_report(chunks: Sequence[SourceChunk]) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    overlaps = []
    for chunk in chunks:
        start = _to_utc(chunk.manifest["requested_start"])
        end = _to_utc(chunk.manifest["requested_end"])
        for symbol in _canonical_batch(chunk.manifest):
            by_symbol[symbol].append((start, end, chunk.chunk_id))
    for symbol, ranges in by_symbol.items():
        ordered = sorted(ranges)
        for left, right in zip(ordered, ordered[1:]):
            if right[0] < left[1]:
                overlaps.append({"symbol": symbol, "left_chunk": left[2], "right_chunk": right[2]})
    return overlaps[:100]


def _preflight_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Alpaca 5m Symbol/Year Preflight",
        "",
        f"- Valid: {payload['valid']}",
        f"- Selected chunks: {payload['selected_production_chunk_count']}",
        f"- Input rows: {payload['selected_input_rows_from_metadata']}",
        f"- Planned partitions: {payload['planned_symbol_year_partitions']}",
        f"- Pending partitions: {payload['pending_partitions']}",
        f"- Blocking issues: {payload['blocking_issues']}",
    ])


def _stddev(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _max_drawdown(values: Sequence[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst


def _failed_partition_keys(failure_root: Path) -> set[tuple[str, int]]:
    keys = set()
    for path in failure_root.glob("*.json"):
        payload = _read_json(path)
        if payload.get("canonical_symbol") and payload.get("year") is not None:
            keys.add((str(payload["canonical_symbol"]).upper(), int(payload["year"])))
    return keys


def _planned_partition_count(chunks: Sequence[SourceChunk]) -> int:
    groups = _group_chunks_by_symbol_year(chunks, set(), set())
    return len(groups)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _write_parquet_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(name, pa.string()) for name in fieldnames])
    table = pa.Table.from_pylist(
        [{name: "" if row.get(name) is None else str(row.get(name)) for name in fieldnames} for row in rows],
        schema=schema,
    )
    pq.write_table(table, path, compression="zstd")


def _min_text(left: str | None, right: Any) -> str | None:
    if not right:
        return left
    text = str(right)
    return text if left is None or text < left else left


def _max_text(left: str | None, right: Any) -> str | None:
    if not right:
        return left
    text = str(right)
    return text if left is None or text > left else left


def _inventory_markdown(report: Mapping[str, Any]) -> str:
    raw = report["raw"]
    converted = report["converted"]
    final = report["final_archive"]
    return "\n".join(
        [
            "# Alpaca 5m Current Inventory",
            "",
            f"- Raw chunk manifests: {raw['chunk_manifest_count']}",
            f"- Raw production chunks: {raw['production_chunk_count']}",
            f"- Raw production rows from manifests: {raw['completed_production_manifest_row_count']}",
            f"- Raw pilot/unknown chunks: {raw['pilot_or_unknown_chunk_count']}",
            f"- Converted bars.parquet files: {converted['bars_parquet_count']}",
            f"- Missing production conversions: {converted['completed_raw_chunks_missing_converted_parquet']}",
            f"- Existing final symbol/year files: {final['bars_parquet_count']}",
            f"- Final archive status: {final['status']}",
            f"- Raw timestamp range: {raw['actual_earliest_timestamp']} through {raw['actual_latest_timestamp']}",
            f"- Raw unique symbols: {raw['unique_symbols']}",
            f"- Production selection: {report['production_selection_rule']}",
            f"- Recommended next action: {report['recommended_next_action']}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize Alpaca SIP 5-minute chunk Parquets into symbol/year partitions.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-only-failed", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--validate-archive", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    args = parser.parse_args(argv)
    if args.config:
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        settings = dict((config.get("ml", {}) or {}).get("alpaca_5m_symbol_year_finalizer", {}) or {})
        if args.execute:
            settings["dry_run"] = False
        if args.dry_run:
            settings["dry_run"] = True
        if args.retry_only_failed:
            settings["retry_only_failed"] = True
        if args.workers is not None:
            settings["workers"] = args.workers
        if args.archive_root is not None:
            settings["archive_root"] = str(args.archive_root)
        if args.report_root is not None:
            settings["report_root"] = str(args.report_root)
        config["ml"]["alpaca_5m_symbol_year_finalizer"] = settings
    else:
        settings = {"dry_run": not args.execute, "retry_only_failed": args.retry_only_failed}
        if args.workers is not None:
            settings["workers"] = args.workers
        if args.archive_root is not None:
            settings["archive_root"] = str(args.archive_root)
        if args.report_root is not None:
            settings["report_root"] = str(args.report_root)
        config = {"ml": {"alpaca_5m_symbol_year_finalizer": settings}}
    if args.inventory_only:
        print(json.dumps(write_current_inventory(), indent=2, default=str))
    elif args.preflight:
        settings = dict((config.get("ml", {}) or {}).get("alpaca_5m_symbol_year_finalizer", {}) or {})
        print(json.dumps(production_preflight(
            raw_root=Path(str(settings.get("raw_root", DEFAULT_RAW_ROOT))),
            parquet_root=Path(str(settings.get("parquet_root", DEFAULT_PARQUET_ROOT))),
            archive_root=Path(str(settings.get("archive_root", DEFAULT_ARCHIVE_ROOT))),
            report_root=Path(str(settings.get("report_root", DEFAULT_REPORT_ROOT))),
            collection_manifest_path=Path(str(settings.get("collection_manifest_path", DEFAULT_COLLECTION_MANIFEST))),
            universe_path=Path(str(settings.get("universe_path", "config/universes/alpaca_514_symbols.txt"))),
            symbols=[str(symbol).upper() for symbol in settings.get("symbols", [])],
            years=[int(year) for year in settings.get("years", [])],
            max_chunks=int(settings.get("max_chunks", 0) or 0),
            max_chunks_per_symbol_year=int(settings.get("max_chunks_per_symbol_year", 0) or 0),
            source_chunk_paths=[Path(str(path)) for path in settings.get("source_chunk_paths", [])],
            workers=int(settings.get("workers", 1) or 1),
        ), indent=2, default=str))
    elif args.validate_archive:
        settings = dict((config.get("ml", {}) or {}).get("alpaca_5m_symbol_year_finalizer", {}) or {})
        print(json.dumps(validate_final_archive(
            archive_root=Path(str(settings.get("archive_root", DEFAULT_ARCHIVE_ROOT))),
            report_root=Path(str(settings.get("report_root", DEFAULT_REPORT_ROOT))),
        ), indent=2, default=str))
    else:
        print(json.dumps(run_from_config(config), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
