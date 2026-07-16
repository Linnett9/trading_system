from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infrastructure.data.alpaca_5m_symbol_year_finalizer import (
    OUTPUT_SCHEMA,
    SourceChunk,
    _file_sha256,
    _iter_table_rows,
    _registry,
    deduplicate_rows,
    finalize_partition,
    normalize_output_row,
    validate_rows,
)
from scripts.convert_completed_alpaca_raw_chunks_to_parquet import chunk_id_from_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded before/after benchmark for Alpaca 5m partition reads.")
    parser.add_argument("--symbol", default="CCRN")
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("data/processed/alpaca/stock_bars_parquet/sip/5m/CBZ-CCBG-CCEP-CCI-CCK-CCL-CCOI-CCRN-CCU-CDE"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/data_quality/alpaca_5m_symbol_year_finalisation_benchmark"),
    )
    args = parser.parse_args()
    files = sorted(path / "bars.parquet" for path in args.batch_root.iterdir() if path.name.startswith(str(args.year)))[: args.max_files]
    if not files:
        raise ValueError("no benchmark source files found")
    chunks = [_source_chunk(path) for path in files]
    registry = _registry(Path("config/universes/alpaca_514_symbols.txt"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    before = _measure(
        lambda: _legacy_partition(args.symbol, args.year, chunks, args.output_root / "before", registry),
        files,
    )
    after = _measure(
        lambda: finalize_partition(
            canonical_symbol=args.symbol,
            year=args.year,
            chunks=chunks,
            archive_root=args.output_root / "after",
            quarantine_root=args.output_root / "after_conflicts",
            registry=registry,
        ),
        files,
    )
    before_path = Path(before["result"]["path"])
    after_path = Path(after["result"]["path"])
    before_rows = pq.read_table(before_path).to_pylist()
    after_rows = pq.read_table(after_path).to_pylist()
    report = {
        "symbol": args.symbol,
        "year": args.year,
        "source_files": [str(path) for path in files],
        "before": before,
        "after": after,
        "output_checksum_equality": before["result"]["output_file_hash"] == after["result"]["output_file_hash"],
        "output_row_equality": before_rows == after_rows,
        "partition_validation_equality": validate_rows(before_rows) == validate_rows(after_rows),
    }
    (args.output_root / "benchmark.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


def _measure(operation, files: list[Path]) -> dict:
    tracemalloc.start()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "elapsed_seconds": elapsed,
        "source_files_opened": len(files),
        "source_bytes": sum(path.stat().st_size for path in files),
        "peak_python_memory_bytes": peak,
        "result": result,
    }


def _legacy_partition(symbol: str, year: int, chunks: list[SourceChunk], archive_root: Path, registry: dict) -> dict:
    rows = []
    source_rows = 0
    read_seconds = 0.0
    filter_seconds = 0.0
    for chunk in chunks:
        started = time.perf_counter()
        table = pq.ParquetFile(chunk.parquet_path).read()
        read_seconds += time.perf_counter() - started
        source_rows += table.num_rows
        started = time.perf_counter()
        for row in _iter_table_rows(table):
            normalized = normalize_output_row(row, chunk=chunk, registry=registry)
            if normalized["canonical_symbol"] == symbol and normalized["timestamp_utc"].year == year:
                rows.append(normalized)
        filter_seconds += time.perf_counter() - started
    started = time.perf_counter()
    deduped, duplicates = deduplicate_rows(rows)
    dedup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    invalid = validate_rows(deduped)
    validation_seconds = time.perf_counter() - started
    if invalid or duplicates["conflicting_duplicate_count"]:
        raise ValueError("legacy benchmark partition failed validation")
    target = archive_root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pq.write_table(pa.Table.from_pylist(deduped, schema=OUTPUT_SCHEMA), target, compression="zstd")
    write_seconds = time.perf_counter() - started
    return {
        "path": str(target),
        "source_row_count": source_rows,
        "rows_decoded": source_rows,
        "output_row_count": len(deduped),
        "read_amplification_ratio": source_rows / max(1, len(deduped)),
        "decoded_row_amplification_ratio": source_rows / max(1, len(deduped)),
        "output_file_hash": _file_sha256(target),
        "phase_timings": {
            "parquet_read_seconds": read_seconds,
            "filter_normalize_seconds": filter_seconds,
            "sort_deduplication_seconds": dedup_seconds,
            "validation_seconds": validation_seconds,
            "parquet_write_seconds": write_seconds,
        },
    }


def _source_chunk(parquet_path: Path) -> SourceChunk:
    relative = parquet_path.relative_to(Path("data/processed/alpaca/stock_bars_parquet"))
    raw_path = Path("data/raw/alpaca/stock_bars") / relative.parent
    manifest = json.loads((raw_path / "manifest.json").read_text(encoding="utf-8"))
    return SourceChunk(
        chunk_id=chunk_id_from_manifest(manifest),
        raw_path=raw_path,
        parquet_path=parquet_path,
        manifest=manifest,
        reason="bounded_benchmark",
    )


if __name__ == "__main__":
    raise SystemExit(main())
