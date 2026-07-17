from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import tempfile
import time
from pathlib import Path
from typing import Any
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.reference.daily_stock_spine import verify_and_register


PROFILE_CONTRACT = "daily_spine_production_shaped_profile.v1"
DEFAULT_BASE = Path(
    "reports/ml/development/ticket_7b3_daily_large_history/"
    "regeneration_canonical_v2/benchmark/stock_level_prediction_artifacts.parquet"
)
DEFAULT_ENRICHED = Path(
    "reports/ml/development/ticket_7b3_daily_large_history/"
    "regeneration_canonical_v2/benchmark/"
    "stock_level_prediction_artifacts_enriched.parquet"
)


def build_bounded_sample(
    *,
    base_source: Path,
    enriched_source: Path,
    output_root: Path,
    rows: int,
) -> dict[str, Any]:
    if rows < 3:
        raise ValueError("sample rows must be at least three")
    base_pf = pq.ParquetFile(base_source)
    enriched_pf = pq.ParquetFile(enriched_source)
    if base_pf.metadata.num_rows != enriched_pf.metadata.num_rows:
        raise ValueError("base and enriched populations must have equal row counts")
    starts = _row_group_starts(base_pf)
    selected_base = sorted({0, len(starts) // 2, len(starts) - 1})
    positions = [starts[index] for index in selected_base]
    allocations = _allocate(rows, len(positions))
    base_tables, enriched_tables = [], []
    base_groups, enriched_groups = set(), set()
    for position, count in zip(positions, allocations):
        base_table, touched = _read_bounded_range(base_pf, position, count)
        enriched_table, enriched_touched = _read_bounded_range(enriched_pf, position, count)
        base_tables.append(base_table)
        enriched_tables.append(enriched_table)
        base_groups.update(touched)
        enriched_groups.update(enriched_touched)
    base_sample = pa.concat_tables(base_tables)
    enriched_sample = pa.concat_tables(enriched_tables)
    if base_sample.num_rows != rows or enriched_sample.num_rows != rows:
        raise RuntimeError("bounded sample did not produce requested population")
    sample_dir = output_root / f"rows={rows}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    base_output = sample_dir / "base.production_shaped.parquet"
    enriched_output = sample_dir / "enriched.production_shaped.parquet"
    pq.write_table(base_sample, base_output, row_group_size=max(1, rows // 6))
    pq.write_table(enriched_sample, enriched_output, row_group_size=max(1, rows // 6))
    manifest = {
        "contract_version": PROFILE_CONTRACT,
        "status": "BOUNDED_BENCHMARK_SAMPLE",
        "production_certification_evidence": False,
        "requested_rows": rows,
        "sample_positions": positions,
        "matched_population_strategy": "same_global_ordinal_ranges",
        "physical_order_match_required_for_certification": False,
        "sample_row_key_overlap": _key_overlap(base_sample, enriched_sample),
        "base": _sample_record(base_source, base_output, base_pf, sorted(base_groups)),
        "enriched": _sample_record(
            enriched_source, enriched_output, enriched_pf, sorted(enriched_groups),
        ),
    }
    manifest_path = sample_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def profile_sample(
    *,
    sample_manifest: Path,
    registry: Path,
    aliases: Path,
    workers: int,
    scratch_root: Path,
) -> dict[str, Any]:
    manifest = json.loads(sample_manifest.read_text())
    report_root = scratch_root / f"profile-w{workers}"
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = verify_and_register(
        base_artifact=Path(manifest["base"]["sample_path"]),
        enriched_artifact=Path(manifest["enriched"]["sample_path"]),
        registry=registry,
        aliases=aliases,
        report_root=report_root,
        verify_only=True,
        dry_run=True,
        max_workers=workers,
        stream_temp_root=scratch_root / "sqlite",
        heartbeat_seconds=3600,
    )
    profiler.disable()
    elapsed = time.perf_counter() - started
    sections = _profile_sections(profiler, elapsed, manifest["requested_rows"] * 2)
    return {
        "contract_version": PROFILE_CONTRACT,
        "sample_manifest": str(sample_manifest),
        "workers": workers,
        "elapsed_seconds": elapsed,
        "status": result["status"],
        "blockers": result["blockers"],
        "peak_working_set_memory_bytes": result["streaming_diagnostics"][
            "peak_working_set_memory_bytes"
        ],
        "scratch_bytes": result["streaming_diagnostics"]["maximum_temporary_bytes"],
        "logical_output_checksum": result["logical_output_checksum"],
        "base_population_checksum": result["base_artifact"]["row_population_checksum"],
        "enriched_population_checksum": result["enriched_artifact"]["row_population_checksum"],
        "spine_dataset_id": result["spine_dataset_id"],
        "price_feature_dataset_id": result["price_feature_dataset_id"],
        "timing_sections": sections,
        "hot_functions": _hot_functions(
            profiler, elapsed, manifest["requested_rows"] * 2,
        ),
    }


def _profile_sections(profiler: cProfile.Profile, elapsed: float, rows: int) -> list[dict]:
    stats = pstats.Stats(profiler, stream=io.StringIO())
    mappings = {
        "source_checksum_calculation": ("file_sha256", "_file_sha256"),
        "parquet_metadata_resolution": ("ParquetFile",),
        "arrow_batch_decoding": ("iter_batches",),
        "column_to_python_projection": ("_columnar_rows", "to_pydict"),
        "timestamp_parsing": ("_parse_temporal_dt_strict", "_parse_dt"),
        "target_temporal_contract_validation": ("_stream_temporal",),
        "symbol_normalisation": ("_date_value", "_timestamp_value"),
        "registry_lookup": ("_resolve_symbols",),
        "canonical_row_id_construction": ("daily_spine_row_id",),
        "sha_hash_operations": ("_hash_json", "openssl_sha256"),
        "sqlite_batch_preparation": ("_augment_row",),
        "sqlite_insertion": ("executemany",),
        "index_construction": ("execute",),
        "duplicate_checks": ("_stream_alignment",),
        "base_enriched_alignment": ("_stream_alignment",),
        "mismatch_queries": ("_stream_target_alignment",),
        "ordered_population_checksums": ("_stream_identities",),
        "dataset_identity_calculation": ("_stream_identities",),
        "dataset_identity_sha": ("_dataset_id_from_rows",),
        "report_creation": ("write_verification_reports",),
        "certification_publication_or_lookup": (
            "build_certification_identity", "load_ready_certification",
            "publish_ready_certification",
        ),
    }
    entries = list(stats.stats.items())
    output = []
    for section, needles in mappings.items():
        calls = 0
        cpu = 0.0
        for (filename, _line, function), (primitive, total, own_time, _cumulative, _callers) in entries:
            label = f"{filename}:{function}"
            if any(needle in label for needle in needles):
                calls += total
                cpu += own_time
        output.append({
            "section": section,
            "call_count": calls,
            "elapsed_seconds": cpu,
            "cpu_seconds": cpu,
            "rows_processed": rows,
            "rows_per_second": rows / cpu if cpu else None,
            "percentage_of_total_runtime": (cpu / elapsed * 100.0) if elapsed else 0.0,
        })
    return output


def _hot_functions(profiler: cProfile.Profile, elapsed: float, rows: int) -> list[dict]:
    stats = pstats.Stats(profiler, stream=io.StringIO())
    ordered = sorted(
        stats.stats.items(), key=lambda item: item[1][2], reverse=True,
    )[:25]
    return [
        {
            "function": f"{Path(key[0]).name}:{key[2]}",
            "call_count": values[1],
            "cpu_seconds": values[2],
            "rows_per_second": rows / values[2] if values[2] else None,
            "percentage_of_total_runtime": values[2] / elapsed * 100.0,
        }
        for key, values in ordered
    ]


def _read_bounded_range(
    parquet: pq.ParquetFile, start: int, count: int,
) -> tuple[pa.Table, list[int]]:
    starts = _row_group_starts(parquet)
    remaining = count
    position = start
    tables = []
    touched = []
    while remaining:
        group = max(index for index, value in enumerate(starts) if value <= position)
        offset = position - starts[group]
        available = parquet.metadata.row_group(group).num_rows - offset
        take = min(remaining, available)
        batches = parquet.iter_batches(
            row_groups=[group], batch_size=max(1, offset + take),
            use_threads=True,
        )
        table = pa.Table.from_batches([next(batches)]).slice(offset, take)
        tables.append(table)
        touched.append(group)
        position += take
        remaining -= take
    return pa.concat_tables(tables), touched


def _row_group_starts(parquet: pq.ParquetFile) -> list[int]:
    starts, position = [], 0
    for index in range(parquet.metadata.num_row_groups):
        starts.append(position)
        position += parquet.metadata.row_group(index).num_rows
    return starts


def _allocate(total: int, buckets: int) -> list[int]:
    values = [total // buckets] * buckets
    for index in range(total % buckets):
        values[index] += 1
    return values


def _key_overlap(base_sample: pa.Table, enriched_sample: pa.Table) -> dict[str, Any]:
    if not {"rebalance_date", "symbol"}.issubset(base_sample.column_names) or not {
        "rebalance_date", "symbol",
    }.issubset(enriched_sample.column_names):
        return {"key_fields": ["rebalance_date", "symbol"], "overlap_count": None}
    base_keys = {
        (str(row["rebalance_date"])[:10], str(row["symbol"]).upper())
        for row in base_sample.select(["rebalance_date", "symbol"]).to_pylist()
    }
    enriched_keys = {
        (str(row["rebalance_date"])[:10], str(row["symbol"]).upper())
        for row in enriched_sample.select(["rebalance_date", "symbol"]).to_pylist()
    }
    return {
        "key_fields": ["rebalance_date", "symbol"],
        "base_unique_key_count": len(base_keys),
        "enriched_unique_key_count": len(enriched_keys),
        "overlap_count": len(base_keys & enriched_keys),
    }


def _sample_record(source: Path, sample: Path, parquet, groups: list[int]) -> dict:
    metadata_payload = {
        "rows": parquet.metadata.num_rows,
        "schema": str(parquet.schema_arrow),
        "selected_row_groups": [
            {"index": index, "rows": parquet.metadata.row_group(index).num_rows}
            for index in groups
        ],
    }
    return {
        "source_path": str(source),
        "source_metadata_checksum": hashlib.sha256(
            json.dumps(metadata_payload, sort_keys=True).encode()
        ).hexdigest(),
        "selected_row_groups": groups,
        "sample_path": str(sample),
        "sample_rows": pq.ParquetFile(sample).metadata.num_rows,
        "sample_sha256": _file_sha256(sample),
        "schema_preserved": pq.ParquetFile(sample).schema_arrow.equals(parquet.schema_arrow),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--enriched-source", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rows", type=int, action="append", default=[])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = args.output_root or Path(tempfile.mkdtemp(prefix="spine-production-shaped-"))
    manifests = [
        build_bounded_sample(
            base_source=args.base_source, enriched_source=args.enriched_source,
            output_root=root, rows=rows,
        )
        for rows in (args.rows or [10_000, 50_000, 100_000])
    ]
    output: dict[str, Any] = {"samples": manifests}
    if args.profile:
        if not args.registry or not args.aliases:
            parser.error("--registry and --aliases are required with --profile")
        output["profiles"] = [
            profile_sample(
                sample_manifest=Path(manifest["manifest_path"]),
                registry=args.registry, aliases=args.aliases, workers=workers,
                scratch_root=root / f"scratch-{manifest['requested_rows']}-w{workers}",
            )
            for manifest in manifests for workers in (1, 3, 6)
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
