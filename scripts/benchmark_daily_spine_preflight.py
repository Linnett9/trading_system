from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import tracemalloc
from pathlib import Path
import sys
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.reference.canonical_assets import (
    build_registry_from_universe,
    file_sha256,
    write_registry_outputs,
)
from core.research.ml.reference.daily_stock_spine import verify_and_register
from core.research.ml.reference.daily_stock_spine_certification import certification_path
from core.research.ml.stock_level.stock_level_artifact_io import iter_stock_level_artifact_batches


def benchmark(rows: int = 5000) -> dict:
    if rows < 1 or rows > 100_000:
        raise ValueError("rows must be between 1 and 100000")
    with tempfile.TemporaryDirectory(prefix="daily-spine-benchmark-") as temporary:
        root = Path(temporary)
        fixture = _fixture(root, rows)
        runs = [
            _run("reference", fixture, verify_only=True, workers=1, emulate_prior_base_scan=True),
            _run("optimized_1_worker", fixture, verify_only=True, workers=1),
            _run("optimized_3_workers", fixture, verify_only=True, workers=3),
            _run("optimized_6_workers", fixture, verify_only=True, workers=6),
        ]
        certification_root = root / "certifications"
        runs.append(_run(
            "certification_seed_full", fixture, verify_only=True, workers=3,
            dry_run=False, certification_root=certification_root,
        ))
        runs.append(_run(
            "certification_same_run_cache_hit", fixture, verify_only=True, workers=3,
            certification_root=certification_root,
        ))
        manifest_b = root / "run=B" / "registry_manifest.json"
        manifest_b.parent.mkdir()
        binding_b = json.loads(fixture["registry_manifest"].read_text())
        binding_b["run_id"] = "B"
        manifest_b.write_text(json.dumps(binding_b), encoding="utf-8")
        runs.append(_run(
            "certification_cross_run_cache_hit", fixture, verify_only=True, workers=3,
            certification_root=certification_root, registry_manifest=manifest_b,
            selector_run_id="B",
        ))
        fixture["config"].write_text(
            fixture["config"].read_text() + "  benchmark_content_change: true\n",
            encoding="utf-8",
        )
        changed = _run(
            "certification_content_change_miss", fixture, verify_only=True, workers=3,
            certification_root=certification_root, dry_run=False,
        )
        runs.append(changed)
        certification_path(certification_root, changed["certification_id"]).write_text(
            "{corrupt", encoding="utf-8",
        )
        runs.append(_run(
            "certification_corrupt_fallback", fixture, verify_only=True, workers=3,
            certification_root=certification_root,
        ))
    optimized_checksums = {
        row["logical_output_checksum"] for row in runs if row["name"].startswith("optimized")
    }
    unchanged = [
        row for row in runs
        if row["name"] not in {"certification_content_change_miss", "certification_corrupt_fallback"}
    ]
    changed = [
        row for row in runs
        if row["name"] in {"certification_content_change_miss", "certification_corrupt_fallback"}
    ]
    return {
        "contract_version": "daily_spine_owner_benchmark.v1",
        "synthetic_owner_level_evidence": True,
        "production_speed_claim_allowed": False,
        "production_data_loaded": False,
        "production_workflow_invoked": False,
        "fixture_rows_per_artifact": rows,
        "runs": runs,
        "optimized_logical_checksums_equivalent": len(optimized_checksums) == 1,
        "unchanged_content_semantic_checksums_equivalent": len({
            row["logical_output_checksum"] for row in unchanged
        }) == 1,
        "changed_content_fallback_semantic_checksums_equivalent": len({
            row["logical_output_checksum"] for row in changed
        }) == 1,
        "all_readiness_results_equivalent": len({row["status"] for row in runs}) == 1,
        "unchanged_content_dataset_identities_equivalent": len({
            (row["spine_dataset_id"], row["price_feature_dataset_id"]) for row in unchanged
        }) == 1,
    }


def _run(
    name: str,
    fixture: dict[str, Path],
    *,
    verify_only: bool,
    workers: int,
    emulate_prior_base_scan: bool = False,
    dry_run: bool = True,
    certification_root: Path | None = None,
    registry_manifest: Path | None = None,
    selector_run_id: str = "A",
) -> dict:
    tracemalloc.start()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    if emulate_prior_base_scan:
        for _batch in iter_stock_level_artifact_batches(
            fixture["base"], required_columns=["symbol"], batch_size=512,
        ):
            pass
    result = verify_and_register(
        base_artifact=fixture["base"],
        enriched_artifact=fixture["enriched"],
        registry=fixture["registry"],
        aliases=fixture["aliases"],
        registry_manifest=registry_manifest or fixture["registry_manifest"],
        daily_archive_manifest=fixture["archive_manifest"],
        expected_config=fixture["config"],
        report_root=fixture["root"] / f"report-{name}",
        dry_run=dry_run,
        verify_only=verify_only,
        stream_batch_size=512,
        max_workers=workers,
        heartbeat_seconds=3600,
        certification_root=certification_root,
        selector_run_id=selector_run_id,
    )
    elapsed = time.perf_counter() - started_wall
    cpu = time.process_time() - started_cpu
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    diagnostics = result.get("streaming_diagnostics", {})
    return {
        "name": name,
        "implementation_path": "verify_and_register",
        "status": result["status"],
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu,
        "peak_python_memory_bytes": peak,
        "base_scans": diagnostics.get("source_scan_counts", {}).get("base", 1) + int(emulate_prior_base_scan),
        "enriched_scans": diagnostics.get("source_scan_counts", {}).get("enriched", 1),
        "sqlite_insert_count": diagnostics.get("sqlite_insert_count", 0),
        "python_row_iterations": diagnostics.get("python_row_iterations", 0),
        "checksum_passes": diagnostics.get("checksum_pass_counts", {}),
        "sqlite_query_count": diagnostics.get("sqlite_query_count", 0) + (
            2 if emulate_prior_base_scan else 0
        ),
        "base_row_count": result["base_artifact"]["row_count"],
        "enriched_row_count": result["enriched_artifact"]["row_count"],
        "spine_dataset_id": result["spine_dataset_id"],
        "price_feature_dataset_id": result["price_feature_dataset_id"],
        "logical_output_checksum": _semantic_checksum(result),
        "native_logical_output_checksum": result.get("logical_output_checksum"),
        "worker_count": workers,
        "reference_emulates_removed_symbol_only_scan": emulate_prior_base_scan,
        "certification_cache_hit": result.get("certification_cache_hit", False),
        "certification_id": result.get("certification_id"),
        "certification_miss_reason": result.get("certification_miss_reason"),
        "reuse_validation_elapsed_seconds": result.get("reuse_validation_elapsed_seconds", 0.0),
    }


def _fixture(root: Path, rows: int) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    symbols = [f"S{index:03d}" for index in range(20)]
    universe = root / "universe.txt"
    universe.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    assets, aliases, _ = build_registry_from_universe(universe)
    registry, alias_path = root / "registry.csv", root / "aliases.csv"
    write_registry_outputs(
        assets, aliases, asset_output=registry, alias_output=alias_path, parquet_output=None,
    )
    records = []
    for index in range(rows):
        decision_date = (date(2000, 1, 1) + timedelta(days=index)).isoformat()
        label_date = (date(2000, 1, 1) + timedelta(days=index + 14)).isoformat()
        records.append({
            "rebalance_date": decision_date,
            "symbol": symbols[index % len(symbols)],
            "decision_timestamp": f"{decision_date}T21:00:00Z",
            "feature_data_cutoff_timestamp": f"{decision_date}T20:00:00Z",
            # The target price anchor is a session date; it is intentionally
            # earlier than the same-session decision when parsed as midnight.
            "target_start_timestamp": decision_date,
            "label_start_timestamp": (
                f"{(date(2000, 1, 1) + timedelta(days=index + 1)).isoformat()}T21:00:00Z"
            ),
            "label_end_timestamp": f"{label_date}T21:00:00Z",
            "label_available_timestamp": f"{label_date}T22:00:00Z",
            "target_horizon_trading_days": 10,
            "actual_forward_return_10d": str(index / 100000),
            "actual_benchmark_return_10d": "0.01",
            "actual_market_residual_return_10d": "0.0",
            "target_provenance_contract_version": (
                "stock_level_target_provenance_v2"
            ),
        })
    table = pa.Table.from_pylist(records)
    base, enriched = root / "base.parquet", root / "enriched.parquet"
    pq.write_table(table, base, row_group_size=250)
    pq.write_table(table, enriched, row_group_size=250)
    registry_manifest = root / "run=A" / "registry_manifest.json"
    registry_manifest.parent.mkdir()
    registry_manifest.write_text(json.dumps({
        "status": "READY", "validation_status": "VERIFIED",
        "publication_status": "complete", "run_id": "A",
        "dataset_id": "synthetic-registry", "symbol_registry_version": "synthetic-v1",
        "registry_path": str(registry), "registry_content_checksum": file_sha256(registry),
        "alias_registry_path": str(alias_path),
        "alias_registry_checksum": file_sha256(alias_path),
    }), encoding="utf-8")
    archive = root / "archive_manifest.json"
    archive.write_text(json.dumps({
        "status": "COMPLETE", "row_count": rows, "symbol_count": 514,
        "dataset_root": "data/processed/market_data/canonical_daily_v2/full",
        "dataset_logical_partition_hash": "synthetic-archive",
    }), encoding="utf-8")
    config = root / "config.yaml"
    config.write_text(
        "ml:\n  historical_data_provider: canonical_daily_v2\n"
        "  stooq_parquet_dir: data/processed/market_data/canonical_daily_v2/full\n",
        encoding="utf-8",
    )
    return {
        "root": root, "base": base, "enriched": enriched, "registry": registry,
        "aliases": alias_path, "registry_manifest": registry_manifest,
        "archive_manifest": archive, "config": config,
    }


def _semantic_checksum(result: dict) -> str:
    payload = {
        key: result.get(key)
        for key in (
            "status", "blockers", "spine_dataset_id", "price_feature_dataset_id",
            "row_grain", "alignment", "target_alignment",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.rows)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
