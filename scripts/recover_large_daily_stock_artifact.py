from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.research.ml.stock_level.prediction_artifacts.targets import _add_cross_sectional_targets
from core.research.ml.stock_level.stock_level_alpha_features import write_stock_level_alpha_features
from core.research.ml.stock_level.stock_level_artifact_io import (
    artifact_identity,
    file_sha256,
    read_stock_level_artifact,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    CONTEXT_COLUMNS,
    PREDICTION_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
)


REPORT_ROOT = Path("reports/data_lineage/daily_stock_artifact_recovery")
REQUIRED_COLUMNS = {
    "rebalance_date",
    "symbol",
    "decision_timestamp",
    "feature_data_cutoff_timestamp",
    "actual_forward_return_10d",
    "actual_benchmark_return_10d",
    "target_horizon_trading_days",
    "target_provenance_contract_version",
}
FINAL_BASE_NAME = "stock_level_prediction_artifacts.parquet"
FINAL_ENRICHED_NAME = "stock_level_prediction_artifacts_enriched.parquet"
MAX_REPORT_ROWS = 1000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover an interrupted large daily stock artifact from symbol partitions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--audit-partitions", action="store_true")
    parser.add_argument("--resume-missing-partitions", action="store_true")
    parser.add_argument("--finalize-from-partitions", action="store_true")
    parser.add_argument("--generate-enriched-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not any([args.audit_partitions, args.resume_missing_partitions, args.finalize_from_partitions, args.generate_enriched_only]):
        raise SystemExit("Select one mode: --audit-partitions, --resume-missing-partitions, --finalize-from-partitions, or --generate-enriched-only")
    if args.resume_missing_partitions:
        raise SystemExit("--resume-missing-partitions audit support is implemented; bounded recomputation is not automatic in this script")
    result = run_recovery(
        config_path=args.config,
        run_dir=args.run_dir,
        report_root=args.report_root,
        audit_partitions=args.audit_partitions,
        finalize_from_partitions=args.finalize_from_partitions,
        generate_enriched_only=args.generate_enriched_only,
        dry_run=args.dry_run,
    )
    print(json.dumps(_summary(result), indent=2, sort_keys=True, default=str))
    return 0


def run_recovery(
    *,
    config_path: Path,
    run_dir: Path,
    report_root: Path = REPORT_ROOT,
    audit_partitions: bool = False,
    finalize_from_partitions: bool = False,
    generate_enriched_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not run_dir:
        raise ValueError("explicit run_dir is required")
    config = _read_yaml(config_path)
    partition_root = _partition_root(config, run_dir)
    expected_symbols = _expected_symbols(config)
    diagnostics = _diagnostics(run_dir / "stock_artifact_symbol_tasks.jsonl")
    inventory = audit_partitions_for_run(partition_root=partition_root, expected_symbols=expected_symbols)
    audit = _partition_audit_payload(config_path, run_dir, partition_root, expected_symbols, diagnostics, inventory)
    if not dry_run:
        write_recovery_reports(audit, inventory, report_root=report_root)

    finalization = None
    enriched = None
    if finalize_from_partitions:
        if dry_run:
            finalization = _dry_run_finalization(run_dir, inventory)
        elif audit["recovery_decision"] != "ALL PARTITIONS REUSABLE":
            finalization = {"status": "BLOCKED", "reason": "partitions_not_all_reusable"}
        else:
            finalization = finalize_base_from_partitions(config=config, run_dir=run_dir, inventory=inventory, report_root=report_root)
    if generate_enriched_only:
        if dry_run:
            enriched = {"status": "DRY_RUN", "expected_path": str(run_dir / FINAL_ENRICHED_NAME)}
        else:
            enriched = generate_enriched_artifact(config=config, run_dir=run_dir, report_root=report_root)
    elapsed = time.perf_counter() - started
    recovery_manifest = {
        "schema_version": "daily_stock_artifact_recovery.v1",
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "partition_root": str(partition_root),
        "audit": audit,
        "finalization": finalization,
        "enriched": enriched,
        "dry_run": dry_run,
        "elapsed_seconds": elapsed,
        "code_commit": _git_commit(),
        "config_hash": _hash_json(config),
        "source_data_modified": False,
        "full_recomputation_triggered": False,
    }
    if not dry_run:
        _write_json(report_root / "recovery_manifest.json", recovery_manifest)
    return recovery_manifest


def audit_partitions_for_run(*, partition_root: Path, expected_symbols: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    expected_date_count = None
    schema_hash = None
    for index, symbol in enumerate(expected_symbols, start=1):
        path = partition_root / f"{symbol}.json"
        row = inspect_partition(path, symbol=symbol)
        if row["status"] == "VALID_COMPLETE":
            expected_date_count = expected_date_count or row["decision_date_count"]
            schema_hash = schema_hash or row["schema_hash"]
            if row["decision_date_count"] != expected_date_count:
                row["status"] = "DATE_COVERAGE_MISMATCH"
            elif row["schema_hash"] != schema_hash:
                row["status"] = "SCHEMA_MISMATCH"
        rows.append(row)
        if index % 50 == 0:
            print(f"[recovery] audited partitions {index}/{len(expected_symbols)}", flush=True)
    return rows


def inspect_partition(path: Path, *, symbol: str) -> dict[str, Any]:
    base = {
        "symbol": symbol,
        "partition_path": str(path),
        "file_size": path.stat().st_size if path.exists() else 0,
        "row_count": 0,
        "column_count": 0,
        "schema_hash": "",
        "date_min": "",
        "date_max": "",
        "decision_date_count": 0,
        "target_horizon_values": "",
        "checksum": "",
        "read_status": "",
        "duplicate_candidate_count": 0,
        "required_column_status": "",
        "status": "UNKNOWN",
    }
    if not path.exists():
        return {**base, "read_status": "missing", "status": "MISSING"}
    if path.stat().st_size <= 0:
        return {**base, "read_status": "empty", "status": "EMPTY"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {**base, "read_status": f"unreadable:{type(exc).__name__}", "status": "UNREADABLE"}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {**base, "read_status": "rows_not_list", "status": "UNREADABLE"}
    if not rows:
        return {**base, "read_status": "empty_rows", "status": "EMPTY"}
    columns = sorted({key for row in rows if isinstance(row, Mapping) for key in row})
    missing_required = sorted(REQUIRED_COLUMNS - set(columns))
    dates = [str(row.get("rebalance_date", "")) for row in rows if isinstance(row, Mapping) and row.get("rebalance_date")]
    horizons = sorted({str(row.get("target_horizon_trading_days") or row.get("target_horizon") or "") for row in rows if isinstance(row, Mapping)})
    keys = [
        (str(row.get("symbol", "")).upper(), str(row.get("decision_timestamp") or row.get("rebalance_date") or ""), str(row.get("target_horizon_trading_days") or ""))
        for row in rows
        if isinstance(row, Mapping)
    ]
    duplicate_count = len(keys) - len(set(keys))
    status = "VALID_COMPLETE"
    if missing_required:
        status = "TARGET_CONTRACT_MISMATCH"
    elif duplicate_count:
        status = "DUPLICATE_ROWS"
    elif not dates:
        status = "DATE_COVERAGE_MISMATCH"
    return {
        **base,
        "file_size": path.stat().st_size,
        "row_count": len(rows),
        "column_count": len(columns),
        "schema_hash": _hash_json(columns),
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
        "decision_date_count": len(set(dates)),
        "target_horizon_values": "|".join(horizons),
        "checksum": file_sha256(path),
        "read_status": "ok",
        "duplicate_candidate_count": duplicate_count,
        "required_column_status": "ok" if not missing_required else "missing:" + "|".join(missing_required),
        "status": status,
    }


def finalize_base_from_partitions(*, config: Mapping[str, Any], run_dir: Path, inventory: Sequence[Mapping[str, Any]], report_root: Path) -> dict[str, Any]:
    output_path = run_dir / FINAL_BASE_NAME
    if output_path.exists():
        started = time.perf_counter()
        rows = read_stock_level_artifact(output_path, required_columns=REQUIRED_COLUMNS)
        identity = _fast_parquet_identity(output_path, rows, list(rows[0]) if rows else [])
        _write_base_audit_files(run_dir, rows, identity)
        output_paths = _base_output_paths(run_dir, output_path)
        _mark_stage_files(run_dir, "stock_artifact", "completed", output_paths)
        report = {
            "status": "EXISTING_VALID",
            "path": str(output_path),
            "identity": identity,
            "row_count": len(rows),
            "symbol_count": len({row["symbol"] for row in rows}),
            "date_min": min(str(row["rebalance_date"]) for row in rows) if rows else "",
            "date_max": max(str(row["rebalance_date"]) for row in rows) if rows else "",
            "elapsed_seconds": time.perf_counter() - started,
            "reused_partition_count": len([row for row in inventory if row["status"] == "VALID_COMPLETE"]),
        }
        _write_json(report_root / "finalization_report.json", report)
        (report_root / "finalization_report.md").write_text(_finalization_markdown(report), encoding="utf-8")
        return report
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    valid = [row for row in inventory if row["status"] == "VALID_COMPLETE"]
    for index, item in enumerate(sorted(valid, key=lambda row: row["symbol"]), start=1):
        payload = json.loads(Path(str(item["partition_path"])).read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        if index % 50 == 0:
            print(f"[recovery] loaded partitions {index}/{len(valid)} rows={len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    if not rows:
        raise ValueError("no partition rows available for finalization")
    rows.sort(key=lambda row: (str(row.get("decision_timestamp") or row.get("rebalance_date") or ""), str(row.get("symbol", "")).upper()))
    _add_cross_sectional_targets(rows)
    _validate_final_rows(rows, expected_symbols=[str(row["symbol"]) for row in valid])
    fieldnames = list(rows[0])
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        _write_recovered_parquet(
            tmp_path,
            rows,
            fieldnames=fieldnames,
            config=config,
        )
        _write_sample_csv(run_dir / "stock_level_prediction_artifacts_sample.csv", rows[:100], fieldnames)
        _validate_written_artifact(tmp_path, expected_rows=len(rows), expected_symbols=len(valid))
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    final_rows = read_stock_level_artifact(output_path, required_columns=REQUIRED_COLUMNS)
    final_identity = _fast_parquet_identity(output_path, final_rows, fieldnames)
    report = {
        "status": "COMPLETED",
        "path": str(output_path),
        "identity": final_identity,
        "row_count": len(final_rows),
        "symbol_count": len({row["symbol"] for row in final_rows}),
        "date_min": min(str(row["rebalance_date"]) for row in final_rows),
        "date_max": max(str(row["rebalance_date"]) for row in final_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "reused_partition_count": len(valid),
    }
    _write_json(report_root / "finalization_report.json", report)
    (report_root / "finalization_report.md").write_text(_finalization_markdown(report), encoding="utf-8")
    _append_recovery_history(run_dir, {"stock_artifact": report})
    _write_base_audit_files(run_dir, rows, final_identity)
    _mark_stage_files(run_dir, "stock_artifact", "completed", _base_output_paths(run_dir, output_path))
    return report


def generate_enriched_artifact(*, config: Mapping[str, Any], run_dir: Path, report_root: Path) -> dict[str, Any]:
    base_path = run_dir / FINAL_BASE_NAME
    if not base_path.exists():
        return {"status": "BLOCKED", "reason": "base_artifact_missing", "path": str(base_path)}
    started = time.perf_counter()
    before = file_sha256(base_path)
    enriched_path = run_dir / FINAL_ENRICHED_NAME
    if enriched_path.exists():
        base_rows = read_stock_level_artifact(base_path, required_columns=REQUIRED_COLUMNS)
        enriched_rows = read_stock_level_artifact(enriched_path, required_columns=REQUIRED_COLUMNS)
        alignment = _base_enriched_alignment(base_rows, enriched_rows)
        if alignment["aligned"]:
            return {"status": "EXISTING_VALID", "path": str(enriched_path), "alignment": alignment}
        raise ValueError(f"existing enriched artifact is not aligned: {alignment}")
    cfg = dict(config)
    cfg["ml"] = dict(cfg.get("ml", {}) or {})
    cfg["ml"]["stock_level_base_prediction_artifacts_path"] = str(base_path)
    cfg["ml"]["output_dir"] = str(run_dir)
    paths = write_stock_level_alpha_features(cfg)
    if file_sha256(base_path) != before:
        raise RuntimeError("base artifact checksum changed during alpha-feature generation")
    base_rows = read_stock_level_artifact(base_path, required_columns=REQUIRED_COLUMNS)
    enriched_rows = read_stock_level_artifact(paths.enriched_parquet_path, required_columns=REQUIRED_COLUMNS)
    alignment = _base_enriched_alignment(base_rows, enriched_rows)
    if not alignment["aligned"]:
        raise ValueError(f"enriched artifact does not preserve base rows: {alignment}")
    report = {
        "status": "COMPLETED",
        "path": str(paths.enriched_parquet_path),
        "file_size_bytes": paths.enriched_parquet_path.stat().st_size,
        "sha256": file_sha256(paths.enriched_parquet_path),
        "row_count": len(enriched_rows),
        "symbol_count": len({row["symbol"] for row in enriched_rows}),
        "date_min": min(str(row["rebalance_date"]) for row in enriched_rows),
        "date_max": max(str(row["rebalance_date"]) for row in enriched_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "alignment": alignment,
    }
    _write_json(report_root / "enriched_generation_report.json", report)
    _append_recovery_history(run_dir, {"alpha_features": report})
    _mark_stage_files(
        run_dir,
        "alpha_features",
        "completed",
        {
            "enriched_parquet_path": str(paths.enriched_parquet_path),
            "audit_csv_path": str(paths.audit_csv_path),
            "audit_json_path": str(paths.audit_json_path),
            "audit_markdown_path": str(paths.audit_markdown_path),
        },
    )
    return report


def write_recovery_reports(audit: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], *, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    _write_csv(report_root / "partition_inventory.csv", inventory, _inventory_fields())
    _write_partition_parquet(report_root / "partition_inventory.parquet", inventory)
    _write_json(report_root / "partition_audit.json", audit)
    (report_root / "partition_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    _write_csv(report_root / "invalid_partitions.csv", [row for row in inventory if row["status"] not in {"VALID_COMPLETE", "MISSING"}], _inventory_fields())
    _write_csv(report_root / "missing_partitions.csv", [row for row in inventory if row["status"] == "MISSING"], _inventory_fields())
    _write_json(report_root / "schema_variants.json", audit["schema_variants"])


def _partition_audit_payload(config_path: Path, run_dir: Path, partition_root: Path, expected_symbols: Sequence[str], diagnostics: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in inventory)
    valid = [row for row in inventory if row["status"] == "VALID_COMPLETE"]
    invalid = [row for row in inventory if row["status"] not in {"VALID_COMPLETE", "MISSING"}]
    missing = [row for row in inventory if row["status"] == "MISSING"]
    schema_variants: dict[str, Any] = defaultdict(list)
    for row in inventory:
        schema_variants[str(row.get("schema_hash", ""))].append(row["symbol"])
    decision = "ALL PARTITIONS REUSABLE" if len(valid) == len(expected_symbols) else ("REUSABLE WITH BOUNDED RECOMPUTATION" if valid else "PARTITIONS NOT REUSABLE")
    return {
        "schema_version": "daily_stock_partition_audit.v1",
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "partition_root": str(partition_root),
        "expected_symbol_count": len(expected_symbols),
        "found_partition_count": sum(1 for row in inventory if Path(str(row["partition_path"])).exists()),
        "valid_partition_count": len(valid),
        "missing_partition_count": len(missing),
        "invalid_partition_count": len(invalid),
        "status_counts": dict(counts),
        "estimated_total_rows": sum(int(row["row_count"]) for row in valid),
        "date_min": min((row["date_min"] for row in valid if row["date_min"]), default=""),
        "date_max": max((row["date_max"] for row in valid if row["date_max"]), default=""),
        "schema_compatibility": "COMPATIBLE" if len([key for key in schema_variants if key]) == 1 and not invalid else "INCOMPATIBLE",
        "schema_variants": {key: {"symbol_count": len(value), "symbols_preview": value[:20]} for key, value in schema_variants.items()},
        "diagnostics": diagnostics,
        "recovery_decision": decision,
        "symbols_to_recompute": [{"symbol": row["symbol"], "reason": row["status"]} for row in inventory if row["status"] != "VALID_COMPLETE"],
        "root_cause_inference": _root_cause(diagnostics),
    }


def _dry_run_finalization(run_dir: Path, inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in inventory if row["status"] == "VALID_COMPLETE"]
    return {
        "status": "DRY_RUN",
        "partitions_to_reuse": len(valid),
        "partitions_to_recompute": len(inventory) - len(valid),
        "expected_output_path": str(run_dir / FINAL_BASE_NAME),
        "expected_symbol_count": len(valid),
        "expected_row_count": sum(int(row["row_count"]) for row in valid),
        "estimated_partition_bytes": sum(int(row["file_size"]) for row in valid),
    }


def _diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    counts = Counter()
    by_event_symbol: dict[str, set[str]] = defaultdict(set)
    last_events = []
    first_ts = last_ts = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = str(row.get("event_type", ""))
            symbol = str(row.get("symbol", "")).upper()
            counts[event] += 1
            if symbol:
                by_event_symbol[event].add(symbol)
            ts = str(row.get("timestamp", ""))
            first_ts = min(first_ts, ts) if first_ts else ts
            last_ts = max(last_ts, ts) if last_ts else ts
            last_events.append(row)
            last_events = last_events[-20:]
    return {
        "exists": True,
        "event_counts": dict(counts),
        "symbol_counts_by_event": {event: len(symbols) for event, symbols in by_event_symbol.items()},
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "last_events": last_events,
    }


def _root_cause(diagnostics: Mapping[str, Any]) -> str:
    completed = int((diagnostics.get("symbol_counts_by_event") or {}).get("completed", 0))
    failed = int((diagnostics.get("symbol_counts_by_event") or {}).get("failed", 0))
    if completed and not failed:
        return "symbol computation completed; interruption occurred after partition writes and before final artifact publication"
    if failed:
        return "symbol task failures observed"
    return "insufficient diagnostics"


def _validate_final_rows(rows: Sequence[Mapping[str, Any]], *, expected_symbols: Sequence[str]) -> None:
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    if symbols != sorted(set(expected_symbols)):
        raise ValueError("final rows symbol set mismatch")
    keys = [(row.get("symbol"), row.get("decision_timestamp") or row.get("rebalance_date"), row.get("target_horizon_trading_days")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate final economic rows")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(f"final rows missing required columns: {missing}")


def _validate_written_artifact(path: Path, *, expected_rows: int, expected_symbols: int) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("temporary artifact row count mismatch")
    table = pq.read_table(path)
    columns = set(table.schema.names)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"temporary artifact missing required columns: {missing}")
    rows = table.to_pylist()
    if len({row["symbol"] for row in rows}) != expected_symbols:
        raise ValueError("temporary artifact symbol count mismatch")


def _write_recovered_parquet(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str], config: Mapping[str, Any]) -> None:
    compression = str((config.get("ml", {}) or {}).get("stock_level_parquet_compression", "zstd")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{name: _normalize_value(row.get(name)) for name in fieldnames} for row in rows]
    table = pa.Table.from_pylist(normalized)
    table = table.select([name for name in fieldnames if name in table.schema.names])
    pq.write_table(table, path, compression=compression)


def _normalize_value(value: Any) -> Any:
    return None if value == "" else value


def _write_sample_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    _write_csv(path, rows, fieldnames)


def _fast_parquet_identity(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    decision_values = [row.get("decision_timestamp") or row.get("rebalance_date") for row in rows]
    target_versions = sorted({str(row.get("target_provenance_contract_version")) for row in rows if row.get("target_provenance_contract_version") not in (None, "")})
    return {
        "artifact_format": "parquet",
        "compression": "zstd",
        "compression_codecs": sorted({
            str(parquet.metadata.row_group(group).column(column).compression)
            for group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.row_group(group).num_columns)
        }),
        "resolved_artifact_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "logical_content_sha256": None,
        "logical_content_hash_skipped_reason": "recovery_fast_path_uses_file_checksum_and_row_identity_audit",
        "schema_fingerprint": _hash_json({"columns": list(fieldnames), "types": {name: str(parquet.schema_arrow.field(name).type) for name in parquet.schema_arrow.names}}),
        "stable_column_order": list(fieldnames),
        "row_count": parquet.metadata.num_rows,
        "column_count": len(fieldnames),
        "symbol_count": len({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")}),
        "decision_date_count": len({str(value)[:10] for value in decision_values if value}),
        "minimum_decision_timestamp": min((str(value) for value in decision_values if value), default=None),
        "maximum_decision_timestamp": max((str(value) for value in decision_values if value), default=None),
        "duplicate_symbol_decision_keys": len(rows) - len({_row_key(row) for row in rows}),
        "null_symbol_count": sum(1 for row in rows if not row.get("symbol")),
        "null_decision_timestamp_count": sum(1 for row in rows if not (row.get("decision_timestamp") or row.get("rebalance_date"))),
        "target_contract_version": target_versions[0] if len(target_versions) == 1 else None,
        "target_contract_versions": target_versions,
        "benchmark_contract_version": "stock_level_benchmark_return_10d_v1",
        "created_at": _utc_now(),
        "completion_status": "complete",
    }


def _base_enriched_alignment(base_rows: Sequence[Mapping[str, Any]], enriched_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ["actual_forward_return_10d", "actual_benchmark_return_10d", "actual_market_residual_return_10d", "target_horizon_trading_days", "target_provenance_contract_version"]
    base_by_key = {_row_key(row): row for row in base_rows}
    enriched_by_key = {_row_key(row): row for row in enriched_rows}
    target_mismatches = 0
    for key, base in base_by_key.items():
        enriched = enriched_by_key.get(key)
        if not enriched:
            continue
        if any(str(base.get(field, "")) != str(enriched.get(field, "")) for field in fields):
            target_mismatches += 1
    return {
        "aligned": set(base_by_key) == set(enriched_by_key) and len(base_rows) == len(enriched_rows) and target_mismatches == 0,
        "base_row_count": len(base_rows),
        "enriched_row_count": len(enriched_rows),
        "base_only_count": len(set(base_by_key) - set(enriched_by_key)),
        "enriched_only_count": len(set(enriched_by_key) - set(base_by_key)),
        "target_mismatch_count": target_mismatches,
    }


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("symbol", "")).upper(), _normalised_timestamp(row.get("decision_timestamp") or row.get("rebalance_date") or ""), str(row.get("target_horizon_trading_days") or ""))


def _normalised_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value)
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_base_audit_files(run_dir: Path, rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]) -> None:
    payload = {
        "row_count": len(rows),
        "symbol_count": len({row["symbol"] for row in rows}),
        "rebalance_date_count": len({row["rebalance_date"] for row in rows}),
        "date_range": [min(str(row["rebalance_date"]) for row in rows), max(str(row["rebalance_date"]) for row in rows)],
        "canonical_artifact": dict(identity),
        "recovered_from_symbol_partitions": True,
        "research_only": True,
        "trading_impact": "none",
    }
    _write_json(run_dir / "stock_level_prediction_artifacts.json", payload)
    (run_dir / "stock_level_prediction_artifacts.md").write_text(
        "\n".join(["# Stock-Level Prediction Artifacts Recovery", "", f"- Rows: {payload['row_count']}", f"- Symbols: {payload['symbol_count']}", f"- Date range: {payload['date_range']}"]),
        encoding="utf-8",
    )


def _append_recovery_history(run_dir: Path, record: Mapping[str, Any]) -> None:
    for name in ("stock_alpha_run_manifest.json", "stock_alpha_run_status.json"):
        path = run_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = list(payload.get("recovery_history", []))
        history.append({"timestamp": _utc_now(), **dict(record)})
        payload["recovery_history"] = history
        payload["original_interruption_preserved"] = True
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _base_output_paths(run_dir: Path, output_path: Path) -> dict[str, str]:
    return {
        "parquet_path": str(output_path),
        "json_path": str(run_dir / "stock_level_prediction_artifacts.json"),
        "markdown_path": str(run_dir / "stock_level_prediction_artifacts.md"),
    }


def _mark_stage_files(run_dir: Path, stage_name: str, status: str, output_paths: Mapping[str, Any]) -> None:
    for name in ("stock_alpha_run_manifest.json", "stock_alpha_run_status.json"):
        _mark_stage(run_dir / name, stage_name, status, output_paths)


def _mark_stage(path: Path, stage_name: str, status: str, output_paths: Mapping[str, Any]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for stage in payload.get("stages", []):
        if stage.get("name") == stage_name:
            stage["status"] = status
            stage["ended_at"] = _utc_now()
            stage["output_paths"] = dict(output_paths)
            stage["output_path_exists"] = {key: Path(str(value)).exists() for key, value in output_paths.items()}
            stage["all_outputs_exist"] = all(stage["output_path_exists"].values())
            stage["any_output_exists"] = any(stage["output_path_exists"].values())
    payload["updated_at"] = _utc_now()
    payload["completed_stages"] = [stage["name"] for stage in payload.get("stages", []) if stage.get("status") == "completed"]
    payload["missing_stages"] = [stage["name"] for stage in payload.get("stages", []) if stage.get("status") in {"pending", "interrupted"}]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _partition_root(config: Mapping[str, Any], run_dir: Path) -> Path:
    ml = dict(config.get("ml", {}) or {})
    return Path(str(ml.get("stock_level_dataset_partition_dir", run_dir / "stock_artifact_symbol_partitions")))


def _expected_symbols(config: Mapping[str, Any]) -> list[str]:
    ml = dict(config.get("ml", {}) or {})
    paths = ml.get("stock_alpha_artifact_universe_paths") or []
    symbols = []
    for raw_path in paths:
        path = Path(str(raw_path))
        payload = _read_yaml(path)
        symbols.extend(str(symbol).upper() for symbol in payload.get("symbols", []) if str(symbol).strip())
    return sorted(dict.fromkeys(symbols))


def _inventory_fields() -> list[str]:
    return [
        "symbol",
        "partition_path",
        "file_size",
        "row_count",
        "column_count",
        "schema_hash",
        "date_min",
        "date_max",
        "decision_date_count",
        "target_horizon_values",
        "checksum",
        "read_status",
        "duplicate_candidate_count",
        "required_column_status",
        "status",
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_partition_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    schema = pa.schema([(name, pa.string()) for name in _inventory_fields()])
    table = pa.Table.from_pylist([{key: str(row.get(key, "")) for key in _inventory_fields()} for row in rows], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _audit_markdown(audit: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Daily Stock Artifact Partition Audit",
        "",
        f"- Expected symbols: {audit['expected_symbol_count']}",
        f"- Found partitions: {audit['found_partition_count']}",
        f"- Valid partitions: {audit['valid_partition_count']}",
        f"- Missing partitions: {audit['missing_partition_count']}",
        f"- Invalid partitions: {audit['invalid_partition_count']}",
        f"- Estimated rows: {audit['estimated_total_rows']}",
        f"- Schema compatibility: {audit['schema_compatibility']}",
        f"- Recovery decision: {audit['recovery_decision']}",
        f"- Root cause inference: {audit['root_cause_inference']}",
        "",
    ])


def _finalization_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Daily Stock Artifact Finalization",
        "",
        f"- Status: {report['status']}",
        f"- Path: {report['path']}",
        f"- Rows: {report['row_count']}",
        f"- Symbols: {report['symbol_count']}",
        f"- Date range: {report['date_min']} to {report['date_max']}",
        f"- Checksum: {report['identity']['sha256']}",
        "",
    ])


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    audit = result["audit"]
    finalization = result.get("finalization") or {}
    enriched = result.get("enriched") or {}
    return {
        "recovery_decision": audit["recovery_decision"],
        "expected_symbols": audit["expected_symbol_count"],
        "found_partitions": audit["found_partition_count"],
        "valid_partitions": audit["valid_partition_count"],
        "missing_partitions": audit["missing_partition_count"],
        "invalid_partitions": audit["invalid_partition_count"],
        "schema_compatibility": audit["schema_compatibility"],
        "finalization_status": finalization.get("status"),
        "enriched_status": enriched.get("status"),
        "dry_run": result["dry_run"],
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
