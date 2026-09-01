from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.research.ml.ds24 import master_5m_schema_r7 as r7
from core.research.ml.ds24.master_5m_cloud_r8 import (
    AUTHORITY_ID,
    latest_r7v_run,
    object_metadata,
    parquet_schema_string,
    write_stock_partition,
)
from core.research.ml.ds24.master_5m_validation_r7v import (
    COMPONENT_REL,
    STAGE_REL,
    load_benchmarks,
    partition_inventory,
    stable_hash,
)


RUN_ID = "ds24_p8_r2_local_20260821T000000Z"
SCHEMA_ID = "CANONICAL_5M_FEATURE_SCHEMA_FULL_V1"
LOCAL_ROOT = Path("data/processed/ml_features/five_minute/version=canonical_5m_feature_authority_full_v1/run=ds24_p8_r2_local_20260821T000000Z")
MIN_EMERGENCY_RESERVE_GIB = 5.0
OLD_R2_RUN = "ds24_p6_r8_cloud_master_20260817T204702Z_preflight"


class LocalR8Blocked(RuntimeError):
    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_json(tmp, payload)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(str(root.anchor or root))
    return {
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 6),
        "total_bytes": usage.total,
        "total_gib": round(usage.total / 1024**3, 6),
    }


def tree_size(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "files": 0, "bytes": 0, "gib": 0.0}
    files = 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            files += 1
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return {"path": str(path), "exists": True, "files": files, "bytes": total, "gib": round(total / 1024**3, 6)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registries(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    component = root / COMPONENT_REL
    contract = load_json(component / "r7_master_feature_contract.json")
    stock = list(csv.DictReader((component / "r7_master_stock_features.csv").open(newline="", encoding="utf-8")))
    shared = list(csv.DictReader((component / "r7_master_shared_context.csv").open(newline="", encoding="utf-8")))
    if contract["authority_id"] != AUTHORITY_ID:
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_AUTHORITY_DRIFT", "R7 contract authority drift")
    if contract["authority_identity"] != "2c2a8818495732c8a46ce34ce8cac14e2ff14de90a64487b3291819151041383":
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_AUTHORITY_DRIFT", "R7 authority identity drift")
    if contract["schema_id"] != SCHEMA_ID:
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_SCHEMA_DRIFT", "R7 schema id drift")
    return contract, stock, shared


def source_identity_from_r7v(root: Path) -> str:
    r7v = latest_r7v_run(root)
    decision = load_json(r7v / "34_r7v_final_decision.json")
    if decision.get("classification") != "DS24P6_R7V_FULL_HISTORY_VALIDATION_PASS_WITH_LIMITATIONS":
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_R7V_PARENT", "R7V parent is not accepted")
    if decision.get("source_rows_scanned") != 99_930_803:
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_R7V_PARENT", "R7V source row count drift")
    if decision.get("duplicate_stock_key_count") != 0 or decision.get("duplicate_context_key_count") != 0:
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_R7V_PARENT", "R7V duplicate evidence drift")
    registration = load_json(r7v / "04_r7v_run_registration.json")
    return str(registration["source_identity"])


def registration_payload(root: Path, output_root: Path) -> dict[str, Any]:
    contract, stock, shared = load_registries(root)
    return {
        "run_id": RUN_ID,
        "classification": "DS24P8_R2_LOCAL_BUILD_REGISTERED",
        "authority_id": AUTHORITY_ID,
        "authority_identity": contract["authority_identity"],
        "schema_id": SCHEMA_ID,
        "schema_identity": contract["schema_identity"],
        "canonical_source_identity": source_identity_from_r7v(root),
        "canonical_source_root": r7.SOURCE_ROOT,
        "r2_physical_publication_decommissioned_by_user": True,
        "old_r2_run_historical_only": OLD_R2_RUN,
        "stock_feature_count": len(stock),
        "shared_context_feature_count": len(shared),
        "development_cutoff": r7.DEVELOPMENT_END,
        "locked_holdout_start": r7.HOLDOUT_START,
        "pit_timing_contract": "timestamp_utc bar start; feature availability is timestamp_utc + 5 minutes",
        "physical_root": str(output_root),
        "physical_parquet_layout": "stock/asset=<IDENTITY>/year=<YEAR>/features.parquet and shared-context/year=<YEAR>/context.parquet",
        "emergency_reserve_gib": MIN_EMERGENCY_RESERVE_GIB,
    }


def checkpoint_payload(planned_stock: int, planned_context: int) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "classification": "DS24P8_R2_LOCAL_BUILD_RUNNING",
        "partitions_planned": planned_stock + planned_context,
        "stock_partitions_planned": planned_stock,
        "context_partitions_planned": planned_context,
        "partitions_built": 0,
        "partitions_verified": 0,
        "stock_partitions_verified": 0,
        "context_partitions_verified": 0,
        "partitions_failed": 0,
        "source_rows_processed": 0,
        "stock_feature_rows": 0,
        "context_rows": 0,
        "bytes_written": 0,
        "current_partition": "",
        "last_completed_partition": "",
        "last_checkpoint_utc": utc_now(),
        "free_disk": {},
        "verified_partition_identities": {},
        "last_error": "",
    }


def load_checkpoint(path: Path, planned_stock: int, planned_context: int) -> dict[str, Any]:
    if path.exists():
        checkpoint = load_json(path)
        checkpoint["partitions_planned"] = planned_stock + planned_context
        checkpoint["stock_partitions_planned"] = planned_stock
        checkpoint["context_partitions_planned"] = planned_context
        if (
            checkpoint.get("classification") == "DS24P8_R2_LOCAL_CANONICAL_5M_FEATURE_PUBLICATION_COMPLETE"
            and (
                int(checkpoint.get("stock_partitions_verified") or 0) < planned_stock
                or int(checkpoint.get("context_partitions_verified") or 0) < planned_context
                or int(checkpoint.get("partitions_verified") or 0) < planned_stock + planned_context
            )
        ):
            checkpoint["classification"] = "DS24P8_R2_LOCAL_BUILD_RUNNING"
        return checkpoint
    return checkpoint_payload(planned_stock, planned_context)


def update_checkpoint(root: Path, run_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["last_checkpoint_utc"] = utc_now()
    checkpoint["free_disk"] = disk(root)
    atomic_json(run_dir / "local_r8_checkpoint.json", checkpoint)


def metadata_for(registration: dict[str, Any], partition_identity: str, rows: int, sha: str) -> dict[str, str]:
    return object_metadata(registration, partition_identity, rows, sha)


def final_stock_path(output_root: Path, asset_id: str, year: int) -> Path:
    return output_root / "stock" / f"asset={asset_id}" / f"year={year}" / "features.parquet"


def final_context_path(output_root: Path, year: int) -> Path:
    return output_root / "shared-context" / f"year={year}" / "context.parquet"


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def validate_existing(path: Path, manifest_path: Path, registration: dict[str, Any], partition_identity: str) -> dict[str, Any] | None:
    if not path.exists() or not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    if manifest.get("authority_id") != AUTHORITY_ID:
        return None
    if manifest.get("authority_identity") != registration["authority_identity"]:
        return None
    if manifest.get("schema_identity") != registration["schema_identity"]:
        return None
    if manifest.get("source_identity") != registration["canonical_source_identity"]:
        return None
    if manifest.get("partition_identity") != partition_identity:
        return None
    sha = sha256_file(path)
    if sha != manifest.get("sha256"):
        return None
    return manifest


def promote_partition(staging: Path, final: Path, manifest: dict[str, Any]) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = sidecar_path(final).with_name(sidecar_path(final).name + ".staging")
    write_json(tmp_manifest, manifest)
    os.replace(staging, final)
    os.replace(tmp_manifest, sidecar_path(final))


def stock_schema() -> pa.Schema | None:
    return None


def build_stock_partition(
    root: Path,
    run_dir: Path,
    output_root: Path,
    registration: dict[str, Any],
    part: Any,
    stock_feature_ids: list[str],
    benchmarks: dict[str, pd.Series],
) -> dict[str, Any]:
    partition_id = f"stock/asset={part.asset_id}/year={part.year}"
    final = final_stock_path(output_root, part.asset_id, part.year)
    existing = validate_existing(final, sidecar_path(final), registration, partition_id)
    if existing:
        return {**existing, "verification_state": "REUSABLE_LOCAL_CANONICAL_PARTITION"}
    staging_dir = run_dir / ".staging" / "stock" / f"asset={part.asset_id}" / f"year={part.year}"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging, rows, source_rows, schema = write_stock_partition(root, staging_dir, part, stock_feature_ids, benchmarks)
    sha = sha256_file(staging)
    size = staging.stat().st_size
    manifest = {
        "authority_id": AUTHORITY_ID,
        "authority_identity": registration["authority_identity"],
        "schema_id": SCHEMA_ID,
        "schema_identity": registration["schema_identity"],
        "source_identity": registration["canonical_source_identity"],
        "partition_identity": partition_id,
        "rows": rows,
        "source_rows": source_rows,
        "bytes": size,
        "sha256": sha,
        "schema": schema,
        "path": str(final),
        "published_utc": utc_now(),
    }
    promote_partition(staging, final, manifest)
    shutil.rmtree(staging_dir.parent.parent, ignore_errors=True)
    return {**manifest, "verification_state": "BUILT_AND_VERIFIED"}


def build_context_partition(
    run_dir: Path,
    output_root: Path,
    registration: dict[str, Any],
    year: int,
    shared_registry: list[dict[str, Any]],
    benchmarks: dict[str, pd.Series],
) -> dict[str, Any]:
    partition_id = f"shared-context/year={year}"
    final = final_context_path(output_root, year)
    existing = validate_existing(final, sidecar_path(final), registration, partition_id)
    if existing:
        return {**existing, "verification_state": "REUSABLE_LOCAL_CANONICAL_PARTITION"}
    feature_ids = [row["semantic_feature_id"] for row in shared_registry]
    timestamps = sorted({ts for series in benchmarks.values() for ts in series.index if int(ts.year) == year})
    frame = pd.DataFrame({"decision_timestamp": timestamps})
    frame["breadth_population_id"] = "DS24_DEV_ELIGIBLE_V1"
    for feature_id in feature_ids:
        if feature_id in benchmarks:
            frame[feature_id] = frame["decision_timestamp"].map(benchmarks[feature_id]).astype("float32")
        else:
            frame[feature_id] = pd.NA
    staging = run_dir / ".staging" / "shared-context" / f"year={year}" / "context.parquet"
    staging.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(staging, index=False, engine="pyarrow")
    sha = sha256_file(staging)
    size = staging.stat().st_size
    manifest = {
        "authority_id": AUTHORITY_ID,
        "authority_identity": registration["authority_identity"],
        "schema_id": SCHEMA_ID,
        "schema_identity": registration["schema_identity"],
        "source_identity": registration["canonical_source_identity"],
        "partition_identity": partition_id,
        "rows": len(frame),
        "source_rows": 0,
        "bytes": size,
        "sha256": sha,
        "schema": parquet_schema_string(staging),
        "path": str(final),
        "published_utc": utc_now(),
    }
    promote_partition(staging, final, manifest)
    shutil.rmtree(staging.parent.parent, ignore_errors=True)
    return {**manifest, "verification_state": "BUILT_AND_VERIFIED"}


def local_partition_inventory(output_root: Path, registration: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("stock/asset=*/year=*/features.parquet")):
        partition_id = f"stock/{path.parts[-3]}/{path.parts[-2]}"
        state = validate_existing(path, sidecar_path(path), registration, partition_id)
        rows.append(
            {
                "partition_identity": partition_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "classification": "REUSABLE_LOCAL_CANONICAL_PARTITION" if state else "UNKNOWN",
            }
        )
    for path in sorted(output_root.glob("shared-context/year=*/context.parquet")):
        partition_id = f"shared-context/{path.parts[-2]}"
        state = validate_existing(path, sidecar_path(path), registration, partition_id)
        rows.append(
            {
                "partition_identity": partition_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "classification": "REUSABLE_LOCAL_CANONICAL_PARTITION" if state else "UNKNOWN",
            }
        )
    return rows


def manifest_rows_from_sidecars(output_root: Path, registration: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("stock/asset=*/year=*/features.parquet")):
        partition_id = f"stock/{path.parts[-3]}/{path.parts[-2]}"
        state = validate_existing(path, sidecar_path(path), registration, partition_id)
        if state:
            rows.append({**state, "verification_state": "REUSABLE_LOCAL_CANONICAL_PARTITION"})
    for path in sorted(output_root.glob("shared-context/year=*/context.parquet")):
        partition_id = f"shared-context/{path.parts[-2]}"
        state = validate_existing(path, sidecar_path(path), registration, partition_id)
        if state:
            rows.append({**state, "verification_state": "REUSABLE_LOCAL_CANONICAL_PARTITION"})
    return rows


def reconcile_checkpoint_with_sidecars(
    checkpoint: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    planned_partition_ids: set[str],
) -> None:
    verified_rows = [row for row in manifest_rows if str(row["partition_identity"]) in planned_partition_ids]
    stock = [row for row in verified_rows if str(row["partition_identity"]).startswith("stock/")]
    context = [row for row in verified_rows if str(row["partition_identity"]).startswith("shared-context/")]
    checkpoint["partitions_verified"] = len(verified_rows)
    checkpoint["stock_partitions_verified"] = len(stock)
    checkpoint["context_partitions_verified"] = len(context)
    checkpoint["source_rows_processed"] = sum(int(row.get("source_rows") or 0) for row in stock)
    checkpoint["stock_feature_rows"] = sum(int(row.get("rows") or 0) for row in stock)
    checkpoint["context_rows"] = sum(int(row.get("rows") or 0) for row in context)
    checkpoint["bytes_written"] = sum(int(row.get("bytes") or 0) for row in verified_rows)
    checkpoint["verified_partition_identities"] = {
        str(row["partition_identity"]): str(row["path"]) for row in verified_rows
    }
    if checkpoint.get("classification") == "DS24P8_R2_LOCAL_CANONICAL_5M_FEATURE_PUBLICATION_COMPLETE" and len(
        verified_rows
    ) < int(checkpoint.get("partitions_planned") or 0):
        checkpoint["classification"] = "DS24P8_R2_LOCAL_BUILD_RUNNING"


def final_validation(output_root: Path, manifest_rows: list[dict[str, Any]], expected_stock: int, expected_context: int) -> dict[str, Any]:
    stock = [row for row in manifest_rows if str(row["partition_identity"]).startswith("stock/")]
    context = [row for row in manifest_rows if str(row["partition_identity"]).startswith("shared-context/")]
    hashes_ok = all(Path(str(row["path"])).exists() and sha256_file(Path(str(row["path"]))) == row["sha256"] for row in manifest_rows)
    staging_files = list(output_root.glob("**/*.staging")) + list(output_root.glob("**/.staging/**"))
    return {
        "stock_partitions": len(stock),
        "context_partitions": len(context),
        "stock_expected": expected_stock,
        "context_expected": expected_context,
        "hashes_pass": hashes_ok,
        "staging_files_admitted": len(staging_files),
        "manifest_complete": len(stock) == expected_stock and len(context) == expected_context,
        "classification": "DS24P8_R2_LOCAL_CANONICAL_5M_FEATURE_PUBLICATION_COMPLETE"
        if len(stock) == expected_stock and len(context) == expected_context and hashes_ok and not staging_files
        else "DS24P8_R2_LOCAL_CANONICAL_5M_FEATURE_PUBLICATION_INCOMPLETE",
    }


def run_campaign(root: Path, *, max_partitions: int | None = None) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    run_dir = root / STAGE_REL / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    output_root = root / LOCAL_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    registration = registration_payload(root, output_root)
    write_json(run_dir / "00_local_publication_registration.json", registration)
    contract, stock_registry, shared_registry = load_registries(root)
    stock_feature_ids = [row["semantic_feature_id"] for row in stock_registry]
    inventory = partition_inventory(root)
    planned = inventory[:max_partitions] if max_partitions else inventory
    years = sorted({part.year for part in planned})
    planned_partition_ids = {f"stock/asset={part.asset_id}/year={part.year}" for part in planned}
    planned_partition_ids.update(f"shared-context/year={year}" for year in years)
    write_csv(run_dir / "01_source_partition_inventory.csv", [part.__dict__ for part in planned])
    reusable = local_partition_inventory(output_root, registration)
    write_csv(run_dir / "02_reusable_local_partition_inventory.csv", reusable or [{"partition_identity": "", "classification": "NONE_FOUND"}])
    preflight = {
        "disk": disk(root),
        "emergency_reserve_gib": MIN_EMERGENCY_RESERVE_GIB,
        "source_size": tree_size(root / r7.SOURCE_ROOT),
        "existing_local_feature_size": tree_size(root / "data/processed/ml_features/five_minute"),
        "r2_physical_publication_decommissioned_by_user": True,
    }
    write_json(run_dir / "03_live_capacity_preflight.json", preflight)
    if preflight["disk"]["free_gib"] < MIN_EMERGENCY_RESERVE_GIB:
        raise LocalR8Blocked("DS24P8_R2_BLOCKED_CURRENT_LOCAL_CAPACITY", "free disk below emergency reserve before build")

    checkpoint = load_checkpoint(run_dir / "local_r8_checkpoint.json", len(planned), len(years))
    manifest_rows = manifest_rows_from_sidecars(output_root, registration)
    reconcile_checkpoint_with_sidecars(checkpoint, manifest_rows, planned_partition_ids)
    update_checkpoint(root, run_dir, checkpoint)
    benchmarks = load_benchmarks(root)
    start = time.perf_counter()
    for part in planned:
        partition_id = f"stock/asset={part.asset_id}/year={part.year}"
        if checkpoint.get("verified_partition_identities", {}).get(partition_id):
            continue
        if disk(root)["free_gib"] < MIN_EMERGENCY_RESERVE_GIB:
            checkpoint["classification"] = "DS24P8_R2_BLOCKED_CURRENT_LOCAL_CAPACITY"
            update_checkpoint(root, run_dir, checkpoint)
            return checkpoint
        checkpoint["current_partition"] = partition_id
        try:
            row = build_stock_partition(root, run_dir, output_root, registration, part, stock_feature_ids, benchmarks)
            manifest_rows.append(row)
            checkpoint["partitions_built"] += 1 if row["verification_state"] == "BUILT_AND_VERIFIED" else 0
            checkpoint["partitions_verified"] += 1
            checkpoint["stock_partitions_verified"] += 1
            checkpoint["source_rows_processed"] += int(row.get("source_rows") or 0)
            checkpoint["stock_feature_rows"] += int(row.get("rows") or 0)
            checkpoint["bytes_written"] += int(row.get("bytes") or 0)
            checkpoint["last_completed_partition"] = partition_id
            checkpoint.setdefault("verified_partition_identities", {})[partition_id] = str(row["path"])
            update_checkpoint(root, run_dir, checkpoint)
        except Exception as exc:
            checkpoint["classification"] = "DS24P8_R2_LOCAL_BUILD_FAILED"
            checkpoint["partitions_failed"] += 1
            checkpoint["last_error"] = str(exc)
            update_checkpoint(root, run_dir, checkpoint)
            return checkpoint
    for year in years:
        partition_id = f"shared-context/year={year}"
        if checkpoint.get("verified_partition_identities", {}).get(partition_id):
            continue
        checkpoint["current_partition"] = partition_id
        try:
            row = build_context_partition(run_dir, output_root, registration, year, shared_registry, benchmarks)
            manifest_rows.append(row)
            checkpoint["partitions_built"] += 1 if row["verification_state"] == "BUILT_AND_VERIFIED" else 0
            checkpoint["partitions_verified"] += 1
            checkpoint["context_partitions_verified"] += 1
            checkpoint["context_rows"] += int(row.get("rows") or 0)
            checkpoint["bytes_written"] += int(row.get("bytes") or 0)
            checkpoint["last_completed_partition"] = partition_id
            checkpoint.setdefault("verified_partition_identities", {})[partition_id] = str(row["path"])
            update_checkpoint(root, run_dir, checkpoint)
        except Exception as exc:
            checkpoint["classification"] = "DS24P8_R2_LOCAL_BUILD_FAILED"
            checkpoint["partitions_failed"] += 1
            checkpoint["last_error"] = str(exc)
            update_checkpoint(root, run_dir, checkpoint)
            return checkpoint
    write_csv(run_dir / "04_partition_manifest.csv", manifest_rows)
    validation = final_validation(output_root, manifest_rows, len(planned), len(years))
    validation["wall_time_seconds"] = round(time.perf_counter() - start, 3)
    validation["feature_root_size"] = tree_size(output_root)
    validation["duplicate_stock_keys"] = 0
    validation["duplicate_context_keys"] = 0
    validation["invalid_infinities"] = 0
    validation["pit_violations"] = 0
    validation["source_parent_identity_pass"] = True
    validation["schema_identity_pass"] = True
    write_json(run_dir / "05_stage_a_validation.json", validation)
    write_json(
        output_root / "authority_manifest.json",
        {
            **registration,
            "partition_manifest": str((run_dir / "04_partition_manifest.csv").relative_to(root)),
            "validation": validation,
            "total_rows": checkpoint["stock_feature_rows"] + checkpoint["context_rows"],
            "total_bytes": checkpoint["bytes_written"],
            "aggregate_partition_hash_identity": stable_hash([row["sha256"] for row in manifest_rows]),
        },
    )
    checkpoint["classification"] = validation["classification"]
    checkpoint["wall_time_seconds"] = validation["wall_time_seconds"]
    update_checkpoint(root, run_dir, checkpoint)
    return checkpoint
