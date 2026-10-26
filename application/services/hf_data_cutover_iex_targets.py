"""Build the physically separate IEX H30/H60 target authority."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, utc_now
from core.research.iex_dataset_builders import build_daily_targets, build_intraday_targets
from infrastructure.data.iex_market_data_store import sha256_file
from infrastructure.data.iex_research_artifact_store import IexResearchArtifactStore


TARGET_WORKERS = 4
MINIMUM_FINAL_RESERVE_BYTES = 15 * 1024**3


def run_iex_target_build(paths: CutoverPaths) -> dict[str, Any]:
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("state") not in {
        "IEX_FEATURE_V1_COMPLETE_TARGET_BUILD_PENDING",
        "IEX_TARGET_BUILD_IN_PROGRESS",
    }:
        raise CutoverAuditError(f"IEX target build requires complete Feature V1, found {run_manifest.get('state')}")
    feature_manifest_path = paths.output_root / "IEX_FEATURE_V1_MANIFEST.json"
    feature_manifest = _read_json(feature_manifest_path)
    if feature_manifest.get("status") != "COMPLETE":
        raise CutoverAuditError("IEX Feature V1 manifest is not complete")
    raw_manifests = sorted((paths.repository_root / "data/market/iex/raw/5m").glob("symbol=*/year=*/manifest.json"))
    adjusted_daily_manifests = sorted(
        (paths.repository_root / "data/market/iex/derived/1d_adjusted").glob("symbol=*/manifest.json")
    )
    store = IexResearchArtifactStore(paths.repository_root / "data/market/iex")
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_TARGET_BUILD_IN_PROGRESS",
            "next_gate": "Build and validate physically separate H30/H60 targets.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=TARGET_WORKERS, thread_name_prefix="iex-targets") as executor:
        futures: dict[Future[dict[str, Any]], Path] = {
            executor.submit(_build_target_partition, store, source): source for source in raw_manifests
        }
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index % 100 == 0:
                elapsed = max(time.monotonic() - started, 1.0)
                print(
                    "IEX TARGET BUILD "
                    f"partitions={index}/{len(raw_manifests)} partitions_per_hour={index / elapsed * 3600:.1f} "
                    f"free_bytes={shutil.disk_usage(paths.repository_root).free}",
                    flush=True,
                )
            if shutil.disk_usage(paths.repository_root).free <= MINIMUM_FINAL_RESERVE_BYTES:
                for pending in futures:
                    pending.cancel()
                raise CutoverAuditError("IEX target build stopped at the 15 GiB reserve gate")
    with ThreadPoolExecutor(max_workers=TARGET_WORKERS, thread_name_prefix="iex-daily-targets") as executor:
        daily_futures = [
            executor.submit(_build_daily_target_partition, store, source)
            for source in adjusted_daily_manifests
        ]
        for future in as_completed(daily_futures):
            completed.append(future.result())
    completed.sort(
        key=lambda row: (
            row["identity"]["symbol"],
            -1 if row["identity"]["year"] is None else int(row["identity"]["year"]),
        )
    )
    mature_rows = sum(int(row["quality"]["mature_row_count"]) for row in completed)
    manifest = {
        "schema_version": "hf_data_cutover_r1.iex_target_v1_manifest.v1",
        "created_at_utc": utc_now(),
        "research_authority_only": True,
        "feature_manifest_sha256_at_target_build": sha256_file(feature_manifest_path),
        "feature_spec_frozen_before_targets": True,
        "targets_physically_separate": True,
        "target_horizons": ["H30", "H60", "H5", "H10"],
        "intraday_contract": {
            "entry_convention": "next contiguous regular-session 5m bar open after decision timestamp",
            "outcome_convention": "close of the sixth/twelfth contiguous regular-session bar after entry",
            "maturity_rule": "entry timestamp plus 30/60 minutes; must remain in the same XNYS session",
            "corporate_action_handling": "same-session targets never cross overnight corporate-action boundaries",
        },
        "daily_contract": {
            "entry_convention": "next listed XNYS session adjusted daily open",
            "outcome_convention": "adjusted close of the fifth/tenth listed session after decision",
            "maturity_rule": "outcome session close",
            "corporate_action_handling": "Alpaca adjustment=all research view, physically separate from raw daily source truth",
        },
        "missingness_state": "UNAVAILABLE_GAP_OR_SESSION_END retained explicitly",
        "partition_count": len(completed),
        "row_count": sum(int(row["row_count"]) for row in completed),
        "mature_row_count": mature_rows,
        "unavailable_row_count": sum(int(row["quality"]["unavailable_row_count"]) for row in completed),
        "parquet_bytes": sum(int(row["parquet_bytes"]) for row in completed),
        "maturity_violation_count": sum(int(row["quality"]["maturity_violation_count"]) for row in completed),
        "duplicate_identity_count": sum(int(row["quality"]["duplicate_identity_count"]) for row in completed),
        "table_counts": _target_table_counts(completed),
        "partition_manifest_sha256": sorted(
            sha256_file(
                paths.repository_root
                / "data/market/iex"
                / Path(row["identity"]["relative_directory"])
                / "manifest.json"
            )
            for row in completed
        ),
        "status": "COMPLETE",
    }
    output_path = paths.output_root / "IEX_TARGET_V1_MANIFEST.json"
    atomic_json(output_path, manifest)
    run_manifest = _read_json(run_manifest_path)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_DATASET_VALIDATION_PENDING",
            "iex_target_v1": {
                "manifest_sha256": sha256_file(output_path),
                "rows": manifest["row_count"],
                "mature_rows": mature_rows,
            },
            "next_gate": "Run dataset-wide quality, determinism, leakage, and SIP-to-IEX migration validation.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def _build_target_partition(
    store: IexResearchArtifactStore,
    source_manifest_path: Path,
) -> dict[str, Any]:
    source_manifest = _read_json(source_manifest_path)
    request = source_manifest["request"]
    symbol = str(request["symbol"])
    year = int(request["year"])
    relative = Path("targets/v1/intraday") / f"symbol={symbol}" / f"year={year}"
    identity = {
        "dataset": "iex_targets_v1",
        "table": "intraday_h30_h60",
        "symbol": symbol,
        "year": year,
        "relative_directory": str(relative),
        "source_partition_id": request["partition_id"],
        "source_parquet_sha256": source_manifest["parquet_sha256"],
        "builder_version": "iex_intraday_targets_v1",
    }
    existing = store.completed(relative, identity)
    if existing is not None:
        return existing
    source_path = Path(source_manifest["parquet_path"])
    if sha256_file(source_path) != source_manifest["parquet_sha256"]:
        raise ValueError(f"Raw source hash changed before target build: {source_path}")
    raw = pd.read_parquet(source_path)
    targets = build_intraday_targets(raw)
    duplicate_count = int(targets.duplicated(["symbol", "decision_timestamp", "target_horizon"]).sum())
    mature = targets["target_status"].eq("MATURE")
    maturity_violations = int(
        (
            mature
            & (
                targets["maturity_timestamp"].isna()
                | targets["maturity_timestamp"].le(targets["decision_timestamp"])
                | targets["target_value"].isna()
            )
        ).sum()
    )
    if duplicate_count or maturity_violations:
        raise ValueError(
            f"Target partition validation failed duplicates={duplicate_count} maturity={maturity_violations}"
        )
    quality = {
        "mature_row_count": int(mature.sum()),
        "unavailable_row_count": int((~mature).sum()),
        "duplicate_identity_count": duplicate_count,
        "maturity_violation_count": maturity_violations,
        "h30_mature_count": int((mature & targets["target_horizon"].eq("H30")).sum()),
        "h60_mature_count": int((mature & targets["target_horizon"].eq("H60")).sum()),
    }
    return store.publish(
        relative,
        targets,
        identity=identity,
        lineage={
            "source_manifest": str(source_manifest_path.resolve()),
            "source_parquet_sha256": source_manifest["parquet_sha256"],
            "feature_paths_read": False,
        },
        quality=quality,
    )


def _build_daily_target_partition(
    store: IexResearchArtifactStore,
    source_manifest_path: Path,
) -> dict[str, Any]:
    source_manifest = _read_json(source_manifest_path)
    source_identity = source_manifest["identity"]
    symbol = str(source_identity["symbol"])
    relative = Path("targets/v1/daily") / f"symbol={symbol}"
    identity = {
        "dataset": "iex_targets_v1",
        "table": "daily_h5_h10",
        "symbol": symbol,
        "year": None,
        "relative_directory": str(relative),
        "source_parquet_sha256": source_manifest["parquet_sha256"],
        "source_adjustment": "all",
        "builder_version": "iex_daily_targets_v1",
    }
    existing = store.completed(relative, identity)
    if existing is not None:
        return existing
    source_path = Path(source_manifest["parquet_path"])
    if sha256_file(source_path) != source_manifest["parquet_sha256"]:
        raise ValueError(f"Adjusted daily source hash changed before target build: {source_path}")
    targets = build_daily_targets(pd.read_parquet(source_path))
    duplicate_count = int(targets.duplicated(["symbol", "decision_timestamp", "target_horizon"]).sum())
    mature = targets["target_status"].eq("MATURE")
    maturity_violations = int(
        (
            mature
            & (
                targets["maturity_timestamp"].isna()
                | targets["maturity_timestamp"].le(targets["decision_timestamp"])
                | targets["target_value"].isna()
            )
        ).sum()
    )
    if duplicate_count or maturity_violations:
        raise ValueError(
            f"Daily target partition validation failed duplicates={duplicate_count} maturity={maturity_violations}"
        )
    quality = {
        "mature_row_count": int(mature.sum()),
        "unavailable_row_count": int((~mature).sum()),
        "duplicate_identity_count": duplicate_count,
        "maturity_violation_count": maturity_violations,
        "h5_mature_count": int((mature & targets["target_horizon"].eq("H5")).sum()),
        "h10_mature_count": int((mature & targets["target_horizon"].eq("H10")).sum()),
    }
    return store.publish(
        relative,
        targets,
        identity=identity,
        lineage={
            "source_manifest": str(source_manifest_path.resolve()),
            "source_parquet_sha256": source_manifest["parquet_sha256"],
            "source_adjustment": "all",
            "feature_paths_read": False,
        },
        quality=quality,
    )


def _target_table_counts(completed: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for row in completed:
        table = str(row["identity"]["table"])
        values = output.setdefault(table, {"partitions": 0, "rows": 0, "mature_rows": 0})
        values["partitions"] += 1
        values["rows"] += int(row["row_count"])
        values["mature_rows"] += int(row["quality"]["mature_row_count"])
    return output


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["run_iex_target_build"]
