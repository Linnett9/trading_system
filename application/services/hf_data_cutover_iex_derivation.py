"""Application orchestration for deterministic IEX 1h derivation."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, utc_now
from core.research.iex_dataset_builders import derive_completed_hour_bars
from infrastructure.data.iex_market_data_store import sha256_file
from infrastructure.data.iex_research_artifact_store import IexResearchArtifactStore


DERIVATION_WORKERS = 4
MINIMUM_FINAL_RESERVE_BYTES = 15 * 1024**3


def run_iex_hour_derivation(paths: CutoverPaths) -> dict[str, Any]:
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("state") not in {
        "IEX_ADJUSTED_DAILY_COMPLETE_1H_PENDING",
        "IEX_1H_DERIVATION_IN_PROGRESS",
    }:
        raise CutoverAuditError(f"IEX 1h derivation requires complete raw and adjusted daily data, found {run_manifest.get('state')}")
    raw_manifest = _read_json(paths.output_root / "IEX_RAW_5M_MANIFEST.json")
    if raw_manifest.get("failed_partition_count") or raw_manifest.get("duplicate_count") or raw_manifest.get("off_grid_count"):
        raise CutoverAuditError("IEX raw 5m manifest fails the 1h derivation gate")
    raw_root = paths.repository_root / "data/market/iex/raw/5m"
    output_root = paths.repository_root / "data/market/iex"
    source_manifests = sorted(raw_root.glob("symbol=*/year=*/manifest.json"))
    if len(source_manifests) != int(raw_manifest["completed_partition_count"]):
        raise CutoverAuditError("IEX raw 5m partition inventory does not match its dataset manifest")
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_1H_DERIVATION_IN_PROGRESS",
            "next_gate": "Complete and validate deterministic completed-hour derivation.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    store = IexResearchArtifactStore(output_root)
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=DERIVATION_WORKERS, thread_name_prefix="iex-1h") as executor:
        futures: dict[Future[dict[str, Any]], Path] = {
            executor.submit(_derive_partition, store, manifest_path): manifest_path
            for manifest_path in source_manifests
        }
        for future in as_completed(futures):
            completed.append(future.result())
            if shutil.disk_usage(output_root).free <= MINIMUM_FINAL_RESERVE_BYTES:
                for pending in futures:
                    pending.cancel()
                raise CutoverAuditError("IEX 1h derivation stopped at the 15 GiB reserve gate")
    completed.sort(key=lambda row: (row["identity"]["symbol"], row["identity"]["year"]))
    manifest = {
        "schema_version": "hf_data_cutover_r1.iex_1h_derivation_manifest.v1",
        "created_at_utc": utc_now(),
        "source_raw_5m_manifest_sha256": sha256_file(paths.output_root / "IEX_RAW_5M_MANIFEST.json"),
        "calendar": "maintained XNYS authority",
        "hour_definition": "session-open-anchored 60 minutes; exactly 12 unique 5m grid bars; partial closing hours excluded",
        "partition_count": len(completed),
        "row_count": sum(int(row["row_count"]) for row in completed),
        "parquet_bytes": sum(int(row["parquet_bytes"]) for row in completed),
        "input_5m_rows": sum(int(row["quality"]["input_rows"]) for row in completed),
        "rth_5m_rows": sum(int(row["quality"]["rth_rows"]) for row in completed),
        "incomplete_hours_excluded": sum(int(row["quality"]["incomplete_hours_excluded"]) for row in completed),
        "partition_manifest_sha256": sorted(
            sha256_file(output_root / Path(row["identity"]["relative_directory"]) / "manifest.json")
            for row in completed
        ),
        "deterministic_rebuild_contract": "same source hash + calendar identity => identical row identities and values",
    }
    output_path = paths.output_root / "IEX_1H_DERIVATION_MANIFEST.json"
    atomic_json(output_path, manifest)
    run_manifest = _read_json(run_manifest_path)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_1H_COMPLETE_FEATURE_BUILD_PENDING",
            "iex_1h_derivation": {
                "manifest_sha256": sha256_file(output_path),
                "rows": manifest["row_count"],
                "partitions": manifest["partition_count"],
            },
            "next_gate": "Build IEX Feature Bank V1 from the frozen pre-target specification.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def _derive_partition(store: IexResearchArtifactStore, source_manifest_path: Path) -> dict[str, Any]:
    source_manifest = _read_json(source_manifest_path)
    request = source_manifest["request"]
    symbol = str(request["symbol"])
    year = int(request["year"])
    relative = Path("derived/1h") / f"symbol={symbol}" / f"year={year}"
    identity = {
        "dataset": "derived_1h",
        "symbol": symbol,
        "year": year,
        "relative_directory": str(relative),
        "source_partition_id": request["partition_id"],
        "source_parquet_sha256": source_manifest["parquet_sha256"],
        "builder_version": "iex_completed_hour_v1",
    }
    existing = store.completed(relative, identity)
    if existing is not None:
        return existing
    source_path = Path(source_manifest["parquet_path"])
    if sha256_file(source_path) != source_manifest["parquet_sha256"]:
        raise ValueError(f"Raw source hash changed before 1h derivation: {source_path}")
    raw = pd.read_parquet(source_path)
    derived, quality = derive_completed_hour_bars(raw)
    if not derived.empty:
        duplicated = int(derived.duplicated(["symbol", "timestamp"]).sum())
        if duplicated:
            raise ValueError(f"Derived 1h partition contains {duplicated} duplicate row identities")
        if bool((derived["available_at"] <= derived["timestamp"]).any()):
            raise ValueError("Derived 1h availability must follow hour start")
    return store.publish(
        relative,
        derived,
        identity=identity,
        lineage={
            "source_manifest": str(source_manifest_path.resolve()),
            "source_parquet_sha256": source_manifest["parquet_sha256"],
            "aggregation_version": "iex_completed_hour_v1",
        },
        quality=quality,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["run_iex_hour_derivation"]
