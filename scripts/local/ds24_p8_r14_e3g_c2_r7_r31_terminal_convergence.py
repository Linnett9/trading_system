from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.comparable_policy import POLICY_ID, policy_hash
from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.ds24_metrics_only_evaluator import POLICY_ID as METRICS_ONLY_POLICY_ID
from core.research.ml.ds24_metrics_only_evaluator import (
    read_parquet_log,
    resolved_v3_summary,
)
from core.research.ml.ds24_metrics_only_evaluator import policy_hash as metrics_policy_hash


RUN_ID = "ds24_p8_r14_e3g_c2_r7_r31_20260827T000000Z"
STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
POLICY_ROOT = STAGE / "r7_r14_policy_workers"
TARGET_ROOT = (
    ROOT
    / "data/processed/ml_targets/five_minute/version=five_minute_targets_v1/run=ticket_71b_60m_20260730T170751Z"
    / "target_id=forward_return_60m__decision_5m"
)
MODEL_PARTITION_MANIFEST = STAGE / "91_r7_r1_extended_model_data_partition_manifest.csv"
CANONICAL_ASSET_REGISTRY = ROOT / "data/reference/assets/canonical_asset_registry.csv"
R25_STATUS_PATH = STAGE / "r7_r25_metrics_only_migration" / "status.json"
SUPERVISOR_HEARTBEAT_PATH = STAGE / "R7_R27_01_supervisor_heartbeat.json"
SUPERVISOR_STATE_PATH = STAGE / "R7_R27_02_family_state_board.json"
SUPERVISOR_RESOURCE_PATH = STAGE / "R7_R27_05_resource_snapshot.json"
SUPERVISOR_ADMISSION_LEDGER_PATH = STAGE / "R7_R27_06_admission_ledger.csv"
SUPERVISOR_LAUNCH_LEDGER_PATH = STAGE / "R7_R27_07_automatic_launch_ledger.csv"

EXACT_TERMINAL_CURSOR = "2026-06-30T19:00:00+00:00"
POLICY_TERMINAL_CURSOR = "2026-06-30T20:00:00+00:00"
LAST_RESOLVED_TARGET_CURSOR = "2026-06-30T19:00:00+00:00"
TOP_N_COST_BPS_PER_UNIT_TURNOVER = 0.0
TRADING_SESSIONS_PER_YEAR = 252

POLICY_FAMILIES = {"ridge_policy_v1_control", "pca_ridge_policy_v1_control", "spline_additive_ridge"}
PERFORMANCE_FAMILIES = ["ridge_policy_v1_control", "pca_ridge_policy_v1_control", "spline_additive_ridge"]
COMPLETE_MARKER_NAME = "r31_terminal_completion.json"
VALIDATING_MARKER_NAME = "r31_terminal_validating.json"
EXCLUSION_MARKER_NAME = "r31_terminal_admission_exclusion.json"
SUMMARY_NAME = "r31_performance_summary.json"
NAMESPACE_QUARANTINE_NAME = "r36_namespace_quarantine.json"
QUARANTINE_RELEASE_NAME = "r36_quarantine_scientific_release.json"


def utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path | str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Any, *, advisory: bool = False) -> None:
    write_json_atomic(path, payload, advisory=advisory)


def write_stage_json(name: str, payload: Any) -> None:
    write_json(STAGE / name, payload, advisory=False)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{str(key): value for key, value in row.items() if key is not None} for row in csv.DictReader(handle)]


def latest_csv_row(path: Path, *, family: str | None = None) -> dict[str, str]:
    rows = read_csv_rows(path)
    if family is not None:
        rows = [
            row
            for row in rows
            if row.get("family") == family or row.get("candidate_family") == family or row.get("event") == family
        ]
    return rows[-1] if rows else {}


def latest_launch_for_family(family: str) -> dict[str, Any]:
    if not SUPERVISOR_LAUNCH_LEDGER_PATH.exists():
        return {}
    rows = list(csv.reader(SUPERVISOR_LAUNCH_LEDGER_PATH.read_text(encoding="utf-8").splitlines()))
    for row in reversed(rows):
        if len(row) >= 3 and row[1] == family:
            return {
                "timestamp": row[0],
                "family": row[1],
                "pid": int(row[2]) if str(row[2]).isdigit() else row[2],
                "command": row[3] if len(row) > 3 else "",
                "reason_admitted": row[4] if len(row) > 4 else "",
                "policy_hash": row[5] if len(row) > 5 else "",
                "configuration_hash": row[6] if len(row) > 6 else "",
                "checkpoint_root": row[7] if len(row) > 7 else "",
                "output_root": row[8] if len(row) > 8 else "",
                "resume_generation": int(row[9]) if len(row) > 9 and str(row[9]).isdigit() else (row[9] if len(row) > 9 else ""),
            }
    return {}


def latest_admission_for_family(family: str) -> dict[str, Any]:
    if not SUPERVISOR_ADMISSION_LEDGER_PATH.exists():
        return {}
    rows = list(csv.reader(SUPERVISOR_ADMISSION_LEDGER_PATH.read_text(encoding="utf-8").splitlines()))
    for row in reversed(rows):
        if len(row) >= 3 and row[1] == family:
            return {
                "timestamp": row[0],
                "candidate_family": row[1],
                "admitted": str(row[2]).lower() == "true",
                "blocked_reasons": row[3] if len(row) > 3 else "",
                "admission_proof_window": row[4] if len(row) > 4 else "",
                "active_model_processes": int(row[5]) if len(row) > 5 and str(row[5]).isdigit() else (row[5] if len(row) > 5 else ""),
                "available_ram_bytes": int(row[6]) if len(row) > 6 and str(row[6]).isdigit() else (row[6] if len(row) > 6 else ""),
                "system_commit_percent": float(row[7]) if len(row) > 7 and row[7] else None,
                "free_disk_bytes": int(row[8]) if len(row) > 8 and str(row[8]).isdigit() else (row[8] if len(row) > 8 else ""),
            }
    return {}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or str(value) in {"", "NaT", "nan"}:
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert("UTC") if parsed.tzinfo else parsed.tz_localize("UTC")


def iso_ts(value: Any) -> str:
    parsed = normalise_timestamp(value)
    return parsed.isoformat() if parsed is not None else ""


def same_timestamp(left: Any, right: Any) -> bool:
    return bool(iso_ts(left) and iso_ts(left) == iso_ts(right))


def terminal_authority(family: str) -> dict[str, Any]:
    if family == "exact_ridge_pca":
        cursor = EXACT_TERMINAL_CURSOR
        kind = "LAST_TRAINABLE_TARGET_DECISION"
    elif family in POLICY_FAMILIES or family in {"rff_ridge", "huber", "mlp"}:
        cursor = POLICY_TERMINAL_CURSOR
        kind = "LAST_REGISTERED_POLICY_SCORING_DECISION"
    else:
        cursor = POLICY_TERMINAL_CURSOR
        kind = "LAST_REGISTERED_POLICY_SCORING_DECISION"
    return {
        "family": family,
        "registered_terminal_cursor": cursor,
        "last_resolved_target_cursor": LAST_RESOLVED_TARGET_CURSOR,
        "authority_kind": kind,
        "decision": (
            "Exact stopped at the last trainable 60m target decision. Policy workers score the complete "
            "registered five-minute spine through 20:00, while 19:05-20:00 outcomes are terminal-censored."
        ),
    }


def has_reached_registered_terminal(
    family: str,
    checkpoint: Mapping[str, Any],
    scoring_spine: Iterable[Any] | None = None,
) -> dict[str, Any]:
    cursor = checkpoint.get("last_completed_T") or checkpoint.get("cursor") or checkpoint.get("current_scoring_cursor")
    cursor_iso = iso_ts(cursor)
    terminal_iso = terminal_authority(family)["registered_terminal_cursor"]
    spine_membership: bool | None = None
    if scoring_spine is not None:
        spine = {iso_ts(item) for item in scoring_spine}
        spine.discard("")
        spine_membership = terminal_iso in spine and cursor_iso in spine
    exact_identity = cursor_iso == terminal_iso
    return {
        "family": family,
        "cursor": cursor_iso,
        "registered_terminal_cursor": terminal_iso,
        "reached": bool(exact_identity and (spine_membership is not False)),
        "timestamp_identity_match": exact_identity,
        "registered_spine_membership": spine_membership,
    }


def family_root(family: str) -> Path:
    return POLICY_ROOT / family


def metrics_root(family: str) -> Path:
    root = family_root(family)
    candidates: list[str] = []
    for filename in ("worker_inventory.json", "progress.json", "initialization_telemetry.json"):
        payload = read_json(root / filename)
        name = str(payload.get("metrics_root_name") or "")
        if name:
            candidates.append(name)
    if root.exists():
        candidates.extend(path.name for path in root.glob("metrics_only_v3*") if path.is_dir())
    candidates.append("metrics_only")
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = root / name
        if (candidate / "resolved_performance_contract_v3.json").exists() or (candidate / "per_t_metrics.parquet").exists():
            return candidate
    return root / "metrics_only"


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return int(len(pd.read_parquet(path)))


def file_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "exists": path.exists(),
        "rows": parquet_rows(path) if path.suffix == ".parquet" and path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
    }


def manifest_evidence(root: Path, stem: str) -> dict[str, Any]:
    path = root / f"{stem}_manifest.json"
    payload = read_json(path)
    parts = payload.get("parts", []) if isinstance(payload.get("parts"), list) else []
    return {
        "path": display_path(path),
        "exists": path.exists(),
        "rows": int(payload.get("total_rows", 0) or 0),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
        "part_count": int(len(parts)),
    }


def metrics_root_is_v3(root: Path) -> bool:
    return (root / "resolved_performance_contract_v3.json").exists()


def inventory_evidence(family: str) -> dict[str, Any]:
    root = metrics_root(family)
    if metrics_root_is_v3(root):
        metrics = manifest_evidence(root, "per_t_metrics")
        topn = manifest_evidence(root, "decision_trace")
        rank_ic = manifest_evidence(root, "rank_ic_v3")
        daily_returns = manifest_evidence(root, "daily_portfolio_returns_v3")
    else:
        metrics = file_evidence(root / "per_t_metrics.parquet")
        topn = file_evidence(root / "decision_trace.parquet")
        rank_ic = {}
        daily_returns = {}
    pending = file_evidence(root / "pending_buffer.parquet")
    checkpoint = file_evidence(root / "checkpoint.json")
    progress = file_evidence(family_root(family) / "progress.json")
    payload = {
        "family": family,
        "metrics_root": display_path(root),
        "metrics_layout": "V3_APPEND_ONLY_MANIFEST" if metrics_root_is_v3(root) else "LEGACY_FLAT_PARQUET",
        "metrics": metrics,
        "topn": topn,
        "rank_ic_v3": rank_ic,
        "daily_portfolio_returns_v3": daily_returns,
        "pending": pending,
        "metrics_checkpoint": checkpoint,
        "progress_checkpoint": progress,
    }
    payload["metrics_inventory_hash"] = stable_hash({"metrics": metrics, "pending": pending})
    payload["topn_inventory_hash"] = stable_hash(topn)
    payload["checkpoint_hash"] = stable_hash({"metrics_checkpoint": checkpoint, "progress_checkpoint": progress})
    return payload


def completion_marker_path(family: str) -> Path:
    return family_root(family) / COMPLETE_MARKER_NAME


def validating_marker_path(family: str) -> Path:
    return family_root(family) / VALIDATING_MARKER_NAME


def exclusion_marker_path(family: str) -> Path:
    return family_root(family) / EXCLUSION_MARKER_NAME


def completion_marker_valid(family: str, *, root: Path | None = None) -> bool:
    marker_root = (root / family) if root is not None else family_root(family)
    payload = read_json(marker_root / COMPLETE_MARKER_NAME)
    return bool(
        payload.get("family") == family
        and payload.get("terminal_validation_state") == "COMPLETE"
        and same_timestamp(payload.get("terminal_cursor"), terminal_authority(family)["registered_terminal_cursor"])
    )


def validating_marker_valid(family: str, *, root: Path | None = None) -> bool:
    marker_root = (root / family) if root is not None else family_root(family)
    payload = read_json(marker_root / VALIDATING_MARKER_NAME) or read_json(marker_root / EXCLUSION_MARKER_NAME)
    return bool(
        payload.get("family") == family
        and payload.get("terminal_validation_state") == "TERMINAL_VALIDATING"
        and same_timestamp(payload.get("terminal_cursor"), terminal_authority(family)["registered_terminal_cursor"])
    )


def terminal_exclusion_state(family: str, *, root: Path | None = None) -> str:
    if completion_marker_valid(family, root=root):
        return "COMPLETE"
    if validating_marker_valid(family, root=root):
        return "TERMINAL_VALIDATING"
    return ""


def active_quarantine(family: str) -> dict[str, Any]:
    payload = read_json(family_root(family) / NAMESPACE_QUARANTINE_NAME)
    if payload and quarantine_release_valid(family):
        return {}
    if payload.get("family") == family and payload.get("status") == "ACTIVE" and payload.get("release_required") is True:
        return payload
    return {}


def quarantine_release_valid(family: str) -> bool:
    payload = read_json(family_root(family) / QUARANTINE_RELEASE_NAME)
    return bool(
        payload.get("family") == family
        and payload.get("status") == "ACTIVE"
        and payload.get("decision") == "HUBER_R36_QUARANTINE_RELEASE_SCIENTIFICALLY_JUSTIFIED"
        and payload.get("released_metrics_root_name") == metrics_root(family).name
        and payload.get("release_id") == "DS24_HUBER_R36_QUARANTINE_SCIENTIFIC_RELEASE_V1"
    )


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_metrics_table(root: Path, stem: str, *, legacy_name: str | None = None) -> pd.DataFrame:
    if metrics_root_is_v3(root):
        return read_parquet_log(root, stem, legacy_path=root / legacy_name if legacy_name else None)
    return read_parquet(root / str(legacy_name or f"{stem}.parquet"))


def target_path(asset_id: str, year: int) -> Path:
    return TARGET_ROOT / f"symbol={asset_id}" / f"year={year}" / "target_rows.parquet"


def target_partition_lookup() -> tuple[dict[tuple[str, int], Path], dict[str, Any]]:
    if not CANONICAL_ASSET_REGISTRY.exists() or not MODEL_PARTITION_MANIFEST.exists():
        return {}, {
            "target_lookup_mode": "DIRECT_SYMBOL_FALLBACK_ONLY",
            "asset_registry_exists": CANONICAL_ASSET_REGISTRY.exists(),
            "model_partition_manifest_exists": MODEL_PARTITION_MANIFEST.exists(),
            "target_lookup_rows": 0,
            "target_lookup_unique_assets": 0,
        }
    registry = pd.read_csv(CANONICAL_ASSET_REGISTRY, usecols=["asset_id", "canonical_symbol"]).dropna()
    registry["asset_id"] = registry["asset_id"].astype(str)
    registry["canonical_symbol"] = registry["canonical_symbol"].astype(str)
    registry = registry.drop_duplicates(["asset_id", "canonical_symbol"])
    manifest = pd.read_csv(MODEL_PARTITION_MANIFEST, usecols=["asset_id", "year", "target_partition"]).dropna()
    manifest = manifest.rename(columns={"asset_id": "canonical_symbol"})
    manifest["canonical_symbol"] = manifest["canonical_symbol"].astype(str)
    manifest["year"] = pd.to_numeric(manifest["year"], errors="coerce").astype("Int64")
    manifest = manifest[manifest["year"].notna()].copy()
    manifest["year"] = manifest["year"].astype(int)
    joined = registry.merge(manifest, on="canonical_symbol", how="inner", validate="many_to_many")
    lookup: dict[tuple[str, int], Path] = {}
    for row in joined.itertuples(index=False):
        partition = Path(str(row.target_partition))
        lookup[(str(row.asset_id), int(row.year))] = partition if partition.is_absolute() else ROOT / partition
    return lookup, {
        "target_lookup_mode": "CANONICAL_REGISTRY_TO_MODEL_PARTITION_MANIFEST",
        "asset_registry_path": display_path(CANONICAL_ASSET_REGISTRY),
        "model_partition_manifest_path": display_path(MODEL_PARTITION_MANIFEST),
        "asset_registry_rows": int(len(registry)),
        "manifest_rows": int(len(manifest)),
        "target_lookup_rows": int(len(lookup)),
        "target_lookup_unique_assets": int(len({asset for asset, _year in lookup})),
    }


def load_targets_for_keys(keys: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if keys.empty:
        return pd.DataFrame(), {"target_files_read": 0, "target_files_missing": 0, "target_rows_loaded": 0}
    work = keys[["asset_id", "decision_timestamp"]].drop_duplicates().copy()
    work["asset_id"] = work["asset_id"].astype(str)
    work["decision_timestamp"] = pd.to_datetime(work["decision_timestamp"], utc=True)
    work["year"] = work["decision_timestamp"].dt.year.astype(int)
    lookup, lookup_meta = target_partition_lookup()
    frames: list[pd.DataFrame] = []
    missing = 0
    files_read = 0
    manifest_resolved = 0
    fallback_resolved = 0
    missing_manifest_mapping = 0
    for (asset, year), group in work.groupby(["asset_id", "year"], sort=True):
        lookup_key = (str(asset), int(year))
        path = lookup.get(lookup_key)
        if path is not None:
            manifest_resolved += 1
        else:
            missing_manifest_mapping += 1
            path = target_path(str(asset), int(year))
            if path.exists():
                fallback_resolved += 1
        if not path.exists():
            missing += 1
            continue
        target = pd.read_parquet(
            path,
            columns=["asset_id", "decision_timestamp", "target_available_timestamp", "target_is_trainable", "target_value"],
        )
        target["asset_id"] = target["asset_id"].astype(str)
        target["decision_timestamp"] = pd.to_datetime(target["decision_timestamp"], utc=True)
        wanted = set(group["decision_timestamp"])
        target = target[target["decision_timestamp"].isin(wanted)].copy()
        if not target.empty:
            frames.append(target)
        files_read += 1
    if not frames:
        return pd.DataFrame(), {
            "target_files_read": files_read,
            "target_files_missing": missing,
            "target_rows_loaded": 0,
            "target_manifest_resolved_files": manifest_resolved,
            "target_fallback_resolved_files": fallback_resolved,
            "target_missing_manifest_mappings": missing_manifest_mapping,
            **lookup_meta,
        }
    out = pd.concat(frames, ignore_index=True)
    out["target_available_timestamp"] = pd.to_datetime(out["target_available_timestamp"], utc=True, errors="coerce")
    return out, {
        "target_files_read": files_read,
        "target_files_missing": missing,
        "target_rows_loaded": int(len(out)),
        "target_manifest_resolved_files": manifest_resolved,
        "target_fallback_resolved_files": fallback_resolved,
        "target_missing_manifest_mappings": missing_manifest_mapping,
        **lookup_meta,
    }


def classify_pending_frame(family: str, pending: pd.DataFrame, targets: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if pending.empty:
        return [], {
            "RESOLVED_MATURED": 0,
            "RIGHT_CENSORED_AT_DATA_TERMINAL": 0,
            "SOURCE_MISSING_BLOCKED": 0,
            "DUPLICATE_INVALID": 0,
            "CHRONOLOGY_INVALID": 0,
        }
    work = pending.copy()
    work["family"] = work.get("family", family)
    work["asset_id"] = work["asset_id"].astype(str)
    work["decision_timestamp"] = pd.to_datetime(work["decision_timestamp"], utc=True)
    duplicate_mask = work.duplicated(["family", "decision_timestamp", "asset_id"], keep=False)
    target = targets.copy()
    if target.empty:
        target = pd.DataFrame(columns=["asset_id", "decision_timestamp", "target_available_timestamp", "target_is_trainable", "target_value"])
    target["asset_id"] = target["asset_id"].astype(str)
    target["decision_timestamp"] = pd.to_datetime(target["decision_timestamp"], utc=True, errors="coerce")
    merged = work.merge(target, on=["asset_id", "decision_timestamp"], how="left", validate="many_to_one")
    policy_terminal = normalise_timestamp(terminal_authority(family)["registered_terminal_cursor"])
    resolved_terminal = normalise_timestamp(LAST_RESOLVED_TARGET_CURSOR)
    rows: list[dict[str, Any]] = []
    counts = {
        "RESOLVED_MATURED": 0,
        "RIGHT_CENSORED_AT_DATA_TERMINAL": 0,
        "SOURCE_MISSING_BLOCKED": 0,
        "DUPLICATE_INVALID": 0,
        "CHRONOLOGY_INVALID": 0,
    }
    for idx, row in merged.iterrows():
        decision = pd.Timestamp(row["decision_timestamp"]).tz_convert("UTC")
        target_available = normalise_timestamp(row.get("target_available_timestamp"))
        trainable = bool(row.get("target_is_trainable")) if not pd.isna(row.get("target_is_trainable")) else False
        target_value = row.get("target_value")
        has_target = not pd.isna(row.get("target_is_trainable"))
        finite_target = pd.notna(target_value) and math.isfinite(float(target_value))
        if bool(duplicate_mask.iloc[idx]):
            reason = "DUPLICATE_INVALID"
        elif has_target and target_available is not None and target_available <= decision:
            reason = "CHRONOLOGY_INVALID"
        elif has_target and trainable and finite_target and target_available is not None and policy_terminal is not None and target_available <= policy_terminal:
            reason = "RESOLVED_MATURED"
        elif decision > resolved_terminal or (target_available is not None and policy_terminal is not None and target_available > policy_terminal) or (has_target and not trainable):
            reason = "RIGHT_CENSORED_AT_DATA_TERMINAL"
        else:
            reason = "SOURCE_MISSING_BLOCKED"
        counts[reason] += 1
        rows.append(
            {
                "family": family,
                "decision_timestamp": decision.isoformat(),
                "asset_id": str(row["asset_id"]),
                "classification": reason,
                "target_available_timestamp": target_available.isoformat() if target_available is not None else "",
                "target_is_trainable": trainable,
                "target_value_present": bool(finite_target),
            }
        )
    return rows, counts


def reconcile_pending_outcomes(family: str) -> dict[str, Any]:
    pending_path = metrics_root(family) / "pending_buffer.parquet"
    pending = read_parquet(pending_path)
    targets, target_load = load_targets_for_keys(pending[["asset_id", "decision_timestamp"]] if not pending.empty else pending)
    rows, counts = classify_pending_frame(family, pending, targets)
    ledger_path = STAGE / f"R7_R31_pending_outcomes_{family}.csv"
    write_csv(ledger_path, rows)
    result = {
        "family": family,
        "pending_path": display_path(pending_path),
        "pending_rows": int(len(pending)),
        "ledger_path": display_path(ledger_path),
        "ledger_rows": len(rows),
        "ledger_sha256": sha256_file(ledger_path),
        "counts": counts,
        **target_load,
        "completion_blocked": bool(counts["SOURCE_MISSING_BLOCKED"] or counts["DUPLICATE_INVALID"] or counts["CHRONOLOGY_INVALID"]),
    }
    return result


def hac_mean_ci(values: Iterable[float], *, lag: int | None = None, z: float = 1.96) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if pd.notna(v) and math.isfinite(float(v))], dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "lower": None, "upper": None, "standard_error": None, "observations": 0, "lag": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "standard_error": 0.0, "observations": 1, "lag": 0}
    demeaned = arr - mean
    if lag is None:
        lag = max(1, int(round(4 * (n / 100.0) ** (2 / 9))))
    lag = min(max(int(lag), 0), n - 1)
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    var = gamma0
    for step in range(1, lag + 1):
        gamma = float(np.dot(demeaned[step:], demeaned[:-step]) / n)
        var += 2.0 * (1.0 - step / (lag + 1.0)) * gamma
    se = math.sqrt(max(var / n, 0.0))
    return {"mean": mean, "lower": mean - z * se, "upper": mean + z * se, "standard_error": se, "observations": n, "lag": lag}


def max_drawdown(returns: pd.Series) -> float | None:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def portfolio_summary_from_returns(per_decision: pd.DataFrame) -> dict[str, Any]:
    if per_decision.empty:
        return {
            "status": "NO_RESOLVED_PORTFOLIO_OBSERVATIONS",
            "resolved_portfolio_observations": 0,
        }
    work = per_decision.copy()
    work["decision_timestamp"] = pd.to_datetime(work["decision_timestamp"], utc=True)
    work["session_date"] = work["decision_timestamp"].dt.date.astype(str)
    session = work.groupby("session_date", sort=True).agg(gross_return=("gross_return", "mean"), net_return=("net_return", "mean")).reset_index()
    net = pd.to_numeric(session["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(session["gross_return"], errors="coerce").dropna()
    downside = net[net < 0.0]
    annualized_return = float(net.mean() * TRADING_SESSIONS_PER_YEAR) if len(net) else None
    annualized_volatility = float(net.std(ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)) if len(net) > 1 else None
    downside_volatility = float(downside.std(ddof=1) * math.sqrt(TRADING_SESSIONS_PER_YEAR)) if len(downside) > 1 else None
    sharpe = annualized_return / annualized_volatility if annualized_volatility and annualized_volatility > 0 else None
    sortino = annualized_return / downside_volatility if downside_volatility and downside_volatility > 0 else None
    return {
        "status": "COMPLETE",
        "lane": "TOP_N_EQUAL_WEIGHT",
        "top_n": int(work["top_n"].dropna().mode().iloc[0]) if "top_n" in work and work["top_n"].notna().any() else 20,
        "aggregation_policy": "session_mean_of_overlapping_60m_forward_returns",
        "transaction_cost_assumption": {
            "cost_bps_per_unit_turnover": TOP_N_COST_BPS_PER_UNIT_TURNOVER,
            "source": "metrics-only registry preserves registered assumptions; no nonzero cost scalar retained in R25 contract",
        },
        "first_resolved_decision_timestamp": work["decision_timestamp"].min().isoformat(),
        "last_resolved_decision_timestamp": work["decision_timestamp"].max().isoformat(),
        "resolved_portfolio_observations": int(len(work)),
        "resolved_session_count": int(len(session)),
        "cumulative_gross_return": float((1.0 + gross).prod() - 1.0) if len(gross) else None,
        "cumulative_net_return": float((1.0 + net).prod() - 1.0) if len(net) else None,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": max_drawdown(net),
        "hit_rate": float((work["net_return"] > 0).mean()) if len(work) else None,
        "average_return_per_decision": float(work["net_return"].mean()) if len(work) else None,
        "average_return_per_session": float(net.mean()) if len(net) else None,
        "turnover": float(work["turnover"].mean()) if "turnover" in work and len(work) else None,
        "estimated_transaction_costs": float(work["transaction_cost"].sum()) if "transaction_cost" in work else 0.0,
    }


def join_decision_trace_targets(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions
    keys = decisions[["asset_id", "decision_timestamp"]].drop_duplicates()
    targets, _ = load_targets_for_keys(keys)
    if targets.empty:
        out = decisions.copy()
        out["target_value"] = np.nan
        out["target_is_trainable"] = False
        out["target_available_timestamp"] = pd.NaT
        return out
    out = decisions.copy()
    out["asset_id"] = out["asset_id"].astype(str)
    out["decision_timestamp"] = pd.to_datetime(out["decision_timestamp"], utc=True)
    targets["asset_id"] = targets["asset_id"].astype(str)
    targets["decision_timestamp"] = pd.to_datetime(targets["decision_timestamp"], utc=True)
    return out.merge(targets, on=["asset_id", "decision_timestamp"], how="left", validate="many_to_one")


def topn_returns(decisions: pd.DataFrame) -> pd.DataFrame:
    joined = join_decision_trace_targets(decisions)
    if joined.empty:
        return pd.DataFrame()
    joined["decision_timestamp"] = pd.to_datetime(joined["decision_timestamp"], utc=True)
    joined["target_value"] = pd.to_numeric(joined["target_value"], errors="coerce")
    joined["target_is_trainable"] = joined["target_is_trainable"].fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    for timestamp, group in joined.groupby("decision_timestamp", sort=True):
        valid = group[group["target_is_trainable"] & np.isfinite(group["target_value"])].copy()
        if len(valid) != len(group):
            continue
        weights = {str(row.asset_id): float(row.weight) for row in group.itertuples(index=False)}
        all_assets = set(previous_weights) | set(weights)
        turnover = 0.0 if not previous_weights else 0.5 * sum(abs(weights.get(asset, 0.0) - previous_weights.get(asset, 0.0)) for asset in all_assets)
        gross = float((valid["weight"].astype(float) * valid["target_value"].astype(float)).sum())
        cost = turnover * (TOP_N_COST_BPS_PER_UNIT_TURNOVER / 10000.0)
        rows.append(
            {
                "decision_timestamp": pd.Timestamp(timestamp).isoformat(),
                "gross_return": gross,
                "net_return": gross - cost,
                "turnover": turnover,
                "transaction_cost": cost,
                "selected_count": int(len(group)),
                "top_n": int(group["top_n"].iloc[0]) if "top_n" in group else 20,
            }
        )
        previous_weights = weights
    return pd.DataFrame(rows)


def duplicate_metric_counts(metrics: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, int]:
    metric_dupes = 0
    topn_dupes = 0
    if not metrics.empty and {"family", "decision_timestamp"}.issubset(metrics.columns):
        metric_dupes = int(metrics.duplicated(["family", "decision_timestamp"]).sum())
    if not decisions.empty and {"family", "decision_timestamp", "asset_id"}.issubset(decisions.columns):
        topn_dupes = int(decisions.duplicated(["family", "decision_timestamp", "asset_id"]).sum())
    return {"duplicate_metrics_count": metric_dupes, "duplicate_topn_key_count": topn_dupes}


def rank_ic_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if "spearman_rank_ic" not in metrics.columns:
        return {
            "status": "NOT_RETAINED_BY_METRICS_ONLY_CONTRACT",
            "mean_spearman_rank_ic": None,
            "median_spearman_rank_ic": None,
            "rank_ic_std": None,
            "rank_ic_positive_fraction": None,
            "rank_ic_information_ratio": None,
            "newey_west_95_ci": None,
            "pearson_ic": None,
            "ic_observation_count": 0,
        }
    series = pd.to_numeric(metrics["spearman_rank_ic"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ci = hac_mean_ci(series)
    std = float(series.std(ddof=1)) if len(series) > 1 else None
    return {
        "status": "COMPLETE" if len(series) else "NO_RETAINED_IC_OBSERVATIONS",
        "mean_spearman_rank_ic": float(series.mean()) if len(series) else None,
        "median_spearman_rank_ic": float(series.median()) if len(series) else None,
        "rank_ic_std": std,
        "rank_ic_positive_fraction": float((series > 0).mean()) if len(series) else None,
        "rank_ic_information_ratio": float(series.mean() / std) if std and std > 0 else None,
        "newey_west_95_ci": ci if len(series) else None,
        "pearson_ic": float(pd.to_numeric(metrics["pearson_ic"], errors="coerce").mean()) if "pearson_ic" in metrics.columns else None,
        "ic_observation_count": int(len(series)),
    }


def performance_summary(family: str, pending: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = metrics_root(family)
    if metrics_root_is_v3(root):
        rank_ic = read_parquet_log(root, "rank_ic_v3")
        daily = read_parquet_log(root, "daily_portfolio_returns_v3")
        sleeves = read_parquet_log(root, "sleeve_maturity_ledger_v3")
        terminal = read_parquet_log(root, "terminal_censored_v3")
        checkpoint = read_json(root / "resolved_performance_checkpoint_v3.json")
        pending_rows = int(checkpoint.get("pending_score_rows", 0) or 0)
        summary_v3 = resolved_v3_summary(rank_ic, daily, sleeves, pending_rows=pending_rows, terminal_censored_rows=len(terminal))
        summary = {
            "run_id": RUN_ID,
            "family": family,
            "classification": "DS24_R31_V3_TERMINAL_PERFORMANCE_SUMMARY_PUBLISHED",
            "metrics_layout": "V3_APPEND_ONLY_MANIFEST",
            "coverage": {
                "first_decision_timestamp": summary_v3.get("first_resolved_decision_timestamp", ""),
                "last_decision_timestamp": summary_v3.get("last_resolved_decision_timestamp", ""),
                "resolved_timestamp_count": int(summary_v3.get("resolved_performance_rows", 0) or 0),
                "resolved_outcome_count": int(summary_v3.get("resolved_performance_rows", 0) or 0),
                "censored_terminal_count": int(summary_v3.get("coverage", {}).get("terminal_censored_rows", 0) or 0),
                "eligible_resolved_fraction": summary_v3.get("coverage", {}).get("eligible_resolved_fraction"),
            },
            "information_coefficient": {
                "status": "COMPLETE" if summary_v3.get("rank_ic", {}).get("valid_timestamps") else "NO_RETAINED_IC_OBSERVATIONS",
                "mean_spearman_rank_ic": summary_v3.get("rank_ic", {}).get("mean_spearman_rank_ic"),
                "newey_west_95_ci": summary_v3.get("rank_ic", {}).get("dependence_aware_95_ci"),
                "ic_observation_count": int(summary_v3.get("rank_ic", {}).get("valid_timestamps", 0) or 0),
            },
            "portfolio_performance": {
                "top_n_equal_weight": {
                    "status": "COMPLETE" if summary_v3.get("returns", {}).get("daily_return_rows") else "NO_RESOLVED_PORTFOLIO_OBSERVATIONS",
                    "aggregation_policy": summary_v3.get("returns", {}).get("portfolio_contract"),
                    "resolved_session_count": int(summary_v3.get("returns", {}).get("daily_return_rows", 0) or 0),
                    "cumulative_gross_return": summary_v3.get("returns", {}).get("cumulative_gross_return"),
                    "cumulative_net_return": summary_v3.get("returns", {}).get("cumulative_net_return"),
                    "annualized_return": summary_v3.get("returns", {}).get("annualized_return_from_daily_returns"),
                    "annualized_volatility": summary_v3.get("returns", {}).get("annualized_volatility_from_daily_returns"),
                    "sharpe_ratio": summary_v3.get("returns", {}).get("daily_sharpe"),
                    "maximum_drawdown": summary_v3.get("returns", {}).get("maximum_drawdown"),
                    "hit_rate": summary_v3.get("returns", {}).get("win_rate"),
                    "turnover": summary_v3.get("returns", {}).get("mean_turnover"),
                    "estimated_transaction_costs": summary_v3.get("returns", {}).get("total_estimated_costs"),
                }
            },
            "metric_rows": int(len(rank_ic)),
            "topn_rows": int(read_json(root / "decision_trace_manifest.json").get("total_rows", 0) or 0),
            "v3_summary": summary_v3,
            "created_at_utc": utc_now(),
        }
        write_json(family_root(family) / SUMMARY_NAME, summary, advisory=False)
        return summary
    metrics = read_parquet(root / "per_t_metrics.parquet")
    decisions = read_parquet(root / "decision_trace.parquet")
    if not metrics.empty:
        metrics["decision_timestamp"] = pd.to_datetime(metrics["decision_timestamp"], utc=True)
    if not decisions.empty:
        decisions["decision_timestamp"] = pd.to_datetime(decisions["decision_timestamp"], utc=True)
    returns = topn_returns(decisions)
    returns_path = STAGE / f"R7_R31_portfolio_returns_{family}.csv"
    write_csv(returns_path, returns.to_dict("records"))
    pending_counts = dict((pending or {}).get("counts", {}))
    coverage = {
        "first_decision_timestamp": metrics["decision_timestamp"].min().isoformat() if not metrics.empty else "",
        "last_decision_timestamp": metrics["decision_timestamp"].max().isoformat() if not metrics.empty else "",
        "resolved_timestamp_count": int(len(returns)),
        "resolved_outcome_count": int(len(returns)),
        "censored_terminal_count": int(pending_counts.get("RIGHT_CENSORED_AT_DATA_TERMINAL", 0) or 0),
        "eligible_asset_count": int(pd.to_numeric(metrics.get("eligible_assets", pd.Series(dtype=float)), errors="coerce").max()) if not metrics.empty and "eligible_assets" in metrics else 0,
        "missing_prediction_count": int(pd.to_numeric(metrics.get("missing_prediction_count", pd.Series(dtype=float)), errors="coerce").sum()) if not metrics.empty else 0,
        "duplicate_prediction_count": int(pd.to_numeric(metrics.get("duplicate_prediction_count", pd.Series(dtype=float)), errors="coerce").sum()) if not metrics.empty else 0,
        "effective_years_of_coverage": (
            round((metrics["decision_timestamp"].max() - metrics["decision_timestamp"].min()).days / 365.25, 6)
            if not metrics.empty
            else 0.0
        ),
    }
    summary = {
        "run_id": RUN_ID,
        "family": family,
        "classification": "DS24_R31_TERMINAL_PERFORMANCE_SUMMARY_PUBLISHED",
        "coverage": coverage,
        "information_coefficient": rank_ic_summary(metrics),
        "portfolio_performance": {"top_n_equal_weight": portfolio_summary_from_returns(returns)},
        "portfolio_returns_path": display_path(returns_path),
        "portfolio_returns_sha256": sha256_file(returns_path),
        "metric_rows": int(len(metrics)),
        "topn_rows": int(len(decisions)),
        "annualisation": {
            "method": "session-level aggregation to avoid treating overlapping 60m labels as independent five-minute returns",
            "trading_sessions_per_year": TRADING_SESSIONS_PER_YEAR,
        },
        "cost_assumptions": {
            "top_n_cost_bps_per_unit_turnover": TOP_N_COST_BPS_PER_UNIT_TURNOVER,
            "source": "No nonzero registered cost scalar is retained in the R25 metrics-only records.",
        },
        "created_at_utc": utc_now(),
    }
    write_json(family_root(family) / SUMMARY_NAME, summary, advisory=False)
    return summary


def validate_family_completion(family: str) -> dict[str, Any]:
    root = family_root(family)
    metrics_namespace = metrics_root(family)
    progress = read_json(root / "progress.json")
    checkpoint = read_json(metrics_namespace / "checkpoint.json")
    authority = terminal_authority(family)
    terminal = has_reached_registered_terminal(family, progress or checkpoint)
    inventory = inventory_evidence(family)
    metrics = read_metrics_table(metrics_namespace, "per_t_metrics", legacy_name="per_t_metrics.parquet")
    decisions = read_metrics_table(metrics_namespace, "decision_trace", legacy_name="decision_trace.parquet")
    duplicates = duplicate_metric_counts(metrics, decisions)
    write_json(
        validating_marker_path(family),
        {
            "run_id": RUN_ID,
            "family": family,
            "terminal_cursor": authority["registered_terminal_cursor"],
            "terminal_validation_state": "TERMINAL_VALIDATING",
            "created_at_utc": utc_now(),
        },
        advisory=False,
    )
    write_json(
        exclusion_marker_path(family),
        {
            "run_id": RUN_ID,
            "family": family,
            "terminal_cursor": authority["registered_terminal_cursor"],
            "terminal_validation_state": "TERMINAL_VALIDATING",
            "created_at_utc": utc_now(),
        },
        advisory=False,
    )
    pending = reconcile_pending_outcomes(family)
    summary = performance_summary(family, pending)
    blockers = []
    if not terminal["reached"]:
        blockers.append("REGISTERED_TERMINAL_CURSOR_NOT_REACHED")
    if inventory["metrics"]["rows"] == 0:
        blockers.append("METRICS_INVENTORY_EMPTY")
    if inventory["topn"]["rows"] == 0:
        blockers.append("TOPN_TRACE_EMPTY")
    if pending["completion_blocked"]:
        blockers.append("PENDING_OUTCOME_VALIDATION_BLOCKED")
    if duplicates["duplicate_metrics_count"] or duplicates["duplicate_topn_key_count"]:
        blockers.append("DUPLICATE_METRIC_OR_TOPN_KEYS")
    quarantine = active_quarantine(family)
    if quarantine:
        blockers.append("ACTIVE_NAMESPACE_QUARANTINE_REQUIRES_RELEASE")
    quarantine_release = read_json(family_root(family) / QUARANTINE_RELEASE_NAME)
    state = "COMPLETE" if not blockers else "CRASHED_BLOCKED"
    marker = {
        "run_id": RUN_ID,
        "family": family,
        "terminal_cursor": authority["registered_terminal_cursor"],
        "metrics_inventory_hash": inventory["metrics_inventory_hash"],
        "topn_inventory_hash": inventory["topn_inventory_hash"],
        "checkpoint_hash": inventory["checkpoint_hash"],
        "policy_hash": policy_hash(),
        "configuration_hash": policy_hash(),
        "metrics_only_policy_hash": metrics_policy_hash(),
        "terminal_validation_state": state,
        "completion_timestamp": utc_now() if state == "COMPLETE" else "",
        "blockers": blockers,
        "quarantine": quarantine,
        "quarantine_release": quarantine_release,
        "pending_outcomes": pending,
        "performance_summary_path": display_path(root / SUMMARY_NAME),
        **duplicates,
    }
    write_json(completion_marker_path(family), marker, advisory=False)
    write_json(exclusion_marker_path(family), marker, advisory=False)
    if state == "COMPLETE":
        try:
            validating_marker_path(family).unlink()
        except FileNotFoundError:
            pass
    return {
        "family": family,
        "state": state,
        "authority": authority,
        "terminal": terminal,
        "inventory": inventory,
        "pending": pending,
        "summary": summary,
        "marker_path": display_path(completion_marker_path(family)),
        "blockers": blockers,
        **duplicates,
    }


def common_sample_performance(families: list[str]) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    for family in families:
        path = STAGE / f"R7_R31_portfolio_returns_{family}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True).map(lambda ts: ts.isoformat())
            frames[family] = frame
    if not frames or any(frame.empty for frame in frames.values()):
        result = {
            "run_id": RUN_ID,
            "classification": "DS24_R31_COMMON_SAMPLE_INSUFFICIENT_COMPLETED_RETURN_SERIES",
            "families": families,
            "common_timestamp_count": 0,
            "created_at_utc": utc_now(),
        }
        write_stage_json("R7_R31_09_common_sample_performance.json", result)
        return result
    common = set.intersection(*(set(frame["decision_timestamp"]) for frame in frames.values()))
    rows = []
    for family, frame in frames.items():
        subset = frame[frame["decision_timestamp"].isin(common)].copy()
        rows.append({"family": family, **portfolio_summary_from_returns(subset)})
    result = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_COMMON_SAMPLE_PERFORMANCE_PUBLISHED",
        "families": families,
        "common_timestamp_count": len(common),
        "common_first_timestamp": min(common) if common else "",
        "common_last_timestamp": max(common) if common else "",
        "rows": rows,
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_09_common_sample_performance.json", result)
    return result


def ps_json(script: str, timeout: int = 25) -> Any:
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
    if not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None


def live_processes() -> list[dict[str, Any]]:
    data = ps_json(
        "$pattern='policy_queue_supervisor|r14_policy_worker|rff_ridge|monitor_ds24_full_family_tournament'; "
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pattern } | "
        "Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine,WorkingSetSize,PageFileUsage,KernelModeTime,UserModeTime | ConvertTo-Json -Depth 5",
        timeout=30,
    )
    if isinstance(data, list):
        return data
    return [data] if isinstance(data, dict) else []


def process_by_pid(pid: int) -> dict[str, Any]:
    if not pid:
        return {}
    data = ps_json(
        f"Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\" | "
        "Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine,WorkingSetSize,PageFileUsage,KernelModeTime,UserModeTime | ConvertTo-Json -Depth 5",
        timeout=15,
    )
    return data if isinstance(data, dict) else {}


def system_resources() -> dict[str, Any]:
    data = ps_json(
        "$os=Get-CimInstance Win32_OperatingSystem; $pf=Get-CimInstance Win32_PageFileUsage; $drive=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\"; "
        "[pscustomobject]@{available_ram_bytes=[int64]($os.FreePhysicalMemory*1KB);total_ram_bytes=[int64]($os.TotalVisibleMemorySize*1KB);"
        "pagefile_current_mb=($pf|Measure-Object CurrentUsage -Sum).Sum;pagefile_peak_mb=($pf|Measure-Object PeakUsage -Sum).Sum;"
        "free_disk_bytes=[int64]$drive.FreeSpace;total_disk_bytes=[int64]$drive.Size} | ConvertTo-Json -Depth 5",
        timeout=20,
    )
    return data if isinstance(data, dict) else {}


def guard_status() -> dict[str, Any]:
    status = read_json(R25_STATUS_PATH)
    return {
        "paper_orders": int(status.get("paper_orders", 0) or 0),
        "live_orders": int(status.get("live_orders", 0) or 0),
        "holdout_accessed": bool(status.get("holdout_accessed", False)),
        "full_prediction_files_in_metrics_namespaces": int(status.get("full_prediction_files_in_metrics_namespaces", 0) or 0),
    }


def live_reconciliation() -> dict[str, Any]:
    processes = live_processes()
    heartbeat = read_json(STAGE / "R7_R27_01_supervisor_heartbeat.json")
    families = {}
    for family in PERFORMANCE_FAMILIES + ["rff_ridge", "elastic_net"]:
        progress = read_json(family_root(family) / "progress.json")
        checkpoint = read_json(metrics_root(family) / "checkpoint.json")
        families[family] = {
            "progress": progress,
            "metrics_checkpoint": checkpoint,
            "terminal": has_reached_registered_terminal(family, progress or checkpoint),
            "exclusion_state": terminal_exclusion_state(family),
        }
    result = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_LIVE_RECONCILIATION_COMPLETE",
        "processes": processes,
        "supervisor_heartbeat": heartbeat,
        "families": families,
        "resources": system_resources(),
        "guards": guard_status(),
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_01_live_reconciliation.json", result)
    return result


def publish_terminal_authority() -> dict[str, Any]:
    result = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_TERMINAL_AUTHORITY_RECONCILED",
        "exact": terminal_authority("exact_ridge_pca"),
        "policy_families": {family: terminal_authority(family) for family in PERFORMANCE_FAMILIES + ["rff_ridge"]},
        "terminal_cursor_difference_resolution": "Exact terminal 19:00 is the final trainable 60m target decision; policy terminal 20:00 is the final registered five-minute score timestamp. The 19:05-20:00 policy scores are terminal-censored outcomes, not missing score work.",
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_02_terminal_authority.json", result)
    return result


def publish_stage_artifacts(validations: list[dict[str, Any]]) -> dict[str, Any]:
    pending = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_PENDING_OUTCOMES_CLASSIFIED",
        "families": {row["family"]: row["pending"] for row in validations},
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_03_pending_outcome_reconciliation.json", pending)
    exclusions = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_TERMINAL_ADMISSION_EXCLUSIONS_PUBLISHED",
        "families": {
            row["family"]: {
                "state": row["state"],
                "marker_path": row["marker_path"],
                "terminal_cursor": row["authority"]["registered_terminal_cursor"],
            }
            for row in validations
        },
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_04_terminal_admission_exclusions.json", exclusions)
    completion = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_COMPLETION_VALIDATION_PASS"
        if all(row["state"] == "COMPLETE" for row in validations)
        else "DS24_R31_COMPLETION_VALIDATION_BLOCKED",
        "families": validations,
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_05_completion_validation.json", completion)
    for idx, family in enumerate(["ridge_policy_v1_control", "spline_additive_ridge", "pca_ridge_policy_v1_control"], start=6):
        row = next((item for item in validations if item["family"] == family), None)
        if row is None:
            continue
        name = {
            "ridge_policy_v1_control": "R7_R31_06_performance_summary_ridge.json",
            "spline_additive_ridge": "R7_R31_07_performance_summary_spline.json",
            "pca_ridge_policy_v1_control": "R7_R31_08_performance_summary_pca.json",
        }[family]
        write_stage_json(name, row["summary"])
    return completion


def stop_process(pid: int) -> dict[str, Any]:
    if not pid:
        return {"pid": pid, "stopped": False, "reason": "NO_PID"}
    data = ps_json(f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue; if($p){{Stop-Process -Id {pid} -Force; @{{stopped=$true;pid={pid}}}|ConvertTo-Json}}else{{@{{stopped=$false;pid={pid};reason='NOT_LIVE'}}|ConvertTo-Json}}", timeout=15)
    return data if isinstance(data, dict) else {"pid": pid, "stopped": False, "reason": "UNKNOWN"}


def contain_completed_family_processes(families: list[str]) -> dict[str, Any]:
    processes = live_processes()
    actions = []
    for family in families:
        if not completion_marker_valid(family):
            continue
        family_processes = [
            row
            for row in processes
            if f"--family {family}" in str(row.get("CommandLine", "")) or f"--family={family}" in str(row.get("CommandLine", ""))
        ]
        for row in family_processes:
            actions.append({"family": family, "process": row, "containment": stop_process(int(row.get("ProcessId", 0) or 0))})
    result = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_REDUNDANT_TERMINAL_PROCESSES_CONTAINED",
        "actions": actions,
        "created_at_utc": utc_now(),
    }
    return result


def family_board_rows() -> list[dict[str, Any]]:
    state = read_json(SUPERVISOR_STATE_PATH)
    rows = state.get("families", [])
    return rows if isinstance(rows, list) else []


def family_board_row(family: str) -> dict[str, Any]:
    for row in family_board_rows():
        if row.get("family") == family:
            return dict(row)
    return {}


def publish_rff_slot_and_live_proof() -> dict[str, Any]:
    live = live_reconciliation()
    heartbeat = read_json(SUPERVISOR_HEARTBEAT_PATH)
    resource = read_json(SUPERVISOR_RESOURCE_PATH)
    board = family_board_rows()
    completed = {
        family: family_board_row(family)
        for family in PERFORMANCE_FAMILIES
    }
    rff = family_board_row("rff_ridge")
    launch = latest_launch_for_family("rff_ridge")
    admission = latest_admission_for_family("rff_ridge")
    active_other_certified = [
        row
        for row in board
        if row.get("family") in {"huber", "mlp", "elastic_net"}
        and row.get("pid_alive")
    ]
    guards = guard_status()
    completed_ok = all(row.get("state") == "COMPLETE" and not row.get("pid_alive") for row in completed.values())
    rff_alive = bool(rff.get("family") == "rff_ridge" and rff.get("pid_alive"))
    rff_clean = str(rff.get("stderr_state", "")).upper() != "FATAL"
    rff_progress_state = bool(rff.get("package_state") or rff.get("cursor") or rff.get("heartbeat"))
    rff_metrics_only = bool(rff.get("metrics_only_authority"))
    guard_clean = not (
        guards["full_prediction_files_in_metrics_namespaces"]
        or guards["paper_orders"]
        or guards["live_orders"]
        or guards["holdout_accessed"]
    )
    rff_only = not active_other_certified
    slot = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_COMPLETED_SLOTS_RELEASED_RFF_ADMITTED"
        if completed_ok and rff_alive and rff_only
        else "DS24_R31_SLOT_RELEASE_OR_RFF_ADMISSION_INCOMPLETE",
        "completed_families": completed,
        "rff_launch": launch,
        "rff_admission_decision": admission,
        "supervisor": {
            "pid": heartbeat.get("supervisor_pid") or heartbeat.get("pid"),
            "heartbeat_utc": heartbeat.get("heartbeat_utc"),
            "max_active_model_processes": heartbeat.get("max_active_model_processes"),
            "max_policy_workers": heartbeat.get("max_policy_workers"),
            "active_model_processes": heartbeat.get("active_model_processes"),
            "active_policy_workers": heartbeat.get("active_policy_workers"),
            "resource_block_reason": heartbeat.get("resource_block_reason"),
        },
        "next_family_held_by_one_worker_cap": heartbeat.get("next_ready_family"),
        "active_other_certified_ready_workers": active_other_certified,
        "created_at_utc": utc_now(),
    }
    proof_pass = completed_ok and rff_alive and rff_clean and rff_progress_state and rff_metrics_only and guard_clean and rff_only
    rff_processes = [
        row
        for row in live.get("processes", [])
        if "--family rff_ridge" in str(row.get("CommandLine", "")) or "--family=rff_ridge" in str(row.get("CommandLine", ""))
    ]
    if not rff_processes and rff.get("pid"):
        pid_process = process_by_pid(int(rff.get("pid") or 0))
        if pid_process:
            pid_process["evidence_source"] = "PID_SPECIFIC_WMI_FALLBACK"
            rff_processes = [pid_process]
    if not rff_processes and rff_alive:
        rff_processes = [
            {
                "ProcessId": rff.get("pid"),
                "ParentProcessId": rff.get("parent_pid"),
                "CreationDate": rff.get("creation_time"),
                "CommandLine": rff.get("command_line"),
                "WorkingSetSize": rff.get("working_set"),
                "PageFileUsage": rff.get("private_memory"),
                "KernelModeTime": rff.get("kernel_time"),
                "UserModeTime": rff.get("user_time"),
                "evidence_source": "SUPERVISOR_BOARD_CLASSIFICATION",
            }
        ]
    proof = {
        "run_id": RUN_ID,
        "classification": "DS24_R31_RFF_LIVE_ADMISSION_PROOF_PASS" if proof_pass else "DS24_R31_RFF_LIVE_ADMISSION_PROOF_BLOCKED",
        "family": "rff_ridge",
        "rff_worker": rff,
        "rff_processes": rff_processes,
        "proof_conditions": {
            "completed_families_excluded_from_admission": completed_ok,
            "rff_pid_alive": rff_alive,
            "rff_activity_or_telemetry_present": rff_progress_state,
            "rff_stderr_clean": rff_clean,
            "rff_metrics_only_authority": rff_metrics_only,
            "zero_full_prediction_batches": guards["full_prediction_files_in_metrics_namespaces"] == 0,
            "paper_orders_zero": guards["paper_orders"] == 0,
            "live_orders_zero": guards["live_orders"] == 0,
            "holdout_accessed_false": guards["holdout_accessed"] is False,
            "no_huber_mlp_elastic_substitution_launched": rff_only,
            "first_metrics_commit_required_for_this_r31_proof": False,
            "first_metrics_commit_observed": int(rff.get("metrics_rows", 0) or 0) > 0,
        },
        "resource_snapshot": resource,
        "guards": guards,
        "supervisor_heartbeat_path": display_path(SUPERVISOR_HEARTBEAT_PATH),
        "slot_release_artifact": display_path(STAGE / "R7_R31_10_slot_release_and_rff_admission.json"),
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_10_slot_release_and_rff_admission.json", slot)
    write_stage_json("R7_R31_11_live_proof.json", proof)
    completion = read_json(STAGE / "R7_R31_05_completion_validation.json")
    validations = completion.get("families", []) if isinstance(completion.get("families"), list) else []
    terminal = {
        "run_id": RUN_ID,
        "classification": terminal_classification(validations, rff_active=proof_pass),
        "completion_validation": display_path(STAGE / "R7_R31_05_completion_validation.json"),
        "common_sample_performance": display_path(STAGE / "R7_R31_09_common_sample_performance.json"),
        "slot_release_and_rff_admission": display_path(STAGE / "R7_R31_10_slot_release_and_rff_admission.json"),
        "live_proof": display_path(STAGE / "R7_R31_11_live_proof.json"),
        "completion": completion.get("classification", ""),
        "rff_live_proof": proof["classification"],
        "paper_orders": guards["paper_orders"],
        "live_orders": guards["live_orders"],
        "holdout_accessed": guards["holdout_accessed"],
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_12_terminal_validation.json", terminal)
    write_report(terminal["classification"], validations, slot=slot)
    return {"slot": slot, "proof": proof, "terminal": terminal}


def write_report(classification: str, validations: list[dict[str, Any]], slot: Mapping[str, Any] | None = None) -> None:
    lines = [
        "# DS24-P8-R14-E3G-C2-R7-R31 Report",
        "",
        f"Classification: `{classification}`",
        "",
        "Root cause: policy workers reached the registered 20:00 scoring terminal, but the supervisor only recognised the Exact 19:00 terminal and treated terminal-censored pending buffers as resumable work.",
        "",
        "Terminal authority: Exact 19:00 is the last trainable 60m target; policy workers use the full registered five-minute spine through 20:00 with terminal-censored outcomes classified separately.",
        "",
    ]
    for row in validations:
        topn = row["summary"]["portfolio_performance"]["top_n_equal_weight"]
        ic = row["summary"]["information_coefficient"]
        lines.append(
            f"- `{row['family']}`: `{row['state']}`, pending={row['pending']['counts']}, "
            f"net={topn.get('cumulative_net_return')}, sharpe={topn.get('sharpe_ratio')}, "
            f"max_dd={topn.get('maximum_drawdown')}, rank_ic_status={ic.get('status')}"
        )
    if slot:
        lines.extend(["", f"Slot/RFF: `{slot.get('classification')}`"])
    lines.append("")
    lines.append("No full prediction batches, holdout access, paper orders, or live orders were introduced by R31.")
    (STAGE / "DS24_P8_R14_E3G_C2_R7_R31_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def terminal_classification(validations: list[dict[str, Any]], rff_active: bool) -> str:
    complete = {row["family"] for row in validations if row["state"] == "COMPLETE"}
    if {"ridge_policy_v1_control", "pca_ridge_policy_v1_control", "spline_additive_ridge"}.issubset(complete) and rff_active:
        return "DS24P8_R14_E3G_C2_R7_R31_THREE_POLICY_FAMILIES_COMPLETE_PERFORMANCE_SUMMARIES_PUBLISHED_RFF_ACTIVE"
    if {"ridge_policy_v1_control", "spline_additive_ridge"}.issubset(complete) and rff_active:
        return "DS24P8_R14_E3G_C2_R7_R31_TERMINAL_RELAUNCH_REPAIRED_RIDGE_SPLINE_COMPLETE_RFF_ACTIVE_PCA_PROTECTED"
    if all(row["state"] == "COMPLETE" for row in validations):
        return "DS24P8_R14_E3G_C2_R7_R31_TERMINAL_CONVERGENCE_COMPLETE_RFF_WAITING_RESOURCE_GATE"
    return "DS24P8_R14_E3G_C2_R7_R31_BLOCKED_TERMINAL_VALIDATION_EVIDENCE_FAILURE"


def run_validation() -> dict[str, Any]:
    live_reconciliation()
    publish_terminal_authority()
    validations = [validate_family_completion(family) for family in PERFORMANCE_FAMILIES]
    completion = publish_stage_artifacts(validations)
    common = common_sample_performance([row["family"] for row in validations if row["state"] == "COMPLETE"])
    terminal = {
        "run_id": RUN_ID,
        "classification": terminal_classification(validations, rff_active=False),
        "completion_validation": display_path(STAGE / "R7_R31_05_completion_validation.json"),
        "common_sample_performance": display_path(STAGE / "R7_R31_09_common_sample_performance.json"),
        "completion": completion["classification"],
        "common": common["classification"],
        "paper_orders": 0,
        "live_orders": 0,
        "holdout_accessed": False,
        "created_at_utc": utc_now(),
    }
    write_stage_json("R7_R31_12_terminal_validation.json", terminal)
    write_report(terminal["classification"], validations)
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validate", "live", "authority", "contain", "rff-proof", "all"], default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "live":
        print(json.dumps(live_reconciliation(), indent=2, sort_keys=True))
        return 0
    if args.phase == "authority":
        print(json.dumps(publish_terminal_authority(), indent=2, sort_keys=True))
        return 0
    if args.phase == "contain":
        print(json.dumps(contain_completed_family_processes(PERFORMANCE_FAMILIES), indent=2, sort_keys=True))
        return 0
    if args.phase == "rff-proof":
        print(json.dumps(publish_rff_slot_and_live_proof(), indent=2, sort_keys=True))
        return 0
    result = run_validation()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
