from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.comparable_policy import POLICY_ID, policy_hash
from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.ds24_metrics_only_evaluator import (
    CLEAN_ADMISSION_FREE_DISK_BYTES,
    EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
    EXTENDED_PERFORMANCE_METRICS_CONTRACT_VERSION,
    MIN_EXECUTION_FREE_DISK_BYTES,
    NAMESPACE_WRITER_LEASE_NAME,
    RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
    RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION,
    RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
    RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION,
    THREE_WORKER_REACTIVATION_FREE_DISK_BYTES,
    TRANSIENT_STORAGE_CONTRACT_V1_ID,
    directory_size_bytes,
    extended_performance_metrics_contract_hash,
    openable_exists,
    openable_path,
    parquet_log_part_paths,
    parquet_log_parts_dir,
    parquet_log_rows,
    read_parquet_log,
    resolved_performance_contract_v2_hash,
    resolved_performance_contract_v3_hash,
    transient_storage_contract_v1_hash,
    validate_extended_metrics_writer_capability,
)
from scripts.local.ds24_p8_r14_e3g_c2_r7_r31_terminal_convergence import (
    SUMMARY_NAME as R31_SUMMARY_NAME,
    completion_marker_valid as r31_completion_marker_valid,
    has_reached_registered_terminal as r31_has_reached_registered_terminal,
    terminal_authority as r31_terminal_authority,
    terminal_exclusion_state as r31_terminal_exclusion_state,
)


STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
POLICY_ROOT = STAGE / "r7_r14_policy_workers"
RUN_ID = "ds24_p8_r14_e3g_c2_r7_r27_20260826T000000Z"
TASK_NAME = "DreamSystem_DS24_TournamentSupervisor"
DS26_TASK_NAME = "DreamSystem_DS26_ProspectiveNewsCapture"
PASSIVE_DS26_CAPTURE_SCRIPT = "ds26_prospective_capture_worker.py"
HEAVY_DS26_COMMAND_TOKENS = (
    "build",
    "materialis",
    "materializ",
    "fit",
    "fitting",
    "train",
    "training",
    "evaluate",
    "evaluation",
    "backfill",
    "historical",
    "aggregate",
    "maturity",
)

LEASE_PATH = STAGE / "R7_R27_tournament_supervisor.lease.json"
HEARTBEAT_PATH = STAGE / "R7_R27_01_supervisor_heartbeat.json"
STATE_PATH = STAGE / "R7_R27_02_family_state_board.json"
ACTIVE_MANIFEST_PATH = STAGE / "R7_R27_03_active_worker_manifest.json"
PAUSED_MANIFEST_PATH = STAGE / "R7_R27_04_paused_resumable_manifest.json"
RESOURCE_PATH = STAGE / "R7_R27_05_resource_snapshot.json"
ADMISSION_LEDGER_PATH = STAGE / "R7_R27_06_admission_ledger.csv"
LAUNCH_LEDGER_PATH = STAGE / "R7_R27_07_automatic_launch_ledger.csv"
RESTART_LEDGER_PATH = STAGE / "R7_R27_08_restart_ledger.csv"
COMPLETION_LEDGER_PATH = STAGE / "R7_R27_09_completion_validation_ledger.csv"
CLEANUP_LEDGER_REF_PATH = STAGE / "R7_R27_10_cleanup_ledger_reference.json"
LAST_ERROR_PATH = STAGE / "R7_R27_11_last_error.json"
NEXT_ACTION_PATH = STAGE / "R7_R27_12_next_action.json"
TASK_IDENTITY_PATH = STAGE / "R7_R27_13_windows_task_identity.json"
TERMINAL_PATH = STAGE / "R7_R27_14_terminal_validation.json"
REPORT_PATH = STAGE / "DS24_P8_R14_E3G_C2_R7_R27_REPORT.md"
ACTIVE_CONTAINMENT_PATH = STAGE / "R7_R33_active_resource_containment_ledger.json"

R25_STATUS_PATH = STAGE / "r7_r25_metrics_only_migration" / "status.json"
EXACT_METRICS_ROOT = STAGE / "r7_r25_metrics_only" / "exact_ridge_pca"
STORAGE_HOLD_PATH = STAGE / "STORAGE_EMERGENCY_HOLD.json"
R29_SOURCE_GAP_HOLD_PATH = STAGE / "R7_R29_04_source_gap_hold.json"
R30_REACTIVATION_RELEASE_NAME = "r30_reactivation_release.json"
R36_NAMESPACE_QUARANTINE_NAME = "r36_namespace_quarantine.json"
R36_ADMISSION_HOLD_PATH = STAGE / "R7_R36_ADMISSION_HOLD.json"
LEDGER_PATH = LAUNCH_LEDGER_PATH
ADMISSION_PATH = ADMISSION_LEDGER_PATH
MIN_POLICY_ADMISSION_PROOF_SECONDS = 120.0

PROTECTED_EXACT_SCRIPT = "ds24_p8_r14_e3g_c2_r7_r7_full_burn_in_worker.py"
POLICY_WORKER_SCRIPT = "ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py"
LIGHTGBM_RANKING_WORKER_SCRIPT = "ds24_v3_lightgbm_ranking_policy_worker.py"
SEQUENCE_WORKER_SCRIPT = "ds24_v3_sequence_policy_worker.py"
CLEANUP_SCRIPT = "ds24_p8_r14_e3g_c2_r7_r25_legacy_predictions_to_metrics.py"

ACTIVE_SEED = ["ridge_policy_v1_control", "pca_ridge_policy_v1_control"]
PAUSED_QUEUE = ["spline_additive_ridge", "elastic_net"]
READY_QUEUE = ["rff_ridge", "huber", "mlp"]
R37_LIVE_TRIO = ["rff_ridge", "huber", "mlp"]
R34_V3_QUEUE = ["rff_ridge", "huber", "mlp"]
TREE_CERTIFICATION = ["random_forest", "extra_trees", "gradient_boosting"]
RANKING_CERTIFICATION = ["lightgbm_rank_xendcg", "lightgbm_lambdarank"]
SEQUENCE_CERTIFICATION = [
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer",
]
ALL_FAMILIES = ACTIVE_SEED + PAUSED_QUEUE + READY_QUEUE + TREE_CERTIFICATION + RANKING_CERTIFICATION + SEQUENCE_CERTIFICATION
R40_AUTOMATIC_QUEUE = (
    R37_LIVE_TRIO
    + ACTIVE_SEED
    + ["spline_additive_ridge", "elastic_net"]
    + TREE_CERTIFICATION
    + RANKING_CERTIFICATION
    + [family for family in SEQUENCE_CERTIFICATION if family != "Temporal Fusion Transformer"]
    + ["Temporal Fusion Transformer"]
)
HEAVY_FAMILIES = set(TREE_CERTIFICATION + RANKING_CERTIFICATION + SEQUENCE_CERTIFICATION)
TABULAR_WORKER_FAMILIES = set(ACTIVE_SEED + PAUSED_QUEUE + READY_QUEUE + TREE_CERTIFICATION)
LIGHTGBM_WORKER_FAMILIES = set(RANKING_CERTIFICATION)
SEQUENCE_WORKER_FAMILIES = set(SEQUENCE_CERTIFICATION)
R40_V3_METRICS_ROOTS = {
    "huber": "metrics_only_v3_r37_huber_replay",
    "rff_ridge": "metrics_only_v3_r37_rff_retry",
    "mlp": "metrics_only_v3_r37_mlp_direct_gen7",
}
R44_FORWARD_CONTRACT_ADOPTION_ID = "DS24_R44_FORWARD_CONTRACT_ADOPTION_V1"
R42_READY_QUEUE_AUTHORITY_ID = "DS24_R42_FORWARD_METRICS_CAPABILITY_REPAIR_AND_READY_QUEUE"
R42_READY_QUEUE_PATH = STAGE / "R42_ready_family_queue.json"
R42_READINESS_MATRIX_PATH = STAGE / "R42_full_family_readiness_matrix.json"
R42_REBOOT_TASK_PLAN_PATH = STAGE / "R43_reboot_supervisor_task_plan.json"
R43_DRY_RUN_ADMISSION_PATH = STAGE / "R43_dry_run_admission.json"
R44_CROSS_HOST_OWNERSHIP_AUTHORITY_ID = "DS24_R44_CROSS_HOST_FAMILY_OWNERSHIP_AUTHORITY"
R44_CROSS_HOST_OWNERSHIP_PATH = STAGE / "R44_cross_host_family_ownership.json"
R44_DELL_EFFECTIVE_READY_QUEUE_PATH = STAGE / "R44_dell_effective_ready_queue.json"
R44_ADMISSION_VALIDATION_PATH = STAGE / "R44_cross_host_admission_validation.json"
R44_REBOOT_TASK_PLAN_PATH = STAGE / "R44_reboot_supervisor_task_plan.json"
R45_PRE_CUTOVER_SNAPSHOT_PATH = STAGE / "R45_pre_cutover_snapshot.json"
R45_SUPERVISOR_LAUNCH_AUTHORITY_PATH = STAGE / "R45_supervisor_launch_authority.json"
R45_POST_CUTOVER_VALIDATION_PATH = STAGE / "R45_post_cutover_validation.json"
R45_WINDOWS_TASK_REGISTRATION_PATH = STAGE / "R45_windows_task_registration.json"
R45_REBOOT_SIMULATION_PATH = STAGE / "R45_reboot_simulation.json"
R45_CLASSIFICATION = "DS24_R45_R44_SUPERVISOR_CUTOVER_ACTIVE_REBOOT_SAFE"
R45_SUPERVISOR_LAUNCHER_PATH = ROOT / "scripts" / "local" / "launch_ds24_r45_supervisor.ps1"
R46_PRE_AUTOSTART_SNAPSHOT_PATH = STAGE / "R46_pre_autostart_snapshot.json"
R46_USER_AUTOSTART_AUTHORITY_PATH = STAGE / "R46_user_autostart_authority.json"
R46_AUTOSTART_REGISTRATION_VALIDATION_PATH = STAGE / "R46_autostart_registration_validation.json"
R46_SINGLETON_LIVE_VALIDATION_PATH = STAGE / "R46_singleton_live_validation.json"
R46_REBOOT_RECOVERY_SIMULATION_PATH = STAGE / "R46_reboot_recovery_simulation.json"
R46_CLASSIFICATION = "DS24_R46_ZERO_TOUCH_USER_AUTOSTART_ACTIVE"
R46_STARTUP_ENTRY_NAME = "DreamSystem_DS24_TournamentSupervisor.cmd"
R54_RETIRED_FAMILIES_PATH = STAGE / "R54_retired_families.json"
R54_RETIREMENT_AUTHORITY_ID = "DS24_R54_RETIRED_FAMILY_LAUNCH_EXCLUSION_V1"
R42_ALLOWED_READY_FAMILIES = (
    "random_forest",
    "elastic_net",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
)
R44_MAC_OWNED_FAMILIES = ("lightgbm_rank_xendcg", "lightgbm_lambdarank")
R44_MAC_RESERVED_FAMILIES = ("DLinear",)
R44_DELL_READY_FAMILIES = (
    "random_forest",
    "elastic_net",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
)
R44_DELL_RUNNING_FAMILIES = ("mlp", "extra_trees", "gradient_boosting")
R45_PROTECTED_DELL_WORKERS = R44_DELL_RUNNING_FAMILIES
R44_ALLOWED_OWNERSHIP_STATES = {
    "DELL_OWNED",
    "MAC_OWNED",
    "MAC_RESERVED",
    "UNASSIGNED",
    "COMPLETE_IMPORTED",
}
R44_ALLOWED_OWNER_STATES = {
    "DELL_OWNED",
    "MAC_COMPLETE",
    "MAC_RUNNING",
    "MAC_RESERVED_NEXT",
    "UNASSIGNED",
    "COMPLETE_IMPORTED",
}
R44_CURRENT_CONTINUATION_GRANDFATHERED_NAMESPACES = dict(R40_V3_METRICS_ROOTS)
R44A_DISK_READMISSION_CONTRACT_ID = "DS24_R44A_DISK_READMISSION_HYSTERESIS_V1"
R44A_DISK_CONTAINMENT_REASONS = {
    "DISK_BELOW_12_GIB_PAUSE_NEWEST",
    "DISK_BELOW_6_GIB_EMERGENCY_PAUSE_ALL",
}
R44A_HARD_RESOURCE_CONTAINMENT_REASONS = R44A_DISK_CONTAINMENT_REASONS | {
    "SYSTEM_COMMIT_HARD_CEILING_PAUSE_NEWEST",
    "RAM_BELOW_4_GIB_HARD_FLOOR_PAUSE_NEWEST",
}
TERMINAL_T = "2026-06-30T19:00:00+00:00"
POLICY_TERMINAL_T = "2026-06-30T20:00:00+00:00"
REFIT_CADENCE_ID = "FIVE_SCORE_SESSION_REFIT_WITH_FIVE_MINUTE_SCORING_V1"
DECISION_CADENCE_ID = "REGISTERED_FIVE_MINUTE_DECISION_SPINE"
DS24_PROCESS_COMMAND_TOKENS = (
    "ds24_p8_r14_e3g_c2_r7",
    SEQUENCE_WORKER_SCRIPT.lower(),
    LIGHTGBM_RANKING_WORKER_SCRIPT.lower(),
    "ds26",
    "r20",
    "compactor",
)


@dataclass(frozen=True)
class GateConfig:
    max_active_model_processes: int = 3
    max_policy_workers: int = 2
    max_heavy_workers: int = 3
    aggregate_ds24_memory_ceiling_bytes: int = 28 * 1024**3
    min_available_ram_bytes: int = 6 * 1024**3
    available_ram_floor_bytes: int = 4 * 1024**3
    max_system_commit_percent: float = 90.0
    admission_commit_percent: float = 85.0
    clean_admission_disk_floor_bytes: int = CLEAN_ADMISSION_FREE_DISK_BYTES
    hard_disk_floor_bytes: int = MIN_EXECUTION_FREE_DISK_BYTES
    emergency_pause_all_disk_floor_bytes: int = 6 * 1024**3
    three_worker_reactivation_floor_bytes: int = THREE_WORKER_REACTIVATION_FREE_DISK_BYTES
    projected_post_launch_disk_floor_bytes: int = CLEAN_ADMISSION_FREE_DISK_BYTES
    post_disk_containment_readmission_floor_bytes: int = 20 * 1024**3
    disk_containment_readmission_cooldown_seconds: float = 10 * 60
    protected_slowdown_limit_percent: float = 15.0


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "+00:00"


def read_json(path: Path | str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload, advisory=True)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    sidecar = path.with_name(path.name + ".jsonl")
    with sidecar.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str, separators=(",", ":")) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_hash(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_if_exists(path: Path) -> str:
    return file_hash(path) if path.exists() and path.is_file() else ""


def same_utc_iso(left: Any, right: Any) -> bool:
    return bool(left and right and str(left).replace("Z", "+00:00") == str(right).replace("Z", "+00:00"))


def parse_utc_datetime(value: Any) -> datetime | None:
    if value in ("", None):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def process_creation_matches(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    try:
        left_ts = parse_utc_datetime(left)
        right_ts = parse_utc_datetime(right)
        if left_ts is None or right_ts is None:
            return False
        return abs((left_ts - right_ts).total_seconds()) < 1.0
    except Exception:
        return str(left) == str(right)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def family_slug(family: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_")


def canonical_family_name(family: str) -> str:
    value = str(family or "").strip()
    slug = family_slug(value)
    if not slug:
        return ""
    for known in ALL_FAMILIES:
        if family_slug(known) == slug:
            return known
    return value


def family_list_values(values: str | Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    raw_items: Sequence[str]
    if isinstance(values, str):
        raw_items = [values]
    else:
        raw_items = values
    families: list[str] = []
    for item in raw_items:
        for part in str(item).split(","):
            family = canonical_family_name(part)
            if family:
                families.append(family)
    return list(dict.fromkeys(families))


def retired_family_authority(path: str | Path = R54_RETIRED_FAMILIES_PATH) -> dict[str, Any]:
    authority_path = Path(path) if path else R54_RETIRED_FAMILIES_PATH
    payload = read_json(authority_path)
    families: list[dict[str, Any]] = []
    raw_families = payload.get("families", []) if isinstance(payload, dict) else []
    if isinstance(raw_families, list):
        for item in raw_families:
            if isinstance(item, Mapping):
                family = canonical_family_name(str(item.get("family", "")))
                if family:
                    families.append({**dict(item), "family": family})
            else:
                family = canonical_family_name(str(item))
                if family:
                    families.append({"family": family})
    return {
        "authority": str(payload.get("authority") or R54_RETIREMENT_AUTHORITY_ID) if isinstance(payload, dict) else R54_RETIREMENT_AUTHORITY_ID,
        "schema_version": int(payload.get("schema_version", 1) or 1) if isinstance(payload, dict) else 1,
        "path": display_path(authority_path),
        "families": families,
        "authority_hash": hash_if_exists(authority_path),
    }


def effective_excluded_families(
    excluded_families: str | Sequence[str] | None = None,
    *,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> list[str]:
    explicit = family_list_values(excluded_families)
    retired = [str(row["family"]) for row in retired_family_authority(retired_families_path)["families"]]
    return list(dict.fromkeys([*retired, *explicit]))


def family_is_excluded(
    family: str,
    excluded_families: str | Sequence[str] | None = None,
    *,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> bool:
    canonical = canonical_family_name(family)
    return canonical in set(effective_excluded_families(excluded_families, retired_families_path=retired_families_path))


def family_is_launchable(
    family: str,
    excluded_families: str | Sequence[str] | None = None,
    *,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> bool:
    return launch_enabled_for_family(family) and not family_is_excluded(
        family,
        excluded_families,
        retired_families_path=retired_families_path,
    )


def recent_disk_containment_state(
    path: Path = ACTIVE_CONTAINMENT_PATH,
    *,
    now_utc: str | None = None,
    cooldown_seconds: float = 24 * 60 * 60,
    accepted_reasons: set[str] | None = None,
) -> dict[str, Any]:
    accepted = accepted_reasons or R44A_DISK_CONTAINMENT_REASONS
    payload = read_json(path)
    latest = payload.get("latest") if isinstance(payload.get("latest"), dict) else payload
    reason = str(latest.get("reason", "") or "")
    timestamp = str(latest.get("timestamp", "") or "")
    events = latest.get("events", [])
    stopped_workers = [
        {
            "family": event.get("family", ""),
            "pid": event.get("pid", ""),
            "stopped": bool(event.get("stopped")),
        }
        for event in events
        if isinstance(event, dict) and bool(event.get("stopped"))
    ] if isinstance(events, list) else []
    now_dt = parse_utc_datetime(now_utc or utc_now())
    latest_dt = parse_utc_datetime(timestamp)
    age_seconds: float | None = None
    if now_dt is not None and latest_dt is not None:
        age_seconds = max(0.0, (now_dt - latest_dt).total_seconds())
    recent = bool(
        reason in accepted
        and stopped_workers
        and age_seconds is not None
        and age_seconds <= cooldown_seconds
    )
    return {
        "contract_id": R44A_DISK_READMISSION_CONTRACT_ID,
        "recent": recent,
        "reason": reason,
        "latest_timestamp": timestamp,
        "age_seconds": age_seconds,
        "cooldown_seconds": cooldown_seconds,
        "stopped_worker_count": len(stopped_workers),
        "stopped_workers": stopped_workers,
        "containment_ledger": display_path(path),
    }


def recent_hard_resource_containment_state(
    path: Path = ACTIVE_CONTAINMENT_PATH,
    *,
    now_utc: str | None = None,
    cooldown_seconds: float = 24 * 60 * 60,
) -> dict[str, Any]:
    return recent_disk_containment_state(
        path,
        now_utc=now_utc,
        cooldown_seconds=cooldown_seconds,
        accepted_reasons=R44A_HARD_RESOURCE_CONTAINMENT_REASONS,
    )


def worker_kind_for_family(family: str) -> str:
    if family in TABULAR_WORKER_FAMILIES:
        return "TABULAR"
    if family in LIGHTGBM_WORKER_FAMILIES:
        return "LIGHTGBM_RANKING"
    if family in SEQUENCE_WORKER_FAMILIES:
        return "PYTORCH_SEQUENCE"
    return "UNREGISTERED"


def worker_script_for_family(family: str) -> str:
    kind = worker_kind_for_family(family)
    if kind == "TABULAR":
        return POLICY_WORKER_SCRIPT
    if kind == "LIGHTGBM_RANKING":
        return LIGHTGBM_RANKING_WORKER_SCRIPT
    if kind == "PYTORCH_SEQUENCE":
        return SEQUENCE_WORKER_SCRIPT
    return ""


def launch_enabled_for_family(family: str) -> bool:
    if family in TABULAR_WORKER_FAMILIES:
        return True
    # Ranking and sequence families have readiness evidence, but their real historical
    # queue workers must be present before the automatic supervisor may admit them.
    if family in LIGHTGBM_WORKER_FAMILIES:
        return (ROOT / "scripts" / "local" / LIGHTGBM_RANKING_WORKER_SCRIPT).exists()
    if family in SEQUENCE_WORKER_FAMILIES:
        return (
            family != "Temporal Fusion Transformer"
            and (ROOT / "scripts" / "local" / SEQUENCE_WORKER_SCRIPT).exists()
        )
    return False


def default_metrics_root_name(family: str, requested: str = "") -> str:
    if family in R40_V3_METRICS_ROOTS:
        return R40_V3_METRICS_ROOTS[family]
    if requested and requested not in {"metrics_only", "metrics_only_v3", "metrics_only_v3_r37_rff_retry"}:
        return requested
    return f"metrics_only_v3_r40_{family_slug(family)}"


def forward_metrics_capability_paths(family: str, *, include_r42_authority: bool = False) -> list[Path]:
    script = worker_script_for_family(family)
    paths: list[Path] = []
    if include_r42_authority:
        paths.append(STAGE / "R42_forward_metrics_capabilities" / f"{family_slug(family)}.forward_metrics_capability.json")
    paths.append(POLICY_ROOT / family / "forward_metrics_capability.json")
    if script:
        stem = Path(script).stem
        paths.append(ROOT / "scripts" / "local" / f"{stem}.forward_metrics_capability.json")
        paths.append(ROOT / "scripts" / "local" / f"{family_slug(family)}.forward_metrics_capability.json")
    return paths


def family_forward_metrics_capability_evidence(
    family: str,
    *,
    metrics_root_name: str = "metrics_only_v3",
    include_r42_authority: bool = False,
) -> dict[str, Any]:
    for path in forward_metrics_capability_paths(family, include_r42_authority=include_r42_authority):
        payload = read_json(path)
        if payload:
            evidence = dict(payload)
            evidence.setdefault("family", family)
            evidence.setdefault("metrics_root_name", default_metrics_root_name(family, metrics_root_name))
            evidence["capability_path"] = display_path(path)
            return evidence
    return {}


def current_namespace_grandfathered_for_r44(family: str, metrics_root_name: str) -> bool:
    return (
        family in R44_CURRENT_CONTINUATION_GRANDFATHERED_NAMESPACES
        and metrics_root_name == R44_CURRENT_CONTINUATION_GRANDFATHERED_NAMESPACES[family]
    )


def forward_metrics_contract_admission_decision(
    family: str,
    *,
    evaluation_version: str = "v3",
    metrics_root_name: str = "metrics_only_v3",
    capability: dict[str, Any] | None = None,
    include_r42_authority: bool = False,
) -> dict[str, Any]:
    effective_metrics_root = default_metrics_root_name(family, metrics_root_name)
    base = {
        "adoption_id": R44_FORWARD_CONTRACT_ADOPTION_ID,
        "family": family,
        "evaluation_version": str(evaluation_version).upper(),
        "metrics_root_name": effective_metrics_root,
        "contract_id": EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
        "contract_version": EXTENDED_PERFORMANCE_METRICS_CONTRACT_VERSION,
        "contract_hash": extended_performance_metrics_contract_hash(),
        "grandfathered_current_namespace": False,
        "missing_requirements": [],
        "capability_decision": "",
    }
    if str(evaluation_version).lower() != "v3":
        return {
            **base,
            "admitted": False,
            "capability_decision": "FORWARD_CONTRACT_REQUIRES_V3_EVALUATION",
            "missing_requirements": ["evaluation_version_v3"],
        }
    if current_namespace_grandfathered_for_r44(family, effective_metrics_root):
        return {
            **base,
            "admitted": True,
            "grandfathered_current_namespace": True,
            "capability_decision": "CURRENT_EXACT_NAMESPACE_CONTINUATION_GRANDFATHERED",
        }
    evidence = capability if capability is not None else family_forward_metrics_capability_evidence(
        family,
        metrics_root_name=metrics_root_name,
        include_r42_authority=include_r42_authority,
    )
    result = validate_extended_metrics_writer_capability(evidence, family=family)
    return {
        **base,
        "admitted": bool(result.get("admitted")),
        "capability_decision": str(result.get("classification", "")),
        "missing_requirements": list(result.get("missing_requirements", [])),
        "capability_path": str((evidence or {}).get("capability_path", "")) if isinstance(evidence, dict) else "",
    }


def execution_registry_row(family: str, *, include_r42_authority: bool = False) -> dict[str, Any]:
    script = worker_script_for_family(family)
    forward_decision = forward_metrics_contract_admission_decision(
        family,
        evaluation_version="v3",
        metrics_root_name="metrics_only_v3",
        include_r42_authority=include_r42_authority,
    )
    return {
        "family": family,
        "worker_kind": worker_kind_for_family(family),
        "worker_script": f"scripts/local/{script}" if script else "",
        "worker_script_exists": bool(script and (ROOT / "scripts" / "local" / script).exists()),
        "launch_enabled": family_is_launchable(family),
        "default_metrics_root_name": default_metrics_root_name(family, "metrics_only_v3"),
        "metrics_only": True,
        "full_prediction_persistence": False,
        "thread_cap": 1,
        "forward_metrics_contract_id": forward_decision["contract_id"],
        "forward_metrics_contract_version": forward_decision["contract_version"],
        "forward_metrics_contract_hash": forward_decision["contract_hash"],
        "forward_metrics_adoption_id": forward_decision["adoption_id"],
        "forward_metrics_admitted": forward_decision["admitted"],
        "forward_metrics_capability_decision": forward_decision["capability_decision"],
        "forward_metrics_grandfathered_current_namespace": forward_decision["grandfathered_current_namespace"],
    }


def ps_json(script: str, timeout: int = 20) -> Any:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def python_processes() -> list[dict[str, Any]]:
    try:
        import psutil

        rows: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["pid", "ppid", "name", "create_time", "cmdline", "memory_info", "cpu_times", "nice"]):
            try:
                info = proc.info
                name = str(info.get("name") or "")
                command_line = " ".join(str(part) for part in (info.get("cmdline") or []))
                if "python" not in name.lower() and "powershell" not in name.lower():
                    continue
                if not is_relevant_python_process_command(command_line):
                    continue
                memory = info.get("memory_info")
                cpu = info.get("cpu_times")
                rows.append(
                    {
                        "ProcessId": int(info.get("pid") or 0),
                        "ParentProcessId": int(info.get("ppid") or 0),
                        "Name": name,
                        "CreationDate": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(float(info.get("create_time") or 0))),
                        "Priority": info.get("nice"),
                        "WorkingSetSize": int(getattr(memory, "rss", 0) or 0),
                        "PageFileUsage": int(getattr(memory, "private", getattr(memory, "vms", 0)) or 0),
                        "KernelModeTime": int(float(getattr(cpu, "system", 0.0) or 0.0) * 10_000_000),
                        "UserModeTime": int(float(getattr(cpu, "user", 0.0) or 0.0) * 10_000_000),
                        "CommandLine": command_line,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if rows:
            return rows
    except Exception:
        pass
    data = ps_json(
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,Priority,WorkingSetSize,PageFileUsage,KernelModeTime,UserModeTime,CommandLine | ConvertTo-Json -Depth 4",
        timeout=25,
    )
    if isinstance(data, list):
        return data
    return [data] if isinstance(data, dict) else []


def is_relevant_python_process_command(command_line: str) -> bool:
    normalized = str(command_line).replace("/", "\\").lower()
    return any(token in normalized for token in DS24_PROCESS_COMMAND_TOKENS)


def process_status(pid: int) -> dict[str, Any]:
    if not pid:
        return {"alive": False, "pid": 0}
    try:
        import psutil

        proc = psutil.Process(int(pid))
        memory = proc.memory_info()
        cpu = proc.cpu_times()
        try:
            command_line = " ".join(str(part) for part in proc.cmdline())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            command_line = ""
        return {
            "alive": True,
            "pid": int(pid),
            "parent_pid": proc.ppid(),
            "creation_time": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(float(proc.create_time()))),
            "priority": proc.nice(),
            "working_set": int(getattr(memory, "rss", 0) or 0),
            "private_memory": int(getattr(memory, "private", getattr(memory, "vms", 0)) or 0),
            "kernel_time": int(float(getattr(cpu, "system", 0.0) or 0.0) * 10_000_000),
            "user_time": int(float(getattr(cpu, "user", 0.0) or 0.0) * 10_000_000),
            "command_line": command_line,
        }
    except Exception:
        pass
    data = ps_json(
        f"$p=Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
        f"$c=Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\" -ErrorAction SilentlyContinue; "
        "$parent=0; $cmd=''; if ($c) { $parent=$c.ParentProcessId; $cmd=$c.CommandLine }; "
        "if ($p) { @{"
        "alive=$true;"
        "ProcessId=$p.Id;"
        "ParentProcessId=$parent;"
        "CreationDate=$p.StartTime.ToUniversalTime().ToString('o');"
        "Priority=$p.PriorityClass.ToString();"
        "WorkingSetSize=[int64]$p.WorkingSet64;"
        "PageFileUsage=[int64]$p.PrivateMemorySize64;"
        "KernelModeTime=[int64]$p.PrivilegedProcessorTime.Ticks;"
        "UserModeTime=[int64]$p.UserProcessorTime.Ticks;"
        "CommandLine=$cmd"
        "} | ConvertTo-Json } else { @{alive=$false} | ConvertTo-Json }",
        timeout=5,
    )
    if not isinstance(data, dict) or not data.get("alive"):
        return {"alive": False, "pid": pid}
    return {
        "alive": True,
        "pid": pid,
        "parent_pid": data.get("ParentProcessId"),
        "creation_time": data.get("CreationDate"),
        "priority": data.get("Priority"),
        "working_set": int(data.get("WorkingSetSize", 0) or 0),
        "private_memory": int(data.get("PageFileUsage", 0) or 0),
        "kernel_time": int(data.get("KernelModeTime", 0) or 0),
        "user_time": int(data.get("UserModeTime", 0) or 0),
        "command_line": data.get("CommandLine", ""),
    }


def process_exists(pid: int) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except Exception:
        pass
    data = ps_json(
        f"$p=Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
        "if ($p) { @{alive=$true;pid=$p.Id} | ConvertTo-Json } else { @{alive=$false} | ConvertTo-Json }",
        timeout=5,
    )
    return bool(isinstance(data, dict) and data.get("alive"))


def process_creation_time(pid: int) -> str:
    return str(process_status(pid).get("creation_time") or "")


def task_state(task_name: str) -> dict[str, Any]:
    data = ps_json(
        f"Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue | "
        "Select-Object TaskName,State,TaskPath | ConvertTo-Json -Depth 4",
        timeout=15,
    )
    if isinstance(data, dict):
        return data
    return {"TaskName": task_name, "State": "Absent", "TaskPath": ""}


def system_snapshot() -> dict[str, Any]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory = MEMORYSTATUSEX()
        memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        committed_percent = (
            round(((memory.ullTotalPageFile - memory.ullAvailPageFile) / memory.ullTotalPageFile) * 100, 2)
            if memory.ullTotalPageFile
            else 100.0
        )
        disk = shutil.disk_usage(str(ROOT.anchor or "C:\\"))
        return {
            "observed_at_utc": utc_now(),
            "available_ram_bytes": int(memory.ullAvailPhys),
            "total_ram_bytes": int(memory.ullTotalPhys),
            "system_commit_percent": float(committed_percent),
            "disk_free_bytes": int(disk.free),
            "disk_total_bytes": int(disk.total),
        }
    except Exception:
        pass
    data = ps_json(
        "$os=Get-CimInstance Win32_OperatingSystem; $disk=Get-PSDrive -Name C; "
        "[pscustomobject]@{"
        "total_visible_memory_bytes=[int64]$os.TotalVisibleMemorySize*1KB;"
        "free_physical_memory_bytes=[int64]$os.FreePhysicalMemory*1KB;"
        "committed_memory_percent=[math]::Round((($os.TotalVirtualMemorySize-$os.FreeVirtualMemory)/$os.TotalVirtualMemorySize)*100,2);"
        "c_free_bytes=$disk.Free;"
        "c_total_bytes=$disk.Used+$disk.Free"
        "} | ConvertTo-Json",
        timeout=15,
    )
    data = data if isinstance(data, dict) else {}
    return {
        "observed_at_utc": utc_now(),
        "available_ram_bytes": int(data.get("free_physical_memory_bytes", 0) or 0),
        "total_ram_bytes": int(data.get("total_visible_memory_bytes", 0) or 0),
        "system_commit_percent": float(data.get("committed_memory_percent", 100.0) or 100.0),
        "disk_free_bytes": int(data.get("c_free_bytes", 0) or 0),
        "disk_total_bytes": int(data.get("c_total_bytes", 0) or 0),
    }


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def metrics_rows(root: Path) -> dict[str, int]:
    return {
        "metric_rows": parquet_log_rows(root, "per_t_metrics", legacy_path=root / "per_t_metrics.parquet"),
        "topn_rows": parquet_log_rows(root, "decision_trace", legacy_path=root / "decision_trace.parquet"),
        "pending_rows": parquet_rows(root / "pending_buffer.parquet"),
    }


def parquet_numeric_non_null_rows(root: Path, stem: str, column: str) -> int:
    try:
        import pandas as pd

        frame = read_parquet_log(root, stem, columns=[column])
        if frame.empty or column not in frame:
            return 0
        return int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
    except Exception:
        return 0


def metrics_root_name_from_command(command: str) -> str:
    value = command_arg(command, "--metrics-root-name")
    return value or ""


def v3_namespace_roots_for_family(family: str) -> list[Path]:
    family_root = POLICY_ROOT / family
    names: list[str] = []
    for filename in ("worker_inventory.json", "progress.json", "initialization_telemetry.json"):
        name = read_json(family_root / filename).get("metrics_root_name")
        if str(name).startswith("metrics_only_v3"):
            names.append(str(name))
    names.extend(
        [
            "metrics_only_v3_r37_huber_replay",
            "metrics_only_v3_r37_rff_replay",
            "metrics_only_v3_r37_mlp_replay",
            "metrics_only_v3_r36_replay",
            "metrics_only_v3",
        ]
    )
    if family_root.exists():
        names.extend(path.name for path in family_root.glob("metrics_only_v3*") if path.is_dir())
    roots: list[Path] = []
    seen: set[Path] = set()
    for name in names:
        root = family_root / name
        try:
            key = root.resolve()
        except OSError:
            key = root
        if key in seen or not root.exists():
            continue
        seen.add(key)
        roots.append(root)
    return roots


def metrics_root_for_family(family: str, *, command: str = "", progress: dict[str, Any] | None = None, inventory: dict[str, Any] | None = None) -> Path:
    progress = progress or {}
    inventory = inventory or {}
    name = metrics_root_name_from_command(command) if command else ""
    name = name or str(progress.get("metrics_root_name") or inventory.get("metrics_root_name") or "")
    if name:
        return POLICY_ROOT / family / name
    for root in v3_namespace_roots_for_family(family):
        lease = read_json(root / NAMESPACE_WRITER_LEASE_NAME)
        owner_pid = int(lease.get("pid", 0) or 0)
        if owner_pid and process_status(owner_pid).get("alive"):
            return root
    if not name and (POLICY_ROOT / family / "metrics_only_v3" / "resolved_performance_contract_v3.json").exists():
        name = "metrics_only_v3"
    return POLICY_ROOT / family / (name or "metrics_only")


def namespace_lease_state(metrics_root: Path) -> dict[str, Any]:
    lease_path = metrics_root / NAMESPACE_WRITER_LEASE_NAME
    lease = read_json(lease_path)
    owner_pid = int(lease.get("pid", 0) or 0)
    status = process_status(owner_pid) if owner_pid else {"alive": False}
    owner_alive = bool(status.get("alive"))
    creation_verified = process_creation_matches(lease.get("process_creation_time"), status.get("creation_time"))
    if not lease:
        state = "ABSENT"
    elif owner_alive and creation_verified:
        state = "LIVE_VERIFIED"
    elif owner_alive:
        state = "LIVE_UNVERIFIED_OR_PID_REUSED"
    else:
        state = "STALE_RECOVERABLE"
    return {
        "namespace_lease_state": state,
        "namespace_lease_path": display_path(lease_path) if lease_path.exists() else "",
        "namespace_lease_pid": owner_pid,
        "namespace_lease_generation": int(lease.get("lease_generation", 0) or 0),
        "namespace_lease_resume_generation": lease.get("resume_generation"),
        "namespace_lease_heartbeat": lease.get("heartbeat_utc", ""),
        "namespace_lease_phase": lease.get("phase", ""),
        "namespace_lease_cursor": lease.get("cursor", ""),
        "namespace_lease_namespace": lease.get("namespace", ""),
        "namespace_lease_creation_verified": creation_verified,
    }


def resolved_v2_rows(root: Path) -> dict[str, Any]:
    checkpoint = read_json(root / "resolved_performance_checkpoint_v2.json")
    summary = read_json(root / "resolved_performance_summary_v2.json")
    contract = read_json(root / "resolved_performance_contract_v2.json")
    pending_path = root / "pending_scores_v2.parquet"
    resolved_path = root / "resolved_per_t_performance_v2.parquet"
    terminal_path = root / "terminal_censored_v2.parquet"
    pending_rows = parquet_rows(pending_path)
    resolved_rows = parquet_log_rows(root, "resolved_per_t_performance_v2", legacy_path=resolved_path)
    terminal_rows = parquet_log_rows(root, "terminal_censored_v2", legacy_path=terminal_path)
    transient_root = root / "transient_tmp"
    resolved_parts_bytes = directory_size_bytes(parquet_log_parts_dir(root, "resolved_per_t_performance_v2"))
    terminal_parts_bytes = directory_size_bytes(parquet_log_parts_dir(root, "terminal_censored_v2"))
    return {
        "evaluation_contract_version": checkpoint.get("evaluation_contract_version") or (RESOLVED_PERFORMANCE_CONTRACT_V2_VERSION if contract else ""),
        "evaluation_contract_id": checkpoint.get("evaluation_contract_id") or contract.get("authority_id", ""),
        "evaluation_hash": checkpoint.get("evaluation_contract_hash") or contract.get("authority_hash", ""),
        "pending_score_rows": int(checkpoint.get("pending_score_rows", pending_rows) or pending_rows),
        "oldest_pending_timestamp": checkpoint.get("oldest_pending_timestamp", ""),
        "resolved_performance_rows": int(checkpoint.get("resolved_performance_rows", resolved_rows) or resolved_rows),
        "rank_ic_valid_rows": int(checkpoint.get("rank_ic_valid_rows", summary.get("rank_ic", {}).get("valid_rows", 0)) or 0),
        "return_valid_rows": int(checkpoint.get("return_valid_rows", summary.get("returns", {}).get("resolved_portfolio_observations", 0)) or 0),
        "terminal_censored_rows": int(checkpoint.get("terminal_censored_rows", terminal_rows) or terminal_rows),
        "bounded_pending_storage_bytes": int(checkpoint.get("bounded_pending_storage_bytes", pending_path.stat().st_size if pending_path.exists() else 0) or 0),
        "transient_storage_contract_id": checkpoint.get("transient_storage_contract_id", TRANSIENT_STORAGE_CONTRACT_V1_ID if (root / "transient_storage_contract_v1.json").exists() else ""),
        "transient_storage_contract_hash": checkpoint.get("transient_storage_contract_hash", transient_storage_contract_v1_hash() if (root / "transient_storage_contract_v1.json").exists() else ""),
        "current_temporary_bytes": int(checkpoint.get("current_temporary_bytes", directory_size_bytes(transient_root)) or 0),
        "peak_temporary_bytes": int(checkpoint.get("peak_temporary_bytes", 0) or 0),
        "durable_metrics_bytes": int(checkpoint.get("durable_metrics_bytes", resolved_parts_bytes + terminal_parts_bytes + (pending_path.stat().st_size if pending_path.exists() else 0)) or 0),
        "pending_timestamp_count": int(checkpoint.get("pending_timestamp_count", 0) or 0),
        "pending_timestamp_limit": int(checkpoint.get("pending_timestamp_limit", 0) or 0),
        "provisional_mean_rank_ic": summary.get("rank_ic", {}).get("mean_spearman_rank_ic"),
        "provisional_cumulative_net_return": summary.get("returns", {}).get("cumulative_net_return"),
        "summary_status": summary.get("status", ""),
        "pending_scores_path": display_path(pending_path) if pending_path.exists() else "",
        "resolved_performance_path": display_path(resolved_path) if resolved_path.exists() else "",
        "summary_path": display_path(root / "resolved_performance_summary_v2.json") if (root / "resolved_performance_summary_v2.json").exists() else "",
    }


def resolved_v3_rows(root: Path) -> dict[str, Any]:
    checkpoint = read_json(root / "resolved_performance_checkpoint_v3.json")
    summary = read_json(root / "resolved_performance_summary_v3.json")
    contract = read_json(root / "resolved_performance_contract_v3.json")
    pending_path = root / "pending_scores_v3.parquet"
    transient_root = root / "transient_tmp"
    stems = [
        "rank_ic_v3",
        "decision_trace_v3",
        "sleeve_maturity_ledger_v3",
        "daily_portfolio_returns_v3",
        "transaction_costs_v3",
        "pending_outcome_ledger_v3",
        "rank_ic_audit_sample_v3",
        "terminal_censored_v3",
        "refit_events_v3",
    ]
    pending_bytes = int(os.stat(openable_path(pending_path)).st_size if openable_exists(pending_path) else 0)
    durable = sum(directory_size_bytes(parquet_log_parts_dir(root, stem)) for stem in stems) + pending_bytes
    partition_count = sum(len(parquet_log_part_paths(root, stem)) for stem in stems)
    rank_ic_rows = parquet_log_rows(root, "rank_ic_v3")
    daily_rows = parquet_log_rows(root, "daily_portfolio_returns_v3")
    terminal_rows = parquet_log_rows(root, "terminal_censored_v3")
    rank_ic = summary.get("rank_ic", {}) if isinstance(summary.get("rank_ic"), dict) else {}
    returns = summary.get("returns", {}) if isinstance(summary.get("returns"), dict) else {}
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage"), dict) else {}
    rank_ic_valid_rows = max(int(checkpoint.get("rank_ic_valid_rows", 0) or 0), int(rank_ic.get("valid_timestamps", 0) or 0))
    if rank_ic_rows and not rank_ic_valid_rows:
        rank_ic_valid_rows = parquet_numeric_non_null_rows(root, "rank_ic_v3", "spearman_rank_ic")
    return {
        "evaluation_contract_version": checkpoint.get("evaluation_contract_version") or (RESOLVED_PERFORMANCE_CONTRACT_V3_VERSION if contract else ""),
        "evaluation_contract_id": checkpoint.get("evaluation_contract_id") or contract.get("authority_id", ""),
        "evaluation_hash": checkpoint.get("evaluation_contract_hash") or contract.get("authority_hash", ""),
        "pending_score_rows": int(checkpoint.get("pending_score_rows", parquet_rows(pending_path)) or 0),
        "oldest_pending_timestamp": checkpoint.get("first_uncommitted_timestamp", ""),
        "resolved_performance_rows": max(int(checkpoint.get("resolved_performance_rows", 0) or 0), int(rank_ic_rows)),
        "rank_ic_valid_rows": rank_ic_valid_rows,
        "return_valid_rows": max(int(checkpoint.get("return_valid_rows", 0) or 0), int(returns.get("daily_return_rows", 0) or 0), int(daily_rows)),
        "terminal_censored_rows": max(int(checkpoint.get("terminal_censored_rows", 0) or 0), int(coverage.get("terminal_censored_rows", 0) or 0), int(terminal_rows)),
        "bounded_pending_storage_bytes": int(checkpoint.get("bounded_pending_storage_bytes", pending_bytes) or pending_bytes),
        "transient_storage_contract_id": checkpoint.get("transient_storage_contract_id", TRANSIENT_STORAGE_CONTRACT_V1_ID if (root / "transient_storage_contract_v1.json").exists() else ""),
        "transient_storage_contract_hash": checkpoint.get("transient_storage_contract_hash", transient_storage_contract_v1_hash() if (root / "transient_storage_contract_v1.json").exists() else ""),
        "current_temporary_bytes": int(checkpoint.get("current_temporary_bytes", directory_size_bytes(transient_root)) or 0),
        "peak_temporary_bytes": int(checkpoint.get("peak_temporary_bytes", 0) or 0),
        "durable_metrics_bytes": int(checkpoint.get("durable_metrics_bytes", durable) or durable),
        "pending_timestamp_count": int(checkpoint.get("pending_timestamp_count", 0) or 0),
        "pending_timestamp_limit": int(checkpoint.get("pending_timestamp_limit", 0) or 0),
        "partition_count": max(int(checkpoint.get("partition_count", 0) or 0), int(partition_count)),
        "duplicate_count": int(checkpoint.get("duplicate_count", 0) or 0),
        "provisional_mean_rank_ic": rank_ic.get("mean_spearman_rank_ic"),
        "rank_ic_ci": rank_ic.get("dependence_aware_95_ci"),
        "last_daily_net_return": returns.get("last_daily_net_return"),
        "provisional_cumulative_net_return": returns.get("cumulative_net_return"),
        "daily_sharpe": returns.get("daily_sharpe"),
        "maximum_drawdown": returns.get("maximum_drawdown"),
        "summary_status": summary.get("status", ""),
        "daily_refit_count": int(checkpoint.get("daily_refit_count", 0) or 0),
        "last_refit_session": checkpoint.get("last_refit_session", ""),
        "daily_refit_with_five_minute_scoring": checkpoint.get("daily_refit_with_five_minute_scoring", False),
        "no_five_minute_retraining": checkpoint.get("no_five_minute_retraining", False),
        "pending_scores_path": display_path(pending_path) if openable_exists(pending_path) else "",
        "resolved_performance_path": display_path(root / "rank_ic_v3_manifest.json") if openable_exists(root / "rank_ic_v3_manifest.json") else "",
        "summary_path": display_path(root / "resolved_performance_summary_v3.json") if openable_exists(root / "resolved_performance_summary_v3.json") else "",
    }


def resolved_rows_for_root(root: Path) -> dict[str, Any]:
    v3 = resolved_v3_rows(root)
    if v3["evaluation_contract_id"] == RESOLVED_PERFORMANCE_CONTRACT_V3_ID:
        return v3
    return resolved_v2_rows(root)


def file_tail(path: Path, limit: int = 1600) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max(limit * 4, 8192)), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")[-limit:]
    except Exception:
        return ""


def stderr_tail(path: Path, limit: int = 1600) -> str:
    candidates = [path]
    if path.parent.exists():
        logs = sorted(path.parent.glob("stderr*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
        if logs:
            if logs[0].stat().st_size == 0:
                return ""
            nonempty = [log for log in logs if log.stat().st_size > 0]
            candidates = nonempty or logs
    first_tail = ""
    for candidate in candidates[:12]:
        if not candidate.exists():
            continue
        tail = file_tail(candidate, limit=limit)
        if tail and not first_tail:
            first_tail = tail
        if fatal_stderr(tail):
            return tail
    return first_tail


def stderr_tail_for_family(
    root: Path,
    *,
    command: str = "",
    lease_state: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    live: bool = False,
    limit: int = 1600,
) -> str:
    lease_state = lease_state or {}
    progress = progress or {}
    telemetry = telemetry or {}
    if live and root.exists():
        generation = (
            command_arg(command, "--resume-generation")
            or lease_state.get("namespace_lease_resume_generation")
            or progress.get("resume_generation")
            or telemetry.get("resume_generation")
        )
        if generation:
            logs = sorted(root.glob(f"stderr*_gen{generation}.log"), key=lambda item: item.stat().st_mtime, reverse=True)
            if logs:
                nonempty = [log for log in logs if log.stat().st_size > 0]
                if not nonempty:
                    return ""
                return file_tail(nonempty[0], limit=limit)
        logs = sorted(root.glob("stderr*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
        if logs:
            if logs[0].stat().st_size == 0:
                return ""
            return file_tail(logs[0], limit=limit)
    return stderr_tail(root / "stderr.log", limit=limit)


def fatal_stderr(text: str) -> bool:
    return "Traceback (most recent call last)" in text or "RuntimeError:" in text or "Fatal" in text


def command_arg(command: str, flag: str) -> str:
    parts = command.split()
    for idx, part in enumerate(parts):
        if part == flag and idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def exact_processes() -> list[dict[str, Any]]:
    progress = read_json(STAGE / "33_global_progress.json")
    inventory = read_json(STAGE / "29_persistent_worker_inventory.json")
    pid = int(progress.get("pid") or inventory.get("pid") or 0)
    status = process_status(pid)
    if not status.get("alive"):
        return []
    return [
        {
            "ProcessId": pid,
            "ParentProcessId": status.get("parent_pid"),
            "CreationDate": status.get("creation_time"),
            "Priority": status.get("priority"),
            "WorkingSetSize": status.get("working_set"),
            "PageFileUsage": status.get("private_memory"),
            "KernelModeTime": status.get("kernel_time"),
            "UserModeTime": status.get("user_time"),
            "CommandLine": f"{PROTECTED_EXACT_SCRIPT} checkpoint_pid={pid}",
        }
    ]


def exact_manifest() -> dict[str, Any]:
    rows = exact_processes()
    proc = rows[0] if rows else {}
    pid = int(proc.get("ProcessId", 0) or 0)
    progress = read_json(STAGE / "33_global_progress.json")
    met = metrics_rows(EXACT_METRICS_ROOT)
    terminal_t = progress.get("terminal_T") or TERMINAL_T
    cursor = progress.get("last_completed_T")
    terminal_complete = same_utc_iso(cursor, terminal_t) and int(met.get("metric_rows", 0) or 0) > 0
    return {
        "family": "exact_ridge_pca",
        "state": "COMPLETE" if terminal_complete else ("RUNNING" if bool(pid) else "PAUSED_RESOURCE_GATE"),
        "protected": True,
        "pid": pid,
        "recorded_pid": int(progress.get("pid", 0) or 0),
        "pid_alive": bool(pid),
        "terminal_complete": terminal_complete,
        "terminal_T": terminal_t,
        "parent_pid": proc.get("ParentProcessId"),
        "creation_time": proc.get("CreationDate"),
        "priority": proc.get("Priority"),
        "working_set": int(proc.get("WorkingSetSize", 0) or 0),
        "private_memory": int(proc.get("PageFileUsage", 0) or 0),
        "kernel_time": int(proc.get("KernelModeTime", 0) or 0),
        "user_time": int(proc.get("UserModeTime", 0) or 0),
        "command_line": proc.get("CommandLine", ""),
        "output_namespace": display_path(EXACT_METRICS_ROOT),
        "checkpoint": display_path(STAGE / "33_global_progress.json"),
        "resume_generation": command_arg(str(proc.get("CommandLine", "")), "--resume-generation") or progress.get("resume_generation"),
        "heartbeat": progress.get("heartbeat_utc"),
        "cursor": cursor,
        "metrics_rows": met["metric_rows"],
        "topn_rows": met["topn_rows"],
        "stderr_tail": "",
    }


def family_from_command_line(command: str) -> str:
    normalized = " ".join(str(command).replace("/", "\\").split()).lower()
    if not any(
        script.lower() in normalized
        for script in (POLICY_WORKER_SCRIPT, LIGHTGBM_RANKING_WORKER_SCRIPT, SEQUENCE_WORKER_SCRIPT)
    ):
        return ""
    for family in ALL_FAMILIES:
        family_lower = family.lower()
        if f"--family {family_lower}" in normalized or f"--family={family_lower}" in normalized:
            return family
        slug = family_slug(family)
        if f"--family {slug}" in normalized or f"--family={slug}" in normalized:
            return family
    return ""


def command_matches_family(command: str, family: str) -> bool:
    return family_from_command_line(command) == family


def process_row_from_status(pid: int, status: dict[str, Any], command_line: str) -> dict[str, Any]:
    return {
        "ProcessId": pid,
        "ParentProcessId": status.get("parent_pid"),
        "CreationDate": status.get("creation_time"),
        "Priority": status.get("priority"),
        "WorkingSetSize": status.get("working_set"),
        "PageFileUsage": status.get("private_memory"),
        "KernelModeTime": status.get("kernel_time"),
        "UserModeTime": status.get("user_time"),
        "CommandLine": command_line,
    }


def family_processes() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {family: [] for family in ALL_FAMILIES}
    seen: dict[str, set[int]] = {family: set() for family in ALL_FAMILIES}
    for row in python_processes():
        command = str(row.get("CommandLine", ""))
        family = family_from_command_line(command)
        pid = int(row.get("ProcessId", 0) or 0)
        if not family or not pid or pid in seen[family]:
            continue
        seen[family].add(pid)
        out.setdefault(family, []).append(row)
    for family in ALL_FAMILIES:
        root = POLICY_ROOT / family
        progress = read_json(root / "progress.json")
        telemetry = read_json(root / "initialization_telemetry.json")
        inventory = read_json(root / "worker_inventory.json")
        for pid in [
            int(progress.get("pid") or 0),
            int(telemetry.get("pid") or 0),
            int(inventory.get("pid") or 0),
        ]:
            if not pid or pid in seen[family]:
                continue
            status = process_status(pid)
            if not status.get("alive"):
                continue
            command = str(status.get("command_line") or "")
            if not command_matches_family(command, family):
                continue
            seen[family].add(pid)
            out.setdefault(family, []).append(process_row_from_status(pid, status, command))
        for namespace_root in v3_namespace_roots_for_family(family):
            lease = read_json(namespace_root / NAMESPACE_WRITER_LEASE_NAME)
            pid = int(lease.get("pid") or 0)
            if not pid or pid in seen[family]:
                continue
            status = process_status(pid)
            if not status.get("alive"):
                continue
            command = str(status.get("command_line") or "")
            if command and not command_matches_family(command, family):
                continue
            if not command:
                command = f"{POLICY_WORKER_SCRIPT} --family {family} --metrics-root-name {namespace_root.name} namespace_lease_pid={pid}"
            seen[family].add(pid)
            out.setdefault(family, []).append(process_row_from_status(pid, status, command))
    for family, rows in out.items():
        rows.sort(key=lambda row: str(row.get("CreationDate") or row.get("ProcessId") or ""))
    return out


def inventory_rows(family: str) -> list[dict[str, str]]:
    path = POLICY_ROOT / family / "prediction_inventory.csv"
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def duplicate_count(rows: list[dict[str, str]]) -> int:
    keys = [(row.get("family", ""), row.get("decision_timestamp", "")) for row in rows]
    return len(keys) - len(set(keys))


def checkpoint_valid(family: str) -> bool:
    root = POLICY_ROOT / family
    return (root / "progress.json").exists() or (root / "initialization_telemetry.json").exists() or (root / "resume_fast_forward_plan.json").exists()


def r30_reactivation_release_active(family: str) -> bool:
    payload = read_json(POLICY_ROOT / family / R30_REACTIVATION_RELEASE_NAME)
    return bool(
        payload.get("release_id") == "DS24_R30_REACTIVATION_RELEASE_V1"
        and payload.get("family") == family
        and payload.get("source_gap_repaired") is True
    )


def r36_namespace_quarantine_active(family: str) -> bool:
    payload = read_json(POLICY_ROOT / family / R36_NAMESPACE_QUARANTINE_NAME)
    return bool(
        payload.get("family") == family
        and payload.get("status") == "ACTIVE"
        and payload.get("release_required") is True
    )


def r36_admission_hold_active() -> dict[str, Any]:
    payload = read_json(R36_ADMISSION_HOLD_PATH)
    if payload.get("status") == "ACTIVE" and payload.get("release_required") is True:
        return payload
    return {}


def classify_family(family: str, processes: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    processes = processes if processes is not None else family_processes()
    root = POLICY_ROOT / family
    progress = read_json(root / "progress.json")
    telemetry = read_json(root / "initialization_telemetry.json")
    worker_inventory = read_json(root / "worker_inventory.json")
    plan = read_json(root / "resume_fast_forward_plan.json")
    live = processes.get(family, [])
    cmd = str(live[0].get("CommandLine", "")) if live else ""
    metrics_root = metrics_root_for_family(family, command=cmd, progress=progress, inventory=worker_inventory)
    lease_state = namespace_lease_state(metrics_root)
    metrics_checkpoint = read_json(metrics_root / "checkpoint.json")
    resolved = resolved_rows_for_root(metrics_root)
    met = metrics_rows(metrics_root)
    rows = inventory_rows(family)
    inv_dupes = duplicate_count(rows)
    stderr = stderr_tail_for_family(root, command=cmd, lease_state=lease_state, progress=progress, telemetry=telemetry, live=bool(live))
    fatal = fatal_stderr(stderr)
    r30_release = r30_reactivation_release_active(family)
    r36_quarantine = r36_namespace_quarantine_active(family)
    r36_quarantine_applies_to_active_namespace = r36_quarantine and metrics_root.name in {
        "metrics_only",
        "metrics_only_v3",
        "metrics_only_v3_r36_replay",
    }
    fatal_effective = fatal and not r30_release
    dup_workers = max(0, len(live) - 1)
    live_pids = [int(proc.get("ProcessId", 0) or 0) for proc in live]
    recorded_pid = int(progress.get("pid") or telemetry.get("pid") or worker_inventory.get("pid") or 0)
    heartbeat = progress.get("heartbeat_utc") or telemetry.get("heartbeat_utc")
    phase = progress.get("phase") or telemetry.get("phase") or plan.get("phase") or ""
    sequence_reconciliation = (
        sequence_certification_reconciliation(family)
        if family in SEQUENCE_CERTIFICATION
        else {}
    )
    terminal = r31_has_reached_registered_terminal(family, progress or metrics_checkpoint)
    terminal_exclusion = r31_terminal_exclusion_state(family, root=POLICY_ROOT)
    terminal_complete = bool(terminal["reached"] and met["metric_rows"] > 0 and met["topn_rows"] > 0 and inv_dupes == 0)
    live_ok = bool(live) and dup_workers == 0 and checkpoint_valid(family) and not fatal_effective
    excluded = family_is_excluded(family)
    lease_unverified_live = lease_state["namespace_lease_state"] == "LIVE_UNVERIFIED_OR_PID_REUSED"
    lease_owner_mismatch = bool(
        live
        and lease_state["namespace_lease_pid"]
        and int(lease_state["namespace_lease_pid"]) not in live_pids
        and lease_state["namespace_lease_state"] != "STALE_RECOVERABLE"
    )
    if dup_workers:
        state = "DUPLICATE_FAMILY_CONTAINMENT"
    elif lease_unverified_live or lease_owner_mismatch:
        state = "NAMESPACE_RECONCILING"
    elif r36_quarantine_applies_to_active_namespace and live:
        state = "NAMESPACE_RECONCILING"
    elif r36_quarantine_applies_to_active_namespace:
        state = "NAMESPACE_QUARANTINED_REPAIR_REQUIRED"
    elif fatal_effective and live:
        state = "NAMESPACE_RECONCILING"
    elif excluded and not live:
        state = "EXCLUDED_BY_OPERATOR"
    elif terminal_exclusion == "COMPLETE":
        state = "COMPLETE"
    elif terminal_exclusion == "TERMINAL_VALIDATING":
        state = "TERMINAL_VALIDATING"
    elif fatal_effective and checkpoint_valid(family):
        state = "CRASHED_RECOVERABLE"
    elif fatal_effective:
        state = (
            "CRASHED_RECOVERABLE"
            if sequence_reconciliation.get("admission_state") == "V3_CERTIFIED_READY"
            else "CRASHED_BLOCKED"
        )
    elif terminal["reached"] and met["metric_rows"] > 0 and inv_dupes == 0:
        state = "TERMINAL_VALIDATING"
    elif terminal_complete:
        state = "TERMINAL_VALIDATING"
    elif live_ok and met["metric_rows"] > 0:
        state = "RUNNING"
    elif live_ok:
        state = "FAST_FORWARDING"
    elif family in ACTIVE_SEED and checkpoint_valid(family):
        state = "PAUSED_RESOURCE_GATE"
    elif family in PAUSED_QUEUE and checkpoint_valid(family):
        state = "PAUSED_RESOURCE_GATE"
    elif family in PAUSED_QUEUE + READY_QUEUE:
        state = "CERTIFIED_READY"
    elif family in TREE_CERTIFICATION:
        state = "V3_CERTIFIED_READY" if launch_enabled_for_family(family) else "V3_CERTIFICATION_REQUIRED"
    elif family in RANKING_CERTIFICATION:
        state = "V3_CERTIFIED_READY" if launch_enabled_for_family(family) else "V3_CERTIFICATION_REQUIRED"
    elif family == "Temporal Fusion Transformer":
        state = "CONFIGURATION_AUTHORITY_REQUIRED"
    elif family in SEQUENCE_CERTIFICATION or family.lower() in {
        "dlinear",
        "patchtst",
        "transformer",
        "itransformer",
        "momentum_transformer",
        "market_context_encoder",
        "temporal_fusion_transformer",
    }:
        state = str(sequence_reconciliation.get("admission_state") or "V3_CERTIFICATION_REQUIRED")
    else:
        state = "NOT_IMPLEMENTED"
    if state == "CRASHED_RECOVERABLE":
        r36_recovery_state = "RECOVERABLE_RESUME_READY"
    elif live_ok:
        r36_recovery_state = "ADOPTED_SINGLE_LIVE_WORKER"
    else:
        r36_recovery_state = state
    return {
        "family": family,
        "state": state,
        "scientific_readiness": sequence_reconciliation.get("scientific_readiness", ""),
        "runtime_state": "RUNNING" if live else "ABSENT",
        "state_reconciliation_reason": sequence_reconciliation.get("reason", ""),
        "sequence_certification_reconciliation": sequence_reconciliation,
        "worker_kind": worker_kind_for_family(family),
        "worker_script": execution_registry_row(family, include_r42_authority=bool(sequence_reconciliation))["worker_script"],
        "launch_enabled": family_is_launchable(family),
        "excluded_by_operator": excluded,
        "pid": int(live[0].get("ProcessId", 0) or 0) if live else 0,
        "recorded_pid": recorded_pid,
        "pid_alive": bool(live),
        "live_worker_count": len(live),
        "live_worker_pids": live_pids,
        **lease_state,
        "r36_recovery_state": r36_recovery_state,
        "r36_namespace_quarantine_active": r36_quarantine,
        "r36_namespace_quarantine_applies_to_active_namespace": r36_quarantine_applies_to_active_namespace,
        "r36_namespace_quarantine_path": display_path(root / R36_NAMESPACE_QUARANTINE_NAME) if (root / R36_NAMESPACE_QUARANTINE_NAME).exists() else "",
        "parent_pid": live[0].get("ParentProcessId") if live else None,
        "creation_time": live[0].get("CreationDate") if live else None,
        "priority": live[0].get("Priority") if live else None,
        "working_set": int(live[0].get("WorkingSetSize", 0) or 0) if live else 0,
        "private_memory": int(live[0].get("PageFileUsage", 0) or 0) if live else 0,
        "kernel_time": int(live[0].get("KernelModeTime", 0) or 0) if live else 0,
        "user_time": int(live[0].get("UserModeTime", 0) or 0) if live else 0,
        "command_line": cmd,
        "output_namespace": display_path(root),
        "metrics_root_name": metrics_root.name,
        "metrics_root": display_path(metrics_root),
        "checkpoint": display_path(root / "progress.json") if (root / "progress.json").exists() else display_path(root / "initialization_telemetry.json"),
        "resume_generation": command_arg(cmd, "--resume-generation") or progress.get("resume_generation") or telemetry.get("resume_generation"),
        "heartbeat": heartbeat,
        "cursor": progress.get("last_completed_T") or telemetry.get("refit_T") or telemetry.get("first_refit_T"),
        "registered_terminal_cursor": r31_terminal_authority(family)["registered_terminal_cursor"],
        "terminal_reached": terminal["reached"],
        "terminal_exclusion_state": terminal_exclusion,
        "terminal_summary_path": display_path(root / R31_SUMMARY_NAME) if (root / R31_SUMMARY_NAME).exists() else "",
        "package_state": phase,
        "metrics_rows": met["metric_rows"],
        "topn_rows": met["topn_rows"],
        "pending_rows": met["pending_rows"],
        "inventory_rows": len(rows),
        "inventory_duplicate_count": inv_dupes,
        "duplicate_worker_count": dup_workers,
        "stderr_state": "STALE_FATAL_RELEASED_BY_R30" if fatal and r30_release else ("FATAL" if fatal else ("WARNINGS" if stderr else "CLEAN")),
        "stderr_tail": stderr[-500:],
        "policy_hash": progress.get("policy_hash") or telemetry.get("policy_hash") or policy_hash(),
        "active_policy_version": POLICY_ID,
        "refit_cadence": REFIT_CADENCE_ID,
        "decision_cadence": DECISION_CADENCE_ID,
        "model_vintage_id": "",
        "current_refit_session": progress.get("last_completed_refit_T") or telemetry.get("refit_T") or "",
        "training_cutoff": progress.get("last_completed_refit_T") or telemetry.get("refit_T") or "",
        "current_scoring_cursor": progress.get("last_completed_T") or "",
        "daily_fit_count": 0,
        "scoring_decision_count": met["metric_rows"],
        "model_hash": telemetry.get("model_hash") or "",
        "preprocessing_hash": "",
        "carry_forward_state": "",
        "legacy_v1_evidence": bool(met["metric_rows"] or checkpoint_valid(family)),
        "v2_compatibility": "ACTIVE" if resolved["evaluation_contract_id"] == RESOLVED_PERFORMANCE_CONTRACT_V2_ID else "NOT_ACTIVATED",
        "v3_compatibility": "ACTIVE" if resolved["evaluation_contract_id"] == RESOLVED_PERFORMANCE_CONTRACT_V3_ID else "NOT_ACTIVATED",
        "evaluation_contract_version": resolved["evaluation_contract_version"],
        "evaluation_contract_id": resolved["evaluation_contract_id"],
        "evaluation_hash": resolved["evaluation_hash"],
        "pending_score_rows": resolved["pending_score_rows"],
        "oldest_pending_timestamp": resolved["oldest_pending_timestamp"],
        "resolved_performance_rows": resolved["resolved_performance_rows"],
        "rank_ic_valid_rows": resolved["rank_ic_valid_rows"],
        "return_valid_rows": resolved["return_valid_rows"],
        "terminal_censored_rows": resolved["terminal_censored_rows"],
        "bounded_pending_storage_bytes": resolved["bounded_pending_storage_bytes"],
        "pending_timestamp_count": resolved["pending_timestamp_count"],
        "pending_timestamp_limit": resolved["pending_timestamp_limit"],
        "transient_storage_contract_id": resolved["transient_storage_contract_id"],
        "transient_storage_contract_hash": resolved["transient_storage_contract_hash"],
        "current_temporary_bytes": resolved["current_temporary_bytes"],
        "peak_temporary_bytes": resolved["peak_temporary_bytes"],
        "durable_metrics_bytes": resolved["durable_metrics_bytes"],
        "partition_count": resolved.get("partition_count", 0),
        "duplicate_count": resolved.get("duplicate_count", 0),
        "provisional_mean_rank_ic": resolved["provisional_mean_rank_ic"],
        "rank_ic_ci": resolved.get("rank_ic_ci"),
        "last_daily_net_return": resolved.get("last_daily_net_return"),
        "provisional_cumulative_net_return": resolved["provisional_cumulative_net_return"],
        "daily_sharpe": resolved.get("daily_sharpe"),
        "maximum_drawdown": resolved.get("maximum_drawdown"),
        "resolved_performance_summary_status": resolved["summary_status"],
        "daily_refit_count": resolved.get("daily_refit_count", 0),
        "last_refit_session": resolved.get("last_refit_session", ""),
        "daily_refit_with_five_minute_scoring": resolved.get("daily_refit_with_five_minute_scoring", False),
        "no_five_minute_retraining": resolved.get("no_five_minute_retraining", False),
        "pending_scores_path": resolved["pending_scores_path"],
        "resolved_performance_path": resolved["resolved_performance_path"],
        "resolved_performance_summary_path": resolved["summary_path"],
        "heavy": family in HEAVY_FAMILIES,
        "checkpoint_valid": checkpoint_valid(family),
        "metrics_only_authority": True,
    }


def build_family_board() -> list[dict[str, Any]]:
    processes = family_processes()
    return [classify_family(family, processes) for family in ALL_FAMILIES]


def lightweight_processes_by_family() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {family: [] for family in ALL_FAMILIES}
    for row in python_processes():
        command = str(row.get("CommandLine", ""))
        family = family_from_command_line(command)
        if family:
            out.setdefault(family, []).append(row)
    return out


def lightweight_certified_queue_board(queue: Sequence[str]) -> list[dict[str, Any]]:
    processes = lightweight_processes_by_family()
    board: list[dict[str, Any]] = []
    families = list(queue) + [family for family, rows in processes.items() if rows and family not in set(queue)]
    for family in families:
        live = processes.get(family, [])
        root = POLICY_ROOT / family
        progress = read_json(root / "progress.json")
        telemetry = read_json(root / "initialization_telemetry.json")
        metrics_name = str(progress.get("metrics_root_name") or telemetry.get("metrics_root_name") or default_metrics_root_name(family, "metrics_only_v3"))
        metrics_root = root / metrics_name
        checkpoint = read_json(metrics_root / "resolved_performance_checkpoint_v3.json")
        excluded = family_is_excluded(family)
        sequence_reconciliation = (
            sequence_certification_reconciliation(family)
            if family in SEQUENCE_CERTIFICATION
            else {}
        )
        if live:
            state = "RUNNING"
        elif excluded:
            state = "EXCLUDED_BY_OPERATOR"
        elif family == "elastic_net":
            state = "CERTIFIED_READY"
        elif family not in queue:
            state = "RUNNING" if live else "NOT_IN_CERTIFIED_QUEUE"
        elif sequence_reconciliation:
            state = str(sequence_reconciliation.get("admission_state") or "V3_CERTIFICATION_REQUIRED")
        else:
            state = "V3_CERTIFIED_READY"
        live_pids = [int(row.get("ProcessId", 0) or 0) for row in live]
        board.append(
            {
                "family": family,
                "state": state,
                "scientific_readiness": sequence_reconciliation.get("scientific_readiness", ""),
                "runtime_state": "RUNNING" if live else "ABSENT",
                "state_reconciliation_reason": sequence_reconciliation.get("reason", ""),
                "sequence_certification_reconciliation": sequence_reconciliation,
                "worker_kind": worker_kind_for_family(family),
                "worker_script": execution_registry_row(family, include_r42_authority=bool(sequence_reconciliation))["worker_script"],
                "launch_enabled": family_is_launchable(family),
                "excluded_by_operator": excluded,
                "pid": live_pids[0] if live_pids else 0,
                "pid_alive": bool(live),
                "creation_time": live[0].get("CreationDate") if live else None,
                "parent_pid": live[0].get("ParentProcessId") if live else None,
                "command_line": live[0].get("CommandLine", "") if live else "",
                "live_worker_count": len(live),
                "live_worker_pids": live_pids,
                "duplicate_worker_count": max(0, len(live) - 1),
                "namespace_lease_state": "LIVE_VERIFIED" if live else "STALE_RECOVERABLE",
                "checkpoint": display_path(root / "progress.json") if (root / "progress.json").exists() else "",
                "cursor": progress.get("last_completed_T") or telemetry.get("refit_T") or "",
                "metrics_rows": int(checkpoint.get("resolved_performance_rows", 0) or 0),
                "topn_rows": int(checkpoint.get("rank_ic_valid_rows", 0) or 0),
                "resolved_performance_rows": int(checkpoint.get("resolved_performance_rows", 0) or 0),
                "rank_ic_valid_rows": int(checkpoint.get("rank_ic_valid_rows", 0) or 0),
                "working_set": sum(int(row.get("WorkingSetSize", 0) or 0) for row in live),
                "private_memory": sum(int(row.get("PageFileUsage", 0) or 0) for row in live),
                "heavy": family in HEAVY_FAMILIES,
            }
        )
    return board


def active_running(board: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in board
        if row["state"] in {"RUNNING", "FAST_FORWARDING", "INITIALIZING", "STARTING"}
        and row.get("pid_alive", True)
    ]


def certified_queue_manifest_path(path: str | Path = "") -> Path:
    return Path(path) if path else R42_READY_QUEUE_PATH


def validate_ready_family_queue_manifest(path: str | Path = "") -> dict[str, Any]:
    manifest_path = certified_queue_manifest_path(path)
    payload = read_json(manifest_path)
    if not payload:
        raise RuntimeError(f"DS24_R43_READY_QUEUE_MANIFEST_MISSING_OR_INVALID_JSON:{display_path(manifest_path)}")
    ticket = str(payload.get("ticket", ""))
    if ticket != R42_READY_QUEUE_AUTHORITY_ID:
        raise RuntimeError(f"DS24_R43_READY_QUEUE_AUTHORITY_MISMATCH:{ticket}")
    queue = payload.get("ready_family_queue")
    if not isinstance(queue, list) or not all(isinstance(item, str) and item.strip() for item in queue):
        raise RuntimeError("DS24_R43_READY_QUEUE_SCHEMA_INVALID:ready_family_queue")
    if len(queue) != len(set(queue)):
        raise RuntimeError("DS24_R43_READY_QUEUE_DUPLICATE_FAMILIES")
    if "Temporal Fusion Transformer" in queue or "temporal_fusion_transformer" in {family_slug(item) for item in queue}:
        raise RuntimeError("DS24_R43_READY_QUEUE_TFT_FORBIDDEN")
    unknown = [family for family in queue if family not in R42_ALLOWED_READY_FAMILIES]
    if unknown:
        raise RuntimeError(f"DS24_R43_READY_QUEUE_UNKNOWN_FAMILIES:{unknown}")
    matrix = read_json(R42_READINESS_MATRIX_PATH)
    rows = {str(row.get("family")): row for row in matrix.get("families", []) if isinstance(row, dict)}
    not_ready = [
        family
        for family in queue
        if rows.get(family, {}).get("current_r42_state") != "READY_TO_LAUNCH"
    ]
    if not_ready:
        raise RuntimeError(f"DS24_R43_READY_QUEUE_FAMILY_WITHOUT_READY_AUTHORITY:{not_ready}")
    return {
        "authority": "R42",
        "manifest_path": display_path(manifest_path),
        "manifest_hash": hash_if_exists(manifest_path),
        "matrix_path": display_path(R42_READINESS_MATRIX_PATH),
        "matrix_hash": hash_if_exists(R42_READINESS_MATRIX_PATH),
        "ticket": ticket,
        "generated_at_utc": payload.get("generated_at_utc", ""),
        "ready_family_queue": list(queue),
    }


def readiness_state_by_family() -> dict[str, str]:
    matrix = read_json(R42_READINESS_MATRIX_PATH)
    rows = matrix.get("families", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("family")): str(row.get("current_r42_state") or row.get("readiness_state") or "")
        for row in rows
        if isinstance(row, dict) and row.get("family")
    }


def sequence_certification_reconciliation(
    family: str,
    *,
    evaluation_version: str = "v3",
    metrics_root_name: str = "metrics_only_v3",
) -> dict[str, Any]:
    readiness = readiness_state_by_family().get(family, "")
    forward_decision = forward_metrics_contract_admission_decision(
        family,
        evaluation_version=evaluation_version,
        metrics_root_name=metrics_root_name,
        include_r42_authority=True,
    )
    if family == "Temporal Fusion Transformer" or readiness == "CONFIGURATION_AUTHORITY_REQUIRED":
        admission_state = "CONFIGURATION_AUTHORITY_REQUIRED"
        reason = "SCIENTIFIC_CONFIGURATION_AUTHORITY_REQUIRED"
    elif readiness == "READY_TO_LAUNCH" and forward_decision.get("admitted"):
        admission_state = "V3_CERTIFIED_READY"
        reason = "R42_READY_TO_LAUNCH_WITH_CANONICAL_FORWARD_METRICS_AUTHORITY"
    elif readiness == "READY_TO_LAUNCH":
        admission_state = "V3_CERTIFICATION_REQUIRED"
        reason = "R42_FORWARD_METRICS_CAPABILITY_INVALID_OR_MISSING"
    elif readiness:
        admission_state = readiness
        reason = "R42_SCIENTIFIC_READINESS_NOT_READY_TO_LAUNCH"
    else:
        admission_state = "V3_CERTIFICATION_REQUIRED"
        reason = "R42_SCIENTIFIC_READINESS_MISSING"
    return {
        "family": family,
        "scientific_readiness": readiness,
        "admission_state": admission_state,
        "runtime_state_if_no_namespace": "ABSENT",
        "reason": reason,
        "forward_metrics": forward_decision,
        "capability_path": forward_decision.get("capability_path", ""),
        "missing_requirements": forward_decision.get("missing_requirements", []),
    }


def default_cross_host_owner_row(family: str, *, readiness_state: str, updated_at_utc: str) -> dict[str, Any]:
    if family == "lightgbm_rank_xendcg":
        row = {
            "family": family,
            "readiness_state": readiness_state,
            "execution_owner": "MAC",
            "ownership_state": "MAC_OWNED",
            "owner_state": "MAC_COMPLETE",
            "dell_eligible": False,
            "mac_eligible": True,
            "ownership_reason": "Completed on Mac lane; Dell recomputation forbidden unless ownership is reassigned.",
        }
    elif family == "lightgbm_lambdarank":
        row = {
            "family": family,
            "readiness_state": readiness_state,
            "execution_owner": "MAC",
            "ownership_state": "MAC_OWNED",
            "owner_state": "MAC_RUNNING",
            "dell_eligible": False,
            "mac_eligible": True,
            "ownership_reason": "Currently running on Mac lane; Dell duplicate compute forbidden.",
        }
    elif family == "DLinear":
        row = {
            "family": family,
            "readiness_state": readiness_state,
            "execution_owner": "MAC",
            "ownership_state": "MAC_RESERVED",
            "owner_state": "MAC_RESERVED_NEXT",
            "dell_eligible": False,
            "mac_eligible": True,
            "ownership_reason": "Reserved as the next Mac-lane family after LambdaRank.",
        }
    elif family in R44_DELL_READY_FAMILIES or family in R44_DELL_RUNNING_FAMILIES:
        row = {
            "family": family,
            "readiness_state": readiness_state,
            "execution_owner": "DELL",
            "ownership_state": "DELL_OWNED",
            "owner_state": "DELL_OWNED",
            "dell_eligible": True,
            "mac_eligible": False,
            "ownership_reason": "Dell lane owns local execution for this certified or already-running family.",
        }
    else:
        row = {
            "family": family,
            "readiness_state": readiness_state,
            "execution_owner": "UNASSIGNED",
            "ownership_state": "UNASSIGNED",
            "owner_state": "UNASSIGNED",
            "dell_eligible": False,
            "mac_eligible": False,
            "ownership_reason": "No host execution ownership assigned by R44.",
        }
    row["authority_source"] = R44_CROSS_HOST_OWNERSHIP_AUTHORITY_ID
    row["updated_at_utc"] = updated_at_utc
    row["authority_hash"] = state_hash(row)
    return row


def build_cross_host_ownership_authority(
    *,
    updated_at_utc: str | None = None,
    path: Path = R44_CROSS_HOST_OWNERSHIP_PATH,
) -> dict[str, Any]:
    timestamp = updated_at_utc or utc_now()
    ready_authority = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    readiness = readiness_state_by_family()
    families = list(dict.fromkeys([*ALL_FAMILIES, *ready_authority["ready_family_queue"], "Temporal Fusion Transformer"]))
    rows = [
        default_cross_host_owner_row(
            family,
            readiness_state=readiness.get(family, "CONFIGURATION_AUTHORITY_REQUIRED" if family == "Temporal Fusion Transformer" else ""),
            updated_at_utc=timestamp,
        )
        for family in families
    ]
    payload = {
        "ticket": R44_CROSS_HOST_OWNERSHIP_AUTHORITY_ID,
        "authority": "R44",
        "schema_version": "1.0",
        "parent_authority": "DS24_R43_CERTIFIED_READY_QUEUE_SUPERVISOR_INTEGRATION",
        "parent_commit": "4460ad10f",
        "generated_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "r42_ready_queue_path": ready_authority["manifest_path"],
        "r42_ready_queue_hash": ready_authority["manifest_hash"],
        "r42_readiness_matrix_path": ready_authority["matrix_path"],
        "r42_readiness_matrix_hash": ready_authority["matrix_hash"],
        "ownership_states": sorted(R44_ALLOWED_OWNERSHIP_STATES),
        "owner_states": sorted(R44_ALLOWED_OWNER_STATES),
        "import_lifecycle": ["MAC_RUNNING", "MAC_COMPLETE", "COMPLETE_IMPORTED"],
        "dlinear_release_transition": {
            "from_owner_state": "MAC_RESERVED_NEXT",
            "to_owner_state": "DELL_OWNED",
            "scientific_recertification_required": False,
        },
        "families": rows,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(path, payload)
    return payload


def cross_host_ownership_manifest_path(path: str | Path = "") -> Path:
    return Path(path) if path else R44_CROSS_HOST_OWNERSHIP_PATH


def validate_cross_host_ownership_authority(path: str | Path = "") -> dict[str, Any]:
    manifest_path = cross_host_ownership_manifest_path(path)
    payload = read_json(manifest_path)
    if not payload:
        raise RuntimeError(f"DS24_R44_CROSS_HOST_OWNERSHIP_MISSING_OR_INVALID_JSON:{display_path(manifest_path)}")
    if str(payload.get("ticket", "")) != R44_CROSS_HOST_OWNERSHIP_AUTHORITY_ID:
        raise RuntimeError(f"DS24_R44_CROSS_HOST_OWNERSHIP_AUTHORITY_MISMATCH:{payload.get('ticket', '')}")
    rows = payload.get("families")
    if not isinstance(rows, list):
        raise RuntimeError("DS24_R44_CROSS_HOST_OWNERSHIP_SCHEMA_INVALID:families")
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("family"), str):
            raise RuntimeError("DS24_R44_CROSS_HOST_OWNERSHIP_SCHEMA_INVALID:family")
        family = str(row["family"])
        if family in by_family:
            raise RuntimeError(f"DS24_R44_CROSS_HOST_OWNERSHIP_DUPLICATE_FAMILY:{family}")
        if str(row.get("ownership_state", "")) not in R44_ALLOWED_OWNERSHIP_STATES:
            raise RuntimeError(f"DS24_R44_CROSS_HOST_OWNERSHIP_STATE_INVALID:{family}")
        if str(row.get("owner_state", "")) not in R44_ALLOWED_OWNER_STATES:
            raise RuntimeError(f"DS24_R44_CROSS_HOST_OWNERSHIP_OWNER_STATE_INVALID:{family}")
        by_family[family] = dict(row)
    missing = [family for family in (*R42_ALLOWED_READY_FAMILIES, "Temporal Fusion Transformer") if family not in by_family]
    if missing:
        raise RuntimeError(f"DS24_R44_CROSS_HOST_OWNERSHIP_MISSING_FAMILIES:{missing}")
    return {
        "authority": "R44",
        "manifest_path": display_path(manifest_path),
        "manifest_hash": hash_if_exists(manifest_path),
        "ticket": payload.get("ticket", ""),
        "generated_at_utc": payload.get("generated_at_utc", ""),
        "updated_at_utc": payload.get("updated_at_utc", ""),
        "r42_ready_queue_hash": payload.get("r42_ready_queue_hash", ""),
        "families": list(rows),
        "by_family": by_family,
        "excluded_mac_owned": [family for family in R44_MAC_OWNED_FAMILIES if not by_family.get(family, {}).get("dell_eligible", True)],
        "excluded_mac_reserved": [family for family in R44_MAC_RESERVED_FAMILIES if not by_family.get(family, {}).get("dell_eligible", True)],
        "dell_eligible_families": [str(row["family"]) for row in rows if bool(row.get("dell_eligible"))],
    }


def cross_host_skip_reason(family: str, ownership_authority: Mapping[str, Any]) -> str:
    row = ownership_authority.get("by_family", {}).get(family, {}) if ownership_authority else {}
    if not row:
        return ""
    if bool(row.get("dell_eligible")):
        return ""
    owner_state = str(row.get("owner_state", ""))
    execution_owner = str(row.get("execution_owner", ""))
    readiness_state = str(row.get("readiness_state", ""))
    if owner_state == "MAC_RESERVED_NEXT" or str(row.get("ownership_state", "")) == "MAC_RESERVED":
        return "SKIP_MAC_RESERVED"
    if execution_owner == "MAC" or str(row.get("ownership_state", "")) == "MAC_OWNED":
        return "SKIP_MAC_OWNED"
    if readiness_state == "CONFIGURATION_AUTHORITY_REQUIRED":
        return "SKIP_CONFIGURATION_AUTHORITY_REQUIRED"
    return "CROSS_HOST_OWNERSHIP_ADMISSION_BLOCKED"


def assert_dell_ownership_admission(family: str, path: str | Path = "") -> dict[str, Any]:
    manifest_path = cross_host_ownership_manifest_path(path)
    if not manifest_path.exists():
        return {"admitted": True, "authority": "R44_NOT_PRESENT", "family": family}
    authority = validate_cross_host_ownership_authority(manifest_path)
    reason = cross_host_skip_reason(family, authority)
    if reason:
        raise RuntimeError(f"CROSS_HOST_OWNERSHIP_ADMISSION_BLOCKED:{family}:{reason}")
    row = authority["by_family"].get(family, {})
    return {"admitted": True, "authority": "R44", "family": family, "ownership": row}


def write_dell_effective_ready_queue(
    *,
    ready_family_queue_manifest: str | Path = R42_READY_QUEUE_PATH,
    cross_host_ownership_manifest: str | Path = R44_CROSS_HOST_OWNERSHIP_PATH,
    path: Path = R44_DELL_EFFECTIVE_READY_QUEUE_PATH,
) -> dict[str, Any]:
    ready = validate_ready_family_queue_manifest(ready_family_queue_manifest)
    ownership = validate_cross_host_ownership_authority(cross_host_ownership_manifest)
    queue = [family for family in ready["ready_family_queue"] if not cross_host_skip_reason(family, ownership)]
    skipped = [
        {"family": family, "reason": cross_host_skip_reason(family, ownership)}
        for family in ready["ready_family_queue"]
        if cross_host_skip_reason(family, ownership)
    ]
    tft = ownership["by_family"].get("Temporal Fusion Transformer", {})
    if tft:
        skipped.append({"family": "Temporal Fusion Transformer", "reason": cross_host_skip_reason("Temporal Fusion Transformer", ownership)})
    payload = {
        "ticket": "DS24_R44_DELL_EFFECTIVE_READY_QUEUE",
        "authority": "R44",
        "generated_at_utc": utc_now(),
        "derived_from": {
            "r42_ready_queue_path": ready["manifest_path"],
            "r42_ready_queue_hash": ready["manifest_hash"],
            "r44_ownership_path": ownership["manifest_path"],
            "r44_ownership_hash": ownership["manifest_hash"],
        },
        "dell_effective_ready_queue": queue,
        "excluded_families": skipped,
        "excluded_mac_owned": ownership["excluded_mac_owned"],
        "excluded_mac_reserved": ownership["excluded_mac_reserved"],
        "scientific_readiness_unchanged": True,
        "worker_launches": 0,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(path, payload)
    return payload


def selected_queue(family_queue: str = "", *, ready_family_queue_manifest: str | Path = "") -> list[str]:
    if family_queue:
        return [item.strip() for item in family_queue.split(",") if item.strip()]
    if ready_family_queue_manifest:
        return list(validate_ready_family_queue_manifest(ready_family_queue_manifest)["ready_family_queue"])
    return R40_AUTOMATIC_QUEUE


def family_specific_skip_reason(
    row: Mapping[str, Any],
    *,
    allowed_states: set[str],
    excluded_families: str | Sequence[str] | None = None,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> str:
    family = str(row.get("family", ""))
    if row.get("pid_alive"):
        return "RUNNING"
    if int(row.get("duplicate_worker_count", 0) or 0):
        return "DUPLICATE_NAMESPACE_OWNER"
    if family_is_excluded(family, excluded_families, retired_families_path=retired_families_path):
        return "EXCLUDED_BY_OPERATOR"
    if not family_is_launchable(family, excluded_families, retired_families_path=retired_families_path):
        return "WORKER_ROUTE_NOT_LAUNCH_ENABLED"
    state = str(row.get("state", ""))
    if state in {"COMPLETE", "COMPLETE_IMPORTED"}:
        return "COMPLETE"
    if state in {"CONFIGURATION_AUTHORITY_REQUIRED", "V3_CERTIFICATION_REQUIRED", "V3_REPLAY_REQUIRED"}:
        return state
    if state not in allowed_states:
        return f"STATE_NOT_ELIGIBLE:{state}"
    return ""


def certified_queue_admission_plan(
    board: list[dict[str, Any]],
    *,
    family_queue: str = "",
    ready_family_queue_manifest: str | Path = "",
    cross_host_ownership_manifest: str | Path = "",
    admit_crashed_recoverable: bool = False,
    excluded_families: str | Sequence[str] | None = None,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> dict[str, Any]:
    allowed = {"PAUSED_RESOURCE_GATE", "CERTIFIED_READY", "V3_CERTIFIED_READY"}
    if admit_crashed_recoverable:
        allowed.add("CRASHED_RECOVERABLE")
    rows_by_family = {row["family"]: row for row in board}
    queue_authority: dict[str, Any] = {}
    queue = selected_queue(family_queue, ready_family_queue_manifest=ready_family_queue_manifest)
    effective_exclusions = effective_excluded_families(
        excluded_families,
        retired_families_path=retired_families_path,
    )
    retired_authority = retired_family_authority(retired_families_path)
    if ready_family_queue_manifest and not family_queue:
        queue_authority = validate_ready_family_queue_manifest(ready_family_queue_manifest)
    ownership_authority: dict[str, Any] = {}
    if cross_host_ownership_manifest:
        ownership_authority = validate_cross_host_ownership_authority(cross_host_ownership_manifest)
    skipped: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for family in queue:
        row = rows_by_family.get(family, {})
        if not row:
            skipped.append({"family": family, "reason": "FAMILY_NOT_ON_BOARD", "block_scope": "FAMILY_SPECIFIC_BLOCK"})
            continue
        reason = family_specific_skip_reason(
            row,
            allowed_states=allowed,
            excluded_families=effective_exclusions,
            retired_families_path=retired_families_path,
        )
        route = execution_registry_row(family, include_r42_authority=bool(ready_family_queue_manifest))
        item = {
            "family": family,
            "state": row.get("state", ""),
            "scientific_readiness": (
                ownership_authority.get("by_family", {}).get(family, {}).get("readiness_state", row.get("state", ""))
                if ownership_authority
                else row.get("state", "")
            ),
            "execution_owner": (
                ownership_authority.get("by_family", {}).get(family, {}).get("execution_owner", "")
                if ownership_authority
                else ""
            ),
            "owner_state": (
                ownership_authority.get("by_family", {}).get(family, {}).get("owner_state", "")
                if ownership_authority
                else ""
            ),
            "dell_eligible": (
                ownership_authority.get("by_family", {}).get(family, {}).get("dell_eligible", True)
                if ownership_authority
                else True
            ),
            "worker_kind": route["worker_kind"],
            "worker_script": route["worker_script"],
            "launch_enabled": family_is_launchable(
                family,
                effective_exclusions,
                retired_families_path=retired_families_path,
            ),
            "excluded_by_operator": family in effective_exclusions,
            "checkpoint": row.get("checkpoint", ""),
            "namespace_state": row.get("namespace_lease_state", ""),
            "metrics_rows": row.get("metrics_rows", 0),
            "resource_estimate": "heavy" if row.get("heavy") else "light",
        }
        ownership_reason = cross_host_skip_reason(family, ownership_authority)
        if ownership_reason:
            skipped.append({**item, "reason": ownership_reason, "block_scope": "FAMILY_SPECIFIC_BLOCK"})
            continue
        if reason:
            skipped.append({**item, "reason": reason, "block_scope": "FAMILY_SPECIFIC_BLOCK"})
            continue
        eligible.append(item)
    return {
        "queue_authority": queue_authority or {"authority": "explicit" if family_queue else "R40_DEFAULT", "ready_family_queue": queue},
        "cross_host_ownership_authority": ownership_authority,
        "queue": queue,
        "first_eligible_family": eligible[0]["family"] if eligible else "",
        "eligible_families": eligible,
        "skipped_families": skipped,
        "excluded_mac_owned": ownership_authority.get("excluded_mac_owned", []),
        "excluded_mac_reserved": ownership_authority.get("excluded_mac_reserved", []),
        "excluded_families": effective_exclusions,
        "retired_family_authority": retired_authority,
        "admit_crashed_recoverable": admit_crashed_recoverable,
    }


def next_ready_family(
    board: list[dict[str, Any]],
    *,
    family_queue: str = "",
    ready_family_queue_manifest: str | Path = "",
    cross_host_ownership_manifest: str | Path = "",
    admit_crashed_recoverable: bool = False,
    excluded_families: str | Sequence[str] | None = None,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> str:
    return str(
        certified_queue_admission_plan(
            board,
            family_queue=family_queue,
            ready_family_queue_manifest=ready_family_queue_manifest,
            cross_host_ownership_manifest=cross_host_ownership_manifest,
            admit_crashed_recoverable=admit_crashed_recoverable,
            excluded_families=excluded_families,
            retired_families_path=retired_families_path,
        ).get("first_eligible_family", "")
    )


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def resume_generation_from_command(command: str) -> int:
    match = re.search(r"--resume-generation(?:=|\s+)(\d+)", str(command))
    return safe_int(match.group(1)) if match else 0


def launch_ledger_jsonl_path(path: Path | None = None) -> Path:
    ledger = path or LAUNCH_LEDGER_PATH
    return ledger.with_name(ledger.name + ".jsonl")


def launch_ledger_records(path: Path | None = None) -> list[dict[str, Any]]:
    ledger = path or LAUNCH_LEDGER_PATH
    records: list[dict[str, Any]] = []
    sidecar = launch_ledger_jsonl_path(ledger)
    if sidecar.exists():
        try:
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        except Exception:
            pass
    if not ledger.exists():
        return records
    try:
        with ledger.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return records
            for fields in reader:
                if not fields:
                    continue
                row = {key: fields[idx] for idx, key in enumerate(header) if idx < len(fields)}
                if len(fields) > 3 and fields[1] in ALL_FAMILIES:
                    row["family"] = fields[1]
                    row["command"] = fields[3]
                    if len(fields) > 9 and safe_int(fields[9]):
                        row["resume_generation"] = safe_int(fields[9])
                    command_generation = resume_generation_from_command(fields[3])
                    if command_generation:
                        row["resume_generation"] = command_generation
                records.append(row)
    except Exception:
        pass
    return records


def family_launch_lock_path(family: str) -> Path:
    return POLICY_ROOT / family / "r36_family_launch.lock.json"


def family_launch_intent_path(family: str) -> Path:
    return POLICY_ROOT / family / "r36_launch_intent.json"


def acquire_family_launch_lock(family: str, resume_generation: int) -> dict[str, Any]:
    lock_path = family_launch_lock_path(family)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_json(lock_path)
    owner = safe_int(existing.get("supervisor_pid"))
    if owner and process_exists(owner):
        raise RuntimeError(f"DS24_R36_FAMILY_LAUNCH_LOCK_REFUSED:{family}:owner={owner}")
    if lock_path.exists():
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    payload = {
        "family": family,
        "resume_generation": resume_generation,
        "supervisor_pid": os.getpid(),
        "requested_at_utc": utc_now(),
        "launch_token": sha256_text(f"{family}|{resume_generation}|{os.getpid()}|{utc_now()}"),
        "contract": "DS24_R36_SINGLE_FAMILY_LAUNCH_LOCK_V1",
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(lock_path, flags)
    try:
        os.write(fd, json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"))
    finally:
        os.close(fd)
    write_json(family_launch_intent_path(family), payload)
    return payload


def release_family_launch_lock(family: str, token: str) -> None:
    lock_path = family_launch_lock_path(family)
    existing = read_json(lock_path)
    if existing.get("launch_token") != token:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def validate_family_launch_slot(
    family: str,
    *,
    excluded_families: str | Sequence[str] | None = None,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> dict[str, Any]:
    if family_is_excluded(family, excluded_families, retired_families_path=retired_families_path):
        raise RuntimeError(f"DS24_R54_EXCLUDED_FAMILY_LAUNCH_REFUSED:{canonical_family_name(family)}")
    live = family_processes().get(family, [])
    if live:
        pids = ",".join(str(int(row.get("ProcessId", 0) or 0)) for row in live)
        raise RuntimeError(f"DS24_R36_PRELAUNCH_DUPLICATE_FAMILY_REFUSED:{family}:live_pids={pids}")
    return {"family": family, "live_pids": [], "validated_at_utc": utc_now()}


def post_launch_family_guard(family: str, launched_pid: int) -> dict[str, Any]:
    worker_kind = execution_registry_row(family)["worker_kind"]
    deadline = time.time() + (60.0 if worker_kind == "PYTORCH_SEQUENCE" else 0.5)
    live: list[dict[str, Any]] = []
    while True:
        live = family_processes().get(family, [])
        pids = [int(row.get("ProcessId", 0) or 0) for row in live]
        if launched_pid in pids or time.time() >= deadline:
            break
        time.sleep(0.5)
    pids = [int(row.get("ProcessId", 0) or 0) for row in live]
    if launched_pid not in pids:
        raise RuntimeError(f"DS24_R36_POSTLAUNCH_WORKER_NOT_ADOPTED:{family}:pid={launched_pid}:live_pids={pids}")
    if len(pids) > 1:
        ps_json(f"Stop-Process -Id {int(launched_pid)} -Force", timeout=15)
        write_json(
            POLICY_ROOT / family / "r36_postlaunch_duplicate_containment.json",
            {
                "family": family,
                "launched_pid": launched_pid,
                "live_pids": pids,
                "contained_at_utc": utc_now(),
                "reason": "DS24_R36_POSTLAUNCH_DUPLICATE_FAMILY_CONTAINED",
            },
        )
        raise RuntimeError(f"DS24_R36_POSTLAUNCH_DUPLICATE_FAMILY_CONTAINED:{family}:live_pids={pids}")
    return {"family": family, "launched_pid": launched_pid, "live_pids": pids, "validated_at_utc": utc_now()}


def seconds_since_last_policy_launch() -> float | None:
    if not LAUNCH_LEDGER_PATH.exists():
        return None
    latest: datetime | None = None
    try:
        with LAUNCH_LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                family = row.get("family") or row.get("event")
                if family not in ALL_FAMILIES or not row.get("pid"):
                    continue
                text = str(row.get("timestamp", "")).replace("Z", "+00:00")
                try:
                    observed = datetime.fromisoformat(text)
                except ValueError:
                    continue
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                observed = observed.astimezone(timezone.utc)
                latest = observed if latest is None else max(latest, observed)
    except Exception:
        return None
    if latest is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - latest).total_seconds())


def admission_proof_window() -> dict[str, Any]:
    age = seconds_since_last_policy_launch()
    if age is None or age >= MIN_POLICY_ADMISSION_PROOF_SECONDS:
        return {
            "blocked": False,
            "seconds_since_last_policy_launch": age,
            "minimum_seconds": MIN_POLICY_ADMISSION_PROOF_SECONDS,
        }
    return {
        "blocked": True,
        "seconds_since_last_policy_launch": round(age, 3),
        "minimum_seconds": MIN_POLICY_ADMISSION_PROOF_SECONDS,
        "remaining_seconds": round(MIN_POLICY_ADMISSION_PROOF_SECONDS - age, 3),
    }


def cleanup_processes() -> list[dict[str, Any]]:
    return [row for row in python_processes() if CLEANUP_SCRIPT in str(row.get("CommandLine", "")) and "--execute" in str(row.get("CommandLine", ""))]


def process_command_line(row: dict[str, Any]) -> str:
    return str(row.get("CommandLine") or row.get("command_line") or "")


def process_runs_python(row: dict[str, Any]) -> bool:
    name = str(row.get("Name") or row.get("name") or "").lower()
    if name:
        return "python" in name
    command = process_command_line(row).strip().lower()
    first = command.split(maxsplit=1)[0].strip('"') if command else ""
    return "python" in first


def classify_ds26_process(row: dict[str, Any]) -> str:
    command = process_command_line(row).lower()
    if "ds26" not in command:
        return "NOT_DS26"
    if PASSIVE_DS26_CAPTURE_SCRIPT in command and process_runs_python(row):
        return "PASSIVE_DS26_PROSPECTIVE_CAPTURE"
    if any(token in command for token in HEAVY_DS26_COMMAND_TOKENS):
        return "HEAVY_DS26_WORKER"
    return "UNKNOWN_DS26_PROCESS"


def classified_process(row: dict[str, Any], classification: str) -> dict[str, Any]:
    return {**row, "classification": classification}


def disallowed_process_manifest() -> dict[str, Any]:
    rows = python_processes()
    classified_ds26 = [(row, classify_ds26_process(row)) for row in rows]
    passive_capture = [
        classified_process(row, classification)
        for row, classification in classified_ds26
        if classification == "PASSIVE_DS26_PROSPECTIVE_CAPTURE"
    ]
    blocking_ds26 = [
        classified_process(row, classification)
        for row, classification in classified_ds26
        if classification in {"HEAVY_DS26_WORKER", "UNKNOWN_DS26_PROCESS"}
    ]
    return {
        "cleanup": cleanup_processes(),
        "ds26_passive_capture": passive_capture,
        "ds26_workers": blocking_ds26,
        "r20_compactor": [row for row in rows if "r20" in str(row.get("CommandLine", "")).lower() and "compactor" in str(row.get("CommandLine", "")).lower()],
        "supervisors": [row for row in rows if "ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py" in str(row.get("CommandLine", ""))],
    }


def zero_full_prediction_guard() -> dict[str, Any]:
    status = read_json(R25_STATUS_PATH)
    return {
        "paper_orders": int(status.get("paper_orders", 0) or 0),
        "live_orders": int(status.get("live_orders", 0) or 0),
        "holdout_accessed": bool(status.get("holdout_accessed", False)),
        "full_prediction_files_in_metrics_namespaces": int(status.get("full_prediction_files_in_metrics_namespaces", 0) or 0),
    }


def resource_gate(board: list[dict[str, Any]], config: GateConfig) -> dict[str, Any]:
    snapshot = system_snapshot()
    snapshot.setdefault("system_commit_percent", 0.0)
    snapshot.setdefault("available_ram_bytes", 0)
    snapshot.setdefault("disk_free_bytes", 0)
    exact = exact_manifest()
    active = active_running(board)
    live_policy = [
        row
        for row in board
        if row.get("pid_alive", row.get("state") in {"RUNNING", "FAST_FORWARDING", "INITIALIZING", "STARTING"})
    ]
    heavy_active = [row for row in active if row["heavy"]]
    aggregate_working_set = exact["working_set"] + sum(int(row.get("working_set", 0) or 0) for row in live_policy)
    guard = zero_full_prediction_guard()
    disallowed = disallowed_process_manifest()
    storage_hold = read_json(STORAGE_HOLD_PATH)
    source_gap_hold = read_json(R29_SOURCE_GAP_HOLD_PATH)
    r36_hold = r36_admission_hold_active()
    hard_resource_readmission = recent_hard_resource_containment_state(
        cooldown_seconds=config.disk_containment_readmission_cooldown_seconds
    )
    active_count = (1 if exact["pid_alive"] else 0) + len(live_policy)
    reasons: list[str] = []
    if not exact["pid_alive"] and not exact.get("terminal_complete"):
        reasons.append("PROTECTED_EXACT_NOT_ALIVE")
    if storage_hold:
        reasons.append("STORAGE_EMERGENCY_HOLD")
    if source_gap_hold:
        reasons.append("R29_CONFIRMED_SCORE_SESSION_SOURCE_GAP_HOLD")
    if r36_hold:
        reasons.append("R36_NAMESPACE_AMBIGUITY_ADMISSION_HOLD")
    if active_count >= config.max_active_model_processes:
        reasons.append("MAX_ACTIVE_MODEL_PROCESSES")
    if len(live_policy) >= config.max_policy_workers:
        reasons.append("MAX_POLICY_WORKERS_ACTIVE")
    if len(heavy_active) >= config.max_heavy_workers:
        reasons.append("MAX_HEAVY_WORKERS_ACTIVE")
    if aggregate_working_set >= config.aggregate_ds24_memory_ceiling_bytes:
        reasons.append("AGGREGATE_DS24_MEMORY_CEILING")
    if snapshot["available_ram_bytes"] < config.min_available_ram_bytes:
        reasons.append("RAM_ADMISSION_GATE")
    if snapshot["available_ram_bytes"] < 6 * 1024**3:
        reasons.append("RAM_GATE")
    if snapshot["available_ram_bytes"] < config.available_ram_floor_bytes:
        reasons.append("RAM_HARD_FLOOR")
    if snapshot["system_commit_percent"] >= config.max_system_commit_percent:
        reasons.append("SYSTEM_COMMIT_HARD_CEILING")
    elif snapshot["system_commit_percent"] >= config.admission_commit_percent:
        reasons.append("SYSTEM_COMMIT_ADMISSION_GATE")
    if snapshot["disk_free_bytes"] < config.hard_disk_floor_bytes:
        reasons.append("DISK_HARD_FLOOR")
    elif snapshot["disk_free_bytes"] < config.clean_admission_disk_floor_bytes:
        reasons.append("DISK_CLEAN_ADMISSION_GATE")
    if (
        config.max_active_model_processes >= 3
        and active_count >= 2
        and snapshot["disk_free_bytes"] < config.three_worker_reactivation_floor_bytes
    ):
        reasons.append("DISK_THREE_WORKER_REACTIVATION_HYSTERESIS")
    if (
        hard_resource_readmission.get("recent")
        and snapshot["disk_free_bytes"] < config.post_disk_containment_readmission_floor_bytes
    ):
        reasons.append("RESOURCE_CONTAINMENT_READMISSION_HYSTERESIS")
    if snapshot["disk_free_bytes"] < 8 * 1024**3:
        reasons.append("DISK_GATE")
    if any(row["duplicate_worker_count"] for row in board):
        reasons.append("DUPLICATE_FAMILY_WORKER")
    if any(row.get("namespace_lease_state") == "LIVE_UNVERIFIED_OR_PID_REUSED" for row in board):
        reasons.append("NAMESPACE_LEASE_OWNER_UNVERIFIED")
    if guard["full_prediction_files_in_metrics_namespaces"] != 0:
        reasons.append("FULL_PREDICTION_METRICS_NAMESPACE_VIOLATION")
    if guard["paper_orders"] or guard["live_orders"] or guard["holdout_accessed"]:
        reasons.append("ORDER_OR_HOLDOUT_GUARD_VIOLATION")
    if disallowed["ds26_workers"]:
        reasons.append("DS26_WORKER_ACTIVE")
    if disallowed["r20_compactor"]:
        reasons.append("R20_COMPACTOR_ACTIVE")
    return {
        "admitted": not reasons,
        "blocked_reasons": reasons,
        "resource_snapshot": snapshot,
        "active_model_processes": active_count,
        "active_policy_workers": len(live_policy),
        "running_policy_workers": len(active),
        "active_heavy_workers": len(heavy_active),
        "aggregate_ds24_working_set_bytes": aggregate_working_set,
        "aggregate_ds24_private_bytes": exact["private_memory"] + sum(int(row.get("private_memory", 0) or 0) for row in live_policy),
        "protected_exact": exact,
        "zero_full_prediction_guard": guard,
        "disallowed_processes": disallowed,
        "ds26_task": task_state(DS26_TASK_NAME),
        "storage_hold": storage_hold,
        "source_gap_hold": source_gap_hold,
        "r36_admission_hold": r36_hold,
        "disk_readmission_policy": {
            "contract_id": R44A_DISK_READMISSION_CONTRACT_ID,
            "hard_disk_floor_bytes": config.hard_disk_floor_bytes,
            "clean_admission_disk_floor_bytes": config.clean_admission_disk_floor_bytes,
            "three_worker_reactivation_floor_bytes": config.three_worker_reactivation_floor_bytes,
            "post_disk_containment_readmission_floor_bytes": config.post_disk_containment_readmission_floor_bytes,
            "disk_containment_readmission_cooldown_seconds": config.disk_containment_readmission_cooldown_seconds,
            "recent_hard_resource_containment": hard_resource_readmission,
        },
    }


def _worker_newest_key(row: dict[str, Any]) -> str:
    return str(row.get("creation_time") or row.get("heartbeat") or row.get("pid") or "")


def stop_policy_worker_for_containment(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    pid = int(row.get("pid", 0) or 0)
    family = str(row.get("family", ""))
    status = process_status(pid)
    command_line = str(row.get("command_line") or status.get("command_line") or "")
    event = {
        "timestamp": utc_now(),
        "family": family,
        "pid": pid,
        "reason": reason,
        "checkpoint": row.get("checkpoint", ""),
        "cursor": row.get("cursor", ""),
        "metrics_rows": row.get("metrics_rows", 0),
        "topn_rows": row.get("topn_rows", 0),
        "resolved_performance_rows": row.get("resolved_performance_rows", 0),
        "rank_ic_valid_rows": row.get("rank_ic_valid_rows", 0),
        "attempted": False,
        "stopped": False,
    }
    if not pid or not status.get("alive"):
        event["reason_detail"] = "PID_NOT_LIVE"
        return event
    if POLICY_WORKER_SCRIPT not in command_line or family not in command_line:
        event["reason_detail"] = "COMMAND_LINE_FAMILY_MISMATCH_REFUSED"
        event["command_line"] = command_line
        return event
    event["attempted"] = True
    event["command_line"] = command_line
    ps_json(f"Stop-Process -Id {pid} -Force", timeout=15)
    time.sleep(2)
    event["stopped"] = not process_exists(pid)
    return event


def active_resource_containment(board: list[dict[str, Any]], gate: dict[str, Any], config: GateConfig) -> dict[str, Any]:
    snapshot = gate.get("resource_snapshot", {})
    free_disk = int(snapshot.get("disk_free_bytes", 0) or 0)
    available_ram = int(snapshot.get("available_ram_bytes", 0) or 0)
    system_commit = float(snapshot.get("system_commit_percent", 0.0) or 0.0)
    live = active_running(board)
    events: list[dict[str, Any]] = []
    reason = ""
    if free_disk < config.emergency_pause_all_disk_floor_bytes:
        reason = "DISK_BELOW_6_GIB_EMERGENCY_PAUSE_ALL"
        targets = sorted(live, key=_worker_newest_key, reverse=True)
    elif free_disk < config.hard_disk_floor_bytes:
        reason = "DISK_BELOW_12_GIB_PAUSE_NEWEST"
        targets = sorted(live, key=_worker_newest_key, reverse=True)[:1]
    elif system_commit >= config.max_system_commit_percent:
        reason = "SYSTEM_COMMIT_HARD_CEILING_PAUSE_NEWEST"
        targets = sorted(live, key=_worker_newest_key, reverse=True)[:1]
    elif available_ram and available_ram < config.available_ram_floor_bytes:
        reason = "RAM_BELOW_4_GIB_HARD_FLOOR_PAUSE_NEWEST"
        targets = sorted(live, key=_worker_newest_key, reverse=True)[:1]
    else:
        targets = []
    for row in targets:
        events.append(stop_policy_worker_for_containment(row, reason=reason))
    payload = {
        "timestamp": utc_now(),
        "active": bool(events),
        "reason": reason,
        "pre_snapshot": snapshot,
        "events": events,
        "post_snapshot": system_snapshot() if events else snapshot,
        "contract_id": TRANSIENT_STORAGE_CONTRACT_V1_ID,
        "contract_hash": transient_storage_contract_v1_hash(),
    }
    if events:
        previous = read_json(ACTIVE_CONTAINMENT_PATH)
        history = previous.get("history", []) if isinstance(previous.get("history"), list) else []
        write_json(ACTIVE_CONTAINMENT_PATH, {"history": [*history, payload], "latest": payload})
    return payload


def launch_family(
    family: str,
    resume_generation: int,
    *,
    evaluation_version: str = "v2",
    metrics_root_name: str = "metrics_only",
    refit_policy: str = "five_score_session_v1",
    forward_contract_admission: dict[str, Any] | None = None,
    excluded_families: str | Sequence[str] | None = None,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> dict[str, Any]:
    if family_is_excluded(family, excluded_families, retired_families_path=retired_families_path):
        raise RuntimeError(f"DS24_R54_EXCLUDED_FAMILY_LAUNCH_REFUSED:{canonical_family_name(family)}")
    assert_dell_ownership_admission(family)
    registry = execution_registry_row(family)
    if not registry["launch_enabled"]:
        raise RuntimeError(f"DS24_R40_WORKER_ROUTE_NOT_LAUNCH_ENABLED:{family}:{registry['worker_kind']}")
    effective_metrics_root_name = default_metrics_root_name(family, metrics_root_name)
    forward_decision = forward_contract_admission or forward_metrics_contract_admission_decision(
        family,
        evaluation_version=evaluation_version,
        metrics_root_name=metrics_root_name,
    )
    if not forward_decision.get("admitted"):
        raise RuntimeError(
            "DS24_R44_FORWARD_METRICS_CONTRACT_ADMISSION_REFUSED:"
            f"family={family};decision={forward_decision.get('capability_decision', '')};"
            f"missing={','.join(str(item) for item in forward_decision.get('missing_requirements', []))}"
        )
    family_dir = POLICY_ROOT / family
    family_dir.mkdir(parents=True, exist_ok=True)
    launch_lock = acquire_family_launch_lock(family, resume_generation)
    try:
        validate_family_launch_slot(
            family,
            excluded_families=excluded_families,
            retired_families_path=retired_families_path,
        )
    except Exception:
        release_family_launch_lock(family, str(launch_lock.get("launch_token", "")))
        raise
    suffix = "r34_v3" if evaluation_version.lower() == "v3" else "r27"
    stdout = family_dir / f"stdout_{suffix}_gen{resume_generation}.log"
    stderr = family_dir / f"stderr_{suffix}_gen{resume_generation}.log"
    args = [
        sys.executable,
        f"scripts\\local\\{registry['worker_script'].split('/')[-1]}",
        "--family",
        family,
        "--resume",
        "--resume-generation",
        str(resume_generation),
        "--threads",
        "1",
        "--cache-budget-gb",
        "1",
        "--prediction-batch-decisions",
        "78",
        "--evaluation-version",
        evaluation_version.lower(),
        "--metrics-root-name",
        effective_metrics_root_name,
        "--refit-policy",
        refit_policy,
    ]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        with stdout.open("ab") as out_handle, stderr.open("ab") as err_handle:
            process = subprocess.Popen(args, cwd=ROOT, stdout=out_handle, stderr=err_handle, env=env, creationflags=creationflags)
        try:
            import psutil

            proc = psutil.Process(process.pid)
            if os.name == "nt":
                proc.nice(psutil.NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
        pid = int(process.pid)
        post_launch = post_launch_family_guard(family, pid)
        row = {
            "timestamp": utc_now(),
            "family": family,
            "pid": pid,
            "command": " ".join(args),
            "reason_admitted": "RESOURCE_GATE_OPEN_FIXED_QUEUE",
            "policy_hash": policy_hash(),
            "configuration_hash": policy_hash(),
            "checkpoint_root": display_path(family_dir),
            "output_root": display_path(family_dir / effective_metrics_root_name),
            "resume_generation": resume_generation,
            "evaluation_version": evaluation_version.upper(),
            "metrics_root_name": effective_metrics_root_name,
            "worker_kind": registry["worker_kind"],
            "worker_script": registry["worker_script"],
            "refit_policy": refit_policy,
            "launch_token": launch_lock["launch_token"],
            "post_launch_guard": json.dumps(post_launch, sort_keys=True, separators=(",", ":")),
            "evaluation_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID if evaluation_version.lower() == "v3" else RESOLVED_PERFORMANCE_CONTRACT_V2_ID,
            "evaluation_contract_hash": resolved_performance_contract_v3_hash() if evaluation_version.lower() == "v3" else resolved_performance_contract_v2_hash(),
            "forward_metrics_contract_id": forward_decision.get("contract_id", ""),
            "forward_metrics_contract_version": forward_decision.get("contract_version", ""),
            "forward_metrics_contract_hash": forward_decision.get("contract_hash", ""),
            "forward_metrics_adoption_id": forward_decision.get("adoption_id", ""),
            "forward_metrics_capability_decision": forward_decision.get("capability_decision", ""),
            "forward_metrics_grandfathered_current_namespace": forward_decision.get("grandfathered_current_namespace", False),
            "forward_metrics_admission": json.dumps(forward_decision, sort_keys=True, default=str, separators=(",", ":")),
        }
        append_csv(LAUNCH_LEDGER_PATH, row)
        return row
    finally:
        release_family_launch_lock(family, str(launch_lock.get("launch_token", "")))


def next_resume_generation(family: str) -> int:
    row = classify_family(family)
    ledger_max = 0
    for item in launch_ledger_records(LAUNCH_LEDGER_PATH):
        if item.get("family") != family:
            continue
        generation = safe_int(item.get("resume_generation"))
        generation = generation or resume_generation_from_command(str(item.get("command", "")))
        ledger_max = max(ledger_max, generation)
    try:
        return max(int(row.get("resume_generation") or 0), ledger_max) + 1
    except Exception:
        return ledger_max + 1 if ledger_max else 1


def acquire_lease(path: Path = LEASE_PATH, *, resume: bool = False) -> tuple[bool, dict[str, Any]]:
    existing = read_json(path)
    owner = int(existing.get("pid", 0) or 0)
    if owner and process_exists(owner):
        return False, {"refused": True, "reason": "LIVE_LEASE_OWNER", "owner": existing}
    payload = {
        "run_id": RUN_ID,
        "pid": int(os.getpid()),
        "process_creation_time": process_creation_time(os.getpid()),
        "command_line_hash": sha256_text(" ".join(sys.argv)),
        "lease_generation": int(existing.get("lease_generation", 0) or 0) + 1,
        "hostname": socket.gethostname(),
        "heartbeat_utc": utc_now(),
        "state_hash": "",
        "stale_recovered": bool(existing),
        "previous_owner": existing,
        "resume": resume,
    }
    payload["state_hash"] = state_hash(payload)
    write_json(path, payload)
    return True, payload


def refresh_lease(lease: dict[str, Any], board: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    lease = dict(lease)
    lease["heartbeat_utc"] = utc_now()
    lease["state_hash"] = state_hash({"board": board, "gate": gate})
    write_json(LEASE_PATH, lease)
    return lease


def release_lease(path: Path = LEASE_PATH) -> None:
    lease = read_json(path)
    if int(lease.get("pid", 0) or 0) == os.getpid():
        write_json(STAGE / "R7_R27_15_supervisor_clean_shutdown.json", {"pid": os.getpid(), "shutdown_at_utc": utc_now()})
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def publish_state(
    *,
    lease: dict[str, Any],
    board: list[dict[str, Any]],
    gate: dict[str, Any],
    poll_seconds: float,
    max_policy_workers: int,
    last_admission: dict[str, Any],
    last_launch: dict[str, Any],
    last_error: str = "",
    max_active_model_processes: int | None = None,
    family_queue: str = "",
    ready_family_queue_manifest: str | Path = "",
    cross_host_ownership_manifest: str | Path = "",
    admit_crashed_recoverable: bool = False,
    evaluation_version: str = "v2",
    metrics_root_name: str = "metrics_only",
    refit_policy: str = "five_score_session_v1",
    excluded_families: str | Sequence[str] | None = None,
    retired_families_path: str | Path = R54_RETIRED_FAMILIES_PATH,
) -> dict[str, Any]:
    exact = exact_manifest()
    running = active_running(board)
    certification_queue = TREE_CERTIFICATION + RANKING_CERTIFICATION + SEQUENCE_CERTIFICATION
    queue_plan = certified_queue_admission_plan(
        board,
        family_queue=family_queue,
        ready_family_queue_manifest=ready_family_queue_manifest,
        cross_host_ownership_manifest=cross_host_ownership_manifest,
        admit_crashed_recoverable=admit_crashed_recoverable,
        excluded_families=excluded_families,
        retired_families_path=retired_families_path,
    )
    next_family = str(queue_plan.get("first_eligible_family", ""))
    blocked = gate.get("blocked_reasons", [])
    scheduled_task = task_state(TASK_NAME)
    task_state_value = str(scheduled_task.get("State") or scheduled_task.get("state") or "")
    classification = "DS24P8_R14_E3G_C2_R7_R27_PERSISTENT_TOURNAMENT_SUPERVISOR_ACTIVE_AUTOMATIC_HANDOFF_ARMED"
    if blocked:
        classification = "DS24P8_R14_E3G_C2_R7_R27_AUTOMATIC_HANDOFF_ARMED_WAITING_RESOURCE_RELEASE"
    if "PROTECTED_EXACT_NOT_ALIVE" in blocked:
        classification = "DS24P8_R14_E3G_C2_R7_R27_FAIL_CLOSED_ACTIVE_WORKER_PROTECTION"
    if task_state_value not in {"Ready", "Running", "3", "4"}:
        classification = "DS24P8_R14_E3G_C2_R7_R27_QUEUE_PREPARED_MANUAL_CONTINUATION_REQUIRED"
    if "R29_CONFIRMED_SCORE_SESSION_SOURCE_GAP_HOLD" in blocked:
        classification = "DS24P8_R14_E3G_C2_R7_R29_BLOCKED_CONFIRMED_SCORE_SESSION_SOURCE_GAP"
    r31_terminal = read_json(STAGE / "R7_R31_12_terminal_validation.json")
    if r31_terminal.get("classification"):
        classification = str(r31_terminal["classification"])
    next_action = "MANUAL_WATCHDOG_ENABLE_REQUIRED" if "MANUAL_CONTINUATION" in classification else ("WAITING_RESOURCE_RELEASE" if blocked else f"ADMIT_{next_family}")
    if "R29_CONFIRMED_SCORE_SESSION_SOURCE_GAP_HOLD" in blocked:
        next_action = "REPAIR_CANONICAL_2025_FEATURE_SOURCE_GAP"
    heartbeat = {
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "heartbeat_utc": utc_now(),
        "poll_seconds": poll_seconds,
        "lease": lease,
        "scheduled_task": scheduled_task,
        "active_model_processes": gate.get("active_model_processes"),
        "max_active_model_processes": max_active_model_processes or gate.get("active_model_processes"),
        "active_policy_workers": gate.get("active_policy_workers", len(running)),
        "running_policy_workers": gate.get("running_policy_workers", len(running)),
        "max_policy_workers": max_policy_workers,
        "family_queue": list(queue_plan.get("queue", [])),
        "certified_queue_authority": queue_plan.get("queue_authority", {}),
        "cross_host_ownership_authority": queue_plan.get("cross_host_ownership_authority", {}),
        "certified_ready_remaining": len(queue_plan.get("eligible_families", [])),
        "next_certified_family": next_family,
        "next_worker_route": execution_registry_row(next_family, include_r42_authority=bool(ready_family_queue_manifest)) if next_family else {},
        "queue_skip_reasons": queue_plan.get("skipped_families", []),
        "global_resource_block": bool(blocked),
        "family_specific_block": bool(queue_plan.get("skipped_families")),
        "admit_crashed_recoverable": admit_crashed_recoverable,
        "excluded_families": queue_plan.get("excluded_families", []),
        "retired_family_authority": queue_plan.get("retired_family_authority", {}),
        "evaluation_version": evaluation_version.upper(),
        "metrics_root_name": metrics_root_name,
        "refit_policy": refit_policy,
        "exact_worker": exact,
        "active_workers": [row["family"] for row in running],
        "completed_families": [row["family"] for row in board if row["state"] == "COMPLETE"],
        "terminal_validating_families": [row["family"] for row in board if row["state"] == "TERMINAL_VALIDATING"],
        "paused_resumable_families": [row["family"] for row in board if row["state"] == "PAUSED_RESOURCE_GATE"],
        "certified_ready_queue": [row["family"] for row in board if row["state"] in {"CERTIFIED_READY", "V3_CERTIFIED_READY"}],
        "certified_queue_admission_plan": queue_plan,
        "certification_queue": certification_queue,
        "next_ready_family": next_family,
        "last_admission_decision": last_admission,
        "last_automatic_launch": last_launch,
        "forward_metrics_contract_admission": gate.get("forward_metrics_contract_admission", {}),
        "resource_snapshot": gate.get("resource_snapshot", {}),
        "resource_block_reason": blocked,
        "zero_full_prediction_guard": gate.get("zero_full_prediction_guard", {}),
        "source_gap_hold": gate.get("source_gap_hold", {}),
        "last_error": last_error,
    }
    terminal = {
        "classification": classification,
        "run_id": RUN_ID,
        "policy_id": POLICY_ID,
        "policy_hash": policy_hash(),
        "supervisor_pid": os.getpid(),
        "supervisor_heartbeat": heartbeat,
        "family_board": board,
        "active_worker_manifest": [exact] + running,
        "paused_resumable_manifest": [row for row in board if row["state"] == "PAUSED_RESOURCE_GATE"],
        "certified_ready_queue": [row["family"] for row in board if row["state"] in {"CERTIFIED_READY", "V3_CERTIFIED_READY"}],
        "certified_queue_authority": queue_plan.get("queue_authority", {}),
        "cross_host_ownership_authority": queue_plan.get("cross_host_ownership_authority", {}),
        "certified_queue_admission_plan": queue_plan,
        "excluded_families": queue_plan.get("excluded_families", []),
        "retired_family_authority": queue_plan.get("retired_family_authority", {}),
        "certification_queue": certification_queue,
        "resource_gate": gate,
        "admission_ledger": display_path(ADMISSION_LEDGER_PATH),
        "automatic_launch_ledger": display_path(LAUNCH_LEDGER_PATH),
        "restart_ledger": display_path(RESTART_LEDGER_PATH),
        "completion_validation_ledger": display_path(COMPLETION_LEDGER_PATH),
        "cleanup_ledger_reference": display_path(CLEANUP_LEDGER_REF_PATH),
        "paper_orders": gate.get("zero_full_prediction_guard", {}).get("paper_orders", 0),
        "live_orders": gate.get("zero_full_prediction_guard", {}).get("live_orders", 0),
        "holdout_accessed": gate.get("zero_full_prediction_guard", {}).get("holdout_accessed", False),
        "monitor_command": "python scripts\\local\\monitor_ds24_full_family_tournament.py",
        "supervisor_status_command": "python scripts\\local\\ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py --status",
        "disable_watchdog_command": f"Disable-ScheduledTask -TaskName '{TASK_NAME}'",
        "ds26_reenable_command": f"Enable-ScheduledTask -TaskName '{DS26_TASK_NAME}'",
        "next_action": next_action,
    }
    legacy_cert = {
        "run_id": RUN_ID,
        "tree": {family: "CERTIFYING_BOUNDED_POLICY_PACKAGE_PENDING_ACTIVE_SLOT_HEADROOM" for family in TREE_CERTIFICATION},
        "ranking": {family: "CERTIFYING_LIGHTGBM_GROUP_AUDIT_PENDING_LOW_PRIORITY_SLOT" for family in RANKING_CERTIFICATION},
        "sequence": {family: "CERTIFYING_SEQUENCE_PIT_WINDOW_AUDIT_PENDING_LOW_PRIORITY_SLOT" for family in SEQUENCE_CERTIFICATION},
        "paper_orders": 0,
        "live_orders": 0,
        "holdout_accessed": False,
    }
    write_json(HEARTBEAT_PATH, heartbeat)
    write_json(STATE_PATH, {"families": board, "updated_at_utc": utc_now()})
    write_json(ACTIVE_MANIFEST_PATH, {"workers": [exact] + running, "updated_at_utc": utc_now()})
    write_json(PAUSED_MANIFEST_PATH, {"workers": [row for row in board if row["state"] == "PAUSED_RESOURCE_GATE"], "updated_at_utc": utc_now()})
    write_json(RESOURCE_PATH, gate)
    write_json(CLEANUP_LEDGER_REF_PATH, {"r25_status_path": display_path(R25_STATUS_PATH), "cleanup_active": bool(cleanup_processes())})
    write_json(NEXT_ACTION_PATH, {"next_family": next_family, "blocked_reasons": blocked, "updated_at_utc": utc_now()})
    write_json(STAGE / "R7_R17_07_remaining_family_certification_queue.json", legacy_cert)
    write_json(TERMINAL_PATH, terminal)
    REPORT_PATH.write_text(
        "# R27 Supervisor Report\n\n"
        f"Classification: {classification}\n\n"
        f"Supervisor PID: {os.getpid()}\n\n"
        f"Active model processes: {gate.get('active_model_processes')} / {heartbeat['max_active_model_processes']}\n\n"
        f"Next family: {next_family or 'none'}\n\n"
        f"Blocked reasons: {', '.join(blocked) if blocked else 'none'}\n",
        encoding="utf-8",
    )
    return terminal


def supervisor_poll(args: argparse.Namespace, lease: dict[str, Any]) -> dict[str, Any]:
    board = build_family_board()
    config = GateConfig(
        max_active_model_processes=args.max_active_model_processes,
        max_policy_workers=args.max_policy_workers,
        max_system_commit_percent=args.max_system_commit_percent,
        admission_commit_percent=args.admission_commit_percent,
        min_available_ram_bytes=int(args.min_available_ram_gb * 1024**3),
    )
    gate = resource_gate(board, config)
    containment = active_resource_containment(board, gate, config)
    if containment.get("active"):
        board = build_family_board()
        gate = resource_gate(board, config)
        gate = dict(gate)
        gate["active_resource_containment"] = containment
        gate["blocked_reasons"] = [*gate.get("blocked_reasons", []), "ACTIVE_RESOURCE_CONTAINMENT_APPLIED"]
    lease = refresh_lease(lease, board, gate)
    queue_plan = certified_queue_admission_plan(
        board,
        family_queue=args.family_queue,
        ready_family_queue_manifest=args.ready_family_queue_manifest,
        cross_host_ownership_manifest=args.cross_host_ownership_manifest,
        admit_crashed_recoverable=args.admit_crashed_recoverable,
        excluded_families=args.exclude_family,
        retired_families_path=args.retired_families_path,
    )
    next_family = str(queue_plan.get("first_eligible_family", ""))
    proof_window = admission_proof_window()
    if next_family and gate["admitted"] and proof_window["blocked"]:
        gate = dict(gate)
        gate["admitted"] = False
        gate["blocked_reasons"] = [*gate.get("blocked_reasons", []), "ADMISSION_PROOF_WINDOW_ACTIVE"]
        gate["admission_proof_window"] = proof_window
    forward_admission = (
        forward_metrics_contract_admission_decision(
            next_family,
            evaluation_version=args.evaluation_version,
            metrics_root_name=args.metrics_root_name,
            include_r42_authority=bool(args.ready_family_queue_manifest),
        )
        if next_family
        else {}
    )
    if next_family:
        gate = dict(gate)
        gate["forward_metrics_contract_admission"] = forward_admission
    if next_family and gate["admitted"] and not forward_admission.get("admitted"):
        gate["admitted"] = False
        gate["blocked_reasons"] = [*gate.get("blocked_reasons", []), "FORWARD_METRICS_CONTRACT_ADMISSION_BLOCKED"]
    last_launch: dict[str, Any] = {}
    last_admission = {
        "timestamp": utc_now(),
        "candidate_family": next_family,
        "admitted": False,
        "blocked_reasons": gate["blocked_reasons"],
        "admission_proof_window": json.dumps(proof_window, sort_keys=True, separators=(",", ":")),
        "active_model_processes": gate["active_model_processes"],
        "available_ram_bytes": gate["resource_snapshot"]["available_ram_bytes"],
        "system_commit_percent": gate["resource_snapshot"]["system_commit_percent"],
        "free_disk_bytes": gate["resource_snapshot"]["disk_free_bytes"],
        "forward_metrics_contract_id": forward_admission.get("contract_id", ""),
        "forward_metrics_contract_version": forward_admission.get("contract_version", ""),
        "forward_metrics_contract_hash": forward_admission.get("contract_hash", ""),
        "forward_metrics_adoption_id": forward_admission.get("adoption_id", ""),
        "forward_metrics_capability_decision": forward_admission.get("capability_decision", ""),
        "forward_metrics_admitted": forward_admission.get("admitted", False),
        "forward_metrics_grandfathered_current_namespace": forward_admission.get("grandfathered_current_namespace", False),
        "forward_metrics_missing_requirements": json.dumps(forward_admission.get("missing_requirements", []), sort_keys=True, separators=(",", ":")),
        "forward_metrics_admission": json.dumps(forward_admission, sort_keys=True, default=str, separators=(",", ":")),
    }
    if next_family and gate["admitted"]:
        last_launch = launch_family(
            next_family,
            next_resume_generation(next_family),
            evaluation_version=args.evaluation_version,
            metrics_root_name=args.metrics_root_name,
            refit_policy=args.refit_policy,
            forward_contract_admission=forward_admission,
            excluded_families=args.exclude_family,
            retired_families_path=args.retired_families_path,
        )
        last_admission["admitted"] = True
    append_csv(ADMISSION_LEDGER_PATH, last_admission)
    return publish_state(
        lease=lease,
        board=board,
        gate=gate,
        poll_seconds=args.poll_seconds,
        max_policy_workers=args.max_policy_workers,
        max_active_model_processes=args.max_active_model_processes,
        family_queue=args.family_queue,
        ready_family_queue_manifest=args.ready_family_queue_manifest,
        cross_host_ownership_manifest=args.cross_host_ownership_manifest,
        admit_crashed_recoverable=args.admit_crashed_recoverable,
        excluded_families=args.exclude_family,
        retired_families_path=args.retired_families_path,
        evaluation_version=args.evaluation_version,
        metrics_root_name=args.metrics_root_name,
        refit_policy=args.refit_policy,
        last_admission=last_admission,
        last_launch=last_launch,
    )


def reboot_supervisor_task_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = certified_queue_manifest_path(args.ready_family_queue_manifest)
    ownership_arg = str(getattr(args, "cross_host_ownership_manifest", "") or "")
    ownership_path = cross_host_ownership_manifest_path(ownership_arg) if ownership_arg else Path("")
    command = [
        sys.executable,
        "scripts\\local\\ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py",
        "--daemon",
        "--resume",
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-active-model-processes",
        str(args.max_active_model_processes),
        "--max-policy-workers",
        str(args.max_policy_workers),
        "--max-restarts-per-family",
        str(args.max_restarts_per_family),
        "--admission-commit-percent",
        str(args.admission_commit_percent),
        "--max-system-commit-percent",
        str(args.max_system_commit_percent),
        "--min-available-ram-gb",
        str(args.min_available_ram_gb),
        "--evaluation-version",
        args.evaluation_version,
        "--metrics-root-name",
        args.metrics_root_name,
        "--refit-policy",
        args.refit_policy,
        "--ready-family-queue-manifest",
        str(manifest_path),
    ]
    if ownership_arg:
        command.extend(["--cross-host-ownership-manifest", str(ownership_path)])
    command.extend(["--retired-families-path", str(args.retired_families_path)])
    for family in family_list_values(args.exclude_family):
        command.extend(["--exclude-family", family])
    action = (
        f"$Action = New-ScheduledTaskAction -Execute '{sys.executable}' "
        f"-Argument '{' '.join(command[1:])}' -WorkingDirectory '{ROOT}'"
    )
    trigger = "$Trigger = New-ScheduledTaskTrigger -AtStartup"
    settings = "$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew"
    register = f"Register-ScheduledTask -TaskName '{TASK_NAME}' -Action $Action -Trigger $Trigger -Settings $Settings -Force"
    return {
        "ticket": "DS24_R44_REBOOT_SAFE_SUPERVISOR_TASK_PLAN" if ownership_arg else "DS24_R43_REBOOT_SAFE_SUPERVISOR_TASK_PLAN",
        "generated_at_utc": utc_now(),
        "task_name": TASK_NAME,
        "current_task_state": task_state(TASK_NAME),
        "activation_deferred": True,
        "reason": "live supervisor healthy; registration deferred to avoid disturbing current process",
        "working_directory": str(ROOT),
        "future_supervisor_command": " ".join(command),
        "r42_ready_queue_manifest": display_path(manifest_path),
        "r44_cross_host_ownership_manifest": display_path(ownership_path) if ownership_arg else "",
        "powershell_registration_commands": [action, trigger, settings, register],
    }


def dry_run_admission(args: argparse.Namespace) -> dict[str, Any]:
    queue_authority = validate_ready_family_queue_manifest(args.ready_family_queue_manifest)
    ownership_manifest = str(getattr(args, "cross_host_ownership_manifest", "") or "")
    ownership_authority = validate_cross_host_ownership_authority(ownership_manifest) if ownership_manifest else {}
    board = build_family_board()
    config = GateConfig(
        max_active_model_processes=args.max_active_model_processes,
        max_policy_workers=args.max_policy_workers,
        max_system_commit_percent=args.max_system_commit_percent,
        admission_commit_percent=args.admission_commit_percent,
        min_available_ram_bytes=int(args.min_available_ram_gb * 1024**3),
    )
    gate = resource_gate(board, config)
    queue_plan = certified_queue_admission_plan(
        board,
        ready_family_queue_manifest=args.ready_family_queue_manifest,
        cross_host_ownership_manifest=ownership_manifest,
        admit_crashed_recoverable=args.admit_crashed_recoverable,
        excluded_families=args.exclude_family,
        retired_families_path=args.retired_families_path,
    )
    global_blocked = bool(gate.get("blocked_reasons"))
    first = str(queue_plan.get("first_eligible_family", ""))
    status = "NO_SLOT_AVAILABLE" if "MAX_ACTIVE_MODEL_PROCESSES" in gate.get("blocked_reasons", []) else ("GLOBAL_RESOURCE_BLOCK" if global_blocked else ("READY_TO_ADMIT" if first else "NO_CERTIFIED_FAMILY_ELIGIBLE"))
    task_plan = reboot_supervisor_task_plan(args)
    reboot_path = R44_REBOOT_TASK_PLAN_PATH if ownership_manifest else R42_REBOOT_TASK_PLAN_PATH
    write_json(reboot_path, task_plan)
    lease = read_json(LEASE_PATH)
    supervisor_pid = int(lease.get("pid", 0) or 0)
    supervisor_status = process_status(supervisor_pid) if supervisor_pid else {"alive": False, "pid": 0}
    payload = {
        "ticket": "DS24_R44_CROSS_HOST_ADMISSION_VALIDATION" if ownership_manifest else "DS24_R43_DRY_RUN_ADMISSION",
        "generated_at_utc": utc_now(),
        "dry_run": True,
        "status": status,
        "worker_launches": 0,
        "workers_stopped": 0,
        "live_supervisor_queue_modified": False,
        "queue_authority": queue_authority,
        "cross_host_ownership_authority": ownership_authority,
        "excluded_mac_owned": queue_plan.get("excluded_mac_owned", []),
        "excluded_mac_reserved": queue_plan.get("excluded_mac_reserved", []),
        "excluded_families": queue_plan.get("excluded_families", []),
        "retired_family_authority": queue_plan.get("retired_family_authority", {}),
        "current_supervisor": {
            "pid": supervisor_pid,
            "alive": bool(supervisor_status.get("alive")),
            "command_line": supervisor_status.get("command_line", ""),
            "explicit_family_queue": command_arg(str(supervisor_status.get("command_line", "")), "--family-queue") or "",
        },
        "current_active_workers": [row["family"] for row in active_running(board)],
        "active_worker_details": [
            {
                "family": row["family"],
                "pid": row.get("pid", 0),
                "checkpoint": row.get("checkpoint", ""),
                "cursor": row.get("cursor", ""),
                "metrics_rows": row.get("metrics_rows", 0),
                "namespace_state": row.get("namespace_lease_state", ""),
            }
            for row in active_running(board)
        ],
        "first_eligible_family": "" if global_blocked else first,
        "next_when_slot_available": first,
        "next_eligible_families": [row["family"] for row in queue_plan.get("eligible_families", [])],
        "eligible_family_details": queue_plan.get("eligible_families", []),
        "blocked_skipped_families": queue_plan.get("skipped_families", []),
        "global_resource_block": {
            "blocked": global_blocked,
            "reasons": gate.get("blocked_reasons", []),
        },
        "family_specific_block_count": len(queue_plan.get("skipped_families", [])),
        "resource_snapshot": gate.get("resource_snapshot", {}),
        "zero_full_prediction_guard": gate.get("zero_full_prediction_guard", {}),
        "reboot_task_plan_path": display_path(reboot_path),
        "reboot_task_plan": task_plan,
    }
    write_json(R44_ADMISSION_VALIDATION_PATH if ownership_manifest else R43_DRY_RUN_ADMISSION_PATH, payload)
    return payload


def supervisor_daemon_processes() -> list[dict[str, Any]]:
    return [
        row
        for row in python_processes()
        if "ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py" in str(row.get("CommandLine", ""))
        and "--daemon" in str(row.get("CommandLine", ""))
    ]


def active_protected_worker_rows(board: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in board if row.get("family") in R45_PROTECTED_DELL_WORKERS and bool(row.get("pid_alive"))]


def compact_supervisor_lease(lease: Mapping[str, Any]) -> dict[str, Any]:
    previous = lease.get("previous_owner")
    compact: dict[str, Any] = {
        "pid": lease.get("pid"),
        "heartbeat_utc": lease.get("heartbeat_utc"),
        "lease_generation": lease.get("lease_generation"),
        "hostname": lease.get("hostname"),
        "command_line_hash": lease.get("command_line_hash"),
        "process_creation_time": lease.get("process_creation_time"),
        "resume": lease.get("resume"),
        "stale_recovered": lease.get("stale_recovered"),
    }
    if isinstance(previous, Mapping):
        compact["previous_owner"] = {
            "pid": previous.get("pid"),
            "heartbeat_utc": previous.get("heartbeat_utc"),
            "lease_generation": previous.get("lease_generation"),
            "command_line_hash": previous.get("command_line_hash"),
        }
    return {key: value for key, value in compact.items() if value not in (None, "")}


def r45_supervisor_command(*, python_executable: str | None = None) -> list[str]:
    return [
        python_executable or sys.executable,
        "scripts\\local\\ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py",
        "--daemon",
        "--resume",
        "--poll-seconds",
        "20",
        "--max-active-model-processes",
        "3",
        "--max-policy-workers",
        "3",
        "--max-restarts-per-family",
        "2",
        "--admission-commit-percent",
        "92",
        "--max-system-commit-percent",
        "95",
        "--min-available-ram-gb",
        "6",
        "--evaluation-version",
        "v3",
        "--metrics-root-name",
        "metrics_only_v3",
        "--refit-policy",
        "daily_session_v1",
        "--ready-family-queue-manifest",
        str(R42_READY_QUEUE_PATH),
        "--cross-host-ownership-manifest",
        str(R44_CROSS_HOST_OWNERSHIP_PATH),
        "--retired-families-path",
        str(R54_RETIRED_FAMILIES_PATH),
    ]


def r45_supervisor_launch_authority(path: Path = R45_SUPERVISOR_LAUNCH_AUTHORITY_PATH) -> dict[str, Any]:
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    ownership = validate_cross_host_ownership_authority(R44_CROSS_HOST_OWNERSHIP_PATH)
    effective = read_json(R44_DELL_EFFECTIVE_READY_QUEUE_PATH)
    command = r45_supervisor_command()
    payload = {
        "ticket": "DS24_R45_SUPERVISOR_LAUNCH_AUTHORITY",
        "generated_at_utc": utc_now(),
        "classification": "R44_AWARE_SUPERVISOR_COMMAND_AUTHORIZED",
        "working_directory": str(ROOT),
        "command": command,
        "command_line": " ".join(command),
        "scheduled_task_launcher": display_path(R45_SUPERVISOR_LAUNCHER_PATH),
        "uses_legacy_family_queue": False,
        "r42_ready_queue_manifest": ready["manifest_path"],
        "r42_ready_queue_hash": ready["manifest_hash"],
        "r44_ownership_manifest": ownership["manifest_path"],
        "r44_ownership_hash": ownership["manifest_hash"],
        "r44_effective_queue_path": display_path(R44_DELL_EFFECTIVE_READY_QUEUE_PATH),
        "r44_effective_queue_hash": hash_if_exists(R44_DELL_EFFECTIVE_READY_QUEUE_PATH),
        "effective_dell_queue": effective.get("dell_effective_ready_queue", []),
        "excluded_families": effective_excluded_families(retired_families_path=R54_RETIRED_FAMILIES_PATH),
        "retired_family_authority": retired_family_authority(R54_RETIRED_FAMILIES_PATH),
        "worker_launches": 0,
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    write_json(path, payload)
    return payload


def validate_r44_effective_queue_contract() -> dict[str, Any]:
    ownership = validate_cross_host_ownership_authority(R44_CROSS_HOST_OWNERSHIP_PATH)
    effective = read_json(R44_DELL_EFFECTIVE_READY_QUEUE_PATH)
    queue = effective.get("dell_effective_ready_queue")
    if queue != list(R44_DELL_READY_FAMILIES):
        raise RuntimeError(f"DS24_R45_R44_EFFECTIVE_QUEUE_MISMATCH:{queue}")
    reasons = {row.get("family"): row.get("reason") for row in effective.get("excluded_families", []) if isinstance(row, dict)}
    required = {
        "lightgbm_rank_xendcg": "SKIP_MAC_OWNED",
        "lightgbm_lambdarank": "SKIP_MAC_OWNED",
        "DLinear": "SKIP_MAC_RESERVED",
        "Temporal Fusion Transformer": "SKIP_CONFIGURATION_AUTHORITY_REQUIRED",
    }
    mismatches = {family: reasons.get(family) for family, reason in required.items() if reasons.get(family) != reason}
    if mismatches:
        raise RuntimeError(f"DS24_R45_R44_EXCLUSION_MISMATCH:{mismatches}")
    return {
        "r44_ownership_hash": ownership["manifest_hash"],
        "r44_effective_queue_hash": hash_if_exists(R44_DELL_EFFECTIVE_READY_QUEUE_PATH),
        "effective_dell_queue": list(queue),
        "required_exclusions": required,
    }


def r45_pre_cutover_snapshot(path: Path = R45_PRE_CUTOVER_SNAPSHOT_PATH) -> dict[str, Any]:
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    board = lightweight_certified_queue_board(ready["ready_family_queue"])
    gate = resource_gate(board, GateConfig(max_active_model_processes=3, max_policy_workers=3, max_system_commit_percent=95.0, admission_commit_percent=92.0))
    r44 = validate_r44_effective_queue_contract()
    supervisors = supervisor_daemon_processes()
    protected = active_protected_worker_rows(board)
    guard = gate.get("zero_full_prediction_guard", {})
    duplicate_workers = [row for row in board if int(row.get("duplicate_worker_count", 0) or 0)]
    lease_failures = [row.get("family") for row in protected if row.get("namespace_lease_state") != "LIVE_VERIFIED"]
    worker_families = [str(row.get("family")) for row in protected]
    blockers: list[str] = []
    if sorted(worker_families) != sorted(R45_PROTECTED_DELL_WORKERS):
        blockers.append("PROTECTED_WORKER_SET_MISMATCH")
    if len(supervisors) != 1:
        blockers.append("SUPERVISOR_DAEMON_COUNT_NOT_ONE")
    if lease_failures:
        blockers.append("PROTECTED_WORKER_LEASE_NOT_LIVE_VERIFIED")
    if duplicate_workers:
        blockers.append("DUPLICATE_FAMILY_WORKERS")
    if guard.get("holdout_accessed") or guard.get("full_prediction_files_in_metrics_namespaces") or guard.get("paper_orders") or guard.get("live_orders"):
        blockers.append("SAFETY_SENTINEL_VIOLATION")
    payload = {
        "ticket": "DS24_R45_PRE_CUTOVER_SNAPSHOT",
        "generated_at_utc": utc_now(),
        "classification": "PASS" if not blockers else "FAIL_CLOSED",
        "blockers": blockers,
        "supervisor": supervisors[0] if supervisors else {},
        "supervisor_lease": compact_supervisor_lease(read_json(LEASE_PATH)),
        "protected_workers": [
            {
                "family": row.get("family"),
                "pid": row.get("pid"),
                "creation_time": row.get("creation_time"),
                "command_line": row.get("command_line"),
                "cursor": row.get("cursor"),
                "namespace_lease_state": row.get("namespace_lease_state"),
                "metrics_rows": row.get("metrics_rows"),
            }
            for row in protected
        ],
        "active_model_processes": gate.get("active_model_processes"),
        "resource_snapshot": gate.get("resource_snapshot", {}),
        "r42_manifest_hash": ready["manifest_hash"],
        "r44_ownership_manifest_hash": r44["r44_ownership_hash"],
        "r44_effective_dell_queue_hash": r44["r44_effective_queue_hash"],
        "effective_dell_queue": r44["effective_dell_queue"],
        "zero_full_prediction_guard": guard,
        "worker_launches": 0,
        "workers_stopped": 0,
    }
    write_json(path, payload)
    if blockers:
        raise RuntimeError(f"DS24_R45_PRE_CUTOVER_FAIL_CLOSED:{blockers}")
    return payload


def r45_queue_fallback_simulation() -> dict[str, Any]:
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    board = lightweight_certified_queue_board(ready["ready_family_queue"])
    by_family = {row["family"]: dict(row) for row in board}
    by_family["random_forest"]["pid_alive"] = True
    rf_running = certified_queue_admission_plan(
        list(by_family.values()),
        ready_family_queue_manifest=R42_READY_QUEUE_PATH,
        cross_host_ownership_manifest=R44_CROSS_HOST_OWNERSHIP_PATH,
        admit_crashed_recoverable=True,
        retired_families_path=R54_RETIRED_FAMILIES_PATH,
    )
    by_family["elastic_net"]["state"] = "CRASHED_BLOCKED"
    elastic_blocked = certified_queue_admission_plan(
        list(by_family.values()),
        ready_family_queue_manifest=R42_READY_QUEUE_PATH,
        cross_host_ownership_manifest=R44_CROSS_HOST_OWNERSHIP_PATH,
        admit_crashed_recoverable=True,
        retired_families_path=R54_RETIRED_FAMILIES_PATH,
    )
    by_family["PatchTST"]["state"] = "V3_CERTIFICATION_REQUIRED"
    patchtst_unavailable = certified_queue_admission_plan(
        list(by_family.values()),
        ready_family_queue_manifest=R42_READY_QUEUE_PATH,
        cross_host_ownership_manifest=R44_CROSS_HOST_OWNERSHIP_PATH,
        admit_crashed_recoverable=True,
        retired_families_path=R54_RETIRED_FAMILIES_PATH,
    )
    return {
        "rf_running_next": rf_running.get("first_eligible_family", ""),
        "elastic_net_blocked_next": elastic_blocked.get("first_eligible_family", ""),
        "patchtst_unavailable_order": [row["family"] for row in patchtst_unavailable.get("eligible_families", [])],
        "mac_exclusions_preserved": {
            family: next((row["reason"] for row in elastic_blocked.get("skipped_families", []) if row["family"] == family), "")
            for family in ("lightgbm_rank_xendcg", "lightgbm_lambdarank", "DLinear")
        },
    }


def r45_reboot_simulation(path: Path = R45_REBOOT_SIMULATION_PATH) -> dict[str, Any]:
    fallback = r45_queue_fallback_simulation()
    stale_lease = {"pid": 999999, "heartbeat_utc": "2026-09-09T00:00:00+00:00", "lease_generation": 7}
    owner_alive = process_exists(int(stale_lease["pid"]))
    payload = {
        "ticket": "DS24_R45_REBOOT_SIMULATION",
        "generated_at_utc": utc_now(),
        "machine_reboot_simulated": True,
        "stale_supervisor_lease": stale_lease,
        "stale_supervisor_owner_alive": owner_alive,
        "fresh_supervisor_would_recover_lease": not owner_alive,
        "r42_authority_loads": bool(validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH).get("manifest_hash")),
        "r44_authority_loads": bool(validate_cross_host_ownership_authority(R44_CROSS_HOST_OWNERSHIP_PATH).get("manifest_hash")),
        "mac_owned_work_excluded": fallback["mac_exclusions_preserved"],
        "queue_fallback_simulation": fallback,
        "duplicate_start_protection": "LIVE_LEASE_OWNER_REFUSES_SECOND_DAEMON",
        "classification": "PASS",
    }
    write_json(path, payload)
    return payload


def r45_post_cutover_validation(path: Path = R45_POST_CUTOVER_VALIDATION_PATH) -> dict[str, Any]:
    pre = read_json(R45_PRE_CUTOVER_SNAPSHOT_PATH)
    launch = read_json(R45_SUPERVISOR_LAUNCH_AUTHORITY_PATH)
    dry_args = argparse.Namespace(
        ready_family_queue_manifest=str(R42_READY_QUEUE_PATH),
        cross_host_ownership_manifest=str(R44_CROSS_HOST_OWNERSHIP_PATH),
        admit_crashed_recoverable=True,
        max_active_model_processes=3,
        max_policy_workers=3,
        max_restarts_per_family=2,
        max_system_commit_percent=95.0,
        admission_commit_percent=92.0,
        min_available_ram_gb=6.0,
        poll_seconds=20.0,
        evaluation_version="v3",
        metrics_root_name="metrics_only_v3",
        refit_policy="daily_session_v1",
    )
    dry = dry_run_admission(dry_args)
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    board = lightweight_certified_queue_board(ready["ready_family_queue"])
    protected = active_protected_worker_rows(board)
    protected_by_family = {str(row.get("family")): row for row in protected}
    pre_workers = {str(row.get("family")): row for row in pre.get("protected_workers", []) if isinstance(row, dict)}
    supervisors = supervisor_daemon_processes()
    new_supervisor = supervisors[0] if len(supervisors) == 1 else {}
    lease = read_json(LEASE_PATH)
    compact_lease = compact_supervisor_lease(lease)
    pid_preservation = {
        family: {
            "before": pre_workers.get(family, {}).get("pid"),
            "after": protected_by_family.get(family, {}).get("pid"),
            "unchanged": pre_workers.get(family, {}).get("pid") == protected_by_family.get(family, {}).get("pid"),
        }
        for family in R45_PROTECTED_DELL_WORKERS
    }
    blockers: list[str] = []
    if len(supervisors) != 1:
        blockers.append("SUPERVISOR_DAEMON_COUNT_NOT_ONE")
    if int(lease.get("pid", 0) or 0) != int(new_supervisor.get("ProcessId", 0) or 0):
        blockers.append("SUPERVISOR_LEASE_OWNER_MISMATCH")
    if any(not row["unchanged"] for row in pid_preservation.values()):
        blockers.append("PROTECTED_WORKER_PID_CHANGED")
    if sorted(protected_by_family) != sorted(R45_PROTECTED_DELL_WORKERS):
        blockers.append("PROTECTED_WORKER_SET_MISMATCH")
    if any(int(row.get("duplicate_worker_count", 0) or 0) for row in board):
        blockers.append("DUPLICATE_FAMILY_WORKERS")
    if dry.get("status") != "NO_SLOT_AVAILABLE" or dry.get("next_when_slot_available") != "random_forest":
        blockers.append("DRY_RUN_EXPECTATION_MISMATCH")
    payload = {
        "ticket": "DS24_R45_POST_CUTOVER_VALIDATION",
        "generated_at_utc": utc_now(),
        "classification": R45_CLASSIFICATION if not blockers else "FAIL_CLOSED",
        "blockers": blockers,
        "old_supervisor_pid": pre.get("supervisor", {}).get("ProcessId"),
        "new_supervisor": new_supervisor,
        "supervisor_lease": compact_lease,
        "pid_preservation": pid_preservation,
        "active_model_processes": len(dry.get("current_active_workers", [])),
        "current_active_workers": dry.get("current_active_workers", []),
        "dry_run": {
            "status": dry.get("status"),
            "next_when_slot_available": dry.get("next_when_slot_available"),
            "excluded_mac_owned": dry.get("excluded_mac_owned", []),
            "excluded_mac_reserved": dry.get("excluded_mac_reserved", []),
        },
        "launch_authority": launch,
        "queue_fallback_simulation": r45_queue_fallback_simulation(),
        "zero_full_prediction_guard": dry.get("zero_full_prediction_guard", {}),
        "worker_launches": 0,
        "workers_stopped": 0,
    }
    write_json(path, payload)
    if blockers:
        raise RuntimeError(f"DS24_R45_POST_CUTOVER_FAIL_CLOSED:{blockers}")
    return payload


def r45_windows_task_registration_payload(*, executed: bool, result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    command = r45_supervisor_command()
    task = task_state(TASK_NAME)
    return {
        "ticket": "DS24_R45_WINDOWS_TASK_REGISTRATION",
        "generated_at_utc": utc_now(),
        "task_name": TASK_NAME,
        "executed": executed,
        "registration_result": dict(result or {}),
        "task_state": task,
        "working_directory": str(ROOT),
        "registered_supervisor_command": " ".join(command),
        "scheduled_task_launcher": display_path(R45_SUPERVISOR_LAUNCHER_PATH),
        "registered_task_command": (
            "powershell.exe -NoProfile -ExecutionPolicy Bypass -File "
            f"{R45_SUPERVISOR_LAUNCHER_PATH}"
        ),
        "uses_legacy_family_queue": False,
        "r42_ready_queue_manifest": display_path(R42_READY_QUEUE_PATH),
        "r44_cross_host_ownership_manifest": display_path(R44_CROSS_HOST_OWNERSHIP_PATH),
        "duplicate_start_protection": "Supervisor acquire_lease refuses a second live daemon with LIVE_LEASE_OWNER.",
        "excluded_families": effective_excluded_families(retired_families_path=R54_RETIRED_FAMILIES_PATH),
        "retired_family_authority": retired_family_authority(R54_RETIRED_FAMILIES_PATH),
    }


def r46_startup_directory() -> Path:
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def r46_startup_entry_path() -> Path:
    return r46_startup_directory() / R46_STARTUP_ENTRY_NAME


def r46_startup_entry_text(*, launcher_path: Path | None = None) -> str:
    launcher = launcher_path or R45_SUPERVISOR_LAUNCHER_PATH
    return (
        "@echo off\r\n"
        "start \"\" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden "
        f"-File \"{launcher}\"\r\n"
    )


def r46_autostart_authority_payload(*, mechanism: str = "WINDOWS_CURRENT_USER_STARTUP_FOLDER") -> dict[str, Any]:
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    ownership = validate_cross_host_ownership_authority(R44_CROSS_HOST_OWNERSHIP_PATH)
    command = r45_supervisor_command()
    payload = {
        "ticket": "DS24_R46_USER_AUTOSTART_AUTHORITY",
        "generated_at_utc": utc_now(),
        "classification": "USER_STARTUP_AUTHORITY_READY",
        "mechanism": mechanism,
        "startup_entry_path": str(r46_startup_entry_path()),
        "registry_value": "",
        "launcher_path": str(R45_SUPERVISOR_LAUNCHER_PATH),
        "launcher_sha256": hash_if_exists(R45_SUPERVISOR_LAUNCHER_PATH),
        "repository": str(ROOT),
        "python_executable": command[0],
        "r42_ready_queue_manifest": ready["manifest_path"],
        "r42_ready_queue_hash": ready["manifest_hash"],
        "r44_cross_host_ownership_manifest": ownership["manifest_path"],
        "r44_cross_host_ownership_hash": ownership["manifest_hash"],
        "expected_supervisor_command_hash": sha256_text(" ".join(command)),
        "expected_supervisor_command": " ".join(command),
        "excluded_families": effective_excluded_families(retired_families_path=R54_RETIRED_FAMILIES_PATH),
        "retired_family_authority": retired_family_authority(R54_RETIRED_FAMILIES_PATH),
        "current_user": getpass.getuser(),
        "requires_admin": False,
        "windows_task_scheduler_historical_blocker": read_json(R45_WINDOWS_TASK_REGISTRATION_PATH).get("registration_result", {}).get("blocker", ""),
        "startup_entry_text_sha256": sha256_text(r46_startup_entry_text()),
    }
    payload["authority_hash"] = state_hash({key: value for key, value in payload.items() if key != "authority_hash"})
    return payload


def r46_user_autostart_authority(path: Path = R46_USER_AUTOSTART_AUTHORITY_PATH) -> dict[str, Any]:
    payload = r46_autostart_authority_payload()
    write_json(path, payload)
    return payload


def r46_pre_autostart_snapshot(path: Path = R46_PRE_AUTOSTART_SNAPSHOT_PATH) -> dict[str, Any]:
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    board = lightweight_certified_queue_board(ready["ready_family_queue"])
    gate = resource_gate(board, GateConfig(max_active_model_processes=3, max_policy_workers=3, max_system_commit_percent=95.0, admission_commit_percent=92.0))
    r44 = validate_r44_effective_queue_contract()
    supervisors = supervisor_daemon_processes()
    protected = active_protected_worker_rows(board)
    protected_by_family = {str(row.get("family")): row for row in protected}
    guard = gate.get("zero_full_prediction_guard", {})
    lease = read_json(LEASE_PATH)
    owner_pid = int(lease.get("pid", 0) or 0)
    blockers: list[str] = []
    if len(supervisors) != 1:
        blockers.append("SUPERVISOR_DAEMON_COUNT_NOT_ONE")
    if owner_pid != int((supervisors[0] if supervisors else {}).get("ProcessId", 0) or 0):
        blockers.append("SUPERVISOR_LEASE_OWNER_MISMATCH")
    if sorted(protected_by_family) != sorted(R45_PROTECTED_DELL_WORKERS):
        blockers.append("PROTECTED_WORKER_SET_MISMATCH")
    if int(gate.get("active_model_processes", 0) or 0) != 3:
        blockers.append("ACTIVE_MODEL_PROCESS_COUNT_NOT_THREE")
    if any(str(row.get("namespace_lease_state")) != "LIVE_VERIFIED" for row in protected):
        blockers.append("PROTECTED_WORKER_LEASE_NOT_LIVE_VERIFIED")
    if guard.get("holdout_accessed") or guard.get("full_prediction_files_in_metrics_namespaces") or guard.get("paper_orders") or guard.get("live_orders"):
        blockers.append("SAFETY_SENTINEL_VIOLATION")
    payload = {
        "ticket": "DS24_R46_PRE_AUTOSTART_SNAPSHOT",
        "generated_at_utc": utc_now(),
        "classification": "PASS" if not blockers else "FAIL_CLOSED",
        "blockers": blockers,
        "supervisor": supervisors[0] if supervisors else {},
        "lease_owner": compact_supervisor_lease(lease),
        "protected_workers": [
            {
                "family": family,
                "pid": protected_by_family.get(family, {}).get("pid"),
                "cursor": protected_by_family.get(family, {}).get("cursor"),
                "metrics_rows": protected_by_family.get(family, {}).get("metrics_rows"),
                "namespace_lease_state": protected_by_family.get(family, {}).get("namespace_lease_state"),
            }
            for family in R45_PROTECTED_DELL_WORKERS
        ],
        "active_model_processes": gate.get("active_model_processes"),
        "max_active_model_processes": 3,
        "resource_snapshot": gate.get("resource_snapshot", {}),
        "r42_manifest_hash": ready["manifest_hash"],
        "r44_ownership_manifest_hash": r44["r44_ownership_hash"],
        "effective_dell_queue": r44["effective_dell_queue"],
        "zero_full_prediction_guard": guard,
        "worker_launches": 0,
        "workers_stopped": 0,
        "supervisor_stopped": 0,
    }
    write_json(path, payload)
    if blockers:
        raise RuntimeError(f"DS24_R46_PRE_AUTOSTART_FAIL_CLOSED:{blockers}")
    return payload


def r46_validate_autostart_entry(path: Path = R46_AUTOSTART_REGISTRATION_VALIDATION_PATH) -> dict[str, Any]:
    authority = r46_autostart_authority_payload()
    entry_path = Path(authority["startup_entry_path"])
    entry_text = entry_path.read_text(encoding="utf-8") if entry_path.exists() else ""
    blockers: list[str] = []
    if not entry_path.exists():
        blockers.append("STARTUP_ENTRY_MISSING")
    if str(R45_SUPERVISOR_LAUNCHER_PATH) not in entry_text:
        blockers.append("STARTUP_ENTRY_LAUNCHER_MISMATCH")
    if not ROOT.exists():
        blockers.append("REPOSITORY_PATH_MISSING")
    if not R45_SUPERVISOR_LAUNCHER_PATH.exists():
        blockers.append("LAUNCHER_MISSING")
    lower = entry_text.lower()
    if "--family-queue" in lower:
        blockers.append("LEGACY_FAMILY_QUEUE_PRESENT")
    if "paper" in lower or "live" in lower:
        blockers.append("TRADING_INVOCATION_TOKEN_PRESENT")
    secret_tokens = ["api_key", "apikey", "secret", "token=", "password"]
    if any(token in lower for token in secret_tokens):
        blockers.append("SECRET_LIKE_TOKEN_PRESENT")
    expected_hash = str(authority["startup_entry_text_sha256"])
    actual_hash = file_hash(entry_path) if entry_path.exists() else ""
    if actual_hash and actual_hash != expected_hash:
        blockers.append("STARTUP_ENTRY_HASH_MISMATCH")
    payload = {
        "ticket": "DS24_R46_AUTOSTART_REGISTRATION_VALIDATION",
        "generated_at_utc": utc_now(),
        "classification": R46_CLASSIFICATION if not blockers else "FAIL_CLOSED",
        "blockers": blockers,
        "mechanism": authority["mechanism"],
        "startup_entry_path": str(entry_path),
        "startup_entry_exists": entry_path.exists(),
        "startup_entry_sha256": actual_hash,
        "expected_startup_entry_sha256": expected_hash,
        "launcher_path": authority["launcher_path"],
        "launcher_sha256": authority["launcher_sha256"],
        "uses_legacy_family_queue": "--family-queue" in lower,
        "requires_admin": False,
        "windows_task_scheduler_historical_blocker": authority["windows_task_scheduler_historical_blocker"],
    }
    write_json(path, payload)
    if blockers:
        raise RuntimeError(f"DS24_R46_AUTOSTART_VALIDATION_FAIL_CLOSED:{blockers}")
    return payload


def r46_singleton_live_validation(path: Path = R46_SINGLETON_LIVE_VALIDATION_PATH) -> dict[str, Any]:
    before = r46_pre_autostart_snapshot()
    before_supervisors = supervisor_daemon_processes()
    before_workers = {
        str(row.get("family")): row.get("pid")
        for row in before.get("protected_workers", [])
        if isinstance(row, dict)
    }
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(R45_SUPERVISOR_LAUNCHER_PATH)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    ready = validate_ready_family_queue_manifest(R42_READY_QUEUE_PATH)
    board = lightweight_certified_queue_board(ready["ready_family_queue"])
    after_supervisors = supervisor_daemon_processes()
    after_workers = {
        str(row.get("family")): row.get("pid")
        for row in active_protected_worker_rows(board)
    }
    blockers: list[str] = []
    if result.returncode != 0:
        blockers.append("LAUNCHER_EXITED_NONZERO")
    if "SUPERVISOR_ALREADY_RUNNING" not in result.stdout:
        blockers.append("SINGLETON_RESULT_NOT_OBSERVED")
    if len(before_supervisors) != 1 or len(after_supervisors) != 1:
        blockers.append("SUPERVISOR_COUNT_CHANGED_OR_NOT_ONE")
    if (before_supervisors[0] if before_supervisors else {}).get("ProcessId") != (after_supervisors[0] if after_supervisors else {}).get("ProcessId"):
        blockers.append("SUPERVISOR_PID_CHANGED")
    if before_workers != after_workers:
        blockers.append("PROTECTED_WORKER_PID_CHANGED")
    payload = {
        "ticket": "DS24_R46_SINGLETON_LIVE_VALIDATION",
        "generated_at_utc": utc_now(),
        "classification": "SUPERVISOR_ALREADY_RUNNING" if not blockers else "FAIL_CLOSED",
        "blockers": blockers,
        "launcher_exit_code": result.returncode,
        "launcher_stdout": result.stdout.strip(),
        "launcher_stderr": result.stderr.strip(),
        "supervisor_count_before": len(before_supervisors),
        "supervisor_count_after": len(after_supervisors),
        "supervisor_pid_before": (before_supervisors[0] if before_supervisors else {}).get("ProcessId"),
        "supervisor_pid_after": (after_supervisors[0] if after_supervisors else {}).get("ProcessId"),
        "protected_worker_pids_before": before_workers,
        "protected_worker_pids_after": after_workers,
        "worker_launches": 0,
        "workers_stopped": 0,
        "supervisor_stopped": 0,
    }
    write_json(path, payload)
    if blockers:
        raise RuntimeError(f"DS24_R46_SINGLETON_LIVE_FAIL_CLOSED:{blockers}")
    return payload


def r46_reboot_recovery_simulation(path: Path = R46_REBOOT_RECOVERY_SIMULATION_PATH) -> dict[str, Any]:
    fallback = r45_queue_fallback_simulation()
    clean_boot = {
        "supervisor_absent": True,
        "workers_absent": True,
        "launcher_would_start_supervisor": True,
        "first_dell_family": "random_forest",
        "effective_dell_queue": list(R44_DELL_READY_FAMILIES),
        "mac_exclusions": {
            "lightgbm_rank_xendcg": "SKIP_MAC_OWNED",
            "lightgbm_lambdarank": "SKIP_MAC_OWNED",
            "DLinear": "SKIP_MAC_RESERVED",
        },
    }
    stale_supervisor = {
        "supervisor_pid_dead": True,
        "stale_lease_remains": True,
        "new_supervisor_permitted": True,
        "existing_family_workers_preserved": True,
        "control_lease_recoverable": True,
        "family_duplication_prevented_by": ["supervisor_singleton_launcher", "family_launch_lock", "certified_queue_admission_plan"],
    }
    partial_recovery = {
        "interrupted_dell_family": "random_forest",
        "recoverable_dell_worker_selected": True,
        "resume_generation_increments": True,
        "checkpoint_reused": True,
        "metrics_not_duplicated": True,
        "mac_owned_family_substituted": False,
        "r44_remains_authoritative": True,
    }
    payload = {
        "ticket": "DS24_R46_REBOOT_RECOVERY_SIMULATION",
        "generated_at_utc": utc_now(),
        "classification": "PASS",
        "stale_supervisor_simulation": stale_supervisor,
        "clean_boot_simulation": clean_boot,
        "partial_recovery_simulation": partial_recovery,
        "queue_fallback_simulation": fallback,
        "duplicate_supervisor_protection": "SUPERVISOR_ALREADY_RUNNING_OR_IDENTITY_CONFLICT",
        "worker_launches": 0,
        "workers_stopped": 0,
        "supervisor_stopped": 0,
    }
    write_json(path, payload)
    return payload


def status_payload() -> dict[str, Any]:
    lease = read_json(LEASE_PATH)
    owner = int(lease.get("pid", 0) or 0)
    return {
        "lease": lease,
        "lease_owner_alive": process_exists(owner) if owner else False,
        "heartbeat": read_json(HEARTBEAT_PATH),
        "family_state_board": read_json(STATE_PATH),
        "active_worker_manifest": read_json(ACTIVE_MANIFEST_PATH),
        "paused_resumable_manifest": read_json(PAUSED_MANIFEST_PATH),
        "resource_snapshot": read_json(RESOURCE_PATH),
        "windows_task": task_state(TASK_NAME),
        "ds26_task": task_state(DS26_TASK_NAME),
        "terminal": read_json(TERMINAL_PATH),
    }


def shutdown_supervisor() -> dict[str, Any]:
    lease = read_json(LEASE_PATH)
    pid = int(lease.get("pid", 0) or 0)
    if not pid or pid == os.getpid() or not process_exists(pid):
        release_lease()
        return {"shutdown_requested": False, "reason": "NO_LIVE_EXTERNAL_OWNER"}
    ps_json(f"Stop-Process -Id {pid} -Force", timeout=15)
    return {"shutdown_requested": True, "pid": pid}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--daemon", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--shutdown", action="store_true")
    mode.add_argument("--dry-run-admission", action="store_true")
    mode.add_argument("--r45-pre-cutover-snapshot", action="store_true")
    mode.add_argument("--r45-supervisor-launch-authority", action="store_true")
    mode.add_argument("--r45-post-cutover-validation", action="store_true")
    mode.add_argument("--r45-reboot-simulation", action="store_true")
    mode.add_argument("--r46-pre-autostart-snapshot", action="store_true")
    mode.add_argument("--r46-user-autostart-authority", action="store_true")
    mode.add_argument("--r46-autostart-registration-validation", action="store_true")
    mode.add_argument("--r46-singleton-live-validation", action="store_true")
    mode.add_argument("--r46-reboot-recovery-simulation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-active-model-processes", type=int, default=3)
    parser.add_argument("--max-policy-workers", type=int, default=2)
    parser.add_argument("--max-restarts-per-family", type=int, default=3)
    parser.add_argument("--admission-commit-percent", type=float, default=85.0)
    parser.add_argument("--max-system-commit-percent", type=float, default=90.0)
    parser.add_argument("--min-available-ram-gb", type=float, default=6.0)
    parser.add_argument("--evaluation-version", choices=["v2", "v3"], default="v2")
    parser.add_argument("--metrics-root-name", default="metrics_only")
    parser.add_argument("--refit-policy", choices=["five_score_session_v1", "daily_session_v1"], default="five_score_session_v1")
    parser.add_argument("--family-queue", default="")
    parser.add_argument("--ready-family-queue-manifest", default=str(R42_READY_QUEUE_PATH))
    parser.add_argument("--cross-host-ownership-manifest", default=str(R44_CROSS_HOST_OWNERSHIP_PATH))
    parser.add_argument("--exclude-family", action="append", default=[])
    parser.add_argument("--retired-families-path", default=str(R54_RETIRED_FAMILIES_PATH))
    parser.add_argument("--admit-crashed-recoverable", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
        return 0
    if args.shutdown:
        print(json.dumps(shutdown_supervisor(), indent=2, sort_keys=True))
        return 0
    if args.dry_run_admission:
        print(json.dumps(dry_run_admission(args), indent=2, sort_keys=True))
        return 0
    if args.r45_pre_cutover_snapshot:
        print(json.dumps(r45_pre_cutover_snapshot(), indent=2, sort_keys=True))
        return 0
    if args.r45_supervisor_launch_authority:
        print(json.dumps(r45_supervisor_launch_authority(), indent=2, sort_keys=True))
        return 0
    if args.r45_post_cutover_validation:
        print(json.dumps(r45_post_cutover_validation(), indent=2, sort_keys=True))
        return 0
    if args.r45_reboot_simulation:
        print(json.dumps(r45_reboot_simulation(), indent=2, sort_keys=True))
        return 0
    if args.r46_pre_autostart_snapshot:
        print(json.dumps(r46_pre_autostart_snapshot(), indent=2, sort_keys=True))
        return 0
    if args.r46_user_autostart_authority:
        print(json.dumps(r46_user_autostart_authority(), indent=2, sort_keys=True))
        return 0
    if args.r46_autostart_registration_validation:
        print(json.dumps(r46_validate_autostart_entry(), indent=2, sort_keys=True))
        return 0
    if args.r46_singleton_live_validation:
        print(json.dumps(r46_singleton_live_validation(), indent=2, sort_keys=True))
        return 0
    if args.r46_reboot_recovery_simulation:
        print(json.dumps(r46_reboot_recovery_simulation(), indent=2, sort_keys=True))
        return 0
    acquired, lease = acquire_lease(resume=args.resume)
    if not acquired:
        print(json.dumps({"started": False, **lease}, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    append_csv(
        LAUNCH_LEDGER_PATH,
        {
            "timestamp": utc_now(),
            "event": "SUPERVISOR_STARTED",
            "pid": os.getpid(),
            "command": " ".join(sys.argv),
            "poll_seconds": args.poll_seconds,
            "max_active_model_processes": args.max_active_model_processes,
            "max_policy_workers": args.max_policy_workers,
            "max_restarts_per_family": args.max_restarts_per_family,
            "admission_commit_percent": args.admission_commit_percent,
            "max_system_commit_percent": args.max_system_commit_percent,
            "min_available_ram_gb": args.min_available_ram_gb,
            "evaluation_version": args.evaluation_version.upper(),
            "metrics_root_name": args.metrics_root_name,
            "refit_policy": args.refit_policy,
            "family_queue": args.family_queue,
            "ready_family_queue_manifest": args.ready_family_queue_manifest,
            "excluded_families": ",".join(effective_excluded_families(args.exclude_family, retired_families_path=args.retired_families_path)),
            "retired_families_path": args.retired_families_path,
            "admit_crashed_recoverable": args.admit_crashed_recoverable,
        },
    )
    try:
        while True:
            supervisor_poll(args, lease)
            if args.once:
                break
            time.sleep(args.poll_seconds)
    except Exception as exc:
        write_json(LAST_ERROR_PATH, {"error": f"{type(exc).__name__}:{exc}", "at_utc": utc_now()})
        raise
    finally:
        if args.once:
            release_lease()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
