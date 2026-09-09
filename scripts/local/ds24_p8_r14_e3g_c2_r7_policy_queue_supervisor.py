from __future__ import annotations

import argparse
import csv
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
from typing import Any

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
        return False
    return False


def default_metrics_root_name(family: str, requested: str = "") -> str:
    if family in R40_V3_METRICS_ROOTS:
        return R40_V3_METRICS_ROOTS[family]
    if requested and requested not in {"metrics_only", "metrics_only_v3", "metrics_only_v3_r37_rff_retry"}:
        return requested
    return f"metrics_only_v3_r40_{family_slug(family)}"


def forward_metrics_capability_paths(family: str) -> list[Path]:
    script = worker_script_for_family(family)
    paths = [POLICY_ROOT / family / "forward_metrics_capability.json"]
    if script:
        stem = Path(script).stem
        paths.append(ROOT / "scripts" / "local" / f"{stem}.forward_metrics_capability.json")
        paths.append(ROOT / "scripts" / "local" / f"{family_slug(family)}.forward_metrics_capability.json")
    return paths


def family_forward_metrics_capability_evidence(family: str, *, metrics_root_name: str = "metrics_only_v3") -> dict[str, Any]:
    for path in forward_metrics_capability_paths(family):
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
    evidence = capability if capability is not None else family_forward_metrics_capability_evidence(family, metrics_root_name=metrics_root_name)
    result = validate_extended_metrics_writer_capability(evidence, family=family)
    return {
        **base,
        "admitted": bool(result.get("admitted")),
        "capability_decision": str(result.get("classification", "")),
        "missing_requirements": list(result.get("missing_requirements", [])),
        "capability_path": str((evidence or {}).get("capability_path", "")) if isinstance(evidence, dict) else "",
    }


def execution_registry_row(family: str) -> dict[str, Any]:
    script = worker_script_for_family(family)
    forward_decision = forward_metrics_contract_admission_decision(family, evaluation_version="v3", metrics_root_name="metrics_only_v3")
    return {
        "family": family,
        "worker_kind": worker_kind_for_family(family),
        "worker_script": f"scripts/local/{script}" if script else "",
        "worker_script_exists": bool(script and (ROOT / "scripts" / "local" / script).exists()),
        "launch_enabled": launch_enabled_for_family(family),
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
                if not any(token in command_line.lower() for token in ["ds24_p8_r14_e3g_c2_r7", "ds26", "r20", "compactor"]):
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
    if POLICY_WORKER_SCRIPT.lower() not in normalized:
        return ""
    for family in ALL_FAMILIES:
        family_lower = family.lower()
        if f"--family {family_lower}" in normalized or f"--family={family_lower}" in normalized:
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
    terminal = r31_has_reached_registered_terminal(family, progress or metrics_checkpoint)
    terminal_exclusion = r31_terminal_exclusion_state(family, root=POLICY_ROOT)
    terminal_complete = bool(terminal["reached"] and met["metric_rows"] > 0 and met["topn_rows"] > 0 and inv_dupes == 0)
    live_ok = bool(live) and dup_workers == 0 and checkpoint_valid(family) and not fatal_effective
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
    elif terminal_exclusion == "COMPLETE":
        state = "COMPLETE"
    elif terminal_exclusion == "TERMINAL_VALIDATING":
        state = "TERMINAL_VALIDATING"
    elif fatal_effective and checkpoint_valid(family):
        state = "CRASHED_RECOVERABLE"
    elif fatal_effective:
        state = "CRASHED_BLOCKED"
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
        state = "V3_CERTIFICATION_REQUIRED"
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
        "worker_kind": worker_kind_for_family(family),
        "worker_script": execution_registry_row(family)["worker_script"],
        "launch_enabled": launch_enabled_for_family(family),
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


def active_running(board: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in board
        if row["state"] in {"RUNNING", "FAST_FORWARDING", "INITIALIZING", "STARTING"}
        and row.get("pid_alive", True)
    ]


def selected_queue(family_queue: str = "") -> list[str]:
    if family_queue:
        return [item.strip() for item in family_queue.split(",") if item.strip()]
    return R40_AUTOMATIC_QUEUE


def next_ready_family(board: list[dict[str, Any]], *, family_queue: str = "", admit_crashed_recoverable: bool = False) -> str:
    allowed = {"PAUSED_RESOURCE_GATE", "CERTIFIED_READY", "V3_CERTIFIED_READY"}
    if admit_crashed_recoverable:
        allowed.add("CRASHED_RECOVERABLE")
    rows_by_family = {row["family"]: row for row in board}
    for family in selected_queue(family_queue):
        row = rows_by_family.get(family, {})
        if row.get("pid_alive") or int(row.get("duplicate_worker_count", 0) or 0):
            continue
        if not launch_enabled_for_family(family):
            continue
        if row.get("state") in allowed:
            return family
    return ""


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


def validate_family_launch_slot(family: str) -> dict[str, Any]:
    live = family_processes().get(family, [])
    if live:
        pids = ",".join(str(int(row.get("ProcessId", 0) or 0)) for row in live)
        raise RuntimeError(f"DS24_R36_PRELAUNCH_DUPLICATE_FAMILY_REFUSED:{family}:live_pids={pids}")
    return {"family": family, "live_pids": [], "validated_at_utc": utc_now()}


def post_launch_family_guard(family: str, launched_pid: int) -> dict[str, Any]:
    time.sleep(0.5)
    live = family_processes().get(family, [])
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
) -> dict[str, Any]:
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
        validate_family_launch_slot(family)
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
    admit_crashed_recoverable: bool = False,
    evaluation_version: str = "v2",
    metrics_root_name: str = "metrics_only",
    refit_policy: str = "five_score_session_v1",
) -> dict[str, Any]:
    exact = exact_manifest()
    running = active_running(board)
    certification_queue = TREE_CERTIFICATION + RANKING_CERTIFICATION + SEQUENCE_CERTIFICATION
    next_family = next_ready_family(board, family_queue=family_queue, admit_crashed_recoverable=admit_crashed_recoverable)
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
        "family_queue": selected_queue(family_queue),
        "admit_crashed_recoverable": admit_crashed_recoverable,
        "evaluation_version": evaluation_version.upper(),
        "metrics_root_name": metrics_root_name,
        "refit_policy": refit_policy,
        "exact_worker": exact,
        "active_workers": [row["family"] for row in running],
        "completed_families": [row["family"] for row in board if row["state"] == "COMPLETE"],
        "terminal_validating_families": [row["family"] for row in board if row["state"] == "TERMINAL_VALIDATING"],
        "paused_resumable_families": [row["family"] for row in board if row["state"] == "PAUSED_RESOURCE_GATE"],
        "certified_ready_queue": [row["family"] for row in board if row["state"] in {"CERTIFIED_READY", "V3_CERTIFIED_READY"}],
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
    next_family = next_ready_family(board, family_queue=args.family_queue, admit_crashed_recoverable=args.admit_crashed_recoverable)
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
        admit_crashed_recoverable=args.admit_crashed_recoverable,
        evaluation_version=args.evaluation_version,
        metrics_root_name=args.metrics_root_name,
        refit_policy=args.refit_policy,
        last_admission=last_admission,
        last_launch=last_launch,
    )


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
