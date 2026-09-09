from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.canonical_prequential_engine import load_predictor_manifest, stable_hash
from core.research.ml.ds24.comparable_policy import POLICY_ID, policy_hash
from core.research.ml.ds24.windows_safe_io import write_json_atomic
from core.research.ml.ds24_metrics_only_evaluator import (
    EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
    RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
    extended_performance_metrics_contract_hash,
    resolved_performance_contract_v3_hash,
)
from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


STAGE = ROOT / (
    "docs/dream_system/components/DS-24_independent_five_minute_selector/"
    "stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
)
POLICY_ROOT = STAGE / "r7_r14_policy_workers"
PREDICTOR_MANIFEST = ROOT / (
    "docs/dream_system/components/DS-24_independent_five_minute_selector/"
    "stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json"
)
OUTPUT_JSON = STAGE / "R41_full_family_readiness_matrix.json"
OUTPUT_CSV = STAGE / "R41_full_family_readiness_matrix.csv"
REPAIR_QUEUE_JSON = STAGE / "R41_readiness_repair_queue.json"
READY_QUEUE_JSON = STAGE / "R41_ready_family_queue.json"

UNFINISHED_FAMILIES = (
    "elastic_net",
    "mlp",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
    "DLinear",
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
    "Temporal Fusion Transformer",
)
COMPLETED_FAMILIES = (
    "ridge_policy_v1_control",
    "pca_ridge_policy_v1_control",
    "spline_additive_ridge",
    "rff_ridge",
    "huber",
    "Exact",
)
RUNNING_EXPECTED = {"mlp", "extra_trees", "gradient_boosting"}
STATE_VALUES = {
    "READY_TO_LAUNCH",
    "RUNNING_VERIFIED",
    "REPAIR_REQUIRED",
    "V3_REPLAY_REQUIRED",
    "FORWARD_METRICS_CONTRACT_REQUIRED",
    "CONFIGURATION_AUTHORITY_REQUIRED",
    "IMPLEMENTATION_BLOCKED",
    "DEPENDENCY_BLOCKED",
    "RESOURCE_BLOCKED_ONLY",
}
CSV_COLUMNS = (
    "family",
    "state",
    "implementation",
    "scientific_contract",
    "v3_forward_metrics",
    "checkpoint",
    "namespace",
    "stderr",
    "bounded_replay",
    "resource_profile",
    "holdout_guard",
    "full_prediction_guard",
    "current_cursor",
    "metrics_rows",
    "next_action",
    "blocker",
    "evidence_paths",
)
DISPLAY_TO_SLUG = {
    "DLinear": "dlinear",
    "PatchTST": "patchtst",
    "Transformer": "transformer",
    "iTransformer": "itransformer",
    "Momentum Transformer": "momentum_transformer",
    "Market Context Encoder": "market_context_encoder",
    "Temporal Fusion Transformer": "temporal_fusion_transformer",
}
SLUG_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_SLUG.items()}


@dataclass(frozen=True)
class EvidenceFile:
    path: Path
    hash: str


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_if_exists(path: Path) -> str:
    return file_hash(path) if path.exists() and path.is_file() else ""


def evidence_entry(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = f"#{hash_if_exists(path)[:16]}" if path.is_file() else ""
    return f"{display_path(path)}{suffix}"


def family_slug(family: str) -> str:
    return DISPLAY_TO_SLUG.get(family, re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_"))


def process_rows() -> list[dict[str, Any]]:
    try:
        rows = supervisor.python_processes()
        if rows:
            return rows
    except Exception:
        pass
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match 'ds24_p8_r14_e3g_c2_r7|ds24_v3_sequence_policy_worker|ds24_v3_lightgbm_ranking_policy_worker' } | "
        "Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    text = result.stdout.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def family_from_command(command: str) -> str:
    normalized = " ".join(command.replace("/", "\\").split()).lower()
    for family in UNFINISHED_FAMILIES:
        lower = family.lower()
        if f"--family {lower}" in normalized or f"--family={lower}" in normalized:
            return family
        slug = family_slug(family)
        if f"--family {slug}" in normalized or f"--family={slug}" in normalized:
            return family
    return ""


def active_processes_by_family() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {family: [] for family in UNFINISHED_FAMILIES}
    for row in process_rows():
        command = str(row.get("CommandLine") or "")
        family = family_from_command(command)
        if family:
            out.setdefault(family, []).append(row)
    return out


def recent_stderr_files(family: str) -> list[Path]:
    root = POLICY_ROOT / family
    if not root.exists():
        return []
    files = [path for path in root.glob("stderr*.log") if path.is_file()]
    files.extend(path for path in root.glob("stderr*.log.gz") if path.is_file())
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def tail_file(path: Path, limit: int = 4000) -> str:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.read()[-limit:]
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max(limit * 4, 8192)), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")[-limit:]
    except Exception as exc:
        return f"STDERR_READ_FAILED:{type(exc).__name__}:{exc}"


def stderr_evidence(family: str, *, active: bool) -> dict[str, Any]:
    first_warning: dict[str, Any] | None = None
    for path in recent_stderr_files(family)[:16]:
        if path.stat().st_size == 0:
            continue
        text = tail_file(path)
        fatal = "Traceback (most recent call last)" in text or "RuntimeError:" in text or "Fatal" in text
        if fatal or text:
            final = text.strip().splitlines()[-1] if text.strip() else ""
            state = "FATAL_REPAIR_REQUIRED" if fatal else "WARNINGS_NON_FATAL"
            evidence = {
                "state": state,
                "path": display_path(path),
                "hash": file_hash(path),
                "final_exception": final,
                "tail_hash": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
                "fatal": fatal,
            }
            if active:
                return evidence
            if fatal:
                return evidence
            if first_warning is None:
                first_warning = evidence
    if first_warning is not None:
        return first_warning
    return {"state": "CLEAN", "path": "", "hash": "", "final_exception": "", "tail_hash": "", "fatal": False}


def latest_metrics_root(family: str) -> Path:
    root = POLICY_ROOT / family
    progress = read_json(root / "progress.json")
    telemetry = read_json(root / "initialization_telemetry.json")
    inventory = read_json(root / "worker_inventory.json")
    name = str(progress.get("metrics_root_name") or telemetry.get("metrics_root_name") or inventory.get("metrics_root_name") or "")
    if name:
        return root / name
    candidates = sorted(root.glob("metrics_only_v3*"), key=lambda path: path.stat().st_mtime, reverse=True) if root.exists() else []
    return candidates[0] if candidates else root / supervisor.default_metrics_root_name(family, "metrics_only_v3")


def metrics_checkpoint(family: str) -> dict[str, Any]:
    root = latest_metrics_root(family)
    checkpoint = read_json(root / "resolved_performance_checkpoint_v3.json")
    basic = read_json(root / "checkpoint.json")
    return {
        "metrics_root": root,
        "checkpoint": checkpoint,
        "basic_checkpoint": basic,
        "rows": int(checkpoint.get("resolved_performance_rows", basic.get("metric_rows", 0)) or 0),
        "topn_rows": int(read_json(root / "decision_trace_v3_manifest.json").get("row_count", 0) or basic.get("decision_rows", 0) or 0),
        "cursor": checkpoint.get("last_resolution_timestamp") or checkpoint.get("first_uncommitted_timestamp") or basic.get("last_completed_T") or "",
        "duplicate_count": int(checkpoint.get("duplicate_count", 0) or 0),
        "full_prediction_files": int(checkpoint.get("permanent_full_prediction_files_written", basic.get("full_prediction_files_written", 0)) or 0),
    }


def namespace_state(family: str, processes: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    root = latest_metrics_root(family)
    lease = read_json(root / supervisor.NAMESPACE_WRITER_LEASE_NAME)
    pid = int(lease.get("pid", 0) or 0)
    live_pids = {int(row.get("ProcessId", 0) or 0) for row in processes.get(family, [])}
    if not lease:
        state = "ABSENT"
    elif pid in live_pids:
        state = "LIVE_VERIFIED"
    else:
        state = "STALE_RECOVERABLE"
    return {
        "state": state,
        "pid": pid,
        "generation": int(lease.get("lease_generation", 0) or 0),
        "resume_generation": lease.get("resume_generation", ""),
        "cursor": lease.get("cursor", ""),
        "phase": lease.get("phase", ""),
        "path": display_path(root / supervisor.NAMESPACE_WRITER_LEASE_NAME) if lease else "",
    }


def load_predictors() -> dict[str, Any]:
    try:
        predictors = load_predictor_manifest(PREDICTOR_MANIFEST)
        predictor_count = int(predictors.predictor_count)
        predictor_hash = str(predictors.manifest_hash)
    except Exception:
        predictor_count = 0
        predictor_hash = hash_if_exists(PREDICTOR_MANIFEST)
    manifest = read_json(PREDICTOR_MANIFEST)
    return {
        "path": display_path(PREDICTOR_MANIFEST),
        "hash": predictor_hash,
        "file_hash": hash_if_exists(PREDICTOR_MANIFEST),
        "count": predictor_count,
        "manifest_keys": sorted(manifest)[:12],
    }


def readiness_manifests() -> dict[str, dict[str, Any]]:
    manifests = [
        STAGE / "R7_LOW_USAGE_CLASSICAL_FAMILY_READINESS.json",
        STAGE / "R7_LOW_USAGE_ELASTIC_NET_READINESS.json",
        STAGE / "R7_LOW_USAGE_LIGHTGBM_RANKING_READINESS.json",
        STAGE / "R7_LOW_USAGE_V3_SEQUENCE_WORKER_READINESS.json",
        STAGE / "R7_LOW_USAGE_TRANSFORMER_ITRANSFORMER_READINESS.json",
        STAGE / "R7_LOW_USAGE_SPECIALISED_SEQUENCE_READINESS.json",
        STAGE / "R7_R40_05_execution_registry.json",
        STAGE / "R7_R40_06_tft_certification.json",
    ]
    out: dict[str, dict[str, Any]] = {}
    for path in manifests:
        payload = read_json(path)
        if not payload:
            continue
        out[path.name] = {"payload": payload, "path": path, "hash": file_hash(path)}
    return out


def records_by_family(manifests: Mapping[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests.values():
        payload = manifest["payload"]
        rows = payload.get("families", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                family = str(row.get("family") or "")
                display = SLUG_TO_DISPLAY.get(family, family)
                out.setdefault(display, []).append({"record": row, "manifest_path": manifest["path"], "manifest_hash": manifest["hash"]})
        family = str(payload.get("family") or "")
        if family:
            display = SLUG_TO_DISPLAY.get(family, family)
            out.setdefault(display, []).append({"record": payload.get("evidence", payload), "manifest_path": manifest["path"], "manifest_hash": manifest["hash"]})
    return out


def forward_metrics_gate(family: str) -> dict[str, Any]:
    decision = supervisor.forward_metrics_contract_admission_decision(
        family,
        evaluation_version="v3",
        metrics_root_name=supervisor.default_metrics_root_name(family, "metrics_only_v3"),
    )
    missing = list(decision.get("missing_requirements", []))
    return {
        "state": "PASS" if decision.get("admitted") else "FORWARD_METRICS_CONTRACT_ADMISSION_BLOCKED",
        "contract_id": EXTENDED_PERFORMANCE_METRICS_CONTRACT_ID,
        "contract_hash": extended_performance_metrics_contract_hash(),
        "base_contract_id": RESOLVED_PERFORMANCE_CONTRACT_V3_ID,
        "base_contract_hash": resolved_performance_contract_v3_hash(),
        "missing_requirements": missing,
        "capability_decision": decision.get("capability_decision", ""),
        "capability_path": decision.get("capability_path", ""),
    }


def implementation_gate(family: str, registry_rows: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    row = supervisor.execution_registry_row(family)
    script = ROOT / str(row.get("worker_script", "")).replace("/", "\\")
    records = registry_rows.get(family, [])
    imported = False
    error = ""
    try:
        if row.get("worker_kind") == "TABULAR":
            from scripts.local.ds24_p8_r14_e3g_c2_r7_r14_policy_worker import FAMILY_MAP

            imported = family in FAMILY_MAP
        elif row.get("worker_kind") == "LIGHTGBM_RANKING":
            imported = script.exists()
        elif row.get("worker_kind") == "PYTORCH_SEQUENCE":
            from scripts.local.ds24_v3_sequence_policy_worker import assert_supported_family

            assert_supported_family(family)
            imported = True
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    return {
        "state": "PASS" if script.exists() and imported else "FAIL",
        "worker_kind": row.get("worker_kind", ""),
        "worker_script": row.get("worker_script", ""),
        "worker_script_exists": bool(script.exists()),
        "launch_enabled": bool(row.get("launch_enabled")),
        "family_resolves": imported,
        "route_error": error,
        "configuration_authority_present": bool(records) or family in RUNNING_EXPECTED or family == "random_forest",
    }


def scientific_gate(family: str, predictor: Mapping[str, Any], registry_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = registry_rows.get(family, [])
    source = rows[0] if rows else {}
    record = source.get("record", {}) if isinstance(source, dict) else {}
    config = record.get("registered_configuration", {}) if isinstance(record, dict) else {}
    ticket63 = config.get("ticket63_family_config", {}) if isinstance(config, dict) else {}
    return {
        "state": "PASS" if predictor["count"] > 0 and (family not in DISPLAY_TO_SLUG or bool(record)) else "FAIL",
        "predictor_manifest": predictor["path"],
        "predictor_manifest_hash": predictor["hash"],
        "feature_count": predictor["count"],
        "target": ticket63.get("target_identity") or "forward_return_60m__decision_5m",
        "decision_cadence": supervisor.DECISION_CADENCE_ID,
        "refit_policy": "daily_session_v1",
        "training_window_semantics": "daily session refit; matured targets only before refit cutoff",
        "pit_availability": True,
        "development_boundary": ticket63.get("development_endpoint", "preholdout DS24 authority"),
        "outer_holdout_inaccessible": True,
        "policy_id": POLICY_ID,
        "policy_hash": policy_hash(),
    }


def bounded_replay_gate(family: str, registry_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = registry_rows.get(family, [])
    has_v3 = False
    path = ""
    for source in rows:
        record = source.get("record", {})
        state = str(record.get("state", ""))
        runtime = record.get("runtime_guards", record.get("v3_metrics_only", {}))
        if state in {"V3_CERTIFIED_READY", "V3_SEQUENCE_WORKER_CERTIFIED_READY"} and runtime:
            has_v3 = True
            path = display_path(source["manifest_path"])
            break
    if family in RUNNING_EXPECTED or family == "random_forest":
        metrics = metrics_checkpoint(family)
        return {
            "state": "PASS" if metrics["rows"] > 0 else "V3_REPLAY_REQUIRED",
            "basis": "live tournament V3 metrics/checkpoint evidence" if metrics["rows"] > 0 else "no live replay metrics",
            "evidence_path": display_path(metrics["metrics_root"]),
        }
    if family in {"DLinear", "PatchTST", "Transformer", "iTransformer", "Momentum Transformer", "Market Context Encoder"}:
        return {
            "state": "V3_REPLAY_REQUIRED",
            "basis": "synthetic worker proof exists, but no genuine bounded tournament replay was found",
            "evidence_path": path,
        }
    return {
        "state": "PASS" if has_v3 else "V3_REPLAY_REQUIRED",
        "basis": "readiness manifest evidence" if has_v3 else "no bounded V3 replay evidence found",
        "evidence_path": path,
    }


def resource_profile(family: str, registry_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    for source in registry_rows.get(family, []):
        record = source.get("record", {})
        estimate = record.get("resource_estimate") or record.get("estimated_ram_disk_refit_cost") or record.get("estimated_ram")
        preflight = record.get("resource_preflight", {})
        if estimate:
            return {
                "thread_requirement": 1,
                "cache_budget_gib": 1,
                "classification": "heavy" if family not in {"elastic_net", "mlp"} else "light",
                "estimate": estimate,
                "preflight": preflight,
                "required_free_disk_floor_gib": preflight.get("preflight_required_headroom_gib", 8),
                "supervisor_compatible": family not in {"Temporal Fusion Transformer"},
            }
    return {
        "thread_requirement": 1,
        "cache_budget_gib": 1,
        "classification": "heavy" if family not in {"elastic_net", "mlp"} else "light",
        "estimate": "unknown",
        "preflight": {},
        "required_free_disk_floor_gib": 8,
        "supervisor_compatible": True,
    }


def safety_gate() -> dict[str, Any]:
    terminal = read_json(STAGE / "R7_R27_14_terminal_validation.json")
    guard = terminal.get("resource_gate", {}).get("zero_full_prediction_guard", {})
    status = read_json(STAGE / "r7_r25_metrics_only_migration" / "status.json")
    return {
        "holdout_accessed": bool(terminal.get("holdout_accessed", False)),
        "full_prediction_files_in_metrics_namespaces": int(
            guard.get("full_prediction_files_in_metrics_namespaces", status.get("full_prediction_files_in_metrics_namespaces", 0)) or 0
        ),
        "paper_orders": int(terminal.get("paper_orders", guard.get("paper_orders", 0)) or 0),
        "live_orders": int(terminal.get("live_orders", guard.get("live_orders", 0)) or 0),
        "evidence_path": display_path(STAGE / "R7_R27_14_terminal_validation.json"),
    }


def classify_state(
    family: str,
    implementation: Mapping[str, Any],
    forward: Mapping[str, Any],
    namespace: Mapping[str, Any],
    stderr: Mapping[str, Any],
    replay: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> tuple[str, str, str]:
    if family in RUNNING_EXPECTED:
        if namespace["state"] == "LIVE_VERIFIED" and not stderr["fatal"] and not any(
            [safety["holdout_accessed"], safety["full_prediction_files_in_metrics_namespaces"], safety["paper_orders"], safety["live_orders"]]
        ):
            return "RUNNING_VERIFIED", "continue read-only monitoring", ""
        return "REPAIR_REQUIRED", "forensic repair before relying on running worker", "running worker failed lease/stderr/safety verification"
    if family == "Temporal Fusion Transformer":
        return (
            "CONFIGURATION_AUTHORITY_REQUIRED",
            "bind accepted TFT tournament authority or repair TFT family adapter",
            "R40 records missing accepted full TFT static/recurrent tournament authority",
        )
    if not implementation["worker_script_exists"] or not implementation["family_resolves"]:
        return "IMPLEMENTATION_BLOCKED", "build or bind production worker route", str(implementation.get("route_error") or "worker route missing")
    if forward["state"] == "FORWARD_METRICS_CONTRACT_ADMISSION_BLOCKED":
        return (
            "FORWARD_METRICS_CONTRACT_REQUIRED",
            "publish extended forward-metrics capability sidecar",
            ",".join(forward["missing_requirements"]),
        )
    if stderr["fatal"] and family == "random_forest":
        final = str(stderr.get("final_exception") or "")
        if "STORAGE_EMERGENCY_HOLD_ACTIVE" not in final and namespace["state"] != "STALE_RECOVERABLE":
            return "REPAIR_REQUIRED", "repair Random Forest crash cause before readmission", final
    if replay["state"] == "V3_REPLAY_REQUIRED":
        return "V3_REPLAY_REQUIRED", "run bounded V3 tournament replay certification", str(replay.get("basis", ""))
    if namespace["state"] in {"ABSENT", "STALE_RECOVERABLE"}:
        return "READY_TO_LAUNCH", "admit when supervisor slot and resource gates open", ""
    return "REPAIR_REQUIRED", "resolve namespace state before launch", str(namespace["state"])


def family_record(
    family: str,
    *,
    processes: Mapping[str, list[dict[str, Any]]],
    predictor: Mapping[str, Any],
    registry_rows: Mapping[str, list[dict[str, Any]]],
    safety: Mapping[str, Any],
) -> dict[str, Any]:
    active = bool(processes.get(family))
    implementation = implementation_gate(family, registry_rows)
    scientific = scientific_gate(family, predictor, registry_rows)
    forward = forward_metrics_gate(family) if family not in COMPLETED_FAMILIES else {"state": "NOT_REOPENED"}
    namespace = namespace_state(family, processes) if family != "Exact" else {"state": "COMPLETE", "pid": 0, "cursor": ""}
    stderr = stderr_evidence(family, active=active)
    replay = bounded_replay_gate(family, registry_rows)
    metrics = metrics_checkpoint(family) if family != "Exact" else {"rows": 0, "cursor": "", "metrics_root": STAGE / "r7_r25_metrics_only" / "exact_ridge_pca", "full_prediction_files": 0}
    resource = resource_profile(family, registry_rows)
    state, next_action, blocker = classify_state(family, implementation, forward, namespace, stderr, replay, safety)
    evidence_paths = [
        evidence_entry(PREDICTOR_MANIFEST),
        evidence_entry(POLICY_ROOT / family / "progress.json"),
        evidence_entry(latest_metrics_root(family) / "resolved_performance_checkpoint_v3.json"),
        evidence_entry(Path(stderr["path"])) if stderr.get("path") else "",
        str(forward.get("capability_path") or ""),
        str(replay.get("evidence_path") or ""),
    ]
    return {
        "family": family,
        "state": state,
        "implementation": implementation,
        "scientific_contract": scientific,
        "v3_forward_metrics": forward,
        "checkpoint": {
            "state": "READABLE" if (POLICY_ROOT / family / "progress.json").exists() or metrics["rows"] > 0 else "ABSENT",
            "current_cursor": metrics["cursor"],
            "metrics_rows": metrics["rows"],
            "topn_rows": metrics["topn_rows"],
            "duplicate_count": metrics["duplicate_count"],
            "full_prediction_files": metrics["full_prediction_files"],
        },
        "namespace": namespace,
        "stderr": stderr,
        "bounded_replay": replay,
        "resource_profile": resource,
        "holdout_guard": "PASS" if not safety["holdout_accessed"] else "FAIL",
        "full_prediction_guard": "PASS" if safety["full_prediction_files_in_metrics_namespaces"] == 0 else "FAIL",
        "current_cursor": metrics["cursor"],
        "metrics_rows": metrics["rows"],
        "next_action": next_action,
        "blocker": blocker,
        "evidence_paths": [item for item in evidence_paths if item],
    }


def completed_record(family: str, predictor: Mapping[str, Any], safety: Mapping[str, Any]) -> dict[str, Any]:
    record = family_record(
        family,
        processes={family: []},
        predictor=predictor,
        registry_rows={},
        safety=safety,
    ) if family != "Exact" else {
        "family": "Exact",
        "state": "COMPLETED_NOT_REOPENED",
        "implementation": "protected exact worker completed; not reopened",
        "scientific_contract": {"state": "NOT_REOPENED", "predictor_manifest_hash": predictor["hash"]},
        "v3_forward_metrics": {"state": "NOT_REOPENED"},
        "checkpoint": {"state": "COMPLETE"},
        "namespace": {"state": "COMPLETE"},
        "stderr": {"state": "CLEAN"},
        "bounded_replay": {"state": "COMPLETE"},
        "resource_profile": {"classification": "completed"},
        "holdout_guard": "PASS" if not safety["holdout_accessed"] else "FAIL",
        "full_prediction_guard": "PASS" if safety["full_prediction_files_in_metrics_namespaces"] == 0 else "FAIL",
        "current_cursor": "",
        "metrics_rows": 0,
        "next_action": "do not reopen completed family",
        "blocker": "",
        "evidence_paths": [evidence_entry(PREDICTOR_MANIFEST)],
    }
    record["state"] = "COMPLETED_NOT_REOPENED"
    record["next_action"] = "do not reopen completed family"
    record["blocker"] = ""
    return record


def repair_priority(record: Mapping[str, Any]) -> int:
    state = str(record["state"])
    if record["family"] == "random_forest":
        return 10
    if state == "FORWARD_METRICS_CONTRACT_REQUIRED":
        return 20
    if state == "V3_REPLAY_REQUIRED":
        return 30
    if state == "IMPLEMENTATION_BLOCKED":
        return 40
    if state == "CONFIGURATION_AUTHORITY_REQUIRED":
        return 50
    if state == "REPAIR_REQUIRED":
        return 25
    return 99


def build_matrix() -> dict[str, Any]:
    manifests = readiness_manifests()
    registry_rows = records_by_family(manifests)
    predictor = load_predictors()
    processes = active_processes_by_family()
    safety = safety_gate()
    unfinished = [
        family_record(family, processes=processes, predictor=predictor, registry_rows=registry_rows, safety=safety)
        for family in UNFINISHED_FAMILIES
    ]
    completed = [completed_record(family, predictor, safety) for family in COMPLETED_FAMILIES]
    bad_states = [row["state"] for row in unfinished if row["state"] not in STATE_VALUES]
    if bad_states:
        raise RuntimeError(f"R41_INVALID_PRIMARY_STATE:{bad_states}")
    ready = [row["family"] for row in unfinished if row["state"] == "READY_TO_LAUNCH"]
    repair_queue = [
        {
            "family": row["family"],
            "state": row["state"],
            "priority": repair_priority(row),
            "next_action": row["next_action"],
            "blocker": row["blocker"],
            "evidence_paths": row["evidence_paths"],
        }
        for row in sorted(unfinished, key=repair_priority)
        if row["state"] not in {"READY_TO_LAUNCH", "RUNNING_VERIFIED", "RESOURCE_BLOCKED_ONLY"}
    ]
    return {
        "ticket": "DS24_R41_FULL_FAMILY_PRE_LAUNCH_READINESS_AUDIT",
        "generated_at_utc": utc_now(),
        "read_only": True,
        "worker_launches": 0,
        "workers_stopped": 0,
        "supervisor_modified": False,
        "stage_root": display_path(STAGE),
        "predictor_manifest": predictor,
        "safety": safety,
        "running_verified_families": [row["family"] for row in unfinished if row["state"] == "RUNNING_VERIFIED"],
        "ready_to_launch_families": ready,
        "recommended_next_three_family_order": ready[:3],
        "recommended_supervisor_queue_when_slot_opens": ready,
        "families": unfinished + completed,
        "repair_queue": repair_queue,
        "source_manifest_hashes": {name: item["hash"] for name, item in manifests.items()},
    }


def csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


def publish(matrix: Mapping[str, Any]) -> None:
    write_json_atomic(OUTPUT_JSON, matrix, advisory=True)
    rows = []
    for row in matrix["families"]:
        rows.append({column: csv_cell(row.get(column, "")) for column in CSV_COLUMNS})
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(
        REPAIR_QUEUE_JSON,
        {
            "ticket": matrix["ticket"],
            "generated_at_utc": matrix["generated_at_utc"],
            "repair_queue": matrix["repair_queue"],
        },
        advisory=True,
    )
    write_json_atomic(
        READY_QUEUE_JSON,
        {
            "ticket": matrix["ticket"],
            "generated_at_utc": matrix["generated_at_utc"],
            "ready_family_queue": matrix["ready_to_launch_families"],
            "recommended_supervisor_queue_when_slot_opens": matrix["recommended_supervisor_queue_when_slot_opens"],
            "not_applied_to_live_supervisor": True,
        },
        advisory=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS24 R41 full-family readiness audit")
    parser.add_argument("--no-publish", action="store_true", help="Build the matrix but do not write R41 artifacts.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = build_matrix()
    if not args.no_publish:
        publish(matrix)
    print(
        json.dumps(
            {
                "matrix_path": display_path(OUTPUT_JSON) if not args.no_publish else "",
                "running_verified": matrix["running_verified_families"],
                "ready_to_launch": matrix["ready_to_launch_families"],
                "repair_queue_count": len(matrix["repair_queue"]),
                "safety": matrix["safety"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
