from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "config/operations/post_finaliser_job_ledger_v1.json"
VALID_STATUSES = {
    "COMPLETE", "ACTIVE", "READY", "WAITING_DEPENDENCY", "WAITING_RESOURCES",
    "BLOCKED", "FAILED_RETRYABLE", "FAILED_NONRETRYABLE", "DEFERRED", "NOT_STARTED",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def load_json(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, "MISSING_FILE"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"MALFORMED_FILE:{exc}"
    return (value, None) if isinstance(value, dict) else (None, "MALFORMED_FILE:ROOT_NOT_OBJECT")


def evaluate_readiness(
    ledger: Mapping[str, Any],
    *,
    progress: Mapping[str, Any] | None,
    archive_validation: Mapping[str, Any] | None,
    selector_state: Mapping[str, Any] | None,
    ops_4a: Mapping[str, Any] | None,
    component_readiness: Mapping[str, Any] | None = None,
    component_inventory: Mapping[str, Any] | None = None,
    finaliser_active: bool,
    free_memory_bytes: int,
    free_disk_bytes: int,
    input_errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    errors = dict(input_errors or {})
    policy = ledger["policy"]
    expected = int(policy["full_archive_partition_count"])
    statuses, blockers = {}, {}

    planned = _integer(progress, "planned_partitions", "planned_symbol_year_partitions")
    completed = _integer(progress, "completed_partitions")
    pending = _integer(progress, "pending_partitions", default=max(0, planned - completed))
    failed = _integer(progress, "failed_partitions")
    invalid = _integer(progress, "invalid_rows")
    conflicts = _integer(progress, "conflicting_duplicates", "conflicting_duplicate_count")
    temporary = _integer(progress, "temporary_files", "temporary_files_left_behind")
    finaliser_complete = (
        progress is not None and planned == expected and completed == expected
        and pending == failed == invalid == conflicts == temporary == 0
    )
    job1_blockers = []
    if errors.get("finaliser_progress"):
        job1_blockers.append(f"FINALISER_PROGRESS_{errors['finaliser_progress']}")
    if failed:
        job1_blockers.append(f"FAILED_PARTITIONS:{failed}")
    if invalid:
        job1_blockers.append(f"INVALID_ROWS:{invalid}")
    if conflicts:
        job1_blockers.append(f"CONFLICTING_DUPLICATES:{conflicts}")
    if temporary:
        job1_blockers.append(f"TEMPORARY_ARTIFACTS:{temporary}")
    if finaliser_complete:
        statuses["JOB-001"] = "COMPLETE"
    elif job1_blockers:
        statuses["JOB-001"] = "FAILED_RETRYABLE"
    elif finaliser_active:
        statuses["JOB-001"] = "ACTIVE"
    elif progress is None:
        statuses["JOB-001"] = "BLOCKED"
    else:
        statuses["JOB-001"] = "FAILED_RETRYABLE"
        job1_blockers.append(f"INCOMPLETE_ARCHIVE:{completed}/{planned or expected}")
    blockers["JOB-001"] = job1_blockers

    validation_count = _integer(archive_validation, "partition_count")
    validation_valid = bool(archive_validation and archive_validation.get("valid") is True)
    validation_invalid = _integer(archive_validation, "invalid_rows")
    validation_temp = len(archive_validation.get("temporary_files_left_behind") or []) if archive_validation else 0
    full_valid = validation_valid and validation_count == expected and validation_invalid == 0 and validation_temp == 0
    job2_blockers = []
    if errors.get("archive_validation"):
        job2_blockers.append(f"ARCHIVE_VALIDATION_{errors['archive_validation']}")
    if validation_count and validation_count != expected:
        job2_blockers.append(f"PARTIAL_OR_WRONG_PARTITION_COUNT:{validation_count}/{expected}")
    if archive_validation and not validation_valid:
        job2_blockers.append("ARCHIVE_VALIDATION_INVALID")
    if validation_invalid:
        job2_blockers.append(f"ARCHIVE_INVALID_ROWS:{validation_invalid}")
    if validation_temp:
        job2_blockers.append(f"ARCHIVE_TEMPORARY_FILES:{validation_temp}")
    if statuses["JOB-001"] != "COMPLETE":
        statuses["JOB-002"] = "WAITING_DEPENDENCY"
        job2_blockers.append("JOB-001_NOT_COMPLETE")
    elif full_valid:
        statuses["JOB-002"] = "COMPLETE"
    elif archive_validation is None:
        statuses["JOB-002"] = "READY"
    else:
        statuses["JOB-002"] = "BLOCKED" if job2_blockers else "READY"
    blockers["JOB-002"] = job2_blockers

    stage_map, selector_errors = _selector_stage_map(selector_state, ledger["selector_run_id"])
    job3_blockers = list(selector_errors)
    if errors.get("selector_run_state"):
        job3_blockers.append(f"SELECTOR_STATE_{errors['selector_run_state']}")
    if errors.get("ops_4a_readiness"):
        job3_blockers.append(f"OPS_4A_{errors['ops_4a_readiness']}")
    if statuses["JOB-001"] != "COMPLETE":
        job3_blockers.append("JOB-001_NOT_COMPLETE")
    if statuses["JOB-002"] != "COMPLETE":
        job3_blockers.append("JOB-002_NOT_COMPLETE")
    if finaliser_active:
        job3_blockers.append("FINALISER_PROCESS_ACTIVE")
    if not ops_4a or ops_4a.get("status") != "READY" or ops_4a.get("whole_table_to_pylist_used") is not False:
        job3_blockers.append("OPS_4A_NOT_READY")
    for stage in (1, 2, 3):
        if stage_map.get(stage) != "complete":
            job3_blockers.append(f"SELECTOR_STAGE_{stage}_NOT_COMPLETE")
    if stage_map.get(4) not in {"failed", "pending"}:
        job3_blockers.append(f"SELECTOR_STAGE_4_NOT_RETRYABLE:{stage_map.get(4)}")
    if free_memory_bytes < int(policy["minimum_free_memory_bytes"]):
        job3_blockers.append(f"FREE_MEMORY_BELOW_POLICY:{free_memory_bytes}")
    if free_disk_bytes < int(policy["minimum_free_disk_bytes"]):
        job3_blockers.append(f"FREE_DISK_BELOW_POLICY:{free_disk_bytes}")
    dependency_block = any(value.endswith("_NOT_COMPLETE") for value in job3_blockers)
    resource_block = any(value.startswith(("FINALISER_PROCESS_ACTIVE", "FREE_MEMORY_", "FREE_DISK_")) for value in job3_blockers)
    if not job3_blockers:
        statuses["JOB-003"] = "READY"
    elif dependency_block:
        statuses["JOB-003"] = "WAITING_DEPENDENCY"
    elif resource_block:
        statuses["JOB-003"] = "WAITING_RESOURCES"
    else:
        statuses["JOB-003"] = "BLOCKED"
    blockers["JOB-003"] = sorted(set(job3_blockers))

    stage10_ready = stage_map.get(10) == "complete" and bool(
        component_readiness and component_readiness.get("status") == "READY"
    )
    if stage10_ready:
        statuses["JOB-004"] = "COMPLETE"
        blockers["JOB-004"] = []
    elif statuses["JOB-003"] == "READY":
        statuses["JOB-004"] = "READY"
        blockers["JOB-004"] = []
    else:
        statuses["JOB-004"] = "WAITING_DEPENDENCY"
        blockers["JOB-004"] = ["JOB-003_NOT_READY"]

    ready_components = _ready_component_keys(component_inventory)
    expected_components = {
        (date, model) for date in policy["component_dates"] for model in policy["component_models"]
    }
    if ready_components == expected_components:
        statuses["JOB-005"] = "COMPLETE"
        blockers["JOB-005"] = []
    elif statuses["JOB-004"] == "COMPLETE" and stage10_ready:
        statuses["JOB-005"] = "READY"
        blockers["JOB-005"] = [f"COMPONENTS_READY:{len(ready_components)}/{len(expected_components)}"]
    else:
        statuses["JOB-005"] = "WAITING_DEPENDENCY"
        blockers["JOB-005"] = ["JOB-004_NOT_COMPLETE"]

    for number in range(6, 12):
        job_id, parent = f"JOB-{number:03d}", f"JOB-{number - 1:03d}"
        statuses[job_id] = "READY" if statuses[parent] == "COMPLETE" else "WAITING_DEPENDENCY"
        blockers[job_id] = [] if statuses[job_id] == "READY" else [f"{parent}_NOT_COMPLETE"]

    ordered = [job["job_id"] for job in ledger["jobs"]]
    next_job = next((job_id for job_id in ordered if statuses[job_id] not in {"COMPLETE", "DEFERRED"}), None)
    command = ledger["commands"]["resume_selector_4_10"]
    result = {
        "contract_version": "post_finaliser_job_readiness.v1",
        "valid": not any(error.startswith("MALFORMED") for error in errors.values()),
        "selector_run_id": ledger["selector_run_id"],
        "job_statuses": statuses,
        "job_blockers": blockers,
        "next_job": next_job,
        "ready_to_resume": statuses["JOB-003"] == "READY",
        "resume_outcome": "READY_TO_RESUME" if statuses["JOB-003"] == "READY" else "NOT_READY",
        "resume_command": command,
        "resume_command_has_required_switch": " -Resume " in f" {command} ",
        "resources": {
            "finaliser_active": finaliser_active,
            "free_memory_bytes": free_memory_bytes,
            "minimum_free_memory_policy_bytes": int(policy["minimum_free_memory_bytes"]),
            "free_disk_bytes": free_disk_bytes,
            "minimum_free_disk_policy_bytes": int(policy["minimum_free_disk_bytes"]),
            "policy_is_operational_not_mathematical": True,
        },
        "archive": {
            "expected_full_partitions": expected,
            "planned": planned, "completed": completed, "pending": pending,
            "failed": failed, "invalid_rows": invalid,
            "validation_partition_count": validation_count,
            "validation_valid": validation_valid,
            "partial_660_validation_rejected": validation_count == int(policy["partial_validation_partition_count"]),
        },
        "components": {
            "expected_count": len(expected_components),
            "ready_count": len(ready_components),
            "complete_roster": ready_components == expected_components,
        },
        "commands": dict(ledger["commands"]),
        "input_errors": errors,
    }
    result["logical_checksum"] = canonical_hash(result)
    return result


def validate_ledger(ledger: Mapping[str, Any]) -> list[str]:
    reasons = []
    if ledger.get("contract_version") != "post_finaliser_job_ledger.v1":
        reasons.append("CONTRACT_VERSION_INVALID")
    jobs = list(ledger.get("jobs") or [])
    ids = [job.get("job_id") for job in jobs]
    if ids != [f"JOB-{number:03d}" for number in range(1, 12)]:
        reasons.append("JOB_ORDER_INVALID")
    known = set(ids)
    for job in jobs:
        if job.get("status") not in VALID_STATUSES:
            reasons.append(f"STATUS_INVALID:{job.get('job_id')}")
        if set(job.get("dependencies") or ()) - known:
            reasons.append(f"DEPENDENCY_UNKNOWN:{job.get('job_id')}")
        supplied = job.get("logical_checksum")
        if supplied and supplied != canonical_hash({key: value for key, value in job.items() if key != "logical_checksum"}):
            reasons.append(f"JOB_CHECKSUM_MISMATCH:{job.get('job_id')}")
    supplied = ledger.get("logical_checksum")
    if supplied and supplied != canonical_hash({key: value for key, value in ledger.items() if key != "logical_checksum"}):
        reasons.append("LEDGER_CHECKSUM_MISMATCH")
    return reasons


def _selector_stage_map(state: Mapping[str, Any] | None, run_id: str) -> tuple[dict[int, str], list[str]]:
    if not state:
        return {}, ["SELECTOR_RUN_STATE_MISSING"]
    reasons = []
    if str(state.get("run_id")) != run_id:
        reasons.append("SELECTOR_RUN_ID_MISMATCH")
    if state.get("run_state_version") != "selector_parent_publication_run_state_v2":
        reasons.append("SELECTOR_STATE_SCHEMA_MISMATCH")
    stages = {
        int(row["stage_number"]): str(row.get("status"))
        for row in state.get("stages", [])
        if isinstance(row, dict) and "stage_number" in row
    }
    return stages, reasons


def _ready_component_keys(payload: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    output = set()
    for row in (payload or {}).get("components", []):
        if row.get("status") == "READY":
            output.add((str(row.get("date")), str(row.get("model"))))
    return output


def _integer(payload: Mapping[str, Any] | None, *names: str, default: int = 0) -> int:
    for name in names:
        if payload and payload.get(name) is not None:
            try:
                return int(payload[name])
            except (TypeError, ValueError):
                return default
    return default


def available_memory_bytes() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended_virtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
    except (AttributeError, OSError):
        pass
    return 0


def finaliser_process_active() -> bool:
    command = [
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'alpaca_5m_symbol_year_finalizer|finalize_alpaca_5m_symbol_year_archive' } | Select-Object -ExpandProperty ProcessId",
    ]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(output.stdout.strip())


def _resolved_path(ledger, key, run_id, override):
    if override:
        return Path(override)
    value = ledger["small_file_paths"][key].format(selector_run_id=run_id)
    return REPO_ROOT / value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only post-finaliser job readiness checker.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--selector-run-id", default="20260716T091011Z")
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--archive-validation", type=Path)
    parser.add_argument("--selector-state", type=Path)
    parser.add_argument("--ops-4a-readiness", type=Path)
    parser.add_argument("--component-readiness", type=Path)
    parser.add_argument("--component-inventory", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--next-job", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    ledger, error = load_json(args.ledger)
    if error or ledger is None:
        print(json.dumps({"status": "INVALID_LEDGER", "error": error}, indent=2))
        return 2
    ledger = dict(ledger)
    ledger["selector_run_id"] = args.selector_run_id
    ledger_errors = validate_ledger(ledger)
    if ledger_errors:
        print(json.dumps({"status": "INVALID_LEDGER", "blockers": ledger_errors}, indent=2))
        return 2
    inputs, errors = {}, {}
    mappings = {
        "finaliser_progress": ("finaliser_progress", args.progress),
        "archive_validation": ("archive_validation", args.archive_validation),
        "selector_run_state": ("selector_run_state", args.selector_state),
        "ops_4a_readiness": ("ops_4a_readiness", args.ops_4a_readiness),
        "component_readiness": ("component_readiness", args.component_readiness),
        "component_inventory": ("component_inventory", args.component_inventory),
    }
    for output_key, (ledger_key, override) in mappings.items():
        value, read_error = load_json(_resolved_path(ledger, ledger_key, args.selector_run_id, override))
        inputs[output_key] = value
        if read_error:
            errors[output_key] = read_error
    result = evaluate_readiness(
        ledger,
        progress=inputs["finaliser_progress"],
        archive_validation=inputs["archive_validation"],
        selector_state=inputs["selector_run_state"],
        ops_4a=inputs["ops_4a_readiness"],
        component_readiness=inputs["component_readiness"],
        component_inventory=inputs["component_inventory"],
        finaliser_active=finaliser_process_active(),
        free_memory_bytes=available_memory_bytes(),
        free_disk_bytes=shutil.disk_usage(REPO_ROOT).free,
        input_errors=errors,
    )
    if args.job:
        payload = {
            "job_id": args.job,
            "status": result["job_statuses"].get(args.job),
            "blockers": result["job_blockers"].get(args.job),
            "definition": next((job for job in ledger["jobs"] if job["job_id"] == args.job), None),
        }
    elif args.next_job:
        payload = {
            "next_job": result["next_job"],
            "status": result["job_statuses"].get(result["next_job"]),
            "blockers": result["job_blockers"].get(result["next_job"]),
            "resume_outcome": result["resume_outcome"],
            "resume_command": result["resume_command"],
        }
    elif args.summary and not args.json:
        payload = {
            "next_job": result["next_job"], "resume_outcome": result["resume_outcome"],
            "archive": result["archive"], "resources": result["resources"],
        }
    else:
        payload = result
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
