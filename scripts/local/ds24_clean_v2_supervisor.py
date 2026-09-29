from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (
    authority_bundle,
    clean_source_hash,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_runtime import (
    MAX_DELL_MODEL_WORKERS,
    REFIT_POLICY_ID,
    RUN_ID,
    validate_ownership,
)
from core.research.ml.ds24.clean_v2_resources import (
    DELL_WORKER_THREAD_ENVIRONMENT,
    HOST_RESOURCE_PRESSURE,
    LOCAL_CAPACITY_DEFERRAL,
    RESOURCE_POLICY,
    RESOURCE_CAPACITY_DEFERRED_EXIT_CODE,
    RESOURCE_PRESSURE_EXIT_CODE,
    SINGLE_WORKER_CAPACITY_BLOCK,
    AllocationEstimate,
    CapacitySnapshot,
    JobMemorySnapshot,
    MemoryDecision,
    ProcessIdentity,
    ResourceReservationLedger,
    WorkerContainment,
    WorkerContainmentError,
    WorkerReservationOwner,
    actual_host_pressure_reasons,
    allocation_failure_decision,
    capacity_change_is_material,
    create_worker_containment,
    estimate_worker_peak_allocation,
    evaluate_memory,
    family_scheduling_profile,
    process_identity,
    process_identity_matches,
    process_memory_snapshot,
    read_json_object_with_retry,
    recovery_policy_payload,
    reconcile_runtime_status_view,
    reservation_payload_is_local_capacity_deferral,
    same_process_creation_time,
    service_file_reservation_requests,
    system_memory_snapshot,
    write_json_object_atomic,
)
from scripts.local.ds24_clean_v2_reconcile_failures import (
    FailureReconciliationError,
    reconcile_registered_control_resume_states,
    validate_registered_control_resume_states,
)


PRIMARY_DELL_FAMILY = "random_forest"
RUN_ROOT = ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID
SUPERVISOR_SCRIPT = Path(__file__).resolve()
QUEUE_POLL_SECONDS = 5.0
PARENT_STARTUP_RECORD_TIMEOUT_SECONDS = 10.0
WINDOWS_CONTROL_C_EXIT_CODE = 0xC000013A


@dataclass(frozen=True)
class OwnedWorker:
    process: subprocess.Popen[Any]
    process_creation_time_utc: str


@dataclass(frozen=True)
class DeferredResourceFamily:
    family: str
    deferred_at_monotonic: float
    capacity_before: CapacitySnapshot
    evidence: Mapping[str, Any]


def _worker_exit_disposition(
    exit_code: int, job_snapshot: JobMemorySnapshot | None
) -> str:
    """Keep scheduler deferral distinct from terminal worker failure."""

    if exit_code == 0:
        return "COMPLETE"
    if exit_code == RESOURCE_CAPACITY_DEFERRED_EXIT_CODE:
        return "DEFERRED_RESOURCE_CAPACITY"
    if exit_code == RESOURCE_PRESSURE_EXIT_CODE or (
        job_snapshot is not None and job_snapshot.limit_violation_detected
    ):
        return "RESOURCE_LIMIT_NO_AUTOMATIC_RETRY"
    return "FAILED_CLOSED_NO_AUTOMATIC_RETRY"


def _record_unexpected_worker_exit(
    *,
    family: str,
    host: str,
    attempt_generation: int,
    worker: OwnedWorker,
    exit_code: int,
    job_snapshot: JobMemorySnapshot | None,
) -> dict[str, Any]:
    """Persist immutable current-attempt evidence for an abnormal worker exit."""

    family_root = RUN_ROOT / f"family={family}"
    state_path = family_root / "resume_state.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    failure_path = family_root / "worker_failure.json"
    failure = _read_json(failure_path) if failure_path.is_file() else {}
    unsigned_exit_code = int(exit_code) & 0xFFFFFFFF
    event_generation = time.time_ns()
    current_failure_artifact = bool(
        int(failure.get("attempt_generation", 0) or 0) == attempt_generation
        and failure.get("family") == family
        and failure.get("host") == host
    )
    evidence = {
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "attempt_generation": attempt_generation,
        "event_generation": event_generation,
        "classification": "DS24_CLEAN_V2_WORKER_EXIT_FAILED_CLOSED",
        "terminal_state": "FAILED_CLOSED",
        "exit_code": int(exit_code),
        "exit_code_hex": f"0x{unsigned_exit_code:08X}",
        "exit_reason": (
            "EXTERNAL_CONTROL_EVENT"
            if unsigned_exit_code == WINDOWS_CONTROL_C_EXIT_CODE
            else "UNEXPECTED_NONZERO_EXIT"
        ),
        "pid": worker.process.pid,
        "process_creation_time_utc": worker.process_creation_time_utc,
        "observed_at_utc": _utc_now(),
        "state_attempt_generation": int(
            state.get("attempt_generation", 0) or 0
        ),
        "state_terminal_state": state.get("terminal_state"),
        "state_heartbeat_utc": state.get("heartbeat_utc"),
        "current_worker_failure_artifact_recorded": current_failure_artifact,
        "worker_failure_artifact_attempt_generation": failure.get(
            "attempt_generation"
        ),
        "job_snapshot": job_snapshot.payload() if job_snapshot else None,
        "automatic_retry": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    _write_json_atomic(
        family_root
        / (
            f"worker_exit_attempt={attempt_generation}_"
            f"event={event_generation}.json"
        ),
        evidence,
    )
    _write_json_atomic(family_root / "worker_exit.json", evidence)
    return evidence


def _host_path(stem: str, host: str) -> Path:
    return RUN_ROOT / f"{stem}_{host}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_runtime_file_retry(event: Mapping[str, Any]) -> None:
    print(
        json.dumps({**dict(event), "component": "supervisor"}, sort_keys=True),
        flush=True,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_object_atomic(path, payload, on_retry=_log_runtime_file_retry)


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object_with_retry(path, on_retry=_log_runtime_file_retry)


def _partition_inventory_matches(
    root: Path,
    partitions: list[dict[str, Any]],
    *,
    relative_path_key: str,
    bytes_key: str,
) -> bool:
    for partition in partitions:
        if not isinstance(partition, dict):
            return False
        relative = Path(str(partition.get(relative_path_key, "")))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        candidate = root / relative
        try:
            if (
                not candidate.is_file()
                or candidate.stat().st_size != int(partition[bytes_key])
            ):
                return False
        except (KeyError, OSError, TypeError, ValueError):
            return False
    return True


def _legacy_ds24_processes() -> list[dict[str, Any]]:
    try:
        import psutil
    except ImportError:
        return []
    patterns = (
        "ds24_p8_r14_e3g_c2_r7_r14_policy_worker",
        "ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor",
        "ds24_v3_sequence_policy_worker",
        "ds24_v3_lightgbm_ranking_policy_worker",
        "r40_full_tournament",
    )
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        lowered = command.lower()
        if process.pid == os.getpid() or "ds24_clean_v2" in lowered:
            continue
        if any(pattern in lowered for pattern in patterns):
            rows.append(
                {
                    "pid": int(process.pid),
                    "started_at_utc": datetime.fromtimestamp(
                        float(process.info["create_time"]), timezone.utc
                    ).isoformat(),
                    "command": command,
                }
            )
    return rows


def _launch_families(host: str) -> list[str]:
    tournament = load_contract("tournament_contract.json")
    ownership = load_contract("cross_host_ownership.json")
    ordered = [
        *tournament["lanes"]["FULL_CLEAN"],
        *tournament["lanes"]["SHORT_REQUALIFICATION"],
        *tournament["lanes"]["UNSCORED_DISCOVERY"],
        *tournament["lanes"]["CONTROLS"],
    ]
    owned = set(ownership["hosts"][host])
    launchable = [family for family in ordered if family in owned]
    if host != "dell":
        return launchable
    # Keep the progressed RF authority first, then establish predictive
    # coverage before controls.  Within a workload class, bounded measured fit
    # time and the guarded memory envelope determine order.  The reservation
    # ledger remains the final compatibility gate and may bypass an unsafe
    # candidate to use otherwise-idle capacity without starving predictors.
    workload_priority = {
        "LIGHT_PREDICTIVE": 0,
        "TREE_PREDICTIVE": 1,
        "PREDICTIVE": 1,
        "SEQUENCE_PREDICTIVE": 2,
        "CONTROL": 3,
    }
    original = {family: index for index, family in enumerate(launchable)}

    def scheduling_key(family: str) -> tuple[int, int, float, int, int]:
        if family == PRIMARY_DELL_FAMILY:
            return (0, 0, 0.0, 0, original[family])
        scheduling = family_scheduling_profile(family)
        guarded = estimate_worker_peak_allocation(family).guarded()
        return (
            1,
            workload_priority.get(scheduling.workload_class, 2),
            round(scheduling.bounded_fit_wall_seconds, 1),
            max(guarded.physical_bytes, guarded.commit_bytes),
            original[family],
        )

    return sorted(
        launchable,
        key=scheduling_key,
    )


def _primary_family_capacity_deferred(
    host: str,
    deferred: Mapping[str, Any],
    complete: set[str],
) -> bool:
    """Keep the progressed Dell RF worker pinned ahead of replacements."""

    return (
        host == "dell"
        and PRIMARY_DELL_FAMILY in deferred
        and PRIMARY_DELL_FAMILY not in complete
    )


def _resource_snapshot() -> tuple[int, Any]:
    disk_free = int(shutil.disk_usage(ROOT.anchor or ROOT).free)
    return disk_free, system_memory_snapshot()


def _reconcile_control_resume_states_for_preflight(
    host: str,
) -> list[dict[str, Any]]:
    """Bind every registered control before admission while runtime is stopped."""

    return reconcile_registered_control_resume_states(
        host=host,
        repository_root=ROOT,
        current_source_hash=clean_source_hash(),
    )


def _validate_control_resume_states_for_live_runtime(
    host: str,
) -> list[dict[str, Any]]:
    """Validate control authority without invoking stopped-state mutation."""

    return validate_registered_control_resume_states(
        host=host,
        repository_root=ROOT,
    )


def _build_preflight_report(
    host: str,
    *,
    write_admission: bool,
    control_state_mode: Literal["stopped_reconcile", "live_read_only"],
) -> dict[str, Any]:
    feature = load_contract("feature_authority.json")
    target = load_contract("target_contract.json")
    tournament = load_contract("tournament_contract.json")
    models = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    validate_ownership(ownership["hosts"])
    bundle = authority_bundle()
    reasons: list[str] = []
    control_reconciliation_reports: list[dict[str, Any]] = []
    stopped_runtime: dict[str, Any] = {
        "stopped": False,
        "classification": "NOT_APPLICABLE_LIVE_RUNTIME_READ_ONLY",
    }
    try:
        if control_state_mode == "stopped_reconcile":
            stopped_runtime = _prepare_stopped_runtime_for_preflight(host)
            if not stopped_runtime["stopped"]:
                reasons.append(str(stopped_runtime["blocking_reason"]))
            else:
                control_reconciliation_reports = (
                    _reconcile_control_resume_states_for_preflight(host)
                )
        else:
            control_reconciliation_reports = (
                _validate_control_resume_states_for_live_runtime(host)
            )
    except FailureReconciliationError as exc:
        failure_classification = (
            "CONTROL_RESUME_STATE_RECONCILIATION_FAILED"
            if control_state_mode == "stopped_reconcile"
            else "CONTROL_RESUME_STATE_LIVE_VALIDATION_FAILED"
        )
        reasons.append(f"{failure_classification}:{exc}")

    if tournament["refit_policy"]["id"] != REFIT_POLICY_ID:
        reasons.append("REFIT_POLICY_ID_MISMATCH")
    if tournament["refit_policy"].get("daily_refit_fallback_allowed") is not False:
        reasons.append("DAILY_REFIT_FALLBACK_NOT_DISABLED")
    all_lanes = {
        family
        for families in tournament["lanes"].values()
        for family in families
    }
    retired = set(ownership.get("retired_source_only", []))
    if all_lanes & retired:
        reasons.append(
            "RETIRED_FAMILY_SCHEDULED:" + ",".join(sorted(all_lanes & retired))
        )

    sidecar_root = ROOT / feature["sidecar"]["path"]
    sidecar_manifest_path = sidecar_root / "authority_manifest.json"
    sidecar_manifest = (
        _read_json(sidecar_manifest_path) if sidecar_manifest_path.is_file() else {}
    )
    expected_rows = int(feature["base_authority"]["physical_snapshot_row_count"])
    sidecar_logical_hash = stable_hash(
        {
            key: value
            for key, value in sidecar_manifest.items()
            if key != "logical_sha256"
        }
    )
    sidecar_partitions = sidecar_manifest.get("partitions", [])
    base_root = ROOT / feature["base_authority"]["path"]
    if not (
        sidecar_manifest.get("complete") is True
        and sidecar_manifest.get("authority_id") == feature["authority_id"]
        and sidecar_manifest.get("repair_columns")
        == feature["sidecar"]["repair_columns"]
        and int(sidecar_manifest.get("row_count", -1)) == expected_rows
        and len(sidecar_partitions)
        == feature["base_authority"]["physical_snapshot_stock_partitions"]
        and _partition_inventory_matches(
            sidecar_root,
            sidecar_partitions,
            relative_path_key="relative_path",
            bytes_key="bytes",
        )
        and _partition_inventory_matches(
            base_root,
            sidecar_partitions,
            relative_path_key="base_relative_path",
            bytes_key="base_bytes",
        )
        and sidecar_manifest.get("logical_sha256") == sidecar_logical_hash
    ):
        reasons.append("V2_SIDECAR_NOT_FULLY_MATERIALIZED")

    certificate_path = RUN_ROOT / "causality_certificate.json"
    certificate = _read_json(certificate_path) if certificate_path.is_file() else {}
    fixture = certificate.get("fixture", {})
    certificate_hash = stable_hash(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_sha256"
        }
    )
    if not (
        certificate.get("passed") is True
        and certificate.get("predictor_count") == 101
        and certificate.get("terminal_classification")
        == "101 / 101 FEATURES CAUSAL UNDER FUTURE-BAR PERTURBATION"
        and certificate.get("certification_scope") == "FULL_MATERIALIZED_AUTHORITY"
        and certificate.get("feature_authority_id") == feature["authority_id"]
        and certificate.get("static_authority_bundle_sha256")
        == bundle["bundle_sha256"]
        and fixture.get("years") == [2017, 2024]
        and set(fixture.get("symbols", []))
        >= {"AAA", "BBB", "SPY", "QQQ", "GLD", "TLT", "XLK"}
        and set(fixture.get("session_types", []))
        >= {"REGULAR", "EARLY_CLOSE", "PRE_MARKET", "AFTER_HOURS"}
        and certificate.get("independent_formula_checks_passed") is True
        and certificate.get("rth_extended_hours_semantics_passed") is True
        and certificate.get("calendar_state_semantics_passed") is True
        and certificate.get("target_maturity_checks_passed") is True
        and certificate.get("zero_target_columns_in_model_inputs") is True
        and certificate.get("certificate_sha256") == certificate_hash
    ):
        reasons.append("FINAL_CAUSALITY_DATA_CERTIFICATE_MISSING_OR_FAILED")

    if (
        target.get("resolved_contract_sha256")
        != "8e2d5458044c17cf60bc1d8e46b71357599f3837ddb53d64d9b00d322d17f419"
    ):
        reasons.append("TARGET_CONTRACT_HASH_MISMATCH")
    target_delta_path = (
        ROOT / target["physical_authority"]["delta_path"] / "authority_manifest.json"
    )
    target_delta = (
        _read_json(target_delta_path) if target_delta_path.is_file() else {}
    )
    target_authority_hash = stable_hash(
        {
            key: value
            for key, value in target_delta.items()
            if key != "logical_sha256"
        }
    )
    target_partitions = target_delta.get("partitions", [])
    target_base_snapshot = target_delta.get("base_snapshot") or {}
    target_base_partitions = target_base_snapshot.get("partitions", [])
    target_code_hashes = {
        code_hash
        for partition in target_partitions
        for code_hash in partition.get("target_code_hashes", [])
    }
    if not (
        target_delta.get("complete") is True
        and target_delta.get("authority_id") == target["authority_id"]
        and target_delta.get("target_id") == target["target_id"]
        and target_delta.get("symbol_count")
        == feature["base_authority"]["physical_snapshot_asset_count"]
        and len(target_partitions) == target_delta.get("symbol_count")
        and _partition_inventory_matches(
            target_delta_path.parent,
            target_partitions,
            relative_path_key="relative_path",
            bytes_key="bytes",
        )
        and len(target_base_partitions)
        == feature["base_authority"]["physical_snapshot_stock_partitions"]
        and target_base_snapshot.get("aggregate_partition_sha256")
        == stable_hash(target_base_partitions)
        and _partition_inventory_matches(
            ROOT / target["physical_authority"]["base_path"],
            target_base_partitions,
            relative_path_key="base_relative_path",
            bytes_key="base_bytes",
        )
        and target_delta.get("maximum_historical_outcome_timestamp_consumed")
        == target["maximum_historical_outcome_timestamp_consumed"].replace(
            "Z", "+00:00"
        )
        and target_code_hashes == {target["target_code_hash"]}
        and target_delta.get("logical_sha256") == target_authority_hash
    ):
        reasons.append("CLEAN_TARGET_DELTA_NOT_FULLY_MATERIALIZED")

    configured = set(models["families"])
    assigned = set(ownership["hosts"].get(host, []))
    missing_models = sorted(
        family
        for family in assigned
        if family not in configured and family not in models["controls"]
    )
    if missing_models:
        reasons.append(f"OWNED_FAMILY_CONFIG_MISSING:{','.join(missing_models)}")
    if host == "dell":
        for family in sorted(assigned & set(models["families"])):
            parameters = models["families"][family].get("parameters", {})
            if "n_jobs" in parameters and int(parameters["n_jobs"]) != 1:
                reasons.append(f"DELL_INNER_THREAD_LIMIT_INVALID:{family}:n_jobs")
            if family in {
                "transformer",
                "momentum_transformer",
                "market_context_encoder",
                "temporal_fusion_transformer",
            } and (
                int(parameters.get("torch_num_threads", -1)) != 1
                or int(parameters.get("dataloader_num_workers", -1)) != 0
            ):
                reasons.append(
                    f"DELL_SEQUENCE_THREAD_LIMIT_INVALID:{family}"
                )
    worker_commands = tournament.get("worker_commands", {}).get(host, {})
    launch_families = _launch_families(host)
    commandless = sorted(
        family for family in launch_families if family not in worker_commands
    )
    if commandless:
        reasons.append(
            f"CLEAN_V2_WORKER_COMMANDS_UNRESOLVED:{','.join(commandless)}"
        )
    forbidden_command = sorted(
        family
        for family, command in worker_commands.items()
        if "ds24_clean_v2_family_worker.py" not in " ".join(map(str, command))
    )
    if forbidden_command:
        reasons.append(
            "NON_CLEAN_WORKER_BINDING:" + ",".join(forbidden_command)
        )

    legacy_processes = _legacy_ds24_processes()
    if legacy_processes:
        reasons.append("OBSOLETE_R40_PROCESS_ACTIVE")

    disk_free, memory_snapshot = _resource_snapshot()
    required_gib = float(tournament["storage"]["minimum_post_launch_free_gib"])
    required_gib += float(
        tournament["storage"]["projected_tournament_gib_upper_bound"]
    )
    if not sidecar_manifest.get("complete"):
        required_gib += float(
            tournament["storage"]["projected_sidecar_gib_upper_bound"]
        )
    if disk_free < int(required_gib * 1024**3):
        reasons.append("INSUFFICIENT_DISK_FOR_DECLARED_RESERVE")
    pending_families = [
        family
        for family in launch_families
        if _family_state(family).get("terminal_state") != "COMPLETE"
    ]
    next_family = pending_families[0] if pending_families else None
    next_estimate = (
        estimate_worker_peak_allocation(next_family)
        if next_family
        else AllocationEstimate(0, 0)
    )
    memory_decision = evaluate_memory(
        stage="manual_preflight_next_worker",
        estimated_physical_bytes=next_estimate.physical_bytes,
        estimated_commit_bytes=next_estimate.commit_bytes,
        snapshot=memory_snapshot,
        purpose="admission",
    )
    reasons.extend(memory_decision.blocking_reasons)

    feature_hash = sidecar_manifest.get("logical_sha256") or stable_hash(feature)
    if not reasons:
        classification = "DS24_CLEAN_V2_READY_FOR_MANUAL_TOURNAMENT_LAUNCH"
    elif "FINAL_CAUSALITY_DATA_CERTIFICATE_MISSING_OR_FAILED" in reasons:
        classification = "DS24_CLEAN_V2_IMPLEMENTATION_READY_MANUAL_DATA_BUILD_REQUIRED"
    elif (
        "INSUFFICIENT_DISK_FOR_DECLARED_RESERVE" in reasons
        or not memory_decision.safe
    ):
        classification = "DS24_CLEAN_V2_BLOCKED_RESOURCE"
    else:
        classification = "DS24_CLEAN_V2_BLOCKED_DATA_OR_CONFIG_AUTHORITY"
    report: dict[str, Any] = {
        "run_id": RUN_ID,
        "host_role": host,
        "hostname": socket.gethostname(),
        "checked_at_utc": _utc_now(),
        "feature_authority_id": feature["authority_id"],
        "feature_authority_hash": feature_hash,
        "target_authority_hash": target_delta.get("logical_sha256"),
        "target_contract_hash": target["resolved_contract_sha256"],
        "static_authority_bundle_sha256": bundle["bundle_sha256"],
        "clean_source_hash": clean_source_hash(),
        "results_ledger_hash": load_contract("prior_evidence_manifest.json")[
            "workbook"
        ]["captured_xlsx_export_sha256"],
        "refit_policy": REFIT_POLICY_ID,
        "launch_families": launch_families,
        "mac_owned_families": ownership["hosts"]["mac"],
        "maximum_model_workers": MAX_DELL_MODEL_WORKERS if host == "dell" else 1,
        "disk_free_bytes": disk_free,
        "ram_available_bytes": memory_snapshot.available_physical_bytes,
        "commit_headroom_bytes": memory_snapshot.commit_headroom_bytes,
        "commit_limit_bytes": memory_snapshot.commit_limit_bytes,
        "committed_bytes": memory_snapshot.committed_bytes,
        "memory_snapshot_source": memory_snapshot.source,
        "resource_policy": recovery_policy_payload(),
        "next_worker_memory_decision": memory_decision.payload(),
        "obsolete_ds24_processes": legacy_processes,
        "control_resume_state_reconciliation": control_reconciliation_reports,
        "control_resume_state_mode": control_state_mode,
        "stopped_runtime_reconciliation": stopped_runtime,
        "blocking_reasons": sorted(set(reasons)),
        "ready": not reasons,
        "classification": classification,
        "paper_orders": 0,
        "live_orders": 0,
    }
    if write_admission and report["ready"]:
        admission_core = {
            "run_id": RUN_ID,
            "host_role": host,
            "hostname": socket.gethostname(),
            "feature_authority_hash": report["feature_authority_hash"],
            "target_authority_hash": report["target_authority_hash"],
            "target_contract_hash": report["target_contract_hash"],
            "static_authority_bundle_sha256": report[
                "static_authority_bundle_sha256"
            ],
            "clean_source_hash": report["clean_source_hash"],
            "refit_policy": REFIT_POLICY_ID,
            "launch_families": launch_families,
            "maximum_model_workers": report["maximum_model_workers"],
            "resource_policy": report["resource_policy"],
            "issued_at_utc": _utc_now(),
        }
        admission = {
            **admission_core,
            "admission_token": stable_hash(admission_core),
        }
        admission_path = _host_path("manual_admission", host)
        _write_json_atomic(admission_path, admission)
        report["manual_admission_path"] = admission_path.relative_to(ROOT).as_posix()
        report["manual_admission_token"] = admission["admission_token"]
    return report


def preflight(host: str, *, write_admission: bool = False) -> dict[str, Any]:
    """Perform stopped-runtime reconciliation and admission preflight."""

    return _build_preflight_report(
        host,
        write_admission=write_admission,
        control_state_mode="stopped_reconcile",
    )


def _runtime_preflight(host: str) -> dict[str, Any]:
    """Revalidate admission inputs without stopped-runtime mutations."""

    return _build_preflight_report(
        host,
        write_admission=False,
        control_state_mode="live_read_only",
    )


def _pid_matches(
    pid: int,
    required_fragments: tuple[str, ...],
    *,
    expected_creation_time: Any,
) -> bool:
    return process_identity_matches(
        process_identity(pid),
        expected_creation_time=expected_creation_time,
        required_command_fragments=required_fragments,
    )


def _validate_admission_against_report(
    host: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    path = _host_path("manual_admission", host)
    if not path.is_file():
        raise RuntimeError("MANUAL_PREFLIGHT_ADMISSION_MISSING")
    admission = _read_json(path)
    token = admission.pop("admission_token", None)
    if token != stable_hash(admission):
        raise RuntimeError("MANUAL_PREFLIGHT_ADMISSION_HASH_MISMATCH")
    if not report["ready"]:
        raise RuntimeError(
            "MANUAL_PREFLIGHT_NO_LONGER_VALID:"
            + ",".join(report["blocking_reasons"])
        )
    checks = {
        "run_id": report["run_id"],
        "host_role": report["host_role"],
        "feature_authority_hash": report["feature_authority_hash"],
        "target_authority_hash": report["target_authority_hash"],
        "target_contract_hash": report["target_contract_hash"],
        "static_authority_bundle_sha256": report[
            "static_authority_bundle_sha256"
        ],
        "clean_source_hash": report["clean_source_hash"],
        "refit_policy": report["refit_policy"],
        "launch_families": report["launch_families"],
        "maximum_model_workers": report["maximum_model_workers"],
        "resource_policy": report["resource_policy"],
    }
    for key, value in checks.items():
        if admission.get(key) != value:
            raise RuntimeError(f"MANUAL_PREFLIGHT_ADMISSION_STALE:{key}")
    return {**admission, "admission_token": token}


def _validated_admission(host: str) -> dict[str, Any]:
    """Validate admission while the runtime is stopped."""

    return _validate_admission_against_report(host, preflight(host))


def _validated_runtime_admission(host: str) -> dict[str, Any]:
    """Read-only admission validation for the already-live supervisor child."""

    return _validate_admission_against_report(host, _runtime_preflight(host))


def _family_state(family: str) -> dict[str, Any]:
    path = RUN_ROOT / f"family={family}" / "resume_state.json"
    return _read_json(path) if path.is_file() else {}


def _move_family_to_queue_back(queue: list[str], family: str) -> None:
    if family in queue:
        queue.remove(family)
    queue.append(family)


def _release_worker_reservation(
    ledger: ResourceReservationLedger,
    *,
    family: str,
    worker: OwnedWorker,
    attempt_generation: int,
) -> bool:
    return ledger.release_worker(
        WorkerReservationOwner(
            family=family,
            attempt_generation=attempt_generation,
            pid=worker.process.pid,
            process_creation_time_utc=worker.process_creation_time_utc,
        )
    )


def _actual_pressure_decision(
    *,
    stage: str,
    memory_snapshot: Any,
    job_snapshot: JobMemorySnapshot | None,
) -> MemoryDecision:
    decision = evaluate_memory(
        stage=stage,
        estimated_allocation_bytes=0,
        snapshot=memory_snapshot,
        purpose="observation",
    )
    reasons = actual_host_pressure_reasons(memory_snapshot, job_snapshot)
    if not reasons:
        return decision
    return MemoryDecision(
        **{
            **decision.__dict__,
            "safe": False,
            "blocking_reasons": reasons,
        }
    )


def _record_supervisor_capacity_deferral(
    *,
    family: str,
    host: str,
    attempt_generation: int,
    resource_decision: Mapping[str, Any],
    single_worker_block: bool = False,
) -> dict[str, Any]:
    family_root = RUN_ROOT / f"family={family}"
    state_path = family_root / "resume_state.json"
    state_exists = state_path.is_file()
    state = _read_json(state_path) if state_exists else {}
    deferred_at = _utc_now()
    terminal_state = (
        "BLOCKED_SINGLE_WORKER_CAPACITY"
        if single_worker_block
        else "DEFERRED_RESOURCE_CAPACITY"
    )
    classification = (
        "DS24_CLEAN_V2_SINGLE_WORKER_CAPACITY_BLOCK"
        if single_worker_block
        else "DS24_CLEAN_V2_DEFERRED_RESOURCE_CAPACITY"
    )
    state_generation = int(state.get("attempt_generation", 0) or 0)
    annotate_resume_state = bool(
        state_exists
        and state_generation <= attempt_generation
        and state.get("terminal_state") != "COMPLETE"
    )
    evidence = {
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "attempt_generation": attempt_generation,
        "classification": classification,
        "terminal_state": terminal_state,
        "deferred_at_utc": deferred_at,
        "resource_decision": dict(resource_decision),
        "completed_refits": list(state.get("completed_refits", [])),
        "metrics_cursor": int(state.get("metrics_cursor", 0) or 0),
        "automatic_retry": False,
        "scheduler_readmission_requires_material_capacity_change": True,
        "scientific_resume_state_updated": False,
        "resume_state_annotation_updated": annotate_resume_state,
        "resume_attempt_generation_preserved": (
            state_generation if state_exists else None
        ),
        "paper_orders": 0,
        "live_orders": 0,
    }
    family_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        family_root
        / (
            f"resource_deferral_attempt={attempt_generation}_"
            f"event={time.time_ns()}.json"
        ),
        evidence,
    )
    _write_json_atomic(family_root / "resource_deferral.json", evidence)
    if annotate_resume_state:
        _write_json_atomic(
            state_path,
            {
                **state,
                "run_id": RUN_ID,
                "family": family,
                "owner_host": host,
                "terminal_state": terminal_state,
                "resource_deferral_classification": classification,
                "resource_deferral_timestamp": deferred_at,
                "resource_decision": dict(resource_decision),
                "completed_refits": evidence["completed_refits"],
                "metrics_cursor": evidence["metrics_cursor"],
                "heartbeat_utc": deferred_at,
                "paper_orders": 0,
                "live_orders": 0,
            },
        )
    return evidence


def _reconcile_persisted_runtime(
    host: str,
    *,
    persist: bool,
) -> dict[str, Any]:
    """Reconcile stale persisted runtime identities without touching progress."""

    status_path = _host_path("supervisor_status", host)
    if not status_path.is_file():
        return {}
    original = _read_json(status_path)
    status = reconcile_runtime_status_view(original, host=host)
    if not status.get("runtime_reconciliation_required"):
        if persist and original.get("runtime_reconciliation_required"):
            status["runtime_reconciliation_required"] = False
            _write_json_atomic(status_path, status)
        return status
    status_generation = int(status.get("attempt_generation", 0) or 0)
    stale_families = {
        str(row.get("family"))
        for row in status.get("stale_worker_records_reconciled", [])
        if row.get("family")
    }
    deferred = dict(status.get("deferred_resource_families") or {})
    paused = dict(status.get("resource_paused_families") or {})
    for family, evidence in list(paused.items()):
        if not isinstance(evidence, Mapping):
            continue
        resource_decision = evidence.get("resource_decision") or {}
        if isinstance(resource_decision, Mapping) and (
            reservation_payload_is_local_capacity_deferral(resource_decision)
        ):
            deferred[family] = {
                **dict(evidence),
                "classification": "DS24_CLEAN_V2_DEFERRED_RESOURCE_CAPACITY",
                "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
                "reconciled_from_legacy_resource_pause": True,
            }
            paused.pop(family, None)
    queued = [str(value) for value in status.get("queued_families", [])]
    for family in sorted(stale_families | set(deferred)):
        if family not in queued:
            queued.append(family)
    status.update(
        {
            "queued_families": queued,
            "resource_paused_families": paused,
            "deferred_resource_families": deferred,
            "capacity_only_deferrals": sorted(deferred),
            "reconciled_at_utc": _utc_now(),
            "ready": False,
        }
    )
    if not paused and not status.get("orphaned_verified_workers"):
        status["blocking_reasons"] = []
    if persist:
        reconciliation_root = RUN_ROOT / "runtime_reconciliation"
        reconciliation_root.mkdir(parents=True, exist_ok=True)
        status_evidence_path = reconciliation_root / (
            f"supervisor_status_{host}.attempt={status_generation}.before.json"
        )
        if status_evidence_path.exists():
            status_evidence_path = reconciliation_root / (
                f"supervisor_status_{host}.attempt={status_generation}."
                f"before={time.time_ns()}.json"
            )
        _write_json_atomic(status_evidence_path, original)
        status["pre_reconciliation_status_evidence_path"] = (
            status_evidence_path.relative_to(ROOT).as_posix()
        )
        for family in sorted(stale_families | set(deferred)):
            state_path = RUN_ROOT / f"family={family}" / "resume_state.json"
            if not state_path.is_file():
                continue
            state = _read_json(state_path)
            state_generation = int(state.get("attempt_generation", 0) or 0)
            if state_generation != status_generation:
                continue
            terminal_state = str(state.get("terminal_state") or "")
            updates: dict[str, Any] = {}
            if family in stale_families and terminal_state == "RUNNING":
                updates = {
                    "terminal_state": "RESUMABLE_STALE_PROCESS",
                    "runtime_reconciliation_classification": (
                        "DEAD_IDENTITY_VERIFIED_WORKER_RECONCILED"
                    ),
                }
            if family in deferred:
                resource_decision = state.get("resource_decision") or {}
                if isinstance(resource_decision, Mapping) and (
                    reservation_payload_is_local_capacity_deferral(
                        resource_decision
                    )
                ):
                    updates = {
                        "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
                        "resource_deferral_classification": (
                            "DS24_CLEAN_V2_DEFERRED_RESOURCE_CAPACITY"
                        ),
                        "reconciled_from_legacy_resource_pause": True,
                    }
            if updates:
                state_evidence_path = reconciliation_root / (
                    f"family={family}.attempt={state_generation}.before.json"
                )
                if state_evidence_path.exists():
                    state_evidence_path = reconciliation_root / (
                        f"family={family}.attempt={state_generation}."
                        f"before={time.time_ns()}.json"
                    )
                _write_json_atomic(state_evidence_path, state)
                _write_json_atomic(
                    state_path,
                    {
                        **state,
                        **updates,
                        "runtime_reconciled_at_utc": status["reconciled_at_utc"],
                        "pre_reconciliation_state_evidence_path": (
                            state_evidence_path.relative_to(ROOT).as_posix()
                        ),
                    },
                )
        status["runtime_reconciliation_required"] = False
        _write_json_atomic(status_path, status)
    return status


def _publish_preflight_status(host: str, report: Mapping[str, Any]) -> None:
    """Publish preflight without discarding a prior terminal attempt status."""

    status_path = _host_path("supervisor_status", host)
    published = dict(report)
    if status_path.is_file():
        previous = _read_json(status_path)
        previous_classification = str(previous.get("classification") or "")
        if (
            previous.get("failed_families")
            or "FAILED_CLOSED" in previous_classification
        ):
            attempt_generation = int(previous.get("attempt_generation", 0) or 0)
            evidence_root = RUN_ROOT / "runtime_reconciliation"
            evidence_root.mkdir(parents=True, exist_ok=True)
            evidence_path = evidence_root / (
                f"supervisor_status_{host}.attempt={attempt_generation}.terminal.json"
            )
            if evidence_path.exists():
                evidence_path = evidence_root / (
                    f"supervisor_status_{host}.attempt={attempt_generation}."
                    f"terminal={time.time_ns()}.json"
                )
            _write_json_atomic(evidence_path, previous)
            published["pre_preflight_terminal_status_evidence_path"] = (
                evidence_path.relative_to(ROOT).as_posix()
            )
    _write_json_atomic(status_path, published)


def _queue_status(
    *,
    host: str,
    classification: str,
    active: Mapping[str, OwnedWorker],
    queued: list[str],
    complete: list[str],
    failed: Mapping[str, int],
    attempt_generation: int,
    blocking_reasons: list[str] | None = None,
    paused: Mapping[str, Mapping[str, Any]] | None = None,
    deferred: Mapping[str, Mapping[str, Any]] | None = None,
    job_snapshot: JobMemorySnapshot | None = None,
    memory_decision: MemoryDecision | None = None,
    reservation_ledger: ResourceReservationLedger | None = None,
    queued_wait_reasons: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    disk_free, memory_snapshot = _resource_snapshot()
    if memory_decision is None:
        memory_decision = evaluate_memory(
            stage="supervisor_heartbeat",
            estimated_allocation_bytes=0,
            snapshot=memory_snapshot,
            purpose="observation",
        )
    supervisor_identity = process_identity(os.getpid())
    supervisor_memory = process_memory_snapshot(os.getpid())
    return {
        "run_id": RUN_ID,
        "host_role": host,
        "hostname": socket.gethostname(),
        "classification": classification,
        "supervisor_pid": os.getpid(),
        "supervisor_process_creation_time_utc": (
            supervisor_identity.creation_time_utc
        ),
        "attempt_generation": attempt_generation,
        "active_workers": [
            {
                "family": family,
                "pid": worker.process.pid,
                "process_creation_time_utc": worker.process_creation_time_utc,
                "attempt_generation": attempt_generation,
                "memory": process_memory_snapshot(worker.process.pid).payload(),
            }
            for family, worker in sorted(active.items())
        ],
        "queued_families": queued,
        "queued_wait_reasons": {
            family: reason
            for family, reason in dict(queued_wait_reasons or {}).items()
            if family in queued
        },
        "complete_families": complete,
        "failed_families": dict(failed),
        "resource_paused_families": dict(paused or {}),
        "deferred_resource_families": dict(deferred or {}),
        "capacity_only_deferrals": sorted(dict(deferred or {})),
        "actual_host_pressure": bool(
            actual_host_pressure_reasons(memory_snapshot, job_snapshot)
        ),
        "actual_host_pressure_reasons": list(
            actual_host_pressure_reasons(memory_snapshot, job_snapshot)
        ),
        "blocking_reasons": blocking_reasons or [],
        "ready": not failed and not (paused or {}) and not (blocking_reasons or []),
        "maximum_model_workers": MAX_DELL_MODEL_WORKERS if host == "dell" else 1,
        "disk_free_bytes": disk_free,
        "ram_available_bytes": memory_snapshot.available_physical_bytes,
        "commit_headroom_bytes": memory_snapshot.commit_headroom_bytes,
        "commit_limit_bytes": memory_snapshot.commit_limit_bytes,
        "committed_bytes": memory_snapshot.committed_bytes,
        "peak_committed_bytes": memory_snapshot.peak_committed_bytes,
        "memory_snapshot_source": memory_snapshot.source,
        "supervisor_memory": supervisor_memory.payload(),
        "worker_job": job_snapshot.payload() if job_snapshot is not None else None,
        "worker_reservations": (
            reservation_ledger.status_payload()
            if reservation_ledger is not None
            else None
        ),
        "resource_policy": recovery_policy_payload(),
        "memory_pressure_safe": memory_decision.safe,
        "memory_admission_allowed": memory_decision.admission_allowed,
        "memory_pressure_level": memory_decision.pressure_level,
        "memory_pressure_warnings": list(memory_decision.warning_reasons),
        "heartbeat_utc": _utc_now(),
        "paper_orders": 0,
        "live_orders": 0,
    }


def _worker_identity(family: str, process: subprocess.Popen[Any]) -> OwnedWorker:
    identity = process_identity(process.pid)
    if not identity.alive or not identity.creation_time_utc:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
        raise RuntimeError(f"WORKER_PROCESS_IDENTITY_UNAVAILABLE:{family}")
    if not process_identity_matches(
        identity,
        expected_creation_time=identity.creation_time_utc,
        required_command_fragments=(
            "ds24_clean_v2_family_worker.py",
            "--family",
            family,
        ),
    ):
        process.terminate()
        raise RuntimeError(f"WORKER_PROCESS_IDENTITY_MISMATCH:{family}")
    return OwnedWorker(
        process=process,
        process_creation_time_utc=identity.creation_time_utc,
    )


def _record_supervisor_resource_pause(
    *,
    family: str,
    host: str,
    attempt_generation: int,
    decision: MemoryDecision,
) -> dict[str, Any]:
    family_root = RUN_ROOT / f"family={family}"
    state_path = family_root / "resume_state.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    paused_at = _utc_now()
    evidence = {
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "attempt_generation": attempt_generation,
        "classification": "DS24_CLEAN_V2_PAUSED_RESOURCE_PRESSURE",
        "terminal_state": "PAUSED_RESOURCE_PRESSURE",
        "paused_at_utc": paused_at,
        "resource_decision": decision.payload(),
        "completed_refits": list(state.get("completed_refits", [])),
        "metrics_cursor": int(state.get("metrics_cursor", 0) or 0),
        "automatic_retry": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    _write_json_atomic(
        family_root / f"resource_pause_attempt={attempt_generation}.json",
        evidence,
    )
    state_generation = int(state.get("attempt_generation", 0) or 0)
    if state_generation <= attempt_generation and state.get("terminal_state") != "COMPLETE":
        _write_json_atomic(
            state_path,
            {
                **state,
                "run_id": RUN_ID,
                "family": family,
                "owner_host": host,
                "attempt_generation": attempt_generation,
                "terminal_state": "PAUSED_RESOURCE_PRESSURE",
                "resource_pause_classification": evidence["classification"],
                "resource_pause_timestamp": paused_at,
                "resource_decision": evidence["resource_decision"],
                "completed_refits": evidence["completed_refits"],
                "metrics_cursor": evidence["metrics_cursor"],
                "heartbeat_utc": paused_at,
                "paper_orders": 0,
                "live_orders": 0,
            },
        )
    return evidence


def _terminate_owned_workers(
    active: Mapping[str, OwnedWorker],
    *,
    host: str,
    attempt_generation: int,
    decision: MemoryDecision | None = None,
    containment: WorkerContainment | None = None,
) -> dict[str, dict[str, Any]]:
    verified: list[tuple[str, OwnedWorker]] = []
    for family, worker in active.items():
        if worker.process.poll() is not None:
            continue
        if not _pid_matches(
            worker.process.pid,
            ("ds24_clean_v2_family_worker.py", "--family", family),
            expected_creation_time=worker.process_creation_time_utc,
        ):
            raise RuntimeError(f"REFUSING_TO_TERMINATE_UNVERIFIED_WORKER:{family}")
        if containment is not None and not containment.contains_pid(
            worker.process.pid
        ):
            raise RuntimeError(
                f"REFUSING_TO_TERMINATE_WORKER_OUTSIDE_OWNED_JOB:{family}"
            )
        verified.append((family, worker))
    for _family, worker in verified:
        worker.process.terminate()
    deadline = time.monotonic() + 30.0
    for _family, worker in verified:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            worker.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait(timeout=10.0)
    paused: dict[str, dict[str, Any]] = {}
    if decision is not None:
        for family, _worker in verified:
            paused[family] = _record_supervisor_resource_pause(
                family=family,
                host=host,
                attempt_generation=attempt_generation,
                decision=decision,
            )
    return paused


def _pressure_drain_target(
    active: Mapping[str, OwnedWorker],
    reservation_ledger: ResourceReservationLedger,
) -> dict[str, OwnedWorker]:
    """Select one high-impact worker for non-emergency pressure drainage."""

    reservation_rows = reservation_ledger.status_payload().get("workers", [])
    scores: dict[str, int] = {}
    for row in reservation_rows:
        owner = row.get("owner") or {}
        family = str(owner.get("family", ""))
        current = row.get("current_memory") or {}
        outstanding = row.get("outstanding") or {}
        scores[family] = int(current.get("private_commit_bytes") or 0) + int(
            outstanding.get("commit_bytes") or 0
        )
    family = max(active, key=lambda candidate: (scores.get(candidate, 0), candidate))
    return {family: active[family]}


def _lease_payload(
    *,
    host: str,
    admission_token: str,
    attempt_generation: int,
    identity: ProcessIdentity,
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "pid": identity.pid,
        "process_creation_time_utc": identity.creation_time_utc,
        "hostname": socket.gethostname(),
        "host_role": host,
        "started_at_utc": _utc_now(),
        "generation": attempt_generation,
        "adopted_r40_state": False,
        "admission_token": admission_token,
    }


def _lease_matches_identity(
    lease: Mapping[str, Any], identity: ProcessIdentity
) -> bool:
    return bool(
        int(lease.get("pid", 0) or 0) == identity.pid
        and same_process_creation_time(
            lease.get("process_creation_time_utc"), identity.creation_time_utc
        )
    )


def _release_owned_lease(path: Path, identity: ProcessIdentity) -> None:
    if not path.is_file():
        return
    lease = _read_json(path)
    if _lease_matches_identity(lease, identity):
        path.unlink()


def _await_parent_startup_record(
    *,
    host: str,
    admission_token: str,
    attempt_generation: int,
    timeout_seconds: float = PARENT_STARTUP_RECORD_TIMEOUT_SECONDS,
) -> ProcessIdentity:
    """Wait until the launcher has durably published this child's ownership."""

    identity = process_identity(os.getpid())
    if not identity.alive or not identity.creation_time_utc:
        raise RuntimeError("SUPERVISOR_PROCESS_IDENTITY_UNAVAILABLE")
    lease_path = _host_path("supervisor_lease", host)
    status_path = _host_path("supervisor_status", host)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        lease = _read_json(lease_path) if lease_path.is_file() else {}
        status = _read_json(status_path) if status_path.is_file() else {}
        lease_matches = bool(
            _lease_matches_identity(lease, identity)
            and int(lease.get("generation", 0) or 0) == attempt_generation
            and lease.get("admission_token") == admission_token
        )
        status_matches = bool(
            int(status.get("attempt_generation", 0) or 0)
            == attempt_generation
            and int(status.get("supervisor_pid", 0) or 0) == identity.pid
            and same_process_creation_time(
                status.get("supervisor_process_creation_time_utc"),
                identity.creation_time_utc,
            )
        )
        if lease_matches and status_matches:
            return identity
        time.sleep(0.05)
    raise RuntimeError("PARENT_STARTUP_OWNERSHIP_RECORD_TIMEOUT")


def _record_unhandled_supervisor_exit(
    *,
    host: str,
    attempt_generation: int,
    error: BaseException,
) -> None:
    """Publish a generation-bound child failure without blaming a family."""

    status_path = _host_path("supervisor_status", host)
    current = _read_json(status_path) if status_path.is_file() else {}
    current_generation = int(current.get("attempt_generation", 0) or 0)
    identity = process_identity(os.getpid())
    if current_generation > attempt_generation:
        return
    if current_generation == attempt_generation:
        recorded_pid = int(current.get("supervisor_pid", 0) or 0)
        recorded_creation = current.get("supervisor_process_creation_time_utc")
        if recorded_pid and (
            recorded_pid != identity.pid
            or not same_process_creation_time(
                recorded_creation, identity.creation_time_utc
            )
        ):
            return
    blocking_reasons = list(current.get("blocking_reasons") or [])
    if "SUPERVISOR_UNHANDLED_EXCEPTION" not in blocking_reasons:
        blocking_reasons.append("SUPERVISOR_UNHANDLED_EXCEPTION")
    _write_json_atomic(
        status_path,
        {
            **current,
            "run_id": RUN_ID,
            "host_role": host,
            "classification": (
                "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_EXIT_FAILED_CLOSED"
            ),
            "attempt_generation": attempt_generation,
            "supervisor_pid": identity.pid,
            "supervisor_process_creation_time_utc": identity.creation_time_utc,
            "supervisor_identity_verified": True,
            "active_workers": [],
            "failed_families": dict(current.get("failed_families") or {}),
            "blocking_reasons": blocking_reasons,
            "supervisor_exit_error_type": type(error).__name__,
            "supervisor_exit_error": str(error),
            "automatic_retry": False,
            "ready": False,
            "heartbeat_utc": _utc_now(),
            "paper_orders": 0,
            "live_orders": 0,
        },
    )
    _release_owned_lease(_host_path("supervisor_lease", host), identity)


def _run_queue_entrypoint(
    host: str,
    admission_token: str,
    *,
    attempt_generation: int,
) -> int:
    """Own detached-child startup evidence and fail closed on unexpected exit."""

    try:
        _await_parent_startup_record(
            host=host,
            admission_token=admission_token,
            attempt_generation=attempt_generation,
        )
        return run_queue(
            host,
            admission_token,
            attempt_generation=attempt_generation,
        )
    except Exception as exc:
        traceback.print_exc()
        _record_unhandled_supervisor_exit(
            host=host,
            attempt_generation=attempt_generation,
            error=exc,
        )
        return 5


def run_queue(
    host: str,
    admission_token: str,
    *,
    attempt_generation: int | None = None,
) -> int:
    admission = _validated_runtime_admission(host)
    if admission["admission_token"] != admission_token:
        raise RuntimeError("SUPERVISOR_ADMISSION_TOKEN_MISMATCH")
    lease_path = _host_path("supervisor_lease", host)
    status_path = _host_path("supervisor_status", host)
    stop_path = _host_path("stop_request", host)
    if stop_path.exists():
        stop_path.unlink()
    attempt_generation = attempt_generation or time.time_ns()
    supervisor_identity = process_identity(os.getpid())
    if not supervisor_identity.alive or not supervisor_identity.creation_time_utc:
        raise RuntimeError("SUPERVISOR_PROCESS_IDENTITY_UNAVAILABLE")
    if lease_path.is_file():
        existing_lease = _read_json(lease_path)
        if not _lease_matches_identity(existing_lease, supervisor_identity):
            raise RuntimeError("SUPERVISOR_LEASE_OWNER_CHANGED_BEFORE_QUEUE_START")
    _write_json_atomic(
        lease_path,
        _lease_payload(
            host=host,
            admission_token=admission_token,
            attempt_generation=attempt_generation,
            identity=supervisor_identity,
        ),
    )
    tournament = load_contract("tournament_contract.json")
    commands = tournament["worker_commands"][host]
    queue = _launch_families(host)
    limit = int(admission["maximum_model_workers"])
    expected_limit = MAX_DELL_MODEL_WORKERS if host == "dell" else 1
    if limit != expected_limit:
        raise RuntimeError("SUPERVISOR_ADMISSION_WORKER_LIMIT_MISMATCH")
    active_worker_limit = (
        min(limit, RESOURCE_POLICY.dell_canary_active_worker_limit)
        if host == "dell"
        else limit
    )
    active: dict[str, OwnedWorker] = {}
    failed: dict[str, int] = {}
    paused: dict[str, dict[str, Any]] = {}
    deferred: dict[str, dict[str, Any]] = {}
    deferred_watches: dict[str, DeferredResourceFamily] = {}
    queued_wait_reasons: dict[str, dict[str, Any]] = {}
    containment: WorkerContainment | None = None
    reservation_ledger: ResourceReservationLedger | None = None
    reservation_root = (
        RUN_ROOT / "resource_reservations" / host / f"attempt={attempt_generation}"
    )
    pressure_started_monotonic: float | None = None
    log_root = RUN_ROOT / "logs" / host
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        try:
            containment = create_worker_containment(host=host)
            initial_job_snapshot = containment.snapshot()
            if os.name == "nt" and (
                initial_job_snapshot is None
                or not initial_job_snapshot.installation_verified
                or containment.contains_pid(os.getpid())
            ):
                raise WorkerContainmentError(
                    "WINDOWS_JOB_LIMIT_OR_SUPERVISOR_SEPARATION_NOT_VERIFIED"
                )
            reservation_ledger = ResourceReservationLedger(
                maximum_workers=limit,
                job_snapshot_provider=containment.snapshot,
            )
            _write_json_atomic(
                status_path,
                _queue_status(
                    host=host,
                    classification=(
                        "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_RUNTIME_INITIALIZED"
                    ),
                    active={},
                    queued=queue,
                    complete=[],
                    failed={},
                    attempt_generation=attempt_generation,
                    job_snapshot=initial_job_snapshot,
                    reservation_ledger=reservation_ledger,
                ),
            )
        except WorkerContainmentError as exc:
            _write_json_atomic(
                status_path,
                _queue_status(
                    host=host,
                    classification=(
                        "DS24_CLEAN_V2_TOURNAMENT_BLOCKED_WORKER_CONTAINMENT"
                    ),
                    active={},
                    queued=queue,
                    complete=[],
                    failed={},
                    attempt_generation=attempt_generation,
                    blocking_reasons=[f"WORKER_CONTAINMENT_UNAVAILABLE:{exc}"],
                ),
            )
            return 4
        while True:
            if containment is None or reservation_ledger is None:
                raise RuntimeError("WORKER_CONTAINMENT_NOT_INITIALIZED")
            service_file_reservation_requests(reservation_root, reservation_ledger)
            reservation_ledger.reconcile_dead_workers()
            job_snapshot = containment.snapshot()
            complete = [
                family
                for family in queue
                if _family_state(family).get("terminal_state") == "COMPLETE"
            ]
            for family, worker in list(active.items()):
                exit_code = worker.process.poll()
                if exit_code is None:
                    continue
                active.pop(family)
                _release_worker_reservation(
                    reservation_ledger,
                    family=family,
                    worker=worker,
                    attempt_generation=attempt_generation,
                )
                exit_job_snapshot = containment.snapshot()
                disposition = _worker_exit_disposition(
                    int(exit_code), exit_job_snapshot
                )
                if disposition == "DEFERRED_RESOURCE_CAPACITY":
                    evidence = _family_state(family)
                    deferred[family] = evidence
                    deferred_watches[family] = DeferredResourceFamily(
                        family=family,
                        deferred_at_monotonic=time.monotonic(),
                        capacity_before=reservation_ledger.capacity_snapshot(),
                        evidence=evidence,
                    )
                    queued_wait_reasons[family] = {
                        "classification": LOCAL_CAPACITY_DEFERRAL,
                        "blocking_reasons": list(
                            (evidence.get("resource_decision") or {}).get(
                                "blocking_reasons", []
                            )
                        ),
                        "readmission": (
                            "COOLDOWN_AND_MATERIAL_CAPACITY_CHANGE_REQUIRED"
                        ),
                        "resource_policy": recovery_policy_payload(),
                    }
                    _move_family_to_queue_back(queue, family)
                elif disposition == "RESOURCE_LIMIT_NO_AUTOMATIC_RETRY" and (
                    exit_code == RESOURCE_PRESSURE_EXIT_CODE
                ):
                    paused[family] = _family_state(family)
                elif disposition == "RESOURCE_LIMIT_NO_AUTOMATIC_RETRY":
                    paused[family] = _record_supervisor_resource_pause(
                        family=family,
                        host=host,
                        attempt_generation=attempt_generation,
                        decision=allocation_failure_decision(
                            stage=f"windows_job_commit_limit:{family}",
                            reason="WINDOWS_JOB_AGGREGATE_COMMIT_LIMIT_REACHED",
                        ),
                    )
                elif disposition == "FAILED_CLOSED_NO_AUTOMATIC_RETRY":
                    _record_unexpected_worker_exit(
                        family=family,
                        host=host,
                        attempt_generation=attempt_generation,
                        worker=worker,
                        exit_code=int(exit_code),
                        job_snapshot=exit_job_snapshot,
                    )
                    failed[family] = int(exit_code)
            if paused:
                if active:
                    _disk_free, current_memory = _resource_snapshot()
                    current_job = containment.snapshot()
                    current_pressure = _actual_pressure_decision(
                        stage="supervisor_paused_worker_pressure",
                        memory_snapshot=current_memory,
                        job_snapshot=current_job,
                    )
                    if current_pressure.emergency or (
                        current_job is not None
                        and current_job.limit_violation_detected
                    ):
                        terminated = _terminate_owned_workers(
                            active,
                            host=host,
                            attempt_generation=attempt_generation,
                            decision=current_pressure,
                            containment=containment,
                        )
                        paused.update(terminated)
                        for active_family, active_worker in list(active.items()):
                            _release_worker_reservation(
                                reservation_ledger,
                                family=active_family,
                                worker=active_worker,
                                attempt_generation=attempt_generation,
                            )
                            active.pop(active_family, None)
                        continue
                    _write_json_atomic(
                        status_path,
                        _queue_status(
                            host=host,
                            classification=(
                                "DS24_CLEAN_V2_TOURNAMENT_DRAINING_AFTER_"
                                "RESOURCE_PRESSURE"
                            ),
                            active=active,
                            queued=[
                                family
                                for family in queue
                                if family not in complete
                                and family not in active
                                and family not in paused
                            ],
                            complete=complete,
                            failed={},
                            paused=paused,
                            deferred=deferred,
                            attempt_generation=attempt_generation,
                            blocking_reasons=["NEW_ADMISSIONS_STOPPED_RESOURCE_PRESSURE"],
                            job_snapshot=containment.snapshot(),
                            reservation_ledger=reservation_ledger,
                            queued_wait_reasons=queued_wait_reasons,
                        ),
                    )
                    time.sleep(QUEUE_POLL_SECONDS)
                    continue
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_PAUSED_RESOURCE_PRESSURE",
                        active={},
                        queued=[
                            family
                            for family in queue
                            if family not in complete and family not in paused
                        ],
                        complete=complete,
                        failed={},
                        paused=paused,
                        deferred=deferred,
                        attempt_generation=attempt_generation,
                        blocking_reasons=["UNSAFE_RUNTIME_MEMORY_PRESSURE"],
                        job_snapshot=containment.snapshot(),
                        reservation_ledger=reservation_ledger,
                        queued_wait_reasons=queued_wait_reasons,
                    ),
                )
                return 3
            if failed:
                _terminate_owned_workers(
                    active,
                    host=host,
                    attempt_generation=attempt_generation,
                    containment=containment,
                )
                for family, worker in list(active.items()):
                    _release_worker_reservation(
                        reservation_ledger,
                        family=family,
                        worker=worker,
                        attempt_generation=attempt_generation,
                    )
                active.clear()
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED",
                        active={},
                        queued=[
                            family
                            for family in queue
                            if family not in active
                            and family not in complete
                            and family not in failed
                        ],
                        complete=complete,
                        failed=failed,
                        attempt_generation=attempt_generation,
                        blocking_reasons=["WORKER_EXIT_NONZERO"],
                        job_snapshot=containment.snapshot(),
                        reservation_ledger=reservation_ledger,
                        deferred=deferred,
                    ),
                )
                return 2
            if stop_path.is_file():
                _terminate_owned_workers(
                    active,
                    host=host,
                    attempt_generation=attempt_generation,
                    containment=containment,
                )
                for family, worker in list(active.items()):
                    _release_worker_reservation(
                        reservation_ledger,
                        family=family,
                        worker=worker,
                        attempt_generation=attempt_generation,
                    )
                active.clear()
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE",
                        active={},
                        queued=[family for family in queue if family not in complete],
                        complete=complete,
                        failed={},
                        attempt_generation=attempt_generation,
                        job_snapshot=containment.snapshot(),
                        reservation_ledger=reservation_ledger,
                        queued_wait_reasons=queued_wait_reasons,
                        deferred=deferred,
                    ),
                )
                stop_path.unlink(missing_ok=True)
                return 0
            if active:
                _disk_free, memory_snapshot = _resource_snapshot()
                runtime_decision = _actual_pressure_decision(
                    stage="supervisor_runtime_pressure_heartbeat",
                    memory_snapshot=memory_snapshot,
                    job_snapshot=containment.snapshot(),
                )
                reservation_runtime = reservation_ledger.observe(
                    stage="supervisor_runtime_reserved_headroom",
                    purpose="allocation",
                )
                if (
                    not reservation_runtime.granted
                    and reservation_runtime.classification
                    == HOST_RESOURCE_PRESSURE
                ):
                    runtime_decision = MemoryDecision(
                        **{
                            **runtime_decision.__dict__,
                            "safe": False,
                            "blocking_reasons": tuple(
                                dict.fromkeys(
                                    (*runtime_decision.blocking_reasons,
                                     *reservation_runtime.blocking_reasons)
                                )
                            ),
                        }
                    )
                job_snapshot = containment.snapshot()
                job_limit_violation = bool(
                    job_snapshot is not None
                    and job_snapshot.limit_violation_detected
                )
                if not runtime_decision.safe or job_limit_violation:
                    if pressure_started_monotonic is None:
                        pressure_started_monotonic = time.monotonic()
                    pressure_seconds = (
                        time.monotonic() - pressure_started_monotonic
                    )
                    must_contain = (
                        runtime_decision.emergency
                        or job_limit_violation
                        or pressure_seconds
                        >= RESOURCE_POLICY.pressure_checkpoint_grace_seconds
                    )
                    if not must_contain:
                        _write_json_atomic(
                            status_path,
                            _queue_status(
                                host=host,
                                classification=(
                                    "DS24_CLEAN_V2_TOURNAMENT_"
                                    "CHECKPOINTING_RESOURCE_PRESSURE"
                                ),
                                active=active,
                                queued=[
                                    family
                                    for family in queue
                                    if family not in complete
                                    and family not in active
                                ],
                                complete=complete,
                                failed={},
                                attempt_generation=attempt_generation,
                                blocking_reasons=list(
                                    runtime_decision.blocking_reasons
                                )
                                or ["WINDOWS_JOB_AGGREGATE_COMMIT_LIMIT_REACHED"],
                                job_snapshot=job_snapshot,
                                memory_decision=runtime_decision,
                                reservation_ledger=reservation_ledger,
                                queued_wait_reasons=queued_wait_reasons,
                                deferred=deferred,
                            ),
                        )
                        time.sleep(QUEUE_POLL_SECONDS)
                        continue
                    containment_decision = runtime_decision
                    if job_limit_violation:
                        containment_decision = allocation_failure_decision(
                            stage="windows_job_commit_limit",
                            reason="WINDOWS_JOB_AGGREGATE_COMMIT_LIMIT_REACHED",
                        )
                    drain_targets = (
                        dict(active)
                        if runtime_decision.emergency or job_limit_violation
                        else _pressure_drain_target(active, reservation_ledger)
                    )
                    paused.update(
                        _terminate_owned_workers(
                            drain_targets,
                            host=host,
                            attempt_generation=attempt_generation,
                            decision=containment_decision,
                            containment=containment,
                        )
                    )
                    for family, worker in drain_targets.items():
                        active.pop(family, None)
                        _release_worker_reservation(
                            reservation_ledger,
                            family=family,
                            worker=worker,
                            attempt_generation=attempt_generation,
                        )
                    if active:
                        _write_json_atomic(
                            status_path,
                            _queue_status(
                                host=host,
                                classification=(
                                    "DS24_CLEAN_V2_TOURNAMENT_DRAINING_AFTER_"
                                    "RESOURCE_PRESSURE"
                                ),
                                active=active,
                                queued=[
                                    family
                                    for family in queue
                                    if family not in complete
                                    and family not in active
                                    and family not in paused
                                ],
                                complete=complete,
                                failed={},
                                paused=paused,
                                attempt_generation=attempt_generation,
                                blocking_reasons=list(
                                    containment_decision.blocking_reasons
                                ),
                                job_snapshot=containment.snapshot(),
                                memory_decision=runtime_decision,
                                reservation_ledger=reservation_ledger,
                                queued_wait_reasons=queued_wait_reasons,
                                deferred=deferred,
                            ),
                        )
                        time.sleep(QUEUE_POLL_SECONDS)
                        continue
                    _write_json_atomic(
                        status_path,
                        _queue_status(
                            host=host,
                            classification="DS24_CLEAN_V2_TOURNAMENT_PAUSED_RESOURCE_PRESSURE",
                            active={},
                            queued=[
                                family
                                for family in queue
                                if family not in complete and family not in paused
                            ],
                            complete=complete,
                            failed={},
                            paused=paused,
                            attempt_generation=attempt_generation,
                            blocking_reasons=list(
                                containment_decision.blocking_reasons
                            ),
                            job_snapshot=containment.snapshot(),
                            memory_decision=runtime_decision,
                            reservation_ledger=reservation_ledger,
                            queued_wait_reasons=queued_wait_reasons,
                            deferred=deferred,
                        ),
                    )
                    return 3
                pressure_started_monotonic = None
            if deferred_watches:
                current_capacity = reservation_ledger.capacity_snapshot()
                now_monotonic = time.monotonic()
                for family, watch in list(deferred_watches.items()):
                    if capacity_change_is_material(
                        watch.capacity_before,
                        current_capacity,
                        elapsed_seconds=(
                            now_monotonic - watch.deferred_at_monotonic
                        ),
                    ):
                        deferred_watches.pop(family, None)
                        deferred.pop(family, None)
                        queued_wait_reasons.pop(family, None)
            pending = [
                family
                for family in queue
                if family not in active
                and family not in complete
                and family not in failed
                and family not in paused
                and family not in deferred
            ]
            if _primary_family_capacity_deferred(host, deferred, complete):
                for waiting_family in pending:
                    queued_wait_reasons[waiting_family] = {
                        "classification": "PINNED_PRIMARY_CAPACITY_WAIT",
                        "blocking_reasons": [
                            "PINNED_PRIMARY_CAPACITY_WAIT"
                        ],
                        "primary_family": PRIMARY_DELL_FAMILY,
                        "resource_policy": recovery_policy_payload(),
                    }
                pending = []
            bypasses_this_pass = 0
            for pending_index, family in enumerate(list(pending)):
                if len(active) >= active_worker_limit:
                    for waiting_family in pending[pending_index:]:
                        queued_wait_reasons[waiting_family] = {
                            "classification": "CANARY_WORKER_LIMIT_REACHED",
                            "blocking_reasons": ["CANARY_WORKER_LIMIT_REACHED"],
                            "active_worker_limit": active_worker_limit,
                            "maximum_model_workers": limit,
                            "resource_policy": recovery_policy_payload(),
                        }
                    break
                unbound_owner = WorkerReservationOwner(
                    family=family,
                    attempt_generation=attempt_generation,
                )
                reservation_decision = reservation_ledger.try_admit(
                    unbound_owner, estimate_worker_peak_allocation(family)
                )
                if not reservation_decision.granted:
                    decision_payload = reservation_decision.payload()
                    single_worker_block = bool(
                        not active
                        and reservation_decision.classification
                        == LOCAL_CAPACITY_DEFERRAL
                    )
                    if single_worker_block:
                        decision_payload["classification"] = (
                            SINGLE_WORKER_CAPACITY_BLOCK
                        )
                    queued_wait_reasons[family] = decision_payload
                    if reservation_decision.classification == LOCAL_CAPACITY_DEFERRAL:
                        evidence = _record_supervisor_capacity_deferral(
                            family=family,
                            host=host,
                            attempt_generation=attempt_generation,
                            resource_decision=decision_payload,
                            single_worker_block=single_worker_block,
                        )
                        deferred[family] = evidence
                        deferred_watches[family] = DeferredResourceFamily(
                            family=family,
                            deferred_at_monotonic=time.monotonic(),
                            capacity_before=reservation_ledger.capacity_snapshot(),
                            evidence=evidence,
                        )
                        _move_family_to_queue_back(queue, family)
                    bypasses_this_pass += 1
                    if (
                        bypasses_this_pass
                        > RESOURCE_POLICY.maximum_admission_bypasses
                    ):
                        break
                    continue
                command = list(commands[family])
                command.extend(
                    ["--resume-generation", str(attempt_generation)]
                )
                log_path = log_root / f"{family}.log"
                worker_environment = os.environ.copy()
                worker_environment["DS24_CLEAN_V2_ADMISSION_TOKEN"] = admission_token
                worker_environment["DS24_CLEAN_V2_RESOURCE_POLICY_ID"] = (
                    RESOURCE_POLICY.policy_id
                )
                worker_environment["DS24_CLEAN_V2_RESERVATION_ROOT"] = str(
                    reservation_root
                )
                if host == "dell":
                    worker_environment.update(DELL_WORKER_THREAD_ENVIRONMENT)
                bound_owner: list[WorkerReservationOwner] = []

                def bind_reservation(pid: int) -> None:
                    identity = process_identity(pid)
                    if not identity.alive or not identity.creation_time_utc:
                        raise WorkerContainmentError(
                            f"WORKER_PROCESS_IDENTITY_UNAVAILABLE:{family}"
                        )
                    owner = WorkerReservationOwner(
                        family=family,
                        attempt_generation=attempt_generation,
                        pid=pid,
                        process_creation_time_utc=identity.creation_time_utc,
                    )
                    reservation_ledger.bind_worker(
                        str(reservation_decision.worker_reservation_id), owner
                    )
                    bound_owner.append(owner)

                try:
                    with log_path.open("a", encoding="utf-8") as log_handle:
                        process = containment.launch(
                            command,
                            cwd=ROOT,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            env=worker_environment,
                            before_resume=bind_reservation,
                        )
                    owned_worker = _worker_identity(family, process)
                    if (
                        not bound_owner
                        or not same_process_creation_time(
                            bound_owner[0].process_creation_time_utc,
                            owned_worker.process_creation_time_utc,
                        )
                    ):
                        raise WorkerContainmentError(
                            f"WORKER_RESERVATION_POST_LAUNCH_IDENTITY_MISMATCH:{family}"
                        )
                except Exception:
                    owner_to_release = (
                        bound_owner[0] if bound_owner else unbound_owner
                    )
                    reservation_ledger.release_worker(
                        owner_to_release, allow_unbound=not bound_owner
                    )
                    raise
                active[family] = owned_worker
                queued_wait_reasons.pop(family, None)
                pending.remove(family)
            if not active and not pending and not deferred:
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_COMPLETE",
                        active={},
                        queued=[],
                        complete=complete,
                        failed={},
                        attempt_generation=attempt_generation,
                        job_snapshot=containment.snapshot(),
                        reservation_ledger=reservation_ledger,
                    ),
                )
                return 0
            status_classification = "DS24_CLEAN_V2_TOURNAMENT_RUNNING"
            if not active and deferred:
                status_classification = (
                    "DS24_CLEAN_V2_TOURNAMENT_WAITING_RESOURCE_CAPACITY"
                )
            elif not active and any(
                (reason or {}).get("classification") == HOST_RESOURCE_PRESSURE
                for reason in queued_wait_reasons.values()
            ):
                status_classification = (
                    "DS24_CLEAN_V2_TOURNAMENT_WAITING_HOST_RESOURCE_PRESSURE"
                )
            _write_json_atomic(
                status_path,
                _queue_status(
                    host=host,
                    classification=status_classification,
                    active=active,
                    queued=pending,
                    complete=complete,
                    failed={},
                    attempt_generation=attempt_generation,
                    job_snapshot=containment.snapshot(),
                    reservation_ledger=reservation_ledger,
                    queued_wait_reasons=queued_wait_reasons,
                    deferred=deferred,
                ),
            )
            time.sleep(QUEUE_POLL_SECONDS)
    finally:
        try:
            if active:
                try:
                    _terminate_owned_workers(
                        active,
                        host=host,
                        attempt_generation=attempt_generation,
                        containment=containment,
                    )
                finally:
                    if reservation_ledger is not None:
                        for family, worker in list(active.items()):
                            _release_worker_reservation(
                                reservation_ledger,
                                family=family,
                                worker=worker,
                                attempt_generation=attempt_generation,
                            )
                    active.clear()
        finally:
            try:
                if containment is not None:
                    containment.close()
            finally:
                _release_owned_lease(lease_path, supervisor_identity)


def _archive_stale_lease(path: Path, lease: Mapping[str, Any]) -> Path:
    stale_root = RUN_ROOT / "stale_leases"
    stale_root.mkdir(parents=True, exist_ok=True)
    generation = int(lease.get("generation", 0) or 0)
    candidate = stale_root / f"{path.stem}.generation={generation}.json"
    if candidate.exists():
        candidate = stale_root / (
            f"{path.stem}.generation={generation}.recovered={time.time_ns()}.json"
        )
    os.replace(path, candidate)
    return candidate


def _prepare_stopped_runtime_for_preflight(host: str) -> dict[str, Any]:
    """Archive only a verified-dead lease before stopped-state reconciliation."""

    lease_path = _host_path("supervisor_lease", host)
    if not lease_path.is_file():
        reconciled_status = _reconcile_persisted_runtime(host, persist=True)
        return {
            "stopped": True,
            "classification": (
                "NO_SUPERVISOR_LEASE_RUNTIME_STATUS_RECONCILED"
                if reconciled_status
                else "NO_SUPERVISOR_LEASE"
            ),
            "archived_lease_path": None,
        }
    lease = _read_json(lease_path)
    pid = int(lease.get("pid", 0) or 0)
    identity = process_identity(pid)
    required = ("ds24_clean_v2_supervisor.py", "--run-queue", "--host", host)
    command_matches = bool(
        identity.alive
        and all(
            fragment.lower() in identity.command_line.lower()
            for fragment in required
        )
    )
    identity_verified = bool(
        command_matches
        and lease.get("process_creation_time_utc")
        and process_identity_matches(
            identity,
            expected_creation_time=lease.get("process_creation_time_utc"),
            required_command_fragments=required,
        )
    )
    if command_matches:
        return {
            "stopped": False,
            "classification": (
                "LIVE_SUPERVISOR_LEASE_VERIFIED"
                if identity_verified
                else "LIVE_SUPERVISOR_LEASE_IDENTITY_UNVERIFIABLE"
            ),
            "blocking_reason": "STOPPED_PREFLIGHT_REQUIRES_NO_LIVE_SUPERVISOR",
            "identity_verified": identity_verified,
            "pid": pid,
            "process_creation_time_utc": identity.creation_time_utc,
        }
    archived = _archive_stale_lease(lease_path, lease)
    _reconcile_persisted_runtime(host, persist=True)
    return {
        "stopped": True,
        "classification": "STALE_SUPERVISOR_LEASE_ARCHIVED",
        "archived_lease_path": archived.relative_to(ROOT).as_posix(),
        "stale_pid": pid,
        "stale_process_creation_time_utc": lease.get(
            "process_creation_time_utc"
        ),
    }


def launch(host: str) -> dict[str, Any]:
    status_path = _host_path("supervisor_status", host)
    stopped_runtime = _prepare_stopped_runtime_for_preflight(host)
    if not stopped_runtime["stopped"]:
        if stopped_runtime.get("identity_verified"):
            status = _read_json(status_path) if status_path.is_file() else {}
            return {
                **status,
                "launch_idempotent": True,
                "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
            }
        raise RuntimeError("LIVE_SUPERVISOR_LEASE_CREATION_TIME_UNVERIFIABLE")
    admission = _validated_admission(host)
    lease_path = _host_path("supervisor_lease", host)
    stop_path = _host_path("stop_request", host)
    if stop_path.exists():
        stop_path.unlink()
    attempt_generation = time.time_ns()
    command = [
        sys.executable,
        str(SUPERVISOR_SCRIPT),
        "--host",
        host,
        "--run-queue",
        "--admission-token",
        admission["admission_token"],
        "--attempt-generation",
        str(attempt_generation),
    ]
    log_path = RUN_ROOT / "logs" / host / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        kwargs: dict[str, Any] = {
            "cwd": ROOT,
            "stdout": handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)  # noqa: S603
    identity = process_identity(process.pid)
    if not identity.alive or not identity.creation_time_utc:
        process.terminate()
        process.wait(timeout=10.0)
        raise RuntimeError("DETACHED_SUPERVISOR_PROCESS_IDENTITY_UNAVAILABLE")
    _write_json_atomic(
        lease_path,
        _lease_payload(
            host=host,
            admission_token=admission["admission_token"],
            attempt_generation=attempt_generation,
            identity=identity,
        ),
    )
    report = {
        "run_id": RUN_ID,
        "host_role": host,
        "classification": "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_STARTING",
        "supervisor_pid": process.pid,
        "supervisor_process_creation_time_utc": identity.creation_time_utc,
        "attempt_generation": attempt_generation,
        "maximum_model_workers": admission["maximum_model_workers"],
        "resource_policy": admission["resource_policy"],
        "admission_path": _host_path("manual_admission", host)
        .relative_to(ROOT)
        .as_posix(),
        "monitor_command": (
            f"python scripts/local/ds24_clean_v2_monitor.py --host {host}"
        ),
        "ready": True,
        "paper_orders": 0,
        "live_orders": 0,
    }
    _write_json_atomic(status_path, report)
    return report


def stop(host: str) -> dict[str, Any]:
    lease_path = _host_path("supervisor_lease", host)
    if not lease_path.is_file():
        return {
            "run_id": RUN_ID,
            "host_role": host,
            "classification": "DS24_CLEAN_V2_TOURNAMENT_ALREADY_STOPPED",
            "ready": True,
        }
    lease = _read_json(lease_path)
    pid = int(lease.get("pid", 0) or 0)
    if not pid or not _pid_matches(
        pid,
        ("ds24_clean_v2_supervisor.py", "--run-queue", "--host", host),
        expected_creation_time=lease.get("process_creation_time_utc"),
    ):
        raise RuntimeError("REFUSING_TO_SIGNAL_UNVERIFIED_SUPERVISOR_PID")
    stop_path = _host_path("stop_request", host)
    _write_json_atomic(
        stop_path,
        {
            "run_id": RUN_ID,
            "host_role": host,
            "requested_at_utc": _utc_now(),
            "requested_by_pid": os.getpid(),
            "supervisor_pid": pid,
            "supervisor_process_creation_time_utc": lease.get(
                "process_creation_time_utc"
            ),
        },
    )
    return {
        "run_id": RUN_ID,
        "host_role": host,
        "classification": "DS24_CLEAN_V2_SAFE_STOP_REQUESTED",
        "supervisor_pid": pid,
        "stop_request_path": stop_path.relative_to(ROOT).as_posix(),
        "ready": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed DS24 clean-V2 queue supervisor."
    )
    parser.add_argument("--host", choices=("dell", "mac"), default="dell")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--resume", action="store_true")
    action.add_argument("--stop", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--run-queue", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--write-admission", action="store_true")
    parser.add_argument("--admission-token", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-generation", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_queue:
        if not args.admission_token:
            raise RuntimeError("SUPERVISOR_ADMISSION_TOKEN_REQUIRED")
        if args.attempt_generation is None:
            raise RuntimeError("SUPERVISOR_ATTEMPT_GENERATION_REQUIRED")
        return _run_queue_entrypoint(
            args.host,
            args.admission_token,
            attempt_generation=args.attempt_generation,
        )
    if args.status:
        path = _host_path("supervisor_status", args.host)
        report = _reconcile_persisted_runtime(args.host, persist=True)
        if not report:
            report = {
                "run_id": RUN_ID,
                "host_role": args.host,
                "classification": "NOT_STARTED",
                "ready": False,
            }
    elif args.stop:
        report = stop(args.host)
    elif args.launch or args.resume:
        report = launch(args.host)
    else:
        report = preflight(args.host, write_admission=args.write_admission)
        _publish_preflight_status(args.host, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
