from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (
    authority_bundle,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_runtime import (
    MAX_DELL_MODEL_WORKERS,
    REFIT_POLICY_ID,
    RUN_ID,
    validate_ownership,
)


RUN_ROOT = ROOT / "research_runs" / "ds24_clean_v2" / RUN_ID
SUPERVISOR_SCRIPT = Path(__file__).resolve()
QUEUE_POLL_SECONDS = 5.0


def _host_path(stem: str, host: str) -> Path:
    return RUN_ROOT / f"{stem}_{host}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


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
    return [family for family in ordered if family in owned]


def _resource_snapshot() -> tuple[int, int | None]:
    disk_free = int(shutil.disk_usage(ROOT.anchor or ROOT).free)
    try:
        import psutil

        ram_available = int(psutil.virtual_memory().available)
    except ImportError:
        ram_available = None
    return disk_free, ram_available


def preflight(host: str, *, write_admission: bool = False) -> dict[str, Any]:
    feature = load_contract("feature_authority.json")
    target = load_contract("target_contract.json")
    tournament = load_contract("tournament_contract.json")
    models = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    validate_ownership(ownership["hosts"])
    bundle = authority_bundle()
    reasons: list[str] = []

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

    disk_free, ram_available = _resource_snapshot()
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

    feature_hash = sidecar_manifest.get("logical_sha256") or stable_hash(feature)
    if not reasons:
        classification = "DS24_CLEAN_V2_READY_FOR_MANUAL_TOURNAMENT_LAUNCH"
    elif "FINAL_CAUSALITY_DATA_CERTIFICATE_MISSING_OR_FAILED" in reasons:
        classification = "DS24_CLEAN_V2_IMPLEMENTATION_READY_MANUAL_DATA_BUILD_REQUIRED"
    elif "INSUFFICIENT_DISK_FOR_DECLARED_RESERVE" in reasons:
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
        "results_ledger_hash": load_contract("prior_evidence_manifest.json")[
            "workbook"
        ]["captured_xlsx_export_sha256"],
        "refit_policy": REFIT_POLICY_ID,
        "launch_families": launch_families,
        "mac_owned_families": ownership["hosts"]["mac"],
        "maximum_model_workers": MAX_DELL_MODEL_WORKERS if host == "dell" else 1,
        "disk_free_bytes": disk_free,
        "ram_available_bytes": ram_available,
        "obsolete_ds24_processes": legacy_processes,
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
            "refit_policy": REFIT_POLICY_ID,
            "launch_families": launch_families,
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


def _pid_matches(pid: int, required_fragments: tuple[str, ...]) -> bool:
    try:
        import psutil

        command = " ".join(psutil.Process(pid).cmdline()).lower()
    except Exception:
        return False
    return all(fragment.lower() in command for fragment in required_fragments)


def _validated_admission(host: str) -> dict[str, Any]:
    path = _host_path("manual_admission", host)
    if not path.is_file():
        raise RuntimeError("MANUAL_PREFLIGHT_ADMISSION_MISSING")
    admission = _read_json(path)
    token = admission.pop("admission_token", None)
    if token != stable_hash(admission):
        raise RuntimeError("MANUAL_PREFLIGHT_ADMISSION_HASH_MISMATCH")
    report = preflight(host)
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
        "refit_policy": report["refit_policy"],
        "launch_families": report["launch_families"],
    }
    for key, value in checks.items():
        if admission.get(key) != value:
            raise RuntimeError(f"MANUAL_PREFLIGHT_ADMISSION_STALE:{key}")
    return {**admission, "admission_token": token}


def _family_state(family: str) -> dict[str, Any]:
    path = RUN_ROOT / f"family={family}" / "resume_state.json"
    return _read_json(path) if path.is_file() else {}


def _queue_status(
    *,
    host: str,
    classification: str,
    active: Mapping[str, subprocess.Popen[Any]],
    queued: list[str],
    complete: list[str],
    failed: Mapping[str, int],
    blocking_reasons: list[str] | None = None,
) -> dict[str, Any]:
    disk_free, ram_available = _resource_snapshot()
    return {
        "run_id": RUN_ID,
        "host_role": host,
        "hostname": socket.gethostname(),
        "classification": classification,
        "supervisor_pid": os.getpid(),
        "active_workers": [
            {"family": family, "pid": process.pid}
            for family, process in sorted(active.items())
        ],
        "queued_families": queued,
        "complete_families": complete,
        "failed_families": dict(failed),
        "blocking_reasons": blocking_reasons or [],
        "ready": not failed and not (blocking_reasons or []),
        "disk_free_bytes": disk_free,
        "ram_available_bytes": ram_available,
        "heartbeat_utc": _utc_now(),
        "paper_orders": 0,
        "live_orders": 0,
    }


def run_queue(host: str, admission_token: str) -> int:
    admission = _validated_admission(host)
    if admission["admission_token"] != admission_token:
        raise RuntimeError("SUPERVISOR_ADMISSION_TOKEN_MISMATCH")
    lease_path = _host_path("supervisor_lease", host)
    status_path = _host_path("supervisor_status", host)
    stop_path = _host_path("stop_request", host)
    if stop_path.exists():
        stop_path.unlink()
    _write_json_atomic(
        lease_path,
        {
            "run_id": RUN_ID,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "host_role": host,
            "started_at_utc": _utc_now(),
            "generation": int(time.time()),
            "adopted_r40_state": False,
            "admission_token": admission_token,
        },
    )
    tournament = load_contract("tournament_contract.json")
    commands = tournament["worker_commands"][host]
    queue = _launch_families(host)
    limit = MAX_DELL_MODEL_WORKERS if host == "dell" else 1
    active: dict[str, subprocess.Popen[Any]] = {}
    failed: dict[str, int] = {}
    log_root = RUN_ROOT / "logs" / host
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            complete = [
                family
                for family in queue
                if _family_state(family).get("terminal_state") == "COMPLETE"
            ]
            for family, process in list(active.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                active.pop(family)
                if exit_code != 0:
                    failed[family] = int(exit_code)
            if failed:
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED",
                        active=active,
                        queued=[
                            family
                            for family in queue
                            if family not in active and family not in complete
                        ],
                        complete=complete,
                        failed=failed,
                        blocking_reasons=["WORKER_EXIT_NONZERO"],
                    ),
                )
                return 2
            if stop_path.is_file():
                for family, process in active.items():
                    if _pid_matches(
                        process.pid,
                        ("ds24_clean_v2_family_worker.py", "--family", family),
                    ):
                        process.terminate()
                deadline = time.monotonic() + 30.0
                for process in active.values():
                    timeout = max(0.0, deadline - time.monotonic())
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE",
                        active={},
                        queued=[family for family in queue if family not in complete],
                        complete=complete,
                        failed={},
                    ),
                )
                return 0
            pending = [
                family
                for family in queue
                if family not in active and family not in complete
            ]
            while pending and len(active) < limit:
                family = pending.pop(0)
                command = list(commands[family])
                log_path = log_root / f"{family}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                worker_environment = os.environ.copy()
                worker_environment["DS24_CLEAN_V2_ADMISSION_TOKEN"] = admission_token
                process = subprocess.Popen(  # noqa: S603 - frozen static authority
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=worker_environment,
                )
                log_handle.close()
                active[family] = process
            if not active and not pending:
                _write_json_atomic(
                    status_path,
                    _queue_status(
                        host=host,
                        classification="DS24_CLEAN_V2_TOURNAMENT_COMPLETE",
                        active={},
                        queued=[],
                        complete=complete,
                        failed={},
                    ),
                )
                return 0
            _write_json_atomic(
                status_path,
                _queue_status(
                    host=host,
                    classification="DS24_CLEAN_V2_TOURNAMENT_RUNNING",
                    active=active,
                    queued=pending,
                    complete=complete,
                    failed={},
                ),
            )
            time.sleep(QUEUE_POLL_SECONDS)
    finally:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass


def launch(host: str) -> dict[str, Any]:
    admission = _validated_admission(host)
    status_path = _host_path("supervisor_status", host)
    lease_path = _host_path("supervisor_lease", host)
    if lease_path.is_file():
        lease = _read_json(lease_path)
        pid = int(lease.get("pid", 0) or 0)
        if pid and _pid_matches(
            pid, ("ds24_clean_v2_supervisor.py", "--run-queue", "--host", host)
        ):
            status = _read_json(status_path) if status_path.is_file() else {}
            return {
                **status,
                "launch_idempotent": True,
                "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
            }
        raise RuntimeError("STALE_OR_UNVERIFIABLE_SUPERVISOR_LEASE")
    stop_path = _host_path("stop_request", host)
    if stop_path.exists():
        stop_path.unlink()
    command = [
        sys.executable,
        str(SUPERVISOR_SCRIPT),
        "--host",
        host,
        "--run-queue",
        "--admission-token",
        admission["admission_token"],
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
    report = {
        "run_id": RUN_ID,
        "host_role": host,
        "classification": "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_STARTING",
        "supervisor_pid": process.pid,
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
        pid, ("ds24_clean_v2_supervisor.py", "--run-queue", "--host", host)
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
    args = parser.parse_args()

    if args.run_queue:
        if not args.admission_token:
            raise RuntimeError("SUPERVISOR_ADMISSION_TOKEN_REQUIRED")
        return run_queue(args.host, args.admission_token)
    if args.status:
        path = _host_path("supervisor_status", args.host)
        report = _read_json(path) if path.is_file() else {
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
        _write_json_atomic(_host_path("supervisor_status", args.host), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
