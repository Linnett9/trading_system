from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "research_runs/ds24_clean_v2/DS24_CLEAN_V2_TOURNAMENT_R1_20260926"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import load_contract  # noqa: E402
from core.research.ml.ds24.clean_v2_resources import (  # noqa: E402
    actual_host_pressure_reasons,
    evaluate_memory,
    process_memory_snapshot,
    read_json_object_with_retry,
    recovery_policy_payload,
    reconcile_runtime_status_view,
    reservation_payload_is_local_capacity_deferral,
    status_worker_identity_matches,
    system_memory_snapshot,
)
from core.research.ml.ds24.clean_v2_runtime import (  # noqa: E402
    MAX_DELL_MODEL_WORKERS,
)


def _log_runtime_file_retry(event: dict[str, Any]) -> None:
    print(
        json.dumps({**event, "component": "monitor"}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object_with_retry(path, on_retry=_log_runtime_file_retry)


def _matching_failure(
    family_root: Path,
    state: dict[str, Any],
    *,
    host: str,
    current_attempt_generation: int,
) -> dict[str, Any]:
    failure = _read_json(family_root / "worker_failure.json")
    if not failure:
        return {}
    if (
        failure.get("run_id") != state.get("run_id")
        or failure.get("family") != state.get("family")
        or failure.get("host") != host
    ):
        return {}
    state_generation = int(state.get("attempt_generation", 0) or 0)
    failure_generation = int(failure.get("attempt_generation", 0) or 0)
    if failure_generation:
        if failure_generation == current_attempt_generation:
            return failure
        if current_attempt_generation > failure_generation:
            return {}
        return failure if state_generation == failure_generation else {}
    if current_attempt_generation > state_generation:
        return {}
    if state_generation:
        return {}
    if state.get("terminal_state") == "COMPLETE":
        return {}
    failed_at = str(failure.get("failed_at_utc") or "")
    heartbeat = str(state.get("heartbeat_utc") or "")
    return failure if failed_at >= heartbeat else {}


def build_report(
    host: str,
    *,
    repository_root: Path = ROOT,
    run_root: Path = RUN_ROOT,
) -> dict[str, Any]:
    """Build one read-only report from current host-scoped authority."""

    status_path = run_root / f"supervisor_status_{host}.json"
    status = reconcile_runtime_status_view(_read_json(status_path), host=host)
    ownership = load_contract("cross_host_ownership.json")
    admitted_families = set(ownership["hosts"][host])
    current_attempt_generation = int(status.get("attempt_generation", 0) or 0)
    reported_active_workers = [
        row
        for row in status.get("active_workers", [])
        if isinstance(row, dict) and row.get("family") in admitted_families
    ]
    verified_worker_records = [
        row
        for row in reported_active_workers
        if status_worker_identity_matches(row)
    ]
    active_workers = [
        {
            **row,
            "memory": process_memory_snapshot(int(row["pid"])).payload(),
        }
        for row in verified_worker_records
    ]
    rejected_active_worker_records = [
        row for row in reported_active_workers if row not in verified_worker_records
    ]
    active_generations = {
        str(row["family"]): int(
            row.get("attempt_generation", current_attempt_generation) or 0
        )
        for row in active_workers
    }
    queued_families = [
        str(family)
        for family in status.get("queued_families", [])
        if family in admitted_families
    ]
    for row in status.get("stale_worker_records_reconciled", []):
        family = str(row.get("family") or "")
        if family in admitted_families and family not in queued_families:
            queued_families.append(family)
    stale_family_ids = {
        str(row.get("family"))
        for row in status.get("stale_worker_records_reconciled", [])
        if row.get("family")
    }
    complete_families = [
        str(family)
        for family in status.get("complete_families", [])
        if family in admitted_families
    ]
    resource_paused_families = {
        str(family): details
        for family, details in dict(
            status.get("resource_paused_families", {})
        ).items()
        if family in admitted_families
    }
    deferred_resource_families = {
        str(family): details
        for family, details in dict(
            status.get("deferred_resource_families", {})
        ).items()
        if family in admitted_families
    }
    for family, details in list(resource_paused_families.items()):
        decision = details.get("resource_decision") if isinstance(details, dict) else {}
        if isinstance(decision, dict) and reservation_payload_is_local_capacity_deferral(
            decision
        ):
            deferred_resource_families[family] = {
                **details,
                "classification": "DS24_CLEAN_V2_DEFERRED_RESOURCE_CAPACITY",
                "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
                "reconciled_from_legacy_resource_pause": True,
            }
            resource_paused_families.pop(family, None)
    for family in deferred_resource_families:
        if family not in queued_families:
            queued_families.append(family)
    disk = shutil.disk_usage(repository_root.anchor or repository_root)
    rows = []
    for family_path in sorted(run_root.glob("family=*/resume_state.json")):
        state = _read_json(family_path)
        if state.get("owner_host") != host:
            continue
        if state.get("family") not in admitted_families:
            continue
        family = str(state["family"])
        state_generation = int(state.get("attempt_generation", 0) or 0)
        recorded_failure = _read_json(family_path.parent / "worker_failure.json")
        failure = _matching_failure(
            family_path.parent,
            state,
            host=host,
            current_attempt_generation=current_attempt_generation,
        )
        historical_failure = None
        if (
            recorded_failure
            and not failure
            and recorded_failure.get("run_id") == state.get("run_id")
            and recorded_failure.get("family") == state.get("family")
            and recorded_failure.get("host") == host
        ):
            historical_failure = {
                "attempt_generation": recorded_failure.get("attempt_generation"),
                "classification": recorded_failure.get("classification"),
                "error": recorded_failure.get("error"),
                "failed_at_utc": recorded_failure.get("failed_at_utc"),
                "log_path": recorded_failure.get("log_path"),
            }
        terminal_state = state.get("terminal_state")
        effective_generation = state_generation
        error = failure.get("error", state.get("error"))
        failure_timestamp = failure.get(
            "failed_at_utc", state.get("failure_timestamp")
        )
        failure_log_path = failure.get(
            "log_path", state.get("failure_log_path")
        )
        if active_generations.get(family, 0) > state_generation:
            terminal_state = "RUNNING"
            effective_generation = active_generations[family]
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif family in stale_family_ids and not status.get(
            "supervisor_identity_verified", False
        ):
            terminal_state = "RESUMABLE_STALE_PROCESS"
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif family in resource_paused_families:
            terminal_state = "PAUSED_RESOURCE_PRESSURE"
            effective_generation = int(
                resource_paused_families[family].get(
                    "attempt_generation", effective_generation
                )
                or effective_generation
            )
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif family in deferred_resource_families:
            terminal_state = str(
                deferred_resource_families[family].get(
                    "terminal_state", "DEFERRED_RESOURCE_CAPACITY"
                )
            )
            effective_generation = int(
                deferred_resource_families[family].get(
                    "attempt_generation", effective_generation
                )
                or effective_generation
            )
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif family in queued_families and (
            current_attempt_generation >= state_generation
            or status.get("classification")
            == "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE"
        ):
            terminal_state = "QUEUED"
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif failure and terminal_state != "COMPLETE":
            terminal_state = failure.get("terminal_state", "FAILED_CLOSED")
        if (
            terminal_state == "RUNNING"
            and family not in active_generations
            and not status.get("supervisor_identity_verified", False)
        ):
            terminal_state = "RESUMABLE_STALE_PROCESS"
            error = None
            failure_timestamp = None
            failure_log_path = None
        rows.append(
            {
                "family": family,
                "state": terminal_state,
                "attempt_generation": state.get("attempt_generation"),
                "effective_attempt_generation": effective_generation,
                "attempt_id": state.get("attempt_id"),
                "current_historical_date": state.get("latest_scored_decision"),
                "completed_refits": len(state.get("completed_refits", [])),
                "expected_refits": state.get("expected_refits"),
                "score_timestamps": state.get("metrics_cursor", 0),
                "current_rank_ic": state.get("current_rank_ic"),
                "top_n_spread": state.get("top_n_spread"),
                "net_10bps": state.get("net_10bps"),
                "error": error,
                "failure_timestamp": failure_timestamp,
                "failure_log_path": failure_log_path,
                "historical_failure_not_current_attempt": historical_failure,
                "causal_authority_hash": state.get("feature_authority_hash"),
            }
        )
    state_failures = {
        str(row["family"]): 1
        for row in rows
        if row["state"] == "FAILED_CLOSED"
    }
    family_rows = {str(row["family"]): row for row in rows}
    reported_status_failures = {
        str(family): exit_code
        for family, exit_code in dict(status.get("failed_families", {})).items()
        if family in admitted_families
    }
    superseded_status_failures: dict[str, Any] = {}
    status_failures: dict[str, Any] = {}
    for family, exit_code in reported_status_failures.items():
        row = family_rows.get(family) or {}
        evidence_generation = int(
            row.get("effective_attempt_generation", 0) or 0
        )
        # A failed scheduler status is written after the worker state observed by
        # that scheduler.  Only a later attempt can supersede it; a same-attempt
        # RUNNING/QUEUED state is older evidence, not proof of recovery.
        superseded = evidence_generation > current_attempt_generation
        if superseded:
            superseded_status_failures[family] = {
                "exit_code": exit_code,
                "status_attempt_generation": current_attempt_generation,
                "superseding_attempt_generation": evidence_generation,
                "superseding_state": row.get("state"),
            }
        else:
            status_failures[family] = exit_code
    failed_families = {
        **status_failures,
        **state_failures,
    }
    active_family_ids = {str(row["family"]) for row in active_workers}
    queued_families = [
        family
        for family in queued_families
        if family not in failed_families
        and family not in complete_families
        and family not in active_family_ids
    ]
    classification = status.get("classification", "NOT_STARTED")
    if state_failures and not resource_paused_families:
        classification = "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED"
    elif (
        reported_status_failures
        and not failed_families
        and classification == "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED"
    ):
        classification = (
            "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_"
            "STALE_FAILURE_RECONCILED"
        )
    memory_snapshot = system_memory_snapshot()
    memory_decision = evaluate_memory(
        stage="monitor_current_pressure",
        estimated_allocation_bytes=0,
        snapshot=memory_snapshot,
        purpose="observation",
    )
    effective_resource_policy = recovery_policy_payload()
    status_policy_matches = status.get("resource_policy") in (
        None,
        effective_resource_policy,
    )
    supervisor_pid = int(status.get("supervisor_pid", 0) or 0)
    supervisor_creation = status.get("supervisor_process_creation_time_utc")
    supervisor_identity_verified = bool(
        status.get("supervisor_identity_verified", False)
    )
    heartbeat_utc = str(status.get("heartbeat_utc") or "")
    heartbeat_age_seconds = None
    if heartbeat_utc:
        try:
            heartbeat = datetime.fromisoformat(heartbeat_utc.replace("Z", "+00:00"))
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            heartbeat_age_seconds = max(
                0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds()
            )
        except ValueError:
            heartbeat_age_seconds = None
    return {
        "run_id": "DS24_CLEAN_V2_TOURNAMENT_R1_20260926",
        "host_role": host,
        "classification": classification,
        "status_authority_path": status_path.relative_to(repository_root).as_posix(),
        "legacy_generic_supervisor_status_ignored": (
            run_root / "supervisor_status.json"
        ).is_file(),
        "attempt_generation": status.get("attempt_generation"),
        "supervisor_pid": status.get("supervisor_pid"),
        "supervisor_process_creation_time_utc": supervisor_creation,
        "supervisor_identity_verified": supervisor_identity_verified,
        "supervisor_process_alive_verified": supervisor_identity_verified,
        "status_heartbeat_utc": heartbeat_utc or None,
        "status_heartbeat_age_seconds": heartbeat_age_seconds,
        "active_workers": active_workers,
        "active_model_worker_count": len(active_workers),
        "rejected_unverifiable_active_worker_records": (
            rejected_active_worker_records
        ),
        "stale_worker_records_reconciled": status.get(
            "stale_worker_records_reconciled", []
        ),
        "stale_reservations_reclaimed": status.get(
            "stale_reservations_reclaimed", []
        ),
        "queued_families": queued_families,
        "queued_wait_reasons": status.get("queued_wait_reasons", {}),
        "complete_families": complete_families,
        "failed_families": failed_families,
        "superseded_historical_status_failures": superseded_status_failures,
        "resource_paused_families": resource_paused_families,
        "deferred_resource_families": deferred_resource_families,
        "capacity_only_deferrals": sorted(deferred_resource_families),
        "families": rows,
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "maximum_model_workers": status.get(
            "maximum_model_workers",
            MAX_DELL_MODEL_WORKERS if host == "dell" else 1,
        ),
        "ram_available_bytes": memory_snapshot.available_physical_bytes,
        "commit_headroom_bytes": memory_snapshot.commit_headroom_bytes,
        "commit_limit_bytes": memory_snapshot.commit_limit_bytes,
        "committed_bytes": memory_snapshot.committed_bytes,
        "peak_committed_bytes": memory_snapshot.peak_committed_bytes,
        "memory_snapshot_source": memory_snapshot.source,
        "supervisor_memory": (
            process_memory_snapshot(supervisor_pid).payload()
            if supervisor_identity_verified
            else None
        ),
        "worker_job": status.get("worker_job"),
        "worker_reservations": status.get("worker_reservations"),
        "resource_policy": effective_resource_policy,
        "status_resource_policy_matches_effective": status_policy_matches,
        "memory_pressure_safe": memory_decision.safe,
        "memory_admission_allowed": memory_decision.admission_allowed,
        "memory_pressure_level": memory_decision.pressure_level,
        "memory_pressure_warnings": list(memory_decision.warning_reasons),
        "memory_pressure_blocking_reasons": list(
            memory_decision.blocking_reasons
        ),
        "actual_host_pressure": bool(
            actual_host_pressure_reasons(memory_snapshot)
        ),
        "actual_host_pressure_reasons": list(
            actual_host_pressure_reasons(memory_snapshot)
        ),
        "blocking_reasons": status.get("blocking_reasons", []),
        "feature_authority_hash": status.get("feature_authority_hash"),
        "target_authority_hash": status.get("target_authority_hash"),
    }


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Project the unattended-operational fields without hiding failures."""

    return {
        "classification": report.get("classification"),
        "supervisor_process_alive_verified": report.get(
            "supervisor_process_alive_verified"
        ),
        "status_heartbeat_age_seconds": report.get("status_heartbeat_age_seconds"),
        "active_model_worker_count": report.get("active_model_worker_count"),
        "active_workers": [
            row.get("family") for row in report.get("active_workers", [])
        ],
        "active_reservations": (
            (report.get("worker_reservations") or {}).get(
                "active_worker_reservations", 0
            )
        ),
        "stale_worker_records_reconciled": len(
            report.get("stale_worker_records_reconciled", [])
        ),
        "stale_reservations_reclaimed": len(
            report.get("stale_reservations_reclaimed", [])
        ),
        "deferred_resource_families": sorted(
            report.get("deferred_resource_families", {})
        ),
        "actual_host_pressure": report.get("actual_host_pressure"),
        "memory_pressure_level": report.get("memory_pressure_level"),
        "failed_families": report.get("failed_families"),
        "families": [
            {
                key: row.get(key)
                for key in (
                    "family",
                    "state",
                    "completed_refits",
                    "score_timestamps",
                    "current_historical_date",
                    "error",
                )
            }
            for row in report.get("families", [])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor one DS24 clean-V2 host queue.")
    parser.add_argument("--host", choices=("dell", "mac"), default="dell")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    report = build_report(args.host)
    if args.compact:
        report = compact_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
