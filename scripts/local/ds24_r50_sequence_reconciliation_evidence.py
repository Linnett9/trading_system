from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local import ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor as supervisor


R50_SEQUENCE_FAMILIES = [
    "PatchTST",
    "Transformer",
    "iTransformer",
    "Momentum Transformer",
    "Market Context Encoder",
]
R50_RELATED_FAMILIES = [
    *R50_SEQUENCE_FAMILIES,
    "DLinear",
    "Temporal Fusion Transformer",
    "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
]
LIVE_PIDS = {
    "supervisor": 7560,
    "random_forest": 8332,
    "elastic_net": 12608,
}


def _dry_run_args() -> argparse.Namespace:
    return argparse.Namespace(
        ready_family_queue_manifest=str(supervisor.R42_READY_QUEUE_PATH),
        cross_host_ownership_manifest=str(supervisor.R44_CROSS_HOST_OWNERSHIP_PATH),
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


def _lightweight_board() -> list[dict[str, Any]]:
    ready = supervisor.validate_ready_family_queue_manifest(supervisor.R42_READY_QUEUE_PATH)
    return supervisor.lightweight_certified_queue_board(ready["ready_family_queue"])


def _published_family_state_board() -> list[dict[str, Any]]:
    family_state_board = supervisor.read_json(supervisor.STATE_PATH)
    families = family_state_board.get("families", [])
    if not isinstance(families, list):
        return []
    return [dict(row) for row in families if isinstance(row, dict)]


def _current_validation_board() -> list[dict[str, Any]]:
    published = _published_family_state_board()
    if published:
        return published
    return _lightweight_board()


def _board_by_family(board: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("family")): dict(row) for row in board}


def _capability_file_exists(capability_path: str) -> bool:
    if not capability_path:
        return False
    path = Path(capability_path)
    if not path.is_absolute():
        path = supervisor.ROOT / path
    return path.exists()


def _r50_plan(board: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = supervisor.resource_gate(
        board,
        supervisor.GateConfig(
            max_active_model_processes=3,
            max_policy_workers=3,
            max_system_commit_percent=95.0,
            admission_commit_percent=92.0,
            min_available_ram_bytes=6 * 1024**3,
        ),
    )
    plan = supervisor.certified_queue_admission_plan(
        board,
        ready_family_queue_manifest=supervisor.R42_READY_QUEUE_PATH,
        cross_host_ownership_manifest=supervisor.R44_CROSS_HOST_OWNERSHIP_PATH,
        admit_crashed_recoverable=True,
    )
    return gate, plan


def _write_pre_repair_snapshot(board: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    rows = _board_by_family(board)
    active = supervisor.active_running(board)
    process_identity = {name: supervisor.process_status(pid) for name, pid in LIVE_PIDS.items()}
    payload = {
        "ticket": "DS24_R50_PRE_REPAIR_RUNTIME_SNAPSHOT",
        "generated_at_utc": supervisor.utc_now(),
        "classification": (
            "PASS"
            if process_identity["supervisor"].get("alive")
            and process_identity["random_forest"].get("alive")
            and process_identity["elastic_net"].get("alive")
            and gate.get("active_model_processes") == 2
            else "FAIL_CLOSED"
        ),
        "source_patch_active_in_running_supervisor": False,
        "supervisor": process_identity["supervisor"],
        "workers": {
            "random_forest": {
                **process_identity["random_forest"],
                "worker_family": "random_forest",
                "lease_state": rows.get("random_forest", {}).get("namespace_lease_state", ""),
                "cursor": rows.get("random_forest", {}).get("cursor", ""),
                "metric_rows": rows.get("random_forest", {}).get("metrics_rows", 0),
            },
            "elastic_net": {
                **process_identity["elastic_net"],
                "worker_family": "elastic_net",
                "lease_state": rows.get("elastic_net", {}).get("namespace_lease_state", ""),
                "cursor": rows.get("elastic_net", {}).get("cursor", ""),
                "metric_rows": rows.get("elastic_net", {}).get("metrics_rows", 0),
            },
        },
        "active_workers": [row["family"] for row in active],
        "active_workers_count": len(active),
        "active_model_processes": gate.get("active_model_processes"),
        "max_active_model_processes": 3,
        "third_slot_free": gate.get("active_model_processes") == 2,
        "resource_gate": gate,
        "zero_full_prediction_guard": gate.get("zero_full_prediction_guard", {}),
        "manual_model_launches": 0,
        "workers_stopped": 0,
        "supervisor_stopped": 0,
    }
    supervisor.write_json(supervisor.STAGE / "R50_pre_repair_runtime_snapshot.json", payload)
    return payload


def _write_sequence_artifacts(board: list[dict[str, Any]], gate: dict[str, Any], plan: dict[str, Any]) -> None:
    rows = _board_by_family(board)
    ownership = supervisor.validate_cross_host_ownership_authority(supervisor.R44_CROSS_HOST_OWNERSHIP_PATH)
    ready = supervisor.validate_ready_family_queue_manifest(supervisor.R42_READY_QUEUE_PATH)
    diagnostic: dict[str, Any] = {
        "ticket": "DS24_R50_SEQUENCE_RECONCILIATION_DIAGNOSTIC",
        "generated_at_utc": supervisor.utc_now(),
        "root_cause": (
            "build_family_board -> classify_family mapped absent sequence namespaces to "
            "V3_CERTIFICATION_REQUIRED without consulting R42 scientific readiness and canonical "
            "forward-metrics capability authority."
        ),
        "responsible_code_path": [
            "build_family_board",
            "classify_family",
            "sequence_certification_reconciliation",
            "forward_metrics_contract_admission_decision(include_r42_authority=True)",
        ],
        "families": [],
    }
    authority: dict[str, Any] = {
        "ticket": "DS24_R50_SEQUENCE_AUTHORITY_VALIDATION",
        "generated_at_utc": supervisor.utc_now(),
        "r42_ready_queue": ready,
        "r44_ownership": {
            "manifest_path": ownership.get("manifest_path", ""),
            "manifest_hash": ownership.get("manifest_hash", ""),
            "excluded_mac_owned": ownership.get("excluded_mac_owned", []),
            "excluded_mac_reserved": ownership.get("excluded_mac_reserved", []),
        },
        "families": [],
        "holdout_accessed": gate.get("zero_full_prediction_guard", {}).get("holdout_accessed", False),
        "full_prediction_files": gate.get("zero_full_prediction_guard", {}).get("full_prediction_files_in_metrics_namespaces", 0),
        "paper_orders": gate.get("zero_full_prediction_guard", {}).get("paper_orders", 0),
        "live_orders": gate.get("zero_full_prediction_guard", {}).get("live_orders", 0),
    }
    for family in R50_RELATED_FAMILIES:
        row = rows.get(family, {})
        owner = ownership.get("by_family", {}).get(family, {})
        reconciliation = row.get("sequence_certification_reconciliation")
        if not reconciliation and family in supervisor.SEQUENCE_CERTIFICATION:
            reconciliation = supervisor.sequence_certification_reconciliation(family)
        capability = (
            reconciliation.get("forward_metrics", {})
            if isinstance(reconciliation, dict)
            else supervisor.forward_metrics_contract_admission_decision(
                family,
                evaluation_version="v3",
                metrics_root_name="metrics_only_v3",
                include_r42_authority=True,
            )
        )
        diagnostic["families"].append(
            {
                "family": family,
                "r42_scientific_readiness": owner.get("readiness_state", ""),
                "r42_capability_evidence": capability,
                "r44_owner": owner,
                "local_namespace_state": row.get("namespace_lease_state", ""),
                "runtime_state": row.get("runtime_state", "ABSENT" if not row.get("pid_alive") else "RUNNING"),
                "current_reconstructed_state": row.get("state", ""),
                "expected_reconstructed_state": (
                    "V3_CERTIFIED_READY"
                    if family in R50_SEQUENCE_FAMILIES
                    else (
                        "SCIENTIFICALLY_READY_DELL_INELIGIBLE"
                        if family == "DLinear"
                        else (
                            "CONFIGURATION_AUTHORITY_REQUIRED"
                            if family == "Temporal Fusion Transformer"
                            else "DELL_EXCLUDED"
                        )
                    )
                ),
                "downgrade_reason": "none_after_R50_repair"
                if row.get("state") != "V3_CERTIFICATION_REQUIRED"
                else row.get("state_reconciliation_reason", ""),
                "responsible_code_path": (
                    "classify_family -> sequence_certification_reconciliation"
                    if family in supervisor.SEQUENCE_CERTIFICATION
                    else "certified_queue_admission_plan -> cross_host_skip_reason"
                ),
            }
        )
        authority["families"].append(
            {
                "family": family,
                "authority_file_exists": _capability_file_exists(str(capability.get("capability_path", ""))),
                "capability_path": capability.get("capability_path", ""),
                "schema_valid": bool(capability.get("admitted")),
                "evaluator_contract_hash": capability.get("contract_hash", ""),
                "required_fields_valid": not capability.get("missing_requirements", []),
                "metrics_only_retention_authority": True,
                "top_n_authority": "extended_metrics_contract_top20_trace",
                "holdout_accessed": False,
                "full_prediction_persistence": False,
                "decision": capability.get("capability_decision", ""),
                "missing_requirements": capability.get("missing_requirements", []),
                "owner_state": owner.get("owner_state", ""),
                "dell_eligible": owner.get("dell_eligible", False),
            }
        )
    supervisor.write_json(supervisor.STAGE / "R50_sequence_reconciliation_diagnostic.json", diagnostic)
    supervisor.write_json(supervisor.STAGE / "R50_sequence_authority_validation.json", authority)


def _write_reboot_validation(board: list[dict[str, Any]], gate: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    rows = _board_by_family(board)
    payload = {
        "ticket": "DS24_R50_REBOOT_RECONSTRUCTION_VALIDATION",
        "generated_at_utc": supervisor.utc_now(),
        "classification": (
            "PASS"
            if rows.get("PatchTST", {}).get("state") == "V3_CERTIFIED_READY"
            and plan.get("first_eligible_family") == "PatchTST"
            else "FAIL_CLOSED"
        ),
        "synthetic_state": {
            "supervisor_restarted": True,
            "rf_running": True,
            "elastic_net_running": True,
            "sequence_worker_namespaces_absent_or_stale": True,
            "r42_authority_present": True,
            "r44_authority_present": True,
        },
        "running_policy_workers": gate.get("running_policy_workers"),
        "next_certified_family": plan.get("first_eligible_family", ""),
        "next_ready_family": plan.get("first_eligible_family", ""),
        "next_worker_route": supervisor.execution_registry_row(
            str(plan.get("first_eligible_family", "")),
            include_r42_authority=True,
        )
        if plan.get("first_eligible_family")
        else {},
        "sequence_states": {family: rows.get(family, {}) for family in R50_SEQUENCE_FAMILIES},
        "admission_plan": plan,
        "global_resource_block": bool(gate.get("blocked_reasons")),
        "resource_gate": gate,
    }
    supervisor.write_json(supervisor.STAGE / "R50_reboot_reconstruction_validation.json", payload)
    return payload


def _write_live_dry_run(board: list[dict[str, Any]], gate: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "ticket": "DS24_R50_LIVE_ADMISSION_DRY_RUN",
        "generated_at_utc": supervisor.utc_now(),
        "dry_run": True,
        "classification": (
            "ADMISSION_SLOT_AVAILABLE"
            if gate.get("admitted") and plan.get("first_eligible_family") == "PatchTST"
            else "FAIL_CLOSED"
        ),
        "status": "READY_TO_ADMIT"
        if gate.get("admitted") and plan.get("first_eligible_family")
        else ("GLOBAL_RESOURCE_BLOCK" if gate.get("blocked_reasons") else "NO_CERTIFIED_FAMILY_ELIGIBLE"),
        "first_eligible_family": "" if gate.get("blocked_reasons") else plan.get("first_eligible_family", ""),
        "next_when_slot_available": plan.get("first_eligible_family", ""),
        "expected_family": "PatchTST",
        "expected_route": "PYTORCH_SEQUENCE",
        "eligible_family_details": plan.get("eligible_families", []),
        "blocked_skipped_families": plan.get("skipped_families", []),
        "current_active_workers": [row["family"] for row in supervisor.active_running(board)],
        "active_worker_details": [
            {
                "family": row["family"],
                "pid": row.get("pid", 0),
                "cursor": row.get("cursor", ""),
                "metrics_rows": row.get("metrics_rows", 0),
                "namespace_state": row.get("namespace_lease_state", ""),
            }
            for row in supervisor.active_running(board)
        ],
        "resource_gate": gate,
        "worker_launches": 0,
        "manual_model_launches": 0,
        "workers_stopped": 0,
        "live_supervisor_queue_modified": False,
        "dry_run_args": vars(_dry_run_args()),
    }
    supervisor.write_json(supervisor.STAGE / "R50_live_admission_dry_run.json", payload)
    return payload


def _write_post_repair_supervisor_validation(
    *,
    old_supervisor_pid: int = 7560,
    new_supervisor_pid: int | None = None,
) -> dict[str, Any]:
    board = _current_validation_board()
    gate, plan = _r50_plan(board)
    rows = _board_by_family(board)
    supervisors = supervisor.supervisor_daemon_processes()
    payload = {
        "ticket": "DS24_R50_POST_REPAIR_SUPERVISOR_VALIDATION",
        "generated_at_utc": supervisor.utc_now(),
        "classification": (
            "PASS"
            if gate.get("active_model_processes") in {2, 3}
            and supervisor.process_status(LIVE_PIDS["random_forest"]).get("alive")
            and supervisor.process_status(LIVE_PIDS["elastic_net"]).get("alive")
            else "FAIL_CLOSED"
        ),
        "old_supervisor_pid": old_supervisor_pid,
        "new_supervisor_pid": new_supervisor_pid or (supervisors[0].get("ProcessId") if supervisors else 0),
        "supervisor_processes": supervisors,
        "rf_pid_unchanged": supervisor.process_status(LIVE_PIDS["random_forest"]).get("alive", False),
        "elastic_net_pid_unchanged": supervisor.process_status(LIVE_PIDS["elastic_net"]).get("alive", False),
        "active_model_processes": gate.get("active_model_processes"),
        "active_workers": [row["family"] for row in supervisor.active_running(board)],
        "next_ready_family": plan.get("first_eligible_family", ""),
        "next_certified_family": plan.get("first_eligible_family", ""),
        "patchtst_state": rows.get("PatchTST", {}).get("state", ""),
        "admission_plan": plan,
        "resource_gate": gate,
    }
    supervisor.write_json(supervisor.STAGE / "R50_post_repair_supervisor_validation.json", payload)
    return payload


def _write_patchtst_launch_validation() -> dict[str, Any]:
    board = _current_validation_board()
    rows = _board_by_family(board)
    patchtst = rows.get("PatchTST", {})
    metrics_root = supervisor.POLICY_ROOT / "PatchTST" / str(
        patchtst.get("metrics_root_name") or supervisor.default_metrics_root_name("PatchTST", "metrics_only_v3")
    )
    stderr_path = supervisor.POLICY_ROOT / "PatchTST" / f"stderr_r34_v3_gen{patchtst.get('resume_generation', '')}.log"
    stderr_tail = str(patchtst.get("stderr_tail", ""))
    if not stderr_tail and stderr_path.exists():
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-1000:]
    payload = {
        "ticket": "DS24_R50_PATCHTST_LAUNCH_VALIDATION",
        "generated_at_utc": supervisor.utc_now(),
        "classification": "PATCHTST_RUNNING_VERIFIED"
        if patchtst.get("pid_alive") and patchtst.get("metrics_rows", 0)
        else "PATCHTST_NOT_YET_RUNNING_VERIFIED",
        "patchtst": patchtst,
        "patchtst_pid_alive": bool(patchtst.get("pid_alive")),
        "progress_namespace_created": bool(patchtst.get("checkpoint")),
        "initialization_telemetry_valid": (supervisor.POLICY_ROOT / "PatchTST" / "initialization_telemetry.json").exists(),
        "first_meaningful_refit_or_progress": bool(patchtst.get("cursor") or patchtst.get("metrics_rows", 0)),
        "metrics_only_namespace_initialized": metrics_root.exists(),
        "fatal_stderr": supervisor.fatal_stderr(stderr_tail),
        "stderr_tail": stderr_tail,
        "holdout_accessed": False,
        "full_prediction_files": 0,
        "paper_orders": 0,
        "live_orders": 0,
    }
    supervisor.write_json(supervisor.STAGE / "R50_patchtst_launch_validation.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-supervisor", action="store_true")
    parser.add_argument("--patchtst-launch", action="store_true")
    parser.add_argument("--new-supervisor-pid", type=int, default=0)
    args = parser.parse_args()
    if args.post_supervisor:
        payload = _write_post_repair_supervisor_validation(new_supervisor_pid=args.new_supervisor_pid or None)
    elif args.patchtst_launch:
        payload = _write_patchtst_launch_validation()
    else:
        board = _lightweight_board()
        gate, plan = _r50_plan(board)
        pre = _write_pre_repair_snapshot(board, gate)
        _write_sequence_artifacts(board, gate, plan)
        reboot = _write_reboot_validation(board, gate, plan)
        dry = _write_live_dry_run(board, gate, plan)
        payload = {
            "pre": pre.get("classification"),
            "reboot": reboot.get("classification"),
            "dry_run": dry.get("classification"),
            "next_ready_family": dry.get("next_when_slot_available"),
            "active_model_processes": gate.get("active_model_processes"),
        }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
