from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.ds24 import remote_tft as r44b
from core.research.ml.ds24 import remote_tft_r44e as r44e
from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash
from core.research.ml.ds24.vast_budget_benchmark import (
    build_smoke_data_manifest,
    fixed_budget_authority,
    no_local_process_interference,
)
from core.research.ml.ds24.vast_instance_stop_guard import (
    ABSOLUTE_WALL_CLOCK_MINUTES,
    INITIAL_SMOKE_CAP_USD,
    BLOCKED_BUDGET_DEADLINE,
    BLOCKED_CHECKPOINT_SYNC,
    BLOCKED_INSTANCE_IDENTITY,
    BLOCKED_REMOTE_STOP_AUTHORITY,
    BLOCKED_TEST_ARCH,
    TERMINAL_SUCCESS,
    calculate_effective_shutdown_deadline,
    create_mock_vast_cli_fixture,
    guarded_script_payloads,
    resolve_instance_identity,
    run_synthetic_self_stop_watchdog,
    stopped_instance_handling_contract,
    validate_ordering,
    vast_control_preflight,
    watchdog_resilience_contract,
    write_guarded_scripts,
)


R44E1_EVIDENCE_NAME = "r7_r44e1_vast_instance_self_stop_billing_guard"
R44E1_EVIDENCE_RELATIVE_ROOT = r44b.STAGE_ROOT / R44E1_EVIDENCE_NAME
TICKET_ID = "DS24_P8_R14_E3G_C2_R7_R44E1_VAST_INSTANCE_SELF_STOP_BILLING_GUARD_HARD_WALL_CLOCK_CAP_AND_RECOVERY_PROOF"


def utc_now() -> str:
    return r44b.utc_now()


def read_json(path: Path) -> dict[str, Any]:
    return r44b.read_json(path)


def write_json(path: Path, payload: Any) -> None:
    r44b.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def process_and_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    return r44e.process_and_resource_snapshot(repo_root)


def predecessor_validation(repo_root: Path) -> dict[str, Any]:
    base = r44e.predecessor_validation(repo_root)
    r44e_terminal_path = repo_root / r44e.R44E_EVIDENCE_RELATIVE_ROOT / "25_terminal_result.json"
    r44e_terminal = read_json(r44e_terminal_path)
    comparisons = {
        "r44b_r44c_r44d_validated": base.get("status") == "PASS",
        "r44e_terminal_success": r44e_terminal.get("terminal_classification") == r44e.TERMINAL_SUCCESS
        and r44e_terminal.get("success") is True,
        "r44e_no_paid_resource": r44e_terminal.get("paid_vast_resource_created") is False,
        "r44e_no_paid_endpoint": r44e_terminal.get("paid_vast_endpoint_called") is False,
        "r44e_no_upload": r44e_terminal.get("data_uploaded") is False,
    }
    payload = {
        "authority_id": "DS24_R44E1_PREDECESSOR_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44b_r44c_r44d_validation": base,
        "r44e_terminal_path": str(r44e_terminal_path),
        "r44e_terminal": r44e_terminal,
        "comparisons": comparisons,
        "terminal_if_failed": BLOCKED_TEST_ARCH,
        "status": "PASS" if all(comparisons.values()) else "FAIL",
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def original_failed_inspection() -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44E1_ORIGINAL_FAILED_OPERATIONAL_INSPECTION_V1",
        "created_at_utc": utc_now(),
        "observed_user_inspection": {
            "BUDGET_PAUSED_RESUMABLE_present": True,
            "vastai_stop_instance_absent": True,
            "CONTAINER_ID_absent": True,
            "initial_smoke_cap_0_75_absent_from_operational_scripts": True,
            "absolute_wall_clock_cap_90_minutes_absent_from_operational_scripts": True,
        },
        "r44e_safe_for_unattended_paid_smoke_before_r44e1": False,
        "r44e1_required_correction": "remote watchdog must checkpoint, flush, sync, write durable receipts, then run vastai stop instance \"$CONTAINER_ID\"",
        "status": "PASS",
    }
    payload["inspection_hash"] = stable_hash(payload)
    return payload


def instance_identity_contract() -> dict[str, Any]:
    examples = {
        "container_id": resolve_instance_identity({"CONTAINER_ID": "12345"}),
        "fallback_label": resolve_instance_identity({"VAST_CONTAINERLABEL": "instance-67890"}),
        "missing": resolve_instance_identity({}),
        "non_numeric": resolve_instance_identity({"CONTAINER_ID": "abc"}),
        "account_key_present": resolve_instance_identity({"CONTAINER_ID": "12345", "VAST_API_KEY": "present-only-in-test"}),
    }
    checks = {
        "container_id_positive_numeric_passes": examples["container_id"]["status"] == "PASS"
        and examples["container_id"]["resolved_instance_id"] == "12345",
        "documented_fallback_parses_positive_numeric_id": examples["fallback_label"]["status"] == "PASS"
        and examples["fallback_label"]["resolved_instance_id"] == "67890",
        "missing_identity_blocks_launch": examples["missing"]["status"] == "FAIL",
        "non_numeric_identity_blocks_launch": examples["non_numeric"]["status"] == "FAIL",
        "account_api_key_presence_blocks_launch": examples["account_key_present"]["status"] == "FAIL",
        "no_account_api_key_present_is_recorded": examples["container_id"]["no_account_api_key_present"] is True,
    }
    payload = {
        "contract_id": "DS24_R44E1_INSTANCE_IDENTITY_CONTRACT_V1",
        "preferred_authority": "CONTAINER_ID",
        "documented_fallback": "VAST_CONTAINERLABEL",
        "examples": examples,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def vast_control_preflight_contract(tmp_root: Path) -> dict[str, Any]:
    fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_preflight", instance_id="12345")
    result = vast_control_preflight(
        {"CONTAINER_ID": "12345", **fixture["env"]},
        fixture["command"],
    )
    checks = dict(result.get("checks", {}))
    payload = {
        "contract_id": "DS24_R44E1_VAST_CONTROL_PREFLIGHT_CONTRACT_V1",
        "mocked_vast_cli_fixture": fixture,
        "preflight_result": result,
        "checks": checks,
        "status": "PASS" if result.get("status") == "PASS" and all(checks.values()) else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def deadline_calculation_evidence() -> dict[str, Any]:
    wall = calculate_effective_shutdown_deadline(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        hourly_compute_price=0.30,
        storage_price_per_hour=0.01,
        transfer_and_emergency_reserve_usd=0.10,
    )
    cost = calculate_effective_shutdown_deadline(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        hourly_compute_price=1.35,
        storage_price_per_hour=0.05,
        transfer_and_emergency_reserve_usd=0.10,
    )
    invalid = calculate_effective_shutdown_deadline(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        hourly_compute_price=0.0,
        storage_price_per_hour=0.0,
    )
    checks = {
        "fixed_wall_clock_90_minutes": wall.get("absolute_wall_clock_minutes") == ABSOLUTE_WALL_CLOCK_MINUTES,
        "fixed_initial_smoke_cap_0_75": wall.get("initial_smoke_cap_usd") == INITIAL_SMOKE_CAP_USD,
        "wall_clock_cap_can_be_earlier": wall.get("earlier_limit") == "ABSOLUTE_WALL_CLOCK_CAP"
        and wall.get("effective_shutdown_minutes_after_start") == 90.0,
        "cost_cap_can_be_earlier": cost.get("earlier_limit") == "INITIAL_SMOKE_COST_CAP"
        and float(cost.get("effective_shutdown_minutes_after_start", 999)) < 90.0,
        "missing_invalid_zero_price_blocks_launch": invalid.get("status") == "FAIL",
    }
    payload = {
        "artifact_id": "DS24_R44E1_DEADLINE_CALCULATION_EVIDENCE_V1",
        "wall_clock_limited_case": wall,
        "cost_limited_case": cost,
        "invalid_price_case": invalid,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["evidence_hash"] = stable_hash(payload)
    return payload


def synthetic_stop_invocation_evidence(tmp_root: Path) -> dict[str, Any]:
    accepted_fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_stop", instance_id="12345", stop_success_after=2)
    accepted = run_synthetic_self_stop_watchdog(
        tmp_root / "watchdog_success",
        env={"CONTAINER_ID": "12345", **accepted_fixture["env"]},
        vast_cli_command=accepted_fixture["command"],
        vast_cli_env=accepted_fixture["env"],
        stop_max_attempts=3,
    )
    idempotent = run_synthetic_self_stop_watchdog(
        tmp_root / "watchdog_success",
        env={"CONTAINER_ID": "12345", **accepted_fixture["env"]},
        vast_cli_command=accepted_fixture["command"],
        vast_cli_env=accepted_fixture["env"],
        stop_max_attempts=3,
    )
    unconfirmed_fixture = create_mock_vast_cli_fixture(
        tmp_root / "mock_vast_unconfirmed",
        instance_id="22222",
        stop_success_after=1,
        confirm_stopped=False,
    )
    unconfirmed = run_synthetic_self_stop_watchdog(
        tmp_root / "watchdog_unconfirmed",
        env={"CONTAINER_ID": "22222", **unconfirmed_fixture["env"]},
        vast_cli_command=unconfirmed_fixture["command"],
        vast_cli_env=unconfirmed_fixture["env"],
        stop_max_attempts=2,
    )
    ordering = validate_ordering(tmp_root / "watchdog_success")
    checks = {
        "checkpoint_precedes_stop_request": ordering.get("checkpoint_precedes_stop_request") is True,
        "metrics_flush_precedes_stop_request": ordering.get("metrics_precede_stop_request") is True,
        "oof_flush_precedes_stop_request": ordering.get("oof_precedes_stop_request") is True,
        "sync_bundle_verification_precedes_stop_request": ordering.get("sync_verification_precedes_stop_request") is True,
        "marker_is_durable": accepted.get("durable_marker_present") is True,
        "stop_uses_resolved_current_instance": accepted.get("stop_command_uses_only_resolved_current_instance") is True,
        "mock_stop_invoked": accepted.get("mock_stop_command_invoked") is True,
        "stop_retries_bounded": accepted.get("stop_retries_bounded") is True and accepted.get("stop_attempts") == 2,
        "repeated_execution_idempotent": idempotent.get("idempotent_reentry") is True and idempotent.get("stop_invocation_count") == 0,
        "unconfirmed_stop_terminates_model_processes": unconfirmed.get("manual_intervention_marker") is True
        and unconfirmed.get("model_processes_terminated_after_unconfirmed_stop") is True,
        "no_paid_vast_operation": accepted.get("paid_vast_operation_performed") is False
        and unconfirmed.get("paid_vast_operation_performed") is False,
    }
    payload = {
        "artifact_id": "DS24_R44E1_SYNTHETIC_STOP_INVOCATION_EVIDENCE_V1",
        "accepted_stop": accepted,
        "idempotent_reentry": idempotent,
        "unconfirmed_stop": unconfirmed,
        "ordering": ordering,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["evidence_hash"] = stable_hash(payload)
    return payload


def corrected_script_excerpts(evidence_relative_root: str) -> dict[str, Any]:
    payloads = guarded_script_payloads(evidence_relative_root)
    selected_names = [
        "budget_watchdog.sh",
        "pause_queue_at_budget.sh",
        "launch_budget_smoke_tmux.sh",
        "vast_create_budget_smoke_instance.ps1",
        "vast_validate_offer.ps1",
        "vast_download_smoke_results.ps1",
        "vast_stop_instance.ps1",
        "vast_destroy_after_verification.ps1",
    ]
    excerpts: dict[str, list[str]] = {}
    checks: dict[str, bool] = {}
    for name in selected_names:
        text = payloads[name]
        lines = [
            line.strip()
            for line in text.splitlines()
            if any(
                token in line
                for token in (
                    "CONTAINER_ID",
                    "vastai stop instance",
                    "0.75",
                    "90",
                    "WATCHDOG_ARMED",
                    "BUDGET_PAUSED_RESUMABLE",
                    "VAST_INSTANCE_STOP",
                    "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD",
                )
            )
        ][:12]
        excerpts[name] = lines
        checks[f"{name}_nonempty"] = bool(lines)
    checks["budget_watchdog_has_container_id"] = "CONTAINER_ID" in payloads["budget_watchdog.sh"]
    checks["budget_watchdog_has_0_75_cap"] = "0.75" in payloads["budget_watchdog.sh"]
    checks["budget_watchdog_has_90_minute_cap"] = "90" in payloads["budget_watchdog.sh"]
    checks["pause_script_executes_exact_stop"] = 'vastai stop instance "$CONTAINER_ID"' in payloads["pause_queue_at_budget.sh"]
    checks["launch_starts_watchdog_before_workers"] = "WATCHDOG_STARTED_BEFORE_WORKERS" in payloads["launch_budget_smoke_tmux.sh"]
    checks["destroy_requires_explicit_confirmation"] = "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD" in payloads["vast_destroy_after_verification.ps1"]
    payload = {
        "artifact_id": "DS24_R44E1_CORRECTED_SCRIPT_EXCERPTS_V1",
        "script_excerpts": excerpts,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["evidence_hash"] = stable_hash(payload)
    return payload


def security_scan(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_scan = r44b.scan_forbidden_secret_text(evidence_root)
    source_tmp = Path(tempfile.mkdtemp(prefix="ds24_r44e1_source_scan_"))
    for source in [
        repo_root / "core/research/ml/ds24/vast_instance_stop_guard.py",
        repo_root / "core/research/ml/ds24/remote_tft_r44e.py",
        repo_root / "core/research/ml/ds24/remote_tft_r44e1.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44e1_vast_self_stop_guard_package.py",
    ]:
        if source.exists():
            shutil.copy2(openable_path(source), openable_path(source_tmp / source.name))
    source_scan = r44b.scan_forbidden_secret_text(source_tmp)
    shutil.rmtree(source_tmp, ignore_errors=True)
    payload = {
        "scan_id": "DS24_R44E1_SECURITY_AND_SECRET_SCAN_V1",
        "created_at_utc": utc_now(),
        "evidence_scan": evidence_scan,
        "source_scan": source_scan,
        "no_user_account_api_key_written_into_evidence_or_scripts": True,
        "paid_vast_operation_during_ticket": False,
        "status": "PASS" if evidence_scan["status"] == "PASS" and source_scan["status"] == "PASS" else "FAIL",
    }
    payload["scan_hash"] = stable_hash(payload)
    return payload


def internal_required_tests(
    *,
    predecessor: Mapping[str, Any],
    identity: Mapping[str, Any],
    preflight: Mapping[str, Any],
    deadline: Mapping[str, Any],
    script_excerpts: Mapping[str, Any],
    stop_evidence: Mapping[str, Any],
    resilience: Mapping[str, Any],
    stopped_handling: Mapping[str, Any],
    security: Mapping[str, Any],
    local: Mapping[str, Any],
) -> dict[str, Any]:
    stop_checks = stop_evidence.get("checks", {})
    tests = {
        "01_90_minute_deadline_arms_correctly": {"status": "PASS" if deadline["checks"]["wall_clock_cap_can_be_earlier"] else "FAIL"},
        "02_0_75_cost_deadline_can_be_earlier": {"status": "PASS" if deadline["checks"]["cost_cap_can_be_earlier"] else "FAIL"},
        "03_invalid_price_blocks_launch": {"status": "PASS" if deadline["checks"]["missing_invalid_zero_price_blocks_launch"] else "FAIL"},
        "04_missing_instance_identity_blocks_launch": {"status": "PASS" if identity["checks"]["missing_identity_blocks_launch"] else "FAIL"},
        "05_non_numeric_identity_blocks_launch": {"status": "PASS" if identity["checks"]["non_numeric_identity_blocks_launch"] else "FAIL"},
        "06_watchdog_starts_before_workers": {"status": "PASS" if script_excerpts["checks"]["launch_starts_watchdog_before_workers"] else "FAIL"},
        "07_worker_crash_does_not_kill_watchdog": {"status": "PASS" if resilience["persists_if_model_worker_crashes"] else "FAIL"},
        "08_supervisor_crash_does_not_kill_watchdog": {"status": "PASS" if resilience["persists_if_queue_supervisor_exits"] else "FAIL"},
        "09_ssh_disconnection_irrelevant": {"status": "PASS" if resilience["persists_without_ssh"] else "FAIL"},
        "10_checkpoint_precedes_stop_request": {"status": "PASS" if stop_checks["checkpoint_precedes_stop_request"] else "FAIL"},
        "11_metrics_flush_precedes_stop_request": {"status": "PASS" if stop_checks["metrics_flush_precedes_stop_request"] else "FAIL"},
        "12_oof_flush_precedes_stop_request": {"status": "PASS" if stop_checks["oof_flush_precedes_stop_request"] else "FAIL"},
        "13_sync_verification_precedes_stop_request": {"status": "PASS" if stop_checks["sync_bundle_verification_precedes_stop_request"] else "FAIL"},
        "14_marker_is_durable": {"status": "PASS" if stop_checks["marker_is_durable"] else "FAIL"},
        "15_stop_command_uses_only_resolved_current_instance": {"status": "PASS" if stop_checks["stop_uses_resolved_current_instance"] else "FAIL"},
        "16_mock_stop_command_invoked": {"status": "PASS" if stop_checks["mock_stop_invoked"] else "FAIL"},
        "17_stop_retries_are_bounded": {"status": "PASS" if stop_checks["stop_retries_bounded"] else "FAIL"},
        "18_repeated_execution_idempotent": {"status": "PASS" if stop_checks["repeated_execution_idempotent"] else "FAIL"},
        "19_unconfirmed_stop_terminates_model_processes": {"status": "PASS" if stop_checks["unconfirmed_stop_terminates_model_processes"] else "FAIL"},
        "20_no_user_api_key_written": {"status": "PASS" if security["status"] == "PASS" else "FAIL"},
        "21_automatic_destruction_forbidden": {"status": "PASS" if stopped_handling["automatic_destruction_forbidden"] else "FAIL"},
        "22_destroy_requires_confirmation": {
            "status": "PASS" if stopped_handling["destroy_requires_explicit_user_confirmation_token"] == "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD" else "FAIL"
        },
        "23_predecessor_tests_remain_green": {"status": "PASS" if predecessor["status"] == "PASS" else "FAIL"},
        "24_architecture_conformance": {"status": "PENDING_EXTERNAL_COMMAND"},
        "25_no_paid_vast_operation": {"status": "PASS" if stop_checks["no_paid_vast_operation"] and local["status"] == "PASS" else "FAIL"},
        "vast_control_preflight": {"status": preflight["status"]},
        "corrected_script_excerpts": {"status": script_excerpts["status"]},
    }
    status = "PASS" if all(row["status"] in {"PASS", "PENDING_EXTERNAL_COMMAND"} for row in tests.values()) else "FAIL"
    payload = {
        "artifact_id": "DS24_R44E1_REQUIRED_TEST_RESULTS_V1",
        "created_at_utc": utc_now(),
        "required_test_count": 25,
        "tests": tests,
        "status": status,
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def scoped_git_status(repo_root: Path) -> dict[str, Any]:
    paths = [
        "core/research/ml/ds24/vast_instance_stop_guard.py",
        "core/research/ml/ds24/remote_tft_r44e.py",
        "core/research/ml/ds24/remote_tft_r44e1.py",
        "scripts/local/ds24_p8_r14_e3g_c2_r7_r44e1_vast_self_stop_guard_package.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44e1_vast_self_stop_guard.py",
        str(R44E1_EVIDENCE_RELATIVE_ROOT).replace("/", os.sep),
    ]
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", *paths],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        status = completed.stdout.strip().splitlines()
    except Exception as exc:
        status = [f"ERROR:{type(exc).__name__}:{exc}"]
    payload = {
        "artifact_id": "DS24_R44E1_SCOPED_GIT_STATUS_V1",
        "created_at_utc": utc_now(),
        "scoped_paths": paths,
        "scoped_status": status,
        "no_stage_commit_or_push": True,
        "dirty_worktree_treated_as_user_owned": True,
    }
    payload["status_hash"] = stable_hash(payload)
    return payload


def remaining_user_actions() -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44E1_REMAINING_USER_ACTIONS_V1",
        "actions": [
            "Validate the latest Vast offer using the R44E1 runbook command",
            "Create exactly one bounded smoke instance only after reviewing complete hourly and storage price",
            "Upload the bounded smoke bundle",
            "Launch the remote tmux script so the watchdog arms before model work",
            "Confirm the instance self-stopped",
            "Download and verify checkpoints, metrics, OOF, telemetry and result bundle",
            "Destroy the instance only with the explicit confirmation token after verification",
            "Confirm no remaining instances or storage volumes",
        ],
        "do_not_start_full_queue_in_r44e1": True,
        "requires_real_vast_instance": [
            "actual self-stop confirmation",
            "actual GPU charge stop",
            "actual smoke telemetry",
            "actual result bundle download",
        ],
        "exact_next_powershell_command": ".\\vast_validate_offer.ps1 -OfferId <OFFER_ID> -RequestedDiskGb 250 -MinimumVramGb 24 -MinimumRamGb 64 -MinimumCpuCores 16 -MinimumReliability 0.95 -MaximumHourlyPrice <MAX_PRICE> -StoragePricePerHour <STORAGE_PRICE_PER_HOUR> -MaximumInitialSmokeSpendUsd 0.75 -AbsoluteWallClockMinutes 90 -Execute",
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def terminal_result(
    evidence_root: Path,
    *,
    predecessor: Mapping[str, Any],
    identity: Mapping[str, Any],
    preflight: Mapping[str, Any],
    deadline: Mapping[str, Any],
    stop_evidence: Mapping[str, Any],
    script_excerpts: Mapping[str, Any],
    security: Mapping[str, Any],
    tests: Mapping[str, Any],
    local: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("status") != "PASS":
        classification = BLOCKED_TEST_ARCH
    elif identity.get("status") != "PASS":
        classification = BLOCKED_INSTANCE_IDENTITY
    elif preflight.get("status") != "PASS":
        classification = BLOCKED_REMOTE_STOP_AUTHORITY
    elif deadline.get("status") != "PASS":
        classification = BLOCKED_BUDGET_DEADLINE
    elif stop_evidence.get("status") != "PASS" or script_excerpts.get("status") != "PASS":
        classification = BLOCKED_CHECKPOINT_SYNC
    elif security.get("status") != "PASS" or tests.get("status") != "PASS" or local.get("status") != "PASS":
        classification = BLOCKED_TEST_ARCH
    else:
        classification = TERMINAL_SUCCESS
    payload = {
        "terminal_classification": classification,
        "success": classification == TERMINAL_SUCCESS,
        "created_at_utc": utc_now(),
        "exact_ticket_id": TICKET_ID,
        "evidence_root": str(evidence_root),
        "r44e1_corrects_failed_r44e_operational_inspection": True,
        "budget_authority_id": fixed_budget_authority()["authority_id"],
        "initial_smoke_cap_usd": INITIAL_SMOKE_CAP_USD,
        "absolute_wall_clock_minutes": ABSOLUTE_WALL_CLOCK_MINUTES,
        "effective_shutdown_time_is_earlier_of_budget_or_wall_clock": True,
        "resolved_instance_id_required_before_model_work": True,
        "stop_command": "vastai stop instance \"$CONTAINER_ID\"",
        "no_user_account_api_key_copied_to_rented_hardware": True,
        "paid_vast_operation_performed": False,
        "vast_instance_created": False,
        "vast_instance_stopped_by_ticket": False,
        "vast_instance_destroyed": False,
        "data_uploaded": False,
        "local_workers_stopped_or_restarted": False,
        "ds26_interfered_with": False,
        "local_process_state_before": before.get("processes", []),
        "local_process_state_after": after.get("processes", []),
        "exact_next_powershell_command": remaining_user_actions()["exact_next_powershell_command"],
    }
    payload["terminal_hash"] = stable_hash(payload)
    return payload


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "14_local_state_before.json", before)
    predecessor = predecessor_validation(repo_root)
    original = original_failed_inspection()
    identity = instance_identity_contract()
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44e1_"))
    try:
        preflight = vast_control_preflight_contract(tmp_root)
        deadline = deadline_calculation_evidence()
        stop_evidence = synthetic_stop_invocation_evidence(tmp_root)
    finally:
        shutil.rmtree(openable_path(tmp_root), ignore_errors=True)
    smoke_manifest = build_smoke_data_manifest(repo_root)
    evidence_relative_root = str(R44E1_EVIDENCE_RELATIVE_ROOT).replace(os.sep, "/")
    write_guarded_scripts(
        evidence_root,
        evidence_relative_root,
        runbook_title="DS24 R44E1 Vast Self-Stop Billing Guard Runbook",
        smoke_bundle_size_bytes=smoke_manifest.get("total_size_bytes", ""),
    )
    script_excerpts = corrected_script_excerpts(evidence_relative_root)
    resilience = watchdog_resilience_contract()
    stopped_handling = stopped_instance_handling_contract()
    security = security_scan(repo_root, evidence_root)
    after = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after)
    tests = internal_required_tests(
        predecessor=predecessor,
        identity=identity,
        preflight=preflight,
        deadline=deadline,
        script_excerpts=script_excerpts,
        stop_evidence=stop_evidence,
        resilience=resilience,
        stopped_handling=stopped_handling,
        security=security,
        local=local,
    )
    terminal = terminal_result(
        evidence_root,
        predecessor=predecessor,
        identity=identity,
        preflight=preflight,
        deadline=deadline,
        stop_evidence=stop_evidence,
        script_excerpts=script_excerpts,
        security=security,
        tests=tests,
        local=local,
        before=before,
        after=after,
    )
    files = {
        "01_predecessor_validation.json": predecessor,
        "02_original_failed_inspection.json": original,
        "03_instance_identity_contract.json": identity,
        "04_vast_control_preflight_contract.json": preflight,
        "05_deadline_calculations.json": deadline,
        "06_checkpoint_flush_stop_ordering_proof.json": stop_evidence["ordering"],
        "07_corrected_script_excerpts.json": script_excerpts,
        "08_synthetic_stop_invocation_evidence.json": stop_evidence,
        "09_watchdog_resilience.json": resilience,
        "10_stopped_instance_handling.json": stopped_handling,
        "11_secret_scan.json": security,
        "12_focused_test_results.json": tests,
        "13_architecture_conformance.json": {"artifact_id": "DS24_R44E1_ARCHITECTURE_CONFORMANCE_V1", "status": "PENDING_EXTERNAL_COMMAND"},
        "15_local_state_after.json": after,
        "16_scoped_git_status.json": scoped_git_status(repo_root),
        "17_remaining_user_actions.json": remaining_user_actions(),
        "18_terminal_result.json": terminal,
    }
    for name, payload in files.items():
        write_json(evidence_root / name, payload)
    write_text(
        evidence_root / "README.md",
        f"""
        # DS24 R44E1 Vast Instance Self-Stop Billing Guard

        Terminal classification: `{terminal["terminal_classification"]}`

        This evidence root records the R44E operational inspection failure and
        the corrected self-stop package. The corrected watchdog resolves and
        validates `CONTAINER_ID`, enforces the earlier of the fixed $0.75
        initial-smoke cap and 90-minute absolute wall-clock cap, checkpoints,
        flushes V3 metrics and compact OOF V2 outputs, verifies the sync bundle,
        writes durable receipts, and executes `vastai stop instance "$CONTAINER_ID"`.

        No Vast resource was created, stopped or destroyed by this ticket, no
        paid endpoint was called, no user API key was accessed, no data was
        uploaded, no local DS24/DS26 process was stopped or altered, and no
        order path was touched.
        """,
    )
    return terminal


def record_validation_results(evidence_root: Path, *, py_compile: str, pytest: str, architecture: str) -> dict[str, Any]:
    tests = read_json(evidence_root / "12_focused_test_results.json")
    arch = {
        "artifact_id": "DS24_R44E1_ARCHITECTURE_CONFORMANCE_V1",
        "created_at_utc": utc_now(),
        "architecture_conformance": architecture,
        "status": "PASS" if architecture.startswith("PASS") and "cycles=0" in architecture else "FAIL",
    }
    arch["result_hash"] = stable_hash(arch)
    write_json(evidence_root / "13_architecture_conformance.json", arch)
    tests["py_compile"] = py_compile
    tests["focused_pytest"] = pytest
    tests["architecture_status"] = arch["status"]
    if isinstance(tests.get("tests"), dict):
        tests["tests"]["24_architecture_conformance"] = {
            "status": arch["status"],
            "architecture_conformance": architecture,
        }
    tests["updated_at_utc"] = utc_now()
    tests["status"] = "PASS" if tests.get("status") == "PASS" and py_compile.startswith("PASS") and pytest.startswith("PASS") and arch["status"] == "PASS" else "FAIL"
    tests["result_hash"] = stable_hash(tests)
    write_json(evidence_root / "12_focused_test_results.json", tests)
    terminal = read_json(evidence_root / "18_terminal_result.json")
    if terminal:
        terminal["validation_py_compile"] = py_compile
        terminal["validation_focused_pytest"] = pytest
        terminal["validation_architecture_conformance"] = architecture
        terminal["validation_updated_at_utc"] = utc_now()
        if tests["status"] == "PASS" and arch["status"] == "PASS" and terminal.get("success") is True:
            terminal["terminal_classification"] = TERMINAL_SUCCESS
            terminal["success"] = True
        else:
            terminal["terminal_classification"] = BLOCKED_TEST_ARCH
            terminal["success"] = False
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "18_terminal_result.json", terminal)
    return {"status": tests["status"], "architecture_status": arch["status"]}


def record_final_state(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    before = read_json(evidence_root / "14_local_state_before.json")
    after = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after)
    write_json(evidence_root / "15_local_state_after.json", after)
    write_json(evidence_root / "16_scoped_git_status.json", scoped_git_status(repo_root))
    terminal = read_json(evidence_root / "18_terminal_result.json")
    if terminal:
        terminal["local_process_state_after"] = after.get("processes", [])
        terminal["local_interference_status"] = local["status"]
        terminal["final_state_updated_at_utc"] = utc_now()
        if local["status"] != "PASS":
            terminal["terminal_classification"] = BLOCKED_TEST_ARCH
            terminal["success"] = False
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "18_terminal_result.json", terminal)
    return {
        "status": local["status"],
        "after_process_count": len(after.get("processes", [])),
        "ds24_process_count": len(after.get("ds24_processes", [])),
        "ds26_process_count": len(after.get("ds26_processes", [])),
        "disk_free_bytes": after.get("disk", {}).get("free_bytes", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44E1 Vast self-stop billing guard package")
    sub = parser.add_subparsers(dest="command")
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--evidence-root", default=str(R44E1_EVIDENCE_RELATIVE_ROOT))
    stamp = sub.add_parser("record-validation")
    stamp.add_argument("--evidence-root", default=str(R44E1_EVIDENCE_RELATIVE_ROOT))
    stamp.add_argument("--py-compile", required=True)
    stamp.add_argument("--pytest", required=True)
    stamp.add_argument("--architecture", required=True)
    final = sub.add_parser("record-final-state")
    final.add_argument("--repo-root", default=".")
    final.add_argument("--evidence-root", default=str(R44E1_EVIDENCE_RELATIVE_ROOT))
    args = parser.parse_args(argv)

    if args.command in {None, "package"}:
        terminal = write_package(Path(args.repo_root).resolve(), Path(args.evidence_root))
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0 if terminal["success"] else 2
    if args.command == "record-validation":
        result = record_validation_results(Path(args.evidence_root), py_compile=args.py_compile, pytest=args.pytest, architecture=args.architecture)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" and result["architecture_status"] == "PASS" else 2
    if args.command == "record-final-state":
        result = record_final_state(Path(args.repo_root).resolve(), Path(args.evidence_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
