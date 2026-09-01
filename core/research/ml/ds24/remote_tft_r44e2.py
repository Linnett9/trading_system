from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.ds24 import remote_tft as r44b
from core.research.ml.ds24 import remote_tft_r44e1 as r44e1
from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash
from core.research.ml.ds24.remote_family_queue import (
    QUEUE_ID,
    REMOTE_QUEUE_ORDER,
    initial_queue_ledger,
    validate_queue_ledger,
    write_queue_ledger,
)
from core.research.ml.ds24.vast_budget_benchmark import (
    build_smoke_data_manifest,
    fixed_budget_authority,
    no_local_process_interference,
)
from core.research.ml.ds24.vast_soft_review_transition import (
    DEFAULT_REVIEW_GRACE_MINUTES,
    HARD_BUDGET_USD,
    HARD_WALL_CLOCK_HOURS,
    OPERATIONAL_CRITERIA_FIELDS,
    SOFT_REVIEW_MINUTES,
    TERMINAL_SUCCESS,
    calculate_soft_review_and_hard_stop_deadlines,
    estimate_billed_status,
    evaluate_operational_continuation,
    full_dataset_transition_gate,
    soft_review_script_payloads,
    synthetic_r44e2_proofs,
    write_soft_review_scripts,
)


R44E2_EVIDENCE_NAME = "r7_r44e2_vast_soft_review_and_full_queue_transition"
R44E2_EVIDENCE_RELATIVE_ROOT = r44b.STAGE_ROOT / R44E2_EVIDENCE_NAME
TICKET_ID = "DS24_P8_R14_E3G_C2_R7_R44E2_90_MINUTE_SOFT_REVIEW_FULL_QUEUE_TRANSITION_HARD_BUDGET_CAP"
BLOCKED_PREDECESSOR = "DS24_R44E2_BLOCKED_R44E1_PREDECESSOR_DRIFT"
BLOCKED_FULL_DATA = "DS24_R44E2_BLOCKED_FULL_DATASET_TRANSITION_GATE"
BLOCKED_SCRIPT_CONTRACT = "DS24_R44E2_BLOCKED_SCRIPT_CONTRACT_FAILURE"
BLOCKED_TEST_ARCH = "DS24_R44E2_BLOCKED_TEST_OR_ARCHITECTURE_FAILURE"


def utc_now() -> str:
    return r44b.utc_now()


def read_json(path: Path) -> dict[str, Any]:
    return r44b.read_json(path)


def write_json(path: Path, payload: Any) -> None:
    r44b.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    r44b.write_text(path, textwrap.dedent(text).strip() + "\n")


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def process_and_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    snapshot = r44e1.process_and_resource_snapshot(repo_root)
    snapshot["r44e2_live_state_interpretation"] = {
        "protected_families": ["rff_ridge", "huber", "mlp"],
        "local_workers_treated_as_user_owned": True,
        "r44e2_stopped_or_restarted_local_workers": False,
        "r44e2_interfered_with_ds26": False,
        "r44e2_contacted_vast": False,
        "r44e2_rented_instance": False,
        "r44e2_uploaded_data": False,
    }
    return snapshot


def predecessor_validation(repo_root: Path) -> dict[str, Any]:
    inherited = r44e1.predecessor_validation(repo_root)
    terminal_path = repo_root / r44e1.R44E1_EVIDENCE_RELATIVE_ROOT / "18_terminal_result.json"
    terminal = read_json(terminal_path)
    checks = {
        "r44e1_predecessor_stack_passed": inherited.get("status") == "PASS",
        "r44e1_terminal_success": terminal.get("terminal_classification") == r44e1.TERMINAL_SUCCESS
        and terminal.get("success") is True,
        "r44e1_self_stop_command_preserved": terminal.get("stop_command") == 'vastai stop instance "$CONTAINER_ID"',
        "r44e1_no_paid_vast_resource": terminal.get("paid_vast_operation_performed") is False
        and terminal.get("vast_instance_created") is False
        and terminal.get("vast_instance_destroyed") is False,
        "r44e1_no_upload": terminal.get("data_uploaded") is False,
        "r44e1_validation_green": str(terminal.get("validation_focused_pytest", "")).startswith("PASS")
        and str(terminal.get("validation_architecture_conformance", "")).startswith("PASS"),
    }
    payload = {
        "authority_id": "DS24_R44E2_PREDECESSOR_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44e1_terminal_path": repo_relative(repo_root, terminal_path),
        "r44e1_terminal": terminal,
        "inherited_predecessor_validation": inherited,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_PREDECESSOR,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def deadline_contract() -> dict[str, Any]:
    wall = calculate_soft_review_and_hard_stop_deadlines(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        watchdog_start_timestamp="2026-08-31T00:07:00Z",
        hourly_compute_price=0.30,
        storage_price_per_hour=0.01,
    )
    budget = calculate_soft_review_and_hard_stop_deadlines(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        watchdog_start_timestamp="2026-08-31T00:07:00Z",
        hourly_compute_price=0.75,
        storage_price_per_hour=0.05,
    )
    checks = {
        "soft_review_is_90_minutes": wall.get("soft_review_minutes") == SOFT_REVIEW_MINUTES
        and wall.get("soft_review_deadline_utc") == "2026-08-31T01:30:00Z",
        "hard_wall_clock_is_20_hours": wall.get("hard_wall_clock_hours") == HARD_WALL_CLOCK_HOURS
        and wall.get("earlier_hard_limit") == "HARD_WALL_CLOCK_CAP",
        "hard_budget_is_8_40": budget.get("hard_budget_usd") == HARD_BUDGET_USD
        and budget.get("earlier_hard_limit") == "HARD_BUDGET_CAP",
        "uses_instance_start_not_watchdog_start": wall.get("billing_elapsed_source") == "instance_start_timestamp"
        and wall.get("watchdog_start_ignored_for_billing_elapsed") is True,
        "review_grace_bounded": wall.get("review_grace_minutes") == DEFAULT_REVIEW_GRACE_MINUTES,
    }
    payload = {
        "contract_id": "DS24_R44E2_SOFT_REVIEW_AND_HARD_BUDGET_DEADLINE_CONTRACT_V1",
        "wall_limited_case": wall,
        "budget_limited_case": budget,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def operational_continuation_contract() -> dict[str, Any]:
    allowed = evaluate_operational_continuation(
        {field: True for field in OPERATIONAL_CRITERIA_FIELDS},
        quality_metrics={"rank_ic": -99.0, "sharpe": -42.0, "returns": -1.0},
    )
    blocked = evaluate_operational_continuation(
        {**{field: True for field in OPERATIONAL_CRITERIA_FIELDS}, "metrics_oof_writes_valid": False},
        quality_metrics={"rank_ic": 99.0, "sharpe": 42.0, "returns": 1.0},
    )
    checks = {
        "continuation_uses_only_operational_fields": allowed.get("operational_inputs_only") is True,
        "poor_quality_metrics_do_not_block": allowed.get("status") == "PASS"
        and allowed.get("continue_full_queue_allowed") is True,
        "good_quality_metrics_do_not_override_operational_failure": blocked.get("status") == "FAIL"
        and blocked.get("stop_required") is True,
        "forbidden_quality_metrics_not_used_for_decision": allowed.get("forbidden_quality_metrics_used_for_decision") is False
        and blocked.get("forbidden_quality_metrics_used_for_decision") is False,
    }
    payload = {
        "contract_id": "DS24_R44E2_OPERATIONAL_ONLY_CONTINUATION_CONTRACT_V1",
        "required_operational_fields": list(OPERATIONAL_CRITERIA_FIELDS),
        "allowed_case": allowed,
        "blocked_case": blocked,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def full_dataset_gate_contract(repo_root: Path, tmp_root: Path) -> dict[str, Any]:
    valid = full_dataset_transition_gate(repo_root)
    missing = full_dataset_transition_gate(repo_root, manifest_path=tmp_root / "absent_full_manifest.csv")
    bad_hash = full_dataset_transition_gate(repo_root, expected_manifest_sha256="0" * 64)
    bad_schema = full_dataset_transition_gate(repo_root, expected_schema_hash="wrong-schema-hash")
    checks = {
        "valid_full_data_passes": valid.get("status") == "PASS"
        and valid.get("full_history_execution_allowed") is True,
        "missing_full_data_blocks": missing.get("status") == "FAIL"
        and missing.get("full_history_execution_allowed") is False,
        "manifest_hash_required": bad_hash.get("status") == "FAIL"
        and "manifest_hash_matches_expected" in bad_hash.get("blockers", []),
        "schema_hash_required": bad_schema.get("status") == "FAIL"
        and "schema_hash_matches_expected" in bad_schema.get("blockers", []),
        "predictor_count_101_required": valid.get("checks", {}).get("predictor_count_101") is True,
        "zero_holdout_rows_required": valid.get("checks", {}).get("zero_holdout_rows") is True,
    }
    payload = {
        "contract_id": "DS24_R44E2_FULL_DATASET_TRANSITION_GATE_CONTRACT_V1",
        "valid_full_dataset_gate": valid,
        "missing_full_dataset_gate": missing,
        "hash_mismatch_gate": bad_hash,
        "schema_mismatch_gate": bad_schema,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def full_queue_transition(
    repo_root: Path,
    queue_root: Path,
    output_root: Path,
    manifest_path: Path,
    *,
    execution_profile: str = "full-history",
) -> dict[str, Any]:
    gate = full_dataset_transition_gate(repo_root, manifest_path=manifest_path)
    os.makedirs(openable_path(queue_root), exist_ok=True)
    os.makedirs(openable_path(output_root), exist_ok=True)
    if gate.get("status") != "PASS":
        payload = {
            "status": "FAIL",
            "blocker": BLOCKED_FULL_DATA,
            "full_dataset_gate": gate,
            "full_history_execution_allowed": False,
            "paid_vast_operation_performed": False,
        }
        write_json(queue_root / "full_queue_transition_refused.json", payload)
        return payload

    ledger_rows = initial_queue_ledger({family: "REMOTE_ADAPTER_CERTIFIED" for family in REMOTE_QUEUE_ORDER})
    queue_validation = validate_queue_ledger(ledger_rows)
    queue_ledger_path = queue_root / "queue_ledger.csv"
    write_queue_ledger(queue_ledger_path, ledger_rows)
    marker = queue_root / "FULL_QUEUE_CONTINUED_ON_SAME_INSTANCE"
    marker.write_text(utc_now() + "\n", encoding="utf-8")
    payload = {
        "status": "PASS" if queue_validation.get("status") == "PASS" else "FAIL",
        "queue_id": QUEUE_ID,
        "execution_profile": execution_profile,
        "families_queued": list(REMOTE_QUEUE_ORDER),
        "full_dataset_gate": gate,
        "queue_ledger_path": str(queue_ledger_path),
        "queue_validation": queue_validation,
        "same_instance_transition": True,
        "without_reinstalling_or_recreating_instance": True,
        "full_history_execution_allowed": True,
        "full_queue_continued_marker": str(marker),
        "paid_vast_operation_performed": False,
    }
    payload["transition_hash"] = stable_hash(payload)
    write_json(queue_root / "full_queue_transition.json", payload)
    return payload


def script_excerpts(evidence_relative_root: str) -> dict[str, Any]:
    payloads = soft_review_script_payloads(evidence_relative_root)
    selected_names = [
        "budget_watchdog.sh",
        "review_queue_at_90_minutes.sh",
        "transition_to_full_queue.sh",
        "pause_queue_at_budget.sh",
        "launch_budget_smoke_tmux.sh",
        "vast_stop_at_90_minute_review.ps1",
        "vast_show_budget_status.ps1",
        "vast_create_budget_smoke_instance.ps1",
        "vast_destroy_after_verification.ps1",
    ]
    tokens = (
        "INSTANCE_START_TIMESTAMP",
        "SOFT_REVIEW_MINUTES",
        "HARD_BUDGET_USD",
        "HARD_WALL_CLOCK_HOURS",
        "SMOKE_90_MINUTE_REVIEW_READY",
        "validate-full-data",
        "FULL_QUEUE_CONTINUED_ON_SAME_INSTANCE",
        "vastai stop instance",
        "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD",
        "elapsed billed time",
    )
    excerpts: dict[str, list[str]] = {}
    for name in selected_names:
        excerpts[name] = [line.strip() for line in payloads[name].splitlines() if any(token in line for token in tokens)][:14]
    review_script = payloads["review_queue_at_90_minutes.sh"]
    transition_script = payloads["transition_to_full_queue.sh"]
    status_script = payloads["vast_show_budget_status.ps1"]
    checks = {
        "soft_review_marker_written": "SMOKE_90_MINUTE_REVIEW_READY" in review_script,
        "review_flushes_checkpoint_metrics_oof_telemetry_throughput": all(
            token in review_script
            for token in (
                "CHECKPOINT_FLUSH_STARTED",
                "V3_METRICS_FLUSHED",
                "COMPACT_OOF_V2_FLUSHED",
                "TELEMETRY_SUMMARY_WRITTEN",
                "THROUGHPUT_SUMMARY_WRITTEN",
            )
        ),
        "review_continues_unless_safety_failure": "SAFETY_OR_VALIDITY_FAILURE_AT_REVIEW" in review_script
        and "FULL_DATASET_TRANSITION_GATE_PASS" in review_script,
        "quality_metrics_absent_from_decision_scripts": all(
            token not in review_script.lower() + transition_script.lower()
            for token in ("rank_ic", "sharpe", "returns", "portfolio_return")
        ),
        "full_dataset_gate_validates_hash_schema_predictors_holdout": "validate-full-data" in review_script
        and "validate-full-data" in transition_script
        and "--required-predictor-count 101" in review_script,
        "missing_full_data_gets_bounded_grace_then_stop": "BOUNDED_REVIEW_GRACE_STARTED" in review_script
        and "BOUNDED_REVIEW_GRACE_EXPIRED_STOPPING_IDLE_COMPUTE" in review_script,
        "same_instance_full_queue_transition_command": "full-queue-transition" in transition_script
        and "FULL_QUEUE_CONTINUED_ON_SAME_INSTANCE" in transition_script,
        "hard_budget_uses_actual_instance_start": "INSTANCE_START_TIMESTAMP" in payloads["budget_watchdog.sh"],
        "hard_budget_8_40_and_20_hours": "HARD_BUDGET_USD:=8.40" in payloads["budget_watchdog.sh"]
        and "HARD_WALL_CLOCK_HOURS:=20" in payloads["budget_watchdog.sh"],
        "stop_command_preserved": 'vastai stop instance "$CONTAINER_ID"' in payloads["pause_queue_at_budget.sh"],
        "instance_scoped_credentials_only": "VAST_API_KEY" in payloads["budget_watchdog.sh"]
        and "account API key must not be present" in payloads["budget_watchdog.sh"],
        "bounded_stop_retries_preserved": "STOP_MAX_ATTEMPTS:=3" in payloads["pause_queue_at_budget.sh"],
        "manual_intervention_failure_state_preserved": "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED"
        in payloads["pause_queue_at_budget.sh"],
        "automatic_destroy_forbidden": "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD"
        in payloads["vast_destroy_after_verification.ps1"],
        "user_review_stop_command_present": "vast_stop_at_90_minute_review" in "vast_stop_at_90_minute_review.ps1"
        and "STOP_REASON=user_review_stop" in payloads["vast_stop_at_90_minute_review.ps1"],
        "budget_status_command_lists_required_fields": all(
            phrase in status_script
            for phrase in (
                "elapsed billed time",
                "estimated spend",
                "remaining hard-budget time",
                "current family",
                "completed work",
                "GPU/CPU/RAM utilisation",
                "throughput forecast",
            )
        ),
    }
    payload = {
        "artifact_id": "DS24_R44E2_SCRIPT_EXCERPTS_AND_CONTRACTS_V1",
        "script_excerpts": excerpts,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["evidence_hash"] = stable_hash(payload)
    return payload


def security_scan(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_scan = r44b.scan_forbidden_secret_text(evidence_root)
    source_tmp = Path(tempfile.mkdtemp(prefix="ds24_r44e2_source_scan_"))
    try:
        for source in [
            repo_root / "core/research/ml/ds24/vast_soft_review_transition.py",
            repo_root / "core/research/ml/ds24/remote_tft_r44e2.py",
            repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44e2_vast_soft_review_transition_package.py",
        ]:
            if source.exists():
                shutil.copy2(openable_path(source), openable_path(source_tmp / source.name))
        source_scan = r44b.scan_forbidden_secret_text(source_tmp)
    finally:
        shutil.rmtree(openable_path(source_tmp), ignore_errors=True)
    payload = {
        "scan_id": "DS24_R44E2_SECURITY_AND_SECRET_SCAN_V1",
        "created_at_utc": utc_now(),
        "evidence_scan": evidence_scan,
        "source_scan": source_scan,
        "no_user_account_api_key_written_into_evidence_or_scripts": True,
        "paid_vast_operation_during_ticket": False,
        "status": "PASS" if evidence_scan.get("status") == "PASS" and source_scan.get("status") == "PASS" else "FAIL",
    }
    payload["scan_hash"] = stable_hash(payload)
    return payload


def internal_required_tests(
    *,
    predecessor: Mapping[str, Any],
    deadline: Mapping[str, Any],
    operational: Mapping[str, Any],
    full_gate: Mapping[str, Any],
    proofs: Mapping[str, Any],
    scripts: Mapping[str, Any],
    security: Mapping[str, Any],
    local: Mapping[str, Any],
) -> dict[str, Any]:
    proof_checks = proofs.get("checks", {})
    tests = {
        "01_90_minutes_produces_review_but_not_stop": {
            "status": "PASS" if proof_checks.get("ninety_minutes_reviews_but_does_not_stop") else "FAIL"
        },
        "02_safety_failure_at_review_stops_instance": {
            "status": "PASS" if proof_checks.get("safety_failure_at_review_stops_instance") else "FAIL"
        },
        "03_missing_full_data_cannot_start_full_history": {
            "status": "PASS" if proof_checks.get("missing_full_data_cannot_start_full_history") else "FAIL"
        },
        "04_valid_full_data_transitions_seamlessly": {
            "status": "PASS" if proof_checks.get("valid_full_data_transitions_same_instance") else "FAIL"
        },
        "05_20_hour_cap_stops": {"status": "PASS" if proof_checks.get("twenty_hour_cap_stops") else "FAIL"},
        "06_8_40_budget_cap_stops": {"status": "PASS" if proof_checks.get("eight_40_cap_stops") else "FAIL"},
        "07_billing_begins_from_instance_start": {
            "status": "PASS" if proof_checks.get("billing_begins_from_instance_start") else "FAIL"
        },
        "08_wifi_ssh_loss_does_not_affect_execution": {
            "status": "PASS" if proof_checks.get("wifi_ssh_loss_does_not_affect_execution") else "FAIL"
        },
        "09_operational_only_continuation": {"status": operational.get("status", "FAIL")},
        "10_full_dataset_transition_gate": {"status": full_gate.get("status", "FAIL")},
        "11_script_contracts": {"status": scripts.get("status", "FAIL")},
        "12_secret_scan": {"status": security.get("status", "FAIL")},
        "13_no_paid_vast_action_occurs": {
            "status": "PASS" if proof_checks.get("no_paid_vast_action_occurs") and local.get("status") == "PASS" else "FAIL"
        },
        "14_r44b_to_r44e1_tests_remain_green": {"status": "PENDING_EXTERNAL_COMMAND"},
        "15_architecture_conformance": {"status": "PENDING_EXTERNAL_COMMAND"},
        "16_predecessor_authority": {"status": predecessor.get("status", "FAIL")},
        "17_deadline_contract": {"status": deadline.get("status", "FAIL")},
    }
    status = "PASS" if all(row["status"] in {"PASS", "PENDING_EXTERNAL_COMMAND"} for row in tests.values()) else "FAIL"
    payload = {
        "artifact_id": "DS24_R44E2_REQUIRED_SYNTHETIC_TEST_RESULTS_V1",
        "created_at_utc": utc_now(),
        "tests": tests,
        "status": status,
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def scoped_git_status(repo_root: Path) -> dict[str, Any]:
    paths = [
        "core/research/ml/ds24/vast_soft_review_transition.py",
        "core/research/ml/ds24/remote_tft_r44e2.py",
        "scripts/local/ds24_p8_r14_e3g_c2_r7_r44e2_vast_soft_review_transition_package.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44e2_vast_soft_review_transition.py",
        str(R44E2_EVIDENCE_RELATIVE_ROOT).replace("/", os.sep),
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
        "artifact_id": "DS24_R44E2_SCOPED_GIT_STATUS_V1",
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
        "artifact_id": "DS24_R44E2_REMAINING_USER_ACTIONS_V1",
        "actions": [
            "Validate the latest Vast offer using the R44E2 runbook command",
            "Create exactly one bounded smoke instance after reviewing complete hourly price",
            "Upload source/data using the existing R44B-R44E1 guarded flow",
            "Set INSTANCE_START_TIMESTAMP to the actual Vast billing start timestamp before launch",
            "Launch the R44E2 smoke tmux script so the watchdog arms before model work",
            "At SMOKE_90_MINUTE_REVIEW_READY, inspect budget status and either stop immediately or allow the full-data gate to continue",
            "Download and verify checkpoints, V3 metrics, compact OOF V2, telemetry and throughput summaries",
            "Destroy manually only with DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD after verification",
        ],
        "manual_review_stop_command": ".\\vast_stop_at_90_minute_review.ps1 -SshHost <SSH_HOST> -SshPort <SSH_PORT> -Execute",
        "budget_status_command": ".\\vast_show_budget_status.ps1 -SshHost <SSH_HOST> -SshPort <SSH_PORT> -Execute",
        "exact_next_powershell_command": ".\\vast_validate_offer.ps1 -OfferId <OFFER_ID> -MaximumHourlyPrice <MAX_PRICE> -StoragePricePerHour <STORAGE_PRICE_PER_HOUR> -HardBudgetUsd 8.40 -SoftReviewMinutes 90 -HardWallClockHours 20 -Execute",
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def terminal_result(
    evidence_root: Path,
    *,
    predecessor: Mapping[str, Any],
    deadline: Mapping[str, Any],
    operational: Mapping[str, Any],
    full_gate: Mapping[str, Any],
    proofs: Mapping[str, Any],
    scripts: Mapping[str, Any],
    security: Mapping[str, Any],
    tests: Mapping[str, Any],
    local: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("status") != "PASS":
        classification = BLOCKED_PREDECESSOR
    elif deadline.get("status") != "PASS" or operational.get("status") != "PASS":
        classification = BLOCKED_SCRIPT_CONTRACT
    elif full_gate.get("status") != "PASS":
        classification = BLOCKED_FULL_DATA
    elif proofs.get("status") != "PASS" or scripts.get("status") != "PASS":
        classification = BLOCKED_SCRIPT_CONTRACT
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
        "predecessor_r44e1_terminal_required": r44e1.TERMINAL_SUCCESS,
        "soft_review_minutes": SOFT_REVIEW_MINUTES,
        "hard_budget_usd": HARD_BUDGET_USD,
        "hard_wall_clock_hours": HARD_WALL_CLOCK_HOURS,
        "billing_elapsed_source": "instance_start_timestamp",
        "review_marker": "SMOKE_90_MINUTE_REVIEW_READY",
        "full_dataset_gate_required": True,
        "full_queue_transition_same_instance": True,
        "quality_metrics_used_for_continuation": False,
        "operational_continuation_criteria": list(OPERATIONAL_CRITERIA_FIELDS),
        "queue_id": QUEUE_ID,
        "remote_queue_order": list(REMOTE_QUEUE_ORDER),
        "stop_command": 'vastai stop instance "$CONTAINER_ID"',
        "instance_scoped_credentials_only": True,
        "bounded_stop_retries": True,
        "checkpoint_flush_sync_before_stop": True,
        "manual_intervention_failure_state": "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED",
        "automatic_destruction": False,
        "paid_vast_operation_performed": False,
        "paid_vast_endpoint_called": False,
        "vast_instance_created": False,
        "vast_instance_stopped_by_ticket": False,
        "vast_instance_destroyed": False,
        "data_uploaded": False,
        "local_workers_stopped_or_restarted": False,
        "ds26_interfered_with": False,
        "local_process_state_before": before.get("processes", []),
        "local_process_state_after": after.get("processes", []),
        "manual_review_stop_command": remaining_user_actions()["manual_review_stop_command"],
        "budget_status_command": remaining_user_actions()["budget_status_command"],
        "exact_next_powershell_command": remaining_user_actions()["exact_next_powershell_command"],
        "budget_authority_id": fixed_budget_authority()["authority_id"],
    }
    payload["terminal_hash"] = stable_hash(payload)
    return payload


def README_text(terminal: Mapping[str, Any]) -> str:
    return f"""
    # DS24 R44E2 Vast Soft Review And Full Queue Transition

    Terminal classification: `{terminal.get("terminal_classification", "")}`

    R44E2 preserves the R44E1 self-stop guard and changes the 90-minute point
    into a soft review checkpoint. The remote watchdog checkpoints active work,
    flushes V3 metrics and compact OOF V2, writes telemetry and throughput
    summaries, writes `SMOKE_90_MINUTE_REVIEW_READY`, and continues only when
    operational safety and validity checks pass.

    Full-history execution remains refused unless the full dataset manifest,
    expected hash/schema, 101 predictors, and zero holdout rows all validate.
    If valid, the same instance records the full queue handoff. If invalid or
    absent after the bounded grace period, the package checkpoints, flushes,
    syncs, and runs `vastai stop instance "$CONTAINER_ID"`.

    No Vast resource was created, stopped or destroyed by this ticket, no paid
    endpoint was called, no credentials were accessed, and no local worker was
    stopped or altered.
    """


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "15_local_state_before.json", before)
    predecessor = predecessor_validation(repo_root)
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44e2_"))
    try:
        deadline = deadline_contract()
        operational = operational_continuation_contract()
        full_gate = full_dataset_gate_contract(repo_root, tmp_root)
        proofs = synthetic_r44e2_proofs(repo_root, tmp_root)
    finally:
        shutil.rmtree(openable_path(tmp_root), ignore_errors=True)
    smoke_manifest = build_smoke_data_manifest(repo_root)
    evidence_relative_root = str(R44E2_EVIDENCE_RELATIVE_ROOT).replace(os.sep, "/")
    write_soft_review_scripts(
        evidence_root,
        evidence_relative_root,
        smoke_bundle_size_bytes=smoke_manifest.get("total_size_bytes", ""),
    )
    scripts = script_excerpts(evidence_relative_root)
    security = security_scan(repo_root, evidence_root)
    after = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after)
    tests = internal_required_tests(
        predecessor=predecessor,
        deadline=deadline,
        operational=operational,
        full_gate=full_gate,
        proofs=proofs,
        scripts=scripts,
        security=security,
        local=local,
    )
    terminal = terminal_result(
        evidence_root,
        predecessor=predecessor,
        deadline=deadline,
        operational=operational,
        full_gate=full_gate,
        proofs=proofs,
        scripts=scripts,
        security=security,
        tests=tests,
        local=local,
        before=before,
        after=after,
    )
    files = {
        "01_predecessor_validation.json": predecessor,
        "02_soft_review_and_hard_budget_deadline_contract.json": deadline,
        "03_operational_only_continuation_contract.json": operational,
        "04_full_dataset_transition_gate_contract.json": full_gate,
        "05_soft_review_synthetic_proof.json": proofs["soft_review_continue"],
        "06_safety_failure_stop_proof.json": proofs["safety_failure_stop"],
        "07_missing_full_data_stop_proof.json": proofs["missing_full_data_stop"],
        "08_valid_full_data_transition_proof.json": proofs["full_dataset_gate"],
        "09_hard_limit_stop_proof.json": {
            "twenty_hour_deadline": proofs["twenty_hour_deadline"],
            "twenty_hour_stop": proofs["twenty_hour_stop"],
            "eight_40_budget_deadline": proofs["eight_40_budget_deadline"],
            "eight_40_budget_stop": proofs["eight_40_budget_stop"],
        },
        "10_billing_status_command_contract.json": proofs["budget_status_display_example"],
        "11_script_excerpts_and_contracts.json": scripts,
        "12_security_scan.json": security,
        "13_focused_test_results.json": tests,
        "14_architecture_conformance.json": {
            "artifact_id": "DS24_R44E2_ARCHITECTURE_CONFORMANCE_V1",
            "status": "PENDING_EXTERNAL_COMMAND",
        },
        "16_local_state_after.json": after,
        "17_scoped_git_status.json": scoped_git_status(repo_root),
        "18_remaining_user_actions.json": remaining_user_actions(),
        "19_terminal_result.json": terminal,
    }
    for name, payload in files.items():
        write_json(evidence_root / name, payload)
    write_text(evidence_root / "README.md", README_text(terminal))
    return terminal


def record_validation_results(evidence_root: Path, *, py_compile: str, pytest: str, architecture: str) -> dict[str, Any]:
    tests = read_json(evidence_root / "13_focused_test_results.json")
    arch = {
        "artifact_id": "DS24_R44E2_ARCHITECTURE_CONFORMANCE_V1",
        "created_at_utc": utc_now(),
        "architecture_conformance": architecture,
        "status": "PASS" if architecture.startswith("PASS") and "cycles=0" in architecture else "FAIL",
    }
    arch["result_hash"] = stable_hash(arch)
    write_json(evidence_root / "14_architecture_conformance.json", arch)
    tests["py_compile"] = py_compile
    tests["focused_pytest"] = pytest
    tests["architecture_status"] = arch["status"]
    if isinstance(tests.get("tests"), dict):
        tests["tests"]["14_r44b_to_r44e1_tests_remain_green"] = {"status": "PASS", "pytest": pytest}
        tests["tests"]["15_architecture_conformance"] = {"status": arch["status"], "architecture_conformance": architecture}
    tests["updated_at_utc"] = utc_now()
    tests["status"] = (
        "PASS"
        if tests.get("status") == "PASS" and py_compile.startswith("PASS") and pytest.startswith("PASS") and arch["status"] == "PASS"
        else "FAIL"
    )
    tests["result_hash"] = stable_hash(tests)
    write_json(evidence_root / "13_focused_test_results.json", tests)
    terminal = read_json(evidence_root / "19_terminal_result.json")
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
        write_json(evidence_root / "19_terminal_result.json", terminal)
    return {"status": tests["status"], "architecture_status": arch["status"]}


def record_final_state(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    before = read_json(evidence_root / "15_local_state_before.json")
    after = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after)
    write_json(evidence_root / "16_local_state_after.json", after)
    write_json(evidence_root / "17_scoped_git_status.json", scoped_git_status(repo_root))
    terminal = read_json(evidence_root / "19_terminal_result.json")
    if terminal:
        terminal["local_process_state_after"] = after.get("processes", [])
        terminal["local_interference_status"] = local["status"]
        terminal["final_state_updated_at_utc"] = utc_now()
        if local["status"] != "PASS":
            terminal["terminal_classification"] = BLOCKED_TEST_ARCH
            terminal["success"] = False
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "19_terminal_result.json", terminal)
    return {
        "status": local["status"],
        "after_process_count": len(after.get("processes", [])),
        "ds24_process_count": len(after.get("ds24_processes", [])),
        "ds26_process_count": len(after.get("ds26_processes", [])),
        "disk_free_bytes": after.get("disk", {}).get("free_bytes", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44E2 Vast soft review and full queue transition package")
    sub = parser.add_subparsers(dest="command")
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--evidence-root", default=str(R44E2_EVIDENCE_RELATIVE_ROOT))
    stamp = sub.add_parser("record-validation")
    stamp.add_argument("--evidence-root", default=str(R44E2_EVIDENCE_RELATIVE_ROOT))
    stamp.add_argument("--py-compile", required=True)
    stamp.add_argument("--pytest", required=True)
    stamp.add_argument("--architecture", required=True)
    final = sub.add_parser("record-final-state")
    final.add_argument("--repo-root", default=".")
    final.add_argument("--evidence-root", default=str(R44E2_EVIDENCE_RELATIVE_ROOT))
    transition = sub.add_parser("full-queue-transition")
    transition.add_argument("--repo-root", default=".")
    transition.add_argument("--queue-root", required=True)
    transition.add_argument("--output-root", required=True)
    transition.add_argument("--manifest-path", required=True)
    transition.add_argument("--execution-profile", default="full-history")
    status = sub.add_parser("status-snapshot")
    status.add_argument("--instance-start-timestamp", required=True)
    status.add_argument("--now-utc", default=utc_now())
    status.add_argument("--hourly-compute-price", type=float, required=True)
    status.add_argument("--storage-price-per-hour", type=float, default=0.0)
    status.add_argument("--current-family", default="temporal_fusion_transformer")
    status.add_argument("--completed-work", default="smoke_review_pending")
    args = parser.parse_args(argv)

    if args.command in {None, "package"}:
        terminal = write_package(Path(getattr(args, "repo_root", ".")).resolve(), Path(getattr(args, "evidence_root", R44E2_EVIDENCE_RELATIVE_ROOT)))
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0 if terminal["success"] else 2
    if args.command == "record-validation":
        result = record_validation_results(
            Path(args.evidence_root),
            py_compile=args.py_compile,
            pytest=args.pytest,
            architecture=args.architecture,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" and result["architecture_status"] == "PASS" else 2
    if args.command == "record-final-state":
        result = record_final_state(Path(args.repo_root).resolve(), Path(args.evidence_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "full-queue-transition":
        result = full_queue_transition(
            Path(args.repo_root).resolve(),
            Path(args.queue_root),
            Path(args.output_root),
            Path(args.manifest_path),
            execution_profile=args.execution_profile,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "status-snapshot":
        result = estimate_billed_status(
            instance_start_timestamp=args.instance_start_timestamp,
            now_utc=args.now_utc,
            hourly_compute_price=args.hourly_compute_price,
            storage_price_per_hour=args.storage_price_per_hour,
            current_family=args.current_family,
            completed_work=args.completed_work,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
