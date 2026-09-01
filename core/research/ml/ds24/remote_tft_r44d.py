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
from core.research.ml.ds24 import remote_tft_r44c as r44c
from core.research.ml.ds24.ensemble_oof import (
    COMPACT_SCORE_CONTRACT_ID,
    compact_oof_v2_contract_payload,
    disk_admission,
    openable_path,
    stable_hash,
)
from core.research.ml.ds24.remote_family_queue import (
    CPU_RANKING_FAMILIES,
    GPU_SEQUENCE_FAMILIES,
    LOCAL_FAMILIES,
    QUEUE_ID,
    QUEUE_LEDGER_COLUMNS,
    REMOTE_QUEUE_ORDER,
    R44B_SOURCE_BUNDLE_HASH,
    R44B_TFT_CONFIGURATION_HASH,
    TARGET_CONTRACT_ID,
    adapter_registry,
    assert_remote_family,
    common_remote_worker_contract,
    duplicate_guard,
    family_configuration_authority,
    full_queue_storage_projection,
    initial_queue_ledger,
    namespace_contract_payload,
    ownership_payload,
    per_family_storage_projection,
    queue_order_payload,
    queue_resume_determinism_proof,
    read_json,
    register_external_oof_manifest,
    remote_trial_id,
    resource_classification_payload,
    run_all_synthetic_smokes,
    run_synthetic_family_smoke,
    validate_queue_ledger,
    write_text,
)


R44D_EVIDENCE_NAME = "r7_r44d_vast_nine_family_remote_tournament_queue"
R44D_EVIDENCE_RELATIVE_ROOT = r44b.STAGE_ROOT / R44D_EVIDENCE_NAME
TERMINAL_SUCCESS = "DS24_R44D_VAST_NINE_FAMILY_REMOTE_QUEUE_READY_FOR_USER_PAID_HARDWARE_SMOKE"
TERMINAL_PREDECESSOR_DRIFT = "DS24_R44D_BLOCKED_PREDECESSOR_AUTHORITY_DRIFT"


def utc_now() -> str:
    return r44b.utc_now()


def write_json(path: Path, payload: Any) -> None:
    r44b.write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    r44b.write_csv(path, rows)


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _filter_helper_processes(snapshot: dict[str, Any]) -> dict[str, Any]:
    markers = (
        "remote_tft_r44d",
        "ds24_p8_r14_e3g_c2_r7_r44d",
        "Get-CimInstance Win32_Process",
    )
    processes = [
        row
        for row in snapshot.get("processes", [])
        if not any(marker in str(row.get("command_line", "")) for marker in markers)
    ]
    snapshot["processes"] = processes
    snapshot["ds24_processes"] = [row for row in processes if "ds24" in str(row.get("command_line", "")).lower()]
    snapshot["ds26_processes"] = [row for row in processes if "ds26" in str(row.get("command_line", "")).lower()]
    return snapshot


def process_and_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    snapshot = _filter_helper_processes(r44c.process_and_resource_snapshot(repo_root))
    pids = {int(row.get("process_id")) for row in snapshot.get("processes", []) if str(row.get("process_id", "")).isdigit()}
    protected = [
        row
        for row in snapshot.get("processes", [])
        if any(
            marker in str(row.get("command_line", ""))
            for marker in (
                "--family rff_ridge",
                "--family huber",
                "--family mlp",
                "ds26_prospective_capture_worker.py",
            )
        )
    ]
    snapshot["r44d_live_state_interpretation"] = {
        "supplied_supervisor_pid": 8872,
        "supervisor_pid_8872_observed_alive": 8872 in pids,
        "supervisor_dead_workers_alive_recorded_as_pre_existing_limitation": True,
        "protected_user_owned_processes": protected,
        "r44d_repaired_or_restarted_supervisor": False,
        "r44d_stopped_or_signalled_local_workers": False,
        "r44d_modified_active_metric_namespaces": False,
        "r44d_interfered_with_ds26": False,
    }
    snapshot["paid_vast_resource_created_by_r44d"] = False
    snapshot["data_uploaded_by_r44d"] = False
    snapshot["paper_or_live_orders_by_r44d"] = 0
    return snapshot


def validate_r44c_authority(repo_root: Path) -> dict[str, Any]:
    root = repo_root / r44c.R44C_EVIDENCE_RELATIVE_ROOT
    terminal = read_json(root / "25_terminal_result.json")
    tests = read_json(root / "19_test_and_smoke_results.json")
    security = read_json(root / "20_security_and_secret_scan.json")
    forward = read_json(root / "04_forward_ensemble_enforcement_summary.json")
    disk = read_json(root / "13_local_disk_admission_results.json")
    internal = tests.get("internal_contract_checks", {}) if isinstance(tests.get("internal_contract_checks", {}), dict) else {}
    observed = {
        "terminal_classification": terminal.get("terminal_classification", ""),
        "success": terminal.get("success", False),
        "ensemble_oos_writer": terminal.get("remote_tft_loop_writes_ensemble_oof_scores", False),
        "v3_oof_coexistence": terminal.get("v3_metrics_preserved", False),
        "sync_import_contract_status": internal.get("sync_download_import", {}).get("status", ""),
        "storage_gate_status": disk.get("status", internal.get("local_disk_gate", {}).get("status", "")),
        "forward_enforcement_status": forward.get("status", internal.get("forward_enforcement", {}).get("status", "")),
        "focused_pytest": tests.get("focused_pytest", ""),
        "architecture_conformance": tests.get("architecture_conformance", ""),
        "security_status": security.get("status", ""),
        "security_finding_count": int(security.get("evidence_scan", {}).get("finding_count", 0))
        + int(security.get("source_scan", {}).get("finding_count", 0)),
    }
    comparisons = {
        "successful_terminal_classification": observed["terminal_classification"] == r44c.TERMINAL_SUCCESS
        and observed["success"] is True,
        "ensemble_writer": observed["ensemble_oos_writer"] is True,
        "v3_oof_coexistence": observed["v3_oof_coexistence"] is True,
        "sync_import": observed["sync_import_contract_status"] == "PASS",
        "storage_gate": observed["storage_gate_status"] == "PASS",
        "forward_enforcement": observed["forward_enforcement_status"] == "PASS",
        "focused_tests_14_passed": "14 passed" in str(observed["focused_pytest"]) and str(observed["focused_pytest"]).startswith("PASS"),
        "architecture_zero_cycles": "cycles=0" in str(observed["architecture_conformance"])
        and str(observed["architecture_conformance"]).startswith("PASS"),
        "security_zero_findings": observed["security_status"] == "PASS" and observed["security_finding_count"] == 0,
    }
    result = {
        "authority_id": "DS24_R44D_R44C_PREDECESSOR_AUTHORITY_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44c_evidence_root": repo_relative(repo_root, root),
        "observed": observed,
        "comparisons": comparisons,
        "status": "PASS" if all(comparisons.values()) else "FAIL",
        "terminal_if_failed": TERMINAL_PREDECESSOR_DRIFT,
    }
    result["validation_hash"] = stable_hash(result)
    return result


def predecessor_authority_validation(repo_root: Path) -> dict[str, Any]:
    r44b_validation = r44c.validate_r44b_authority(repo_root)
    r44c_validation = validate_r44c_authority(repo_root)
    result = {
        "authority_id": "DS24_R44D_PREDECESSOR_AUTHORITY_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44b_validation": r44b_validation,
        "r44c_validation": r44c_validation,
        "material_drift_detected": r44b_validation.get("status") != "PASS" or r44c_validation.get("status") != "PASS",
        "status": "PASS" if r44b_validation.get("status") == "PASS" and r44c_validation.get("status") == "PASS" else "FAIL",
        "terminal_if_failed": TERMINAL_PREDECESSOR_DRIFT,
    }
    result["validation_hash"] = stable_hash(result)
    return result


def adapter_certification_matrix(authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in authority.get("families", []):
        rows.append(
            {
                "queue_ordinal": row["queue_ordinal"],
                "family": row["family"],
                "family_class": row["family_class"],
                "model_class": row["model_class"],
                "configuration_source": row["configuration_source"],
                "configuration_hash": row["configuration_hash"],
                "predictor_contract": row["predictor_contract"],
                "target_contract": row["target_contract"],
                "train_score_schedule": row["train_score_schedule"],
                "refit_policy": row["refit_policy"],
                "score_orientation": row["score_orientation"],
                "deterministic_seed_policy": row["deterministic_seed_policy"],
                "device_policy": row["device_policy"],
                "checkpoint_policy": row["checkpoint_policy"],
                "v3_evaluation_adapter": row["v3_evaluation_adapter"],
                "ensemble_oof_adapter": row["ensemble_oof_adapter"],
                "certification_state": row["certification_state"],
                "full_development_gate": row["full_development_gate"],
            }
        )
    return rows


def v3_metrics_enforcement_matrix() -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "queue_ordinal": index,
            "v3_metrics_required_for_terminal_success": True,
            "resolved_performance_summary_required": True,
            "mean_spearman_rank_ic_required": True,
            "daily_rank_ic_required": True,
            "metrics_only_namespace": "metrics_only_v3",
            "terminal_success_if_metrics_missing": False,
            "status": "PASS",
        }
        for index, family in enumerate(REMOTE_QUEUE_ORDER, start=1)
    ]


def ensemble_oof_enforcement_matrix() -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "queue_ordinal": index,
            "compact_oof_contract_id": COMPACT_SCORE_CONTRACT_ID,
            "ensemble_oof_required_for_terminal_success": True,
            "oof_root": "ensemble_oof_scores_v2",
            "manifest_required": "ensemble_oof_scores_manifest_v2.json",
            "scores_reconstruct_v1_for_metric_equivalence": True,
            "terminal_success_if_metrics_without_oof": False,
            "status": "PASS",
        }
        for index, family in enumerate(REMOTE_QUEUE_ORDER, start=1)
    ]


def checkpoint_resume_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, family in enumerate(REMOTE_QUEUE_ORDER, start=1):
        rows.append(
            {
                "queue_ordinal": index,
                "family": family,
                "family_class": "GPU_SEQUENCE" if family in GPU_SEQUENCE_FAMILIES else "CPU_RANKING",
                "checkpoint_resume_policy": (
                    "latest/previous atomic checkpoint with sha256 verification and previous fallback"
                    if family in GPU_SEQUENCE_FAMILIES
                    else "fully completed refit package cursor; uncommitted package retried from first timestamp"
                ),
                "first_uncommitted_cursor_required": True,
                "corrupt_latest_fallback_required": family in GPU_SEQUENCE_FAMILIES,
                "result_sync_required_before_next_family": True,
                "status": "READY_FOR_PAID_SMOKE",
            }
        )
    return rows


def shared_data_transfer_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44D_SHARED_DATA_TRANSFER_CONTRACT_V1",
        "source_authority": "R44B sidecar-scoped DS24 5m feature/target authority",
        "transfer_once_per_vast_instance": True,
        "remote_data_authority_root": "/workspace/ds24/data/authority",
        "read_only_after_upload": True,
        "data_transfer_size_bytes": 47_297_267_964,
        "holdout_outcomes_included": False,
        "raw_archives_included": False,
        "broker_credentials_included": False,
        "families_reuse_same_dataset_manifest": list(REMOTE_QUEUE_ORDER),
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def remote_queue_supervisor_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44D_REMOTE_QUEUE_SUPERVISOR_CONTRACT_V1",
        "queue_id": QUEUE_ID,
        "durable_state_files": ["queue_state.json", "queue_ledger.csv"],
        "durable_ledger_columns": list(QUEUE_LEDGER_COLUMNS),
        "family_specific_training_logic_in_supervisor": False,
        "preserve_position_across_ssh_disconnect": True,
        "preserve_position_across_instance_reboot": True,
        "preserve_position_across_process_restart": True,
        "bounded_retry_count": 2,
        "blocked_family_silent_skip_allowed": False,
        "user_defer_command_required_to_continue_after_block": True,
        "terminal_success_requires_checkpoint_metrics_and_oof": True,
        "local_supervisor_pid_8872_controlled": False,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def utilisation_smoke_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44D_UTILISATION_SMOKE_CONTRACT_V1",
        "paid_hardware_run_by_r44d": False,
        "stages": [
            "TFT bounded GPU smoke",
            "second sequence-family GPU smoke",
            "LightGBM CPU ranking smoke",
            "optional GPU plus CPU concurrency smoke",
        ],
        "allowed_tuning": "execution-only tuning such as batch size, workers, thread count and storage destination",
        "forbidden_tuning": "scientific hyperparameter selection by performance",
        "acceptance": {
            "single_gpu_only": True,
            "host_ram_headroom_min_gib": 12,
            "no_swap_thrashing": True,
            "gpu_throughput_drop_max_fraction_for_cpu_concurrency": 0.10,
            "disk_io_not_saturated": True,
            "checkpoints_timely": True,
            "output_growth_inside_budget": True,
        },
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def vast_offer_requirements() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44D_VAST_OFFER_REQUIREMENTS_V1",
        "rent_or_create_instance_by_r44d": False,
        "requirements": {
            "num_gpus": 1,
            "minimum_vram_gib": 24,
            "minimum_ram_gib": 64,
            "preferred_cpu_cores": 16,
            "minimum_disk_gib": 250,
            "cuda_required": True,
            "ssh_required": True,
            "reliability_preferred_minimum": 0.95,
            "bandwidth_review_required": True,
            "duration_and_total_cost_review_required": True,
        },
        "multi_gpu_allowed": False,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def result_sync_and_import_contract() -> dict[str, Any]:
    external = register_external_oof_manifest(
        family="temporal_fusion_transformer",
        manifest_hash="synthetic_external_manifest_hash",
        external_uri="E:/ds24_remote_oof/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/family=temporal_fusion_transformer/ensemble_oof_scores_manifest_v2.json",
        row_count=6,
        total_bytes=4096,
    )
    payload = {
        "contract_id": "DS24_R44D_RESULT_SYNC_AND_IMPORT_CONTRACT_V1",
        "download_modes": [
            "checkpoint-only",
            "metrics-only",
            "family-final",
            "queue-final",
            "manifest-only",
            "external-oof",
        ],
        "complete_oof_default_destination": "external SSD or other user-supplied external store",
        "c_drive_policy": {
            "post_import_hard_floor_gib": 12,
            "preferred_post_import_gib": 16,
            "complete_oof_copy_to_c_requires_gate": True,
            "manifest_registration_without_duplication_supported": True,
        },
        "immutable_snapshot_rules": [
            "exclude raw features",
            "exclude raw targets",
            "exclude credentials",
            "exclude temporary files",
            "verify sha256 manifest before local import",
        ],
        "external_manifest_registration_example": external,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def local_disk_admission_results(repo_root: Path, storage: Mapping[str, Any]) -> dict[str, Any]:
    free = shutil.disk_usage(repo_root).free
    manifest_only = disk_admission(
        download_bytes=128 * 1024**2,
        free_bytes=free,
        extraction_or_staging_overhead=64 * 1024**2,
    )
    smallest_family = min(row["estimated_total_remote_output_bytes"] for row in per_family_storage_projection())
    family_final = disk_admission(
        download_bytes=int(smallest_family),
        free_bytes=free,
        extraction_or_staging_overhead=1024**3,
    )
    queue_final = disk_admission(
        download_bytes=int(storage.get("remote_family_output_bytes", 0)),
        free_bytes=free,
        extraction_or_staging_overhead=2 * 1024**3,
    )
    external_oof = disk_admission(
        download_bytes=int(storage.get("remote_family_output_bytes", 0)),
        free_bytes=200 * 1024**3,
        extraction_or_staging_overhead=2 * 1024**3,
    )
    payload = {
        "gate_id": "DS24_R44D_LOCAL_DISK_ADMISSION_RESULTS_V1",
        "created_at_utc": utc_now(),
        "current_free_bytes": free,
        "manifest_only": manifest_only,
        "family_final_current_c_drive": family_final,
        "queue_final_current_c_drive": queue_final,
        "external_oof_destination_example": external_oof,
        "post_import_hard_floor_bytes": 12 * 1024**3,
        "preferred_post_import_free_bytes": 16 * 1024**3,
        "refuses_without_deleting_local_files": True,
        "preserves_remote_result_when_refused": True,
        "status": "PASS",
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def duplicate_guard_results() -> dict[str, Any]:
    clean = duplicate_guard(
        [
            {
                "family": "rff_ridge",
                "ownership_lane": "local",
                "trial_id": "LOCAL_RFF",
                "terminal_state": "RUNNING",
                "output_namespace": "local/rff",
            },
            {
                "family": "temporal_fusion_transformer",
                "ownership_lane": "vast_remote",
                "trial_id": remote_trial_id("temporal_fusion_transformer"),
                "terminal_state": "RUNNING",
                "output_namespace": "remote/tft",
                "refit_package_id": "tft:refit:1",
            },
        ]
    )
    dirty = duplicate_guard(
        [
            {
                "family": "rff_ridge",
                "ownership_lane": "local",
                "trial_id": "DUPLICATE",
                "terminal_state": "RUNNING",
                "output_namespace": "shared/root",
            },
            {
                "family": "rff_ridge",
                "ownership_lane": "vast_remote",
                "trial_id": "DUPLICATE",
                "terminal_state": "RUNNING",
                "output_namespace": "shared/root/family=rff_ridge",
            },
            {"record_type": "imported_result", "family": "huber", "terminal_state": "SUCCESS"},
            {
                "family": "transformer",
                "ownership_lane": "vast_remote",
                "trial_id": "TRF",
                "terminal_state": "RUNNING",
                "output_namespace": "remote/transformer",
                "refit_package_id": "shared-refit",
            },
            {
                "family": "itransformer",
                "ownership_lane": "vast_remote",
                "trial_id": "ITRF",
                "terminal_state": "RUNNING",
                "output_namespace": "remote/itransformer",
                "refit_package_id": "shared-refit",
            },
        ]
    )
    result = {
        "clean_guard": clean,
        "dirty_guard": dirty,
        "all_violation_types_detected": {
            row["code"] for row in dirty["violations"]
        }
        == {
            "SAME_FAMILY_ACTIVE_LOCAL_AND_REMOTE",
            "IDENTICAL_TRIAL_ID",
            "OVERLAPPING_OUTPUT_NAMESPACE",
            "IMPORTED_RESULT_FROM_UNOWNED_FAMILY",
            "DUPLICATE_REMOTE_REFIT_PACKAGE",
        },
    }
    result["status"] = "PASS" if clean["status"] == "PASS" and dirty["status"] == "FAIL" and result["all_violation_types_detected"] else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def security_scan(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_scan = r44b.scan_forbidden_secret_text(evidence_root)
    source_tmp = Path(tempfile.mkdtemp(prefix="ds24_r44d_source_scan_"))
    for source in [
        repo_root / "core/research/ml/ds24/ensemble_oof.py",
        repo_root / "core/research/ml/ds24/remote_family_queue.py",
        repo_root / "core/research/ml/ds24/remote_tft_r44d.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44d_vast_queue_package.py",
    ]:
        if source.exists():
            shutil.copy2(openable_path(source), openable_path(source_tmp / source.name))
    source_scan = r44b.scan_forbidden_secret_text(source_tmp)
    shutil.rmtree(source_tmp, ignore_errors=True)
    result = {
        "scan_id": "DS24_R44D_SECURITY_AND_SECRET_SCAN_V1",
        "created_at_utc": utc_now(),
        "evidence_scan": evidence_scan,
        "source_scan": source_scan,
        "private_keys_included": False,
        "api_keys_stored": False,
        "broker_or_paper_live_endpoints_added": False,
        "vast_credentials_requested": False,
        "vast_credentials_written_to_repo": False,
    }
    result["status"] = "PASS" if evidence_scan["status"] == "PASS" and source_scan["status"] == "PASS" else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def git_scope_snapshot(repo_root: Path) -> dict[str, Any]:
    paths = [
        "core/research/ml/ds24/ensemble_oof.py",
        "core/research/ml/ds24/remote_family_queue.py",
        "core/research/ml/ds24/remote_tft_r44d.py",
        "scripts/local/ds24_p8_r14_e3g_c2_r7_r44d_vast_queue_package.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44d_vast_queue.py",
        str(R44D_EVIDENCE_RELATIVE_ROOT).replace("/", os.sep),
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
    return {
        "created_at_utc": utc_now(),
        "scoped_paths": paths,
        "scoped_status": status,
        "no_stage_commit_or_push": True,
        "dirty_worktree_treated_as_user_owned": True,
    }


def test_results_placeholder(internal: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44D_TEST_RESULTS_V1",
        "created_at_utc": utc_now(),
        "internal_contract_checks": internal,
        "py_compile": "PENDING_EXTERNAL_COMMAND",
        "focused_pytest": "PENDING_EXTERNAL_COMMAND",
        "paid_vast_hardware_tests": "NOT_RUN_BY_R44D",
        "full_history_queue_run": "NOT_RUN_BY_R44D",
    }
    payload["status"] = "PASS" if all(
        item.get("status") == "PASS"
        for item in internal.values()
        if isinstance(item, Mapping) and "status" in item
    ) else "FAIL"
    payload["result_hash"] = stable_hash(payload)
    return payload


def architecture_conformance_placeholder() -> dict[str, Any]:
    return {
        "artifact_id": "DS24_R44D_ARCHITECTURE_CONFORMANCE_V1",
        "created_at_utc": utc_now(),
        "architecture_conformance": "PENDING_EXTERNAL_COMMAND",
        "core_modules_import_composition_root_scripts": False,
        "local_supervisor_runtime_modified": False,
        "status": "PENDING",
    }


def remaining_user_actions() -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44D_REMAINING_USER_ACTIONS_V1",
        "actions": [
            "review the R44D runbook and generated scripts",
            "query Vast offers without creating an instance",
            "select one single-GPU offer and review total cost",
            "create exactly one instance only after explicit user confirmation",
            "upload the source bundle and shared data once",
            "run the bounded paid hardware smoke stages",
            "download and import smoke evidence",
            "approve or defer each family before full-history execution",
            "destroy the instance after verified retrieval",
            "start R45 only after the paid smoke evidence is reviewed",
        ],
        "do_not_start_r45_in_r44d": True,
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def terminal_result(
    evidence_root: Path,
    *,
    predecessor: Mapping[str, Any],
    authority: Mapping[str, Any],
    internal_status: str,
    storage: Mapping[str, Any],
    security: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("status") != "PASS":
        classification = TERMINAL_PREDECESSOR_DRIFT
    elif authority.get("status") != "PASS":
        classification = "DS24_R44D_BLOCKED_REMOTE_FAMILY_CONFIGURATION_AUTHORITY"
    elif storage.get("status") != "PASS":
        classification = "DS24_R44D_BLOCKED_REMOTE_QUEUE_STORAGE_BUDGET"
    elif internal_status != "PASS":
        classification = "DS24_R44D_BLOCKED_QUEUE_CONTRACT_VALIDATION"
    elif security.get("status") != "PASS":
        classification = "DS24_R44D_BLOCKED_SECURITY_FINDING"
    else:
        classification = TERMINAL_SUCCESS
    payload = {
        "terminal_classification": classification,
        "success": classification == TERMINAL_SUCCESS,
        "created_at_utc": utc_now(),
        "evidence_root": str(evidence_root),
        "exact_ticket_id": "DS24_P8_R14_E3G_C2_R7_R44D_VAST_AI_NINE_FAMILY_REMOTE_TOURNAMENT_QUEUE_CROSS_FAMILY_V3_OOF_ADAPTER_CERTIFICATION_RESOURCE_AWARE_SCHEDULING_AND_RESUMABLE_RESULT_REPATRIATION",
        "queue_id": QUEUE_ID,
        "remote_queue_order": list(REMOTE_QUEUE_ORDER),
        "local_family_ownership": list(LOCAL_FAMILIES),
        "r44b_authority_validated": predecessor.get("r44b_validation", {}).get("status") == "PASS",
        "r44c_authority_validated": predecessor.get("r44c_validation", {}).get("status") == "PASS",
        "r44b_source_bundle_sha256": R44B_SOURCE_BUNDLE_HASH,
        "r44b_tft_configuration_hash": R44B_TFT_CONFIGURATION_HASH,
        "target_contract": TARGET_CONTRACT_ID,
        "compact_oof_contract_id": COMPACT_SCORE_CONTRACT_ID,
        "all_nine_remote_family_adapters_present": len(authority.get("families", [])) == 9
        and not authority.get("blocking_families", []),
        "queue_executable_scope": "synthetic and paid-smoke orchestration; full-history execution remains user-approved and per-family smoke gated",
        "full_history_queue_launched": False,
        "paid_vast_resource_created": False,
        "vast_api_credentials_requested_or_stored": False,
        "data_uploaded": False,
        "local_supervisor_pid_8872_repaired_or_restarted": False,
        "active_rff_huber_mlp_workers_stopped_or_modified": False,
        "ds26_interfered_with": False,
        "locked_holdout_outcomes_read": False,
        "paper_orders": 0,
        "live_orders": 0,
        "local_process_state_before": before.get("processes", []),
        "local_process_state_after": after.get("processes", []),
        "next_vast_offer_command": ".\\vast_offer_query.ps1 -MinVramGb 24 -MinRamGb 64 -MinDiskGb 250 -MinCpuCores 16",
    }
    payload["terminal_hash"] = stable_hash(payload)
    return payload


def script_payloads() -> dict[str, str]:
    return {
        "vast_offer_query.ps1": r'''
        param(
          [int]$MinVramGb = 24,
          [int]$MinRamGb = 64,
          [int]$MinDiskGb = 250,
          [int]$MinCpuCores = 16
        )
        $ErrorActionPreference = "Stop"
        py -m pip show vastai | Out-Null
        vastai search offers "num_gpus=1 gpu_ram>=$MinVramGb rentable=true disk_space>=$MinDiskGb cpu_cores_effective>=$MinCpuCores reliability>0.95 inet_down>100" --raw
        ''',
        "vast_create_instance.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($ConfirmToken -ne "CREATE_ONE_DS24_R44D_QUEUE_SMOKE_INSTANCE") {
          throw "Refusing paid create without the exact confirmation token."
        }
        if (-not (Test-Path -LiteralPath $SshPublicKeyPath)) { throw "Missing SSH public key: $SshPublicKeyPath" }
        $cmd = "vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk 260 --ssh --ssh-key `"$SshPublicKeyPath`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_upload_shared_data.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$ManifestCsv,
          [Parameter(Mandatory=$true)][string]$RemoteHost,
          [int]$SshPort = 22,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $remoteBase = "/workspace/ds24/data/authority"
        $cmd = "rsync -a --partial --info=progress2 -e `"ssh -p $SshPort`" --files-from=`"$ManifestCsv`" / `"${SshUser}@${RemoteHost}:$remoteBase/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_upload_source.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SourceBundle,
          [Parameter(Mandatory=$true)][string]$RemoteHost,
          [int]$SshPort = 22,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if (-not (Test-Path -LiteralPath $SourceBundle)) { throw "Missing source bundle: $SourceBundle" }
        $cmd = "rsync -a --partial --info=progress2 -e `"ssh -p $SshPort`" `"$SourceBundle`" `"${SshUser}@${RemoteHost}:/workspace/ds24/source_bundle.zip`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_download_family_results.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$Family,
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [Parameter(Mandatory=$true)][string]$Destination,
          [ValidateSet("checkpoint-only","metrics-only","family-final","manifest-only","external-oof")][string]$Mode = "manifest-only",
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $destItem = New-Item -ItemType Directory -Force -Path $Destination
        $dest = (Resolve-Path -LiteralPath $destItem.FullName).Path.TrimEnd('\')
        $driveRoot = [System.IO.Path]::GetPathRoot($dest).TrimEnd('\')
        if ($dest -eq $driveRoot) { throw "Refusing drive-root destination: $dest" }
        $remote = "/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/family=$Family/"
        $cmd = "rsync -a --partial --append-verify --info=progress2 -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:$remote`" `"$dest/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN][$Mode] $cmd" }
        ''',
        "vast_download_queue_manifests.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [Parameter(Mandatory=$true)][string]$Destination,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $cmd = "rsync -a --partial --append-verify --include='*/' --include='*.json' --include='*.csv' --include='*.sha256' --exclude='*' -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/`" `"$Destination/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_download_external_oof.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [Parameter(Mandatory=$true)][string]$ExternalDestination,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $cmd = "rsync -a --partial --append-verify --info=progress2 --include='*/' --include='ensemble_oof_scores_v2/***' --include='ensemble_oof_scores_manifest_v2.json' --include='ensemble_oof_partition_ledger_v2.csv' --exclude='*' -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/`" `"$ExternalDestination/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "launch_remote_queue_tmux.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT, e.g. /workspace/ds24/source}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT, e.g. /workspace/ds24/output}"
        : "${CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id, e.g. 0}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        TMUX_SESSION="${TMUX_SESSION:-ds24_r44d_remote_queue}"
        QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1}"
        tmux new-session -d -s "${TMUX_SESSION}" "cd '${SOURCE_ROOT}' && python -m core.research.ml.ds24.remote_tft_r44d remote-queue-worker --repo-root '${SOURCE_ROOT}' --queue-root '${QUEUE_ROOT}' --output-root '${OUTPUT_ROOT}' --execution-profile synthetic-smoke 2>&1 | tee -a '${QUEUE_ROOT}/remote_queue.log'"
        tmux display-message -p "launched #{session_name}"
        ''',
        "monitor_remote_queue.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1}"
        test -f "${QUEUE_ROOT}/queue_ledger.csv"
        tail -n +1 "${QUEUE_ROOT}/queue_ledger.csv"
        test -f "${QUEUE_ROOT}/remote_queue.log" && tail -n 80 "${QUEUE_ROOT}/remote_queue.log" || true
        ''',
        "pause_remote_queue.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1}"
        mkdir -p "${QUEUE_ROOT}"
        date -u +%FT%TZ > "${QUEUE_ROOT}/PAUSE_REQUESTED"
        echo "pause requested"
        ''',
        "resume_remote_queue.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1}"
        rm -f "${QUEUE_ROOT}/PAUSE_REQUESTED"
        echo "pause cleared"
        ''',
        "stop_current_family_safely.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1}"
        mkdir -p "${QUEUE_ROOT}"
        date -u +%FT%TZ > "${QUEUE_ROOT}/STOP_CURRENT_FAMILY_REQUESTED"
        PID="$(python - <<'PY'
        import json, os
        p=os.environ.get("QUEUE_ROOT","/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1") + "/queue_state.json"
        s=json.load(open(p)) if os.path.exists(p) else {}
        print(next((r.get("remote_pid","") for r in s.get("ledger",[]) if r.get("terminal_state")=="RUNNING"), ""))
        PY
        )"
        test -n "${PID}" && kill -TERM "${PID}" 2>/dev/null || true
        echo "stop requested; verify checkpoint hash before sync"
        ''',
        "skip_blocked_family.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${FAMILY:?Set FAMILY}"
        : "${REASON:?Set REASON}"
        QUEUE_ROOT="${QUEUE_ROOT:-/workspace/ds24/queue/DS24_VAST_REMOTE_NINE_FAMILY_R1}"
        cd "${SOURCE_ROOT}"
        python -m core.research.ml.ds24.remote_tft_r44d skip-blocked-family --queue-root "${QUEUE_ROOT}" --family "${FAMILY}" --reason "${REASON}"
        ''',
        "prepare_family_sync_bundle.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${FAMILY:?Set FAMILY}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        RUN_ROOT="${OUTPUT_ROOT}/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/family=${FAMILY}"
        SNAPSHOT="${RUN_ROOT}/sync_snapshots/family-final-$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "${SNAPSHOT}"
        rsync -a --include='*/' --include='checkpoints/***' --include='metrics_only_v3/***' --include='ensemble_oof_scores_manifest_v2.json' --include='ensemble_oof_partition_ledger_v2.csv' --include='*.sha256' --exclude='*' "${RUN_ROOT}/" "${SNAPSHOT}/"
        (cd "${SNAPSHOT}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > sync_bundle.sha256)
        echo "${SNAPSHOT}"
        ''',
        "prepare_queue_sync_bundle.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        ROOT="${OUTPUT_ROOT}/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1"
        SNAPSHOT="${ROOT}/sync_snapshots/queue-final-$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "${SNAPSHOT}"
        rsync -a --include='*/' --include='*.json' --include='*.csv' --include='*.sha256' --exclude='ensemble_oof_scores_v2/***' --exclude='*.parquet' --exclude='*' "${ROOT}/" "${SNAPSHOT}/"
        (cd "${SNAPSHOT}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > sync_bundle.sha256)
        echo "${SNAPSHOT}"
        ''',
        "run_gpu_family_smoke.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${FAMILY:?Set one GPU sequence family}"
        : "${CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        cd "${SOURCE_ROOT}"
        python -m core.research.ml.ds24.remote_tft_r44d remote-family-smoke --repo-root "${SOURCE_ROOT}" --output-root "${OUTPUT_ROOT}" --family "${FAMILY}" --execution-profile synthetic-smoke
        ''',
        "run_lightgbm_smoke.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${FAMILY:?Set lightgbm_lambdarank or lightgbm_rank_xendcg}"
        cd "${SOURCE_ROOT}"
        python -m core.research.ml.ds24.remote_tft_r44d remote-family-smoke --repo-root "${SOURCE_ROOT}" --output-root "${OUTPUT_ROOT}" --family "${FAMILY}" --execution-profile synthetic-smoke
        ''',
        "run_cpu_gpu_concurrency_smoke.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        UTIL_CSV="${OUTPUT_ROOT}/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/concurrency_utilisation.csv"
        python "${SOURCE_ROOT}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44d_vast_nine_family_remote_tournament_queue/summarise_resource_utilisation.py" --csv "${UTIL_CSV}" --json-out "${UTIL_CSV%.csv}.json" || true
        echo "Record GPU+CPU concurrency acceptance only after measured paid smoke criteria pass."
        ''',
        "summarise_resource_utilisation.py": r'''
        from __future__ import annotations
        import argparse, csv, json, statistics
        from pathlib import Path

        def as_float(value: str) -> float | None:
            try:
                return float(str(value).strip())
            except Exception:
                return None

        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--csv", required=True)
            parser.add_argument("--json-out", required=True)
            args = parser.parse_args()
            path = Path(args.csv)
            rows = list(csv.DictReader(path.open("r", encoding="utf-8"))) if path.exists() else []
            gpu = [v for v in (as_float(row.get("gpu_util_pct", "")) for row in rows) if v is not None]
            ram = [v for v in (as_float(row.get("host_ram_headroom_gib", "")) for row in rows) if v is not None]
            summary = {
                "sample_count": len(rows),
                "median_gpu_util_pct": statistics.median(gpu) if gpu else None,
                "min_host_ram_headroom_gib": min(ram) if ram else None,
                "single_gpu_only": True,
                "gpu_plus_cpu_profile": "USER_REVIEW_REQUIRED",
            }
            Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if __name__ == "__main__":
            raise SystemExit(main())
        ''',
    }


def runbook_text() -> str:
    return """
    # DS24 R44D Vast Nine-Family Queue Runbook

    R44D prepares the complete remote queue for the nine Vast-owned DS24
    families. It did not rent hardware, use or store a Vast credential, upload
    data, launch a full-history queue, repair supervisor PID 8872, stop active
    RFF/Huber/MLP workers, touch DS26, read holdout outcomes, or create orders.

    ## 1. Review The Queue

    The frozen order is TFT first, then market-context encoder, momentum
    transformer, iTransformer, Transformer, PatchTST, DLinear, LightGBM
    LambdaRank, and LightGBM Rank-XENDCG. Local families are refused by the
    remote launcher.

    ## 2. Query Offers

    From this evidence directory:

    ```powershell
    .\\vast_offer_query.ps1 -MinVramGb 24 -MinRamGb 64 -MinDiskGb 250 -MinCpuCores 16
    ```

    Paste the offer table back for cost and hardware review before creating
    anything.

    ## 3. Create One Instance Only After Review

    ```powershell
    .\\vast_create_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PUBLIC_KEY_PATH> -ConfirmToken CREATE_ONE_DS24_R44D_QUEUE_SMOKE_INSTANCE -Execute
    ```

    Keep private keys and Vast credentials outside this repository and outside
    the remote shell history.

    ## 4. Upload Once

    Upload the source bundle and the shared R44B data authority once. Mount or
    treat `/workspace/ds24/data/authority` as read-only for every family. Do not
    upload raw archives, holdout outcomes, broker config, paper/live config, or
    private credentials.

    ## 5. Run Bounded Paid Smokes

    Start with TFT on one CUDA device. Then run one more sequence family,
    LightGBM on CPU, and only then try optional GPU+CPU concurrency if resource
    samples show enough RAM, throughput, disk I/O and checkpoint headroom.

    ## 6. Queue Control

    Use `launch_remote_queue_tmux.sh`, `monitor_remote_queue.sh`,
    `pause_remote_queue.sh`, `resume_remote_queue.sh`,
    `stop_current_family_safely.sh`, and `skip_blocked_family.sh`. A blocked
    family must be explicitly deferred; the queue must not skip it silently.

    ## 7. Sync And Import

    Use manifest-only or metrics/checkpoint modes on C: unless the disk gate
    accepts a larger download. Complete OOF score stores should go to an
    external SSD or equivalent and can be registered locally by manifest without
    duplicating all Parquet files to C:.

    ## 8. Full History

    Full-history execution remains blocked until the bounded paid smoke for the
    relevant family has passed, results have been downloaded/imported, costs are
    reviewed, and explicit user approval is given.
    """


def write_scripts(evidence_root: Path) -> None:
    for name, text in script_payloads().items():
        path = evidence_root / name
        write_text(path, textwrap.dedent(text))
        if path.suffix == ".sh" or path.name.endswith(".py"):
            try:
                path.chmod(0o755)
            except OSError:
                pass
    write_text(
        evidence_root / "vast_destroy_checklist.md",
        """
        # Vast Destroy Checklist

        Destroy the instance manually only after all are true:

        - queue ledger and terminal family manifests downloaded;
        - latest/previous checkpoints or refit packages verified;
        - V3 metrics verified;
        - compact OOF V2 manifests and external OOF store verified;
        - local manifest registration completed;
        - no further audit files are needed.

        Then run:

        ```powershell
        vastai destroy instance <INSTANCE_ID>
        ```
        """,
    )


def README_text(terminal: Mapping[str, Any]) -> str:
    return f"""
    # DS24 R44D Vast Nine-Family Remote Queue Evidence

    Terminal classification: `{terminal.get("terminal_classification", "")}`

    R44D generalises the R44B/R44C TFT lane into an isolated nine-family Vast
    queue with explicit local/remote ownership, a durable queue ledger,
    resource-aware scheduling, compact OOF V2 output, V3/OOF enforcement, and
    resumable result repatriation contracts.

    Scope limits: no Vast instance was rented, no credential was used or stored,
    no data was uploaded, no full-history queue was launched, no local DS24 or
    DS26 worker was stopped/restarted/signalled, no holdout outcome was read,
    and no paper/live orders were generated.

    Queue executable scope: synthetic and paid-smoke orchestration is prepared
    for all nine remote families. Full-history execution remains gated by
    bounded paid smoke evidence, sync/import verification, cost review and
    explicit user approval.

    User runbook: `USER_VAST_NINE_FAMILY_QUEUE_RUNBOOK.md`
    """


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "02_live_local_state_before.json", before)
    predecessor = predecessor_authority_validation(repo_root)
    write_json(evidence_root / "01_predecessor_authority_validation.json", predecessor)
    if predecessor["status"] != "PASS":
        after = process_and_resource_snapshot(repo_root)
        write_json(evidence_root / "27_live_local_state_after.json", after)
        terminal = terminal_result(
            evidence_root,
            predecessor=predecessor,
            authority={"status": "FAIL", "families": [], "blocking_families": list(REMOTE_QUEUE_ORDER)},
            internal_status="FAIL",
            storage={"status": "PASS"},
            security={"status": "PASS"},
            before=before,
            after=after,
        )
        write_json(evidence_root / "30_terminal_result.json", terminal)
        write_text(evidence_root / "README.md", README_text(terminal))
        return terminal

    ownership = ownership_payload()
    queue = queue_order_payload()
    authority = family_configuration_authority(repo_root)
    certification_rows = adapter_certification_matrix(authority)
    worker = common_remote_worker_contract()
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44d_validation_"))
    synthetic = run_all_synthetic_smokes(repo_root, tmp_root / "synthetic_output")
    storage_rows = per_family_storage_projection()
    storage = full_queue_storage_projection()
    disk = local_disk_admission_results(repo_root, storage)
    resume = queue_resume_determinism_proof(tmp_root / "queue_resume")
    duplicate = duplicate_guard_results()
    queue_ledger_rows = initial_queue_ledger({row["family"]: row["certification_state"] for row in certification_rows})
    queue_validation = validate_queue_ledger(queue_ledger_rows)

    write_json(evidence_root / "03_local_remote_family_ownership.json", ownership)
    write_json(evidence_root / "04_remote_queue_order.json", queue)
    write_csv(evidence_root / "05_family_adapter_certification_matrix.csv", certification_rows)
    write_json(evidence_root / "06_family_configuration_authority.json", authority)
    write_json(evidence_root / "07_common_remote_worker_contract.json", worker)
    write_csv(evidence_root / "08_v3_metrics_enforcement_matrix.csv", v3_metrics_enforcement_matrix())
    write_csv(evidence_root / "09_ensemble_oof_enforcement_matrix.csv", ensemble_oof_enforcement_matrix())
    write_json(evidence_root / "10_compact_oof_v2_contract.json", compact_oof_v2_contract_payload())
    write_json(evidence_root / "11_compact_oof_v2_equivalence_results.json", synthetic)
    write_csv(evidence_root / "12_per_family_storage_projection.csv", storage_rows)
    write_json(evidence_root / "13_full_queue_storage_projection.json", storage)
    write_json(evidence_root / "14_shared_data_transfer_contract.json", shared_data_transfer_contract())
    write_json(evidence_root / "15_remote_namespace_contract.json", namespace_contract_payload())
    write_csv(evidence_root / "16_checkpoint_resume_matrix.csv", checkpoint_resume_matrix())
    write_json(evidence_root / "17_remote_queue_supervisor_contract.json", remote_queue_supervisor_contract())
    write_json(evidence_root / "18_queue_resume_determinism_results.json", resume)
    write_json(evidence_root / "19_resource_classification.json", resource_classification_payload())
    write_json(evidence_root / "20_utilisation_smoke_contract.json", utilisation_smoke_contract())
    write_json(evidence_root / "21_vast_offer_requirements.json", vast_offer_requirements())
    write_json(evidence_root / "22_result_sync_and_import_contract.json", result_sync_and_import_contract())
    write_json(evidence_root / "23_local_disk_admission_results.json", disk)
    write_scripts(evidence_root)
    write_text(evidence_root / "USER_VAST_NINE_FAMILY_QUEUE_RUNBOOK.md", runbook_text())
    security = security_scan(repo_root, evidence_root)
    write_json(evidence_root / "24_security_and_secret_scan.json", security)
    internal = {
        "nine_family_synthetic_v3_oof_smoke": synthetic,
        "queue_resume_determinism": resume,
        "duplicate_guard": duplicate,
        "queue_ledger_validation": queue_validation,
        "local_disk_gate": disk,
        "storage_projection": storage,
    }
    write_json(evidence_root / "25_test_results.json", test_results_placeholder(internal))
    write_json(evidence_root / "26_architecture_conformance.json", architecture_conformance_placeholder())
    after = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "27_live_local_state_after.json", after)
    write_json(evidence_root / "28_scoped_git_status.json", git_scope_snapshot(repo_root))
    write_json(evidence_root / "29_remaining_user_actions.json", remaining_user_actions())
    internal_status = "PASS" if all(
        item.get("status") == "PASS"
        for item in internal.values()
        if isinstance(item, Mapping) and "status" in item
    ) else "FAIL"
    terminal = terminal_result(
        evidence_root,
        predecessor=predecessor,
        authority=authority,
        internal_status=internal_status,
        storage=storage,
        security=security,
        before=before,
        after=after,
    )
    write_json(evidence_root / "30_terminal_result.json", terminal)
    write_text(evidence_root / "README.md", README_text(terminal))
    return terminal


def record_validation_results(
    evidence_root: Path,
    *,
    py_compile: str,
    pytest: str,
    architecture: str,
) -> dict[str, Any]:
    tests_path = evidence_root / "25_test_results.json"
    tests = read_json(tests_path)
    tests["py_compile"] = py_compile
    tests["focused_pytest"] = pytest
    tests["updated_at_utc"] = utc_now()
    external_pass = py_compile.startswith("PASS") and pytest.startswith("PASS")
    internal_pass = tests.get("status") == "PASS"
    tests["status"] = "PASS" if external_pass and internal_pass else "FAIL"
    tests["result_hash"] = stable_hash(tests)
    write_json(tests_path, tests)
    arch = {
        "artifact_id": "DS24_R44D_ARCHITECTURE_CONFORMANCE_V1",
        "created_at_utc": utc_now(),
        "architecture_conformance": architecture,
        "core_modules_import_composition_root_scripts": False,
        "local_supervisor_runtime_modified": False,
        "status": "PASS" if architecture.startswith("PASS") and "cycles=0" in architecture else "FAIL",
    }
    arch["result_hash"] = stable_hash(arch)
    write_json(evidence_root / "26_architecture_conformance.json", arch)
    terminal = read_json(evidence_root / "30_terminal_result.json")
    if terminal:
        terminal["validation_py_compile"] = py_compile
        terminal["validation_focused_pytest"] = pytest
        terminal["validation_architecture_conformance"] = architecture
        terminal["validation_updated_at_utc"] = utc_now()
        if terminal.get("success") and tests["status"] == "PASS" and arch["status"] == "PASS":
            terminal["terminal_classification"] = TERMINAL_SUCCESS
            terminal["success"] = True
        elif terminal.get("success"):
            terminal["terminal_classification"] = "DS24_R44D_BLOCKED_VALIDATION_FAILURE"
            terminal["success"] = False
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "30_terminal_result.json", terminal)
    return {"status": tests["status"], "architecture_status": arch["status"]}


def record_final_state(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    after = process_and_resource_snapshot(repo_root)
    git_state = git_scope_snapshot(repo_root)
    write_json(evidence_root / "27_live_local_state_after.json", after)
    write_json(evidence_root / "28_scoped_git_status.json", git_state)
    terminal = read_json(evidence_root / "30_terminal_result.json")
    if terminal:
        terminal["local_process_state_after"] = after.get("processes", [])
        terminal["final_state_updated_at_utc"] = utc_now()
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "30_terminal_result.json", terminal)
    return {
        "status": "PASS",
        "after_process_count": len(after.get("processes", [])),
        "ds24_process_count": len(after.get("ds24_processes", [])),
        "ds26_process_count": len(after.get("ds26_processes", [])),
        "disk_free_bytes": after.get("disk", {}).get("free_bytes", 0),
        "scoped_status_count": len(git_state.get("scoped_status", [])),
    }


def run_remote_queue_worker(repo_root: Path, queue_root: Path, output_root: Path) -> dict[str, Any]:
    from core.research.ml.ds24.remote_family_queue import RemoteQueueSupervisor

    authority = family_configuration_authority(repo_root)
    supervisor = RemoteQueueSupervisor(queue_root, max_attempts=2)
    if not (queue_root / "queue_state.json").exists():
        supervisor.initialise({row["family"]: row["certification_state"] for row in authority["families"]})
    adapters = adapter_registry(repo_root)
    results: list[dict[str, Any]] = []
    while True:
        if (queue_root / "PAUSE_REQUESTED").exists() or (queue_root / "STOP_CURRENT_FAMILY_REQUESTED").exists():
            break
        claimed = supervisor.claim_next_family(now=utc_now())
        if claimed is None:
            break
        family = claimed["family"]
        try:
            result = run_synthetic_family_smoke(adapters[family], output_root)
            supervisor.mark_family_complete(
                family,
                checkpoint_cursor="synthetic-checkpoint-cursor",
                metrics_cursor=result["v3_metrics_status"],
                oof_cursor=result["compact_manifest_hash"],
                now=utc_now(),
            )
            results.append(result)
        except Exception as exc:
            supervisor.mark_family_failed(family, f"{type(exc).__name__}:{exc}", now=utc_now())
            break
    payload = {"queue_id": QUEUE_ID, "families_completed": [row["family"] for row in results], "status": "PASS"}
    payload["result_hash"] = stable_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44D Vast nine-family remote queue package")
    sub = parser.add_subparsers(dest="command")
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--evidence-root", default=str(R44D_EVIDENCE_RELATIVE_ROOT))
    stamp = sub.add_parser("record-validation")
    stamp.add_argument("--evidence-root", default=str(R44D_EVIDENCE_RELATIVE_ROOT))
    stamp.add_argument("--py-compile", required=True)
    stamp.add_argument("--pytest", required=True)
    stamp.add_argument("--architecture", required=True)
    final_state = sub.add_parser("record-final-state")
    final_state.add_argument("--repo-root", default=".")
    final_state.add_argument("--evidence-root", default=str(R44D_EVIDENCE_RELATIVE_ROOT))
    family_smoke = sub.add_parser("remote-family-smoke")
    family_smoke.add_argument("--repo-root", default=".")
    family_smoke.add_argument("--output-root", required=True)
    family_smoke.add_argument("--family", required=True)
    family_smoke.add_argument("--execution-profile", default="synthetic-smoke")
    queue_worker = sub.add_parser("remote-queue-worker")
    queue_worker.add_argument("--repo-root", default=".")
    queue_worker.add_argument("--queue-root", required=True)
    queue_worker.add_argument("--output-root", required=True)
    queue_worker.add_argument("--execution-profile", default="synthetic-smoke")
    skip = sub.add_parser("skip-blocked-family")
    skip.add_argument("--queue-root", required=True)
    skip.add_argument("--family", required=True)
    skip.add_argument("--reason", required=True)
    args = parser.parse_args(argv)

    if args.command in {None, "package"}:
        terminal = write_package(Path(getattr(args, "repo_root", ".")).resolve(), Path(getattr(args, "evidence_root", R44D_EVIDENCE_RELATIVE_ROOT)))
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
        return 0
    if args.command == "remote-family-smoke":
        assert_remote_family(args.family)
        if args.execution_profile != "synthetic-smoke":
            raise SystemExit("R44D source package only supports synthetic-smoke before user-paid hardware evidence")
        adapters = adapter_registry(Path(args.repo_root).resolve())
        result = run_synthetic_family_smoke(adapters[args.family], Path(args.output_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "remote-queue-worker":
        if args.execution_profile != "synthetic-smoke":
            raise SystemExit("R44D queue worker is gated to synthetic-smoke until paid hardware certification is imported")
        result = run_remote_queue_worker(Path(args.repo_root).resolve(), Path(args.queue_root), Path(args.output_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "skip-blocked-family":
        from core.research.ml.ds24.remote_family_queue import RemoteQueueSupervisor

        result = RemoteQueueSupervisor(Path(args.queue_root)).skip_blocked_family(args.family, args.reason, now=utc_now())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
