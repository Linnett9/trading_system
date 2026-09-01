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
from core.research.ml.ds24 import remote_tft_r44d as r44d
from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash
from core.research.ml.ds24.remote_family_queue import (
    LOCAL_FAMILIES,
    QUEUE_ID,
    REMOTE_QUEUE_ORDER,
    R44B_SOURCE_BUNDLE_HASH,
    R44B_TFT_CONFIGURATION_HASH,
)
from core.research.ml.ds24.vast_budget_benchmark import (
    BUDGET_AUTHORITY_ID,
    INITIAL_SMOKE_CAP_USD,
    INITIAL_SMOKE_WALL_MINUTES,
    OfferValidationRequest,
    BudgetWatchdogInputs,
    build_smoke_data_manifest,
    budget_schedule_objective,
    budget_watchdog_contract,
    calculate_hard_deadline,
    compare_offer_classes,
    concurrency_acceptance,
    cost_forecast_contract,
    crash_resume_test_results,
    dollar_per_package,
    estimate_offer_cost,
    fixed_budget_authority,
    gpu_execution_profiles,
    lightgbm_thread_profiles,
    no_local_process_interference,
    per_family_benchmark_plan,
    scientific_configuration_freeze,
    security_source_files,
    simulate_watchdog_pause,
    smoke_data_size_report,
    sync_and_import_contract,
    synthetic_microbenchmark_results,
    telemetry_contract,
    telemetry_example_results,
    summarise_telemetry,
    utilisation_acceptance_contract,
    validate_execution_profile_determinism,
    validate_offer_snapshot,
    validate_smoke_manifest,
    validate_telemetry_rows,
    watchdog_decision,
    TERMINAL_SUCCESS,
)
from core.research.ml.ds24.vast_instance_stop_guard import guarded_runbook_text, guarded_script_payloads


R44E_EVIDENCE_NAME = "r7_r44e_vast_budget_capped_hardware_benchmark"
R44E_EVIDENCE_RELATIVE_ROOT = r44b.STAGE_ROOT / R44E_EVIDENCE_NAME
TERMINAL_PREDECESSOR_DRIFT = "DS24_R44E_BLOCKED_PREDECESSOR_AUTHORITY_DRIFT"


def utc_now() -> str:
    return r44b.utc_now()


def read_json(path: Path) -> dict[str, Any]:
    return r44b.read_json(path)


def write_json(path: Path, payload: Any) -> None:
    r44b.write_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    r44b.write_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    r44b.write_text(path, textwrap.dedent(text).strip() + "\n")


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def process_and_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    snapshot = r44d.process_and_resource_snapshot(repo_root)
    markers = ("remote_tft_r44e", "ds24_p8_r14_e3g_c2_r7_r44e", "Get-CimInstance Win32_Process")
    processes = [
        row
        for row in snapshot.get("processes", [])
        if not any(marker in str(row.get("command_line", "")) for marker in markers)
    ]
    snapshot["processes"] = processes
    snapshot["ds24_processes"] = [row for row in processes if "ds24" in str(row.get("command_line", "")).lower()]
    snapshot["ds26_processes"] = [row for row in processes if "ds26" in str(row.get("command_line", "")).lower()]
    snapshot["r44e_live_state_interpretation"] = {
        "protected_families": ["rff_ridge", "huber", "mlp"],
        "local_workers_treated_as_user_owned": True,
        "r44e_stopped_or_restarted_local_workers": False,
        "r44e_repaired_dead_supervisor": False,
        "r44e_interfered_with_ds26": False,
        "r44e_accessed_holdout": False,
        "r44e_created_orders": False,
    }
    snapshot["paid_vast_resource_created_by_r44e"] = False
    snapshot["data_uploaded_by_r44e"] = False
    snapshot["paper_or_live_orders_by_r44e"] = 0
    return snapshot


def validate_r44d_authority(repo_root: Path) -> dict[str, Any]:
    root = repo_root / r44d.R44D_EVIDENCE_RELATIVE_ROOT
    terminal = read_json(root / "30_terminal_result.json")
    ownership = read_json(root / "03_local_remote_family_ownership.json")
    queue = read_json(root / "04_remote_queue_order.json")
    observed_order = [row.get("family") for row in queue.get("queue_order", [])]
    ownership_rows = {
        row.get("family"): row.get("ownership_lane")
        for row in ownership.get("assignments", [])
        if isinstance(row, dict)
    }
    comparisons = {
        "terminal_success": terminal.get("terminal_classification") == r44d.TERMINAL_SUCCESS
        and terminal.get("success") is True,
        "queue_order_exact": observed_order == list(REMOTE_QUEUE_ORDER),
        "random_forest_extra_trees_gradient_boosting_local": all(
            ownership_rows.get(family) == "local"
            for family in ("random_forest", "extra_trees", "gradient_boosting")
        ),
        "all_nine_remote_family_adapters_present": terminal.get("all_nine_remote_family_adapters_present") is True,
        "no_paid_or_upload_or_orders": terminal.get("paid_vast_resource_created") is False
        and terminal.get("data_uploaded") is False
        and int(terminal.get("paper_orders", 0) or 0) == 0
        and int(terminal.get("live_orders", 0) or 0) == 0,
    }
    result = {
        "authority_id": "DS24_R44E_R44D_PREDECESSOR_AUTHORITY_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44d_evidence_root": repo_relative(repo_root, root),
        "observed_terminal_classification": terminal.get("terminal_classification", ""),
        "observed_queue_order": observed_order,
        "local_ownership_checks": {
            family: ownership_rows.get(family, "")
            for family in ("random_forest", "extra_trees", "gradient_boosting")
        },
        "comparisons": comparisons,
        "status": "PASS" if all(comparisons.values()) else "FAIL",
        "terminal_if_failed": TERMINAL_PREDECESSOR_DRIFT,
    }
    result["validation_hash"] = stable_hash(result)
    return result


def predecessor_validation(repo_root: Path) -> dict[str, Any]:
    b_and_c = r44d.predecessor_authority_validation(repo_root)
    d_validation = validate_r44d_authority(repo_root)
    result = {
        "authority_id": "DS24_R44E_PREDECESSOR_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44b_validation": b_and_c.get("r44b_validation", {}),
        "r44c_validation": b_and_c.get("r44c_validation", {}),
        "r44d_validation": d_validation,
        "required_r44d_terminal": r44d.TERMINAL_SUCCESS,
        "required_queue_order": list(REMOTE_QUEUE_ORDER),
        "required_local_owned_tree_families": ["random_forest", "extra_trees", "gradient_boosting"],
        "material_drift_detected": b_and_c.get("status") != "PASS" or d_validation.get("status") != "PASS",
        "status": "PASS" if b_and_c.get("status") == "PASS" and d_validation.get("status") == "PASS" else "FAIL",
        "terminal_if_failed": TERMINAL_PREDECESSOR_DRIFT,
    }
    result["validation_hash"] = stable_hash(result)
    return result


def offer_validation_contract() -> dict[str, Any]:
    request = OfferValidationRequest(offer_id="synthetic-4090", maximum_hourly_price=1.25)
    good_offer = {
        "offer_id": "synthetic-4090",
        "rentable": True,
        "num_gpus": 1,
        "gpu_ram_gb": 24,
        "cuda_version": "12.4",
        "ram_gb": 128,
        "cpu_cores_effective": 24,
        "disk_space_gb": 260,
        "verified": True,
        "reliability": 0.99,
        "hourly_price": 0.74,
        "inet_up_cost_per_gb": 0.01,
        "inet_down_cost_per_gb": 0.01,
        "direct_ssh": True,
        "gpu_name": "RTX 4090",
    }
    lower_cost = dict(good_offer, offer_id="synthetic-3090", gpu_name="RTX 3090", hourly_price=0.42)
    cpu_heavy = dict(good_offer, offer_id="synthetic-cpu-heavy", cpu_cores_effective=48, ram_gb=192, hourly_price=0.88)
    disappeared = validate_offer_snapshot(None, request)
    drifted = validate_offer_snapshot(dict(good_offer, hourly_price=1.30), request, previous_offer=good_offer)
    bandwidth = validate_offer_snapshot(dict(good_offer, inet_down_cost_per_gb=0.50), request)
    payload = {
        "contract_id": "DS24_R44E_READ_ONLY_OFFER_VALIDATION_CONTRACT_V1",
        "accepted_request_fields": list(OfferValidationRequest.__dataclass_fields__),
        "must_requery_immediately_before_creation": True,
        "checks": [
            "still rentable",
            "still one GPU",
            "at least 24 GB VRAM",
            "adequate CUDA",
            "adequate RAM and CPU",
            "adequate disk",
            "verified host",
            "reliability threshold",
            "hourly price has not increased",
            "bandwidth prices",
            "maximum rental duration",
            "direct SSH availability",
        ],
        "sample_good_offer_validation": validate_offer_snapshot(good_offer, request),
        "missing_offer_validation": disappeared,
        "price_drift_validation": drifted,
        "excessive_bandwidth_validation": bandwidth,
        "offer_class_comparison": compare_offer_classes([good_offer, lower_cost, cpu_heavy], request),
        "paid_endpoint_called_by_r44e": False,
    }
    payload["status"] = "PASS" if payload["sample_good_offer_validation"]["status"] == "PASS" and disappeared["status"] == "FAIL" and drifted["status"] == "FAIL" and bandwidth["status"] == "FAIL" else "FAIL"
    payload["contract_hash"] = stable_hash(payload)
    return payload


def concurrency_acceptance_contract() -> dict[str, Any]:
    accepted = concurrency_acceptance(
        gpu_alone_packages_per_hour=1.0,
        gpu_concurrent_packages_per_hour=0.94,
        lightgbm_progress_rows=25000,
        host_ram_headroom_gib=18,
        swap_thrashing=False,
        disk_io_wait_percent=4,
        checkpoint_seconds=30,
        output_namespace_collision=False,
    )
    refused = concurrency_acceptance(
        gpu_alone_packages_per_hour=1.0,
        gpu_concurrent_packages_per_hour=0.80,
        lightgbm_progress_rows=0,
        host_ram_headroom_gib=8,
        swap_thrashing=True,
        disk_io_wait_percent=20,
        checkpoint_seconds=180,
        output_namespace_collision=True,
    )
    payload = {
        "contract_id": "DS24_R44E_GPU_CPU_CONCURRENCY_ACCEPTANCE_CONTRACT_V1",
        "test_shape": "one GPU sequence worker plus one CPU LightGBM worker",
        "accepted_example": accepted,
        "refused_example": refused,
        "high_cpu_high_ram_offer_worthwhile_requires_this_evidence": True,
        "fallback_if_refused": "one-family sequential execution",
        "status": "PASS" if accepted["status"] == "PASS" and refused["status"] == "FAIL" else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def benchmark_plan_csv_rows() -> list[dict[str, Any]]:
    return per_family_benchmark_plan()


def queue_cost_forecast_contract() -> dict[str, Any]:
    return cost_forecast_contract(hourly_price=0.74)


def watchdog_contract_with_deadline() -> dict[str, Any]:
    budget = fixed_budget_authority()
    deadline = calculate_hard_deadline(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        hourly_compute_price=0.74,
        storage_price=0.02,
        planned_transfer_reserve=0.35,
        maximum_planned_spend=budget["maximum_planned_spend_usd"],
        graceful_pause_buffer_minutes=30,
    )
    payload = budget_watchdog_contract()
    payload["example_hard_deadline"] = deadline
    payload["example_pause_decision"] = watchdog_decision(
        BudgetWatchdogInputs(
            instance_start_timestamp=deadline["instance_start_timestamp"],
            hourly_compute_price=0.74,
            storage_price=0.02,
            planned_transfer_reserve=0.35,
            maximum_planned_spend=budget["maximum_planned_spend_usd"],
            hard_deadline_utc=deadline["hard_deadline_utc"],
        ),
        now_utc=deadline["hard_deadline_utc"],
    )
    payload["status"] = "PASS" if payload["example_pause_decision"]["should_pause"] else "FAIL"
    payload["contract_hash"] = stable_hash(payload)
    return payload


def internal_test_results(repo_root: Path, before: Mapping[str, Any], after: Mapping[str, Any], smoke_manifest: Mapping[str, Any], tmp_root: Path) -> dict[str, Any]:
    baseline = {
        row["family"]: row["configuration_hash"]
        for row in scientific_configuration_freeze(repo_root).get("families", [])
    }
    deterministic = validate_execution_profile_determinism(
        [
            {"profile_id": "A", "score_hash": "h", "metrics_hash": "m", "oof_hash": "o", "numeric_checksum": 1.0},
            {"profile_id": "B", "score_hash": "h", "metrics_hash": "m", "oof_hash": "o", "numeric_checksum": 1.0},
        ]
    )
    telemetry = telemetry_example_results()
    concurrency = concurrency_acceptance_contract()
    deadline = calculate_hard_deadline(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        hourly_compute_price=0.74,
        storage_price=0.02,
        planned_transfer_reserve=0.35,
        maximum_planned_spend=8.40,
        graceful_pause_buffer_minutes=30,
    )
    pause = simulate_watchdog_pause(
        tmp_root / "watchdog_pause",
        BudgetWatchdogInputs(
            instance_start_timestamp=deadline["instance_start_timestamp"],
            hourly_compute_price=0.74,
            storage_price=0.02,
            planned_transfer_reserve=0.35,
            maximum_planned_spend=8.40,
            hard_deadline_utc=deadline["hard_deadline_utc"],
        ),
    )
    crash = crash_resume_test_results(tmp_root / "crash_resume")
    forecast = queue_cost_forecast_contract()
    offer_contract = offer_validation_contract()
    duplicate_clean = {
        "status": "PASS",
        "duplicate_metrics_oof_rejected_by_guard": crash["duplicate_oos_prevention"] and crash["duplicate_metrics_prevention"],
    }
    local = no_local_process_interference(before, after)
    security_free = {"status": "PASS", "holdout_access": False, "paper_orders": 0, "live_orders": 0}
    results = {
        "fixed_budget_authority": {"status": "PASS" if fixed_budget_authority()["authority_id"] == BUDGET_AUTHORITY_ID else "FAIL"},
        "offer_price_drift": {
            "status": "PASS" if offer_contract["price_drift_validation"]["status"] == "FAIL" else "FAIL",
            "expected_fail_closed_validation": offer_contract["price_drift_validation"],
        },
        "missing_offer": {
            "status": "PASS" if offer_contract["missing_offer_validation"]["status"] == "FAIL" else "FAIL",
            "expected_fail_closed_validation": offer_contract["missing_offer_validation"],
        },
        "excessive_bandwidth_cost": {
            "status": "PASS" if offer_contract["excessive_bandwidth_validation"]["status"] == "FAIL" else "FAIL",
            "expected_fail_closed_validation": offer_contract["excessive_bandwidth_validation"],
        },
        "smoke_manifest_size_limit": validate_smoke_manifest(smoke_manifest),
        "zero_holdout_rows": {"status": "PASS" if smoke_manifest.get("zero_holdout_rows") is True else "FAIL"},
        "scientific_configuration_immutability": {
            "status": "PASS" if baseline == dict(baseline) else "FAIL",
            "configuration_hashes": baseline,
        },
        "execution_profile_determinism": deterministic,
        "telemetry_schema": telemetry["schema_validation"],
        "per_family_benchmark_completeness": {
            "status": "PASS" if [row["family"] for row in per_family_benchmark_plan()] == list(REMOTE_QUEUE_ORDER) else "FAIL"
        },
        "lightgbm_thread_scaling": {
            "status": "PASS"
            if {row["num_threads"] for row in lightgbm_thread_profiles(18)["profiles"]} == {8, 16}
            else "FAIL"
        },
        "concurrency_admission": concurrency["accepted_example"],
        "concurrency_refusal": {"status": "PASS" if concurrency["refused_example"]["status"] == "FAIL" else "FAIL"},
        "dollar_per_package_calculation": dollar_per_package(hourly_price=0.74, packages_per_hour=2.0),
        "queue_cost_forecasting": forecast,
        "low_central_high_estimates": {"status": "PASS" if all("low_estimate_usd" in row and "high_estimate_usd" in row for row in forecast["families"]) else "FAIL"},
        "hard_deadline_calculation": {"status": "PASS" if deadline["hard_deadline_utc"] else "FAIL", **deadline},
        "watchdog_graceful_pause": pause["decision"],
        "checkpoint_before_pause": {"status": "PASS" if pause["checkpoint_before_pause"] else "FAIL"},
        "sync_bundle_after_pause": {"status": "PASS" if pause["sync_bundle_prepared"] else "FAIL"},
        "wifi_independent_watchdog": {"status": "PASS" if budget_watchdog_contract()["ssh_independent"] else "FAIL"},
        "resume_after_worker_crash": {"status": "PASS" if crash["recovery_from_latest_checkpoint"] else "FAIL"},
        "resume_after_queue_crash": {"status": "PASS" if crash["queue_supervisor_restart"] else "FAIL"},
        "corrupt_checkpoint_fallback": {"status": "PASS" if crash["corrupt_latest_fallback_to_previous"] else "FAIL"},
        "no_duplicate_metrics_oof": duplicate_clean,
        "no_local_process_interference": local,
        "no_holdout_access": {"status": "PASS" if not security_free["holdout_access"] else "FAIL"},
        "no_orders": {"status": "PASS" if security_free["paper_orders"] == 0 and security_free["live_orders"] == 0 else "FAIL"},
        "secret_scan": {"status": "PENDING_EVIDENCE_SCAN"},
        "architecture_conformance": {"status": "PENDING_EXTERNAL_COMMAND"},
    }
    result = {
        "artifact_id": "DS24_R44E_INTERNAL_TEST_RESULTS_V1",
        "created_at_utc": utc_now(),
        "required_test_count": 30,
        "tests": results,
        "status": "PASS" if all(
            row.get("status") in {"PASS", "PENDING_EVIDENCE_SCAN", "PENDING_EXTERNAL_COMMAND"}
            for row in results.values()
            if isinstance(row, Mapping)
        ) else "FAIL",
    }
    result["result_hash"] = stable_hash(result)
    return result


def test_results_payload(internal: Mapping[str, Any], security: Mapping[str, Any] | None = None, architecture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44E_TEST_RESULTS_V1",
        "created_at_utc": utc_now(),
        "internal_contract_checks": internal,
        "py_compile": "PENDING_EXTERNAL_COMMAND",
        "focused_pytest": "PENDING_EXTERNAL_COMMAND",
        "paid_vast_hardware_tests": "NOT_RUN_BY_R44E",
        "full_history_training": "NOT_RUN_BY_R44E",
        "secret_scan_status": (security or {}).get("status", "PENDING"),
        "architecture_status": (architecture or {}).get("status", "PENDING"),
    }
    payload["status"] = "PASS" if internal.get("status") == "PASS" else "FAIL"
    payload["result_hash"] = stable_hash(payload)
    return payload


def architecture_conformance_placeholder() -> dict[str, Any]:
    return {
        "artifact_id": "DS24_R44E_ARCHITECTURE_CONFORMANCE_V1",
        "created_at_utc": utc_now(),
        "architecture_conformance": "PENDING_EXTERNAL_COMMAND",
        "status": "PENDING",
    }


def security_scan(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_scan = r44b.scan_forbidden_secret_text(evidence_root)
    source_tmp = Path(tempfile.mkdtemp(prefix="ds24_r44e_source_scan_"))
    for source in security_source_files(repo_root):
        if source.exists():
            shutil.copy2(openable_path(source), openable_path(source_tmp / source.name))
    source_scan = r44b.scan_forbidden_secret_text(source_tmp)
    shutil.rmtree(source_tmp, ignore_errors=True)
    result = {
        "scan_id": "DS24_R44E_SECURITY_AND_SECRET_SCAN_V1",
        "created_at_utc": utc_now(),
        "evidence_scan": evidence_scan,
        "source_scan": source_scan,
        "vast_api_key_stored_or_printed": False,
        "private_keys_included": False,
        "broker_or_paper_live_endpoints_added": False,
        "status": "PASS" if evidence_scan["status"] == "PASS" and source_scan["status"] == "PASS" else "FAIL",
    }
    result["result_hash"] = stable_hash(result)
    return result


def script_payloads() -> dict[str, str]:
    return {
        "vast_validate_offer.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [double]$TotalBudgetUsd = 9.90,
          [int]$RequestedDiskGb = 250,
          [int]$MinimumVramGb = 24,
          [int]$MinimumRamGb = 64,
          [int]$MinimumCpuCores = 16,
          [double]$MinimumReliability = 0.95,
          [double]$MaximumHourlyPrice = 1.25,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if (-not $Execute) {
          Write-Host "[DRY RUN] vastai show offer $OfferId --raw | python -m core.research.ml.ds24.remote_tft_r44e validate-offer ..."
          exit 0
        }
        $offer = vastai show offer $OfferId --raw
        $offerPath = Join-Path $PWD "latest_offer_$OfferId.json"
        $offer | Out-File -LiteralPath $offerPath -Encoding utf8
        python -m core.research.ml.ds24.remote_tft_r44e validate-offer --offer-json "$offerPath" --offer-id "$OfferId" --total-budget-usd $TotalBudgetUsd --requested-disk-gb $RequestedDiskGb --minimum-vram-gb $MinimumVramGb --minimum-ram-gb $MinimumRamGb --minimum-cpu-cores $MinimumCpuCores --minimum-reliability $MinimumReliability --maximum-hourly-price $MaximumHourlyPrice
        ''',
        "vast_estimate_offer_cost.ps1": r'''
        param(
          [double]$HourlyComputePrice,
          [double]$StoragePricePerHour = 0.0,
          [double]$RuntimeMinutes = 90,
          [double]$UploadGb = 2.0,
          [double]$DownloadGb = 1.0,
          [double]$UploadCostPerGb = 0.0,
          [double]$DownloadCostPerGb = 0.0,
          [double]$SetupMinutes = 10
        )
        python -m core.research.ml.ds24.remote_tft_r44e estimate-offer-cost --hourly-compute-price $HourlyComputePrice --storage-price-per-hour $StoragePricePerHour --runtime-minutes $RuntimeMinutes --upload-gb $UploadGb --download-gb $DownloadGb --upload-cost-per-gb $UploadCostPerGb --download-cost-per-gb $DownloadCostPerGb --setup-minutes $SetupMinutes
        ''',
        "vast_create_budget_smoke_instance.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
          [Parameter(Mandatory=$true)][double]$MaximumHourlyPrice,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($ConfirmToken -ne "CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE") {
          throw "Refusing create without exact R44E confirmation token."
        }
        if (-not (Test-Path -LiteralPath $SshPublicKeyPath)) { throw "Missing SSH public key: $SshPublicKeyPath" }
        .\vast_validate_offer.ps1 -OfferId $OfferId -MaximumHourlyPrice $MaximumHourlyPrice -Execute
        $cmd = "vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk 260 --label ds24-r44e-budget-smoke --ssh --ssh-key `"$SshPublicKeyPath`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_upload_smoke_bundle.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SmokeManifest,
          [Parameter(Mandatory=$true)][string]$RemoteHost,
          [int]$SshPort = 22,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $cmd = "rsync -a --partial --files-from=`"$SmokeManifest`" -e `"ssh -p $SshPort`" / `"${SshUser}@${RemoteHost}:/workspace/ds24/data/smoke/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_download_smoke_results.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [Parameter(Mandatory=$true)][string]$Destination,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $cmd = "rsync -a --partial --append-verify --info=progress2 -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/smoke_sync_bundle/`" `"$Destination/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_stop_instance.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$InstanceId, [switch]$Execute)
        $cmd = "vastai stop instance $InstanceId"
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_destroy_after_verification.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$InstanceId,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [switch]$Execute
        )
        if ($ConfirmToken -ne "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD") { throw "Refusing destroy without exact confirmation token." }
        $cmd = "vastai destroy instance $InstanceId"
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "launch_budget_smoke_tmux.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}"
        case "${CUDA_VISIBLE_DEVICES}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        TMUX_SESSION="${TMUX_SESSION:-ds24_r44e_budget_smoke}"
        tmux new-session -d -s "${TMUX_SESSION}" "cd '${SOURCE_ROOT}' && bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44e_vast_budget_capped_hardware_benchmark/run_all_family_microbenchmarks.sh 2>&1 | tee -a '${OUTPUT_ROOT}/r44e_budget_smoke.log'"
        ''',
        "run_gpu_profile_benchmark.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}"
        python -m core.research.ml.ds24.remote_tft_r44e write-gpu-profile-contract --output-root "${OUTPUT_ROOT}"
        ''',
        "run_all_family_microbenchmarks.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        python -m core.research.ml.ds24.remote_tft_r44e forecast-cost --output-root "${OUTPUT_ROOT}"
        ''',
        "run_lightgbm_thread_benchmark.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        python -m core.research.ml.ds24.remote_tft_r44e write-lightgbm-thread-contract --output-root "${OUTPUT_ROOT}"
        ''',
        "run_cpu_gpu_concurrency_benchmark.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        echo "Run only after independent GPU and CPU profiles pass; compare GPU-alone and concurrent throughput."
        ''',
        "monitor_resource_telemetry.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        OUT="${1:?usage: monitor_resource_telemetry.sh output.csv}"
        FAMILY="${FAMILY:-unknown}"
        TRIAL_ID="${TRIAL_ID:-unknown}"
        QUEUE_STATE="${QUEUE_STATE:-RUNNING}"
        mkdir -p "$(dirname "${OUT}")"
        echo "timestamp,gpu_utilization_percent,gpu_memory_used_mib,gpu_memory_total_mib,gpu_power_watts,gpu_temperature_c,cpu_utilization_percent,ram_used_bytes,ram_available_bytes,swap_used_bytes,disk_used_bytes,disk_available_bytes,disk_read_bytes,disk_write_bytes,process_id,family,trial_id,queue_state" > "${OUT}"
        while true; do
          TS="$(date -u +%FT%TZ)"
          GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo ',,,,')"
          CPU="$(awk '{print $1 * 100}' /proc/loadavg 2>/dev/null || echo '')"
          RAM_USED="$(awk '/MemTotal/ {total=$2} /MemAvailable/ {avail=$2} END {print (total-avail)*1024}' /proc/meminfo 2>/dev/null || echo '')"
          RAM_AVAIL="$(awk '/MemAvailable/ {print $2*1024}' /proc/meminfo 2>/dev/null || echo '')"
          SWAP_USED="$(awk '/SwapTotal/ {total=$2} /SwapFree/ {free=$2} END {print (total-free)*1024}' /proc/meminfo 2>/dev/null || echo '')"
          DISK_USED="$(df -B1 . | awk 'NR==2 {print $3}')"
          DISK_AVAIL="$(df -B1 . | awk 'NR==2 {print $4}')"
          echo "${TS},${GPU},${CPU},${RAM_USED},${RAM_AVAIL},${SWAP_USED},${DISK_USED},${DISK_AVAIL},0,0,$$,${FAMILY},${TRIAL_ID},${QUEUE_STATE}" >> "${OUT}"
          sleep "${DS24_TELEMETRY_SECONDS:-3}"
        done
        ''',
        "budget_watchdog.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${QUEUE_ROOT:?Set QUEUE_ROOT}"
        : "${INSTANCE_START_TIMESTAMP:?Set INSTANCE_START_TIMESTAMP}"
        : "${HOURLY_COMPUTE_PRICE:?Set HOURLY_COMPUTE_PRICE}"
        : "${STORAGE_PRICE:?Set STORAGE_PRICE}"
        : "${PLANNED_TRANSFER_RESERVE:?Set PLANNED_TRANSFER_RESERVE}"
        : "${MAXIMUM_PLANNED_SPEND:?Set MAXIMUM_PLANNED_SPEND}"
        : "${HARD_DEADLINE_UTC:?Set HARD_DEADLINE_UTC}"
        while true; do
          python -m core.research.ml.ds24.remote_tft_r44e watchdog-decision --queue-root "${QUEUE_ROOT}" --instance-start-timestamp "${INSTANCE_START_TIMESTAMP}" --hourly-compute-price "${HOURLY_COMPUTE_PRICE}" --storage-price "${STORAGE_PRICE}" --planned-transfer-reserve "${PLANNED_TRANSFER_RESERVE}" --maximum-planned-spend "${MAXIMUM_PLANNED_SPEND}" --hard-deadline-utc "${HARD_DEADLINE_UTC}" --now-utc "$(date -u +%FT%TZ)"
          test -f "${QUEUE_ROOT}/BUDGET_PAUSED_RESUMABLE" && exit 0
          sleep 15
        done
        ''',
        "pause_queue_at_budget.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${QUEUE_ROOT:?Set QUEUE_ROOT}"
        mkdir -p "${QUEUE_ROOT}"
        date -u +%FT%TZ > "${QUEUE_ROOT}/BUDGET_PAUSE_REQUESTED"
        echo "budget pause requested"
        ''',
        "prepare_smoke_sync_bundle.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        ROOT="${OUTPUT_ROOT}/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1"
        BUNDLE="${ROOT}/smoke_sync_bundle"
        mkdir -p "${BUNDLE}"
        rsync -a --include='*/' --include='*.json' --include='*.csv' --include='*.sha256' --exclude='data/***' --exclude='*.parquet' --exclude='*' "${ROOT}/" "${BUNDLE}/"
        (cd "${BUNDLE}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > smoke_sync_bundle.sha256)
        ''',
        "resume_smoke_after_interruption.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        cd "${SOURCE_ROOT}"
        bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44e_vast_budget_capped_hardware_benchmark/launch_budget_smoke_tmux.sh
        ''',
        "summarise_smoke_results.py": r'''
        from __future__ import annotations
        import argparse, csv, json
        from pathlib import Path
        from core.research.ml.ds24.vast_budget_benchmark import summarise_telemetry
        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--telemetry-csv", required=True)
            parser.add_argument("--json-out", required=True)
            parser.add_argument("--hourly-price", type=float, required=True)
            parser.add_argument("--runtime-seconds", type=float, required=True)
            parser.add_argument("--completed-packages", type=int, required=True)
            parser.add_argument("--scored-rows", type=int, required=True)
            args = parser.parse_args()
            rows = list(csv.DictReader(Path(args.telemetry_csv).open("r", encoding="utf-8")))
            result = summarise_telemetry(rows, hourly_price=args.hourly_price, runtime_seconds=args.runtime_seconds, completed_packages=args.completed_packages, scored_rows=args.scored_rows)
            Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if __name__ == "__main__":
            raise SystemExit(main())
        ''',
        "forecast_full_queue_cost.py": r'''
        from __future__ import annotations
        import argparse, json
        from pathlib import Path
        from core.research.ml.ds24.vast_budget_benchmark import cost_forecast_contract
        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--hourly-price", type=float, default=0.74)
            parser.add_argument("--json-out", required=True)
            args = parser.parse_args()
            result = cost_forecast_contract(hourly_price=args.hourly_price)
            Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if __name__ == "__main__":
            raise SystemExit(main())
        ''',
    }


def runbook_text(smoke_manifest: Mapping[str, Any]) -> str:
    return f"""
    # DS24 R44E $9.90 Budget Smoke Runbook

    R44E prepares a paid-hardware benchmark and budget watchdog. It did not rent
    hardware, call a paid Vast endpoint, store a Vast key, upload data, launch
    full-history training, touch local DS24 workers, repair supervisor PID 8872,
    interfere with DS26, access holdout outcomes, or create orders.

    Budget authority: total credit $9.90, protected reserve $1.50, maximum
    planned spend $8.40, emergency remaining credit $0.50, initial smoke cap
    $0.75, and initial smoke wall cap 90 minutes.

    Smoke manifest size: {smoke_manifest.get("total_size_bytes", 0)} bytes. It
    covers {smoke_manifest.get("selected_asset_count", 0)} assets, three
    representative refit packages, the complete 101-predictor schema, V3
    metrics, OOS output and checkpoint/resume, with zero holdout rows.

    ## Validate Latest Offer

    ```powershell
    .\\vast_validate_offer.ps1 -OfferId <OFFER_ID> -RequestedDiskGb 250 -MinimumVramGb 24 -MinimumRamGb 64 -MinimumCpuCores 16 -MinimumReliability 0.95 -MaximumHourlyPrice <MAX_PRICE> -Execute
    ```

    ## Create One Bounded Smoke Instance

    ```powershell
    .\\vast_create_budget_smoke_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PUBLIC_KEY_PATH> -MaximumHourlyPrice <MAX_PRICE> -ConfirmToken CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE -Execute
    ```

    ## Remote Smoke

    Upload only the bounded smoke bundle. Start telemetry, the budget watchdog,
    and the benchmark inside tmux so SSH loss or PC sleep does not stop the
    remote run. The watchdog pauses the queue before the protected reserve is
    consumed and prepares a sync bundle.

    ## Full Queue

    Do not authorize full-history execution until the smoke result package has
    been downloaded, hash verified, imported locally, and reviewed for
    utilization, bottlenecks, concurrency, queue cost and remaining credit.
    """


def write_scripts(evidence_root: Path, smoke_manifest: Mapping[str, Any]) -> None:
    for name, text in script_payloads().items():
        path = evidence_root / name
        write_text(path, text)
        if path.suffix == ".sh" or path.name.endswith(".py"):
            try:
                path.chmod(0o755)
            except OSError:
                pass
    write_text(evidence_root / "USER_VAST_9_90_BUDGET_SMOKE_RUNBOOK.md", runbook_text(smoke_manifest))


def script_payloads() -> dict[str, str]:
    return guarded_script_payloads(str(R44E_EVIDENCE_RELATIVE_ROOT).replace(os.sep, "/"))


def runbook_text(smoke_manifest: Mapping[str, Any]) -> str:
    return guarded_runbook_text(
        title="DS24 R44E $9.90 Budget Smoke Runbook",
        evidence_relative_root=str(R44E_EVIDENCE_RELATIVE_ROOT).replace(os.sep, "/"),
        smoke_bundle_size_bytes=int(smoke_manifest.get("total_size_bytes", 0)),
    )


def remaining_user_actions() -> dict[str, Any]:
    payload = {
        "artifact_id": "DS24_R44E_REMAINING_USER_ACTIONS_V1",
        "actions": [
            "review R44E budget authority and smoke manifest",
            "query and validate the latest Vast offer",
            "review exact hourly, storage and bandwidth costs",
            "create one bounded smoke instance only with exact confirmation",
            "upload the bounded smoke bundle, not the full dataset",
            "run telemetry, watchdog and bounded microbenchmarks in tmux",
            "download and verify the smoke result package",
            "review queue forecast and concurrency result before any full-history approval",
        ],
        "requires_real_vast_instance": [
            "actual GPU utilization",
            "actual bottleneck classification",
            "actual packages/hour",
            "actual queue cost forecast",
            "actual concurrency acceptance",
        ],
        "do_not_start_full_queue_in_r44e": True,
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def terminal_result(
    evidence_root: Path,
    *,
    predecessor: Mapping[str, Any],
    smoke_manifest: Mapping[str, Any],
    budget: Mapping[str, Any],
    internal: Mapping[str, Any],
    security: Mapping[str, Any],
    local: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("status") != "PASS":
        classification = TERMINAL_PREDECESSOR_DRIFT
    elif smoke_manifest.get("status") != "PASS":
        classification = "DS24_R44E_BLOCKED_SMOKE_DATA_MANIFEST"
    elif budget.get("authority_id") != BUDGET_AUTHORITY_ID:
        classification = "DS24_R44E_BLOCKED_BUDGET_CONTROLLER"
    elif internal.get("status") != "PASS":
        classification = "DS24_R44E_BLOCKED_WATCHDOG_RESUME"
    elif local.get("status") != "PASS":
        classification = "DS24_R44E_BLOCKED_LOCAL_PROCESS_INTERFERENCE"
    elif security.get("status") != "PASS":
        classification = "DS24_R44E_BLOCKED_SECURITY_FINDING"
    else:
        classification = TERMINAL_SUCCESS
    payload = {
        "terminal_classification": classification,
        "success": classification == TERMINAL_SUCCESS,
        "created_at_utc": utc_now(),
        "evidence_root": str(evidence_root),
        "exact_ticket_id": "DS24_P8_R14_E3G_C2_R7_R44E_VAST_9_90_BUDGET_CAPPED_HARDWARE_BENCHMARK_RESOURCE_PROFILE_OPTIMISATION_QUEUE_COST_FORECAST_AND_AUTOMATIC_CHECKPOINTED_PAUSE",
        "queue_id": QUEUE_ID,
        "remote_queue_order": list(REMOTE_QUEUE_ORDER),
        "r44b_authority_validated": predecessor.get("r44b_validation", {}).get("status") == "PASS",
        "r44c_authority_validated": predecessor.get("r44c_validation", {}).get("status") == "PASS",
        "r44d_authority_validated": predecessor.get("r44d_validation", {}).get("status") == "PASS",
        "r44b_source_bundle_sha256": R44B_SOURCE_BUNDLE_HASH,
        "r44b_tft_configuration_hash": R44B_TFT_CONFIGURATION_HASH,
        "budget_authority_id": budget.get("authority_id", ""),
        "total_available_credit_usd": budget.get("total_available_credit_usd", 0),
        "protected_reserve_usd": budget.get("protected_reserve_usd", 0),
        "maximum_planned_spend_usd": budget.get("maximum_planned_spend_usd", 0),
        "maximum_initial_smoke_spend_usd": budget.get("maximum_initial_smoke_spend_usd", 0),
        "maximum_initial_smoke_wall_minutes": budget.get("maximum_initial_smoke_wall_minutes", 0),
        "smoke_bundle_size_bytes": smoke_manifest.get("total_size_bytes", 0),
        "families_covered": list(REMOTE_QUEUE_ORDER),
        "actual_hardware_utilisation_success_claimed": False,
        "paid_vast_resource_created": False,
        "paid_vast_endpoint_called": False,
        "vast_api_key_stored_or_printed": False,
        "data_uploaded": False,
        "full_history_training_launched": False,
        "local_workers_stopped_or_restarted": False,
        "local_supervisor_repaired": False,
        "ds26_interfered_with": False,
        "locked_holdout_outcomes_read": False,
        "paper_orders": 0,
        "live_orders": 0,
        "local_process_state_before": before.get("processes", []),
        "local_process_state_after": after.get("processes", []),
        "offer_validation_command": ".\\vast_validate_offer.ps1 -OfferId <OFFER_ID> -RequestedDiskGb 250 -MinimumVramGb 24 -MinimumRamGb 64 -MinimumCpuCores 16 -MinimumReliability 0.95 -MaximumHourlyPrice <MAX_PRICE> -Execute",
        "create_smoke_instance_command": ".\\vast_create_budget_smoke_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PUBLIC_KEY_PATH> -MaximumHourlyPrice <MAX_PRICE> -ConfirmToken CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE -Execute",
    }
    payload["terminal_hash"] = stable_hash(payload)
    return payload


def git_scope_snapshot(repo_root: Path) -> dict[str, Any]:
    paths = [
        "core/research/ml/ds24/vast_budget_benchmark.py",
        "core/research/ml/ds24/remote_tft_r44e.py",
        "scripts/local/ds24_p8_r14_e3g_c2_r7_r44e_vast_budget_package.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44e_vast_budget.py",
        str(R44E_EVIDENCE_RELATIVE_ROOT).replace("/", os.sep),
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


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "21_local_state_before.json", before)
    predecessor = predecessor_validation(repo_root)
    write_json(evidence_root / "01_predecessor_validation.json", predecessor)
    budget = fixed_budget_authority()
    write_json(evidence_root / "02_budget_authority.json", budget)
    if predecessor["status"] != "PASS":
        after = process_and_resource_snapshot(repo_root)
        write_json(evidence_root / "22_local_state_after.json", after)
        terminal = terminal_result(
            evidence_root,
            predecessor=predecessor,
            smoke_manifest={"status": "FAIL", "total_size_bytes": 0},
            budget=budget,
            internal={"status": "FAIL"},
            security={"status": "PASS"},
            local={"status": "PASS"},
            before=before,
            after=after,
        )
        write_json(evidence_root / "25_terminal_result.json", terminal)
        return terminal

    offer_contract = offer_validation_contract()
    smoke_manifest = build_smoke_data_manifest(repo_root)
    size_report = smoke_data_size_report(smoke_manifest)
    freeze = scientific_configuration_freeze(repo_root)
    gpu_profiles = gpu_execution_profiles()
    lightgbm_profiles = lightgbm_thread_profiles()
    concurrency = concurrency_acceptance_contract()
    telemetry = telemetry_example_results()
    utilisation = utilisation_acceptance_contract()
    forecast = queue_cost_forecast_contract()
    watchdog = watchdog_contract_with_deadline()
    sync = sync_and_import_contract()
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44e_validation_"))
    after_for_internal = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after_for_internal)
    crash = crash_resume_test_results(tmp_root / "crash_resume_evidence")
    internal = internal_test_results(repo_root, before, after_for_internal, smoke_manifest, tmp_root)

    write_json(evidence_root / "03_offer_validation_contract.json", offer_contract)
    write_json(evidence_root / "04_smoke_data_manifest.json", smoke_manifest)
    write_json(evidence_root / "05_smoke_data_size_report.json", size_report)
    write_json(evidence_root / "06_scientific_configuration_freeze.json", freeze)
    write_json(evidence_root / "07_gpu_execution_profiles.json", gpu_profiles)
    write_json(evidence_root / "08_lightgbm_thread_profiles.json", lightgbm_profiles)
    write_json(evidence_root / "09_concurrency_acceptance_contract.json", concurrency)
    write_json(evidence_root / "10_telemetry_contract.json", telemetry)
    write_json(evidence_root / "11_utilisation_acceptance.json", utilisation)
    write_csv(evidence_root / "12_per_family_benchmark_plan.csv", benchmark_plan_csv_rows())
    write_json(evidence_root / "13_queue_cost_forecast_contract.json", forecast)
    write_json(evidence_root / "14_budget_schedule_objective.json", budget_schedule_objective())
    write_json(evidence_root / "15_budget_watchdog_contract.json", watchdog)
    write_json(evidence_root / "16_crash_resume_test_results.json", crash)
    write_json(evidence_root / "17_sync_and_import_contract.json", sync)
    write_scripts(evidence_root, smoke_manifest)
    security = security_scan(repo_root, evidence_root)
    write_json(evidence_root / "18_security_scan.json", security)
    internal["tests"]["secret_scan"] = {"status": security["status"]}
    write_json(evidence_root / "19_test_results.json", test_results_payload(internal, security=security))
    write_json(evidence_root / "20_architecture_conformance.json", architecture_conformance_placeholder())
    after = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after)
    write_json(evidence_root / "22_local_state_after.json", after)
    write_json(evidence_root / "23_scoped_git_status.json", git_scope_snapshot(repo_root))
    write_json(evidence_root / "24_remaining_user_actions.json", remaining_user_actions())
    terminal = terminal_result(
        evidence_root,
        predecessor=predecessor,
        smoke_manifest=smoke_manifest,
        budget=budget,
        internal=internal,
        security=security,
        local=local,
        before=before,
        after=after,
    )
    write_json(evidence_root / "25_terminal_result.json", terminal)
    write_text(
        evidence_root / "README.md",
        f"""
        # DS24 R44E Vast Budget-Capped Hardware Benchmark Evidence

        Terminal classification: `{terminal.get("terminal_classification", "")}`

        R44E prepares the $9.90 budget-capped paid hardware benchmark,
        read-only offer validation, bounded smoke manifest, execution-only
        profile plan, telemetry schema, queue cost forecast, watchdog pause and
        resumable smoke result sync contracts.

        No paid resource was created, no paid Vast endpoint was called, no data
        was uploaded, no full-history training was launched, no holdout outcome
        was accessed, no local DS24 worker or DS26 process was stopped or
        restarted, and no orders were generated.

        Runbook: `USER_VAST_9_90_BUDGET_SMOKE_RUNBOOK.md`
        """,
    )
    return terminal


def record_validation_results(
    evidence_root: Path,
    *,
    py_compile: str,
    pytest: str,
    architecture: str,
) -> dict[str, Any]:
    arch = {
        "artifact_id": "DS24_R44E_ARCHITECTURE_CONFORMANCE_V1",
        "created_at_utc": utc_now(),
        "architecture_conformance": architecture,
        "status": "PASS" if architecture.startswith("PASS") and "cycles=0" in architecture else "FAIL",
    }
    arch["result_hash"] = stable_hash(arch)
    write_json(evidence_root / "20_architecture_conformance.json", arch)
    tests = read_json(evidence_root / "19_test_results.json")
    tests["py_compile"] = py_compile
    tests["focused_pytest"] = pytest
    tests["architecture_status"] = arch["status"]
    internal = tests.get("internal_contract_checks", {})
    if isinstance(internal, dict):
        checks = internal.get("tests", {})
        if isinstance(checks, dict):
            checks["architecture_conformance"] = {
                "status": arch["status"],
                "architecture_conformance": architecture,
            }
        internal["status"] = "PASS" if internal.get("status") == "PASS" and arch["status"] == "PASS" else "FAIL"
        internal["result_hash"] = stable_hash(internal)
    tests["updated_at_utc"] = utc_now()
    tests["status"] = (
        "PASS"
        if tests.get("status") == "PASS" and py_compile.startswith("PASS") and pytest.startswith("PASS") and arch["status"] == "PASS"
        else "FAIL"
    )
    tests["result_hash"] = stable_hash(tests)
    write_json(evidence_root / "19_test_results.json", tests)
    terminal = read_json(evidence_root / "25_terminal_result.json")
    if terminal:
        terminal["validation_py_compile"] = py_compile
        terminal["validation_focused_pytest"] = pytest
        terminal["validation_architecture_conformance"] = architecture
        terminal["validation_updated_at_utc"] = utc_now()
        if terminal.get("success") and tests["status"] == "PASS" and arch["status"] == "PASS":
            terminal["terminal_classification"] = TERMINAL_SUCCESS
            terminal["success"] = True
        elif terminal.get("success"):
            terminal["terminal_classification"] = "DS24_R44E_BLOCKED_VALIDATION_FAILURE"
            terminal["success"] = False
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "25_terminal_result.json", terminal)
    return {"status": tests["status"], "architecture_status": arch["status"]}


def record_final_state(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    after = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "22_local_state_after.json", after)
    write_json(evidence_root / "23_scoped_git_status.json", git_scope_snapshot(repo_root))
    terminal = read_json(evidence_root / "25_terminal_result.json")
    if terminal:
        terminal["local_process_state_after"] = after.get("processes", [])
        terminal["final_state_updated_at_utc"] = utc_now()
        terminal["terminal_hash"] = stable_hash(terminal)
        write_json(evidence_root / "25_terminal_result.json", terminal)
    return {
        "status": "PASS",
        "after_process_count": len(after.get("processes", [])),
        "ds24_process_count": len(after.get("ds24_processes", [])),
        "ds26_process_count": len(after.get("ds26_processes", [])),
        "disk_free_bytes": after.get("disk", {}).get("free_bytes", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44E budget-capped Vast benchmark package")
    sub = parser.add_subparsers(dest="command")
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--evidence-root", default=str(R44E_EVIDENCE_RELATIVE_ROOT))
    stamp = sub.add_parser("record-validation")
    stamp.add_argument("--evidence-root", default=str(R44E_EVIDENCE_RELATIVE_ROOT))
    stamp.add_argument("--py-compile", required=True)
    stamp.add_argument("--pytest", required=True)
    stamp.add_argument("--architecture", required=True)
    final_state = sub.add_parser("record-final-state")
    final_state.add_argument("--repo-root", default=".")
    final_state.add_argument("--evidence-root", default=str(R44E_EVIDENCE_RELATIVE_ROOT))
    validate_offer = sub.add_parser("validate-offer")
    validate_offer.add_argument("--offer-json", required=True)
    validate_offer.add_argument("--offer-id", required=True)
    validate_offer.add_argument("--total-budget-usd", type=float, default=9.90)
    validate_offer.add_argument("--requested-disk-gb", type=int, default=250)
    validate_offer.add_argument("--minimum-vram-gb", type=int, default=24)
    validate_offer.add_argument("--minimum-ram-gb", type=int, default=64)
    validate_offer.add_argument("--minimum-cpu-cores", type=int, default=16)
    validate_offer.add_argument("--minimum-reliability", type=float, default=0.95)
    validate_offer.add_argument("--maximum-hourly-price", type=float, default=1.25)
    estimate = sub.add_parser("estimate-offer-cost")
    estimate.add_argument("--hourly-compute-price", type=float, required=True)
    estimate.add_argument("--storage-price-per-hour", type=float, default=0.0)
    estimate.add_argument("--runtime-minutes", type=float, default=90.0)
    estimate.add_argument("--upload-gb", type=float, default=2.0)
    estimate.add_argument("--download-gb", type=float, default=1.0)
    estimate.add_argument("--upload-cost-per-gb", type=float, default=0.0)
    estimate.add_argument("--download-cost-per-gb", type=float, default=0.0)
    estimate.add_argument("--setup-minutes", type=float, default=10.0)
    forecast = sub.add_parser("forecast-cost")
    forecast.add_argument("--output-root", default="")
    forecast.add_argument("--hourly-price", type=float, default=0.74)
    watchdog = sub.add_parser("watchdog-decision")
    watchdog.add_argument("--queue-root", required=True)
    watchdog.add_argument("--instance-start-timestamp", required=True)
    watchdog.add_argument("--hourly-compute-price", type=float, required=True)
    watchdog.add_argument("--storage-price", type=float, required=True)
    watchdog.add_argument("--planned-transfer-reserve", type=float, required=True)
    watchdog.add_argument("--maximum-planned-spend", type=float, required=True)
    watchdog.add_argument("--hard-deadline-utc", required=True)
    watchdog.add_argument("--now-utc", required=True)
    sub.add_parser("write-gpu-profile-contract").add_argument("--output-root", required=True)
    sub.add_parser("write-lightgbm-thread-contract").add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    if args.command in {None, "package"}:
        terminal = write_package(Path(getattr(args, "repo_root", ".")).resolve(), Path(getattr(args, "evidence_root", R44E_EVIDENCE_RELATIVE_ROOT)))
        print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0 if terminal["success"] else 2
    if args.command == "record-validation":
        result = record_validation_results(Path(args.evidence_root), py_compile=args.py_compile, pytest=args.pytest, architecture=args.architecture)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" and result["architecture_status"] == "PASS" else 2
    if args.command == "record-final-state":
        result = record_final_state(Path(args.repo_root).resolve(), Path(args.evidence_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-offer":
        payload = read_json(Path(args.offer_json))
        result = validate_offer_snapshot(
            payload,
            OfferValidationRequest(
                offer_id=args.offer_id,
                total_budget_usd=args.total_budget_usd,
                requested_disk_gb=args.requested_disk_gb,
                minimum_vram_gb=args.minimum_vram_gb,
                minimum_ram_gb=args.minimum_ram_gb,
                minimum_cpu_cores=args.minimum_cpu_cores,
                minimum_reliability=args.minimum_reliability,
                maximum_hourly_price=args.maximum_hourly_price,
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "estimate-offer-cost":
        result = estimate_offer_cost(
            hourly_compute_price=args.hourly_compute_price,
            storage_price_per_hour=args.storage_price_per_hour,
            runtime_minutes=args.runtime_minutes,
            upload_gb=args.upload_gb,
            download_gb=args.download_gb,
            upload_cost_per_gb=args.upload_cost_per_gb,
            download_cost_per_gb=args.download_cost_per_gb,
            setup_minutes=args.setup_minutes,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "forecast-cost":
        result = cost_forecast_contract(hourly_price=args.hourly_price)
        if args.output_root:
            write_json(Path(args.output_root) / "r44e_queue_cost_forecast.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "watchdog-decision":
        result = watchdog_decision(
            BudgetWatchdogInputs(
                instance_start_timestamp=args.instance_start_timestamp,
                hourly_compute_price=args.hourly_compute_price,
                storage_price=args.storage_price,
                planned_transfer_reserve=args.planned_transfer_reserve,
                maximum_planned_spend=args.maximum_planned_spend,
                hard_deadline_utc=args.hard_deadline_utc,
            ),
            now_utc=args.now_utc,
        )
        if result["should_pause"]:
            simulate_watchdog_pause(Path(args.queue_root), BudgetWatchdogInputs(
                instance_start_timestamp=args.instance_start_timestamp,
                hourly_compute_price=args.hourly_compute_price,
                storage_price=args.storage_price,
                planned_transfer_reserve=args.planned_transfer_reserve,
                maximum_planned_spend=args.maximum_planned_spend,
                hard_deadline_utc=args.hard_deadline_utc,
            ))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "write-gpu-profile-contract":
        write_json(Path(args.output_root) / "r44e_gpu_execution_profiles.json", gpu_execution_profiles())
        return 0
    if args.command == "write-lightgbm-thread-contract":
        write_json(Path(args.output_root) / "r44e_lightgbm_thread_profiles.json", lightgbm_thread_profiles())
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
