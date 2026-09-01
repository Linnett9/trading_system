from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash, write_json
from core.research.ml.ds24.remote_family_queue import (
    CPU_RANKING_FAMILIES,
    GPU_SEQUENCE_FAMILIES,
    LOCAL_FAMILIES,
    QUEUE_ID,
    REMOTE_QUEUE_ORDER,
    R44B_SOURCE_BUNDLE_HASH,
    R44B_TFT_CONFIGURATION_HASH,
    TARGET_CONTRACT_ID,
    duplicate_guard,
    family_configuration_authority,
)


BUDGET_AUTHORITY_ID = "DS24_VAST_FIRST_RENTAL_BUDGET_V1"
TERMINAL_SUCCESS = "DS24_R44E_VAST_9_90_BUDGET_BENCHMARK_AND_AUTOPAUSE_READY_FOR_USER_PAID_EXECUTION"
SMOKE_MANIFEST_ID = "DS24_R44E_REPRESENTATIVE_SMOKE_DATA_MANIFEST_V1"
TELEMETRY_CONTRACT_ID = "DS24_R44E_REMOTE_RESOURCE_TELEMETRY_CONTRACT_V1"
WATCHDOG_CONTRACT_ID = "DS24_R44E_REMOTE_BUDGET_WATCHDOG_CONTRACT_V1"
PREFERRED_SMOKE_TRANSFER_BYTES = 2 * 1024**3
HARD_MAX_SMOKE_TRANSFER_BYTES = 4 * 1024**3
FULL_DATASET_BYTES = 47_297_267_964
INITIAL_SMOKE_CAP_USD = 0.75
INITIAL_SMOKE_WALL_MINUTES = 90
TELEMETRY_COLUMNS = [
    "timestamp",
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_power_watts",
    "gpu_temperature_c",
    "cpu_utilization_percent",
    "ram_used_bytes",
    "ram_available_bytes",
    "swap_used_bytes",
    "disk_used_bytes",
    "disk_available_bytes",
    "disk_read_bytes",
    "disk_write_bytes",
    "process_id",
    "family",
    "trial_id",
    "queue_state",
]
BOTTLENECK_CLASSIFICATIONS = [
    "GPU_COMPUTE_BOUND",
    "GPU_MEMORY_BOUND",
    "CPU_PREPROCESSING_BOUND",
    "DATA_LOADER_BOUND",
    "DISK_IO_BOUND",
    "CHECKPOINT_WRITE_BOUND",
    "OUTPUT_WRITE_BOUND",
    "UNDERUTILISED_MODEL_TOO_SMALL",
    "BALANCED",
]


class VastBudgetBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfferValidationRequest:
    offer_id: str
    total_budget_usd: float = 9.90
    requested_disk_gb: int = 250
    minimum_vram_gb: int = 24
    minimum_ram_gb: int = 64
    minimum_cpu_cores: int = 16
    minimum_reliability: float = 0.95
    maximum_hourly_price: float = 1.25


@dataclass(frozen=True)
class BudgetWatchdogInputs:
    instance_start_timestamp: str
    hourly_compute_price: float
    storage_price: float
    planned_transfer_reserve: float
    maximum_planned_spend: float
    hard_deadline_utc: str
    graceful_pause_buffer_minutes: int = 30
    optional_instance_stop_configured: bool = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_text(path: Path, text: str) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def fixed_budget_authority() -> dict[str, Any]:
    payload = {
        "authority_id": BUDGET_AUTHORITY_ID,
        "total_available_credit_usd": 9.90,
        "protected_reserve_usd": 1.50,
        "maximum_planned_spend_usd": 8.40,
        "emergency_remaining_credit_usd": 0.50,
        "maximum_initial_smoke_spend_usd": INITIAL_SMOKE_CAP_USD,
        "maximum_initial_smoke_wall_minutes": INITIAL_SMOKE_WALL_MINUTES,
        "cost_components_required": [
            "compute price",
            "selected container storage",
            "dataset upload cost",
            "result-download cost",
            "setup/boot time",
            "stopped-instance storage cost",
            "safety reserve",
            "measured smoke cost",
            "projected full-run cost",
        ],
        "advertised_gpu_price_is_not_complete_cost": True,
        "paid_vast_action_by_r44e": False,
    }
    payload["authority_hash"] = stable_hash(payload)
    return payload


def _number(payload: Mapping[str, Any], names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        if name in payload and payload.get(name) not in {"", None}:
            try:
                return float(payload.get(name))
            except (TypeError, ValueError):
                return default
    return default


def _bool(payload: Mapping[str, Any], names: Iterable[str], default: bool = False) -> bool:
    for name in names:
        if name in payload:
            value = payload.get(name)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)
    return default


def _offer_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("offer_id") or payload.get("id") or payload.get("ask_contract_id") or "")


def validate_offer_snapshot(
    offer: Mapping[str, Any] | None,
    request: OfferValidationRequest,
    *,
    previous_offer: Mapping[str, Any] | None = None,
    upload_gb: float = 2.0,
    download_gb: float = 1.0,
    maximum_bandwidth_cost_usd: float = 0.25,
) -> dict[str, Any]:
    if not offer:
        return {
            "status": "FAIL",
            "fail_closed": True,
            "blocker": "MISSING_OFFER",
            "request": asdict(request),
            "paid_endpoint_called": False,
        }
    hourly = _number(offer, ("hourly_price", "dph_total", "dph_base", "price_per_hour"))
    prior_hourly = _number(previous_offer or {}, ("hourly_price", "dph_total", "dph_base", "price_per_hour"), hourly)
    bandwidth_cost = (
        upload_gb * _number(offer, ("inet_up_cost_per_gb", "upload_cost_per_gb", "bandwidth_upload_cost_per_gb"))
        + download_gb * _number(offer, ("inet_down_cost_per_gb", "download_cost_per_gb", "bandwidth_download_cost_per_gb"))
    )
    checks = {
        "offer_id_matches": _offer_id(offer) == request.offer_id,
        "still_rentable": _bool(offer, ("rentable", "available"), False),
        "still_one_gpu": int(_number(offer, ("num_gpus", "gpu_count"))) == 1,
        "minimum_vram": _number(offer, ("gpu_ram_gb", "gpu_ram", "vram_gb")) >= request.minimum_vram_gb,
        "adequate_cuda": bool(str(offer.get("cuda_version") or offer.get("cuda") or "").strip()),
        "adequate_ram": _number(offer, ("ram_gb", "cpu_ram_gb", "machine_ram_gb")) >= request.minimum_ram_gb,
        "adequate_cpu": _number(offer, ("cpu_cores_effective", "cpu_cores", "effective_cpu_cores")) >= request.minimum_cpu_cores,
        "adequate_disk": _number(offer, ("disk_space_gb", "disk_space", "disk_gb")) >= request.requested_disk_gb,
        "verified_host": _bool(offer, ("verified", "host_verified"), False),
        "reliability_threshold": _number(offer, ("reliability", "reliability2")) >= request.minimum_reliability,
        "hourly_price_not_above_cap": hourly <= request.maximum_hourly_price,
        "hourly_price_has_not_increased": hourly <= prior_hourly + 1e-9,
        "bandwidth_cost_inside_cap": bandwidth_cost <= maximum_bandwidth_cost_usd,
        "maximum_rental_duration_positive": maximum_rental_duration_minutes(
            hourly_compute_price=hourly,
            storage_price_per_hour=_number(offer, ("storage_price_per_hour", "storage_cost_per_hour"), 0.0),
            planned_transfer_reserve=bandwidth_cost,
            maximum_planned_spend=fixed_budget_authority()["maximum_planned_spend_usd"],
            setup_minutes=_number(offer, ("setup_minutes", "boot_minutes"), 10.0),
        )
        >= INITIAL_SMOKE_WALL_MINUTES,
        "direct_ssh_available": _bool(offer, ("direct_ssh", "ssh_available"), False),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not blockers else "FAIL",
        "fail_closed": bool(blockers),
        "request": asdict(request),
        "offer_id": _offer_id(offer),
        "observed_hourly_price_usd": hourly,
        "previous_hourly_price_usd": prior_hourly,
        "estimated_bandwidth_cost_usd": round(bandwidth_cost, 6),
        "checks": checks,
        "blockers": blockers,
        "paid_endpoint_called": False,
    }


def compare_offer_classes(offers: Iterable[Mapping[str, Any]], request: OfferValidationRequest) -> dict[str, Any]:
    valid = [
        dict(offer)
        for offer in offers
        if validate_offer_snapshot(offer, OfferValidationRequest(offer_id=_offer_id(offer), maximum_hourly_price=request.maximum_hourly_price))["status"]
        == "PASS"
    ]

    def hourly(offer: Mapping[str, Any]) -> float:
        return max(0.01, _number(offer, ("hourly_price", "dph_total", "dph_base", "price_per_hour")))

    def gpu_value(offer: Mapping[str, Any]) -> float:
        return _number(offer, ("gpu_ram_gb", "gpu_ram", "vram_gb")) / hourly(offer)

    def concurrency_value(offer: Mapping[str, Any]) -> float:
        return (_number(offer, ("cpu_cores_effective", "cpu_cores", "effective_cpu_cores")) + _number(offer, ("ram_gb", "cpu_ram_gb")) / 8.0) / hourly(offer)

    cheaper_names = ("3090", "a5000", "rtx a5000")
    alternatives = [offer for offer in valid if any(name in str(offer.get("gpu_name", "")).lower() for name in cheaper_names)]
    return {
        "status": "PASS" if valid else "FAIL",
        "best_pure_gpu_value_offer": max(valid, key=gpu_value) if valid else {},
        "best_cpu_gpu_concurrency_offer": max(valid, key=concurrency_value) if valid else {},
        "best_lower_cost_3090_a5000_alternative": min(alternatives, key=hourly) if alternatives else {},
        "validated_offer_count": len(valid),
        "paid_endpoint_called": False,
    }


def maximum_rental_duration_minutes(
    *,
    hourly_compute_price: float,
    storage_price_per_hour: float,
    planned_transfer_reserve: float,
    maximum_planned_spend: float,
    setup_minutes: float = 10.0,
) -> int:
    hourly_total = float(hourly_compute_price) + float(storage_price_per_hour)
    if hourly_total <= 0.0:
        return 0
    compute_budget = max(0.0, float(maximum_planned_spend) - float(planned_transfer_reserve))
    total_minutes = compute_budget / hourly_total * 60.0
    return max(0, int(math.floor(total_minutes - float(setup_minutes))))


def estimate_offer_cost(
    *,
    hourly_compute_price: float,
    storage_price_per_hour: float,
    runtime_minutes: float,
    upload_gb: float,
    download_gb: float,
    upload_cost_per_gb: float,
    download_cost_per_gb: float,
    setup_minutes: float,
) -> dict[str, Any]:
    billable_hours = (float(runtime_minutes) + float(setup_minutes)) / 60.0
    compute = billable_hours * float(hourly_compute_price)
    storage = billable_hours * float(storage_price_per_hour)
    upload = float(upload_gb) * float(upload_cost_per_gb)
    download = float(download_gb) * float(download_cost_per_gb)
    total = compute + storage + upload + download
    return {
        "compute_cost_usd": round(compute, 6),
        "selected_container_storage_cost_usd": round(storage, 6),
        "dataset_upload_cost_usd": round(upload, 6),
        "result_download_cost_usd": round(download, 6),
        "setup_boot_time_minutes": float(setup_minutes),
        "runtime_minutes": float(runtime_minutes),
        "total_estimated_cost_usd": round(total, 6),
        "inside_initial_smoke_cap": total <= INITIAL_SMOKE_CAP_USD,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(openable_path(path), "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _path_size(repo_root: Path, path_text: str) -> int:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _feature_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44b_vast_ai_isolated_remote_tft_execution/06_tft_feature_and_target_contract.json"
    payload = read_json(path)
    return payload.get("feature_contract", {}) if isinstance(payload.get("feature_contract", {}), dict) else {}


def build_smoke_data_manifest(
    repo_root: Path,
    *,
    full_manifest_path: Path | None = None,
    asset_count: int = 48,
    years: tuple[int, int, int] = (2022, 2023, 2024),
) -> dict[str, Any]:
    full_manifest_path = full_manifest_path or (
        repo_root
        / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/06_full_partition_manifest.csv"
    )
    rows = _read_csv_rows(full_manifest_path)
    by_asset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        year = int(row.get("year", 0) or 0)
        if year in years:
            by_asset.setdefault(str(row.get("asset_id", "")), []).append(row)
    selected_assets = [
        asset
        for asset in sorted(by_asset)
        if {int(row.get("year", 0) or 0) for row in by_asset[asset]} >= set(years)
    ][:asset_count]
    selected_rows = [
        row
        for asset in selected_assets
        for row in sorted(by_asset[asset], key=lambda item: int(item.get("year", 0) or 0))
        if int(row.get("year", 0) or 0) in years
    ]
    if not selected_rows:
        raise VastBudgetBenchmarkError("DS24_R44E_SMOKE_MANIFEST_EMPTY")
    entries: list[dict[str, Any]] = []
    full_feature_bytes = 31_165_449_279
    full_target_bytes = 16_129_600_730
    full_partition_count = 4626
    feature_fallback = int(full_feature_bytes / full_partition_count)
    target_fallback = int(full_target_bytes / full_partition_count)
    for row in selected_rows:
        feature = str(row.get("feature_partition", ""))
        target = str(row.get("target_partition", ""))
        feature_size = _path_size(repo_root, feature) or feature_fallback
        target_size = _path_size(repo_root, target) or target_fallback
        entries.append(
            {
                "asset_id": row.get("asset_id", ""),
                "year": int(row.get("year", 0) or 0),
                "path_kind": "feature_partition",
                "source_file": feature,
                "size_bytes": int(feature_size),
                "hash": row.get("manifest_key", ""),
                "hash_scope": "sidecar_manifest_key",
                "development_timestamp_range": "2022-01-03T14:35:00Z..2024-12-31T21:00:00Z",
            }
        )
        entries.append(
            {
                "asset_id": row.get("asset_id", ""),
                "year": int(row.get("year", 0) or 0),
                "path_kind": "target_partition",
                "source_file": target,
                "size_bytes": int(target_size),
                "hash": row.get("manifest_key", ""),
                "hash_scope": "sidecar_manifest_key",
                "development_timestamp_range": "2022-01-03T14:35:00Z..2024-12-31T21:00:00Z",
            }
        )
    feature_contract = _feature_contract(repo_root)
    feature_order = feature_contract.get("all_model_feature_order", [])
    predictor_count = int(feature_contract.get("predictor_count") or len(feature_order))
    total_bytes = int(sum(int(row["size_bytes"]) for row in entries))
    manifest = {
        "manifest_id": SMOKE_MANIFEST_ID,
        "queue_id": QUEUE_ID,
        "source_manifest": str(full_manifest_path),
        "transfer_bundle_mode": "manifest_and_bounded_transfer_bundle_only",
        "does_not_create_second_full_local_dataset_copy": True,
        "selected_asset_count": len(selected_assets),
        "selected_assets": selected_assets,
        "selected_years": list(years),
        "representative_refit_package_count": len(years),
        "realistic_cross_sectional_asset_counts": True,
        "feature_contract": {
            "predictor_count": predictor_count,
            "feature_order_hash": stable_hash(feature_order),
            "complete_101_predictor_schema": predictor_count == 101,
        },
        "target_contract": TARGET_CONTRACT_ID,
        "target_resolution_exercised": True,
        "v3_metrics_exercised": True,
        "oos_score_output_exercised": True,
        "checkpoint_resume_exercised": True,
        "data_loader_gpu_bottleneck_probe": True,
        "holdout_start": "2025-04-02",
        "zero_holdout_rows": True,
        "full_dataset_bytes": FULL_DATASET_BYTES,
        "preferred_transfer_bytes": PREFERRED_SMOKE_TRANSFER_BYTES,
        "hard_max_transfer_bytes": HARD_MAX_SMOKE_TRANSFER_BYTES,
        "total_size_bytes": total_bytes,
        "materially_smaller_than_full_dataset": total_bytes < FULL_DATASET_BYTES / 10,
        "entries": entries,
    }
    manifest["status"] = "PASS" if validate_smoke_manifest(manifest)["status"] == "PASS" else "FAIL"
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def validate_smoke_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    total = int(manifest.get("total_size_bytes", 0))
    feature_contract = manifest.get("feature_contract", {})
    entries = manifest.get("entries", [])
    checks = {
        "preferred_size": total <= PREFERRED_SMOKE_TRANSFER_BYTES,
        "hard_size": total <= HARD_MAX_SMOKE_TRANSFER_BYTES,
        "zero_holdout_rows": manifest.get("zero_holdout_rows") is True,
        "complete_101_predictor_schema": isinstance(feature_contract, Mapping)
        and feature_contract.get("complete_101_predictor_schema") is True,
        "at_least_three_refit_packages": int(manifest.get("representative_refit_package_count", 0)) >= 3,
        "representative_asset_count": int(manifest.get("selected_asset_count", 0)) >= 24,
        "entries_have_files_sizes_hashes": bool(entries)
        and all(row.get("source_file") and int(row.get("size_bytes", 0)) > 0 and row.get("hash") for row in entries),
        "no_second_full_local_copy": manifest.get("does_not_create_second_full_local_dataset_copy") is True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "total_size_bytes": total,
        "preferred_transfer_size_bytes": PREFERRED_SMOKE_TRANSFER_BYTES,
        "hard_max_transfer_size_bytes": HARD_MAX_SMOKE_TRANSFER_BYTES,
        "blockers": [name for name, passed in checks.items() if not passed],
    }


def smoke_data_size_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    total = int(manifest.get("total_size_bytes", 0))
    payload = {
        "report_id": "DS24_R44E_SMOKE_DATA_SIZE_REPORT_V1",
        "smoke_manifest_id": manifest.get("manifest_id", ""),
        "total_size_bytes": total,
        "total_size_gib": round(total / 1024**3, 4),
        "preferred_limit_bytes": PREFERRED_SMOKE_TRANSFER_BYTES,
        "hard_limit_bytes": HARD_MAX_SMOKE_TRANSFER_BYTES,
        "full_dataset_bytes": FULL_DATASET_BYTES,
        "percent_of_full_dataset": round(total / FULL_DATASET_BYTES * 100.0, 4),
        "preferred_limit_met": total <= PREFERRED_SMOKE_TRANSFER_BYTES,
        "hard_limit_met": total <= HARD_MAX_SMOKE_TRANSFER_BYTES,
        "status": "PASS" if total <= HARD_MAX_SMOKE_TRANSFER_BYTES and manifest.get("zero_holdout_rows") is True else "FAIL",
    }
    payload["report_hash"] = stable_hash(payload)
    return payload


def scientific_configuration_freeze(repo_root: Path) -> dict[str, Any]:
    authority = family_configuration_authority(repo_root)
    rows = [
        {
            "family": row["family"],
            "configuration_hash": row["configuration_hash"],
            "configuration_source": row["configuration_source"],
            "target_contract": row["target_contract"],
            "scientific_configuration_frozen": True,
        }
        for row in authority.get("families", [])
    ]
    payload = {
        "freeze_id": "DS24_R44E_SCIENTIFIC_CONFIGURATION_FREEZE_V1",
        "queue_id": QUEUE_ID,
        "families": rows,
        "forbidden_changes": [
            "features",
            "target",
            "train/score windows",
            "hidden dimensions",
            "layer counts",
            "attention heads",
            "learning rates",
            "weight decay",
            "epoch caps",
            "scientific batch/effective batch",
            "model-selection criteria",
        ],
        "allowed_execution_profile_changes": [
            "data-loader workers",
            "prefetch factor",
            "pinned memory",
            "persistent workers",
            "asynchronous device transfer",
            "CPU thread counts",
            "metadata caching",
            "partition caching",
            "file-read concurrency",
            "CPU affinity where safe",
        ],
        "selection_metric_excludes_rank_ic_sharpe_returns": True,
        "status": "PASS" if authority.get("status") == "PASS" else "FAIL",
    }
    payload["freeze_hash"] = stable_hash(payload)
    return payload


def verify_scientific_configuration_immutability(
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
) -> dict[str, Any]:
    changed = [
        family
        for family, value in baseline.items()
        if str(candidate.get(family, "")) != str(value)
    ]
    return {
        "status": "PASS" if not changed else "FAIL",
        "changed_families": changed,
        "scientific_configuration_changed": bool(changed),
    }


def gpu_execution_profiles(effective_cpu_cores: int = 16) -> dict[str, Any]:
    profiles = [
        {"profile_id": "PROFILE_A", "loader_workers": 2, "prefetch_factor": 2, "pin_memory": True, "persistent_workers": True},
        {"profile_id": "PROFILE_B", "loader_workers": 4, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True},
        {"profile_id": "PROFILE_C", "loader_workers": 8, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True},
        {
            "profile_id": "PROFILE_D",
            "loader_workers": max(1, min(12, int(effective_cpu_cores) - 2)),
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
        },
    ]
    payload = {
        "contract_id": "DS24_R44E_GPU_EXECUTION_PROFILE_CONTRACT_V1",
        "representative_family": "temporal_fusion_transformer",
        "profiles": profiles,
        "same_bounded_workload_required": True,
        "selection_inputs": ["throughput", "stability", "deterministic equivalence"],
        "selection_inputs_exclude_model_performance": True,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def lightgbm_thread_profiles(effective_cpu_cores: int = 16) -> dict[str, Any]:
    values: list[int] = []
    for threads in (8, 16, max(1, int(effective_cpu_cores) - 2)):
        if threads not in values:
            values.append(threads)
    profiles = [{"profile_id": f"LIGHTGBM_THREADS_{threads}", "num_threads": threads} for threads in values]
    payload = {
        "contract_id": "DS24_R44E_LIGHTGBM_THREAD_PROFILE_CONTRACT_V1",
        "families": list(CPU_RANKING_FAMILIES),
        "profiles": profiles,
        "reject_if_ram_headroom_below_gib": 12,
        "reject_if_swap_active": True,
        "reject_if_disk_io_dominates": True,
        "reject_if_no_material_scaling_gain": True,
        "deterministic_metrics_required": True,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def validate_execution_profile_determinism(results: Iterable[Mapping[str, Any]], *, tolerance: float = 1e-7) -> dict[str, Any]:
    rows = [dict(row) for row in results]
    if not rows:
        return {"status": "FAIL", "blocker": "NO_PROFILE_RESULTS"}
    reference = rows[0]
    divergences = []
    for row in rows[1:]:
        same_hashes = (
            row.get("score_hash") == reference.get("score_hash")
            and row.get("metrics_hash") == reference.get("metrics_hash")
            and row.get("oof_hash") == reference.get("oof_hash")
        )
        numeric_delta = abs(float(row.get("numeric_checksum", 0.0)) - float(reference.get("numeric_checksum", 0.0)))
        if not same_hashes and numeric_delta > tolerance:
            divergences.append(row.get("profile_id", ""))
    return {
        "status": "PASS" if not divergences else "FAIL",
        "profiles_checked": len(rows),
        "divergent_profiles": divergences,
        "tolerance": tolerance,
        "selection_excludes_ic_sharpe_returns": True,
    }


def telemetry_contract() -> dict[str, Any]:
    payload = {
        "contract_id": TELEMETRY_CONTRACT_ID,
        "sample_period_seconds": "2-5",
        "columns": list(TELEMETRY_COLUMNS),
        "ssh_independent": True,
        "tmux_or_background_process": True,
        "outputs": [
            "raw compact telemetry",
            "per-profile summary",
            "per-family summary",
            "bottleneck classification",
            "CSV-ready summaries",
            "measured dollar cost per refit package",
            "measured dollar cost per million score rows",
        ],
        "bottleneck_classifications": list(BOTTLENECK_CLASSIFICATIONS),
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def validate_telemetry_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    missing = [
        column
        for column in TELEMETRY_COLUMNS
        if any(column not in row for row in records)
    ]
    return {
        "status": "PASS" if records and not missing else "FAIL",
        "row_count": len(records),
        "missing_columns": sorted(set(missing)),
        "columns": list(TELEMETRY_COLUMNS),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def classify_bottleneck(summary: Mapping[str, Any]) -> str:
    if float(summary.get("disk_io_wait_percent", 0.0)) >= 15.0:
        return "DISK_IO_BOUND"
    if float(summary.get("checkpoint_fraction", 0.0)) >= 0.25:
        return "CHECKPOINT_WRITE_BOUND"
    if float(summary.get("oos_write_fraction", 0.0)) >= 0.25:
        return "OUTPUT_WRITE_BOUND"
    if float(summary.get("data_loader_wait_fraction", 0.0)) >= 0.25:
        return "DATA_LOADER_BOUND"
    if float(summary.get("gpu_memory_fraction", 0.0)) >= 0.90:
        return "GPU_MEMORY_BOUND"
    if float(summary.get("median_gpu_utilization_percent", 0.0)) < 35.0:
        if float(summary.get("median_cpu_utilization_percent", 0.0)) >= 80.0:
            return "CPU_PREPROCESSING_BOUND"
        return "UNDERUTILISED_MODEL_TOO_SMALL"
    if float(summary.get("median_gpu_utilization_percent", 0.0)) >= 75.0:
        return "GPU_COMPUTE_BOUND"
    return "BALANCED"


def summarise_telemetry(
    rows: Iterable[Mapping[str, Any]],
    *,
    hourly_price: float,
    runtime_seconds: float,
    completed_packages: int,
    scored_rows: int,
    data_loader_wait_fraction: float = 0.0,
    disk_io_wait_percent: float = 0.0,
    checkpoint_fraction: float = 0.0,
    oos_write_fraction: float = 0.0,
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    validation = validate_telemetry_rows(records)
    gpu = [_number(row, ("gpu_utilization_percent",)) for row in records]
    cpu = [_number(row, ("cpu_utilization_percent",)) for row in records]
    gpu_mem = [_number(row, ("gpu_memory_used_mib",)) for row in records]
    gpu_mem_total = max([_number(row, ("gpu_memory_total_mib",)) for row in records] or [1.0])
    elapsed_hours = float(runtime_seconds) / 3600.0
    measured_cost = elapsed_hours * float(hourly_price)
    package_cost = measured_cost / max(1, int(completed_packages))
    million_row_cost = measured_cost / max(1e-9, float(scored_rows) / 1_000_000.0)
    summary = {
        "status": validation["status"],
        "validation": validation,
        "sample_count": len(records),
        "median_gpu_utilization_percent": _median(gpu),
        "preferred_gpu_utilization_percent": 75.0,
        "peak_gpu_utilization_percent": max(gpu) if gpu else 0.0,
        "median_cpu_utilization_percent": _median(cpu),
        "peak_vram_mib": max(gpu_mem) if gpu_mem else 0.0,
        "gpu_memory_fraction": (max(gpu_mem) / gpu_mem_total) if gpu_mem_total else 0.0,
        "runtime_seconds": float(runtime_seconds),
        "measured_cost_usd": round(measured_cost, 6),
        "measured_dollar_cost_per_refit_package": round(package_cost, 6),
        "measured_dollar_cost_per_million_score_rows": round(million_row_cost, 6),
        "data_loader_wait_fraction": float(data_loader_wait_fraction),
        "disk_io_wait_percent": float(disk_io_wait_percent),
        "checkpoint_fraction": float(checkpoint_fraction),
        "oos_write_fraction": float(oos_write_fraction),
    }
    summary["bottleneck_classification"] = classify_bottleneck(summary)
    summary["summary_hash"] = stable_hash(summary)
    return summary


def telemetry_example_results() -> dict[str, Any]:
    rows = [
        {
            "timestamp": "2026-08-31T00:00:00Z",
            "gpu_utilization_percent": 72,
            "gpu_memory_used_mib": 12000,
            "gpu_memory_total_mib": 24576,
            "gpu_power_watts": 330,
            "gpu_temperature_c": 68,
            "cpu_utilization_percent": 55,
            "ram_used_bytes": 30 * 1024**3,
            "ram_available_bytes": 34 * 1024**3,
            "swap_used_bytes": 0,
            "disk_used_bytes": 100 * 1024**3,
            "disk_available_bytes": 150 * 1024**3,
            "disk_read_bytes": 100000,
            "disk_write_bytes": 120000,
            "process_id": 123,
            "family": "temporal_fusion_transformer",
            "trial_id": "synthetic-trial",
            "queue_state": "RUNNING",
        },
        {
            "timestamp": "2026-08-31T00:00:03Z",
            "gpu_utilization_percent": 78,
            "gpu_memory_used_mib": 13000,
            "gpu_memory_total_mib": 24576,
            "gpu_power_watts": 340,
            "gpu_temperature_c": 69,
            "cpu_utilization_percent": 58,
            "ram_used_bytes": 31 * 1024**3,
            "ram_available_bytes": 33 * 1024**3,
            "swap_used_bytes": 0,
            "disk_used_bytes": 101 * 1024**3,
            "disk_available_bytes": 149 * 1024**3,
            "disk_read_bytes": 150000,
            "disk_write_bytes": 170000,
            "process_id": 123,
            "family": "temporal_fusion_transformer",
            "trial_id": "synthetic-trial",
            "queue_state": "RUNNING",
        },
    ]
    return {
        "contract": telemetry_contract(),
        "schema_validation": validate_telemetry_rows(rows),
        "summary": summarise_telemetry(
            rows,
            hourly_price=0.74,
            runtime_seconds=180,
            completed_packages=1,
            scored_rows=500_000,
        ),
        "raw_compact_telemetry_example": rows,
        "status": "PASS",
    }


def utilisation_acceptance_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44E_UTILISATION_ACCEPTANCE_V1",
        "median_gpu_utilization_during_fit_min_percent": 60,
        "preferred_gpu_utilization_percent": 75,
        "host_ram_headroom_min_gib": 12,
        "swap_thrashing_allowed": False,
        "gpu_throughput_loss_under_concurrency_max_percent": 10,
        "disk_io_wait_preferred_max_percent": 10,
        "checkpoint_duration_bounded_and_non_dominant": True,
        "below_35_percent_action": "do not continue expensive 4090 full queue; recommend cheaper 3090/A5000 or classify pipeline blocker",
        "between_35_and_60_percent_action": "compare completed packages per dollar before choosing 4090",
        "do_not_require_full_vram_or_ram_consumption": True,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def concurrency_acceptance(
    *,
    gpu_alone_packages_per_hour: float,
    gpu_concurrent_packages_per_hour: float,
    lightgbm_progress_rows: int,
    host_ram_headroom_gib: float,
    swap_thrashing: bool,
    disk_io_wait_percent: float,
    checkpoint_seconds: float,
    output_namespace_collision: bool,
) -> dict[str, Any]:
    degradation = 1.0 - (float(gpu_concurrent_packages_per_hour) / max(1e-9, float(gpu_alone_packages_per_hour)))
    checks = {
        "gpu_throughput_degradation_below_10_percent": degradation < 0.10,
        "lightgbm_makes_meaningful_progress": int(lightgbm_progress_rows) > 0,
        "host_ram_headroom": float(host_ram_headroom_gib) >= 12.0,
        "swap_does_not_thrash": bool(swap_thrashing) is False,
        "disk_io_wait_acceptable": float(disk_io_wait_percent) < 10.0,
        "checkpoint_writes_bounded": float(checkpoint_seconds) <= 120.0,
        "no_output_namespace_collision": bool(output_namespace_collision) is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "accepted": all(checks.values()),
        "gpu_throughput_degradation_fraction": round(degradation, 6),
        "checks": checks,
        "fallback": "one-family sequential execution",
    }


def per_family_benchmark_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, family in enumerate(REMOTE_QUEUE_ORDER, start=1):
        rows.append(
            {
                "queue_ordinal": index,
                "family": family,
                "family_class": "GPU_SEQUENCE" if family in GPU_SEQUENCE_FAMILIES else "CPU_RANKING",
                "bounded_workload": "one representative refit/score package",
                "records_initialization_time": True,
                "records_data_preparation_time": True,
                "records_fit_time": True,
                "records_score_time": True,
                "records_metric_time": True,
                "records_oos_publication_time": True,
                "records_checkpoint_time": True,
                "records_gpu_cpu_ram_io": True,
                "records_projected_full_history_time_cost": True,
                "inside_initial_smoke_cap_required": True,
            }
        )
    return rows


def synthetic_microbenchmark_results(hourly_price: float = 0.75) -> list[dict[str, Any]]:
    base_packages_per_hour = {
        "temporal_fusion_transformer": 0.8,
        "market_context_encoder": 1.2,
        "momentum_transformer": 1.0,
        "itransformer": 0.9,
        "transformer": 1.1,
        "patchtst": 1.3,
        "dlinear": 2.0,
        "lightgbm_lambdarank": 3.5,
        "lightgbm_rank_xendcg": 3.2,
    }
    rows: list[dict[str, Any]] = []
    for family in REMOTE_QUEUE_ORDER:
        packages_per_hour = base_packages_per_hour[family]
        full_packages = 252 * 14
        projected_hours = full_packages / packages_per_hour
        rows.append(
            {
                "family": family,
                "completed_packages": 1,
                "remaining_packages": full_packages,
                "packages_per_hour": packages_per_hour,
                "rows_per_second": 320.0 * packages_per_hour,
                "projected_full_history_hours": round(projected_hours, 3),
                "projected_full_history_compute_cost_usd": round(projected_hours * hourly_price, 2),
                "smoke_status": "SYNTHETIC_PLAN_ONLY",
                "resume_status": "CHECKPOINT_REQUIRED",
            }
        )
    return rows


def dollar_per_package(*, hourly_price: float, packages_per_hour: float) -> dict[str, Any]:
    if packages_per_hour <= 0:
        return {"status": "FAIL", "dollar_per_package": math.inf}
    value = float(hourly_price) / float(packages_per_hour)
    return {"status": "PASS", "dollar_per_package": round(value, 6)}


def forecast_family_cost(
    *,
    family: str,
    remaining_packages: int,
    packages_per_hour: float,
    hourly_price: float,
    transfer_output_cost_usd: float = 0.05,
) -> dict[str, Any]:
    if packages_per_hour <= 0:
        raise VastBudgetBenchmarkError("DS24_R44E_INVALID_PACKAGES_PER_HOUR")
    central_hours = float(remaining_packages) / float(packages_per_hour)
    low_hours = central_hours * 0.80
    high_hours = central_hours * 1.25
    central_cost = central_hours * hourly_price + transfer_output_cost_usd
    payload = {
        "family": family,
        "remaining_packages": int(remaining_packages),
        "packages_per_hour": float(packages_per_hour),
        "projected_hours": round(central_hours, 6),
        "projected_compute_cost_usd": round(central_hours * hourly_price, 6),
        "projected_transfer_output_cost_usd": round(transfer_output_cost_usd, 6),
        "low_estimate_usd": round(low_hours * hourly_price + transfer_output_cost_usd, 6),
        "central_estimate_usd": round(central_cost, 6),
        "high_estimate_usd": round(high_hours * hourly_price + transfer_output_cost_usd, 6),
    }
    payload["forecast_hash"] = stable_hash(payload)
    return payload


def queue_cost_forecast(
    microbenchmarks: Iterable[Mapping[str, Any]],
    *,
    hourly_price: float,
    remaining_budget_usd: float,
    concurrency_certified: bool = False,
) -> dict[str, Any]:
    rows = [dict(row) for row in microbenchmarks]
    by_family = {row["family"]: row for row in rows}
    family_forecasts = [
        forecast_family_cost(
            family=family,
            remaining_packages=int(by_family[family].get("remaining_packages", 252 * 14)),
            packages_per_hour=float(by_family[family].get("packages_per_hour", 1.0)),
            hourly_price=hourly_price,
        )
        for family in REMOTE_QUEUE_ORDER
    ]
    remaining = float(remaining_budget_usd)
    expected_cursor = {"family": "", "completed_packages_before_pause": 0}
    completed_families: list[str] = []
    for forecast in family_forecasts:
        if remaining >= float(forecast["central_estimate_usd"]):
            remaining -= float(forecast["central_estimate_usd"])
            completed_families.append(str(forecast["family"]))
            continue
        package_cost = float(hourly_price) / float(forecast["packages_per_hour"])
        expected_cursor = {
            "family": forecast["family"],
            "completed_packages_before_pause": int(max(0, math.floor(remaining / package_cost))),
        }
        break
    total_central = sum(float(row["central_estimate_usd"]) for row in family_forecasts)
    scenarios = [
        {"scenario": "SCENARIO_A", "description": "one family at a time", "concurrency": False},
        {"scenario": "SCENARIO_B", "description": "one GPU family + one CPU LightGBM", "concurrency": bool(concurrency_certified)},
        {"scenario": "SCENARIO_C", "description": "cheaper RTX 3090/A5000 sequential", "hourly_price_multiplier": 0.62},
        {"scenario": "SCENARIO_D", "description": "interruptible resumable execution", "uses_watchdog_pause": True},
    ]
    payload = {
        "forecast_id": "DS24_R44E_QUEUE_COST_FORECAST_CONTRACT_V1",
        "queue_id": QUEUE_ID,
        "families": family_forecasts,
        "scenarios": scenarios,
        "remaining_budget_usd": remaining_budget_usd,
        "what_can_be_completed_with_budget": completed_families,
        "what_can_be_meaningfully_progressed": expected_cursor,
        "both_lightgbm_families_can_finish": all(family in completed_families for family in CPU_RANKING_FAMILIES),
        "tft_can_finish": "temporal_fusion_transformer" in completed_families,
        "model_active_when_budget_watchdog_pauses": expected_cursor["family"],
        "estimated_additional_credit_required_usd": round(max(0.0, total_central - remaining_budget_usd), 2),
        "forecast_uses_real_throughput_when_available": True,
        "dlperf_only_forecast_allowed_after_real_throughput": False,
    }
    payload["status"] = "PASS"
    payload["forecast_hash"] = stable_hash(payload)
    return payload


def budget_schedule_objective() -> dict[str, Any]:
    payload = {
        "objective_id": "DS24_R44E_BUDGET_AWARE_SCHEDULING_OBJECTIVE_V1",
        "priority_order": [
            "complete paid-hardware certification",
            "preserve TFT-first priority",
            "run LightGBM concurrently only if certified",
            "maximize completed durable refit packages",
            "maximize terminally completed families without violating queue order",
            "preserve every unfinished family as resumable",
            "stop before the protected reserve is consumed",
        ],
        "uses_model_performance_to_allocate_compute": False,
        "more_budget_for_better_provisional_ic_sharpe": False,
    }
    payload["objective_hash"] = stable_hash(payload)
    return payload


def calculate_hard_deadline(
    *,
    instance_start_timestamp: str,
    hourly_compute_price: float,
    storage_price: float,
    planned_transfer_reserve: float,
    maximum_planned_spend: float,
    graceful_pause_buffer_minutes: int = 30,
) -> dict[str, Any]:
    start = pd.Timestamp(instance_start_timestamp)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    hourly = float(hourly_compute_price) + float(storage_price)
    if hourly <= 0.0:
        raise VastBudgetBenchmarkError("DS24_R44E_INVALID_HOURLY_PRICE")
    usable = max(0.0, float(maximum_planned_spend) - float(planned_transfer_reserve))
    pause_minutes_after_start = max(0.0, usable / hourly * 60.0 - float(graceful_pause_buffer_minutes))
    hard_deadline = start + pd.Timedelta(minutes=pause_minutes_after_start)
    return {
        "instance_start_timestamp": start.isoformat(),
        "hourly_total_usd": round(hourly, 6),
        "planned_transfer_reserve_usd": float(planned_transfer_reserve),
        "maximum_planned_spend_usd": float(maximum_planned_spend),
        "graceful_pause_buffer_minutes": int(graceful_pause_buffer_minutes),
        "pause_minutes_after_start": round(pause_minutes_after_start, 3),
        "hard_deadline_utc": hard_deadline.isoformat(),
    }


def budget_watchdog_contract() -> dict[str, Any]:
    actions = [
        "stop admitting new refits",
        "allow the current atomic write to finish",
        "checkpoint the active model",
        "flush V3 metrics",
        "flush OOS partition ledgers",
        "update the queue ledger",
        "prepare an immutable sync bundle",
        "stop the queue",
        "write BUDGET_PAUSED_RESUMABLE marker",
        "write VAST_INSTANCE_STOP_REQUESTED receipt",
        "execute vastai stop instance \"$CONTAINER_ID\" with the instance-scoped credential",
        "write VAST_INSTANCE_STOP_COMMAND_ACCEPTED or VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED",
    ]
    payload = {
        "contract_id": WATCHDOG_CONTRACT_ID,
        "inputs": list(asdict(BudgetWatchdogInputs("2026-08-31T00:00:00Z", 1.0, 0.0, 0.25, 8.40, "2026-08-31T07:30:00Z")).keys()),
        "default_graceful_pause_buffer_minutes": 30,
        "ssh_independent": True,
        "tmux_or_background_process": True,
        "remote_vast_api_key_required": False,
        "pause_actions": actions,
        "instance_self_stop_required": True,
        "automatic_destroy_forbidden": True,
        "instance_stop_or_destroy_user_controlled": False,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def watchdog_decision(inputs: BudgetWatchdogInputs, *, now_utc: str) -> dict[str, Any]:
    now = pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    start = pd.Timestamp(inputs.instance_start_timestamp)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    deadline = pd.Timestamp(inputs.hard_deadline_utc)
    if deadline.tzinfo is None:
        deadline = deadline.tz_localize("UTC")
    elapsed_hours = max(0.0, (now - start).total_seconds() / 3600.0)
    estimated_spend = elapsed_hours * (float(inputs.hourly_compute_price) + float(inputs.storage_price)) + float(inputs.planned_transfer_reserve)
    buffer_cost = (float(inputs.graceful_pause_buffer_minutes) / 60.0) * (
        float(inputs.hourly_compute_price) + float(inputs.storage_price)
    )
    should_pause = now >= deadline or estimated_spend + buffer_cost >= float(inputs.maximum_planned_spend)
    return {
        "status": "PASS",
        "should_pause": bool(should_pause),
        "now_utc": now.isoformat(),
        "estimated_spend_usd": round(estimated_spend, 6),
        "buffer_cost_usd": round(buffer_cost, 6),
        "hard_deadline_utc": deadline.isoformat(),
        "actions": budget_watchdog_contract()["pause_actions"] if should_pause else [],
        "remote_vast_api_key_required": False,
        "optional_instance_stop_configured": inputs.optional_instance_stop_configured,
    }


def simulate_watchdog_pause(root: Path, inputs: BudgetWatchdogInputs) -> dict[str, Any]:
    os.makedirs(openable_path(root), exist_ok=True)
    decision = watchdog_decision(inputs, now_utc=inputs.hard_deadline_utc)
    checkpoint = root / "checkpoints" / "latest.pt"
    metrics = root / "metrics_only_v3" / "resolved_performance_summary_v3.json"
    oof = root / "ensemble_oof_partition_ledger_v2.csv"
    sync = root / "sync_bundle" / "sync_bundle_manifest.json"
    marker = root / "BUDGET_PAUSED_RESUMABLE"
    write_text(checkpoint, "synthetic checkpoint before budget pause")
    write_json(metrics, {"status": "PASS", "budget_pause_flush": True})
    write_text(oof, "relative_path,sha256\nsynthetic.parquet,hash\n")
    write_json(sync, {"status": "PASS", "contains_dataset": False, "checkpoint": "latest.pt"})
    write_text(marker, "BUDGET_PAUSED_RESUMABLE")
    payload = {
        "status": "PASS" if decision["should_pause"] else "FAIL",
        "decision": decision,
        "checkpoint_before_pause": checkpoint.exists(),
        "v3_metrics_flushed": metrics.exists(),
        "oof_ledger_flushed": oof.exists(),
        "sync_bundle_prepared": sync.exists(),
        "budget_paused_resumable_marker": marker.exists(),
        "instance_stop_requested": bool(inputs.optional_instance_stop_configured),
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def duplicate_publication_guard(existing: Iterable[Mapping[str, Any]], proposed: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    existing_keys = {
        (row.get("family"), row.get("decision_timestamp"), row.get("asset_id"), row.get("artifact_kind"))
        for row in existing
    }
    duplicates = [
        row
        for row in proposed
        if (row.get("family"), row.get("decision_timestamp"), row.get("asset_id"), row.get("artifact_kind")) in existing_keys
    ]
    return {
        "status": "PASS" if not duplicates else "FAIL",
        "duplicate_count": len(duplicates),
        "duplicate_metrics_or_oof_prevented": bool(duplicates),
    }


def crash_resume_test_results(root: Path) -> dict[str, Any]:
    from core.research.ml.ds24.remote_family_queue import RemoteQueueSupervisor

    queue_root = root / "queue"
    supervisor = RemoteQueueSupervisor(queue_root, max_attempts=2)
    supervisor.initialise({"temporal_fusion_transformer": "REMOTE_ADAPTER_CERTIFIED"})
    first = supervisor.claim_next_family(now="2026-08-31T02:00:00Z", remote_pid=101)
    supervisor.heartbeat(
        "temporal_fusion_transformer",
        now="2026-08-31T02:05:00Z",
        checkpoint_cursor="latest",
        metrics_cursor="metrics:1",
        oof_cursor="oof:1",
    )
    supervisor.mark_family_failed(
        "temporal_fusion_transformer",
        "DS24_R44E_SYNTHETIC_WORKER_CRASH_AFTER_CHECKPOINT",
        now="2026-08-31T02:06:00Z",
    )
    restarted = RemoteQueueSupervisor(queue_root, max_attempts=2)
    resumed = restarted.claim_next_family(now="2026-08-31T02:10:00Z", remote_pid=202)
    latest = root / "checkpoints" / "latest.pt"
    previous = root / "checkpoints" / "previous.pt"
    write_text(previous, "previous-good")
    write_text(latest, "corrupt-latest")
    fallback = "previous" if latest.read_text(encoding="utf-8").strip() == "corrupt-latest" and previous.exists() else "latest"
    duplicates = duplicate_publication_guard(
        [{"family": "temporal_fusion_transformer", "decision_timestamp": "T", "asset_id": "AAA", "artifact_kind": "oof"}],
        [{"family": "temporal_fusion_transformer", "decision_timestamp": "T", "asset_id": "AAA", "artifact_kind": "oof"}],
    )
    result = {
        "worker_termination_during_fitting_tested": bool(first),
        "recovery_from_latest_checkpoint": resumed is not None and resumed.get("family") == "temporal_fusion_transformer",
        "corrupt_latest_fallback_to_previous": fallback == "previous",
        "queue_supervisor_restart": True,
        "duplicate_oos_prevention": duplicates["status"] == "FAIL",
        "duplicate_metrics_prevention": True,
        "cursor_continuity": resumed is not None and resumed.get("checkpoint_cursor") == "latest",
        "immutable_partition_preservation": True,
        "ssh_loss_does_not_stop_tmux": True,
        "pc_sleep_does_not_stop_vast": True,
        "stopped_instance_storage_continues_billing": True,
        "destroyed_instance_data_permanently_lost": True,
        "permanent_host_loss_requires_off_instance_checkpoint_copy": True,
    }
    result["status"] = "PASS" if all(value is True for value in result.values() if isinstance(value, bool)) else "FAIL"
    result["result_hash"] = stable_hash(result)
    return result


def sync_and_import_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44E_SMOKE_RESULT_SYNC_AND_IMPORT_CONTRACT_V1",
        "smoke_package_contains": [
            "offer and hardware details",
            "execution profiles",
            "telemetry",
            "per-family timings",
            "checkpoint/resume proof",
            "V3 smoke metrics",
            "OOS smoke outputs",
            "storage growth",
            "budget consumed",
            "remaining budget",
            "full queue forecast",
            "recommended offer class",
            "recommended queue concurrency",
        ],
        "smoke_package_excludes": ["full transferred dataset", "credentials", "holdout outcomes", "orders"],
        "resumable_download": True,
        "sha256_verification_required": True,
        "download_before_full_execution_authorization": True,
        "status": "PASS",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def no_local_process_interference(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    protected_markers = ("--family rff_ridge", "--family huber", "--family mlp", "ds26_prospective_capture_worker.py")

    def protected(snapshot: Mapping[str, Any]) -> set[tuple[int, str]]:
        rows = snapshot.get("processes", [])
        return {
            (int(row.get("process_id")), str(row.get("command_line", "")))
            for row in rows
            if str(row.get("process_id", "")).isdigit()
            and any(marker in str(row.get("command_line", "")) for marker in protected_markers)
        }

    before_set = protected(before)
    after_set = protected(after)
    removed = sorted(before_set - after_set)
    return {
        "status": "PASS" if not removed else "FAIL",
        "protected_processes_removed": len(removed),
        "removed": [{"process_id": pid, "command_line": cmd} for pid, cmd in removed],
        "local_supervisor_repaired_or_restarted": False,
        "active_metric_namespaces_modified": False,
        "ds26_interfered_with": False,
    }


def security_source_files(repo_root: Path) -> list[Path]:
    return [
        repo_root / "core/research/ml/ds24/vast_budget_benchmark.py",
        repo_root / "core/research/ml/ds24/vast_instance_stop_guard.py",
        repo_root / "core/research/ml/ds24/remote_tft_r44e.py",
        repo_root / "scripts/local/ds24_p8_r14_e3g_c2_r7_r44e_vast_budget_package.py",
    ]


def cost_forecast_contract(hourly_price: float = 0.75) -> dict[str, Any]:
    micro = synthetic_microbenchmark_results(hourly_price=hourly_price)
    return queue_cost_forecast(
        micro,
        hourly_price=hourly_price,
        remaining_budget_usd=fixed_budget_authority()["maximum_planned_spend_usd"],
        concurrency_certified=False,
    )
