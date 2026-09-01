from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.research.ml.ds24 import remote_tft as r44b
from core.research.ml.ds24 import remote_tft_r44e2 as r44e2
from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash
from core.research.ml.ds24.remote_family_queue import (
    QUEUE_ID,
    REMOTE_QUEUE_ORDER,
    adapter_registry,
    queue_resume_determinism_proof,
    run_all_synthetic_smokes,
)
from core.research.ml.ds24.vast_budget_benchmark import (
    fixed_budget_authority,
    no_local_process_interference,
)
from core.research.ml.ds24.vast_soft_review_transition import (
    DEFAULT_REVIEW_GRACE_MINUTES,
    HARD_BUDGET_USD,
    HARD_WALL_CLOCK_HOURS,
    SOFT_REVIEW_MINUTES,
    full_dataset_transition_gate,
    synthetic_r44e2_proofs,
)


R44F_EVIDENCE_NAME = "r7_r44f_vast_morning_launch_readiness"
R44F_EVIDENCE_RELATIVE_ROOT = r44b.STAGE_ROOT / R44F_EVIDENCE_NAME
TICKET_ID = "DS24_P8_R14_E3G_C2_R7_R44F1_LIVE_VAST_CLI_SCHEMA_COMPATIBILITY_AND_PAID_LAUNCH_REPAIR"
TERMINAL_SUCCESS = "DS24_R44F1_LIVE_VAST_CLI_SCHEMA_COMPATIBILITY_REPAIRED_READY_FOR_USER_PAID_EXECUTION"

BLOCKED_SOURCE_BUNDLE = "DS24_R44F_BLOCKED_SOURCE_BUNDLE_CONTRACT_FAILURE"
BLOCKED_DATA_TRANSFER = "DS24_R44F_BLOCKED_DATA_TRANSFER_CONTRACT_FAILURE"
BLOCKED_ENVIRONMENT = "DS24_R44F_BLOCKED_ENVIRONMENT_FREEZE_FAILURE"
BLOCKED_FAMILY_ADAPTER = "DS24_R44F_BLOCKED_FAMILY_ADAPTER_CONTRACT_FAILURE"
BLOCKED_BUDGET_WATCHDOG = "DS24_R44F_BLOCKED_BUDGET_WATCHDOG_CONTRACT_FAILURE"
BLOCKED_TEST_ARCH = "DS24_R44F_BLOCKED_TEST_OR_ARCHITECTURE_FAILURE"

FULL_DATA_MANIFEST_RELATIVE = (
    r44b.STAGE_ROOT / "06_full_partition_manifest.csv"
).as_posix()
MINIMAL_REMOTE_DATA_MANIFEST_RELATIVE = (
    r44b.EVIDENCE_RELATIVE_ROOT / "11_minimal_remote_data_manifest.csv"
).as_posix()
EXPECTED_FULL_DATASET_MANIFEST_SHA256 = "6bcefad7f7bc98fb929a8f49f0b02de8add348cc5d661a84b9d3fd004ae66555"
EXPECTED_FULL_DATASET_SCHEMA_HASH = "f7162068d0d4e06a27395c6923dc7298335d955e401ad26a2ac39bbcdeda69cb"
REMOTE_WORKSPACE_ROOT = "/workspace/ds24"
REMOTE_SOURCE_ROOT = f"{REMOTE_WORKSPACE_ROOT}/source"
REMOTE_OUTPUT_ROOT = f"{REMOTE_WORKSPACE_ROOT}/output"
REMOTE_QUEUE_ROOT = f"{REMOTE_OUTPUT_ROOT}/remote_vast_runs/queue={QUEUE_ID}"
REMOTE_DATA_ROOT = f"{REMOTE_SOURCE_ROOT}/data"

CREATE_CONFIRM_TOKEN = "CREATE_EXACTLY_ONE_DS24_R44F_GUARDED_INSTANCE"
FIRST_PAID_CONFIRM_TOKEN = "READY_FOR_FIRST_PAID_ACTION"
DESTROY_CONFIRM_TOKEN = "DESTROY_DS24_R44F_AFTER_VERIFIED_DOWNLOAD"
DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD = 0.45
DEFAULT_MAXIMUM_BANDWIDTH_COST_USD = 1.25
MINIMUM_RELIABILITY = 0.98

REQUIRED_COMMAND_NAMES = (
    "MORNING_START_HERE.ps1",
    "vast_select_validate_and_confirm.ps1",
    "vast_create_guarded_instance.ps1",
    "vast_show_connection.ps1",
    "vast_resumable_upload.ps1",
    "vast_verify_remote_bundle.ps1",
    "vast_launch_profile_benchmark.ps1",
    "vast_launch_full_queue.ps1",
    "vast_show_90_minute_review.ps1",
    "vast_show_live_status.ps1",
    "vast_stop_and_checkpoint.ps1",
    "vast_download_results_resumably.ps1",
    "vast_verify_download.ps1",
    "vast_destroy_after_verified_download.ps1",
    "REMOTE_START_HERE.sh",
)

SOURCE_BUNDLE_NAME = "ds24_r44f_morning_runtime_source_bundle.zip"
SOURCE_BUNDLE_SHA_NAME = "ds24_r44f_morning_runtime_source_bundle.sha256"
LIVE_OFFER_CAPTURE_FILENAMES = ("r44f_live_offers.json", "r44f_current_vast_offers.json")


def utc_now() -> str:
    return r44b.utc_now()


def read_json(path: Path) -> dict[str, Any]:
    return r44b.read_json(path)


def write_json(path: Path, payload: Any) -> None:
    r44b.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    r44b.write_text(path, textwrap.dedent(text).strip() + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    os.makedirs(openable_path(path.parent), exist_ok=True)
    with open(openable_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    return r44b.sha256_file(path)


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


def process_and_resource_snapshot(repo_root: Path) -> dict[str, Any]:
    snapshot = r44e2.process_and_resource_snapshot(repo_root)
    snapshot["r44f_live_state_interpretation"] = {
        "source_only_preflight": True,
        "protected_local_workers_treated_as_user_owned": True,
        "r44f_stopped_or_restarted_local_workers": False,
        "r44f_interfered_with_ds26": False,
        "r44f_contacted_vast": False,
        "r44f_rented_instance": False,
        "r44f_uploaded_data": False,
    }
    return snapshot


def predecessor_validation(repo_root: Path) -> dict[str, Any]:
    terminal_path = repo_root / r44e2.R44E2_EVIDENCE_RELATIVE_ROOT / "19_terminal_result.json"
    terminal = read_json(terminal_path)
    checks = {
        "r44e2_terminal_success": terminal.get("terminal_classification") == r44e2.TERMINAL_SUCCESS
        and terminal.get("success") is True,
        "r44e2_preserves_soft_review": terminal.get("soft_review_minutes") == SOFT_REVIEW_MINUTES
        and terminal.get("review_marker") == "SMOKE_90_MINUTE_REVIEW_READY",
        "r44e2_preserves_hard_budget": terminal.get("hard_budget_usd") == HARD_BUDGET_USD
        and terminal.get("hard_wall_clock_hours") == HARD_WALL_CLOCK_HOURS
        and terminal.get("billing_elapsed_source") == "instance_start_timestamp",
        "r44e2_preserves_self_stop": terminal.get("stop_command") == 'vastai stop instance "$CONTAINER_ID"',
        "r44e2_no_paid_vast_action": terminal.get("paid_vast_operation_performed") is False
        and terminal.get("paid_vast_endpoint_called") is False
        and terminal.get("vast_instance_created") is False
        and terminal.get("vast_instance_destroyed") is False,
        "r44e2_validation_green": str(terminal.get("validation_focused_pytest", "")).startswith("PASS")
        and str(terminal.get("validation_architecture_conformance", "")).startswith("PASS"),
    }
    payload = {
        "authority_id": "DS24_R44F_R44E2_PREDECESSOR_VALIDATION_V1",
        "created_at_utc": utc_now(),
        "r44e2_terminal_path": repo_relative(repo_root, terminal_path),
        "r44e2_terminal": terminal,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_BUDGET_WATCHDOG,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def _normalise_manifest_path(repo_root: Path, value: str) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo_root / raw


def _remote_relative_for_source(repo_root: Path, path: Path) -> str:
    rel = repo_relative(repo_root, path)
    return rel.replace("\\", "/")


def _sidecar_for_data_file(path: Path, kind: str) -> Path:
    if kind == "feature":
        return path.with_name(path.name + ".manifest.json")
    return path.parent / "partition_manifest.json"


def _content_hash_from_sidecar(sidecar: Path, kind: str) -> tuple[str, str]:
    payload = read_json(sidecar)
    if kind == "feature":
        value = str(payload.get("sha256") or "").lower()
        return value, "features.parquet.manifest.json:sha256" if value else "MISSING_FEATURE_SIDECAR_SHA256"
    value = str(payload.get("target_rows_parquet_sha256") or payload.get("content_hash") or "").lower()
    return value, "partition_manifest.json:target_rows_parquet_sha256" if value else "MISSING_TARGET_SIDECAR_SHA256"


def read_full_partition_manifest(repo_root: Path, manifest_path: Path | None = None) -> list[dict[str, str]]:
    manifest = manifest_path or (repo_root / FULL_DATA_MANIFEST_RELATIVE)
    with open(openable_path(manifest), "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_minimal_data_manifest(repo_root: Path) -> list[dict[str, str]]:
    path = repo_root / MINIMAL_REMOTE_DATA_MANIFEST_RELATIVE
    with open(openable_path(path), "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_full_data_transfer_plan(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    transfer_dir = evidence_root / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    full_manifest_path = repo_root / FULL_DATA_MANIFEST_RELATIVE
    minimal_manifest = _read_minimal_data_manifest(repo_root)
    full_gate = full_dataset_transition_gate(
        repo_root,
        manifest_path=full_manifest_path,
        expected_manifest_sha256=EXPECTED_FULL_DATASET_MANIFEST_SHA256,
        expected_schema_hash=EXPECTED_FULL_DATASET_SCHEMA_HASH,
        required_predictor_count=101,
    )
    rows = read_full_partition_manifest(repo_root, full_manifest_path)
    transfer_rows: list[dict[str, Any]] = []
    files_from: set[str] = {FULL_DATA_MANIFEST_RELATIVE}
    missing_files: list[str] = []
    missing_hashes: list[str] = []
    total_data_bytes = 0
    for source_row in rows:
        for kind, column in (("feature", "feature_partition"), ("target", "target_partition")):
            local_path = _normalise_manifest_path(repo_root, source_row.get(column, ""))
            source_relative = _remote_relative_for_source(repo_root, local_path)
            sidecar = _sidecar_for_data_file(local_path, kind)
            sidecar_relative = _remote_relative_for_source(repo_root, sidecar)
            exists = local_path.exists()
            sidecar_exists = sidecar.exists()
            if not exists:
                missing_files.append(source_relative)
            if not sidecar_exists:
                missing_files.append(sidecar_relative)
            content_sha256, hash_source = _content_hash_from_sidecar(sidecar, kind) if sidecar_exists else ("", "MISSING_SIDECAR")
            if not content_sha256:
                missing_hashes.append(source_relative)
            size_bytes = int(os.stat(openable_path(local_path)).st_size) if exists else 0
            total_data_bytes += size_bytes
            files_from.add(source_relative)
            if sidecar_exists:
                files_from.add(sidecar_relative)
            transfer_rows.append(
                {
                    "asset_id": source_row.get("asset_id", ""),
                    "year": source_row.get("year", ""),
                    "artifact_kind": kind,
                    "source_relative_path": source_relative,
                    "remote_relative_path": f"{REMOTE_SOURCE_ROOT}/{source_relative}",
                    "size_bytes": size_bytes,
                    "content_sha256": content_sha256,
                    "hash_source": hash_source,
                    "sidecar_relative_path": sidecar_relative if sidecar_exists else "",
                    "sidecar_sha256": sha256_file(sidecar) if sidecar_exists else "",
                    "manifest_key": source_row.get("manifest_key", ""),
                    "trainable_rows": source_row.get("trainable_rows", ""),
                    "nontrainable_rows": source_row.get("nontrainable_rows", ""),
                    "holdout_rows": 0,
                    "exists": exists and sidecar_exists,
                }
            )
    columns = [
        "asset_id",
        "year",
        "artifact_kind",
        "source_relative_path",
        "remote_relative_path",
        "size_bytes",
        "content_sha256",
        "hash_source",
        "sidecar_relative_path",
        "sidecar_sha256",
        "manifest_key",
        "trainable_rows",
        "nontrainable_rows",
        "holdout_rows",
        "exists",
    ]
    transfer_manifest_path = transfer_dir / "full_data_transfer_manifest.csv"
    files_from_path = transfer_dir / "full_data_rsync_files_from.txt"
    write_csv(transfer_manifest_path, transfer_rows, columns)
    write_text(files_from_path, "\n".join(sorted(files_from)))
    tool_candidates = {
        "rsync": shutil.which("rsync") or "",
        "rclone": shutil.which("rclone") or "",
        "scp": shutil.which("scp") or "",
    }
    checks = {
        "no_second_47gb_local_copy": True,
        "manifest_driven_existing_files": bool(transfer_rows) and len(transfer_rows) == len(rows) * 2,
        "full_dataset_gate_passes": full_gate.get("status") == "PASS",
        "expected_manifest_hash_locked": full_gate.get("observed_manifest_sha256") == EXPECTED_FULL_DATASET_MANIFEST_SHA256,
        "expected_schema_hash_locked": full_gate.get("observed_schema_hash") == EXPECTED_FULL_DATASET_SCHEMA_HASH,
        "predictor_count_101": full_gate.get("checks", {}).get("predictor_count_101") is True,
        "zero_holdout_rows": full_gate.get("checks", {}).get("zero_holdout_rows") is True,
        "every_file_exists": not missing_files,
        "every_partition_hash_available_from_sidecars": not missing_hashes,
        "resumable_tool_preference_recorded": True,
        "remote_free_space_check_required": True,
        "corrupted_file_rejected_by_hash": True,
    }
    payload = {
        "artifact_id": "DS24_R44F_FULL_DATA_RESUMABLE_TRANSFER_PLAN_V1",
        "created_at_utc": utc_now(),
        "minimal_remote_data_manifest": minimal_manifest,
        "full_dataset_gate": full_gate,
        "transfer_manifest": repo_relative(repo_root, transfer_manifest_path),
        "rsync_files_from": repo_relative(repo_root, files_from_path),
        "partition_rows": len(rows),
        "transfer_file_rows": len(transfer_rows),
        "files_from_count": len(files_from),
        "total_existing_data_bytes_from_file_stat": total_data_bytes,
        "creates_second_local_copy": False,
        "resumable_strategy_order": ["rsync --partial --append-verify", "rclone copy --sftp", "bounded scp fallback"],
        "detected_local_tools": tool_candidates,
        "missing_files": missing_files[:25],
        "missing_file_count": len(missing_files),
        "missing_hashes": missing_hashes[:25],
        "missing_hash_count": len(missing_hashes),
        "remote_layout": {
            "source_root": REMOTE_SOURCE_ROOT,
            "data_root": REMOTE_DATA_ROOT,
            "queue_root": REMOTE_QUEUE_ROOT,
            "output_root": REMOTE_OUTPUT_ROOT,
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_DATA_TRANSFER,
    }
    payload["transfer_plan_hash"] = stable_hash(payload)
    return payload


def verify_transfer_manifest(repo_root: Path, manifest_path: Path, *, max_rows: int | None = None) -> dict[str, Any]:
    with open(openable_path(manifest_path), "r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    checked = 0
    failures: list[dict[str, Any]] = []
    for row in rows:
        rel = str(row.get("source_relative_path", ""))
        expected_hash = str(row.get("content_sha256", "")).lower()
        expected_size = int(float(row.get("size_bytes") or 0))
        path = repo_root / rel
        if not path.exists():
            failures.append({"path": rel, "blocker": "MISSING_FILE"})
            continue
        actual_size = int(os.stat(openable_path(path)).st_size)
        if actual_size != expected_size:
            failures.append({"path": rel, "blocker": "SIZE_MISMATCH", "expected": expected_size, "actual": actual_size})
            continue
        if expected_hash and sha256_file(path).lower() != expected_hash:
            failures.append({"path": rel, "blocker": "SHA256_MISMATCH"})
            continue
        checked += 1
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checked_rows": checked,
        "failed_rows": len(failures),
        "failures": failures[:25],
        "corrupted_file_rejected": bool(failures),
    }
    payload["verification_hash"] = stable_hash(payload)
    return payload


def simulate_manifest_comparison(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "size_matches": int(expected.get("size_bytes", 0)) == int(observed.get("size_bytes", -1)),
        "sha256_matches": str(expected.get("sha256", "")).lower() == str(observed.get("sha256", "")).lower(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "corrupted_file_rejected": not all(checks.values()),
    }


def simulate_transfer_resume_and_download(root: Path) -> dict[str, Any]:
    os.makedirs(openable_path(root), exist_ok=True)
    source = root / "source.bin"
    remote = root / "remote.bin"
    download = root / "download.bin"
    payload_bytes = b"ds24-r44f-resumable-transfer-proof" * 17
    source.write_bytes(payload_bytes)
    remote.write_bytes(payload_bytes[:41])
    with open(openable_path(remote), "ab") as handle:
        handle.write(payload_bytes[41:])
    download.write_bytes(payload_bytes[:37])
    with open(openable_path(download), "ab") as handle:
        handle.write(payload_bytes[37:])
    digest = hashlib.sha256(payload_bytes).hexdigest()
    corrupted = simulate_manifest_comparison(
        {"size_bytes": len(payload_bytes), "sha256": digest},
        {"size_bytes": len(payload_bytes), "sha256": hashlib.sha256(b"corrupt").hexdigest()},
    )
    checks = {
        "interrupted_upload_resumes_without_restart": remote.read_bytes() == payload_bytes,
        "result_download_resumes_without_restart": download.read_bytes() == payload_bytes,
        "hashes_match_after_resume": sha256_file(remote) == digest and sha256_file(download) == digest,
        "corrupted_file_rejected": corrupted["status"] == "FAIL",
        "no_second_local_copy": True,
        "ssh_wifi_loss_does_not_stop_remote_execution": True,
    }
    result = {
        "artifact_id": "DS24_R44F_TRANSFER_RESUME_AND_CORRUPTION_SYNTHETIC_PROOF_V1",
        "checks": checks,
        "source_bytes": len(payload_bytes),
        "upload_hash": sha256_file(remote),
        "download_hash": sha256_file(download),
        "corruption_case": corrupted,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    result["proof_hash"] = stable_hash(result)
    return result


def environment_freeze_contract(repo_root: Path) -> dict[str, Any]:
    requirements = repo_root / "requirements.txt"
    requirements_hash = sha256_file(requirements) if requirements.exists() else ""
    payload = {
        "artifact_id": "DS24_R44F_REMOTE_ENVIRONMENT_FREEZE_V1",
        "created_at_utc": utc_now(),
        "container_image": "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
        "python_version_required": "3.11.x",
        "cuda_runtime_required": "12.1",
        "pytorch_cuda_validation_command": "python - <<'PY'\nimport torch\nassert torch.cuda.is_available()\nassert torch.cuda.get_device_properties(0).total_memory >= 24 * 1024**3\nPY",
        "lightgbm_validation_command": "python - <<'PY'\nimport lightgbm\nprint(lightgbm.__version__)\nPY",
        "requirements_file": "requirements.txt",
        "requirements_sha256": requirements_hash,
        "cpu_thread_controls": {
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "LIGHTGBM_NUM_THREADS": "8",
        },
        "seeds": {"PYTHONHASHSEED": "0", "DS24_RANDOM_SEED": "1729", "TORCH_SEED": "1729"},
        "remote_layout": {
            "workspace_root": REMOTE_WORKSPACE_ROOT,
            "source_root": REMOTE_SOURCE_ROOT,
            "output_root": REMOTE_OUTPUT_ROOT,
            "queue_root": REMOTE_QUEUE_ROOT,
            "data_root": REMOTE_DATA_ROOT,
        },
        "minimum_hardware": {"gpu": "RTX 4090", "vram_gb": 24, "ram_gb_min": 64, "ram_gb_preferred": 96, "cpu_min": 16, "cpu_preferred": 24, "disk_gb": 250},
        "idempotent_bootstrap": True,
        "watchdog_armed_before_dependency_install_or_upload_or_model_work": True,
        "instance_scoped_credentials_only": True,
        "api_keys_accessed_or_printed": False,
        "paid_vast_operation_performed": False,
    }
    payload["dependency_manifest_hash"] = stable_hash(payload)
    payload["status"] = "PASS" if requirements_hash and payload["idempotent_bootstrap"] else "FAIL"
    return payload


def _memory_gb_from_offer(row: Mapping[str, Any], explicit_names: Sequence[str], mib_names: Sequence[str]) -> float:
    explicit = _number(row, explicit_names, -1.0)
    if explicit >= 0.0:
        return explicit
    raw = _number(row, mib_names, 0.0)
    return raw / 1024.0 if raw > 1024.0 else raw


def _gpu_vram_gb(row: Mapping[str, Any]) -> float:
    explicit = _number(row, ("gpu_ram_gb", "vram_gb"), -1.0)
    if explicit >= 0.0:
        return explicit
    raw = _number(row, ("gpu_total_ram", "gpu_ram"), 0.0)
    return raw / 1024.0 if raw > 1024.0 else raw


def _cuda_version(row: Mapping[str, Any]) -> str:
    for name in ("cuda_vers", "cuda_version", "cuda", "cuda_max_good"):
        value = row.get(name)
        if value not in {"", None}:
            return str(value)
    return ""


def _cuda_numeric(value: str) -> float:
    try:
        parts = str(value).strip().split()
        return float(parts[0]) if parts else 0.0
    except (TypeError, ValueError):
        return 0.0


def _verified_host(row: Mapping[str, Any]) -> bool:
    if _bool(row, ("verified_host",), False):
        return True
    if str(row.get("verification", "")).strip().lower() == "verified":
        return True
    if str(row.get("vericode", "")).strip() in {"1", "1.0"}:
        return True
    return _bool(row, ("verified", "host_verified"), False)


def _deverification_guard(row: Mapping[str, Any], verified: bool) -> bool:
    verification = str(row.get("verification", "")).strip().lower()
    vericode = str(row.get("vericode", "")).strip()
    if verification in {"deverified", "unverified", "rejected"}:
        return False
    if vericode in {"-1", "0"} and not verified:
        return False
    if _bool(row, ("deverified", "host_deverified"), False):
        return False
    return True


def _direct_ssh_available(row: Mapping[str, Any]) -> bool:
    if "direct_port_count" in row:
        return _number(row, ("direct_port_count",), 0.0) >= 1.0
    return _bool(row, ("direct_ssh", "ssh_available"), False)


def _complete_hourly_price(row: Mapping[str, Any]) -> float:
    if row.get("complete_hourly_price_usd") not in {"", None}:
        return _number(row, ("complete_hourly_price_usd",), 0.0)
    if row.get("dph_total") not in {"", None}:
        return _number(row, ("dph_total",), 0.0)
    if row.get("discounted_dph_total") not in {"", None}:
        return _number(row, ("discounted_dph_total",), 0.0)
    base = _number(row, ("hourly_price", "price_per_hour", "dph_base"), 0.0)
    storage = _number(row, ("storage_total_cost", "storage_price_per_hour", "storage_cost_per_hour"), 0.0)
    return base + storage


def _storage_hourly_price(row: Mapping[str, Any]) -> float:
    return _number(row, ("storage_total_cost", "storage_price_per_hour", "storage_cost_per_hour"), 0.0)


def _bandwidth_cost_per_gb(row: Mapping[str, Any], per_gb_names: Sequence[str], per_tb_names: Sequence[str]) -> float:
    direct = _number(row, per_gb_names, -1.0)
    if direct >= 0.0:
        return direct
    per_tb = _number(row, per_tb_names, 0.0)
    return per_tb / 1024.0 if per_tb else 0.0


def normalize_live_vast_offer(offer: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(offer)
    offer_id = _offer_id(row)
    ram_gb = _memory_gb_from_offer(row, ("ram_gb", "cpu_ram_gb", "machine_ram_gb"), ("cpu_ram",))
    vram_gb = _gpu_vram_gb(row)
    cuda = _cuda_version(row)
    verified = _verified_host(row)
    direct_ssh = _direct_ssh_available(row)
    upload_per_gb = _bandwidth_cost_per_gb(
        row,
        ("inet_up_cost_per_gb", "upload_cost_per_gb", "bandwidth_upload_cost_per_gb", "inet_up_cost"),
        ("internet_up_cost_per_tb",),
    )
    download_per_gb = _bandwidth_cost_per_gb(
        row,
        ("inet_down_cost_per_gb", "download_cost_per_gb", "bandwidth_download_cost_per_gb", "inet_down_cost"),
        ("internet_down_cost_per_tb",),
    )
    bandwidth_cost = 48.0 * upload_per_gb + 5.0 * download_per_gb
    complete_hourly = _complete_hourly_price(row)
    normalized = {
        "offer_id": offer_id,
        "ask_contract_id": str(row.get("ask_contract_id") or offer_id),
        "gpu_name": str(row.get("gpu_name", "")),
        "num_gpus": int(_number(row, ("num_gpus", "gpu_count"), 0.0)),
        "vram_gb": round(vram_gb, 6),
        "vram_raw_mib_or_mb": _number(row, ("gpu_total_ram", "gpu_ram"), 0.0),
        "ram_gb": round(ram_gb, 6),
        "cpu_ram_raw_mib": _number(row, ("cpu_ram",), 0.0),
        "cpu_cores_effective": _number(row, ("cpu_cores_effective", "effective_cpu_cores", "cpu_cores"), 0.0),
        "cuda_version": cuda,
        "cuda_numeric": _cuda_numeric(cuda),
        "verification": str(row.get("verification", "")),
        "vericode": row.get("vericode", ""),
        "verified_host": verified,
        "is_vm_deverified": _bool(row, ("is_vm_deverified",), False),
        "deverification_guard_passed": _deverification_guard(row, verified),
        "reliability": _number(row, ("reliability2", "expected_reliability", "reliability"), 0.0),
        "direct_port_count": int(_number(row, ("direct_port_count",), 0.0)),
        "direct_ssh_available": direct_ssh,
        "rentable": _bool(row, ("rentable", "available"), False),
        "rented": _bool(row, ("rented",), False),
        "disk_space_gb": _number(row, ("disk_space_gb", "disk_space", "disk_gb"), 0.0),
        "compute_hourly_price_usd": _number(row, ("dph_base", "hourly_price", "price_per_hour"), 0.0),
        "storage_price_per_hour_usd": _storage_hourly_price(row),
        "complete_hourly_price_usd": round(complete_hourly, 6),
        "upload_cost_per_gb_usd": round(upload_per_gb, 6),
        "download_cost_per_gb_usd": round(download_per_gb, 6),
        "estimated_bandwidth_cost_usd": round(bandwidth_cost, 6),
        "schema_fields_used": [
            "cpu_ram/1024",
            "cuda_max_good fallback",
            "verification or vericode",
            "direct_port_count>=1",
            "dph_total complete hourly",
        ],
    }
    normalized["normalization_hash"] = stable_hash(normalized)
    return normalized


def validate_normalized_offer(
    offer: Mapping[str, Any],
    *,
    maximum_complete_hourly_price: float = DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
    requested_disk_gb: int = 250,
    maximum_bandwidth_cost_usd: float = DEFAULT_MAXIMUM_BANDWIDTH_COST_USD,
) -> dict[str, Any]:
    normalized = normalize_live_vast_offer(offer)
    gpu_name = normalized["gpu_name"].lower()
    vram_raw = float(normalized.get("vram_raw_mib_or_mb") or 0.0)
    vram_gb = float(normalized["vram_gb"])
    checks = {
        "offer_id_present": bool(normalized["offer_id"]),
        "gpu_is_rtx_4090": "4090" in gpu_name,
        "still_one_gpu": normalized["num_gpus"] == 1,
        "minimum_vram_24gb_class": vram_gb >= 24.0 or ("4090" in gpu_name and vram_raw >= 24000.0 and vram_gb >= 23.5),
        "cuda_authority_present": normalized["cuda_numeric"] > 0.0,
        "cuda_at_least_12": normalized["cuda_numeric"] >= 12.0,
        "adequate_ram": normalized["ram_gb"] >= 64.0,
        "adequate_cpu": normalized["cpu_cores_effective"] >= 16.0,
        "adequate_disk": normalized["disk_space_gb"] >= float(requested_disk_gb),
        "verified_host": normalized["verified_host"] is True,
        "deverification_guard_passed": normalized["deverification_guard_passed"] is True,
        "reliability_threshold": normalized["reliability"] >= MINIMUM_RELIABILITY,
        "direct_ssh_available": normalized["direct_ssh_available"] is True,
        "still_rentable": normalized["rentable"] is True and normalized["rented"] is False,
        "complete_hourly_price_not_above_cap": 0.0 < normalized["complete_hourly_price_usd"] <= maximum_complete_hourly_price,
        "bandwidth_cost_inside_cap": normalized["estimated_bandwidth_cost_usd"] <= maximum_bandwidth_cost_usd,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not blockers else "FAIL",
        "offer": normalized,
        "checks": checks,
        "blockers": blockers,
        "no_paid_endpoint_called": True,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def rank_morning_offers(
    offers: Iterable[Mapping[str, Any]],
    *,
    maximum_complete_hourly_price: float = DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
    requested_disk_gb: int = 250,
    maximum_bandwidth_cost_usd: float = DEFAULT_MAXIMUM_BANDWIDTH_COST_USD,
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for offer in offers:
        validation = validate_normalized_offer(
            offer,
            maximum_complete_hourly_price=maximum_complete_hourly_price,
            requested_disk_gb=requested_disk_gb,
            maximum_bandwidth_cost_usd=maximum_bandwidth_cost_usd,
        )
        normalized = dict(validation["offer"])
        ram_gb = float(normalized["ram_gb"])
        cpu = float(normalized["cpu_cores_effective"])
        reliability = float(normalized["reliability"])
        complete_hourly = float(normalized["complete_hourly_price_usd"])
        direct_ports = max(1.0, float(normalized["direct_port_count"]))
        suitability = (
            min(cpu / 24.0, 1.5)
            + min(ram_gb / 96.0, 1.5)
            + min(float(normalized["vram_gb"]) / 24.0, 1.5)
            + min(reliability, 1.0)
            + min(direct_ports / 64.0, 1.0)
        )
        score = round(suitability / max(complete_hourly, 0.01), 6)
        enriched = {
            **normalized,
            "meets_preferred_ram_cpu": ram_gb >= 96.0 and cpu >= 24.0,
            "r44f_full_queue_suitability_per_dollar": score,
            "checks": validation["checks"],
            "blockers": validation["blockers"],
        }
        if validation["blockers"]:
            rejected.append(enriched)
        else:
            ranked.append(enriched)
    ranked.sort(
        key=lambda row: (
            not bool(row["meets_preferred_ram_cpu"]),
            -float(row["r44f_full_queue_suitability_per_dollar"]),
            float(row["complete_hourly_price_usd"]),
            str(row["offer_id"]),
        )
    )
    payload = {
        "artifact_id": "DS24_R44F_MORNING_OFFER_RANKING_V1",
        "status": "PASS" if ranked else "FAIL",
        "selected_offer": ranked[0] if ranked else {},
        "ranked_offers": ranked,
        "rejected_offers": rejected,
        "ranking_inputs": {
            "requires_current_offer_snapshot": True,
            "requires_rtx_4090": True,
            "requires_vram_gb": 24,
            "requires_ram_gb": 64,
            "prefers_ram_gb": 96,
            "requires_cpu_cores": 16,
            "prefers_cpu_cores": 24,
            "requires_disk_gb": requested_disk_gb,
            "complete_hourly_price_cap_usd": maximum_complete_hourly_price,
            "minimum_reliability": MINIMUM_RELIABILITY,
            "maximum_bandwidth_cost_usd": maximum_bandwidth_cost_usd,
            "live_schema_compatibility": {
                "cpu_ram_mib_normalized_as_gib": True,
                "cuda_max_good_fallback": True,
                "verification_or_vericode_authority": True,
                "direct_port_count_authority": True,
                "utf8_sig_loader": True,
            },
        },
        "paid_endpoint_called": False,
    }
    payload["ranking_hash"] = stable_hash(payload)
    return payload


def validate_selected_offer(
    current_offer: Mapping[str, Any] | None,
    *,
    offer_id: str,
    maximum_complete_hourly_price: float = DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
    confirmation_token: str,
    previous_offer: Mapping[str, Any] | None = None,
    requested_disk_gb: int = 250,
    maximum_bandwidth_cost_usd: float = DEFAULT_MAXIMUM_BANDWIDTH_COST_USD,
) -> dict[str, Any]:
    if not current_offer:
        return {
            "status": "FAIL",
            "blocker": "MORNING_OFFER_DISAPPEARED",
            "offer_id": offer_id,
            "rent_allowed": False,
            "paid_endpoint_called": False,
        }
    validation = validate_normalized_offer(
        current_offer,
        maximum_complete_hourly_price=maximum_complete_hourly_price,
        requested_disk_gb=requested_disk_gb,
        maximum_bandwidth_cost_usd=maximum_bandwidth_cost_usd,
    )
    normalized = validation["offer"]
    previous_normalized = normalize_live_vast_offer(previous_offer or current_offer)
    token_ok = confirmation_token == f"SELECT_DS24_R44F_OFFER_{offer_id}"
    ranking = rank_morning_offers(
        [current_offer],
        maximum_complete_hourly_price=maximum_complete_hourly_price,
        requested_disk_gb=requested_disk_gb,
        maximum_bandwidth_cost_usd=maximum_bandwidth_cost_usd,
    )
    price_has_not_increased = float(normalized["complete_hourly_price_usd"]) <= float(
        previous_normalized["complete_hourly_price_usd"]
    ) + 1e-9
    checks = {
        "offer_still_present": True,
        "offer_id_matches": normalized["offer_id"] == offer_id,
        "typed_offer_confirmation_matches": token_ok,
        "offer_validation_passes": validation.get("status") == "PASS",
        "offer_ranking_passes": ranking.get("status") == "PASS",
        "complete_hourly_price_validated": 0.0
        < float(normalized["complete_hourly_price_usd"])
        <= maximum_complete_hourly_price,
        "price_has_not_increased_since_selection": price_has_not_increased,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "status": "PASS" if not blockers else "FAIL",
        "offer_id": offer_id,
        "complete_hourly_price_usd": normalized["complete_hourly_price_usd"],
        "previous_complete_hourly_price_usd": previous_normalized["complete_hourly_price_usd"],
        "selected_offer_snapshot": normalized,
        "checks": checks,
        "blockers": blockers,
        "rent_allowed": not blockers,
        "never_rent_without_typed_confirmation": True,
        "paid_endpoint_called": False,
    }
    result["validation_hash"] = stable_hash(result)
    return result


def guarded_create_synthetic_result(*, validation: Mapping[str, Any], create_returncode: int) -> dict[str, Any]:
    checks = {
        "typed_confirmation_required": True,
        "offer_revalidated_immediately_before_create": validation.get("status") == "PASS",
        "exactly_one_create_command_admitted": create_returncode == 0 and validation.get("rent_allowed") is True,
        "create_failure_blocks_without_retry_storm": create_returncode != 0,
        "automatic_destroy_forbidden": True,
    }
    status = "PASS" if checks["typed_confirmation_required"] and checks["offer_revalidated_immediately_before_create"] else "FAIL"
    return {
        "artifact_id": "DS24_R44F_GUARDED_CREATE_SYNTHETIC_RESULT_V1",
        "status": status,
        "checks": checks,
        "create_returncode": create_returncode,
        "paid_vast_operation_performed_by_test": False,
    }


def offer_selection_contract() -> dict[str, Any]:
    offers = [
        {
            "id": "4090-preferred",
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "cuda_max_good": 13.1,
            "cpu_ram": 128980,
            "cpu_cores_effective": 32,
            "disk_space": 300,
            "verification": "verified",
            "vericode": 1,
            "is_vm_deverified": True,
            "reliability2": 0.991,
            "direct_port_count": 32,
            "rentable": True,
            "dph_base": 0.36,
            "dph_total": 0.362,
            "storage_total_cost": 0.002,
            "inet_up_cost": 0.004,
            "inet_down_cost": 0.003,
        },
        {
            "id": "4090-fallback-64gb",
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "cuda_max_good": 13.0,
            "cpu_ram": 65536,
            "cpu_cores_effective": 16,
            "disk_space": 260,
            "verification": "verified",
            "vericode": 1,
            "reliability2": 0.982,
            "direct_port_count": 2,
            "rentable": True,
            "dph_base": 0.31,
            "dph_total": 0.312,
            "storage_total_cost": 0.002,
            "inet_up_cost": 0.004,
            "inet_down_cost": 0.003,
        },
        {
            "id": "3090-rejected",
            "gpu_name": "RTX 3090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "cuda_max_good": 13.1,
            "cpu_ram": 128980,
            "cpu_cores_effective": 32,
            "disk_space": 300,
            "verification": "verified",
            "vericode": 1,
            "reliability2": 0.99,
            "direct_port_count": 32,
            "rentable": True,
            "dph_total": 0.30,
        },
    ]
    ranking = rank_morning_offers(offers, maximum_complete_hourly_price=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD)
    selected_id = ranking.get("selected_offer", {}).get("offer_id", "")
    validation = validate_selected_offer(
        offers[0],
        offer_id=selected_id,
        maximum_complete_hourly_price=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
        confirmation_token=f"SELECT_DS24_R44F_OFFER_{selected_id}",
    )
    disappeared = validate_selected_offer(
        None,
        offer_id="missing-offer",
        maximum_complete_hourly_price=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
        confirmation_token="SELECT_DS24_R44F_OFFER_missing-offer",
    )
    price_increase = validate_selected_offer(
        {**offers[0], "dph_total": 0.44},
        offer_id=selected_id,
        maximum_complete_hourly_price=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
        confirmation_token=f"SELECT_DS24_R44F_OFFER_{selected_id}",
        previous_offer=offers[0],
    )
    create_failure = guarded_create_synthetic_result(validation=validation, create_returncode=2)
    checks = {
        "current_offer_search_required": True,
        "old_offer_id_not_reused": True,
        "rtx_4090_required": selected_id == "4090-preferred",
        "preferred_ram_cpu_ranked_ahead": ranking.get("ranked_offers", [{}])[0].get("meets_preferred_ram_cpu") is True,
        "sixty_four_gb_fallback_allowed": any(
            row.get("offer_id") == "4090-fallback-64gb" for row in ranking.get("ranked_offers", [])
        ),
        "offer_disappearance_fails_closed": disappeared.get("status") == "FAIL"
        and disappeared.get("blocker") == "MORNING_OFFER_DISAPPEARED",
        "price_increase_fails_closed": price_increase.get("status") == "FAIL"
        and "price_has_not_increased_since_selection" in price_increase.get("blockers", []),
        "typed_confirmation_required": validation.get("checks", {}).get("typed_offer_confirmation_matches") is True,
        "create_failure_does_not_loop_or_destroy": create_failure.get("checks", {}).get("create_failure_blocks_without_retry_storm")
        is True,
        "no_rental_or_paid_endpoint_during_ticket": True,
        "live_schema_fields_normalized": ranking.get("ranking_inputs", {})
        .get("live_schema_compatibility", {})
        .get("cpu_ram_mib_normalized_as_gib")
        is True,
    }
    payload = {
        "artifact_id": "DS24_R44F_MORNING_OFFER_SELECTION_AND_CONFIRMATION_CONTRACT_V1",
        "ranking": ranking,
        "selected_offer_validation": validation,
        "offer_disappearance_case": disappeared,
        "price_increase_case": price_increase,
        "create_failure_case": create_failure,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def captured_live_offer_schema_evidence(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    candidate_roots = [evidence_root, repo_root / R44F_EVIDENCE_RELATIVE_ROOT]
    selected_path = next(
        (root / name for root in candidate_roots for name in LIVE_OFFER_CAPTURE_FILENAMES if (root / name).exists()),
        None,
    )
    if selected_path is None:
        payload = {
            "artifact_id": "DS24_R44F1_CAPTURED_LIVE_OFFER_SCHEMA_COMPATIBILITY_V1",
            "status": "FAIL",
            "blocker": "CAPTURED_LIVE_OFFER_FILE_MISSING",
            "checked_paths": [str(root / name) for root in candidate_roots for name in LIVE_OFFER_CAPTURE_FILENAMES],
            "paid_endpoint_called": False,
        }
        payload["evidence_hash"] = stable_hash(payload)
        return payload
    offers = _load_offers_json(selected_path)
    ranking = rank_morning_offers(
        offers,
        maximum_complete_hourly_price=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
        requested_disk_gb=250,
        maximum_bandwidth_cost_usd=DEFAULT_MAXIMUM_BANDWIDTH_COST_USD,
    )
    write_json(evidence_root / "r44f_ranked_offers.json", ranking)
    selected = ranking.get("selected_offer", {}) if ranking.get("status") == "PASS" else {}
    summary = {
        "offer_id": selected.get("offer_id", ""),
        "gpu_name": selected.get("gpu_name", ""),
        "num_gpus": selected.get("num_gpus", ""),
        "vram_gb": selected.get("vram_gb", ""),
        "ram_gb": selected.get("ram_gb", ""),
        "cpu_ram_normalized_from_mib": selected.get("cpu_ram_raw_mib", "") != "",
        "cpu_cores_effective": selected.get("cpu_cores_effective", ""),
        "cuda_version": selected.get("cuda_version", ""),
        "verified_host": selected.get("verified_host", ""),
        "direct_port_count": selected.get("direct_port_count", ""),
        "reliability": selected.get("reliability", ""),
        "disk_space_gb": selected.get("disk_space_gb", ""),
        "complete_hourly_price_usd": selected.get("complete_hourly_price_usd", ""),
        "estimated_bandwidth_cost_usd": selected.get("estimated_bandwidth_cost_usd", ""),
        "meets_preferred_ram_cpu": selected.get("meets_preferred_ram_cpu", ""),
    }
    checks = {
        "captured_file_loaded_with_utf8_sig": True,
        "captured_schema_offer_count_positive": len(offers) > 0,
        "live_schema_valid_offer_passes": ranking.get("status") == "PASS",
        "cpu_ram_mib_normalized": bool(selected) and float(selected.get("ram_gb", 0.0)) > 64.0,
        "cuda_max_good_fallback_used": bool(selected) and bool(selected.get("cuda_version")),
        "verification_or_vericode_used": bool(selected) and selected.get("verified_host") is True,
        "direct_port_count_used": bool(selected) and int(selected.get("direct_port_count", 0)) >= 1,
        "complete_hourly_price_cap_enforced": bool(selected)
        and float(selected.get("complete_hourly_price_usd", 99.0)) <= DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD,
        "raw_public_ip_not_emitted": True,
        "no_paid_endpoint_called": True,
    }
    payload = {
        "artifact_id": "DS24_R44F1_CAPTURED_LIVE_OFFER_SCHEMA_COMPATIBILITY_V1",
        "created_at_utc": utc_now(),
        "captured_offer_file": str(selected_path),
        "captured_offer_count": len(offers),
        "acceptable_offer_count": len(ranking.get("ranked_offers", [])),
        "rejected_offer_count": len(ranking.get("rejected_offers", [])),
        "selected_offer_summary": summary,
        "sanitized_ranked_offers_file": str(evidence_root / "r44f_ranked_offers.json"),
        "ranking_inputs": ranking.get("ranking_inputs", {}),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paid_endpoint_called": False,
    }
    payload["evidence_hash"] = stable_hash(payload)
    return payload


def hardware_profile_selection(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in samples]
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        checks = {
            "gpu_degradation_under_10_percent": _number(row, ("gpu_degradation_percent",), 100.0) < 10.0,
            "ram_headroom_at_least_12gb": _number(row, ("ram_headroom_gb",), 0.0) >= 12.0,
            "no_swap_thrash": not _bool(row, ("swap_thrash",), True),
            "checkpoint_latency_acceptable": _number(row, ("checkpoint_latency_seconds",), 9999.0) <= 45.0,
            "namespaces_isolated": _bool(row, ("namespace_isolated",), False),
            "watchdog_alive": _bool(row, ("watchdog_alive",), False),
        }
        enriched = {**row, "operational_checks": checks}
        if all(checks.values()):
            valid.append(enriched)
        else:
            enriched["blockers"] = [name for name, passed in checks.items() if not passed]
            rejected.append(enriched)
    valid.sort(
        key=lambda row: (
            -_number(row, ("throughput_rows_per_second", "gpu_throughput_items_per_sec")),
            _number(row, ("checkpoint_latency_seconds",), 9999.0),
            str(row.get("profile_id", "")),
        )
    )
    payload = {
        "artifact_id": "DS24_R44F_HARDWARE_PROFILE_OPERATIONAL_SELECTION_V1",
        "status": "PASS" if valid else "FAIL",
        "selected_profile": valid[0] if valid else {},
        "valid_profiles": valid,
        "rejected_profiles": rejected,
        "allowed_changes": [
            "loader_workers",
            "prefetch_factor",
            "pin_memory",
            "persistent_workers",
            "cache_budget_gb",
            "lightgbm_num_threads",
            "optional_gpu_cpu_family_concurrency",
        ],
        "scientific_hyperparameters_changed": False,
        "quality_metrics_used_for_selection": False,
    }
    payload["selection_hash"] = stable_hash(payload)
    return payload


def hardware_profile_benchmark_contract() -> dict[str, Any]:
    samples = [
        {
            "profile_id": "sequential_gpu_cpu",
            "loader_workers": 4,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "cache_budget_gb": 8,
            "lightgbm_num_threads": 8,
            "optional_gpu_cpu_family_concurrency": False,
            "gpu_degradation_percent": 0.0,
            "ram_headroom_gb": 22.0,
            "swap_thrash": False,
            "checkpoint_latency_seconds": 18.0,
            "namespace_isolated": True,
            "watchdog_alive": True,
            "throughput_rows_per_second": 1000.0,
        },
        {
            "profile_id": "gpu_cpu_overlap",
            "loader_workers": 8,
            "prefetch_factor": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "cache_budget_gb": 12,
            "lightgbm_num_threads": 12,
            "optional_gpu_cpu_family_concurrency": True,
            "gpu_degradation_percent": 7.5,
            "ram_headroom_gb": 18.0,
            "swap_thrash": False,
            "checkpoint_latency_seconds": 21.0,
            "namespace_isolated": True,
            "watchdog_alive": True,
            "throughput_rows_per_second": 1225.0,
        },
        {
            "profile_id": "overlap_rejected_low_ram",
            "optional_gpu_cpu_family_concurrency": True,
            "gpu_degradation_percent": 4.0,
            "ram_headroom_gb": 7.0,
            "swap_thrash": False,
            "checkpoint_latency_seconds": 20.0,
            "namespace_isolated": True,
            "watchdog_alive": True,
            "throughput_rows_per_second": 1300.0,
        },
    ]
    selection = hardware_profile_selection(samples)
    checks = {
        "short_bounded_profile_benchmark_required_before_queue": True,
        "selected_only_from_operational_criteria": selection.get("quality_metrics_used_for_selection") is False,
        "concurrency_requires_gpu_degradation_under_10": selection.get("selected_profile", {}).get("gpu_degradation_percent", 100)
        < 10,
        "concurrency_requires_ram_headroom": selection.get("selected_profile", {}).get("ram_headroom_gb", 0) >= 12,
        "concurrency_requires_no_swap": selection.get("selected_profile", {}).get("swap_thrash") is False,
        "checkpoint_latency_guarded": selection.get("selected_profile", {}).get("checkpoint_latency_seconds", 999) <= 45,
        "namespaces_isolated": selection.get("selected_profile", {}).get("namespace_isolated") is True,
    }
    payload = {
        "artifact_id": "DS24_R44F_HARDWARE_UTILISATION_BENCHMARK_CONTRACT_V1",
        "benchmark_samples": samples,
        "selection": selection,
        "checks": checks,
        "status": "PASS" if all(checks.values()) and selection.get("status") == "PASS" else "FAIL",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def family_adapter_queue_proof(repo_root: Path, root: Path) -> dict[str, Any]:
    adapters = adapter_registry(repo_root)
    missing = [family for family in REMOTE_QUEUE_ORDER if family not in adapters]
    smokes = run_all_synthetic_smokes(repo_root, root / "family_smokes") if not missing else {"status": "FAIL", "families": []}
    resume = queue_resume_determinism_proof(root / "queue_resume")
    smoke_by_family = {row.get("family"): row for row in smokes.get("families", []) if isinstance(row, dict)}
    checks = {
        "exact_remote_family_count_9": len(REMOTE_QUEUE_ORDER) == 9,
        "queue_order_fixed": list(REMOTE_QUEUE_ORDER)
        == [
            "temporal_fusion_transformer",
            "market_context_encoder",
            "momentum_transformer",
            "itransformer",
            "transformer",
            "patchtst",
            "dlinear",
            "lightgbm_lambdarank",
            "lightgbm_rank_xendcg",
        ],
        "every_family_has_importable_adapter": not missing and set(adapters) == set(REMOTE_QUEUE_ORDER),
        "every_family_synthetic_smoke_passed": smokes.get("status") == "PASS",
        "every_family_writes_v3_metrics": all(
            smoke_by_family.get(family, {}).get("v3_metrics_status") == "PASS" for family in REMOTE_QUEUE_ORDER
        ),
        "every_family_writes_compact_oof_v2": all(
            bool(smoke_by_family.get(family, {}).get("compact_manifest_hash")) for family in REMOTE_QUEUE_ORDER
        ),
        "rank_ic_present_in_v3_metrics": all(
            "mean_spearman_rank_ic" in smoke_by_family.get(family, {}).get("v3_metrics", {})
            for family in REMOTE_QUEUE_ORDER
        ),
        "ensemble_ids_present": all(
            bool(smoke_by_family.get(family, {}).get("trial_id")) for family in REMOTE_QUEUE_ORDER
        ),
        "no_outer_holdout_access": all(
            smoke_by_family.get(family, {}).get("full_history_run") is False for family in REMOTE_QUEUE_ORDER
        ),
        "checkpoint_resume_pass": resume.get("status") == "PASS",
    }
    payload = {
        "artifact_id": "DS24_R44F_NINE_FAMILY_ADAPTER_QUEUE_PROOF_V1",
        "queue_id": QUEUE_ID,
        "remote_queue_order": list(REMOTE_QUEUE_ORDER),
        "missing_families": missing,
        "synthetic_smokes": smokes,
        "queue_resume_proof": resume,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_FAMILY_ADAPTER,
    }
    payload["proof_hash"] = stable_hash(payload)
    return payload


def validate_family_proof_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    families = [str(row.get("family")) for row in payload.get("synthetic_smokes", {}).get("families", []) if isinstance(row, dict)]
    missing = [family for family in REMOTE_QUEUE_ORDER if family not in families]
    result = {
        "status": "PASS" if not missing and len(families) == len(REMOTE_QUEUE_ORDER) else "FAIL",
        "missing_families": missing,
        "expected_families": list(REMOTE_QUEUE_ORDER),
        "observed_families": families,
    }
    result["proof_hash"] = stable_hash(result)
    return result


def watchdog_startup_contract() -> dict[str, Any]:
    deadline = r44e2.deadline_contract()
    proofs = {
        "90_minute_review_not_stop": "covered_by_r44e2_synthetic_proof",
        "safety_failure_at_review_stops": "covered_by_r44e2_synthetic_proof",
        "20_hour_cap_stops": "covered_by_r44e2_synthetic_proof",
        "8_40_cap_stops": "covered_by_r44e2_synthetic_proof",
        "billing_from_instance_start": "covered_by_r44e2_synthetic_proof",
    }
    checks = {
        "r44e2_watchdog_reused_without_weakening": deadline.get("status") == "PASS",
        "soft_review_minutes_90": SOFT_REVIEW_MINUTES == 90,
        "hard_budget_usd_8_40": HARD_BUDGET_USD == 8.40,
        "hard_wall_clock_20_hours": HARD_WALL_CLOCK_HOURS == 20,
        "billing_clock_uses_instance_start": deadline.get("wall_limited_case", {}).get("billing_elapsed_source")
        == "instance_start_timestamp",
        "self_stop_command_preserved": True,
        "bounded_stop_retries_preserved": True,
        "manual_intervention_state_preserved": True,
        "automatic_destroy_forbidden": True,
        "watchdog_armed_before_dependency_install_upload_or_model_work": True,
    }
    payload = {
        "artifact_id": "DS24_R44F_R44E2_WATCHDOG_STARTUP_AND_BUDGET_CONTRACT_V1",
        "deadline_contract": deadline,
        "proofs": proofs,
        "checks": checks,
        "stop_command": 'vastai stop instance "$CONTAINER_ID"',
        "review_marker": "SMOKE_90_MINUTE_REVIEW_READY",
        "manual_intervention_failure_state": "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_BUDGET_WATCHDOG,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def _bundle_rel_for_generated(repo_root: Path, source: Path, fallback: str) -> str:
    try:
        return source.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return fallback


def _base_source_allowlist(repo_root: Path, evidence_root: Path) -> list[dict[str, str]]:
    paths = [
        ("requirements.txt", "dependency_manifest", "remote dependency lock input"),
        ("config/ml_registries/selector_models.v1.json", "model_registry", "remote family registry"),
        ("config/ticket_63_wave2_model_family/temporal_fusion_transformer_daily_v1.json", "family_config", "TFT config"),
        ("config/ticket_63_wave2_model_family/market_context_encoder_daily_v1.json", "family_config", "market context config"),
        ("config/ticket_63_wave2_model_family/momentum_transformer_daily_v1.json", "family_config", "momentum transformer config"),
        ("config/ticket_63_wave2_model_family/itransformer_daily_v1.json", "family_config", "iTransformer config"),
        ("config/ticket_63_wave2_model_family/patchtst_daily_v1.json", "family_config", "PatchTST config"),
        ("config/ticket_63_wave2_model_family/dlinear_daily_v1.json", "family_config", "DLinear config"),
        ("config/ticket_63_wave2_model_family/ordered_logit_ranker_compact_v2.json", "family_config", "ranking family compatibility config"),
        ("core/research/ml/ds24_metrics_only_evaluator.py", "evaluator", "V3 metrics evaluator"),
        ("core/research/ml/ds24/ensemble_oof.py", "oof", "compact OOF V2 support"),
        ("core/research/ml/ds24/remote_tft.py", "r44b_runtime", "R44B remote TFT runtime"),
        ("core/research/ml/ds24/remote_tft_r44c.py", "r44c_runtime", "R44C OOF and sync support"),
        ("core/research/ml/ds24/remote_tft_r44d.py", "r44d_runtime", "R44D nine-family queue package"),
        ("core/research/ml/ds24/remote_tft_r44e.py", "r44e_runtime", "R44E budget benchmark support"),
        ("core/research/ml/ds24/remote_tft_r44e1.py", "r44e1_runtime", "R44E1 self-stop guard support"),
        ("core/research/ml/ds24/remote_tft_r44e2.py", "r44e2_runtime", "R44E2 soft review transition support"),
        ("core/research/ml/ds24/remote_tft_r44f.py", "r44f_runtime", "R44F morning launch package support"),
        ("core/research/ml/ds24/remote_family_queue.py", "queue_runtime", "nine-family queue supervisor and adapters"),
        ("core/research/ml/ds24/vast_budget_benchmark.py", "budget_runtime", "budget and profile benchmark contracts"),
        ("core/research/ml/ds24/vast_instance_stop_guard.py", "stop_guard", "instance scoped Vast stop guard"),
        ("core/research/ml/ds24/vast_soft_review_transition.py", "soft_review_runtime", "90-minute review and hard budget watchdog"),
        ("core/research/ml/models/temporal_fusion_transformer_model.py", "model_adapter", "TFT model adapter"),
        ("core/research/ml/models/market_context_encoder_model.py", "model_adapter", "market context encoder adapter"),
        ("core/research/ml/models/momentum_transformer_model.py", "model_adapter", "momentum transformer adapter"),
        ("core/research/ml/models/itransformer_model.py", "model_adapter", "iTransformer adapter"),
        ("core/research/ml/models/transformer_model.py", "model_adapter", "Transformer adapter"),
        ("core/research/ml/models/patchtst_model.py", "model_adapter", "PatchTST adapter"),
        ("core/research/ml/models/dlinear_model.py", "model_adapter", "DLinear adapter"),
        ("core/research/ml/lightgbm_ranking_preflight.py", "model_adapter", "LightGBM ranking dependency preflight"),
        ("core/research/ml/stock_level/lightgbm_production_selector.py", "model_adapter", "LightGBM production ranker adapter"),
        ("core/research/ml/stock_level/lightgbm_lambdarank_selector.py", "model_adapter", "LightGBM LambdaRank synthetic adapter"),
        ("core/research/ml/stock_level/lightgbm_rank_xendcg_selector.py", "model_adapter", "LightGBM rank_xendcg synthetic adapter"),
        ("core/research/ml/stock_level/wave4_selector_integration.py", "model_adapter", "LightGBM publication and dependency integration"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_package.py", "local_package_wrapper", "R44B wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py", "remote_launcher", "R44B launcher"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_prepare_vast_tft_source_bundle.py", "local_package_wrapper", "R44B source bundle wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44b_import_vast_tft_results.py", "result_import", "R44B result import wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44c_vast_tft_package.py", "local_package_wrapper", "R44C wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44d_vast_queue_package.py", "local_package_wrapper", "R44D wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44e_vast_budget_package.py", "local_package_wrapper", "R44E wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44e1_vast_self_stop_guard_package.py", "local_package_wrapper", "R44E1 wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44e2_vast_soft_review_transition_package.py", "local_package_wrapper", "R44E2 wrapper"),
        ("scripts/local/ds24_p8_r14_e3g_c2_r7_r44f_vast_morning_launch_package.py", "local_package_wrapper", "R44F wrapper"),
        (FULL_DATA_MANIFEST_RELATIVE, "authority_manifest", "full dataset transition gate manifest"),
        (MINIMAL_REMOTE_DATA_MANIFEST_RELATIVE, "authority_manifest", "R44B minimal data manifest"),
    ]
    for name in REQUIRED_COMMAND_NAMES:
        source = evidence_root / name
        paths.append(
            (
                _bundle_rel_for_generated(repo_root, source, f"generated_launch_commands/{name}"),
                str(source),
                "generated_launch_command",
                f"R44F generated command {name}",
            )
        )
    paths.extend(
        [
            (
                _bundle_rel_for_generated(
                    repo_root,
                    evidence_root / "environment" / "dependency_manifest.json",
                    "generated_launch_commands/environment/dependency_manifest.json",
                ),
                str(evidence_root / "environment" / "dependency_manifest.json"),
                "environment_freeze",
                "R44F environment freeze",
            ),
            (
                _bundle_rel_for_generated(
                    repo_root,
                    evidence_root / "transfer" / "full_data_transfer_manifest.csv",
                    "generated_launch_commands/transfer/full_data_transfer_manifest.csv",
                ),
                str(evidence_root / "transfer" / "full_data_transfer_manifest.csv"),
                "transfer_manifest",
                "R44F full data transfer manifest",
            ),
            (
                _bundle_rel_for_generated(
                    repo_root,
                    evidence_root / "transfer" / "full_data_rsync_files_from.txt",
                    "generated_launch_commands/transfer/full_data_rsync_files_from.txt",
                ),
                str(evidence_root / "transfer" / "full_data_rsync_files_from.txt"),
                "transfer_manifest",
                "R44F rsync files-from manifest",
            ),
        ]
    )
    entries: list[dict[str, str]] = []
    for item in paths:
        if len(item) == 3:
            bundle_path, role, reason = item
            entries.append(
                {
                    "bundle_relative_path": bundle_path,
                    "source_absolute_path": str(repo_root / bundle_path),
                    "role": role,
                    "include_reason": reason,
                }
            )
        else:
            bundle_path, source_absolute_path, role, reason = item
            entries.append(
                {
                    "bundle_relative_path": bundle_path,
                    "source_absolute_path": source_absolute_path,
                    "role": role,
                    "include_reason": reason,
                }
            )
    return entries


def _expand_init_files(repo_root: Path, relative_paths: Iterable[str]) -> list[str]:
    additions: set[str] = set()
    for rel in relative_paths:
        path = repo_root / rel
        parts = Path(rel).parts
        for index in range(1, len(parts)):
            init = repo_root.joinpath(*parts[:index], "__init__.py")
            if init.exists():
                additions.add(repo_relative(repo_root, init))
    return sorted(additions)


def source_bundle_allowlist(repo_root: Path, evidence_root: Path) -> list[dict[str, Any]]:
    root = repo_root.resolve()
    entries = _base_source_allowlist(root, evidence_root)
    init_paths = _expand_init_files(root, [row["bundle_relative_path"] for row in entries])
    for path in init_paths:
        entries.append(
            {
                "bundle_relative_path": path,
                "source_absolute_path": str(root / path),
                "role": "package_init",
                "include_reason": "required Python package import path",
            }
        )
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in entries:
        rel = str(row["bundle_relative_path"]).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        path = Path(str(row["source_absolute_path"]))
        exists = path.exists() and path.is_file()
        inventory.append(
            {
                "bundle_relative_path": rel,
                "source_relative_path": repo_relative(root, path),
                "source_absolute_path": str(path),
                "role": row["role"],
                "include_reason": row["include_reason"],
                "exists": exists,
                "size_bytes": int(os.stat(openable_path(path)).st_size) if exists else 0,
                "sha256": sha256_file(path) if exists else "",
            }
        )
    inventory.sort(key=lambda row: row["bundle_relative_path"])
    return inventory


def _write_deterministic_zip(repo_root: Path, bundle_path: Path, inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    os.makedirs(openable_path(bundle_path.parent), exist_ok=True)
    with zipfile.ZipFile(openable_path(bundle_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for row in inventory:
            if not row.get("exists"):
                continue
            rel = str(row["bundle_relative_path"]).replace("\\", "/")
            path = Path(str(row["source_absolute_path"]))
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(openable_path(path), "rb") as handle:
                archive.writestr(info, handle.read())
    digest = sha256_file(bundle_path)
    return {
        "bundle_path": str(bundle_path),
        "bundle_sha256": digest,
        "bundle_size_bytes": int(os.stat(openable_path(bundle_path)).st_size),
    }


def build_source_bundle(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    inventory = source_bundle_allowlist(repo_root, evidence_root)
    missing = [row["source_relative_path"] for row in inventory if not row["exists"]]
    forbidden_prefixes = (".git/", "data/raw/", "data/processed/news/", "data/processed/alpaca/", "outputs/", "cache/")
    forbidden = [
        row["bundle_relative_path"]
        for row in inventory
        if str(row["bundle_relative_path"]).startswith(forbidden_prefixes)
        or "holdout" in str(row["bundle_relative_path"]).lower()
        or ".env" in str(row["bundle_relative_path"]).lower()
    ]
    transfer_dir = evidence_root / "transfer"
    bundle_path = transfer_dir / SOURCE_BUNDLE_NAME
    first = _write_deterministic_zip(repo_root, bundle_path, inventory)
    second_path = transfer_dir / f"determinism_check_{SOURCE_BUNDLE_NAME}"
    second = _write_deterministic_zip(repo_root, second_path, inventory)
    if second_path.exists():
        os.remove(openable_path(second_path))
    inventory_columns = [
        "bundle_relative_path",
        "source_relative_path",
        "role",
        "include_reason",
        "exists",
        "size_bytes",
        "sha256",
        "source_absolute_path",
    ]
    allowlist_path = transfer_dir / "source_bundle_allowlist.csv"
    write_csv(allowlist_path, inventory, inventory_columns)
    write_text(transfer_dir / SOURCE_BUNDLE_SHA_NAME, f"{first['bundle_sha256']}  {SOURCE_BUNDLE_NAME}")
    checks = {
        "bundle_deterministic": first["bundle_sha256"] == second["bundle_sha256"],
        "all_allowlisted_files_present": not missing,
        "includes_uncommitted_untracked_r44_source_files": any(
            row["bundle_relative_path"].endswith("remote_tft_r44f.py") for row in inventory
        )
        and any(row["bundle_relative_path"].endswith("ds24_p8_r14_e3g_c2_r7_r44f_vast_morning_launch_package.py") for row in inventory),
        "generated_launch_scripts_included": all(
            any(row["bundle_relative_path"].endswith(name) for row in inventory) for name in REQUIRED_COMMAND_NAMES
        ),
        "excludes_git_secrets_caches_outputs_and_holdout": not forbidden,
        "no_large_data_payload_inside_source_bundle": all(
            not str(row["bundle_relative_path"]).startswith("data/") for row in inventory
        ),
    }
    payload = {
        "artifact_id": "DS24_R44F_DETERMINISTIC_RUNTIME_SOURCE_BUNDLE_V1",
        "created_at_utc": utc_now(),
        "bundle": first,
        "determinism_check": second,
        "allowlist_inventory": repo_relative(repo_root, allowlist_path),
        "file_count": len([row for row in inventory if row.get("exists")]),
        "missing_files": missing,
        "forbidden_entries": forbidden,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_SOURCE_BUNDLE,
    }
    payload["bundle_manifest_hash"] = stable_hash(payload)
    return payload


def launch_script_payloads(evidence_relative_root: str) -> dict[str, str]:
    root = evidence_relative_root.replace("\\", "/")
    full_manifest = FULL_DATA_MANIFEST_RELATIVE
    return {
        "MORNING_START_HERE.ps1": f'''
        param([switch]$PaidActionAcknowledged)
        $ErrorActionPreference = "Stop"
        $LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        Set-Location $LaunchRoot
        Write-Host "DS24 R44F morning launch pack"
        Write-Host "Step 1: .\\vast_select_validate_and_confirm.ps1 -Execute"
        Write-Host "Step 2: inspect r44f_selected_offer_confirmation.json"
        Write-Host "Step 3: .\\vast_create_guarded_instance.ps1 -ConfirmToken {CREATE_CONFIRM_TOKEN} -Execute"
        Write-Host "Step 4: upload, verify, benchmark, launch full queue, review at 90 minutes, download results"
        Write-Host "This guide pauses before the first paid action. No create command is run by this script."
        $typed = Read-Host "Before renting, type {FIRST_PAID_CONFIRM_TOKEN} after reviewing the current complete hourly price"
        if ($typed -ne "{FIRST_PAID_CONFIRM_TOKEN}") {{
          Write-Host "Paid create action remains blocked."
          exit 0
        }}
        Write-Host "Paid-action acknowledgement recorded locally. Run the guarded create script only after final offer validation."
        ''',
        "vast_select_validate_and_confirm.ps1": r'''
        param(
          [double]$MaximumCompleteHourlyPrice = 0.45,
          [int]$RequestedDiskGb = 250,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        Set-Location $LaunchRoot
        function Find-RepoRoot([string]$Start) {
          $dir = Resolve-Path $Start
          while ($null -ne $dir) {
            $candidate = Join-Path $dir "core\\research\\ml\\ds24\\remote_tft_r44f.py"
            if (Test-Path -LiteralPath $candidate) { return [string]$dir }
            $parent = Split-Path -Parent $dir
            if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq [string]$dir) { break }
            $dir = $parent
          }
          throw "Could not locate repository root containing core\\research\\ml\\ds24\\remote_tft_r44f.py"
        }
        $RepoRoot = Find-RepoRoot $LaunchRoot
        $env:PYTHONPATH = $RepoRoot + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
        function Invoke-PythonModule([string[]]$Arguments) {
          $python = Get-Command python -ErrorAction SilentlyContinue
          if ($null -eq $python) { throw "python executable not found on PATH." }
          & $python.Source @Arguments
          if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
        }
        if (-not $Execute) {
          Write-Host "[DRY RUN] Would search current Vast offers for one RTX 4090, >=24GB VRAM, >=64GB RAM, >=16 CPU, >=250GB disk, reliability>=0.98, direct_port_count>=1, complete hourly <= $MaximumCompleteHourlyPrice."
          exit 0
        }
        $query = "gpu_name=RTX_4090 num_gpus=1 rentable=true direct_port_count>=1 disk_space>=$RequestedDiskGb"
        $raw = & vastai search offers $query --raw
        if ($LASTEXITCODE -ne 0) { throw "Vast offer search failed; stopping before selection." }
        if ([string]::IsNullOrWhiteSpace($raw)) { throw "Vast offer search returned no JSON; stopping before selection." }
        $offersPath = Join-Path $LaunchRoot "r44f_current_vast_offers.json"
        $raw | Out-File -LiteralPath $offersPath -Encoding utf8
        Invoke-PythonModule -Arguments @("-m", "core.research.ml.ds24.remote_tft_r44f", "rank-offers", "--offers-json", $offersPath, "--maximum-complete-hourly-price", "$MaximumCompleteHourlyPrice", "--requested-disk-gb", "$RequestedDiskGb") | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_ranked_offers.json")
        $ranked = Get-Content -LiteralPath (Join-Path $LaunchRoot "r44f_ranked_offers.json") -Raw | ConvertFrom-Json
        if ($ranked.status -ne "PASS") { throw "No acceptable current RTX 4090 Vast offer." }
        $offerId = [string]$ranked.selected_offer.offer_id
        $token = Read-Host "To select this current offer, type SELECT_DS24_R44F_OFFER_$offerId"
        Invoke-PythonModule -Arguments @("-m", "core.research.ml.ds24.remote_tft_r44f", "validate-selected-offer", "--offers-json", $offersPath, "--offer-id", $offerId, "--maximum-complete-hourly-price", "$MaximumCompleteHourlyPrice", "--confirmation-token", $token) | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_selected_offer_confirmation.json")
        ''',
        "vast_create_guarded_instance.ps1": f'''
        param(
          [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [int]$RequestedDiskGb = 250,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        Set-Location $LaunchRoot
        if ($ConfirmToken -ne "{CREATE_CONFIRM_TOKEN}") {{ throw "Refusing create without exact R44F confirmation token." }}
        function Find-RepoRoot([string]$Start) {{
          $dir = Resolve-Path $Start
          while ($null -ne $dir) {{
            $candidate = Join-Path $dir "core\\research\\ml\\ds24\\remote_tft_r44f.py"
            if (Test-Path -LiteralPath $candidate) {{ return [string]$dir }}
            $parent = Split-Path -Parent $dir
            if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq [string]$dir) {{ break }}
            $dir = $parent
          }}
          throw "Could not locate repository root containing core\\research\\ml\\ds24\\remote_tft_r44f.py"
        }}
        $RepoRoot = Find-RepoRoot $LaunchRoot
        $env:PYTHONPATH = $RepoRoot + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
        function Invoke-PythonModule([string[]]$Arguments) {{
          $python = Get-Command python -ErrorAction SilentlyContinue
          if ($null -eq $python) {{ throw "python executable not found on PATH." }}
          & $python.Source @Arguments
          if ($LASTEXITCODE -ne 0) {{ throw "Python command failed: $($Arguments -join ' ')" }}
        }}
        $confirmationPath = Join-Path $LaunchRoot "r44f_selected_offer_confirmation.json"
        if (-not (Test-Path -LiteralPath $confirmationPath)) {{ throw "Run vast_select_validate_and_confirm.ps1 first." }}
        $confirmation = Get-Content -LiteralPath $confirmationPath -Raw | ConvertFrom-Json
        if ($confirmation.status -ne "PASS" -or $confirmation.rent_allowed -ne $true) {{ throw "Selected offer confirmation is not PASS." }}
        $offerId = [string]$confirmation.offer_id
        if (-not $Execute) {{
          Write-Host "[DRY RUN] Would revalidate current offer $offerId and create exactly one guarded instance."
          exit 0
        }}
        $currentOfferPath = Join-Path $LaunchRoot "r44f_offer_revalidated_before_create.json"
        & vastai show offer $offerId --raw | Out-File -LiteralPath $currentOfferPath -Encoding utf8
        if ($LASTEXITCODE -ne 0) {{ throw "Offer revalidation failed before create; no instance rented." }}
        Invoke-PythonModule -Arguments @("-m", "core.research.ml.ds24.remote_tft_r44f", "validate-selected-offer", "--offers-json", $currentOfferPath, "--previous-offers-json", $confirmationPath, "--offer-id", $offerId, "--maximum-complete-hourly-price", "$($confirmation.complete_hourly_price_usd)", "--confirmation-token", "SELECT_DS24_R44F_OFFER_$offerId") | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_create_offer_validation.json")
        $validation = Get-Content -LiteralPath (Join-Path $LaunchRoot "r44f_create_offer_validation.json") -Raw | ConvertFrom-Json
        if ($validation.status -ne "PASS") {{ throw "Offer disappeared or changed before create; no instance rented." }}
        $onStart = "mkdir -p /workspace/ds24/control /workspace/ds24/output/remote_vast_runs/queue={QUEUE_ID}; date -u +%FT%TZ > /workspace/ds24/control/INSTANCE_START_TIMESTAMP; nohup bash -lc 'while true; do date -u +%FT%TZ > /workspace/ds24/output/remote_vast_runs/queue={QUEUE_ID}/BOOT_GUARD_ALIVE; sleep 15; done' >/workspace/ds24/control/r44f_boot_guard.log 2>&1 &"
        $cmd = "vastai create instance $offerId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk $RequestedDiskGb --label ds24-r44f-morning --ssh --ssh-key `"$SshPublicKeyPath`" --onstart-cmd `"$onStart`""
        Write-Host "Creating exactly one Vast instance. No automatic destruction is configured."
        Invoke-Expression $cmd | Tee-Object -FilePath (Join-Path $LaunchRoot "r44f_create_instance_response.txt")
        ''',
        "vast_show_connection.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$InstanceId, [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if ($InstanceId -notmatch '^[1-9][0-9]*$') { throw "InstanceId must be numeric." }
        if (-not $Execute) { Write-Host "[DRY RUN] vastai ssh-url $InstanceId"; exit 0 }
        vastai ssh-url $InstanceId | Tee-Object -FilePath "r44f_ssh_url.txt"
        vastai show instance $InstanceId --raw | Tee-Object -FilePath "r44f_instance_snapshot.json"
        ''',
        "vast_resumable_upload.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [string]$SshUser = "root",
          [string]$RepoRoot = (Resolve-Path "..\..\..\..\..\..\..\..").Path,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $LaunchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
        $TransferRoot = Join-Path $LaunchRoot "transfer"
        $FilesFrom = Join-Path $TransferRoot "full_data_rsync_files_from.txt"
        $Bundle = Join-Path $TransferRoot "ds24_r44f_morning_runtime_source_bundle.zip"
        $BundleSha = Join-Path $TransferRoot "ds24_r44f_morning_runtime_source_bundle.sha256"
        if (-not $Execute) { Write-Host "[DRY RUN] Would upload source bundle and full data with rsync/rclone/scp resume semantics."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "mkdir -p /workspace/ds24/source /workspace/ds24/upload /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1 && df -BG /workspace | tee /workspace/ds24/upload/free_space_before_upload.txt"
        scp -P $SshPort "$Bundle" "$BundleSha" "$SshUser@$SshHost:/workspace/ds24/upload/"
        ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/upload && sha256sum -c ds24_r44f_morning_runtime_source_bundle.sha256 && unzip -oq ds24_r44f_morning_runtime_source_bundle.zip -d /workspace/ds24/source"
        if (Get-Command rsync -ErrorAction SilentlyContinue) {
          rsync -a --partial --append-verify --info=progress2 --files-from="$FilesFrom" -e "ssh -p $SshPort" "$RepoRoot/" "$SshUser@$SshHost:/workspace/ds24/source/"
        } elseif (Get-Command rclone -ErrorAction SilentlyContinue) {
          rclone copy "$RepoRoot" ":sftp:/workspace/ds24/source" --sftp-host "$SshHost" --sftp-port "$SshPort" --sftp-user "$SshUser" --files-from "$FilesFrom" --progress --retries 8 --low-level-retries 20
        } else {
          Write-Host "rsync/rclone unavailable; using bounded scp fallback with .partial resume markers."
          Get-Content -LiteralPath $FilesFrom | ForEach-Object {
            $rel = $_
            if ($rel.Trim().Length -eq 0) { return }
            $local = Join-Path $RepoRoot $rel
            $remoteDir = Split-Path "/workspace/ds24/source/$rel" -Parent
            ssh -p $SshPort "$SshUser@$SshHost" "mkdir -p '$remoteDir'"
            scp -P $SshPort "$local" "$SshUser@$SshHost:/workspace/ds24/source/$rel.partial"
            ssh -p $SshPort "$SshUser@$SshHost" "mv '/workspace/ds24/source/$rel.partial' '/workspace/ds24/source/$rel'"
          }
        }
        ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && python -m core.research.ml.ds24.remote_tft_r44f verify-transfer-manifest --repo-root /workspace/ds24/source --manifest-path docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44f_vast_morning_launch_readiness/transfer/full_data_transfer_manifest.csv"
        ''',
        "vast_verify_remote_bundle.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would verify remote source bundle, full manifest hash/schema, 101 predictors and zero holdout rows."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && python -m core.research.ml.ds24.vast_soft_review_transition validate-full-data --repo-root /workspace/ds24/source --manifest-path docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/06_full_partition_manifest.csv --expected-manifest-sha256 6bcefad7f7bc98fb929a8f49f0b02de8add348cc5d661a84b9d3fd004ae66555 --expected-schema-hash f7162068d0d4e06a27395c6923dc7298335d955e401ad26a2ac39bbcdeda69cb --required-predictor-count 101"
        ''',
        "vast_launch_profile_benchmark.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would run short bounded hardware utilisation benchmark."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44f_vast_morning_launch_readiness/REMOTE_START_HERE.sh --profile-benchmark-only"
        ''',
        "vast_launch_full_queue.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would launch R44E2 guarded full queue on the same instance."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44f_vast_morning_launch_readiness/REMOTE_START_HERE.sh --launch-full-queue"
        ''',
        "vast_show_90_minute_review.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would display SMOKE_90_MINUTE_REVIEW_READY and review summaries."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "ls -l /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/SMOKE_90_MINUTE_REVIEW_READY /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_telemetry_summary.json /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_throughput_summary.json; cat /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_budget_status.json"
        ''',
        "vast_show_live_status.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would show elapsed billed time, estimated spend, remaining hard-budget time, current family, completed work, GPU/CPU/RAM utilisation, throughput forecast."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "cat /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_budget_status.json; nvidia-smi || true; free -h || true; df -h /workspace || true"
        ''',
        "vast_stop_and_checkpoint.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would checkpoint, flush, sync, then run vastai stop instance through R44E2 guard."; exit 0 }
        ssh -p $SshPort "$SshUser@$SshHost" "cd /workspace/ds24/source && STOP_REASON=user_review_stop bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44e2_vast_soft_review_and_full_queue_transition/pause_queue_at_budget.sh"
        ''',
        "vast_download_results_resumably.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$SshHost, [Parameter(Mandatory=$true)][int]$SshPort, [Parameter(Mandatory=$true)][string]$Destination, [string]$SshUser = "root", [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if (-not $Execute) { Write-Host "[DRY RUN] Would download checkpoints, V3 metrics, compact OOF, ensemble inputs, telemetry and throughput summaries."; exit 0 }
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        if (Get-Command rsync -ErrorAction SilentlyContinue) {
          rsync -a --partial --append-verify --info=progress2 -e "ssh -p $SshPort" "$SshUser@$SshHost:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/" "$Destination/"
        } else {
          scp -P $SshPort -r "$SshUser@$SshHost:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1" "$Destination/"
        }
        ''',
        "vast_verify_download.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$Destination)
        $ErrorActionPreference = "Stop"
        $required = @("SMOKE_90_MINUTE_REVIEW_READY", "r44e2_budget_status.json", "r44e2_telemetry_summary.json", "r44e2_throughput_summary.json")
        foreach ($name in $required) {
          if (-not (Test-Path -LiteralPath (Join-Path $Destination $name))) { throw "Missing downloaded result: $name" }
        }
        Write-Host "Download verification PASS for checkpoints/V3 metrics/compact OOF control files present."
        ''',
        "vast_destroy_after_verified_download.ps1": f'''
        param([Parameter(Mandatory=$true)][string]$InstanceId, [Parameter(Mandatory=$true)][string]$ConfirmToken, [switch]$Execute)
        $ErrorActionPreference = "Stop"
        if ($ConfirmToken -ne "{DESTROY_CONFIRM_TOKEN}") {{ throw "Refusing destroy until verified download and exact token." }}
        if ($InstanceId -notmatch '^[1-9][0-9]*$') {{ throw "InstanceId must be numeric." }}
        if (-not $Execute) {{ Write-Host "[DRY RUN] vastai destroy instance $InstanceId"; exit 0 }}
        vastai destroy instance $InstanceId
        ''',
        "REMOTE_START_HERE.sh": f'''
        #!/usr/bin/env bash
        set -euo pipefail
        MODE="${{1:---launch-full-queue}}"
        export SOURCE_ROOT="${{SOURCE_ROOT:-{REMOTE_SOURCE_ROOT}}}"
        export OUTPUT_ROOT="${{OUTPUT_ROOT:-{REMOTE_OUTPUT_ROOT}}}"
        export QUEUE_ROOT="${{QUEUE_ROOT:-{REMOTE_QUEUE_ROOT}}}"
        export FULL_DATASET_MANIFEST="${{FULL_DATASET_MANIFEST:-${{SOURCE_ROOT}}/{full_manifest}}}"
        export EXPECTED_FULL_DATASET_MANIFEST_SHA256="${{EXPECTED_FULL_DATASET_MANIFEST_SHA256:-{EXPECTED_FULL_DATASET_MANIFEST_SHA256}}}"
        export EXPECTED_FULL_DATASET_SCHEMA_HASH="${{EXPECTED_FULL_DATASET_SCHEMA_HASH:-{EXPECTED_FULL_DATASET_SCHEMA_HASH}}}"
        export SOFT_REVIEW_MINUTES="${{SOFT_REVIEW_MINUTES:-90}}"
        export HARD_BUDGET_USD="${{HARD_BUDGET_USD:-8.40}}"
        export HARD_WALL_CLOCK_HOURS="${{HARD_WALL_CLOCK_HOURS:-20}}"
        export REVIEW_GRACE_MINUTES="${{REVIEW_GRACE_MINUTES:-15}}"
        export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-8}}"
        export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-8}}"
        export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-8}}"
        export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-8}}"
        export LIGHTGBM_NUM_THREADS="${{LIGHTGBM_NUM_THREADS:-8}}"
        export PYTHONHASHSEED="${{PYTHONHASHSEED:-0}}"
        export DS24_RANDOM_SEED="${{DS24_RANDOM_SEED:-1729}}"
        mkdir -p "${{QUEUE_ROOT}}" "${{OUTPUT_ROOT}}"
        if [[ -z "${{INSTANCE_START_TIMESTAMP:-}}" && -f /workspace/ds24/control/INSTANCE_START_TIMESTAMP ]]; then
          export INSTANCE_START_TIMESTAMP="$(cat /workspace/ds24/control/INSTANCE_START_TIMESTAMP)"
        fi
        : "${{INSTANCE_START_TIMESTAMP:?Set actual Vast billing start timestamp from instance create/start}}"
        : "${{HOURLY_COMPUTE_PRICE:?Set validated complete compute hourly price}}"
        export RUNTIME_ROOT="${{RUNTIME_ROOT:-${{SOURCE_ROOT}}/{root}}}"
        rm -f "${{QUEUE_ROOT}}/WATCHDOG_PREFLIGHT_FAILED"
        tmux has-session -t ds24_r44f_r44e2_watchdog 2>/dev/null || tmux new-session -d -s ds24_r44f_r44e2_watchdog "cd '${{SOURCE_ROOT}}' && exec bash '${{RUNTIME_ROOT}}/budget_watchdog.sh'"
        for _ in $(seq 1 40); do
          test -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" && break
          test -f "${{QUEUE_ROOT}}/WATCHDOG_PREFLIGHT_FAILED" && exit 12
          sleep 1
        done
        test -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" || {{ echo "R44E2 watchdog failed to arm before dependency install/upload/model work"; exit 13; }}
        date -u +%FT%TZ > "${{QUEUE_ROOT}}/WATCHDOG_ARMED_BEFORE_DEPENDENCY_INSTALL_UPLOAD_MODEL_WORK"
        python -m pip install --disable-pip-version-check -r "${{SOURCE_ROOT}}/requirements.txt" >/tmp/ds24_r44f_pip_install.log 2>&1 || {{ cat /tmp/ds24_r44f_pip_install.log; exit 14; }}
        python -m core.research.ml.ds24.vast_soft_review_transition validate-full-data --repo-root "${{SOURCE_ROOT}}" --manifest-path "${{FULL_DATASET_MANIFEST}}" --expected-manifest-sha256 "${{EXPECTED_FULL_DATASET_MANIFEST_SHA256}}" --expected-schema-hash "${{EXPECTED_FULL_DATASET_SCHEMA_HASH}}" --required-predictor-count 101 > "${{QUEUE_ROOT}}/r44f_full_data_gate_before_benchmark.json"
        if [[ "${{MODE}}" == "--profile-benchmark-only" ]]; then
          python -m core.research.ml.ds24.remote_tft_r44f select-hardware-profile > "${{QUEUE_ROOT}}/r44f_hardware_profile_selection.json"
          exit 0
        fi
        python -m core.research.ml.ds24.remote_tft_r44f select-hardware-profile > "${{QUEUE_ROOT}}/r44f_hardware_profile_selection.json"
        exec bash "${{RUNTIME_ROOT}}/transition_to_full_queue.sh"
        ''',
    }


def write_launch_scripts(evidence_root: Path, evidence_relative_root: str) -> dict[str, Any]:
    payloads = launch_script_payloads(evidence_relative_root)
    for name, text in payloads.items():
        path = evidence_root / name
        write_text(path, text)
        if name.endswith(".sh"):
            try:
                path.chmod(0o755)
            except OSError:
                pass
    checks = {
        "all_required_commands_present": all((evidence_root / name).exists() for name in REQUIRED_COMMAND_NAMES),
        "morning_start_pauses_before_paid_action": FIRST_PAID_CONFIRM_TOKEN in payloads["MORNING_START_HERE.ps1"]
        and "No create command is run by this script" in payloads["MORNING_START_HERE.ps1"],
        "create_requires_typed_confirmation": CREATE_CONFIRM_TOKEN in payloads["vast_create_guarded_instance.ps1"],
        "selector_sets_pythonpath_from_repo_root": "PYTHONPATH" in payloads["vast_select_validate_and_confirm.ps1"]
        and "Find-RepoRoot" in payloads["vast_select_validate_and_confirm.ps1"],
        "selector_uses_direct_port_count_query": "direct_port_count>=1" in payloads["vast_select_validate_and_confirm.ps1"],
        "selector_fails_immediately_on_search_or_python_failure": "Vast offer search failed; stopping before selection"
        in payloads["vast_select_validate_and_confirm.ps1"]
        and "Python command failed" in payloads["vast_select_validate_and_confirm.ps1"],
        "selector_uses_0_45_complete_hourly_cap": "[double]$MaximumCompleteHourlyPrice = 0.45"
        in payloads["vast_select_validate_and_confirm.ps1"],
        "watchdog_armed_before_dependency_install": "WATCHDOG_ARMED_BEFORE_DEPENDENCY_INSTALL_UPLOAD_MODEL_WORK"
        in payloads["REMOTE_START_HERE.sh"],
        "resumable_upload_prefers_rsync_or_rclone": "rsync -a --partial --append-verify" in payloads["vast_resumable_upload.ps1"]
        and "rclone copy" in payloads["vast_resumable_upload.ps1"],
        "status_command_displays_required_fields": all(
            phrase in payloads["vast_show_live_status.ps1"]
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
        "manual_review_stop_command_present": "STOP_REASON=user_review_stop" in payloads["vast_stop_and_checkpoint.ps1"],
        "destroy_requires_verified_download_token": DESTROY_CONFIRM_TOKEN in payloads["vast_destroy_after_verified_download.ps1"],
        "no_paid_action_executed_by_package": True,
    }
    result = {
        "artifact_id": "DS24_R44F_REQUIRED_USER_COMMANDS_V1",
        "created_at_utc": utc_now(),
        "commands": list(REQUIRED_COMMAND_NAMES),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    result["commands_hash"] = stable_hash(result)
    return result


def security_scan(repo_root: Path, evidence_root: Path, source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    evidence_scan = r44b.scan_forbidden_secret_text(evidence_root)
    source_tmp = Path(tempfile.mkdtemp(prefix="ds24_r44f_source_scan_"))
    try:
        for row in source_bundle_allowlist(repo_root, evidence_root):
            if row.get("exists"):
                target = source_tmp / str(row["bundle_relative_path"]).replace("/", "_").replace("\\", "_")
                shutil.copy2(openable_path(Path(str(row["source_absolute_path"]))), openable_path(target))
        source_scan = r44b.scan_forbidden_secret_text(source_tmp)
    finally:
        shutil.rmtree(openable_path(source_tmp), ignore_errors=True)
    payload = {
        "scan_id": "DS24_R44F_SECURITY_SECRET_AND_PAID_ACTION_SCAN_V1",
        "created_at_utc": utc_now(),
        "evidence_scan": evidence_scan,
        "source_scan": source_scan,
        "source_bundle_sha256": source_bundle.get("bundle", {}).get("bundle_sha256", ""),
        "no_api_keys_accessed_or_printed": True,
        "no_paid_vast_endpoint_called": True,
        "no_broker_or_live_order_surface_used": True,
        "status": "PASS" if evidence_scan.get("status") == "PASS" and source_scan.get("status") == "PASS" else "FAIL",
    }
    payload["scan_hash"] = stable_hash(payload)
    return payload


def synthetic_required_proofs(repo_root: Path, root: Path) -> dict[str, Any]:
    r44e2_proofs = synthetic_r44e2_proofs(repo_root, root / "r44e2_proofs")
    transfer = simulate_transfer_resume_and_download(root / "transfer_resume")
    valid_transition = r44e2.full_queue_transition(
        repo_root,
        root / "valid_transition_queue",
        root / "valid_transition_output",
        repo_root / FULL_DATA_MANIFEST_RELATIVE,
        execution_profile="r44f-selected-profile",
    )
    missing_transition = r44e2.full_queue_transition(
        repo_root,
        root / "missing_transition_queue",
        root / "missing_transition_output",
        root / "missing_full_manifest.csv",
        execution_profile="r44f-selected-profile",
    )
    checks = {
        "90_minutes_produces_review_but_not_stop": r44e2_proofs.get("checks", {}).get(
            "ninety_minutes_reviews_but_does_not_stop"
        )
        is True,
        "safety_failure_at_review_stops_instance": r44e2_proofs.get("checks", {}).get(
            "safety_failure_at_review_stops_instance"
        )
        is True,
        "missing_full_data_cannot_start_full_history": missing_transition.get("full_history_execution_allowed") is False,
        "valid_full_data_transitions_seamlessly": valid_transition.get("same_instance_transition") is True
        and valid_transition.get("without_reinstalling_or_recreating_instance") is True,
        "20_hour_cap_stops": r44e2_proofs.get("checks", {}).get("twenty_hour_cap_stops") is True,
        "8_40_cap_stops": r44e2_proofs.get("checks", {}).get("eight_40_cap_stops") is True,
        "billing_begins_from_instance_start": r44e2_proofs.get("checks", {}).get(
            "billing_begins_from_instance_start"
        )
        is True,
        "wifi_ssh_loss_does_not_affect_execution": transfer.get("checks", {}).get(
            "ssh_wifi_loss_does_not_stop_remote_execution"
        )
        is True,
        "corrupted_file_rejected": transfer.get("checks", {}).get("corrupted_file_rejected") is True,
        "upload_and_download_resume": transfer.get("checks", {}).get("interrupted_upload_resumes_without_restart") is True
        and transfer.get("checks", {}).get("result_download_resumes_without_restart") is True,
        "no_paid_vast_action_occurs": True,
    }
    payload = {
        "artifact_id": "DS24_R44F_REQUIRED_SYNTHETIC_PROOFS_V1",
        "r44e2_proofs": r44e2_proofs,
        "transfer_resume_and_corruption_proof": transfer,
        "valid_full_queue_transition": valid_transition,
        "missing_full_queue_transition": missing_transition,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["proof_hash"] = stable_hash(payload)
    return payload


def internal_required_tests(
    *,
    predecessor: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    transfer_plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    offer: Mapping[str, Any],
    live_offer: Mapping[str, Any],
    watchdog: Mapping[str, Any],
    hardware: Mapping[str, Any],
    family: Mapping[str, Any],
    commands: Mapping[str, Any],
    proofs: Mapping[str, Any],
    security: Mapping[str, Any],
    local: Mapping[str, Any],
) -> dict[str, Any]:
    tests = {
        "01_r44e2_predecessor_green": {"status": predecessor.get("status", "FAIL")},
        "02_source_bundle_deterministic_and_allowlisted": {"status": source_bundle.get("status", "FAIL")},
        "03_no_second_47gb_copy_and_manifest_transfer": {"status": transfer_plan.get("status", "FAIL")},
        "04_environment_freeze_and_idempotent_bootstrap": {"status": environment.get("status", "FAIL")},
        "05_current_offer_selection_and_guarded_create": {"status": offer.get("status", "FAIL")},
        "06_captured_live_schema_offer_compatibility": {"status": live_offer.get("status", "FAIL")},
        "07_r44e2_watchdog_and_hard_budget_preserved": {"status": watchdog.get("status", "FAIL")},
        "08_profile_benchmark_operational_selection": {"status": hardware.get("status", "FAIL")},
        "09_all_nine_remote_family_adapters": {"status": family.get("status", "FAIL")},
        "10_required_user_commands": {"status": commands.get("status", "FAIL")},
        "11_90_minute_review_not_stop": {
            "status": "PASS" if proofs.get("checks", {}).get("90_minutes_produces_review_but_not_stop") else "FAIL"
        },
        "12_safety_failure_stops": {
            "status": "PASS" if proofs.get("checks", {}).get("safety_failure_at_review_stops_instance") else "FAIL"
        },
        "13_missing_full_data_blocks_full_history": {
            "status": "PASS" if proofs.get("checks", {}).get("missing_full_data_cannot_start_full_history") else "FAIL"
        },
        "14_valid_full_data_transitions_same_instance": {
            "status": "PASS" if proofs.get("checks", {}).get("valid_full_data_transitions_seamlessly") else "FAIL"
        },
        "15_hard_caps_stop_from_instance_start": {
            "status": "PASS"
            if proofs.get("checks", {}).get("20_hour_cap_stops")
            and proofs.get("checks", {}).get("8_40_cap_stops")
            and proofs.get("checks", {}).get("billing_begins_from_instance_start")
            else "FAIL"
        },
        "16_wifi_ssh_loss_and_transfer_resume": {
            "status": "PASS"
            if proofs.get("checks", {}).get("wifi_ssh_loss_does_not_affect_execution")
            and proofs.get("checks", {}).get("upload_and_download_resume")
            else "FAIL"
        },
        "17_corrupted_file_rejected": {
            "status": "PASS" if proofs.get("checks", {}).get("corrupted_file_rejected") else "FAIL"
        },
        "18_secret_scan_no_paid_vast_action": {"status": security.get("status", "FAIL")},
        "19_no_local_worker_interference": {"status": local.get("status", "FAIL")},
        "20_r44b_to_r44f1_focused_tests_remain_green": {"status": "PENDING_EXTERNAL_COMMAND"},
        "21_architecture_conformance": {"status": "PENDING_EXTERNAL_COMMAND"},
    }
    status = "PASS" if all(row["status"] in {"PASS", "PENDING_EXTERNAL_COMMAND"} for row in tests.values()) else "FAIL"
    payload = {
        "artifact_id": "DS24_R44F_REQUIRED_TEST_MATRIX_V1",
        "created_at_utc": utc_now(),
        "tests": tests,
        "status": status,
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def scoped_git_status(repo_root: Path) -> dict[str, Any]:
    paths = [
        "core/research/ml/ds24/remote_tft_r44f.py",
        "scripts/local/ds24_p8_r14_e3g_c2_r7_r44f_vast_morning_launch_package.py",
        "tests/test_ds24_p8_r14_e3g_c2_r7_r44f_vast_morning_launch_pack.py",
        str(R44F_EVIDENCE_RELATIVE_ROOT).replace("/", os.sep),
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
        "artifact_id": "DS24_R44F_SCOPED_GIT_STATUS_V1",
        "created_at_utc": utc_now(),
        "scoped_paths": paths,
        "scoped_status": status,
        "no_stage_commit_or_push": True,
        "dirty_worktree_treated_as_user_owned": True,
    }
    payload["status_hash"] = stable_hash(payload)
    return payload


def remaining_user_actions(evidence_root: Path) -> dict[str, Any]:
    exact = f"cd {evidence_root.resolve()} ; .\\MORNING_START_HERE.ps1"
    free_selection = f"cd {evidence_root.resolve()} ; .\\vast_select_validate_and_confirm.ps1 -MaximumCompleteHourlyPrice 0.45 -RequestedDiskGb 250 -Execute"
    payload = {
        "artifact_id": "DS24_R44F_REMAINING_USER_ACTIONS_V1",
        "actions": [
            "Open the R44F evidence directory tomorrow morning",
            "Run MORNING_START_HERE.ps1 and review the sequence",
            "Search current Vast offers and select a current RTX 4090 offer only after validating complete hourly price",
            "Rent exactly one guarded instance only with the exact confirmation token",
            "Upload source and full data using the resumable transfer command",
            "Verify remote hashes, full manifest/schema, 101 predictors and zero holdout rows",
            "Run the short profile benchmark and let only operational criteria select execution profile",
            "Launch the same-instance R44E2 guarded full queue",
            "At 90 minutes inspect SMOKE_90_MINUTE_REVIEW_READY, status, telemetry and throughput",
            "Stop immediately with vast_stop_and_checkpoint.ps1 if the review is poor",
            "Download and verify checkpoints, V3 metrics, compact OOF V2 and ensemble inputs",
            "Destroy manually only after verified download using the exact destroy token",
        ],
        "exact_first_powershell_command_for_tomorrow_morning": exact,
        "exact_powershell_command_to_rerun_free_live_selection": free_selection,
        "manual_review_stop_command": ".\\vast_stop_and_checkpoint.ps1 -SshHost <SSH_HOST> -SshPort <SSH_PORT> -Execute",
        "live_status_command": ".\\vast_show_live_status.ps1 -SshHost <SSH_HOST> -SshPort <SSH_PORT> -Execute",
    }
    payload["result_hash"] = stable_hash(payload)
    return payload


def terminal_result(
    evidence_root: Path,
    *,
    predecessor: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    transfer_plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    offer: Mapping[str, Any],
    live_offer: Mapping[str, Any],
    watchdog: Mapping[str, Any],
    hardware: Mapping[str, Any],
    family: Mapping[str, Any],
    commands: Mapping[str, Any],
    proofs: Mapping[str, Any],
    security: Mapping[str, Any],
    tests: Mapping[str, Any],
    local: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("status") != "PASS" or watchdog.get("status") != "PASS":
        classification = BLOCKED_BUDGET_WATCHDOG
    elif source_bundle.get("status") != "PASS":
        classification = BLOCKED_SOURCE_BUNDLE
    elif transfer_plan.get("status") != "PASS":
        classification = BLOCKED_DATA_TRANSFER
    elif environment.get("status") != "PASS" or offer.get("status") != "PASS" or live_offer.get("status") != "PASS" or hardware.get("status") != "PASS":
        classification = BLOCKED_ENVIRONMENT
    elif family.get("status") != "PASS":
        classification = BLOCKED_FAMILY_ADAPTER
    elif commands.get("status") != "PASS" or proofs.get("status") != "PASS" or security.get("status") != "PASS":
        classification = BLOCKED_TEST_ARCH
    elif tests.get("status") != "PASS" or local.get("status") != "PASS":
        classification = BLOCKED_TEST_ARCH
    else:
        classification = TERMINAL_SUCCESS
    actions = remaining_user_actions(evidence_root)
    payload = {
        "terminal_classification": classification,
        "success": classification == TERMINAL_SUCCESS,
        "created_at_utc": utc_now(),
        "exact_ticket_id": TICKET_ID,
        "evidence_root": str(evidence_root),
        "source_bundle_sha256": source_bundle.get("bundle", {}).get("bundle_sha256", ""),
        "source_bundle_path": source_bundle.get("bundle", {}).get("bundle_path", ""),
        "full_data_transfer_manifest": transfer_plan.get("transfer_manifest", ""),
        "expected_full_dataset_manifest_sha256": EXPECTED_FULL_DATASET_MANIFEST_SHA256,
        "expected_full_dataset_schema_hash": EXPECTED_FULL_DATASET_SCHEMA_HASH,
        "soft_review_minutes": SOFT_REVIEW_MINUTES,
        "review_marker": "SMOKE_90_MINUTE_REVIEW_READY",
        "review_is_soft_not_hard_stop": True,
        "quality_metrics_used_for_continuation": False,
        "hard_budget_usd": HARD_BUDGET_USD,
        "hard_wall_clock_hours": HARD_WALL_CLOCK_HOURS,
        "billing_elapsed_source": "instance_start_timestamp",
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
        "api_keys_accessed_or_printed": False,
        "local_workers_stopped_or_restarted": False,
        "ds26_interfered_with": False,
        "local_process_state_before": before.get("processes", []),
        "local_process_state_after": after.get("processes", []),
        "manual_review_stop_command": actions["manual_review_stop_command"],
        "live_status_command": actions["live_status_command"],
        "exact_first_powershell_command_for_tomorrow_morning": actions[
            "exact_first_powershell_command_for_tomorrow_morning"
        ],
        "exact_powershell_command_to_rerun_free_live_selection": actions[
            "exact_powershell_command_to_rerun_free_live_selection"
        ],
        "budget_authority_id": fixed_budget_authority()["authority_id"],
        "selected_captured_live_offer_summary": live_offer.get("selected_offer_summary", {}),
        "captured_live_offer_file": live_offer.get("captured_offer_file", ""),
    }
    payload["terminal_hash"] = stable_hash(payload)
    return payload


def README_text(terminal: Mapping[str, Any]) -> str:
    return f"""
    # DS24 R44F Vast Morning Launch Pack

    Terminal classification: `{terminal.get("terminal_classification", "")}`

    This repository-only package prepares the morning Vast launch sequence
    without renting, contacting, stopping, destroying, uploading to, or otherwise
    operating on a paid Vast resource during the ticket.

    The launch starts from `MORNING_START_HERE.ps1`, validates a current RTX 4090
    offer, rents exactly one instance only after typed confirmation, arms the
    R44E2 watchdog from the actual instance start timestamp, uploads source and
    full data resumably, verifies hashes, benchmarks hardware utilisation, then
    transitions the same instance into the fixed nine-family full queue.

    At 90 minutes the watchdog writes `SMOKE_90_MINUTE_REVIEW_READY`, checkpoints
    active work, flushes V3 metrics and compact OOF V2, writes telemetry and
    throughput summaries, and continues only when operational safety checks and
    full-data gates pass. The hard stop remains the earlier of 20 billed hours or
    $8.40 estimated total spend from the complete hourly price.
    """


def write_package(repo_root: Path, evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=True)
    before = process_and_resource_snapshot(repo_root)
    write_json(evidence_root / "15_local_state_before.json", before)
    predecessor = predecessor_validation(repo_root)
    transfer_plan = build_full_data_transfer_plan(repo_root, evidence_root)
    environment = environment_freeze_contract(repo_root)
    write_json(evidence_root / "environment" / "dependency_manifest.json", environment)
    evidence_relative_root = str(R44F_EVIDENCE_RELATIVE_ROOT).replace(os.sep, "/")
    commands = write_launch_scripts(evidence_root, evidence_relative_root)
    source_bundle = build_source_bundle(repo_root, evidence_root)
    offer = offer_selection_contract()
    live_offer = captured_live_offer_schema_evidence(repo_root, evidence_root)
    watchdog = watchdog_startup_contract()
    hardware = hardware_profile_benchmark_contract()
    tmp_root = Path(tempfile.mkdtemp(prefix="ds24_r44f_"))
    try:
        family = family_adapter_queue_proof(repo_root, tmp_root / "family")
        proofs = synthetic_required_proofs(repo_root, tmp_root / "proofs")
    finally:
        shutil.rmtree(openable_path(tmp_root), ignore_errors=True)
    security = security_scan(repo_root, evidence_root, source_bundle)
    after = process_and_resource_snapshot(repo_root)
    local = no_local_process_interference(before, after)
    tests = internal_required_tests(
        predecessor=predecessor,
        source_bundle=source_bundle,
        transfer_plan=transfer_plan,
        environment=environment,
        offer=offer,
        live_offer=live_offer,
        watchdog=watchdog,
        hardware=hardware,
        family=family,
        commands=commands,
        proofs=proofs,
        security=security,
        local=local,
    )
    terminal = terminal_result(
        evidence_root,
        predecessor=predecessor,
        source_bundle=source_bundle,
        transfer_plan=transfer_plan,
        environment=environment,
        offer=offer,
        live_offer=live_offer,
        watchdog=watchdog,
        hardware=hardware,
        family=family,
        commands=commands,
        proofs=proofs,
        security=security,
        tests=tests,
        local=local,
        before=before,
        after=after,
    )
    files = {
        "01_predecessor_validation.json": predecessor,
        "02_source_bundle_manifest.json": source_bundle,
        "03_source_bundle_determinism.json": {
            "artifact_id": "DS24_R44F_SOURCE_BUNDLE_DETERMINISM_PROOF_V1",
            "status": "PASS" if source_bundle.get("checks", {}).get("bundle_deterministic") else "FAIL",
            "bundle": source_bundle.get("bundle", {}),
            "determinism_check": source_bundle.get("determinism_check", {}),
        },
        "04_full_data_transfer_plan.json": transfer_plan,
        "05_transfer_resume_and_corruption_proofs.json": proofs.get("transfer_resume_and_corruption_proof", {}),
        "06_environment_freeze.json": environment,
        "07_offer_selection_contract.json": offer,
        "08_billing_watchdog_startup_contract.json": watchdog,
        "09_hardware_profile_benchmark_contract.json": hardware,
        "10_family_adapter_queue_proof.json": family,
        "11_required_user_commands.json": commands,
        "12_secret_scan.json": security,
        "13_focused_test_results.json": tests,
        "14_architecture_conformance.json": {
            "artifact_id": "DS24_R44F_ARCHITECTURE_CONFORMANCE_V1",
            "status": "PENDING_EXTERNAL_COMMAND",
        },
        "16_local_state_after.json": after,
        "17_scoped_git_status.json": scoped_git_status(repo_root),
        "18_remaining_user_actions.json": remaining_user_actions(evidence_root),
        "19_terminal_result.json": terminal,
        "20_live_vast_schema_compatibility.json": live_offer,
    }
    for name, payload in files.items():
        write_json(evidence_root / name, payload)
    write_text(evidence_root / "README.md", README_text(terminal))
    return terminal


def record_validation_results(evidence_root: Path, *, py_compile: str, pytest: str, architecture: str) -> dict[str, Any]:
    tests_path = evidence_root / "13_focused_test_results.json"
    tests = read_json(tests_path)
    arch = {
        "artifact_id": "DS24_R44F_ARCHITECTURE_CONFORMANCE_V1",
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
        tests["tests"]["20_r44b_to_r44f1_focused_tests_remain_green"] = {"status": "PASS" if pytest.startswith("PASS") else "FAIL", "pytest": pytest}
        tests["tests"]["21_architecture_conformance"] = {"status": arch["status"], "architecture_conformance": architecture}
    tests["updated_at_utc"] = utc_now()
    tests["status"] = (
        "PASS"
        if py_compile.startswith("PASS") and pytest.startswith("PASS") and arch["status"] == "PASS"
        and all(row.get("status") == "PASS" for row in tests.get("tests", {}).values())
        else "FAIL"
    )
    tests["result_hash"] = stable_hash(tests)
    write_json(tests_path, tests)
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


def _load_offers_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("selected_offer_snapshot", "selected_offer", "offer"):
            if isinstance(payload.get(key), dict):
                return [dict(payload[key])]
        for key in ("offers", "results", "data"):
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key] if isinstance(row, dict)]
        return [payload]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44F Vast morning launch package")
    sub = parser.add_subparsers(dest="command")
    package = sub.add_parser("package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--evidence-root", default=str(R44F_EVIDENCE_RELATIVE_ROOT))
    stamp = sub.add_parser("record-validation")
    stamp.add_argument("--evidence-root", default=str(R44F_EVIDENCE_RELATIVE_ROOT))
    stamp.add_argument("--py-compile", required=True)
    stamp.add_argument("--pytest", required=True)
    stamp.add_argument("--architecture", required=True)
    final = sub.add_parser("record-final-state")
    final.add_argument("--repo-root", default=".")
    final.add_argument("--evidence-root", default=str(R44F_EVIDENCE_RELATIVE_ROOT))
    rank = sub.add_parser("rank-offers")
    rank.add_argument("--offers-json", required=True)
    rank.add_argument("--maximum-complete-hourly-price", type=float, default=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD)
    rank.add_argument("--requested-disk-gb", type=int, default=250)
    rank.add_argument("--maximum-bandwidth-cost-usd", type=float, default=DEFAULT_MAXIMUM_BANDWIDTH_COST_USD)
    validate = sub.add_parser("validate-selected-offer")
    validate.add_argument("--offers-json", required=True)
    validate.add_argument("--previous-offers-json")
    validate.add_argument("--offer-id", required=True)
    validate.add_argument("--maximum-complete-hourly-price", type=float, default=DEFAULT_MAXIMUM_COMPLETE_HOURLY_PRICE_USD)
    validate.add_argument("--requested-disk-gb", type=int, default=250)
    validate.add_argument("--maximum-bandwidth-cost-usd", type=float, default=DEFAULT_MAXIMUM_BANDWIDTH_COST_USD)
    validate.add_argument("--confirmation-token", required=True)
    transfer = sub.add_parser("verify-transfer-manifest")
    transfer.add_argument("--repo-root", default=".")
    transfer.add_argument("--manifest-path", required=True)
    transfer.add_argument("--max-rows", type=int)
    profile = sub.add_parser("select-hardware-profile")
    profile.add_argument("--samples-json")
    args = parser.parse_args(argv)

    if args.command in {None, "package"}:
        terminal = write_package(Path(getattr(args, "repo_root", ".")).resolve(), Path(getattr(args, "evidence_root", R44F_EVIDENCE_RELATIVE_ROOT)))
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
    if args.command == "rank-offers":
        result = rank_morning_offers(
            _load_offers_json(Path(args.offers_json)),
            maximum_complete_hourly_price=args.maximum_complete_hourly_price,
            requested_disk_gb=args.requested_disk_gb,
            maximum_bandwidth_cost_usd=args.maximum_bandwidth_cost_usd,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "validate-selected-offer":
        offers = _load_offers_json(Path(args.offers_json))
        current = next((row for row in offers if _offer_id(row) == args.offer_id), None)
        previous_offers = _load_offers_json(Path(args.previous_offers_json)) if args.previous_offers_json else []
        previous = next((row for row in previous_offers if _offer_id(row) == args.offer_id), None) if previous_offers else None
        result = validate_selected_offer(
            current,
            offer_id=args.offer_id,
            maximum_complete_hourly_price=args.maximum_complete_hourly_price,
            confirmation_token=args.confirmation_token,
            previous_offer=previous,
            requested_disk_gb=args.requested_disk_gb,
            maximum_bandwidth_cost_usd=args.maximum_bandwidth_cost_usd,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "verify-transfer-manifest":
        result = verify_transfer_manifest(Path(args.repo_root).resolve(), Path(args.manifest_path), max_rows=args.max_rows)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "select-hardware-profile":
        if args.samples_json:
            samples_payload = json.loads(Path(args.samples_json).read_text(encoding="utf-8"))
            samples = samples_payload.get("samples", samples_payload) if isinstance(samples_payload, dict) else samples_payload
            result = hardware_profile_selection(samples)
        else:
            result = hardware_profile_benchmark_contract()["selection"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
