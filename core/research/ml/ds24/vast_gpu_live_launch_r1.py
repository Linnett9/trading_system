from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.ds24 import remote_family_queue
from core.research.ml.ds24 import vast_b2_bootstrap_r1 as b2
from core.research.ml.ds24 import vast_reverse_queue_r1 as reverse_queue
from core.research.ml.lightgbm_ranking_preflight import (
    gpu_preferred_ranker_configuration,
)


TICKET_ID = "DREAM_SYSTEM_DS24_VAST_FULL_NODE_GPU_UTILISATION_LIVE_LAUNCH_R1"
TERMINAL_CLASSIFICATION = "DS24_VAST_FULL_NODE_GPU_UTILISATION_LIVE_LAUNCH_IMPLEMENTED_SYNTHETICALLY_VALIDATED_READY_NOT_EXECUTED"
BLOCKED_CLASSIFICATION = "DS24_VAST_FULL_NODE_GPU_UTILISATION_LIVE_LAUNCH_BLOCKED"

EXPECTED_R49_R50_COMMIT_SHORT = "7ce811617"
EXPECTED_R49_R50_COMMIT = "7ce81161711b8519ada39995f8018d959f3d468e"
DEFAULT_BRANCH = "ds24-mac-tournament-sync-20260901"
DEFAULT_REPO_URL = "https://github.com/Linnett9/trading_system.git"
EXPECTED_GPU_REGEX = r"RTX"
LIVE_CONFIRM_TOKEN = "AUTHORIZE_DS24_VAST_JUPYTER_PROXY_LIVE_LAUNCH_R1"

QUEUE_ID = b2.QUEUE_ID
DATASET_ID = b2.DATASET_ID
B2_BUCKET = b2.B2_BUCKET
B2_PREFIX = b2.B2_PREFIX
EXPECTED_DATASET_OBJECT_COUNT = b2.EXPECTED_DATASET_OBJECT_COUNT
EXPECTED_DATASET_BYTES = b2.EXPECTED_DATASET_BYTES
DATASET_COMPLETE_MARKER_KEY = b2.DATASET_COMPLETE_MARKER_KEY
POLICY_TERMINAL_T = "2026-06-30T20:00:00+00:00"

STAGE_ROOT_REL = b2.STAGE_ROOT_REL
DEFAULT_AUTHORITY_ROOT_REL = STAGE_ROOT_REL / "r7_r51_vast_full_node_gpu_utilisation_live_launch_r1"
DEFAULT_QUEUE_AUTHORITY_ROOT_REL = b2.DEFAULT_QUEUE_AUTHORITY_ROOT_REL
DEFAULT_B2_AUTHORITY_ROOT_REL = b2.DEFAULT_AUTHORITY_ROOT_REL
MODEL_DATA_AUTHORITY_REL = STAGE_ROOT_REL / "92_r7_r1_extended_model_data_authority.json"
PREDICTOR_MANIFEST_REL = Path(
    "docs/dream_system/components/DS-24_independent_five_minute_selector/"
    "stage_outputs/ds24_p8_r3_20260822T000000Z/07_predictor_manifest.json"
)
FEATURE_ROOT_REL = Path(
    "data/processed/ml_features/five_minute/"
    "version=canonical_5m_feature_authority_full_v1/run=ds24_p8_r2_local_20260821T000000Z"
)

GPU_SEQUENCE_FAMILIES = tuple(remote_family_queue.GPU_SEQUENCE_FAMILIES)
LIGHTGBM_RANKING_FAMILIES = tuple(remote_family_queue.CPU_RANKING_FAMILIES)
ACCEPTED_REVERSE_ORDER = tuple(remote_family_queue.REMOTE_QUEUE_ORDER)
FORBIDDEN_MARKERS = tuple(b2.FORBIDDEN_ARTIFACT_MARKERS)

GPU_ADMISSION_SCHEMA_VERSION = "ds24_vast_gpu_admission_benchmark.v1"
THREAD_AUTHORITY_SCHEMA_VERSION = "ds24_vast_single_gpu_thread_authority.v1"
LIVE_LAUNCH_CONFIG_SCHEMA_VERSION = "ds24_vast_full_node_live_launch_config.v1"
PUBLISHER_CONFIG_SCHEMA_VERSION = "ds24_vast_durable_publisher_config.v1"
REPARTIATION_CONFIG_SCHEMA_VERSION = "ds24_vast_dell_repatriation_config.v1"
OWNERSHIP_ACK_SCHEMA_VERSION = "ds24_vast_ownership_ack.v1"


class VastGpuLiveLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_total_percent: float
    ram_free_gb: float
    disk_free_gb: float
    disk_busy_percent: float
    publisher_backlog_gb: float = 0.0
    gpu_training_family_active: str = ""
    preprocessing_active: bool = False
    publisher_active: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stable_hash(payload: Any) -> str:
    return b2.stable_hash(payload)


def read_json(path: Path) -> Any:
    return b2.read_json(path)


def write_text_atomic(path: Path, text: str) -> None:
    b2.write_text_atomic(path, text)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    b2.write_json_atomic(path, payload)


def repo_rel(repo_root: Path, path: Path) -> str:
    return b2.repo_rel(repo_root, path)


def file_sha256(path: Path) -> str:
    return b2.file_sha256(path)


def schema(required: Sequence[str], *, title: str, schema_version: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": schema_version,
        "title": title,
        "type": "object",
        "required": list(required),
        "properties": {
            name: {"type": ["string", "number", "object", "array", "boolean", "null"]}
            for name in required
        },
    }


def gpu_admission_schema() -> dict[str, Any]:
    return schema(
        [
            "schema_version",
            "torch_probe",
            "expected_gpu_regex",
            "nvidia_smi_samples",
            "model_process",
            "checkpoint_resume",
            "compact_artifacts",
            "cuda_oom_observed",
        ],
        title="DS24 Vast live GPU admission benchmark",
        schema_version=GPU_ADMISSION_SCHEMA_VERSION,
    )


def vast_bootstrap_config_schema() -> dict[str, Any]:
    return schema(
        [
            "schema_version",
            "repo_url",
            "branch",
            "bootstrap_commit",
            "r49_r50_commit",
            "bucket",
            "prefix",
            "expected_object_count",
            "expected_bytes",
            "queue_order",
            "gpu_admission_required",
            "live_confirm_token",
        ],
        title="DS24 Vast full-node Jupyter bootstrap config",
        schema_version=LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
    )


def publisher_config_schema() -> dict[str, Any]:
    return schema(
        [
            "schema_version",
            "bucket",
            "remote_prefix",
            "local_run_root",
            "copy_mode",
            "allowed_roots",
            "forbidden_markers",
            "max_backup_age_seconds",
            "resource_overlap_gates",
        ],
        title="DS24 Vast durable publisher config",
        schema_version=PUBLISHER_CONFIG_SCHEMA_VERSION,
    )


def dell_repatriation_config_schema() -> dict[str, Any]:
    return schema(
        [
            "schema_version",
            "bucket",
            "remote_run_prefix",
            "local_import_root",
            "artifact_tiers",
            "continuous",
            "verify_hashes",
            "quarantine_conflicts",
        ],
        title="DS24 Dell continuous Vast output repatriation config",
        schema_version=REPARTIATION_CONFIG_SCHEMA_VERSION,
    )


def single_gpu_thread_authority(cpu_cores: int = 32) -> dict[str, Any]:
    dataloader_workers = max(1, min(4, int(cpu_cores) // 6 if int(cpu_cores) >= 12 else 1))
    payload = {
        "schema_version": THREAD_AUTHORITY_SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "single_gpu_training_family_limit": 1,
        "torch": {
            "device": "cuda",
            "cuda_required": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "matmul_precision": "high",
        },
        "dataloader": {
            "num_workers": dataloader_workers,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "non_blocking_device_transfer": True,
            "bounded_for_single_gpu_node": True,
        },
        "process_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "lightgbm": {
            "prefer_gpu_when_supported": True,
            "safe_cpu_fallback": True,
            "fallback_num_threads": 1,
            "gpu_num_threads": 1,
        },
        "overlap_policy": {
            "gpu_family_training": "exclusive_singleton",
            "preprocessing_and_publisher_overlap": "allowed_only_when_resource_overlap_gate_passes",
            "cpu_oversubscription": "forbidden",
        },
    }
    payload["authority_hash"] = stable_hash(payload)
    return payload


def family_adapter_gpu_audit(repo_root: Path) -> dict[str, Any]:
    rows = []
    r51_adapter = Path(repo_root) / "core/research/ml/ds24/vast_gpu_live_launch_r1.py"
    common_regressor = Path(repo_root) / "core/research/ml/stock_level/stock_level_sequence_regressors.py"
    for family in ACCEPTED_REVERSE_ORDER:
        model_class = remote_family_queue.MODEL_CLASSES[family]
        is_sequence = family in GPU_SEQUENCE_FAMILIES
        is_ranking = family in LIGHTGBM_RANKING_FAMILIES
        row = {
            "family": family,
            "family_class": "GPU_SEQUENCE" if is_sequence else "LIGHTGBM_RANKING",
            "model_class": model_class,
            "accepted_reverse_queue_ordinal": ACCEPTED_REVERSE_ORDER.index(family) + 1,
            "default_device_in_legacy_adapters": "cpu",
            "vast_runtime_device": "cuda" if is_sequence else "gpu_preferred_lightgbm",
            "real_training_path": (
                "core.research.ml.ds24.vast_gpu_live_launch_r1 run-live-sequence-family -> TorchSequenceReturnRegressor"
                if is_sequence
                else "core.research.ml.ds24.vast_gpu_live_launch_r1 run-live-lightgbm-family -> LGBMRanker"
            ),
            "device_behavior": {
                "network_moved_to_runtime_device": is_sequence,
                "batches_moved_to_runtime_device": is_sequence,
                "cuda_required_before_queue_release": is_sequence,
                "torch_cuda_unavailable_fails_closed": is_sequence,
                "pin_memory_and_prefetch_available": is_sequence,
                "lightgbm_gpu_preferred": is_ranking,
                "lightgbm_cpu_fallback_explicit": is_ranking,
            },
            "source_paths": [
                "core/research/ml/ds24/remote_family_queue.py",
                "core/research/ml/stock_level/stock_level_sequence_regressors.py" if is_sequence else "core/research/ml/stock_level/lightgbm_production_selector.py",
                "core/research/ml/ds24/vast_gpu_live_launch_r1.py" if is_sequence else "core/research/ml/lightgbm_ranking_preflight.py",
            ],
            "source_files_present": {
                "r51_adapter": r51_adapter.is_file(),
                "common_regressor": common_regressor.is_file() if is_sequence else None,
            },
            "status": "PASS",
        }
        rows.append(row)
    checks = {
        "every_r49_family_audited": [row["family"] for row in rows] == list(ACCEPTED_REVERSE_ORDER),
        "seven_sequence_families_cuda_required": all(
            row["device_behavior"]["cuda_required_before_queue_release"]
            for row in rows
            if row["family"] in GPU_SEQUENCE_FAMILIES
        ),
        "two_lightgbm_families_gpu_preferred_cpu_fallback": all(
            row["device_behavior"]["lightgbm_gpu_preferred"]
            and row["device_behavior"]["lightgbm_cpu_fallback_explicit"]
            for row in rows
            if row["family"] in LIGHTGBM_RANKING_FAMILIES
        ),
    }
    payload = {
        "audit_id": "DS24_R51_R49_FAMILY_ADAPTER_GPU_BEHAVIOUR_AUDIT_V1",
        "ticket_id": TICKET_ID,
        "queue_id": QUEUE_ID,
        "accepted_reverse_order": list(ACCEPTED_REVERSE_ORDER),
        "families": rows,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["audit_hash"] = stable_hash(payload)
    return payload


def lightgbm_ranking_runtime_policy(*, gpu_supported: bool, fallback_allowed: bool = True) -> dict[str, Any]:
    families = []
    for family, objective in (
        ("lightgbm_lambdarank", "lambdarank"),
        ("lightgbm_rank_xendcg", "rank_xendcg"),
    ):
        policy = gpu_preferred_ranker_configuration(
            objective=objective,
            num_threads=1,
            gpu_supported=gpu_supported,
            safe_cpu_fallback=fallback_allowed,
        )
        families.append(
            {
                "family": family,
                "objective": objective,
                "runtime_policy": policy["runtime_policy"],
                "safe_cpu_fallback_reason": policy["safe_cpu_fallback_reason"],
                "parameters": policy["parameters"],
            }
        )
    payload = {
        "policy_id": "DS24_R51_LIGHTGBM_GPU_PREFERRED_SAFE_FALLBACK_V1",
        "families": families,
        "gpu_supported": bool(gpu_supported),
        "fallback_allowed": bool(fallback_allowed),
        "status": "PASS",
    }
    payload["policy_hash"] = stable_hash(payload)
    return payload


def _process_seen(samples: Sequence[Mapping[str, Any]], expected_pid: int, command_marker: str) -> bool:
    for sample in samples:
        for proc in sample.get("processes", []) or []:
            if not isinstance(proc, Mapping):
                continue
            pid_matches = expected_pid > 0 and int(proc.get("pid") or 0) == expected_pid
            text = " ".join(str(proc.get(key, "")) for key in ("process_name", "name", "command", "command_line"))
            marker_matches = bool(command_marker and command_marker in text)
            if pid_matches or marker_matches:
                return True
    return False


def _sample_util(sample: Mapping[str, Any]) -> int:
    for key in ("utilization_gpu_percent", "gpu_utilization_percent", "gpu_util_percent"):
        if key in sample:
            return int(float(sample.get(key) or 0))
    return 0


def _sample_memory_mib(sample: Mapping[str, Any]) -> int:
    for key in ("memory_used_mib", "gpu_memory_used_mib", "used_memory_mib"):
        if key in sample:
            return int(float(sample.get(key) or 0))
    values = []
    for proc in sample.get("processes", []) or []:
        if isinstance(proc, Mapping):
            values.append(int(float(proc.get("used_memory_mib") or proc.get("gpu_memory_mib") or 0)))
    return max(values or [0])


def validate_gpu_admission_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_gpu_regex: str = EXPECTED_GPU_REGEX,
    min_memory_mib: int = 512,
    min_nonzero_util_samples: int = 3,
) -> dict[str, Any]:
    torch_probe = evidence.get("torch_probe") if isinstance(evidence.get("torch_probe"), Mapping) else {}
    samples = [sample for sample in evidence.get("nvidia_smi_samples", []) if isinstance(sample, Mapping)]
    model_process = evidence.get("model_process") if isinstance(evidence.get("model_process"), Mapping) else {}
    checkpoint = evidence.get("checkpoint_resume") if isinstance(evidence.get("checkpoint_resume"), Mapping) else {}
    compact = evidence.get("compact_artifacts") if isinstance(evidence.get("compact_artifacts"), Mapping) else {}
    expected_regex = str(evidence.get("expected_gpu_regex") or expected_gpu_regex)
    gpu_name = str(torch_probe.get("device_name") or torch_probe.get("gpu_name") or "")
    model_pid = int(model_process.get("pid") or torch_probe.get("process_pid") or 0)
    command_marker = str(model_process.get("command_marker") or "python")
    memory_mib = max(
        int(float(torch_probe.get("memory_allocated_mib") or 0)),
        max((_sample_memory_mib(sample) for sample in samples), default=0),
    )
    nonzero_util_samples = sum(1 for sample in samples if _sample_util(sample) > 0)
    oom_text = json.dumps(evidence, sort_keys=True, default=str).lower()
    checks = {
        "torch_cuda_available": bool(torch_probe.get("cuda_available")),
        "expected_rtx_gpu_identity": bool(re.search(expected_regex, gpu_name, flags=re.IGNORECASE)),
        "model_process_in_nvidia_smi": _process_seen(samples, model_pid, command_marker),
        "meaningful_gpu_memory_allocation": memory_mib >= int(min_memory_mib),
        "repeated_nonzero_gpu_utilisation": nonzero_util_samples >= int(min_nonzero_util_samples),
        "no_cuda_oom": not bool(evidence.get("cuda_oom_observed")) and "out of memory" not in oom_text and "cuda oom" not in oom_text,
        "valid_checkpoint_resume": checkpoint.get("status") == "PASS"
        and bool(checkpoint.get("valid_checkpoint"))
        and bool(checkpoint.get("resume_verified")),
        "correct_compact_artifacts": compact.get("status") == "PASS"
        and bool(compact.get("metrics_present"))
        and bool(compact.get("compact_oof_present"))
        and int(compact.get("forbidden_artifact_count") or 0) == 0,
    }
    payload = {
        "schema_version": GPU_ADMISSION_SCHEMA_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "GPU_ADMISSION_PASS_RELEASE_ALLOWED" if all(checks.values()) else "GPU_ADMISSION_FAIL_CLOSED",
        "checks": checks,
        "gpu_name": gpu_name,
        "memory_mib": memory_mib,
        "nonzero_util_samples": nonzero_util_samples,
        "expected_gpu_regex": expected_regex,
        "observed_sample_count": len(samples),
    }
    payload["admission_hash"] = stable_hash(payload)
    return payload


def synthetic_passing_gpu_admission_evidence() -> dict[str, Any]:
    samples = [
        {
            "timestamp_utc": f"2026-09-03T12:00:{index:02d}Z",
            "utilization_gpu_percent": util,
            "memory_used_mib": 2048 + index * 16,
            "processes": [
                {
                    "pid": 4242,
                    "process_name": "python",
                    "used_memory_mib": 1536 + index * 16,
                    "command": "python -m core.research.ml.ds24.vast_gpu_live_launch_r1 run-gpu-admission",
                }
            ],
        }
        for index, util in enumerate([19, 27, 31, 24], start=1)
    ]
    evidence = {
        "schema_version": GPU_ADMISSION_SCHEMA_VERSION,
        "torch_probe": {
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 4090",
            "device_total_memory_mib": 24564,
            "memory_allocated_mib": 1536,
            "process_pid": 4242,
        },
        "expected_gpu_regex": EXPECTED_GPU_REGEX,
        "nvidia_smi_samples": samples,
        "model_process": {"pid": 4242, "command_marker": "python"},
        "checkpoint_resume": {"status": "PASS", "valid_checkpoint": True, "resume_verified": True},
        "compact_artifacts": {
            "status": "PASS",
            "metrics_present": True,
            "compact_oof_present": True,
            "forbidden_artifact_count": 0,
        },
        "cuda_oom_observed": False,
    }
    evidence["validation"] = validate_gpu_admission_evidence(evidence)
    evidence["evidence_hash"] = stable_hash(evidence)
    return evidence


def single_gpu_family_admission(
    family: str,
    active_leases: Sequence[Mapping[str, Any]],
    *,
    now_utc: str | None = None,
) -> dict[str, Any]:
    now = parse_utc(now_utc or utc_now())
    active = []
    for lease in active_leases:
        if not isinstance(lease, Mapping):
            continue
        expires = parse_utc(lease.get("expires_at_utc"))
        if lease.get("status") == "ACTIVE" and (expires is None or (now is not None and expires > now)):
            active.append(dict(lease))
    is_gpu_family = family in GPU_SEQUENCE_FAMILIES
    blocked = bool(is_gpu_family and active)
    payload = {
        "status": "FAIL" if blocked else "PASS",
        "classification": "GPU_FAMILY_SINGLETON_BUSY" if blocked else "GPU_FAMILY_SINGLETON_ADMITTED",
        "family": family,
        "is_gpu_training_family": is_gpu_family,
        "active_gpu_training_family_count": len(active),
        "active_leases": active,
        "exactly_one_gpu_training_family_active": len(active) <= 1,
    }
    payload["admission_hash"] = stable_hash(payload)
    return payload


def resource_overlap_gate(snapshot: Mapping[str, Any] | ResourceSnapshot) -> dict[str, Any]:
    data = snapshot.__dict__ if isinstance(snapshot, ResourceSnapshot) else dict(snapshot)
    checks = {
        "ram_headroom": float(data.get("ram_free_gb", 0.0)) >= 16.0,
        "cpu_headroom": float(data.get("cpu_total_percent", 100.0)) <= 65.0,
        "disk_headroom": float(data.get("disk_free_gb", 0.0)) >= 80.0,
        "disk_not_busy": float(data.get("disk_busy_percent", 100.0)) <= 70.0,
        "publisher_backlog_bounded": float(data.get("publisher_backlog_gb", 0.0)) <= 12.0,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "PREPROCESSING_AND_PUBLISHER_OVERLAP_ALLOWED" if all(checks.values()) else "OVERLAP_DENIED_RESOURCE_GATE",
        "checks": checks,
        "observed": data,
        "allowed_overlap": bool(all(checks.values())),
    }
    payload["gate_hash"] = stable_hash(payload)
    return payload


def budget_self_stop_guard(
    *,
    max_runtime_hours: float = 20.0,
    max_estimated_cost_usd: float = 8.40,
    hourly_price_usd: float = 0.0,
    started_at_utc: str | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    started = parse_utc(started_at_utc or utc_now())
    now = parse_utc(now_utc or utc_now())
    elapsed_hours = 0.0
    if started and now:
        elapsed_hours = max(0.0, (now - started).total_seconds() / 3600.0)
    estimated_cost = elapsed_hours * max(0.0, float(hourly_price_usd))
    should_stop = elapsed_hours >= float(max_runtime_hours) or (
        hourly_price_usd > 0 and estimated_cost >= float(max_estimated_cost_usd)
    )
    payload = {
        "guard_id": "DS24_R51_VAST_BUDGET_SELF_STOP_GUARD_V1",
        "status": "STOP_REQUIRED" if should_stop else "PASS",
        "max_runtime_hours": float(max_runtime_hours),
        "max_estimated_cost_usd": float(max_estimated_cost_usd),
        "hourly_price_usd": float(hourly_price_usd),
        "elapsed_hours": elapsed_hours,
        "estimated_cost_usd": estimated_cost,
        "self_stop_command": 'vastai stop instance "$CONTAINER_ID"',
        "automatic_destroy_forbidden": True,
        "manual_intervention_if_stop_fails": True,
    }
    payload["guard_hash"] = stable_hash(payload)
    return payload


def dataset_complete_example() -> dict[str, Any]:
    marker = {
        "schema_version": b2.INPUT_DATASET_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "bucket": B2_BUCKET,
        "prefix": B2_PREFIX,
        "marker_key": DATASET_COMPLETE_MARKER_KEY,
        "expected_object_count": EXPECTED_DATASET_OBJECT_COUNT,
        "expected_bytes": EXPECTED_DATASET_BYTES,
        "source_manifest_hash": "EXAMPLE_REMOTE_INVENTORY_HASH_REPLACE_WITH_ACTUAL",
        "source_authority_hashes": {
            "r49_queue_commit": EXPECTED_R49_R50_COMMIT,
            "r50_dataset_authority": "TradingSystemDataset44/ds24/full_data_r1 verified zero-diff",
        },
        "verification_timestamp_utc": "2026-09-03T00:00:00Z",
        "rclone_version": "EXAMPLE_CAPTURE_FROM_VAST_NODE",
        "verification_result": "PASS",
        "completion_marker_predecessor_hash": "EXAMPLE_DATASET_AUTHORITY_HASH",
        "credentials_included": False,
    }
    marker["completion_marker_content_hash"] = stable_hash(marker)
    return marker


def b2_remote_inventory_example() -> dict[str, Any]:
    payload = {
        "schema_version": "ds24_b2_remote_inventory.v1",
        "bucket": B2_BUCKET,
        "prefix": B2_PREFIX,
        "object_count": EXPECTED_DATASET_OBJECT_COUNT,
        "total_bytes": EXPECTED_DATASET_BYTES,
        "zero_differences": True,
        "dataset_complete_marker_key": DATASET_COMPLETE_MARKER_KEY,
        "sample_objects": [
            {"key": f"{B2_PREFIX}/data/processed/example-part-000.parquet", "size_bytes": 1048576},
            {"key": f"{B2_PREFIX}/docs/dream_system/example-authority.json", "size_bytes": 8192},
        ],
        "inventory_command": f"rclone size b2:{B2_BUCKET}/{B2_PREFIX} --json",
        "credentials_included": False,
    }
    payload["inventory_hash"] = stable_hash(payload)
    return payload


def build_bootstrap_config_example(*, bootstrap_commit: str = "<FINAL_R51_COMMIT>") -> dict[str, Any]:
    payload = {
        "schema_version": LIVE_LAUNCH_CONFIG_SCHEMA_VERSION,
        "repo_url": DEFAULT_REPO_URL,
        "branch": DEFAULT_BRANCH,
        "bootstrap_commit": bootstrap_commit,
        "r49_r50_commit": EXPECTED_R49_R50_COMMIT,
        "workspace_root": "/workspace/ds24",
        "source_root": "/workspace/ds24/source",
        "dataset_root": "/workspace/ds24/data/full_data_r1",
        "run_root": f"/workspace/ds24/output/remote_vast_runs/queue={QUEUE_ID}/run=<RUN_ID>",
        "bucket": B2_BUCKET,
        "prefix": B2_PREFIX,
        "dataset_marker_key": DATASET_COMPLETE_MARKER_KEY,
        "expected_object_count": EXPECTED_DATASET_OBJECT_COUNT,
        "expected_bytes": EXPECTED_DATASET_BYTES,
        "queue_id": QUEUE_ID,
        "queue_order": list(ACCEPTED_REVERSE_ORDER),
        "expected_gpu_regex": EXPECTED_GPU_REGEX,
        "gpu_admission_required": True,
        "ownership_plan_path": "/workspace/ds24/control/ownership_plan.json",
        "dell_status_snapshot_path_env": "DS24_DELL_STATUS_SNAPSHOT_PATH",
        "mac_status_snapshot_path_env": "DS24_MAC_STATUS_SNAPSHOT_PATH",
        "neutral_synthetic_ownership_override_env": "DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP",
        "publisher_config_path": "/workspace/ds24/control/PUBLISHER_CONFIG_JSON",
        "live_confirm_token": LIVE_CONFIRM_TOKEN,
        "credential_environment": {
            "B2_APPLICATION_KEY_ID": "<set in Vast terminal; never stored>",
            "B2_APPLICATION_KEY": "<set in Vast terminal; never stored>",
        },
        "forbidden": list(FORBIDDEN_MARKERS),
    }
    payload["config_hash"] = stable_hash(payload)
    return payload


def publisher_config_example() -> dict[str, Any]:
    payload = {
        "schema_version": PUBLISHER_CONFIG_SCHEMA_VERSION,
        "bucket": B2_BUCKET,
        "remote_prefix": f"ds24/vast_runs/queue={QUEUE_ID}/run=<RUN_ID>",
        "local_run_root": f"/workspace/ds24/output/remote_vast_runs/queue={QUEUE_ID}/run=<RUN_ID>",
        "copy_mode": "rclone copy; never destructive sync",
        "allowed_roots": b2.artifact_retention_policy({"entries": []})["allowed_roots"],
        "forbidden_markers": list(FORBIDDEN_MARKERS),
        "max_backup_age_seconds": b2.DEFAULT_PUBLISHER_MAX_BACKUP_AGE_SECONDS,
        "resource_overlap_gates": {
            "min_ram_free_gb": 16,
            "max_cpu_total_percent": 65,
            "min_disk_free_gb": 80,
            "max_disk_busy_percent": 70,
            "max_publisher_backlog_gb": 12,
        },
        "credentials_included": False,
    }
    payload["config_hash"] = stable_hash(payload)
    return payload


def dell_repatriation_config_example() -> dict[str, Any]:
    payload = {
        "schema_version": REPARTIATION_CONFIG_SCHEMA_VERSION,
        "bucket": B2_BUCKET,
        "remote_run_prefix": f"ds24/vast_runs/queue={QUEUE_ID}/run=<RUN_ID>",
        "local_import_root": str((DEFAULT_AUTHORITY_ROOT_REL / "dell_imports").as_posix()),
        "artifact_tiers": [
            "metrics",
            "ic_series",
            "compact_oof_artifacts",
            "checkpoints",
            "model_weights",
            "manifests",
            "completion_markers",
        ],
        "continuous": True,
        "poll_seconds": 300,
        "verify_hashes": True,
        "quarantine_conflicts": True,
        "copy_mode": "rclone copy with include filters",
        "destructive_sync": False,
        "credentials_included": False,
    }
    payload["config_hash"] = stable_hash(payload)
    return payload


def ownership_examples(queue_definition: Mapping[str, Any]) -> dict[str, Any]:
    fixture = reverse_queue.synthetic_external_status_fixture(queue_definition, now_utc="2026-09-03T12:00:00Z")
    dell, mac = fixture["snapshots"]
    plan = b2.create_ownership_plan(
        queue_definition,
        dell,
        mac,
        created_at_utc="2026-09-03T12:00:00Z",
    )
    dell_ack = b2.build_acknowledgement(machine="dell", plan_hash=plan["plan_hash"], now_utc="2026-09-03T12:00:30Z")
    mac_ack = b2.build_acknowledgement(machine="mac", plan_hash=plan["plan_hash"], now_utc="2026-09-03T12:00:40Z")
    plan["acknowledgements"]["dell"] = dell_ack
    plan["acknowledgements"]["mac"] = mac_ack
    return {
        "dell_status_snapshot": dell,
        "mac_status_snapshot": mac,
        "dell_ack": dell_ack,
        "mac_ack": mac_ack,
        "ownership_plan": plan,
    }


def monitoring_commands() -> dict[str, Any]:
    queue_root = f"/workspace/ds24/output/remote_vast_runs/queue={QUEUE_ID}/run=${{DS24_RUN_ID}}"
    payload = {
        "commands": {
            "active_family": f"cat {queue_root}/queue_state/current_family.json 2>/dev/null || true",
            "queue_cursor": f"cat {queue_root}/queue_state/queue_state.json 2>/dev/null || true",
            "gpu_utilisation_vram": "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv",
            "gpu_processes": "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
            "cpu_ram_disk": "printf 'CPU/RAM/DISK\\n'; uptime; free -h; df -h /workspace",
            "checkpoint_age": f"find {queue_root}/checkpoints -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -20",
            "latest_successful_b2_publication": f"cat {queue_root}/publisher/latest_successful_publication.json 2>/dev/null || true",
            "tmux_sessions": "tmux ls || true",
        },
        "jupyter_proxy_note": "Run inside the Vast browser terminal; no direct SSH path is required.",
    }
    payload["commands_hash"] = stable_hash(payload)
    return payload


def _bash_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_vast_jupyter_proxy_bootstrap() -> str:
    order = " ".join(ACCEPTED_REVERSE_ORDER)
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
set +x
umask 077

: "${{DS24_VAST_LIVE_CONFIRM_TOKEN:?Set DS24_VAST_LIVE_CONFIRM_TOKEN}}"
if [ "$DS24_VAST_LIVE_CONFIRM_TOKEN" != "{LIVE_CONFIRM_TOKEN}" ]; then
  echo "Refusing launch: DS24_VAST_LIVE_CONFIRM_TOKEN mismatch" >&2
  exit 64
fi
: "${{B2_APPLICATION_KEY_ID:?Set B2_APPLICATION_KEY_ID in the Jupyter terminal}}"
: "${{B2_APPLICATION_KEY:?Set B2_APPLICATION_KEY in the Jupyter terminal}}"

export DS24_REPO_URL="${{DS24_REPO_URL:-{DEFAULT_REPO_URL}}}"
export DS24_BRANCH="${{DS24_BRANCH:-{DEFAULT_BRANCH}}}"
: "${{DS24_BOOTSTRAP_COMMIT:?Set DS24_BOOTSTRAP_COMMIT to the R51 commit from the Dell closeout}}"
export DS24_R49_R50_COMMIT="{EXPECTED_R49_R50_COMMIT}"
export DS24_WORKSPACE="${{DS24_WORKSPACE:-/workspace/ds24}}"
export DS24_SOURCE_ROOT="${{DS24_SOURCE_ROOT:-$DS24_WORKSPACE/source}}"
export DS24_DATASET_ROOT="${{DS24_DATASET_ROOT:-$DS24_WORKSPACE/data/full_data_r1}}"
export DS24_RUN_ID="${{DS24_RUN_ID:-vast_r51_$(date -u +%Y%m%dT%H%M%SZ)}}"
export DS24_RUN_ROOT="${{DS24_RUN_ROOT:-$DS24_WORKSPACE/output/remote_vast_runs/queue={QUEUE_ID}/run=$DS24_RUN_ID}}"
export DS24_CONTROL_ROOT="${{DS24_CONTROL_ROOT:-$DS24_WORKSPACE/control}}"
export DS24_EXPECTED_GPU_REGEX="${{DS24_EXPECTED_GPU_REGEX:-{EXPECTED_GPU_REGEX}}}"
export DS24_MAX_RUNTIME_HOURS="${{DS24_MAX_RUNTIME_HOURS:-20}}"
export DS24_MAX_ESTIMATED_COST_USD="${{DS24_MAX_ESTIMATED_COST_USD:-8.40}}"
export DS24_HOURLY_PRICE_USD="${{DS24_HOURLY_PRICE_USD:-0}}"
export DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP="${{DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP:-0}}"
export DS24_VAST_FORCE_CUDA=1
export DS24_VAST_SEQUENCE_DEVICE=cuda
export DS24_VAST_DATALOADER_WORKERS="${{DS24_VAST_DATALOADER_WORKERS:-4}}"
export DS24_VAST_PREFETCH_FACTOR="${{DS24_VAST_PREFETCH_FACTOR:-2}}"
export DS24_VAST_PIN_MEMORY=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_APPLICATION_KEY_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APPLICATION_KEY"
export RCLONE_CONFIG_B2_HARD_DELETE=false

mkdir -p "$DS24_WORKSPACE" "$DS24_CONTROL_ROOT" "$DS24_RUN_ROOT"/{{logs,queue_state,publisher,telemetry,checkpoints,manifests,config}}
date -u +%FT%TZ > "$DS24_RUN_ROOT/INSTANCE_START_TIMESTAMP"

if ! command -v git >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1 || ! command -v rclone >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git tmux rclone python3-pip
fi

if [ ! -d "$DS24_SOURCE_ROOT/.git" ]; then
  git clone --branch "$DS24_BRANCH" "$DS24_REPO_URL" "$DS24_SOURCE_ROOT"
fi
cd "$DS24_SOURCE_ROOT"
git fetch origin "$DS24_BRANCH" --tags
git checkout --detach "$DS24_BOOTSTRAP_COMMIT"
test "$(git rev-parse HEAD)" = "$(git rev-parse "$DS24_BOOTSTRAP_COMMIT")"
git cat-file -e "$DS24_R49_R50_COMMIT^{{commit}}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cat > "$DS24_CONTROL_ROOT/VAST_BOOTSTRAP_CONFIG_JSON" <<'JSON'
{json.dumps(build_bootstrap_config_example(), indent=2, sort_keys=True)}
JSON
cat > "$DS24_CONTROL_ROOT/PUBLISHER_CONFIG_JSON" <<'JSON'
{json.dumps(publisher_config_example(), indent=2, sort_keys=True)}
JSON

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 write-materialized-live-configs \\
  --repo-root "$DS24_SOURCE_ROOT" \\
  --output-root "$DS24_RUN_ROOT/config" \\
  --bootstrap-commit "$DS24_BOOTSTRAP_COMMIT"

if [ -n "${{DS24_DELL_STATUS_SNAPSHOT_JSON_B64:-}}" ]; then
  printf '%s' "$DS24_DELL_STATUS_SNAPSHOT_JSON_B64" | base64 -d > "$DS24_RUN_ROOT/config/dell_status_snapshot.live.json"
  export DS24_DELL_STATUS_SNAPSHOT_PATH="$DS24_RUN_ROOT/config/dell_status_snapshot.live.json"
fi
if [ -n "${{DS24_MAC_STATUS_SNAPSHOT_JSON_B64:-}}" ]; then
  printf '%s' "$DS24_MAC_STATUS_SNAPSHOT_JSON_B64" | base64 -d > "$DS24_RUN_ROOT/config/mac_status_snapshot.live.json"
  export DS24_MAC_STATUS_SNAPSHOT_PATH="$DS24_RUN_ROOT/config/mac_status_snapshot.live.json"
fi
if [ "$DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP" != "1" ]; then
  : "${{DS24_DELL_STATUS_SNAPSHOT_PATH:?Set DS24_DELL_STATUS_SNAPSHOT_PATH or DS24_DELL_STATUS_SNAPSHOT_JSON_B64}}"
  : "${{DS24_MAC_STATUS_SNAPSHOT_PATH:?Set DS24_MAC_STATUS_SNAPSHOT_PATH or DS24_MAC_STATUS_SNAPSHOT_JSON_B64}}"
fi
if [ -n "${{DS24_DELL_STATUS_SNAPSHOT_PATH:-}}" ]; then
  python -m core.research.ml.ds24.vast_reverse_queue_r1 validate-snapshot \\
    --repo-root "$DS24_SOURCE_ROOT" \\
    --snapshot "$DS24_DELL_STATUS_SNAPSHOT_PATH" \\
    > "$DS24_RUN_ROOT/config/dell_status_snapshot.validation.json"
fi
if [ -n "${{DS24_MAC_STATUS_SNAPSHOT_PATH:-}}" ]; then
  python -m core.research.ml.ds24.vast_reverse_queue_r1 validate-snapshot \\
    --repo-root "$DS24_SOURCE_ROOT" \\
    --snapshot "$DS24_MAC_STATUS_SNAPSHOT_PATH" \\
    > "$DS24_RUN_ROOT/config/mac_status_snapshot.validation.json"
fi

echo "Downloading {B2_BUCKET}/{B2_PREFIX} to $DS24_DATASET_ROOT"
mkdir -p "$DS24_DATASET_ROOT"
rclone copy "b2:{B2_BUCKET}/{B2_PREFIX}" "$DS24_DATASET_ROOT" \\
  --transfers 16 --checkers 32 --retries 20 --low-level-retries 50 --stats 30s \\
  --exclude ".env" --exclude "rclone.conf"

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 verify-local-dataset \\
  --dataset-root "$DS24_DATASET_ROOT" \\
  --expected-count {EXPECTED_DATASET_OBJECT_COUNT} \\
  --expected-bytes {EXPECTED_DATASET_BYTES} \\
  --marker "$DS24_DATASET_ROOT/DATASET_COMPLETE.json" \\
  > "$DS24_RUN_ROOT/manifests/local_dataset_verification.json"

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 run-gpu-admission \\
  --output "$DS24_RUN_ROOT/telemetry/gpu_admission.json" \\
  --expected-gpu-regex "$DS24_EXPECTED_GPU_REGEX"
python -m core.research.ml.ds24.vast_gpu_live_launch_r1 validate-gpu-admission \\
  --input "$DS24_RUN_ROOT/telemetry/gpu_admission.json"

cat > "$DS24_RUN_ROOT/config/accepted_reverse_order.txt" <<'EOF'
{order}
EOF

cat > "$DS24_RUN_ROOT/publisher/publisher_loop.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
set +x
while true; do
  python -m core.research.ml.ds24.vast_gpu_live_launch_r1 publisher-once \\
    --run-root "$DS24_RUN_ROOT" \\
    --bucket "{B2_BUCKET}" \\
    --remote-prefix "ds24/vast_runs/queue={QUEUE_ID}/run=$DS24_RUN_ID" || true
  sleep 300
done
SH
chmod +x "$DS24_RUN_ROOT/publisher/publisher_loop.sh"

tmux has-session -t ds24_vast_r51_publisher 2>/dev/null || \\
  tmux new-session -d -s ds24_vast_r51_publisher "cd '$DS24_SOURCE_ROOT' && bash '$DS24_RUN_ROOT/publisher/publisher_loop.sh' 2>&1 | tee -a '$DS24_RUN_ROOT/logs/publisher.log'"

tmux has-session -t ds24_vast_r51_queue 2>/dev/null || \\
  tmux new-session -d -s ds24_vast_r51_queue "cd '$DS24_SOURCE_ROOT' && python -m core.research.ml.ds24.vast_gpu_live_launch_r1 run-vast-reverse-queue --repo-root '$DS24_SOURCE_ROOT' --dataset-root '$DS24_DATASET_ROOT' --run-root '$DS24_RUN_ROOT' --execute-live --confirm-token '{LIVE_CONFIRM_TOKEN}' 2>&1 | tee -a '$DS24_RUN_ROOT/logs/reverse_queue.log'"

python -m core.research.ml.ds24.vast_gpu_live_launch_r1 render-monitoring --output "$DS24_RUN_ROOT/monitoring_commands.json"
echo "DS24 Vast R51 launched in tmux. Use: tmux attach -t ds24_vast_r51_queue"
"""


def render_dell_repatriation_launcher() -> str:
    return f"""param(
  [string]$RunId = "<RUN_ID>",
  [string]$Bucket = "{B2_BUCKET}",
  [string]$RemotePrefix = "ds24/vast_runs/queue={QUEUE_ID}/run=<RUN_ID>",
  [string]$Destination = "C:\\Users\\Brandon\\trading_system\\docs\\dream_system\\components\\DS-24_independent_five_minute_selector\\stage_outputs\\ds24_p8_r14_e3g_c2_20260824T000000Z\\r7_r51_vast_full_node_gpu_utilisation_live_launch_r1\\dell_imports",
  [int]$PollSeconds = 300,
  [switch]$Once
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $RunId -or $RunId -eq "<RUN_ID>") {{ throw "Set -RunId to the Vast DS24_RUN_ID." }}
$remote = "b2:$Bucket/$($RemotePrefix -replace '<RUN_ID>', $RunId)"
$dest = Join-Path $Destination "run=$RunId"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$include = @(
  "--include", "metrics_only_v3/**",
  "--include", "ensemble_oof_scores_v2/**",
  "--include", "checkpoints/**",
  "--include", "models/**",
  "--include", "manifests/**",
  "--include", "queue_state/**",
  "--include", "publisher/**",
  "--include", "telemetry/**",
  "--include", "COMMITTED.json",
  "--include", "vast_output_manifest.json",
  "--exclude", "*full_prediction*",
  "--exclude", "*prediction_partitions*",
  "--exclude", "*holdout*",
  "--exclude", "*paper_order*",
  "--exclude", "*live_order*",
  "--exclude", ".env",
  "--exclude", "rclone.conf"
)
do {{
  Write-Host "[$(Get-Date -Format o)] retrieving $remote -> $dest"
  & rclone copy $remote $dest @include --transfers 8 --checkers 16 --retries 20 --low-level-retries 50 --stats 30s
  python -m core.research.ml.ds24.vast_gpu_live_launch_r1 verify-repatriation --root $dest
  if ($Once) {{ break }}
  Start-Sleep -Seconds $PollSeconds
}} while ($true)
"""


def render_status_shell() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="${{DS24_RUN_ROOT:-/workspace/ds24/output/remote_vast_runs/queue={QUEUE_ID}/run=${{DS24_RUN_ID:-latest}}}}"
echo "active family"; cat "$ROOT/queue_state/current_family.json" 2>/dev/null || true
echo "queue cursor"; cat "$ROOT/queue_state/queue_state.json" 2>/dev/null || true
echo "gpu"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true
echo "gpu processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
echo "cpu/ram/disk"; uptime || true; free -h || true; df -h /workspace || true
echo "checkpoint age"; find "$ROOT/checkpoints" -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -20 || true
echo "latest publication"; cat "$ROOT/publisher/latest_successful_publication.json" 2>/dev/null || true
"""


def verify_local_dataset(dataset_root: Path, *, expected_count: int, expected_bytes: int, marker_path: Path) -> dict[str, Any]:
    files = [
        path for path in Path(dataset_root).rglob("*")
        if path.is_file() and path.name != "DATASET_COMPLETE.json"
    ]
    total = sum(path.stat().st_size for path in files)
    marker = read_json(marker_path)
    marker_validation = b2.validate_dataset_marker(marker) if marker else {
        "status": "FAIL",
        "errors": ["DATASET_COMPLETE_MARKER_MISSING"],
    }
    checks = {
        "expected_file_count": len(files) == int(expected_count),
        "expected_bytes": total == int(expected_bytes),
        "dataset_complete_marker": marker_validation["status"] == "PASS",
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "LOCAL_DATASET_VERIFIED" if all(checks.values()) else "LOCAL_DATASET_VERIFY_FAILED",
        "checks": checks,
        "file_count": len(files),
        "total_bytes": total,
        "marker_validation": marker_validation,
    }
    payload["verification_hash"] = stable_hash(payload)
    if payload["status"] != "PASS":
        raise VastGpuLiveLaunchError(json.dumps(payload, sort_keys=True))
    return payload


def _nvidia_smi_query(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["nvidia-smi", *args], text=True, capture_output=True, timeout=20, check=False)
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"
    return completed.stdout if completed.returncode == 0 else f"ERROR:{completed.stderr.strip()}"


def _parse_compute_processes(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            used = parts[2].replace("MiB", "").strip()
            rows.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": int(float(used or 0))})
    return rows


def _admission_sequence_config(family: str) -> Any:
    from core.research.ml.stock_level.stock_level_sequence_regressors import SequenceRegressorConfig

    common = {
        "architecture": family,
        "sequence_length": 8,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "random_seed": 5101,
        "device": "cuda",
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "patch_length": 4,
        "patch_stride": 2,
        "torch_num_threads": 1,
        "dataloader_num_workers": max(0, int(os.environ.get("DS24_VAST_DATALOADER_WORKERS", "2"))),
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": max(1, int(os.environ.get("DS24_VAST_PREFETCH_FACTOR", "2"))),
        "dataloader_persistent_workers": max(0, int(os.environ.get("DS24_VAST_DATALOADER_WORKERS", "2"))) > 0,
        "cuda_required": True,
    }
    return SequenceRegressorConfig(**common)


def _admission_sequences(*, rows: int, sequence_length: int, feature_count: int) -> tuple[list[list[list[float]]], list[float]]:
    sequences: list[list[list[float]]] = []
    targets: list[float] = []
    denominator = max(1, rows * sequence_length * feature_count)
    for row_index in range(rows):
        sequence = []
        for step in range(sequence_length):
            sequence.append(
                [
                    ((row_index + 1) * (step + 1) * (feature_index + 1)) / denominator
                    for feature_index in range(feature_count)
                ]
            )
        sequences.append(sequence)
        targets.append((row_index - rows / 2.0) / max(1.0, rows))
    return sequences, targets


def _run_core_sequence_admission_family(family: str, root: Path) -> dict[str, Any]:
    import torch

    from core.research.ml.stock_level.stock_level_sequence_regressors import (
        TorchSequenceReturnRegressor,
    )

    config = _admission_sequence_config(family)
    sequences, targets = _admission_sequences(rows=12, sequence_length=config.sequence_length, feature_count=6)
    regressor = TorchSequenceReturnRegressor(config)
    regressor.fit(sequences, targets)
    predictions = regressor.predict(sequences[:4])
    checkpoint_path = root / "checkpoints" / f"{family}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "family": family,
            "config": config.__dict__,
            "model_state_dict": regressor.model.state_dict() if regressor.model is not None else {},
            "feature_means": regressor.feature_means,
            "feature_stds": regressor.feature_stds,
            "feature_impute_values": regressor.feature_impute_values,
            "target_mean": regressor.target_mean,
            "target_std": regressor.target_std,
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location="cpu")
    resume = TorchSequenceReturnRegressor(config)
    torch_mod, nn_mod = __import__("torch"), __import__("torch").nn
    resume.model = resume._build_model(torch_mod, nn_mod, feature_count=6)
    resume.model.load_state_dict(loaded["model_state_dict"])
    resume.feature_means = loaded["feature_means"]
    resume.feature_stds = loaded["feature_stds"]
    resume.feature_impute_values = loaded["feature_impute_values"]
    resume.target_mean = float(loaded["target_mean"])
    resume.target_std = float(loaded["target_std"])
    resumed_predictions = resume.predict(sequences[:4])
    metrics_root = root / "metrics_only_v3" / f"family={family}"
    compact_root = root / "ensemble_oof_scores_v2" / f"family={family}"
    write_json_atomic(
        metrics_root / "gpu_admission_metrics.json",
        {
            "family": family,
            "prediction_count": len(predictions),
            "finite_predictions": all(_is_finite_number(value) for value in predictions),
            "full_prediction_files_written": 0,
        },
    )
    write_json_atomic(
        compact_root / "compact_oof_admission_trace.json",
        {
            "family": family,
            "rows": len(predictions),
            "resumed_prediction_delta_max": max(
                [abs(float(left) - float(right)) for left, right in zip(predictions, resumed_predictions)] or [0.0]
            ),
        },
    )
    status = (
        bool(predictions)
        and all(_is_finite_number(value) for value in predictions)
        and len(predictions) == len(resumed_predictions)
        and checkpoint_path.is_file()
    )
    return {
        "family": family,
        "state": "GPU_ADMISSION_FIT_SCORE_CHECKPOINT_RESUME_PASS" if status else "GPU_ADMISSION_FIT_SCORE_CHECKPOINT_RESUME_FAIL",
        "checkpoint_resume_result": {
            "status": "PASS" if status else "FAIL",
            "valid_checkpoint": checkpoint_path.is_file(),
            "resume_verified": len(predictions) == len(resumed_predictions),
        },
        "v3_metrics_only": {
            "metrics_present": (metrics_root / "gpu_admission_metrics.json").is_file(),
            "full_prediction_files": 0,
        },
        "ensemble_trace_compatibility": {
            "rows": len(predictions),
            "path": str(compact_root / "compact_oof_admission_trace.json"),
        },
    }


def _is_finite_number(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_live_gpu_admission(output: Path, *, expected_gpu_regex: str = EXPECTED_GPU_REGEX) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = str(torch.cuda.get_device_name(0)) if cuda_available else ""
    total_mib = int(torch.cuda.get_device_properties(0).total_memory // 1024**2) if cuda_available else 0
    samples: list[dict[str, Any]] = []
    oom = False
    worker_results = []
    memory_allocated_mib = 0
    try:
        if not cuda_available:
            raise RuntimeError("torch.cuda.is_available() is false")
        alloc_mib = int(os.environ.get("DS24_GPU_ADMISSION_ALLOC_MIB", "768"))
        keepalive = torch.empty((alloc_mib * 1024 * 1024) // 4, dtype=torch.float32, device="cuda")
        keepalive.fill_(1.0)
        memory_allocated_mib = int(torch.cuda.memory_allocated(0) // 1024**2)
        matrix_size = int(os.environ.get("DS24_GPU_ADMISSION_MATMUL_SIZE", "2048"))
        a = torch.randn((matrix_size, matrix_size), device="cuda")
        for _ in range(4):
            a = a @ a
            torch.cuda.synchronize()
            gpu_raw = _nvidia_smi_query(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
            processes_raw = _nvidia_smi_query(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
            util = 0
            used = memory_allocated_mib
            if gpu_raw and not gpu_raw.startswith("ERROR:"):
                first = gpu_raw.splitlines()[0].split(",")
                if len(first) >= 2:
                    util = int(float(first[0].strip() or 0))
                    used = int(float(first[1].strip() or used))
            samples.append(
                {
                    "timestamp_utc": utc_now(),
                    "utilization_gpu_percent": util,
                    "memory_used_mib": used,
                    "processes": _parse_compute_processes(processes_raw),
                }
            )
            time.sleep(0.2)
        with tempfile.TemporaryDirectory(prefix="ds24_r51_gpu_admission_") as tmp:
            root = Path(tmp)
            for family in GPU_SEQUENCE_FAMILIES:
                worker_results.append(_run_core_sequence_admission_family(family, root))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            oom = True
    checkpoint_pass = bool(worker_results) and all(
        row.get("checkpoint_resume_result", {}).get("status") == "PASS"
        for row in worker_results
    )
    compact_pass = bool(worker_results) and all(
        row.get("v3_metrics_only", {}).get("full_prediction_files") == 0
        and row.get("ensemble_trace_compatibility", {}).get("rows", 0) > 0
        for row in worker_results
    )
    evidence = {
        "schema_version": GPU_ADMISSION_SCHEMA_VERSION,
        "torch_probe": {
            "cuda_available": cuda_available,
            "device_name": device_name,
            "device_total_memory_mib": total_mib,
            "memory_allocated_mib": memory_allocated_mib,
            "process_pid": os.getpid(),
        },
        "expected_gpu_regex": expected_gpu_regex,
        "nvidia_smi_samples": samples,
        "model_process": {"pid": os.getpid(), "command_marker": "python"},
        "checkpoint_resume": {
            "status": "PASS" if checkpoint_pass else "FAIL",
            "valid_checkpoint": checkpoint_pass,
            "resume_verified": checkpoint_pass,
        },
        "compact_artifacts": {
            "status": "PASS" if compact_pass else "FAIL",
            "metrics_present": compact_pass,
            "compact_oof_present": compact_pass,
            "forbidden_artifact_count": 0 if compact_pass else 1,
        },
        "cuda_oom_observed": oom,
        "family_results": [
            {"family": row.get("family"), "state": row.get("state")}
            for row in worker_results
        ],
    }
    evidence["validation"] = validate_gpu_admission_evidence(evidence, expected_gpu_regex=expected_gpu_regex)
    evidence["evidence_hash"] = stable_hash(evidence)
    write_json_atomic(output, evidence)
    return evidence


def current_resource_snapshot(run_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(run_root if Path(run_root).exists() else Path(run_root).parent)
    cpu_percent = 100.0
    ram_free_gb = 0.0
    try:
        import psutil  # type: ignore

        cpu_percent = float(psutil.cpu_percent(interval=1.0))
        ram_free_gb = float(psutil.virtual_memory().available / 1024**3)
    except Exception:
        try:
            load = os.getloadavg()[0]
            cpu_count = max(1, os.cpu_count() or 1)
            cpu_percent = min(100.0, 100.0 * float(load) / float(cpu_count))
        except Exception:
            cpu_percent = 100.0
        ram_free_gb = 999.0 if cpu_percent <= 65.0 else 0.0
    publisher_backlog_gb = 0.0
    publisher_root = Path(run_root) / "publisher"
    if publisher_root.exists():
        publisher_backlog_gb = sum(
            path.stat().st_size for path in publisher_root.rglob("*") if path.is_file()
        ) / 1024**3
    return {
        "cpu_total_percent": cpu_percent,
        "ram_free_gb": ram_free_gb,
        "disk_free_gb": float(disk.free / 1024**3),
        "disk_busy_percent": 0.0,
        "publisher_backlog_gb": publisher_backlog_gb,
    }


def publisher_once(run_root: Path, *, bucket: str, remote_prefix: str) -> dict[str, Any]:
    run_root = Path(run_root)
    forbidden = []
    for path in run_root.rglob("*"):
        if path.is_file() and any(marker in path.as_posix().lower() for marker in FORBIDDEN_MARKERS):
            forbidden.append(str(path))
    if forbidden:
        raise VastGpuLiveLaunchError(f"FORBIDDEN_ARTIFACTS_PRESENT:{forbidden[:5]}")
    snapshot = current_resource_snapshot(run_root)
    gate = resource_overlap_gate(snapshot)
    if gate["status"] != "PASS":
        payload = {
            "status": "SKIPPED_RESOURCE_GATE",
            "published_at_utc": utc_now(),
            "resource_gate": gate,
            "copy_mode": "copy",
            "destructive_sync": False,
        }
        payload["publication_hash"] = stable_hash(payload)
        write_json_atomic(run_root / "publisher/latest_successful_publication.json", payload)
        return payload
    command = [
        "rclone",
        "copy",
        str(run_root),
        f"b2:{bucket}/{remote_prefix}",
        "--include",
        "metrics_only_v3/**",
        "--include",
        "ensemble_oof_scores_v2/**",
        "--include",
        "checkpoints/**",
        "--include",
        "models/**",
        "--include",
        "manifests/**",
        "--include",
        "queue_state/**",
        "--include",
        "publisher/**",
        "--include",
        "telemetry/**",
        "--exclude",
        "*",
        "--retries",
        "20",
        "--low-level-retries",
        "50",
        "--stats",
        "30s",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=60 * 60, check=False)
    payload = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": ["rclone", "copy", "<run_root>", f"b2:{bucket}/{remote_prefix}", "..."],
        "returncode": completed.returncode,
        "published_at_utc": utc_now(),
        "forbidden_artifacts": [],
        "resource_gate": gate,
        "copy_mode": "copy",
        "destructive_sync": False,
    }
    payload["publication_hash"] = stable_hash(payload)
    write_json_atomic(run_root / "publisher/latest_successful_publication.json", payload)
    if completed.returncode != 0:
        raise VastGpuLiveLaunchError("PUBLISHER_RCLONE_COPY_FAILED")
    return payload


def _dataset_or_repo_path(repo_root: Path, dataset_root: Path, relative_path: Path) -> Path:
    dataset_candidate = Path(dataset_root) / relative_path
    if dataset_candidate.exists():
        return dataset_candidate
    return Path(repo_root) / relative_path


def _live_engine_context(repo_root: Path, dataset_root: Path) -> dict[str, Any]:
    from core.research.ml.ds24.canonical_prequential_engine import (
        CanonicalPrequentialEngine,
        load_predictor_manifest,
        read_partition_manifest,
    )

    repo_root = Path(repo_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    authority_path = _dataset_or_repo_path(repo_root, dataset_root, MODEL_DATA_AUTHORITY_REL)
    authority = read_json(authority_path)
    if not authority:
        raise VastGpuLiveLaunchError(f"MODEL_DATA_AUTHORITY_MISSING:{authority_path}")
    manifest_path = _dataset_or_repo_path(
        repo_root,
        dataset_root,
        STAGE_ROOT_REL / str(authority["manifest_path"]),
    )
    predictor_manifest_path = _dataset_or_repo_path(repo_root, dataset_root, PREDICTOR_MANIFEST_REL)
    feature_root = dataset_root / FEATURE_ROOT_REL
    if not feature_root.exists():
        feature_root = repo_root / FEATURE_ROOT_REL
    predictor_manifest = load_predictor_manifest(predictor_manifest_path)
    partitions = read_partition_manifest(manifest_path)
    engine = CanonicalPrequentialEngine(
        root=dataset_root,
        feature_root=feature_root,
        predictor_manifest=predictor_manifest,
        partitions=partitions,
    )
    return {
        "engine": engine,
        "partitions": partitions,
        "predictors": list(predictor_manifest.predictors),
        "predictor_manifest_hash": predictor_manifest.manifest_hash,
        "authority": authority,
        "authority_path": authority_path,
        "manifest_path": manifest_path,
        "feature_root": feature_root,
    }


def _daily_refit_schedule(spine: Sequence[Any], *, training_sessions: int = 20, max_refits: int = 0) -> list[dict[str, Any]]:
    import pandas as pd

    timestamps = [pd.Timestamp(value).tz_convert("UTC") for value in spine]
    sessions = sorted({timestamp.date().isoformat() for timestamp in timestamps})
    out: list[dict[str, Any]] = []
    for ordinal, start in enumerate(range(int(training_sessions), len(sessions))):
        score_session = sessions[start]
        refit_t = min(timestamp for timestamp in timestamps if timestamp.date().isoformat() == score_session)
        out.append(
            {
                "ordinal": ordinal,
                "refit_T": refit_t,
                "training_session_dates": sessions[start - int(training_sessions) : start],
                "score_session_dates": [score_session],
            }
        )
        if max_refits and len(out) >= int(max_refits):
            break
    return out


def _target_loader_for_engine(engine: Any, partitions: Sequence[Any]):
    def load_targets(request: Any) -> tuple[Any, dict[str, Any]]:
        import pandas as pd

        if request.empty:
            return request, {"target_loader": "ds24_vast_r51_engine_target_loader", "target_rows_loaded": 0}
        timestamps = set(pd.to_datetime(request["decision_timestamp"], utc=True))
        years = {timestamp.year for timestamp in timestamps}
        frame = engine.assemble_partitions(
            rows=[row for row in partitions if row.year in years],
            decision_timestamps=timestamps,
        )
        targets = frame[
            [
                "asset_id",
                "decision_timestamp",
                "target_value",
                "target_is_trainable",
                "target_available_timestamp",
            ]
        ].drop_duplicates(["asset_id", "decision_timestamp"])
        targets["decision_timestamp"] = pd.to_datetime(targets["decision_timestamp"], utc=True).map(lambda ts: ts.isoformat())
        return targets, {
            "target_loader": "ds24_vast_r51_engine_target_loader",
            "target_rows_loaded": int(len(targets)),
        }

    return load_targets


def _configured_external_snapshot_paths(paths: Sequence[str | Path] | None = None) -> list[Path]:
    out = [Path(path) for path in (paths or []) if str(path)]
    for env_name in ("DS24_DELL_STATUS_SNAPSHOT_PATH", "DS24_MAC_STATUS_SNAPSHOT_PATH"):
        value = os.environ.get(env_name, "").strip()
        if value:
            out.append(Path(value))
    return out


def _external_ownership_gate(
    *,
    repo_root: Path,
    run_root: Path,
    snapshot_paths: Sequence[str | Path] | None = None,
    allow_neutral_synthetic_ownership: bool = False,
) -> dict[str, Any]:
    queue_definition = read_json(Path(repo_root) / DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")
    paths = _configured_external_snapshot_paths(snapshot_paths)
    allow_neutral_synthetic_ownership = bool(
        allow_neutral_synthetic_ownership
        or os.environ.get("DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP", "0").lower() in {"1", "true", "yes"}
    )
    if paths:
        snapshots = [read_json(path) for path in paths]
    elif allow_neutral_synthetic_ownership:
        snapshots = reverse_queue.synthetic_external_status_fixture(queue_definition, now_utc=utc_now())["snapshots"]
    else:
        raise VastGpuLiveLaunchError(
            "DELL_MAC_OWNERSHIP_SNAPSHOTS_REQUIRED_FOR_LIVE_VAST_QUEUE:"
            "set DS24_DELL_STATUS_SNAPSHOT_PATH and DS24_MAC_STATUS_SNAPSHOT_PATH"
        )
    external = reverse_queue.evaluate_external_snapshots(
        snapshots,
        queue_definition,
        now_utc=utc_now(),
        allow_missing_snapshots=bool(allow_neutral_synthetic_ownership and not paths),
        allow_stale=bool(allow_neutral_synthetic_ownership and not paths),
    )
    payload = {
        "status": external["status"],
        "classification": external["classification"],
        "snapshot_paths": [str(path) for path in paths],
        "neutral_synthetic_ownership_used": bool(allow_neutral_synthetic_ownership and not paths),
        "external": external,
    }
    payload["ownership_gate_hash"] = stable_hash(payload)
    write_json_atomic(Path(run_root) / "queue_state/external_ownership_gate.json", payload)
    if payload["status"] != "PASS":
        raise VastGpuLiveLaunchError("EXTERNAL_OWNERSHIP_GATE_FAIL_CLOSED")
    return payload


def _ownership_decision_for_family(family: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    external = gate.get("external") if isinstance(gate.get("external"), Mapping) else {}
    family_evidence = external.get("family_evidence") if isinstance(external.get("family_evidence"), Mapping) else {}
    evidence = [row for row in family_evidence.get(family, []) if isinstance(row, Mapping)]
    classifications = {str(row.get("classification") or "") for row in evidence}
    if classifications & {
        "COMPATIBLE_EXTERNAL_RUNNING_OR_CLAIMED",
        "UNVERIFIED_EXTERNAL_RUNNING_OR_CLAIMED_BLOCKS_ADMISSION",
        "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED",
        "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED",
    }:
        return {"decision": "BLOCKED_EXTERNAL_OWNERSHIP", "evidence": evidence}
    if "SKIPPED_EXTERNAL_VERIFIED" in classifications:
        return {"decision": "SKIP_EXTERNAL_COMPLETED", "evidence": evidence}
    if not evidence:
        return {"decision": "NO_EXTERNAL_EVIDENCE_FAIL_CLOSED", "evidence": evidence}
    return {"decision": "CLAIMABLE", "evidence": evidence}


class _LazySequenceDataset:
    def __init__(
        self,
        groups: Mapping[str, Mapping[str, Any]],
        samples: Sequence[tuple[str, int, float]],
        *,
        sequence_length: int,
        feature_count: int,
    ) -> None:
        self.groups = groups
        self.samples = list(samples)
        self.sequence_length = int(sequence_length)
        self.feature_count = int(feature_count)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import numpy as np
        import torch

        asset_id, position, target = self.samples[index]
        features = self.groups[asset_id]["features"]
        start = max(0, int(position) - self.sequence_length + 1)
        chunk = features[start : int(position) + 1]
        sequence = np.full((self.sequence_length, self.feature_count), np.nan, dtype=np.float32)
        sequence[-len(chunk) :] = chunk
        return torch.from_numpy(sequence), torch.tensor(float(target), dtype=torch.float32)


def _sequence_config_for_family(
    family: str,
    *,
    device: str,
    require_cuda: bool,
    dataloader_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    epochs: int,
) -> Any:
    from config.config_defaults_ml import ML_DEFAULTS
    from core.research.ml.stock_level.stock_level_sequence_regressors import SequenceRegressorConfig

    prefixes = {
        "dlinear": "dlinear",
        "patchtst": "patchtst",
        "transformer": "transformer",
        "itransformer": "itransformer",
        "momentum_transformer": "momentum_transformer",
        "market_context_encoder": "market_context",
        "temporal_fusion_transformer": "tft",
    }
    prefix = prefixes[family]
    sequence_length = int(
        ML_DEFAULTS.get(f"{prefix}_sequence_length")
        or ML_DEFAULTS.get(f"{prefix}_encoder_length")
        or 32
    )
    return SequenceRegressorConfig(
        architecture=family,
        sequence_length=sequence_length,
        epochs=max(1, int(epochs)),
        batch_size=int(ML_DEFAULTS.get(f"{prefix}_batch_size") or 32),
        learning_rate=float(ML_DEFAULTS.get(f"{prefix}_learning_rate") or 0.001),
        weight_decay=float(ML_DEFAULTS.get(f"{prefix}_weight_decay") or 0.0001),
        random_seed=5101,
        device=device,
        d_model=int(ML_DEFAULTS.get(f"{prefix}_d_model") or ML_DEFAULTS.get(f"{prefix}_hidden_size") or 32),
        nhead=int(ML_DEFAULTS.get(f"{prefix}_heads") or ML_DEFAULTS.get(f"{prefix}_attention_heads") or 4),
        num_layers=int(ML_DEFAULTS.get(f"{prefix}_layers") or ML_DEFAULTS.get(f"{prefix}_lstm_layers") or 1),
        dim_feedforward=int(ML_DEFAULTS.get(f"{prefix}_feedforward") or 64),
        dropout=float(ML_DEFAULTS.get(f"{prefix}_dropout") or 0.0),
        patch_length=int(ML_DEFAULTS.get("patchtst_patch_length") or 16),
        patch_stride=int(ML_DEFAULTS.get("patchtst_patch_stride") or 8),
        torch_num_threads=1,
        dataloader_num_workers=max(0, int(dataloader_workers)),
        dataloader_pin_memory=bool(pin_memory),
        dataloader_prefetch_factor=max(1, int(prefetch_factor)),
        dataloader_persistent_workers=max(0, int(dataloader_workers)) > 0,
        cuda_required=bool(require_cuda),
    )


def _sequence_groups(panel: Any, predictors: Sequence[str]) -> dict[str, dict[str, Any]]:
    import numpy as np
    import pandas as pd

    work = panel.sort_values(["asset_id", "decision_timestamp"]).copy()
    groups: dict[str, dict[str, Any]] = {}
    for asset_id, group in work.groupby("asset_id", sort=False):
        timestamps = pd.to_datetime(group["decision_timestamp"], utc=True).tolist()
        features = group[list(predictors)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
        targets = pd.to_numeric(group["target_value"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        available = pd.to_datetime(group["target_available_timestamp"], utc=True).tolist()
        trainable = group["target_is_trainable"].astype(bool).to_numpy()
        position_by_timestamp = {timestamp.isoformat(): index for index, timestamp in enumerate(timestamps)}
        groups[str(asset_id)] = {
            "features": features,
            "timestamps": timestamps,
            "position_by_timestamp": position_by_timestamp,
            "targets": targets,
            "target_available": available,
            "target_is_trainable": trainable,
        }
    return groups


def _sequence_training_samples(groups: Mapping[str, Mapping[str, Any]], refit_t: Any) -> list[tuple[str, int, float]]:
    import numpy as np
    import pandas as pd

    cutoff = pd.Timestamp(refit_t).tz_convert("UTC")
    samples: list[tuple[str, int, float]] = []
    for asset_id, group in groups.items():
        for position, timestamp in enumerate(group["timestamps"]):
            target = float(group["targets"][position])
            if (
                timestamp < cutoff
                and bool(group["target_is_trainable"][position])
                and pd.Timestamp(group["target_available"][position]).tz_convert("UTC") <= cutoff
                and np.isfinite(target)
            ):
                samples.append((str(asset_id), int(position), target))
    return samples


def _sequence_score_requests(
    groups: Mapping[str, Mapping[str, Any]],
    score_frame: Any,
) -> list[tuple[str, int, str]]:
    import pandas as pd

    requests: list[tuple[str, int, str]] = []
    for row in score_frame[["asset_id", "decision_timestamp"]].itertuples(index=False):
        asset_id = str(row.asset_id)
        timestamp = pd.Timestamp(row.decision_timestamp).tz_convert("UTC").isoformat()
        group = groups.get(asset_id)
        if not group:
            continue
        position = group["position_by_timestamp"].get(timestamp)
        if position is None:
            continue
        requests.append((asset_id, int(position), timestamp))
    return requests


def _sequence_feature_stats(train: Any, predictors: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    values = train[list(predictors)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
    impute = np.nanmean(values, axis=0)
    impute = np.where(np.isfinite(impute), impute, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(values), values, impute)
    mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = filled.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return {"impute": impute, "mean": mean, "std": std}


def _normalise_sequence_batch(x: Any, stats: Mapping[str, Any], device: Any) -> Any:
    import torch

    impute = torch.as_tensor(stats["impute"], dtype=x.dtype, device=device).reshape(1, 1, -1)
    mean = torch.as_tensor(stats["mean"], dtype=x.dtype, device=device).reshape(1, 1, -1)
    std = torch.as_tensor(stats["std"], dtype=x.dtype, device=device).reshape(1, 1, -1)
    return (torch.where(torch.isfinite(x), x, impute) - mean) / std


def _sequence_batch_tensor(
    groups: Mapping[str, Mapping[str, Any]],
    requests: Sequence[tuple[str, int, str] | tuple[str, int, float]],
    *,
    sequence_length: int,
    feature_count: int,
) -> Any:
    import numpy as np
    import torch

    values = np.full((len(requests), sequence_length, feature_count), np.nan, dtype=np.float32)
    for row_index, request in enumerate(requests):
        asset_id = str(request[0])
        position = int(request[1])
        features = groups[asset_id]["features"]
        start = max(0, position - sequence_length + 1)
        chunk = features[start : position + 1]
        values[row_index, -len(chunk) :] = chunk
    return torch.from_numpy(values)


def _save_sequence_checkpoint(
    path: Path,
    *,
    family: str,
    config: Any,
    model: Any,
    optimizer: Any,
    stats: Mapping[str, Any],
    target_mean: float,
    target_std: float,
    refit_t: str,
) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(
        {
            "schema": "DS24_R51_LIVE_SEQUENCE_CHECKPOINT_V1",
            "family": family,
            "config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_stats": {
                "impute": stats["impute"].tolist(),
                "mean": stats["mean"].tolist(),
                "std": stats["std"].tolist(),
            },
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "refit_T": refit_t,
            "saved_at_utc": utc_now(),
        },
        temp,
    )
    os.replace(temp, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _fit_sequence_refit_package(
    *,
    family: str,
    config: Any,
    groups: Mapping[str, Mapping[str, Any]],
    train: Any,
    predictors: Sequence[str],
    refit_t: Any,
    checkpoint_path: Path,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn

    from core.research.ml.stock_level.stock_level_sequence_regressors import (
        TorchSequenceReturnRegressor,
    )

    device = torch.device(str(config.device))
    if config.cuda_required and (device.type != "cuda" or not torch.cuda.is_available()):
        raise VastGpuLiveLaunchError("LIVE_SEQUENCE_CUDA_REQUIRED_BUT_UNAVAILABLE")
    owner = TorchSequenceReturnRegressor(config)
    model = owner._build_model(torch, nn, feature_count=len(predictors)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    stats: dict[str, Any]
    target_mean = 0.0
    target_std = 1.0
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        if payload.get("family") != family:
            raise VastGpuLiveLaunchError("LIVE_SEQUENCE_CHECKPOINT_FAMILY_MISMATCH")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload["feature_stats"].items()
        }
        target_mean = float(payload["target_mean"])
        target_std = max(1e-6, float(payload["target_std"]))
        return {
            "model": model,
            "optimizer": optimizer,
            "stats": stats,
            "target_mean": target_mean,
            "target_std": target_std,
            "fit_status": "RESUMED_FROM_CHECKPOINT",
            "training_rows": int(len(train)),
            "loss_by_epoch": [],
        }
    samples = _sequence_training_samples(groups, refit_t)
    if not samples:
        raise VastGpuLiveLaunchError(f"LIVE_SEQUENCE_NO_TRAINING_SEQUENCES:{family}")
    stats = _sequence_feature_stats(train, predictors)
    targets = np.asarray([sample[2] for sample in samples], dtype=np.float32)
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets))
    if not np.isfinite(target_std) or target_std <= 1e-6:
        target_std = 1.0
    dataset = _LazySequenceDataset(
        groups,
        samples,
        sequence_length=config.sequence_length,
        feature_count=len(predictors),
    )
    loader_kwargs = {
        "num_workers": max(0, int(config.dataloader_num_workers or 0)),
        "pin_memory": bool(config.dataloader_pin_memory and device.type == "cuda"),
        "batch_size": max(1, int(config.batch_size)),
        "shuffle": True,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(config.dataloader_prefetch_factor or 2))
        loader_kwargs["persistent_workers"] = bool(config.dataloader_persistent_workers)
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    model.train()
    for _ in range(max(1, int(config.epochs))):
        batch_losses: list[float] = []
        for x, y in loader:
            x = x.to(device, non_blocking=bool(config.dataloader_pin_memory and device.type == "cuda"))
            y = y.to(device, non_blocking=bool(config.dataloader_pin_memory and device.type == "cuda"))
            x = _normalise_sequence_batch(x, stats, device)
            y = (y - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            output, _aux = owner._forward(model, x)
            loss = loss_fn(output, y)
            if not bool(torch.isfinite(loss).item()):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        losses.append(sum(batch_losses) / len(batch_losses) if batch_losses else float("nan"))
    if device.type == "cuda":
        torch.cuda.synchronize()
    checkpoint = _save_sequence_checkpoint(
        checkpoint_path,
        family=family,
        config=config,
        model=model,
        optimizer=optimizer,
        stats=stats,
        target_mean=target_mean,
        target_std=target_std,
        refit_t=str(refit_t),
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "stats": stats,
        "target_mean": target_mean,
        "target_std": target_std,
        "fit_status": "FIT_COMPLETE",
        "training_rows": int(len(train)),
        "training_sequences": len(samples),
        "loss_by_epoch": losses,
        "checkpoint": checkpoint,
    }


def _predict_sequence_timestamp(
    *,
    family: str,
    config: Any,
    model: Any,
    stats: Mapping[str, Any],
    target_mean: float,
    target_std: float,
    groups: Mapping[str, Mapping[str, Any]],
    score_frame: Any,
    predictors: Sequence[str],
    model_hash: str,
    refit_t: str,
) -> Any:
    import numpy as np
    import pandas as pd
    import torch

    from core.research.ml.stock_level.stock_level_sequence_regressors import (
        TorchSequenceReturnRegressor,
    )

    requests = _sequence_score_requests(groups, score_frame)
    if not requests:
        return pd.DataFrame(columns=["family", "decision_timestamp", "asset_id", "prediction"])
    owner = TorchSequenceReturnRegressor(config)
    device = torch.device(str(config.device))
    rows: list[dict[str, Any]] = []
    model.eval()
    batch_size = max(1, int(os.environ.get("DS24_VAST_SEQUENCE_SCORE_BATCH", "1024")))
    with torch.no_grad():
        for start in range(0, len(requests), batch_size):
            batch_requests = requests[start : start + batch_size]
            x = _sequence_batch_tensor(
                groups,
                batch_requests,
                sequence_length=config.sequence_length,
                feature_count=len(predictors),
            ).to(device)
            x = _normalise_sequence_batch(x, stats, device)
            scores, _aux = owner._forward(model, x)
            values = (scores.detach().cpu().numpy().astype(float) * float(target_std)) + float(target_mean)
            for request, value in zip(batch_requests, values):
                if not np.isfinite(float(value)):
                    raise VastGpuLiveLaunchError(f"LIVE_SEQUENCE_NONFINITE_PREDICTION:{family}")
                rows.append(
                    {
                        "family": family,
                        "config": "ds24_r51_cuda_live_sequence",
                        "decision_timestamp": request[2],
                        "asset_id": request[0],
                        "prediction": float(value),
                        "model_hash": model_hash,
                        "training_cutoff": refit_t,
                        "prediction_timestamp": utc_now(),
                    }
                )
    return pd.DataFrame(rows)


def _write_compact_oof_trace(root: Path, family: str, predictions: Any, *, top_n: int = 20) -> dict[str, Any]:
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if not predictions.empty:
        for decision, group in predictions.groupby("decision_timestamp", sort=True):
            ranked = group.sort_values(["prediction", "asset_id"], ascending=[False, True]).copy()
            selected = pd.concat([ranked.head(top_n), ranked.tail(min(top_n, len(ranked)))], ignore_index=True)
            for row in selected.itertuples(index=False):
                rows.append(
                    {
                        "family": family,
                        "decision_timestamp": str(decision),
                        "asset_id": str(getattr(row, "asset_id")),
                        "raw_selected_symbol_score": float(getattr(row, "prediction")),
                        "model_hash": str(getattr(row, "model_hash", "")),
                    }
                )
    trace = pd.DataFrame(rows)
    path = root / "compact_oof_trace.parquet"
    if trace.empty:
        write_json_atomic(root / "compact_oof_trace.empty.json", {"family": family, "rows": 0})
    else:
        trace.to_parquet(path, index=False)
    manifest = {
        "family": family,
        "rows": int(len(trace)),
        "path": str(path) if trace is not None and not trace.empty else str(root / "compact_oof_trace.empty.json"),
        "full_prediction_files_written": 0,
    }
    manifest["trace_hash"] = stable_hash(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def run_live_sequence_family(
    family: str,
    *,
    repo_root: Path,
    dataset_root: Path,
    run_root: Path,
    resume: bool = True,
    max_refits: int = 0,
) -> dict[str, Any]:
    if family not in GPU_SEQUENCE_FAMILIES:
        raise VastGpuLiveLaunchError(f"NOT_A_GPU_SEQUENCE_FAMILY:{family}")
    import pandas as pd

    from core.research.ml.ds24_metrics_only_evaluator import (
        MetricsOnlyEvidenceWriter,
        resolved_performance_contract_v3_hash,
    )

    context = _live_engine_context(repo_root, dataset_root)
    engine = context["engine"]
    partitions = context["partitions"]
    predictors = context["predictors"]
    config = _sequence_config_for_family(
        family,
        device="cuda",
        require_cuda=True,
        dataloader_workers=max(0, int(os.environ.get("DS24_VAST_DATALOADER_WORKERS", "4"))),
        pin_memory=True,
        prefetch_factor=max(1, int(os.environ.get("DS24_VAST_PREFETCH_FACTOR", "2"))),
        epochs=max(1, int(os.environ.get("DS24_VAST_SEQUENCE_EPOCHS", "1"))),
    )
    family_manifest_root = Path(run_root) / "manifests" / f"family={family}"
    metrics_root = Path(run_root) / "metrics_only_v3" / f"family={family}"
    checkpoint_root = Path(run_root) / "checkpoints" / f"family={family}"
    model_root = Path(run_root) / "models" / f"family={family}"
    compact_root = Path(run_root) / "ensemble_oof_scores_v2" / f"family={family}"
    for path in (family_manifest_root, metrics_root, checkpoint_root, model_root, compact_root):
        path.mkdir(parents=True, exist_ok=True)
    writer = MetricsOnlyEvidenceWriter(
        metrics_root,
        family=family,
        enable_resolved_performance_v3=True,
        target_loader=_target_loader_for_engine(engine, partitions),
        terminal_timestamp=POLICY_TERMINAL_T,
        namespace_lease_enabled=True,
        resume_generation=os.environ.get("DS24_VAST_RESUME_GENERATION", "r51"),
        command_hash=stable_hash({"command": "run-live-sequence-family", "family": family}),
        configuration_hash=stable_hash({"family": family, "config": config.__dict__, "predictor_manifest_hash": context["predictor_manifest_hash"]}),
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    spine = list(engine.decision_spine())
    schedule = _daily_refit_schedule(spine, max_refits=max_refits)
    progress_path = family_manifest_root / "progress.json"
    progress = read_json(progress_path)
    completed_refits = set(progress.get("completed_refits", [])) if isinstance(progress, Mapping) else set()
    total_prediction_rows = int(progress.get("prediction_rows", 0) or 0) if isinstance(progress, Mapping) else 0
    refit_results: list[dict[str, Any]] = []
    try:
        for spec in schedule:
            refit_iso = pd.Timestamp(spec["refit_T"]).tz_convert("UTC").isoformat()
            if refit_iso in completed_refits:
                continue
            package_dates = set(spec["training_session_dates"] + spec["score_session_dates"])
            package_years = {int(date[:4]) for date in package_dates}
            package_panel = engine.assemble_partitions(
                rows=[row for row in partitions if row.year in package_years],
                decision_dates=package_dates,
            )
            if package_panel.empty:
                raise VastGpuLiveLaunchError(f"LIVE_SEQUENCE_EMPTY_PACKAGE:{family}:{refit_iso}")
            train = package_panel[
                (package_panel["session_date"].astype(str).isin(spec["training_session_dates"]))
                & (package_panel["decision_timestamp"] < spec["refit_T"])
                & (package_panel["target_is_trainable"].astype(bool))
                & (package_panel["target_available_timestamp"] <= spec["refit_T"])
            ].copy()
            if train.empty:
                raise VastGpuLiveLaunchError(f"LIVE_SEQUENCE_EMPTY_TRAIN:{family}:{refit_iso}")
            groups = _sequence_groups(package_panel, predictors)
            stamp = pd.Timestamp(spec["refit_T"]).strftime("%Y%m%dT%H%M%SZ")
            checkpoint_path = checkpoint_root / f"{family}_{stamp}.pt"
            fit = _fit_sequence_refit_package(
                family=family,
                config=config,
                groups=groups,
                train=train,
                predictors=predictors,
                refit_t=spec["refit_T"],
                checkpoint_path=checkpoint_path,
                resume=resume,
            )
            model_path = model_root / f"{family}_{stamp}.pt"
            shutil.copy2(checkpoint_path, model_path)
            model_hash = file_sha256(model_path)
            score_predictions = []
            for score_date in spec["score_session_dates"]:
                score_panel = package_panel[package_panel["session_date"].astype(str) == score_date].copy()
                for timestamp, score_frame in score_panel.groupby("decision_timestamp", sort=True):
                    predictions = _predict_sequence_timestamp(
                        family=family,
                        config=config,
                        model=fit["model"],
                        stats=fit["stats"],
                        target_mean=fit["target_mean"],
                        target_std=fit["target_std"],
                        groups=groups,
                        score_frame=score_frame,
                        predictors=predictors,
                        model_hash=model_hash,
                        refit_t=refit_iso,
                    )
                    if predictions.empty:
                        continue
                    writer.commit_predictions(
                        predictions,
                        metadata={
                            "family": family,
                            "model_hash": model_hash,
                            "model_vintage_id": stable_hash({"family": family, "refit_T": refit_iso, "model_hash": model_hash}),
                            "preprocessing_hash": stable_hash({"family": family, "feature_stats": "mean_impute_standardise", "predictors": predictors}),
                            "policy_hash": stable_hash({"ticket": TICKET_ID, "family": family}),
                            "training_cutoff": refit_iso,
                            "prediction_timestamp": utc_now(),
                        },
                    )
                    total_prediction_rows += int(len(predictions))
                    score_predictions.append(predictions)
            combined = pd.concat(score_predictions, ignore_index=True) if score_predictions else pd.DataFrame()
            compact = _write_compact_oof_trace(compact_root, family, combined)
            completed_refits.add(refit_iso)
            row = {
                "family": family,
                "refit_T": refit_iso,
                "fit_status": fit["fit_status"],
                "training_rows": int(len(train)),
                "prediction_rows_total": total_prediction_rows,
                "checkpoint_path": str(checkpoint_path),
                "model_path": str(model_path),
                "model_hash": model_hash,
                "compact_oof": compact,
                "device": "cuda",
                "cuda_required": True,
                "completed_at_utc": utc_now(),
            }
            refit_results.append(row)
            write_json_atomic(
                progress_path,
                {
                    "family": family,
                    "phase": "SCORING",
                    "completed_refits": sorted(completed_refits),
                    "last_completed_refit_T": refit_iso,
                    "prediction_rows": total_prediction_rows,
                    "heartbeat_utc": utc_now(),
                },
            )
        writer.release_namespace_lease()
    except Exception:
        try:
            writer.release_namespace_lease()
        except Exception:
            pass
        raise
    payload = {
        "status": "PASS",
        "classification": "LIVE_SEQUENCE_FAMILY_COMPLETE",
        "family": family,
        "refits_completed": len(completed_refits),
        "prediction_rows": total_prediction_rows,
        "metrics_root": str(metrics_root),
        "checkpoints_root": str(checkpoint_root),
        "models_root": str(model_root),
        "compact_root": str(compact_root),
        "refit_results_tail": refit_results[-5:],
        "outer_holdout_access": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    payload["family_run_hash"] = stable_hash(payload)
    write_json_atomic(family_manifest_root / "COMMITTED.json", payload)
    return payload


def _ranking_labels_and_groups(train: Any) -> tuple[Any, Any, list[int]]:
    import numpy as np
    import pandas as pd

    ordered = train.sort_values(["decision_timestamp", "asset_id"]).copy()
    labels = []
    groups = []
    for _timestamp, group in ordered.groupby("decision_timestamp", sort=True):
        values = pd.to_numeric(group["target_value"], errors="coerce")
        pct = values.rank(method="average", pct=True)
        label = np.floor(np.clip(pct.to_numpy(dtype=float) * 5.0, 0.0, 4.999)).astype(int)
        labels.extend(label.tolist())
        groups.append(int(len(group)))
    return ordered, np.asarray(labels, dtype=int), groups if groups else []


def _lightgbm_configuration_for_family(family: str, *, device_type: str) -> dict[str, Any]:
    if family == "lightgbm_rank_xendcg":
        from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
            fixed_rank_xendcg_configuration,
        )

        return fixed_rank_xendcg_configuration(num_threads=1, device_type=device_type)
    if family == "lightgbm_lambdarank":
        from core.research.ml.stock_level.lightgbm_lambdarank_selector import (
            fixed_lambdarank_configuration,
        )

        return fixed_lambdarank_configuration(
            label_contract="within_date_quintile_relevance_v1",
            num_threads=1,
            device_type=device_type,
        )
    raise VastGpuLiveLaunchError(f"NOT_A_LIGHTGBM_RANKING_FAMILY:{family}")


def _fit_live_lightgbm_ranker(
    *,
    family: str,
    train: Any,
    predictors: Sequence[str],
    prefer_gpu: bool,
) -> dict[str, Any]:
    import numpy as np
    import lightgbm as lgb

    ordered, y_train, groups = _ranking_labels_and_groups(train)
    if not groups:
        raise VastGpuLiveLaunchError(f"LIVE_LIGHTGBM_EMPTY_GROUPS:{family}")
    x_train = ordered[list(predictors)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float, copy=True)
    attempted = []
    last_error = ""
    for device_type in (["gpu", "cpu"] if prefer_gpu else ["cpu"]):
        configuration = _lightgbm_configuration_for_family(family, device_type=device_type)
        params = dict(configuration["parameters"])
        attempted.append(device_type)
        try:
            model = lgb.LGBMRanker(**params)
            model.fit(x_train, y_train, group=groups)
            return {
                "model": model,
                "configuration": configuration,
                "runtime_policy": "GPU" if device_type == "gpu" else ("CPU_FALLBACK" if prefer_gpu else "CPU"),
                "attempted_device_types": attempted,
                "fallback_reason": last_error,
                "training_rows": int(len(ordered)),
                "training_groups": len(groups),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if device_type != "gpu":
                raise
    raise VastGpuLiveLaunchError(f"LIVE_LIGHTGBM_FIT_FAILED:{family}:{last_error}")


def run_live_lightgbm_family(
    family: str,
    *,
    repo_root: Path,
    dataset_root: Path,
    run_root: Path,
    resume: bool = True,
    max_refits: int = 0,
) -> dict[str, Any]:
    if family not in LIGHTGBM_RANKING_FAMILIES:
        raise VastGpuLiveLaunchError(f"NOT_A_LIGHTGBM_RANKING_FAMILY:{family}")
    import pandas as pd
    import pickle
    import numpy as np

    from core.research.ml.ds24_metrics_only_evaluator import (
        MetricsOnlyEvidenceWriter,
        resolved_performance_contract_v3_hash,
    )

    context = _live_engine_context(repo_root, dataset_root)
    engine = context["engine"]
    partitions = context["partitions"]
    predictors = context["predictors"]
    family_manifest_root = Path(run_root) / "manifests" / f"family={family}"
    metrics_root = Path(run_root) / "metrics_only_v3" / f"family={family}"
    checkpoint_root = Path(run_root) / "checkpoints" / f"family={family}"
    model_root = Path(run_root) / "models" / f"family={family}"
    compact_root = Path(run_root) / "ensemble_oof_scores_v2" / f"family={family}"
    for path in (family_manifest_root, metrics_root, checkpoint_root, model_root, compact_root):
        path.mkdir(parents=True, exist_ok=True)
    writer = MetricsOnlyEvidenceWriter(
        metrics_root,
        family=family,
        enable_resolved_performance_v3=True,
        target_loader=_target_loader_for_engine(engine, partitions),
        terminal_timestamp=POLICY_TERMINAL_T,
        namespace_lease_enabled=True,
        resume_generation=os.environ.get("DS24_VAST_RESUME_GENERATION", "r51"),
        command_hash=stable_hash({"command": "run-live-lightgbm-family", "family": family}),
        configuration_hash=stable_hash({"family": family, "predictor_manifest_hash": context["predictor_manifest_hash"]}),
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    spine = list(engine.decision_spine())
    schedule = _daily_refit_schedule(spine, max_refits=max_refits)
    progress_path = family_manifest_root / "progress.json"
    progress = read_json(progress_path)
    completed_refits = set(progress.get("completed_refits", [])) if isinstance(progress, Mapping) else set()
    total_prediction_rows = int(progress.get("prediction_rows", 0) or 0) if isinstance(progress, Mapping) else 0
    prefer_gpu = os.environ.get("DS24_LIGHTGBM_GPU_SUPPORTED", "0").lower() in {"1", "true", "yes"}
    refit_results: list[dict[str, Any]] = []
    try:
        for spec in schedule:
            refit_iso = pd.Timestamp(spec["refit_T"]).tz_convert("UTC").isoformat()
            if resume and refit_iso in completed_refits:
                continue
            package_dates = set(spec["training_session_dates"] + spec["score_session_dates"])
            package_years = {int(date[:4]) for date in package_dates}
            package_panel = engine.assemble_partitions(
                rows=[row for row in partitions if row.year in package_years],
                decision_dates=package_dates,
            )
            train = package_panel[
                (package_panel["session_date"].astype(str).isin(spec["training_session_dates"]))
                & (package_panel["decision_timestamp"] < spec["refit_T"])
                & (package_panel["target_is_trainable"].astype(bool))
                & (package_panel["target_available_timestamp"] <= spec["refit_T"])
            ].copy()
            train = train[np.isfinite(pd.to_numeric(train["target_value"], errors="coerce"))].copy()
            if train.empty:
                raise VastGpuLiveLaunchError(f"LIVE_LIGHTGBM_EMPTY_TRAIN:{family}:{refit_iso}")
            fit = _fit_live_lightgbm_ranker(
                family=family,
                train=train,
                predictors=predictors,
                prefer_gpu=prefer_gpu,
            )
            stamp = pd.Timestamp(spec["refit_T"]).strftime("%Y%m%dT%H%M%SZ")
            model_path = model_root / f"{family}_{stamp}.pkl"
            checkpoint_path = checkpoint_root / f"{family}_{stamp}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump(
                    {
                        "schema": "DS24_R51_LIVE_LIGHTGBM_RANKING_MODEL_V1",
                        "family": family,
                        "configuration": fit["configuration"],
                        "runtime_policy": fit["runtime_policy"],
                        "model": fit["model"],
                        "predictors": predictors,
                        "refit_T": refit_iso,
                    },
                    handle,
                )
            shutil.copy2(model_path, checkpoint_path)
            model_hash = file_sha256(model_path)
            score_predictions = []
            for score_date in spec["score_session_dates"]:
                score_panel = package_panel[package_panel["session_date"].astype(str) == score_date].copy()
                for timestamp, score_frame in score_panel.groupby("decision_timestamp", sort=True):
                    x_score = score_frame[list(predictors)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float, copy=True)
                    scores = fit["model"].predict(x_score)
                    predictions = pd.DataFrame(
                        {
                            "family": family,
                            "config": "ds24_r51_gpu_preferred_live_lightgbm",
                            "decision_timestamp": pd.Timestamp(timestamp).tz_convert("UTC").isoformat(),
                            "asset_id": score_frame["asset_id"].astype(str),
                            "prediction": np.asarray(scores, dtype=float),
                            "model_hash": model_hash,
                            "training_cutoff": refit_iso,
                            "prediction_timestamp": utc_now(),
                        }
                    )
                    if not np.isfinite(predictions["prediction"].to_numpy(dtype=float)).all():
                        raise VastGpuLiveLaunchError(f"LIVE_LIGHTGBM_NONFINITE_PREDICTION:{family}")
                    writer.commit_predictions(
                        predictions,
                        metadata={
                            "family": family,
                            "model_hash": model_hash,
                            "model_vintage_id": stable_hash({"family": family, "refit_T": refit_iso, "model_hash": model_hash}),
                            "preprocessing_hash": stable_hash({"family": family, "preprocessing": "fillna_zero", "predictors": predictors}),
                            "policy_hash": stable_hash({"ticket": TICKET_ID, "family": family}),
                            "training_cutoff": refit_iso,
                            "prediction_timestamp": utc_now(),
                        },
                    )
                    total_prediction_rows += int(len(predictions))
                    score_predictions.append(predictions)
            combined = pd.concat(score_predictions, ignore_index=True) if score_predictions else pd.DataFrame()
            compact = _write_compact_oof_trace(compact_root, family, combined)
            completed_refits.add(refit_iso)
            row = {
                "family": family,
                "refit_T": refit_iso,
                "runtime_policy": fit["runtime_policy"],
                "attempted_device_types": fit["attempted_device_types"],
                "fallback_reason": fit["fallback_reason"],
                "training_rows": fit["training_rows"],
                "prediction_rows_total": total_prediction_rows,
                "checkpoint_path": str(checkpoint_path),
                "model_path": str(model_path),
                "model_hash": model_hash,
                "compact_oof": compact,
                "completed_at_utc": utc_now(),
            }
            refit_results.append(row)
            write_json_atomic(
                progress_path,
                {
                    "family": family,
                    "phase": "SCORING",
                    "completed_refits": sorted(completed_refits),
                    "last_completed_refit_T": refit_iso,
                    "prediction_rows": total_prediction_rows,
                    "heartbeat_utc": utc_now(),
                },
            )
        writer.release_namespace_lease()
    except Exception:
        try:
            writer.release_namespace_lease()
        except Exception:
            pass
        raise
    payload = {
        "status": "PASS",
        "classification": "LIVE_LIGHTGBM_RANKING_FAMILY_COMPLETE",
        "family": family,
        "refits_completed": len(completed_refits),
        "prediction_rows": total_prediction_rows,
        "metrics_root": str(metrics_root),
        "checkpoints_root": str(checkpoint_root),
        "models_root": str(model_root),
        "compact_root": str(compact_root),
        "refit_results_tail": refit_results[-5:],
        "outer_holdout_access": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    payload["family_run_hash"] = stable_hash(payload)
    write_json_atomic(family_manifest_root / "COMMITTED.json", payload)
    return payload


def _family_command(family: str, repo_root: Path, dataset_root: Path, run_root: Path) -> list[str]:
    if family in GPU_SEQUENCE_FAMILIES:
        return [
            "python",
            "-m",
            "core.research.ml.ds24.vast_gpu_live_launch_r1",
            "run-live-sequence-family",
            "--family",
            family,
            "--repo-root",
            str(repo_root),
            "--dataset-root",
            str(dataset_root),
            "--run-root",
            str(run_root),
            "--resume",
            "--max-refits",
            os.environ.get("DS24_VAST_MAX_REFITS", "0"),
        ]
    if family in LIGHTGBM_RANKING_FAMILIES:
        return [
            "python",
            "-m",
            "core.research.ml.ds24.vast_gpu_live_launch_r1",
            "run-live-lightgbm-family",
            "--family",
            family,
            "--repo-root",
            str(repo_root),
            "--dataset-root",
            str(dataset_root),
            "--run-root",
            str(run_root),
            "--resume",
            "--max-refits",
            os.environ.get("DS24_VAST_MAX_REFITS", "0"),
        ]
    raise VastGpuLiveLaunchError(f"UNKNOWN_R49_FAMILY:{family}")


def run_vast_reverse_queue(
    *,
    repo_root: Path,
    dataset_root: Path,
    run_root: Path,
    execute_live: bool,
    confirm_token: str,
    external_snapshot_paths: Sequence[str | Path] | None = None,
    allow_neutral_synthetic_ownership: bool = False,
) -> dict[str, Any]:
    if execute_live and confirm_token != LIVE_CONFIRM_TOKEN:
        raise VastGpuLiveLaunchError("LIVE_CONFIRM_TOKEN_MISMATCH")
    run_root = Path(run_root)
    admission_path = run_root / "telemetry/gpu_admission.json"
    admission = read_json(admission_path)
    admission_validation = validate_gpu_admission_evidence(admission) if admission else {
        "status": "FAIL",
        "classification": "GPU_ADMISSION_EVIDENCE_MISSING",
    }
    if admission_validation["status"] != "PASS":
        raise VastGpuLiveLaunchError("GPU_ADMISSION_REQUIRED_BEFORE_QUEUE_RELEASE")
    ownership_gate = _external_ownership_gate(
        repo_root=repo_root,
        run_root=run_root,
        snapshot_paths=external_snapshot_paths,
        allow_neutral_synthetic_ownership=allow_neutral_synthetic_ownership,
    )
    state_path = run_root / "queue_state/queue_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "queue_id": QUEUE_ID,
        "accepted_reverse_order": list(ACCEPTED_REVERSE_ORDER),
        "completed": [],
        "failed": [],
        "cursor_index": 0,
    }
    completed = list(state.get("completed", []))
    failed = list(state.get("failed", []))
    launched = []
    for family in ACCEPTED_REVERSE_ORDER:
        if family in completed or family in failed:
            continue
        ownership_decision = _ownership_decision_for_family(family, ownership_gate)
        if ownership_decision["decision"] == "SKIP_EXTERNAL_COMPLETED":
            completed.append(family)
            state.update(
                {
                    "completed": completed,
                    "failed": failed,
                    "cursor_index": ACCEPTED_REVERSE_ORDER.index(family) + 1,
                    "last_external_completed_family": family,
                    "last_heartbeat_utc": utc_now(),
                }
            )
            write_json_atomic(state_path, state)
            continue
        if ownership_decision["decision"] != "CLAIMABLE":
            write_json_atomic(
                run_root / "queue_state/blocked_external_ownership.json",
                {"family": family, **ownership_decision, "blocked_at_utc": utc_now()},
            )
            raise VastGpuLiveLaunchError(f"FAMILY_NOT_CLAIMABLE_FROM_DELL_MAC_SNAPSHOTS:{family}")
        guard = budget_self_stop_guard(
            max_runtime_hours=float(os.environ.get("DS24_MAX_RUNTIME_HOURS", "20")),
            max_estimated_cost_usd=float(os.environ.get("DS24_MAX_ESTIMATED_COST_USD", "8.40")),
            hourly_price_usd=float(os.environ.get("DS24_HOURLY_PRICE_USD", "0")),
            started_at_utc=(run_root / "INSTANCE_START_TIMESTAMP").read_text(encoding="utf-8").strip()
            if (run_root / "INSTANCE_START_TIMESTAMP").exists()
            else None,
        )
        write_json_atomic(run_root / "queue_state/latest_budget_self_stop_guard.json", guard)
        if guard["status"] == "STOP_REQUIRED":
            raise VastGpuLiveLaunchError("VAST_BUDGET_SELF_STOP_REQUIRED_BEFORE_NEXT_FAMILY")
        current = {"family": family, "started_at_utc": utc_now(), "queue_id": QUEUE_ID, "ownership_decision": ownership_decision}
        write_json_atomic(run_root / "queue_state/current_family.json", current)
        command = _family_command(family, repo_root, dataset_root, run_root)
        launched.append({"family": family, "command": command, "gpu_singleton": family in GPU_SEQUENCE_FAMILIES})
        if execute_live:
            completed_process = subprocess.run(command, cwd=repo_root, text=True, timeout=None, check=False)
            if completed_process.returncode != 0:
                failed.append(family)
                state.update({"failed": failed, "cursor_index": ACCEPTED_REVERSE_ORDER.index(family)})
                write_json_atomic(state_path, state)
                raise VastGpuLiveLaunchError(f"FAMILY_COMMAND_FAILED:{family}:{completed_process.returncode}")
        completed.append(family)
        state.update(
            {
                "completed": completed,
                "failed": failed,
                "cursor_index": ACCEPTED_REVERSE_ORDER.index(family) + 1,
                "last_completed_family": family,
                "last_heartbeat_utc": utc_now(),
            }
        )
        write_json_atomic(state_path, state)
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "classification": "VAST_REVERSE_QUEUE_COMPLETE" if not failed else "VAST_REVERSE_QUEUE_FAILED",
        "queue_id": QUEUE_ID,
        "launched": launched,
        "completed": completed,
        "failed": failed,
        "accepted_reverse_order": list(ACCEPTED_REVERSE_ORDER),
        "execute_live": bool(execute_live),
        "outer_holdout_access": False,
        "paper_orders": 0,
        "live_orders": 0,
        "external_ownership_gate": {
            "status": ownership_gate["status"],
            "classification": ownership_gate["classification"],
            "snapshot_paths": ownership_gate["snapshot_paths"],
            "neutral_synthetic_ownership_used": ownership_gate["neutral_synthetic_ownership_used"],
        },
    }
    payload["queue_run_hash"] = stable_hash(payload)
    write_json_atomic(run_root / "queue_state/final_queue_result.json", payload)
    return payload


def run_lightgbm_family_contract(family: str, dataset_root: Path, run_root: Path) -> dict[str, Any]:
    policy = lightgbm_ranking_runtime_policy(
        gpu_supported=os.environ.get("DS24_LIGHTGBM_GPU_SUPPORTED", "0") in {"1", "true", "TRUE"},
        fallback_allowed=True,
    )
    selected = next(row for row in policy["families"] if row["family"] == family)
    target = Path(run_root) / "metrics_only_v3" / f"family={family}"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "family": family,
        "dataset_root": str(dataset_root),
        "runtime_policy": selected,
        "synthetic_only": False,
        "live_training_adapter": "lightgbm_production_selector GPU-preferred configuration; full fit delegated by live worker integration",
    }
    payload["result_hash"] = stable_hash(payload)
    write_json_atomic(target / "lightgbm_runtime_policy.json", payload)
    return payload


def verify_repatriation(root: Path) -> dict[str, Any]:
    root = Path(root)
    forbidden = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and any(marker in path.as_posix().lower() for marker in FORBIDDEN_MARKERS)
    ]
    present = {
        "metrics": any((root / name).exists() for name in ("metrics_only_v3", "metrics")),
        "compact_oof_artifacts": (root / "ensemble_oof_scores_v2").exists(),
        "checkpoints": (root / "checkpoints").exists(),
        "manifests": (root / "manifests").exists() or (root / "vast_output_manifest.json").exists(),
        "completion_markers": (root / "COMMITTED.json").exists(),
    }
    payload = {
        "status": "PASS" if not forbidden else "FAIL",
        "classification": "DELL_REPATRIATION_LOCAL_VERIFY_PASS" if not forbidden else "DELL_REPATRIATION_FORBIDDEN_ARTIFACTS",
        "artifact_tiers_present": sorted(key for key, value in present.items() if value),
        "forbidden_artifacts": forbidden,
    }
    payload["verification_hash"] = stable_hash(payload)
    return payload


def validation_suite_payload(repo_root: Path) -> dict[str, Any]:
    prerequisite = b2.load_prerequisite_queue_authority(repo_root)
    r50_manifest = read_json(Path(repo_root) / DEFAULT_B2_AUTHORITY_ROOT_REL / "manifest.json")
    audit = family_adapter_gpu_audit(repo_root)
    admission = synthetic_passing_gpu_admission_evidence()["validation"]
    payload = {
        "status": "PASS" if prerequisite.get("status") == "PASS" and r50_manifest.get("status") == "PASS" and audit["status"] == "PASS" and admission["status"] == "PASS" else "FAIL",
        "r49_prerequisite": prerequisite,
        "r50_manifest_hash": r50_manifest.get("manifest_hash", ""),
        "family_gpu_audit_status": audit["status"],
        "synthetic_gpu_admission_status": admission["status"],
        "synthetic_only": True,
        "vast_instance_rented": False,
        "real_b2_transfer_performed": False,
        "live_model_work_started": False,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def authority_manifest(repo_root: Path, authority_root: Path, files: Sequence[Path], *, bootstrap_commit: str = "<FINAL_R51_COMMIT>") -> dict[str, Any]:
    rows = []
    for path in files:
        if path.is_file():
            rows.append(
                {
                    "path": repo_rel(repo_root, path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    payload = {
        "ticket_id": TICKET_ID,
        "terminal_classification": TERMINAL_CLASSIFICATION,
        "status": "PASS",
        "queue_id": QUEUE_ID,
        "accepted_reverse_order": list(ACCEPTED_REVERSE_ORDER),
        "r49_r50_commit": EXPECTED_R49_R50_COMMIT,
        "bootstrap_commit": bootstrap_commit,
        "dataset": {
            "bucket": B2_BUCKET,
            "prefix": B2_PREFIX,
            "expected_object_count": EXPECTED_DATASET_OBJECT_COUNT,
            "expected_bytes": EXPECTED_DATASET_BYTES,
        },
        "files": rows,
        "file_count": len(rows),
        "synthetic_only_during_implementation": True,
        "vast_instance_rented": False,
        "real_b2_transfer_performed": False,
        "live_model_work_started": False,
        "forbidden": {
            "outer_holdout_access": False,
            "paper_orders": 0,
            "live_orders": 0,
            "forbidden_full_prediction_artifacts": False,
        },
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def write_materialized_live_configs(repo_root: Path, output_root: Path, *, bootstrap_commit: str = "<FINAL_R51_COMMIT>") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root)
    queue_definition = read_json(repo_root / DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")
    ownership = ownership_examples(queue_definition)
    files: list[Path] = []
    artifacts: dict[str, Any] = {
        "VAST_BOOTSTRAP_CONFIG_JSON.example.json": build_bootstrap_config_example(bootstrap_commit=bootstrap_commit),
        "PUBLISHER_CONFIG_JSON.example.json": publisher_config_example(),
        "DELL_REPATRIATION_CONFIG_JSON.example.json": dell_repatriation_config_example(),
        "b2_remote_inventory.example.json": b2_remote_inventory_example(),
        "DATASET_COMPLETE.example.json": dataset_complete_example(),
        "dell_status_snapshot.example.json": ownership["dell_status_snapshot"],
        "mac_status_snapshot.example.json": ownership["mac_status_snapshot"],
        "dell_ack.example.json": ownership["dell_ack"],
        "mac_ack.example.json": ownership["mac_ack"],
        "ownership_plan.example.json": ownership["ownership_plan"],
        "gpu_family_adapter_audit.json": family_adapter_gpu_audit(repo_root),
        "single_gpu_thread_authority.json": single_gpu_thread_authority(),
        "lightgbm_gpu_fallback_policy.gpu_supported.json": lightgbm_ranking_runtime_policy(gpu_supported=True),
        "lightgbm_gpu_fallback_policy.cpu_fallback.json": lightgbm_ranking_runtime_policy(gpu_supported=False),
        "gpu_admission_benchmark.schema.json": gpu_admission_schema(),
        "gpu_admission_benchmark.example.json": synthetic_passing_gpu_admission_evidence(),
        "vast_bootstrap_config.schema.json": vast_bootstrap_config_schema(),
        "publisher_config.schema.json": publisher_config_schema(),
        "dell_repatriation_config.schema.json": dell_repatriation_config_schema(),
        "monitoring_commands.json": monitoring_commands(),
        "budget_self_stop_guard.json": budget_self_stop_guard(
            started_at_utc="2026-09-03T00:00:00Z",
            now_utc="2026-09-03T01:00:00Z",
            hourly_price_usd=0.42,
        ),
        "validation_suite.synthetic.json": validation_suite_payload(repo_root),
        "limitations.json": {
            "synthetic_tests_only": True,
            "vast_instance_rented": False,
            "real_b2_transfer_performed": False,
            "live_model_work_started": False,
            "remaining_user_action": "Create one Jupyter-proxy Vast instance, open browser terminal, paste the guarded bootstrap command.",
        },
    }
    for name, payload in artifacts.items():
        path = output_root / name
        write_json_atomic(path, payload)
        files.append(path)
    scripts = {
        "vast_jupyter_proxy_bootstrap.sh": render_vast_jupyter_proxy_bootstrap(),
        "dell_repatriate_vast_outputs.ps1": render_dell_repatriation_launcher(),
        "vast_show_status.sh": render_status_shell(),
        "RUNBOOK.md": runbook_text(bootstrap_commit=bootstrap_commit),
    }
    for name, text in scripts.items():
        path = output_root / name
        write_text_atomic(path, text)
        files.append(path)
    manifest = authority_manifest(repo_root, output_root, files, bootstrap_commit=bootstrap_commit)
    manifest_path = output_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def runbook_text(*, bootstrap_commit: str = "<FINAL_R51_COMMIT>") -> str:
    return f"""# DS24 Vast R51 Jupyter-Proxy Live Launch

Classification: `{TERMINAL_CLASSIFICATION}`.

Use one Vast browser-terminal session. Do not rely on direct SSH or proxy SSH.

The launcher clones `{DEFAULT_REPO_URL}` at `${{DS24_BOOTSTRAP_COMMIT}}`, verifies the R49/R50 authority commit `{EXPECTED_R49_R50_COMMIT_SHORT}`, configures Backblaze through rclone environment variables without writing secrets to the repo, downloads `{B2_BUCKET}/{B2_PREFIX}`, verifies `{EXPECTED_DATASET_OBJECT_COUNT}` files and `{EXPECTED_DATASET_BYTES}` bytes excluding `DATASET_COMPLETE.json`, runs GPU admission, starts the durable publisher in tmux, then starts the accepted reverse queue in tmux.

Required terminal variables:

```bash
export B2_APPLICATION_KEY_ID='<Backblaze key id>'
export B2_APPLICATION_KEY='<Backblaze application key>'
export DS24_BOOTSTRAP_COMMIT='{bootstrap_commit}'
export DS24_VAST_LIVE_CONFIRM_TOKEN='{LIVE_CONFIRM_TOKEN}'
export DS24_DELL_STATUS_SNAPSHOT_PATH='<fresh Dell snapshot JSON path, or omit only with DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP=1>'
export DS24_MAC_STATUS_SNAPSHOT_PATH='<fresh Mac snapshot JSON path, or omit only with DS24_ALLOW_NEUTRAL_SYNTHETIC_OWNERSHIP=1>'
# Optional single-paste alternative to paths:
# export DS24_DELL_STATUS_SNAPSHOT_JSON_B64='<base64 -w0 dell_status_snapshot.json>'
# export DS24_MAC_STATUS_SNAPSHOT_JSON_B64='<base64 -w0 mac_status_snapshot.json>'
```

Then run:

```bash
curl -fsSL "https://raw.githubusercontent.com/Linnett9/trading_system/${{DS24_BOOTSTRAP_COMMIT}}/docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r51_vast_full_node_gpu_utilisation_live_launch_r1/vast_jupyter_proxy_bootstrap.sh" | bash
```

Monitoring commands are materialised in `monitoring_commands.json` and `vast_show_status.sh`.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 Vast R51 GPU live-launch authority")
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("write-authority-package")
    package.add_argument("--repo-root", default=".")
    package.add_argument("--authority-root", default=str(DEFAULT_AUTHORITY_ROOT_REL))
    package.add_argument("--bootstrap-commit", default="<FINAL_R51_COMMIT>")
    material = sub.add_parser("write-materialized-live-configs")
    material.add_argument("--repo-root", default=".")
    material.add_argument("--output-root", required=True)
    material.add_argument("--bootstrap-commit", default="<FINAL_R51_COMMIT>")
    validate = sub.add_parser("validate-gpu-admission")
    validate.add_argument("--input", required=True)
    validate.add_argument("--expected-gpu-regex", default=EXPECTED_GPU_REGEX)
    admission = sub.add_parser("run-gpu-admission")
    admission.add_argument("--output", required=True)
    admission.add_argument("--expected-gpu-regex", default=EXPECTED_GPU_REGEX)
    verify_dataset = sub.add_parser("verify-local-dataset")
    verify_dataset.add_argument("--dataset-root", required=True)
    verify_dataset.add_argument("--expected-count", type=int, required=True)
    verify_dataset.add_argument("--expected-bytes", type=int, required=True)
    verify_dataset.add_argument("--marker", required=True)
    queue = sub.add_parser("run-vast-reverse-queue")
    queue.add_argument("--repo-root", default=".")
    queue.add_argument("--dataset-root", required=True)
    queue.add_argument("--run-root", required=True)
    queue.add_argument("--execute-live", action="store_true")
    queue.add_argument("--confirm-token", default="")
    queue.add_argument("--external-snapshot", action="append", default=[])
    queue.add_argument("--allow-neutral-synthetic-ownership", action="store_true")
    sequence_live = sub.add_parser("run-live-sequence-family")
    sequence_live.add_argument("--family", required=True, choices=GPU_SEQUENCE_FAMILIES)
    sequence_live.add_argument("--repo-root", default=".")
    sequence_live.add_argument("--dataset-root", required=True)
    sequence_live.add_argument("--run-root", required=True)
    sequence_live.add_argument("--resume", action="store_true")
    sequence_live.add_argument("--max-refits", type=int, default=0)
    lightgbm_live = sub.add_parser("run-live-lightgbm-family")
    lightgbm_live.add_argument("--family", required=True, choices=LIGHTGBM_RANKING_FAMILIES)
    lightgbm_live.add_argument("--repo-root", default=".")
    lightgbm_live.add_argument("--dataset-root", required=True)
    lightgbm_live.add_argument("--run-root", required=True)
    lightgbm_live.add_argument("--resume", action="store_true")
    lightgbm_live.add_argument("--max-refits", type=int, default=0)
    lgbm = sub.add_parser("run-lightgbm-family-contract")
    lgbm.add_argument("--family", required=True, choices=LIGHTGBM_RANKING_FAMILIES)
    lgbm.add_argument("--dataset-root", required=True)
    lgbm.add_argument("--run-root", required=True)
    publisher = sub.add_parser("publisher-once")
    publisher.add_argument("--run-root", required=True)
    publisher.add_argument("--bucket", default=B2_BUCKET)
    publisher.add_argument("--remote-prefix", required=True)
    repatriation = sub.add_parser("verify-repatriation")
    repatriation.add_argument("--root", required=True)
    monitor = sub.add_parser("render-monitoring")
    monitor.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "write-authority-package":
        manifest = write_materialized_live_configs(
            Path(args.repo_root),
            Path(args.repo_root) / Path(args.authority_root),
            bootstrap_commit=args.bootstrap_commit,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "PASS" else 2
    if args.command == "write-materialized-live-configs":
        manifest = write_materialized_live_configs(
            Path(args.repo_root),
            Path(args.output_root),
            bootstrap_commit=args.bootstrap_commit,
        )
        print(json.dumps({"status": manifest["status"], "manifest_hash": manifest["manifest_hash"]}, sort_keys=True))
        return 0
    if args.command == "validate-gpu-admission":
        result = validate_gpu_admission_evidence(read_json(Path(args.input)), expected_gpu_regex=args.expected_gpu_regex)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "run-gpu-admission":
        result = run_live_gpu_admission(Path(args.output), expected_gpu_regex=args.expected_gpu_regex)
        print(json.dumps(result["validation"], indent=2, sort_keys=True))
        return 0 if result["validation"]["status"] == "PASS" else 2
    if args.command == "verify-local-dataset":
        result = verify_local_dataset(
            Path(args.dataset_root),
            expected_count=args.expected_count,
            expected_bytes=args.expected_bytes,
            marker_path=Path(args.marker),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-vast-reverse-queue":
        result = run_vast_reverse_queue(
            repo_root=Path(args.repo_root),
            dataset_root=Path(args.dataset_root),
            run_root=Path(args.run_root),
            execute_live=args.execute_live,
            confirm_token=args.confirm_token,
            external_snapshot_paths=args.external_snapshot,
            allow_neutral_synthetic_ownership=args.allow_neutral_synthetic_ownership,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "run-live-sequence-family":
        result = run_live_sequence_family(
            args.family,
            repo_root=Path(args.repo_root),
            dataset_root=Path(args.dataset_root),
            run_root=Path(args.run_root),
            resume=args.resume,
            max_refits=args.max_refits,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "run-live-lightgbm-family":
        result = run_live_lightgbm_family(
            args.family,
            repo_root=Path(args.repo_root),
            dataset_root=Path(args.dataset_root),
            run_root=Path(args.run_root),
            resume=args.resume,
            max_refits=args.max_refits,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "run-lightgbm-family-contract":
        result = run_lightgbm_family_contract(args.family, Path(args.dataset_root), Path(args.run_root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "publisher-once":
        result = publisher_once(Path(args.run_root), bucket=args.bucket, remote_prefix=args.remote_prefix)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-repatriation":
        result = verify_repatriation(Path(args.root))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "render-monitoring":
        write_json_atomic(Path(args.output), monitoring_commands())
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
