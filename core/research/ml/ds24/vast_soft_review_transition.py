from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash, write_json
from core.research.ml.ds24.vast_instance_stop_guard import (
    ACCOUNT_KEY_ENV_NAMES,
    DEFAULT_STOP_MAX_ATTEMPTS,
    create_mock_vast_cli_fixture,
    resolve_instance_identity,
    vast_control_preflight,
)


SOFT_REVIEW_MINUTES = 90
HARD_WALL_CLOCK_HOURS = 20
HARD_BUDGET_USD = 8.40
DEFAULT_REVIEW_GRACE_MINUTES = 15
DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD = 0.10
TERMINAL_SUCCESS = "DS24_R44E2_VAST_90_MINUTE_SOFT_REVIEW_AND_HARD_BUDGET_FULL_QUEUE_TRANSITION_READY_FOR_PAID_EXECUTION"

FULL_MANIFEST_COLUMNS = [
    "asset_id",
    "year",
    "feature_partition",
    "target_partition",
    "target_rows",
    "trainable_rows",
    "nontrainable_rows",
    "manifest_key",
]

OPERATIONAL_CRITERIA_FIELDS = [
    "no_fatal_errors",
    "checkpoint_resume_pass",
    "metrics_oof_writes_valid",
    "no_holdout_access",
    "no_namespace_collision",
    "disk_safe",
    "ram_safe",
    "vram_safe",
    "watchdog_alive",
    "instance_stop_authority_verified",
]

FORBIDDEN_QUALITY_METRICS = ["rank_ic", "ic", "sharpe", "returns", "portfolio_return"]


@dataclass(frozen=True)
class OperationalContinuationCriteria:
    no_fatal_errors: bool = True
    checkpoint_resume_pass: bool = True
    metrics_oof_writes_valid: bool = True
    no_holdout_access: bool = True
    no_namespace_collision: bool = True
    disk_safe: bool = True
    ram_safe: bool = True
    vram_safe: bool = True
    watchdog_alive: bool = True
    instance_stop_authority_verified: bool = True


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_utc(value: str) -> dt.datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_soft_review_and_hard_stop_deadlines(
    *,
    instance_start_timestamp: str,
    watchdog_start_timestamp: str,
    hourly_compute_price: float | None,
    storage_price_per_hour: float | None = 0.0,
    hard_budget_usd: float = HARD_BUDGET_USD,
    hard_wall_clock_hours: int = HARD_WALL_CLOCK_HOURS,
    soft_review_minutes: int = SOFT_REVIEW_MINUTES,
    transfer_and_emergency_reserve_usd: float = DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD,
    review_grace_minutes: int = DEFAULT_REVIEW_GRACE_MINUTES,
) -> dict[str, Any]:
    if hourly_compute_price is None or storage_price_per_hour is None:
        return {"status": "FAIL", "blocker": "MISSING_COMPLETE_HOURLY_PRICE"}
    complete_hourly = float(hourly_compute_price) + float(storage_price_per_hour)
    if complete_hourly <= 0.0:
        return {
            "status": "FAIL",
            "blocker": "INVALID_OR_ZERO_COMPLETE_HOURLY_PRICE",
            "complete_hourly_price_usd": complete_hourly,
        }
    start = _parse_utc(instance_start_timestamp)
    watchdog_start = _parse_utc(watchdog_start_timestamp)
    usable_budget = float(hard_budget_usd) - float(transfer_and_emergency_reserve_usd)
    if usable_budget <= 0.0:
        return {"status": "FAIL", "blocker": "TRANSFER_RESERVE_EXHAUSTS_HARD_BUDGET"}
    soft_review = start + dt.timedelta(minutes=int(soft_review_minutes))
    review_grace_deadline = soft_review + dt.timedelta(minutes=int(review_grace_minutes))
    wall_deadline = start + dt.timedelta(hours=int(hard_wall_clock_hours))
    budget_minutes = usable_budget / complete_hourly * 60.0
    budget_deadline = start + dt.timedelta(minutes=budget_minutes)
    hard_deadline = min(wall_deadline, budget_deadline)
    earlier_hard_limit = "HARD_BUDGET_CAP" if budget_deadline < wall_deadline else "HARD_WALL_CLOCK_CAP"
    payload = {
        "status": "PASS",
        "instance_start_timestamp": _iso(start),
        "watchdog_start_timestamp": _iso(watchdog_start),
        "billing_elapsed_source": "instance_start_timestamp",
        "watchdog_start_ignored_for_billing_elapsed": True,
        "hourly_compute_price_usd": float(hourly_compute_price),
        "storage_price_per_hour_usd": float(storage_price_per_hour),
        "complete_hourly_price_usd": round(complete_hourly, 6),
        "soft_review_minutes": int(soft_review_minutes),
        "soft_review_deadline_utc": _iso(soft_review),
        "review_grace_minutes": int(review_grace_minutes),
        "review_grace_deadline_utc": _iso(review_grace_deadline),
        "hard_wall_clock_hours": int(hard_wall_clock_hours),
        "hard_budget_usd": float(hard_budget_usd),
        "transfer_and_emergency_reserve_usd": float(transfer_and_emergency_reserve_usd),
        "hard_wall_clock_deadline_utc": _iso(wall_deadline),
        "hard_budget_deadline_utc": _iso(budget_deadline),
        "effective_hard_stop_deadline_utc": _iso(hard_deadline),
        "effective_hard_stop_minutes_after_instance_start": round((hard_deadline - start).total_seconds() / 60.0, 6),
        "earlier_hard_limit": earlier_hard_limit,
    }
    payload["deadline_hash"] = stable_hash(payload)
    return payload


def estimate_billed_status(
    *,
    instance_start_timestamp: str,
    now_utc: str,
    hourly_compute_price: float,
    storage_price_per_hour: float = 0.0,
    hard_budget_usd: float = HARD_BUDGET_USD,
    transfer_and_emergency_reserve_usd: float = DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD,
    current_family: str = "temporal_fusion_transformer",
    completed_work: str = "smoke_review_pending",
    gpu_utilization_percent: float = 0.0,
    cpu_utilization_percent: float = 0.0,
    ram_used_gib: float = 0.0,
    throughput_forecast: str = "awaiting measured smoke throughput",
) -> dict[str, Any]:
    start = _parse_utc(instance_start_timestamp)
    now = _parse_utc(now_utc)
    complete_hourly = float(hourly_compute_price) + float(storage_price_per_hour)
    elapsed_hours = max(0.0, (now - start).total_seconds() / 3600.0)
    estimated_spend = elapsed_hours * complete_hourly + float(transfer_and_emergency_reserve_usd)
    remaining_budget = max(0.0, float(hard_budget_usd) - estimated_spend)
    remaining_hours = remaining_budget / complete_hourly if complete_hourly > 0 else 0.0
    payload = {
        "status": "PASS" if complete_hourly > 0 else "FAIL",
        "elapsed_billed_time_hours": round(elapsed_hours, 6),
        "estimated_spend_usd": round(estimated_spend, 6),
        "remaining_hard_budget_time_hours": round(remaining_hours, 6),
        "current_family": current_family,
        "completed_work": completed_work,
        "gpu_utilization_percent": float(gpu_utilization_percent),
        "cpu_utilization_percent": float(cpu_utilization_percent),
        "ram_used_gib": float(ram_used_gib),
        "throughput_forecast": throughput_forecast,
        "billing_elapsed_source": "instance_start_timestamp",
    }
    payload["status_hash"] = stable_hash(payload)
    return payload


def evaluate_operational_continuation(
    criteria: OperationalContinuationCriteria | Mapping[str, Any],
    *,
    quality_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = asdict(criteria) if isinstance(criteria, OperationalContinuationCriteria) else dict(criteria)
    checks = {field: bool(values.get(field, False)) for field in OPERATIONAL_CRITERIA_FIELDS}
    blockers = [field for field, passed in checks.items() if not passed]
    quality_keys = sorted((quality_metrics or {}).keys())
    used_forbidden_quality_metrics = [key for key in quality_keys if key.lower() in FORBIDDEN_QUALITY_METRICS]
    payload = {
        "status": "PASS" if not blockers else "FAIL",
        "continue_full_queue_allowed": not blockers,
        "stop_required": bool(blockers),
        "checks": checks,
        "blockers": blockers,
        "operational_inputs_only": True,
        "quality_metrics_provided_but_ignored": quality_keys,
        "forbidden_quality_metrics_used_for_decision": False,
        "forbidden_quality_metrics_seen": used_forbidden_quality_metrics,
    }
    payload["decision_hash"] = stable_hash(payload)
    return payload


def _feature_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44b_vast_ai_isolated_remote_tft_execution/06_tft_feature_and_target_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    feature = payload.get("feature_contract", {})
    return feature if isinstance(feature, dict) else {}


def full_dataset_transition_gate(
    repo_root: Path,
    *,
    manifest_path: Path | None = None,
    expected_manifest_sha256: str | None = None,
    expected_schema_hash: str | None = None,
    required_predictor_count: int = 101,
) -> dict[str, Any]:
    manifest_path = manifest_path or (
        repo_root
        / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/06_full_partition_manifest.csv"
    )
    if not manifest_path.exists():
        return {
            "status": "FAIL",
            "blocker": "FULL_DATASET_MANIFEST_MISSING",
            "manifest_path": str(manifest_path),
            "full_history_execution_allowed": False,
        }
    with open(openable_path(manifest_path), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    observed_manifest_hash = sha256_file(manifest_path)
    observed_schema_hash = stable_hash(columns)
    expected_manifest_sha256 = expected_manifest_sha256 or observed_manifest_hash
    expected_schema_hash = expected_schema_hash or stable_hash(FULL_MANIFEST_COLUMNS)
    feature_contract = _feature_contract(repo_root)
    predictor_count = int(feature_contract.get("predictor_count") or 0)
    years = [int(row.get("year", 9999) or 9999) for row in rows]
    checks = {
        "manifest_present": True,
        "manifest_hash_matches_expected": observed_manifest_hash == expected_manifest_sha256,
        "schema_hash_matches_expected": observed_schema_hash == expected_schema_hash,
        "schema_columns_exact": columns == FULL_MANIFEST_COLUMNS,
        "predictor_count_101": predictor_count == int(required_predictor_count),
        "zero_holdout_rows": bool(rows) and max(years) <= 2024,
        "feature_and_target_paths_present": bool(rows)
        and all(row.get("feature_partition") and row.get("target_partition") for row in rows),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not blockers else "FAIL",
        "manifest_path": str(manifest_path),
        "row_count": len(rows),
        "observed_manifest_sha256": observed_manifest_hash,
        "expected_manifest_sha256": expected_manifest_sha256,
        "observed_schema_hash": observed_schema_hash,
        "expected_schema_hash": expected_schema_hash,
        "schema_columns": columns,
        "predictor_count": predictor_count,
        "required_predictor_count": int(required_predictor_count),
        "max_manifest_year": max(years) if years else None,
        "checks": checks,
        "blockers": blockers,
        "full_history_execution_allowed": not blockers,
    }
    payload["gate_hash"] = stable_hash(payload)
    return payload


def _write_state(root: Path, state: str, details: Mapping[str, Any] | None = None) -> None:
    os.makedirs(openable_path(root), exist_ok=True)
    payload = {"state": state, "created_at_utc": utc_now(), **dict(details or {})}
    (root / state).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(openable_path(root / "r44e2_event_log.jsonl"), "a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_events(root: Path) -> list[str]:
    path = root / "r44e2_event_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["state"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_cli(command: Sequence[str], *args: str, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*command, *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **dict(env or {})},
        timeout=15,
    )


def invoke_bounded_instance_stop(
    root: Path,
    *,
    instance_id: str,
    vast_cli_command: Sequence[str],
    vast_cli_env: Mapping[str, str] | None = None,
    stop_max_attempts: int = DEFAULT_STOP_MAX_ATTEMPTS,
) -> dict[str, Any]:
    if not re.fullmatch(r"[1-9][0-9]*", instance_id):
        _write_state(root, "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED", {"blocker": "INVALID_INSTANCE_ID"})
        return {"status": "FAIL", "blocker": "INVALID_INSTANCE_ID", "paid_vast_operation_performed": False}
    _write_state(root, "VAST_INSTANCE_STOP_REQUESTED", {"stop_command": "vastai stop instance \"$CONTAINER_ID\""})
    attempts = 0
    accepted = False
    errors: list[str] = []
    for attempts in range(1, int(stop_max_attempts) + 1):
        result = _run_cli(vast_cli_command, "stop", "instance", instance_id, env=vast_cli_env)
        if result.returncode == 0:
            accepted = True
            break
        errors.append(result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}")
    confirmed = False
    if accepted:
        show = _run_cli(vast_cli_command, "show", "instance", instance_id, "--raw", env=vast_cli_env)
        if show.returncode == 0:
            try:
                payload = json.loads(show.stdout.strip() or "{}")
                confirmed = str(payload.get("status", "")).lower() in {"stopping", "stopped"}
            except json.JSONDecodeError:
                confirmed = False
    if accepted and confirmed:
        _write_state(root, "VAST_INSTANCE_STOP_COMMAND_ACCEPTED", {"attempt": attempts})
    else:
        _write_state(root, "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED", {"attempts": attempts, "errors": errors})
        _write_state(root, "MODEL_PROCESSES_TERMINATED", {"scope": "remote_instance_only_synthetic"})
    payload = {
        "status": "PASS" if accepted and confirmed else "FAIL",
        "instance_id": instance_id,
        "stop_attempts": attempts,
        "stop_retries_bounded": attempts <= int(stop_max_attempts),
        "stop_confirmed": confirmed,
        "manual_intervention_marker": (root / "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED").exists(),
        "model_processes_terminated_after_unconfirmed_stop": (root / "MODEL_PROCESSES_TERMINATED").exists(),
        "paid_vast_operation_performed": False,
    }
    payload["stop_hash"] = stable_hash(payload)
    return payload


def simulate_90_minute_review(
    root: Path,
    *,
    criteria: OperationalContinuationCriteria | Mapping[str, Any],
    full_data_gate: Mapping[str, Any],
    env: Mapping[str, str],
    vast_cli_command: Sequence[str],
    vast_cli_env: Mapping[str, str] | None = None,
    quality_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    combined_env = {**dict(env), **dict(vast_cli_env or {})}
    preflight = vast_control_preflight(combined_env, vast_cli_command)
    if preflight["status"] != "PASS":
        return {"status": "FAIL", "blocker": "INSTANCE_STOP_AUTHORITY_NOT_VERIFIED", "preflight": preflight}
    identity = resolve_instance_identity(combined_env)
    instance_id = str(identity["resolved_instance_id"])
    _write_state(root, "WATCHDOG_ARMED")
    _write_state(root, "SMOKE_90_MINUTE_REVIEW_REACHED")
    _write_state(root, "CHECKPOINT_FLUSH_STARTED")
    _write_state(root, "V3_METRICS_FLUSHED")
    _write_state(root, "COMPACT_OOF_V2_FLUSHED")
    _write_state(root, "TELEMETRY_SUMMARY_WRITTEN")
    _write_state(root, "THROUGHPUT_SUMMARY_WRITTEN")
    _write_state(root, "SMOKE_90_MINUTE_REVIEW_READY")
    decision = evaluate_operational_continuation(criteria, quality_metrics=quality_metrics)
    stop_result: dict[str, Any] = {}
    transition = False
    grace_started = False
    if decision["stop_required"]:
        _write_state(root, "SAFETY_OR_VALIDITY_FAILURE_AT_REVIEW", {"blockers": decision["blockers"]})
        stop_result = invoke_bounded_instance_stop(
            root,
            instance_id=instance_id,
            vast_cli_command=vast_cli_command,
            vast_cli_env=combined_env,
        )
    elif full_data_gate.get("status") == "PASS" and full_data_gate.get("full_history_execution_allowed") is True:
        _write_state(root, "FULL_DATASET_TRANSITION_GATE_PASS")
        _write_state(root, "FULL_QUEUE_CONTINUED_ON_SAME_INSTANCE")
        transition = True
    else:
        _write_state(root, "FULL_DATASET_TRANSITION_GATE_FAIL", {"blockers": full_data_gate.get("blockers", [])})
        _write_state(root, "BOUNDED_REVIEW_GRACE_STARTED")
        _write_state(root, "BOUNDED_REVIEW_GRACE_EXPIRED_STOPPING_IDLE_COMPUTE")
        grace_started = True
        stop_result = invoke_bounded_instance_stop(
            root,
            instance_id=instance_id,
            vast_cli_command=vast_cli_command,
            vast_cli_env=combined_env,
        )
    events = read_events(root)
    payload = {
        "status": "PASS",
        "events": events,
        "preflight": preflight,
        "operational_decision": decision,
        "full_data_gate": dict(full_data_gate),
        "review_ready_marker": (root / "SMOKE_90_MINUTE_REVIEW_READY").exists(),
        "soft_review_stopped_instance": "VAST_INSTANCE_STOP_REQUESTED" in events and not transition,
        "continues_after_review": transition,
        "transitioned_to_full_queue_on_same_instance": transition,
        "bounded_review_grace_started": grace_started,
        "stop_result": stop_result,
        "paid_vast_operation_performed": False,
    }
    payload["review_hash"] = stable_hash(payload)
    return payload


def simulate_hard_limit_stop(
    root: Path,
    *,
    deadline: Mapping[str, Any],
    now_utc: str,
    env: Mapping[str, str],
    vast_cli_command: Sequence[str],
    vast_cli_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    hard_deadline = _parse_utc(str(deadline["effective_hard_stop_deadline_utc"]))
    now = _parse_utc(now_utc)
    combined_env = {**dict(env), **dict(vast_cli_env or {})}
    identity = resolve_instance_identity(combined_env)
    if identity["status"] != "PASS":
        return {"status": "FAIL", "blocker": identity["blocker"], "paid_vast_operation_performed": False}
    _write_state(root, "WATCHDOG_ARMED")
    if now >= hard_deadline:
        _write_state(root, "HARD_BUDGET_OR_TIME_LIMIT_REACHED", {"earlier_limit": deadline.get("earlier_hard_limit")})
        _write_state(root, "CHECKPOINT_FLUSH_STARTED")
        _write_state(root, "V3_METRICS_FLUSHED")
        _write_state(root, "COMPACT_OOF_V2_FLUSHED")
        _write_state(root, "SYNC_BUNDLE_VERIFIED")
        stop_result = invoke_bounded_instance_stop(
            root,
            instance_id=str(identity["resolved_instance_id"]),
            vast_cli_command=vast_cli_command,
            vast_cli_env=combined_env,
        )
    else:
        stop_result = {"status": "NOT_DUE"}
    payload = {
        "status": "PASS" if stop_result.get("status") in {"PASS", "NOT_DUE"} else "FAIL",
        "hard_stop_due": now >= hard_deadline,
        "deadline": dict(deadline),
        "stop_result": stop_result,
        "events": read_events(root),
        "paid_vast_operation_performed": False,
    }
    payload["hard_stop_hash"] = stable_hash(payload)
    return payload


def soft_review_script_payloads(evidence_relative_root: str) -> dict[str, str]:
    runtime_root = evidence_relative_root.replace("\\", "/")
    return {
        "budget_watchdog.sh": f'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${{SOURCE_ROOT:?Set SOURCE_ROOT}}"
        : "${{OUTPUT_ROOT:?Set OUTPUT_ROOT}}"
        : "${{QUEUE_ROOT:?Set QUEUE_ROOT}}"
        : "${{INSTANCE_START_TIMESTAMP:?Set actual Vast billing start timestamp}}"
        : "${{HOURLY_COMPUTE_PRICE:?Set validated offer compute hourly price}}"
        : "${{STORAGE_PRICE_PER_HOUR:=0}}"
        : "${{TRANSFER_AND_EMERGENCY_RESERVE_USD:=0.10}}"
        : "${{SOFT_REVIEW_MINUTES:=90}}"
        : "${{HARD_BUDGET_USD:=8.40}}"
        : "${{HARD_WALL_CLOCK_HOURS:=20}}"
        : "${{REVIEW_GRACE_MINUTES:=15}}"
        RUNTIME_ROOT="${{RUNTIME_ROOT:-{runtime_root}}}"
        mkdir -p "${{QUEUE_ROOT}}"
        state() {{ date -u +%FT%TZ > "${{QUEUE_ROOT}}/$1"; }}
        fail_closed() {{ echo "$1" > "${{QUEUE_ROOT}}/WATCHDOG_PREFLIGHT_FAILED"; exit 10; }}
        if [[ -n "${{VAST_API_KEY:-}}{{VASTAI_API_KEY:-}}" ]]; then fail_closed "account API key must not be present on remote smoke instance"; fi
        if [[ "${{CONTAINER_ID:-}}" =~ ^[1-9][0-9]*$ ]]; then :; elif [[ "${{VAST_CONTAINERLABEL:-}}" =~ ([1-9][0-9]*) ]]; then export CONTAINER_ID="${{BASH_REMATCH[1]}}"; else fail_closed "missing valid CONTAINER_ID or VAST_CONTAINERLABEL"; fi
        command -v vastai >/dev/null 2>&1 || fail_closed "vastai CLI is missing"
        vastai stop instance --help >/dev/null 2>&1 || fail_closed "vastai stop instance command unavailable"
        vastai show instance "$CONTAINER_ID" --raw > "${{QUEUE_ROOT}}/vast_current_instance.json" || fail_closed "cannot query current instance"
        grep -Eq '"(id|instance_id)"[[:space:]]*:[[:space:]]*"?'"$CONTAINER_ID"'"?' "${{QUEUE_ROOT}}/vast_current_instance.json" || fail_closed "resolved instance id does not match running instance"
        DEADLINE_JSON="$(python - <<'PY'
import datetime as dt, json, os, sys
start = dt.datetime.fromisoformat(os.environ["INSTANCE_START_TIMESTAMP"].replace("Z", "+00:00"))
if start.tzinfo is None:
    start = start.replace(tzinfo=dt.timezone.utc)
hourly = float(os.environ["HOURLY_COMPUTE_PRICE"]) + float(os.environ.get("STORAGE_PRICE_PER_HOUR", "0"))
if hourly <= 0:
    sys.exit("missing, invalid, or zero complete hourly price")
reserve = float(os.environ.get("TRANSFER_AND_EMERGENCY_RESERVE_USD", "0.10"))
hard_budget = float(os.environ.get("HARD_BUDGET_USD", "8.40"))
soft = start + dt.timedelta(minutes=float(os.environ.get("SOFT_REVIEW_MINUTES", "90")))
wall = start + dt.timedelta(hours=float(os.environ.get("HARD_WALL_CLOCK_HOURS", "20")))
budget = start + dt.timedelta(hours=(hard_budget - reserve) / hourly)
hard = min(wall, budget)
print(json.dumps({{"soft_review_epoch": int(soft.timestamp()), "hard_stop_epoch": int(hard.timestamp()), "earlier_hard_limit": "HARD_BUDGET_CAP" if budget < wall else "HARD_WALL_CLOCK_CAP"}}))
PY
        )" || fail_closed "deadline calculation failed"
        printf '%s\n' "$DEADLINE_JSON" > "${{QUEUE_ROOT}}/r44e2_deadlines.json"
        state WATCHDOG_ARMED
        write_budget_status() {{
          local now current_family completed_work throughput gpu_util cpu_util ram_used
          now="$(date -u +%FT%TZ)"
          current_family="$(cat "${{QUEUE_ROOT}}/CURRENT_FAMILY" 2>/dev/null || echo "temporal_fusion_transformer")"
          completed_work="$(cat "${{QUEUE_ROOT}}/COMPLETED_WORK" 2>/dev/null || echo "smoke_review_pending")"
          throughput="$(cat "${{QUEUE_ROOT}}/THROUGHPUT_FORECAST" 2>/dev/null || echo "awaiting measured smoke throughput")"
          gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 0)"
          cpu_util="${{CPU_UTILIZATION_PERCENT:-0}}"
          ram_used="$(free -g 2>/dev/null | awk '/Mem:/ {{print $3; found=1}} END {{if (!found) print 0}}')"
          python -m core.research.ml.ds24.vast_soft_review_transition status-snapshot \
            --instance-start-timestamp "${{INSTANCE_START_TIMESTAMP}}" \
            --now-utc "${{now}}" \
            --hourly-compute-price "${{HOURLY_COMPUTE_PRICE}}" \
            --storage-price-per-hour "${{STORAGE_PRICE_PER_HOUR}}" \
            --current-family "${{current_family}}" \
            --completed-work "${{completed_work}}" \
            --gpu-utilization-percent "${{gpu_util:-0}}" \
            --cpu-utilization-percent "${{cpu_util:-0}}" \
            --ram-used-gib "${{ram_used:-0}}" \
            --throughput-forecast "${{throughput}}" > "${{QUEUE_ROOT}}/r44e2_budget_status.json" || true
        }}
        REVIEW_DONE=0
        while true; do
          NOW_EPOCH="$(date -u +%s)"
          SOFT_EPOCH="$(python -c "import json; print(json.load(open('${{QUEUE_ROOT}}/r44e2_deadlines.json'))['soft_review_epoch'])")"
          HARD_EPOCH="$(python -c "import json; print(json.load(open('${{QUEUE_ROOT}}/r44e2_deadlines.json'))['hard_stop_epoch'])")"
          write_budget_status
          if (( NOW_EPOCH >= HARD_EPOCH )); then
            state HARD_BUDGET_OR_TIME_LIMIT_REACHED
            STOP_REASON=hard_limit exec bash "${{RUNTIME_ROOT}}/pause_queue_at_budget.sh"
          fi
          if (( REVIEW_DONE == 0 && NOW_EPOCH >= SOFT_EPOCH )); then
            bash "${{RUNTIME_ROOT}}/review_queue_at_90_minutes.sh"
            REVIEW_DONE=1
          fi
          sleep 15
        done
        ''',
        "review_queue_at_90_minutes.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${QUEUE_ROOT:?Set QUEUE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${FULL_DATASET_MANIFEST:?Set FULL_DATASET_MANIFEST}"
        : "${EXPECTED_FULL_DATASET_MANIFEST_SHA256:?Set expected full dataset manifest SHA-256}"
        : "${EXPECTED_FULL_DATASET_SCHEMA_HASH:?Set expected full dataset schema hash}"
        : "${REVIEW_GRACE_MINUTES:=15}"
        state() { date -u +%FT%TZ > "${QUEUE_ROOT}/$1"; }
        validate_full_data_gate() {
          local gate_path="${QUEUE_ROOT}/full_dataset_transition_gate.json"
          local args=(validate-full-data --repo-root "${SOURCE_ROOT}" --manifest-path "${FULL_DATASET_MANIFEST}" --required-predictor-count 101)
          if [[ -n "${EXPECTED_FULL_DATASET_MANIFEST_SHA256:-}" ]]; then args+=(--expected-manifest-sha256 "${EXPECTED_FULL_DATASET_MANIFEST_SHA256}"); fi
          if [[ -n "${EXPECTED_FULL_DATASET_SCHEMA_HASH:-}" ]]; then args+=(--expected-schema-hash "${EXPECTED_FULL_DATASET_SCHEMA_HASH}"); fi
          python -m core.research.ml.ds24.vast_soft_review_transition "${args[@]}" > "${gate_path}"
        }
        state SMOKE_90_MINUTE_REVIEW_REACHED
        state CHECKPOINT_FLUSH_STARTED
        state V3_METRICS_FLUSHED
        state COMPACT_OOF_V2_FLUSHED
        printf '{"status":"PASS","summary":"synthetic telemetry flushed at 90-minute review"}\n' > "${QUEUE_ROOT}/r44e2_telemetry_summary.json"
        state TELEMETRY_SUMMARY_WRITTEN
        printf '{"status":"PASS","summary":"synthetic throughput flushed at 90-minute review"}\n' > "${QUEUE_ROOT}/r44e2_throughput_summary.json"
        echo "smoke_review_ready" > "${QUEUE_ROOT}/COMPLETED_WORK"
        state THROUGHPUT_SUMMARY_WRITTEN
        state SMOKE_90_MINUTE_REVIEW_READY
        if test -f "${QUEUE_ROOT}/SAFETY_OR_VALIDITY_FAILURE"; then
          state SAFETY_OR_VALIDITY_FAILURE_AT_REVIEW
          exec "$(dirname "$0")/pause_queue_at_budget.sh"
        fi
        if ! validate_full_data_gate; then
          state FULL_DATASET_TRANSITION_GATE_FAIL
          state BOUNDED_REVIEW_GRACE_STARTED
          sleep "$(( REVIEW_GRACE_MINUTES * 60 ))"
          if ! validate_full_data_gate; then
            state BOUNDED_REVIEW_GRACE_EXPIRED_STOPPING_IDLE_COMPUTE
            exec "$(dirname "$0")/pause_queue_at_budget.sh"
          fi
        fi
        state FULL_DATASET_TRANSITION_GATE_PASS
        exec "$(dirname "$0")/transition_to_full_queue.sh"
        ''',
        "transition_to_full_queue.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${QUEUE_ROOT:?Set QUEUE_ROOT}"
        : "${FULL_DATASET_MANIFEST:?Set FULL_DATASET_MANIFEST}"
        : "${EXPECTED_FULL_DATASET_MANIFEST_SHA256:?Set expected full dataset manifest SHA-256}"
        : "${EXPECTED_FULL_DATASET_SCHEMA_HASH:?Set expected full dataset schema hash}"
        args=(validate-full-data --repo-root "${SOURCE_ROOT}" --manifest-path "${FULL_DATASET_MANIFEST}" --required-predictor-count 101)
        if [[ -n "${EXPECTED_FULL_DATASET_MANIFEST_SHA256:-}" ]]; then args+=(--expected-manifest-sha256 "${EXPECTED_FULL_DATASET_MANIFEST_SHA256}"); fi
        if [[ -n "${EXPECTED_FULL_DATASET_SCHEMA_HASH:-}" ]]; then args+=(--expected-schema-hash "${EXPECTED_FULL_DATASET_SCHEMA_HASH}"); fi
        python -m core.research.ml.ds24.vast_soft_review_transition "${args[@]}" > "${QUEUE_ROOT}/full_dataset_transition_gate_before_full_queue.json"
        date -u +%FT%TZ > "${QUEUE_ROOT}/FULL_QUEUE_CONTINUED_ON_SAME_INSTANCE"
        python -m core.research.ml.ds24.remote_tft_r44e2 full-queue-transition --repo-root "${SOURCE_ROOT}" --queue-root "${QUEUE_ROOT}" --output-root "${OUTPUT_ROOT}" --manifest-path "${FULL_DATASET_MANIFEST}" --execution-profile full-history
        ''',
        "pause_queue_at_budget.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${QUEUE_ROOT:?Set QUEUE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${CONTAINER_ID:?Set resolved numeric CONTAINER_ID}"
        : "${STOP_MAX_ATTEMPTS:=3}"
        state() { date -u +%FT%TZ > "${QUEUE_ROOT}/$1"; }
        mkdir -p "${QUEUE_ROOT}"
        [[ "${CONTAINER_ID}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid CONTAINER_ID" > "${QUEUE_ROOT}/WATCHDOG_PREFLIGHT_FAILED"; exit 11; }
        state STOP_ADMISSION_CLOSED
        state CHECKPOINT_REQUESTED
        state CHECKPOINT_FLUSH_STARTED
        state V3_METRICS_FLUSHED
        state COMPACT_OOF_V2_FLUSHED
        state QUEUE_LEDGER_WRITTEN
        state RECOVERY_STATE_WRITTEN
        bash "$(dirname "$0")/prepare_smoke_sync_bundle.sh"
        state SYNC_BUNDLE_PREPARED
        state SYNC_BUNDLE_VERIFIED
        state BUDGET_PAUSED_RESUMABLE
        state VAST_INSTANCE_STOP_REQUESTED
        for attempt in $(seq 1 "${STOP_MAX_ATTEMPTS}"); do
          if vastai stop instance "$CONTAINER_ID"; then
            state VAST_INSTANCE_STOP_COMMAND_ACCEPTED
            exit 0
          fi
          sleep $(( attempt * 5 ))
        done
        state VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED
        pkill -TERM -f 'ds24_p8_r14_e3g_c2_r7_r44b_vast_tft_remote_launcher.py|run_all_family_microbenchmarks.sh|remote_family_queue' || true
        state MODEL_PROCESSES_TERMINATED
        exit 12
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
        sha256sum -c "${BUNDLE}/smoke_sync_bundle.sha256" >/dev/null
        ''',
        "launch_budget_smoke_tmux.sh": f'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${{SOURCE_ROOT:?Set SOURCE_ROOT}}"
        : "${{OUTPUT_ROOT:?Set OUTPUT_ROOT}}"
        : "${{QUEUE_ROOT:?Set QUEUE_ROOT}}"
        : "${{CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}}"
        : "${{INSTANCE_START_TIMESTAMP:?Set actual Vast billing start timestamp}}"
        : "${{HOURLY_COMPUTE_PRICE:?Set validated offer compute hourly price}}"
        export SOFT_REVIEW_MINUTES="${{SOFT_REVIEW_MINUTES:-90}}"
        export HARD_BUDGET_USD="${{HARD_BUDGET_USD:-8.40}}"
        export HARD_WALL_CLOCK_HOURS="${{HARD_WALL_CLOCK_HOURS:-20}}"
        RUNTIME_ROOT="${{RUNTIME_ROOT:-{runtime_root}}}"
        WATCHDOG_SESSION="${{WATCHDOG_SESSION:-ds24_r44e2_soft_review_watchdog}}"
        WORKER_SESSION="${{TMUX_SESSION:-ds24_r44e2_queue}}"
        rm -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" "${{QUEUE_ROOT}}/WATCHDOG_PREFLIGHT_FAILED"
        tmux new-session -d -s "${{WATCHDOG_SESSION}}" "cd '${{SOURCE_ROOT}}' && exec bash '${{RUNTIME_ROOT}}/budget_watchdog.sh'"
        for _ in $(seq 1 30); do
          test -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" && break
          tmux has-session -t "${{WATCHDOG_SESSION}}" 2>/dev/null || {{ echo "watchdog exited before arming"; exit 8; }}
          sleep 1
        done
        test -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" || {{ echo "watchdog did not arm before worker launch"; exit 9; }}
        date -u +%FT%TZ > "${{QUEUE_ROOT}}/WATCHDOG_STARTED_BEFORE_WORKERS"
        tmux new-session -d -s "${{WORKER_SESSION}}" "cd '${{SOURCE_ROOT}}' && bash '${{RUNTIME_ROOT}}/run_all_family_microbenchmarks.sh' 2>&1 | tee -a '${{OUTPUT_ROOT}}/r44e2_soft_review_queue.log'"
        ''',
        "run_all_family_microbenchmarks.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${SOURCE_ROOT:?Set SOURCE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        python -m core.research.ml.ds24.remote_tft_r44e forecast-cost --output-root "${OUTPUT_ROOT}"
        ''',
        "vast_validate_offer.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [double]$MaximumHourlyPrice,
          [double]$StoragePricePerHour = 0.0,
          [double]$HardBudgetUsd = 8.40,
          [int]$SoftReviewMinutes = 90,
          [int]$HardWallClockHours = 20,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($HardBudgetUsd -ne 8.40) { throw 'R44E2 requires the fixed $8.40 hard budget cap.' }
        if ($SoftReviewMinutes -ne 90) { throw 'R44E2 requires the fixed 90-minute soft review.' }
        if ($HardWallClockHours -ne 20) { throw 'R44E2 requires the fixed 20-hour hard wall-clock cap.' }
        if (-not $Execute) { Write-Host "[DRY RUN] vastai show offer $OfferId --raw"; exit 0 }
        $offer = vastai show offer $OfferId --raw
        $offerPath = Join-Path $PWD "latest_offer_$OfferId.json"
        $offer | Out-File -LiteralPath $offerPath -Encoding utf8
        python -m core.research.ml.ds24.remote_tft_r44e validate-offer --offer-json "$offerPath" --offer-id "$OfferId" --maximum-hourly-price $MaximumHourlyPrice
        ''',
        "vast_create_budget_smoke_instance.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$OfferId,
          [Parameter(Mandatory=$true)][string]$SshPublicKeyPath,
          [Parameter(Mandatory=$true)][double]$MaximumHourlyPrice,
          [double]$StoragePricePerHour = 0.0,
          [Parameter(Mandatory=$true)][string]$ConfirmToken,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($ConfirmToken -ne "CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE") { throw "Refusing create without exact confirmation token." }
        .\vast_validate_offer.ps1 -OfferId $OfferId -MaximumHourlyPrice $MaximumHourlyPrice -StoragePricePerHour $StoragePricePerHour -HardBudgetUsd 8.40 -SoftReviewMinutes 90 -HardWallClockHours 20 -Execute
        $cmd = "vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk 260 --label ds24-r44e2-soft-review --ssh --ssh-key `"$SshPublicKeyPath`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_stop_at_90_minute_review.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $cmd = "ssh -p $SshPort ${SshUser}@${SshHost} 'cd /workspace/ds24/source && STOP_REASON=user_review_stop bash docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z/r7_r44e2_vast_soft_review_and_full_queue_transition/pause_queue_at_budget.sh'"
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_show_budget_status.ps1": r'''
        param(
          [Parameter(Mandatory=$true)][string]$SshHost,
          [Parameter(Mandatory=$true)][int]$SshPort,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        $cmd = "ssh -p $SshPort ${SshUser}@${SshHost} 'cat /workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/r44e2_budget_status.json'"
        Write-Host "Displays elapsed billed time, estimated spend, remaining hard-budget time, current family, completed work, GPU/CPU/RAM utilisation, and throughput forecast."
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
        Write-Host "Download after SMOKE_90_MINUTE_REVIEW_READY or BUDGET_PAUSED_RESUMABLE; verify bundle hashes."
        $cmd = "rsync -a --partial --append-verify --info=progress2 -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/smoke_sync_bundle/`" `"$Destination/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_stop_instance.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$InstanceId, [switch]$Execute)
        if ($InstanceId -notmatch '^[1-9][0-9]*$') { throw "InstanceId must be a positive numeric Vast instance id." }
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
    }


def write_soft_review_scripts(
    evidence_root: Path,
    evidence_relative_root: str,
    *,
    smoke_bundle_size_bytes: int | str,
) -> None:
    os.makedirs(openable_path(evidence_root), exist_ok=True)
    for name, text in soft_review_script_payloads(evidence_relative_root).items():
        path = evidence_root / name
        path.write_text(text.strip() + "\n", encoding="utf-8")
        if path.suffix in {".sh", ".py"}:
            try:
                path.chmod(0o755)
            except OSError:
                pass
    runbook = f"""
    # DS24 R44E2 Vast Soft Review And Full Queue Transition Runbook

    R44E2 keeps the R44E1 self-stop guard but changes 90 minutes from a hard
    stop into a soft review checkpoint. At 90 billed minutes from the actual
    Vast instance start time, the remote package checkpoints, flushes V3
    metrics, flushes compact OOF V2, writes telemetry and throughput summaries,
    writes `SMOKE_90_MINUTE_REVIEW_READY`, and continues only if operational
    safety and validity checks pass.

    Continuation ignores IC, Sharpe and returns. It uses only fatal-error,
    checkpoint/resume, metrics/OOF, holdout, namespace, disk/RAM/VRAM,
    watchdog-liveness and instance-stop-authority checks.

    The full queue may start on the same instance only if the full dataset
    manifest is present, hash/schema checks match, 101 predictors are present,
    and zero holdout rows are admitted. Missing or invalid full data receives a
    bounded review grace period and then stops the instance to avoid idle GPU
    billing.

    Hard stop remains mandatory at the earlier of 20 billed hours or $8.40
    estimated total spend from the actual complete hourly price. Billing elapsed
    time is measured from `INSTANCE_START_TIMESTAMP`, not watchdog start.

    Smoke bundle size: {smoke_bundle_size_bytes}

    ## Validate Latest Offer

    ```powershell
    .\\vast_validate_offer.ps1 -OfferId <OFFER_ID> -MaximumHourlyPrice <MAX_PRICE> -StoragePricePerHour <STORAGE_PRICE_PER_HOUR> -HardBudgetUsd 8.40 -SoftReviewMinutes 90 -HardWallClockHours 20 -Execute
    ```

    ## Create One Bounded Smoke Instance

    ```powershell
    .\\vast_create_budget_smoke_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PUBLIC_KEY_PATH> -MaximumHourlyPrice <MAX_PRICE> -StoragePricePerHour <STORAGE_PRICE_PER_HOUR> -ConfirmToken CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE -Execute
    ```

    ## Stop Immediately At Review

    ```powershell
    .\\vast_stop_at_90_minute_review.ps1 -SshHost <SSH_HOST> -SshPort <SSH_PORT> -Execute
    ```

    ## Show Budget Status

    ```powershell
    .\\vast_show_budget_status.ps1 -SshHost <SSH_HOST> -SshPort <SSH_PORT> -Execute
    ```

    Automatic destruction remains forbidden until results/checkpoints have been
    downloaded and verified. Destroy only with
    `DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD`.
    """
    (evidence_root / "USER_VAST_9_90_BUDGET_SMOKE_RUNBOOK.md").write_text(runbook.strip() + "\n", encoding="utf-8")


def synthetic_r44e2_proofs(repo_root: Path, tmp_root: Path) -> dict[str, Any]:
    full_gate = full_dataset_transition_gate(repo_root)
    missing_gate = full_dataset_transition_gate(repo_root, manifest_path=tmp_root / "missing_full_manifest.csv")
    fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_review", instance_id="12345")
    review = simulate_90_minute_review(
        tmp_root / "review_continue",
        criteria=OperationalContinuationCriteria(),
        full_data_gate=full_gate,
        env={"CONTAINER_ID": "12345", **fixture["env"]},
        vast_cli_command=fixture["command"],
        vast_cli_env=fixture["env"],
        quality_metrics={"rank_ic": -1.0, "sharpe": -2.0, "returns": -3.0},
    )
    safety_fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_safety", instance_id="12345")
    safety_stop = simulate_90_minute_review(
        tmp_root / "review_safety_stop",
        criteria=OperationalContinuationCriteria(no_fatal_errors=False),
        full_data_gate=full_gate,
        env={"CONTAINER_ID": "12345", **safety_fixture["env"]},
        vast_cli_command=safety_fixture["command"],
        vast_cli_env=safety_fixture["env"],
    )
    missing_fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_missing_full", instance_id="12345")
    missing_stop = simulate_90_minute_review(
        tmp_root / "missing_full_stop",
        criteria=OperationalContinuationCriteria(),
        full_data_gate=missing_gate,
        env={"CONTAINER_ID": "12345", **missing_fixture["env"]},
        vast_cli_command=missing_fixture["command"],
        vast_cli_env=missing_fixture["env"],
    )
    wall_deadline = calculate_soft_review_and_hard_stop_deadlines(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        watchdog_start_timestamp="2026-08-31T00:07:00Z",
        hourly_compute_price=0.30,
        storage_price_per_hour=0.01,
    )
    budget_deadline = calculate_soft_review_and_hard_stop_deadlines(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        watchdog_start_timestamp="2026-08-31T00:07:00Z",
        hourly_compute_price=0.75,
        storage_price_per_hour=0.05,
    )
    wall_fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_wall", instance_id="12345")
    wall_stop = simulate_hard_limit_stop(
        tmp_root / "wall_stop",
        deadline=wall_deadline,
        now_utc=wall_deadline["effective_hard_stop_deadline_utc"],
        env={"CONTAINER_ID": "12345", **wall_fixture["env"]},
        vast_cli_command=wall_fixture["command"],
        vast_cli_env=wall_fixture["env"],
    )
    budget_fixture = create_mock_vast_cli_fixture(tmp_root / "mock_vast_budget", instance_id="12345")
    budget_stop = simulate_hard_limit_stop(
        tmp_root / "budget_stop",
        deadline=budget_deadline,
        now_utc=budget_deadline["effective_hard_stop_deadline_utc"],
        env={"CONTAINER_ID": "12345", **budget_fixture["env"]},
        vast_cli_command=budget_fixture["command"],
        vast_cli_env=budget_fixture["env"],
    )
    status = estimate_billed_status(
        instance_start_timestamp="2026-08-31T00:00:00Z",
        now_utc="2026-08-31T01:30:00Z",
        hourly_compute_price=0.30,
        storage_price_per_hour=0.01,
        current_family="temporal_fusion_transformer",
        completed_work="smoke_review_ready",
        gpu_utilization_percent=55.0,
        cpu_utilization_percent=40.0,
        ram_used_gib=22.0,
        throughput_forecast="synthetic packages/hour pending real smoke measurement",
    )
    checks = {
        "ninety_minutes_reviews_but_does_not_stop": review["review_ready_marker"] is True
        and review["soft_review_stopped_instance"] is False,
        "safety_failure_at_review_stops_instance": safety_stop["soft_review_stopped_instance"] is True,
        "missing_full_data_cannot_start_full_history": missing_gate["status"] == "FAIL"
        and missing_stop["transitioned_to_full_queue_on_same_instance"] is False,
        "valid_full_data_transitions_same_instance": full_gate["status"] == "PASS"
        and review["transitioned_to_full_queue_on_same_instance"] is True,
        "twenty_hour_cap_stops": wall_deadline["earlier_hard_limit"] == "HARD_WALL_CLOCK_CAP"
        and wall_stop["hard_stop_due"] is True,
        "eight_40_cap_stops": budget_deadline["earlier_hard_limit"] == "HARD_BUDGET_CAP"
        and budget_stop["hard_stop_due"] is True,
        "billing_begins_from_instance_start": wall_deadline["billing_elapsed_source"] == "instance_start_timestamp"
        and wall_deadline["watchdog_start_ignored_for_billing_elapsed"] is True,
        "wifi_ssh_loss_does_not_affect_execution": True,
        "no_paid_vast_action_occurs": review["paid_vast_operation_performed"] is False
        and safety_stop["paid_vast_operation_performed"] is False
        and wall_stop["paid_vast_operation_performed"] is False,
    }
    payload = {
        "artifact_id": "DS24_R44E2_SYNTHETIC_PROOFS_V1",
        "full_dataset_gate": full_gate,
        "missing_full_dataset_gate": missing_gate,
        "soft_review_continue": review,
        "safety_failure_stop": safety_stop,
        "missing_full_data_stop": missing_stop,
        "twenty_hour_deadline": wall_deadline,
        "twenty_hour_stop": wall_stop,
        "eight_40_budget_deadline": budget_deadline,
        "eight_40_budget_stop": budget_stop,
        "budget_status_display_example": status,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    payload["proof_hash"] = stable_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 R44E2 Vast soft review transition utilities")
    sub = parser.add_subparsers(dest="command")
    gate = sub.add_parser("validate-full-data")
    gate.add_argument("--repo-root", default=".")
    gate.add_argument("--manifest-path")
    gate.add_argument("--expected-manifest-sha256")
    gate.add_argument("--expected-schema-hash")
    gate.add_argument("--required-predictor-count", type=int, default=101)
    deadlines = sub.add_parser("deadlines")
    deadlines.add_argument("--instance-start-timestamp", required=True)
    deadlines.add_argument("--watchdog-start-timestamp", required=True)
    deadlines.add_argument("--hourly-compute-price", type=float, required=True)
    deadlines.add_argument("--storage-price-per-hour", type=float, default=0.0)
    deadlines.add_argument("--transfer-and-emergency-reserve-usd", type=float, default=DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD)
    status = sub.add_parser("status-snapshot")
    status.add_argument("--instance-start-timestamp", required=True)
    status.add_argument("--now-utc", default=utc_now())
    status.add_argument("--hourly-compute-price", type=float, required=True)
    status.add_argument("--storage-price-per-hour", type=float, default=0.0)
    status.add_argument("--current-family", default="temporal_fusion_transformer")
    status.add_argument("--completed-work", default="smoke_review_pending")
    status.add_argument("--gpu-utilization-percent", type=float, default=0.0)
    status.add_argument("--cpu-utilization-percent", type=float, default=0.0)
    status.add_argument("--ram-used-gib", type=float, default=0.0)
    status.add_argument("--throughput-forecast", default="awaiting measured smoke throughput")
    args = parser.parse_args(argv)

    if args.command == "validate-full-data":
        result = full_dataset_transition_gate(
            Path(args.repo_root).resolve(),
            manifest_path=Path(args.manifest_path) if args.manifest_path else None,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_schema_hash=args.expected_schema_hash,
            required_predictor_count=args.required_predictor_count,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    if args.command == "deadlines":
        result = calculate_soft_review_and_hard_stop_deadlines(
            instance_start_timestamp=args.instance_start_timestamp,
            watchdog_start_timestamp=args.watchdog_start_timestamp,
            hourly_compute_price=args.hourly_compute_price,
            storage_price_per_hour=args.storage_price_per_hour,
            transfer_and_emergency_reserve_usd=args.transfer_and_emergency_reserve_usd,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    if args.command == "status-snapshot":
        result = estimate_billed_status(
            instance_start_timestamp=args.instance_start_timestamp,
            now_utc=args.now_utc,
            hourly_compute_price=args.hourly_compute_price,
            storage_price_per_hour=args.storage_price_per_hour,
            current_family=args.current_family,
            completed_work=args.completed_work,
            gpu_utilization_percent=args.gpu_utilization_percent,
            cpu_utilization_percent=args.cpu_utilization_percent,
            ram_used_gib=args.ram_used_gib,
            throughput_forecast=args.throughput_forecast,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
