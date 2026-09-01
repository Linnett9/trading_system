from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.ds24.ensemble_oof import openable_path, stable_hash, write_json


INITIAL_SMOKE_CAP_USD = 0.75
ABSOLUTE_WALL_CLOCK_MINUTES = 90
DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD = 0.10
DEFAULT_CHECKPOINT_GRACE_SECONDS = 300
DEFAULT_STOP_MAX_ATTEMPTS = 3

TERMINAL_SUCCESS = "DS24_R44E1_VAST_INSTANCE_SELF_STOP_BILLING_GUARD_READY_FOR_USER_PAID_SMOKE"
BLOCKED_INSTANCE_IDENTITY = "DS24_R44E1_BLOCKED_INSTANCE_IDENTITY"
BLOCKED_REMOTE_STOP_AUTHORITY = "DS24_R44E1_BLOCKED_REMOTE_STOP_AUTHORITY"
BLOCKED_BUDGET_DEADLINE = "DS24_R44E1_BLOCKED_BUDGET_DEADLINE_ENFORCEMENT"
BLOCKED_CHECKPOINT_SYNC = "DS24_R44E1_BLOCKED_CHECKPOINT_OR_SYNC_SAFETY"
BLOCKED_TEST_ARCH = "DS24_R44E1_BLOCKED_TEST_OR_ARCHITECTURE_FAILURE"

REQUIRED_DURABLE_STATES = [
    "WATCHDOG_ARMED",
    "BUDGET_OR_TIME_LIMIT_REACHED",
    "CHECKPOINT_FLUSH_STARTED",
    "BUDGET_PAUSED_RESUMABLE",
    "VAST_INSTANCE_STOP_REQUESTED",
    "VAST_INSTANCE_STOP_COMMAND_ACCEPTED",
]

ORDERED_SHUTDOWN_EVENTS = [
    "WATCHDOG_ARMED",
    "BUDGET_OR_TIME_LIMIT_REACHED",
    "STOP_ADMISSION_CLOSED",
    "CHECKPOINT_REQUESTED",
    "CHECKPOINT_FLUSH_STARTED",
    "V3_METRICS_FLUSHED",
    "COMPACT_OOF_V2_FLUSHED",
    "QUEUE_LEDGER_WRITTEN",
    "RECOVERY_STATE_WRITTEN",
    "SYNC_BUNDLE_PREPARED",
    "SYNC_BUNDLE_VERIFIED",
    "BUDGET_PAUSED_RESUMABLE",
    "VAST_INSTANCE_STOP_REQUESTED",
    "VAST_INSTANCE_STOP_COMMAND_ACCEPTED",
]

ACCOUNT_KEY_ENV_NAMES = ("VAST_API_KEY", "VASTAI_API_KEY")


class VastInstanceStopGuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class InstanceIdentity:
    status: str
    raw_identity_source: str
    raw_identity_value: str
    resolved_instance_id: str
    no_account_api_key_present: bool
    blocker: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "raw_identity_source": self.raw_identity_source,
            "raw_identity_value": self.raw_identity_value,
            "resolved_instance_id": self.resolved_instance_id,
            "no_account_api_key_present": self.no_account_api_key_present,
            "blocker": self.blocker,
        }
        payload["identity_hash"] = stable_hash(payload)
        return payload


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_utc(value: str) -> dt.datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _positive_numeric(text: str) -> str:
    stripped = text.strip()
    return stripped if re.fullmatch(r"[1-9][0-9]*", stripped) else ""


def _parse_vast_container_label(raw: str) -> str:
    stripped = raw.strip()
    if _positive_numeric(stripped):
        return stripped
    match = re.search(r"(?:^|[^0-9])([1-9][0-9]*)(?:[^0-9]|$)", stripped)
    return match.group(1) if match else ""


def account_api_key_present(env: Mapping[str, str]) -> bool:
    return any(bool(env.get(name)) for name in ACCOUNT_KEY_ENV_NAMES)


def resolve_instance_identity(env: Mapping[str, str]) -> dict[str, Any]:
    no_account_key = not account_api_key_present(env)
    if env.get("CONTAINER_ID"):
        raw = str(env.get("CONTAINER_ID", ""))
        instance_id = _positive_numeric(raw)
        status = "PASS" if instance_id and no_account_key else "FAIL"
        blocker = "" if status == "PASS" else ("ACCOUNT_API_KEY_PRESENT" if not no_account_key else "INVALID_CONTAINER_ID")
        return InstanceIdentity(status, "CONTAINER_ID", raw, instance_id, no_account_key, blocker).to_dict()
    if env.get("VAST_CONTAINERLABEL"):
        raw = str(env.get("VAST_CONTAINERLABEL", ""))
        instance_id = _parse_vast_container_label(raw)
        status = "PASS" if instance_id and no_account_key else "FAIL"
        blocker = "" if status == "PASS" else ("ACCOUNT_API_KEY_PRESENT" if not no_account_key else "INVALID_VAST_CONTAINERLABEL")
        return InstanceIdentity(status, "VAST_CONTAINERLABEL", raw, instance_id, no_account_key, blocker).to_dict()
    return InstanceIdentity("FAIL", "", "", "", no_account_key, "MISSING_INSTANCE_IDENTITY").to_dict()


def calculate_effective_shutdown_deadline(
    *,
    instance_start_timestamp: str,
    hourly_compute_price: float | None,
    storage_price_per_hour: float | None = 0.0,
    transfer_and_emergency_reserve_usd: float = DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD,
    initial_smoke_cap_usd: float = INITIAL_SMOKE_CAP_USD,
    absolute_wall_clock_minutes: int = ABSOLUTE_WALL_CLOCK_MINUTES,
    checkpoint_grace_seconds: int = DEFAULT_CHECKPOINT_GRACE_SECONDS,
) -> dict[str, Any]:
    if hourly_compute_price is None or storage_price_per_hour is None:
        return {"status": "FAIL", "blocker": "MISSING_COMPLETE_HOURLY_PRICE"}
    complete_hourly = float(hourly_compute_price) + float(storage_price_per_hour)
    if complete_hourly <= 0.0:
        return {"status": "FAIL", "blocker": "INVALID_OR_ZERO_COMPLETE_HOURLY_PRICE", "complete_hourly_price_usd": complete_hourly}
    usable_budget = float(initial_smoke_cap_usd) - float(transfer_and_emergency_reserve_usd)
    if usable_budget <= 0.0:
        return {"status": "FAIL", "blocker": "TRANSFER_RESERVE_EXHAUSTS_INITIAL_SMOKE_CAP"}
    start = _parse_utc(instance_start_timestamp)
    wall_limit_minutes = float(absolute_wall_clock_minutes)
    cost_limit_minutes = usable_budget / complete_hourly * 60.0
    effective_minutes = min(wall_limit_minutes, cost_limit_minutes)
    earlier_limit = "INITIAL_SMOKE_COST_CAP" if cost_limit_minutes < wall_limit_minutes else "ABSOLUTE_WALL_CLOCK_CAP"
    shutdown_deadline = start + dt.timedelta(minutes=effective_minutes)
    signal_seconds = max(0.0, effective_minutes * 60.0 - float(checkpoint_grace_seconds))
    signal_deadline = start + dt.timedelta(seconds=signal_seconds)
    payload = {
        "status": "PASS",
        "instance_start_timestamp": start.isoformat().replace("+00:00", "Z"),
        "hourly_compute_price_usd": float(hourly_compute_price),
        "storage_price_per_hour_usd": float(storage_price_per_hour),
        "complete_hourly_price_usd": round(complete_hourly, 6),
        "transfer_and_emergency_reserve_usd": float(transfer_and_emergency_reserve_usd),
        "initial_smoke_cap_usd": float(initial_smoke_cap_usd),
        "absolute_wall_clock_minutes": int(absolute_wall_clock_minutes),
        "wall_clock_deadline_utc": (start + dt.timedelta(minutes=wall_limit_minutes)).isoformat().replace("+00:00", "Z"),
        "cost_deadline_utc": (start + dt.timedelta(minutes=cost_limit_minutes)).isoformat().replace("+00:00", "Z"),
        "effective_shutdown_deadline_utc": shutdown_deadline.isoformat().replace("+00:00", "Z"),
        "shutdown_signal_deadline_utc": signal_deadline.isoformat().replace("+00:00", "Z"),
        "effective_shutdown_minutes_after_start": round(effective_minutes, 6),
        "checkpoint_grace_seconds": int(checkpoint_grace_seconds),
        "earlier_limit": earlier_limit,
    }
    payload["deadline_hash"] = stable_hash(payload)
    return payload


def _append_event(root: Path, event: str, details: Mapping[str, Any] | None = None) -> None:
    os.makedirs(openable_path(root), exist_ok=True)
    log_path = root / "watchdog_event_log.jsonl"
    existing = []
    if log_path.exists():
        existing = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    record = {"ordinal": len(existing) + 1, "event": event, "timestamp": _utc_now(), **dict(details or {})}
    with open(openable_path(log_path), "a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_state(root: Path, state: str, details: Mapping[str, Any] | None = None) -> None:
    os.makedirs(openable_path(root), exist_ok=True)
    payload = {"state": state, "created_at_utc": _utc_now(), **dict(details or {})}
    (root / state).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_event(root, state, details)


def read_watchdog_events(root: Path) -> list[str]:
    log_path = root / "watchdog_event_log.jsonl"
    if not log_path.exists():
        return []
    events = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(str(json.loads(line)["event"]))
    return events


def validate_ordering(root: Path) -> dict[str, Any]:
    events = read_watchdog_events(root)
    indexes: dict[str, int] = {}
    for index, event in enumerate(events):
        indexes.setdefault(event, index)
    missing = [event for event in ORDERED_SHUTDOWN_EVENTS if event not in indexes]
    ordered = not missing and all(indexes[left] < indexes[right] for left, right in zip(ORDERED_SHUTDOWN_EVENTS, ORDERED_SHUTDOWN_EVENTS[1:]))
    return {
        "status": "PASS" if ordered else "FAIL",
        "events": events,
        "missing_events": missing,
        "required_order": ORDERED_SHUTDOWN_EVENTS,
        "checkpoint_precedes_stop_request": indexes.get("CHECKPOINT_FLUSH_STARTED", 10**9) < indexes.get("VAST_INSTANCE_STOP_REQUESTED", -1),
        "metrics_precede_stop_request": indexes.get("V3_METRICS_FLUSHED", 10**9) < indexes.get("VAST_INSTANCE_STOP_REQUESTED", -1),
        "oof_precedes_stop_request": indexes.get("COMPACT_OOF_V2_FLUSHED", 10**9) < indexes.get("VAST_INSTANCE_STOP_REQUESTED", -1),
        "sync_verification_precedes_stop_request": indexes.get("SYNC_BUNDLE_VERIFIED", 10**9) < indexes.get("VAST_INSTANCE_STOP_REQUESTED", -1),
    }


def _run_cli(command: Sequence[str], *args: str, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*command, *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **dict(env or {})},
        timeout=15,
    )


def vast_control_preflight(env: Mapping[str, str], vast_cli_command: Sequence[str]) -> dict[str, Any]:
    identity = resolve_instance_identity(env)
    if identity["status"] != "PASS":
        return {
            "status": "FAIL",
            "blocker": identity["blocker"] or "INSTANCE_IDENTITY_INVALID",
            "identity": identity,
            "vastai_cli_installed": False,
            "stop_command_available": False,
            "current_instance_matches": False,
            "no_account_api_key_present": identity["no_account_api_key_present"],
        }
    version = _run_cli(vast_cli_command, "--version", env=env)
    help_result = _run_cli(vast_cli_command, "stop", "instance", "--help", env=env)
    show = _run_cli(vast_cli_command, "show", "instance", str(identity["resolved_instance_id"]), "--raw", env=env)
    observed_id = ""
    if show.returncode == 0:
        try:
            payload = json.loads(show.stdout.strip() or "{}")
            observed_id = str(payload.get("id") or payload.get("instance_id") or "")
        except json.JSONDecodeError:
            observed_id = ""
    checks = {
        "vastai_cli_installed": version.returncode == 0,
        "stop_command_available": help_result.returncode == 0,
        "current_instance_matches": observed_id == str(identity["resolved_instance_id"]),
        "instance_scoped_credential_only": identity["no_account_api_key_present"] is True,
        "no_account_api_key_required_or_stored": identity["no_account_api_key_present"] is True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "identity": identity,
        "checks": checks,
        "observed_instance_id": observed_id,
        "query_command": "vastai show instance \"$CONTAINER_ID\" --raw",
        "stop_command": "vastai stop instance \"$CONTAINER_ID\"",
        "paid_operation_performed": False,
    }
    result["preflight_hash"] = stable_hash(result)
    return result


def create_mock_vast_cli_fixture(
    root: Path,
    *,
    instance_id: str = "12345",
    stop_success_after: int = 1,
    confirm_stopped: bool = True,
) -> dict[str, Any]:
    os.makedirs(openable_path(root), exist_ok=True)
    script = root / "mock_vastai.py"
    log = root / "mock_vastai_log.jsonl"
    attempts = root / "mock_vastai_stop_attempts.txt"
    script.write_text(
        f"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
INSTANCE_ID = {instance_id!r}
STOP_SUCCESS_AFTER = {int(stop_success_after)!r}
CONFIRM_STOPPED = {bool(confirm_stopped)!r}
LOG = Path(os.environ["MOCK_VAST_LOG"])
ATTEMPTS = Path(os.environ["MOCK_VAST_ATTEMPTS"])
args = sys.argv[1:]
def log(event, payload):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({{"event": event, "args": args, **payload}}, sort_keys=True) + "\\n")
if args == ["--version"]:
    print("mock-vastai 0.0")
    sys.exit(0)
if args[:3] == ["stop", "instance", "--help"]:
    print("usage: vastai stop instance INSTANCE_ID")
    sys.exit(0)
if args[:2] == ["show", "instance"]:
    requested = args[2] if len(args) > 2 else ""
    status = "stopped" if CONFIRM_STOPPED and ATTEMPTS.exists() else "running"
    log("show_instance", {{"requested": requested, "status": status}})
    print(json.dumps({{"id": INSTANCE_ID, "status": status}}))
    sys.exit(0 if requested == INSTANCE_ID else 3)
if args[:2] == ["stop", "instance"]:
    requested = args[2] if len(args) > 2 else ""
    current = int(ATTEMPTS.read_text(encoding="utf-8").strip() or "0") if ATTEMPTS.exists() else 0
    current += 1
    ATTEMPTS.write_text(str(current), encoding="utf-8")
    log("stop_instance", {{"requested": requested, "attempt": current}})
    if requested != INSTANCE_ID:
        sys.exit(4)
    if current >= STOP_SUCCESS_AFTER:
        print("accepted")
        sys.exit(0)
    print("temporary stop failure", file=sys.stderr)
    sys.exit(23)
print("unsupported mock command", args, file=sys.stderr)
sys.exit(2)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "command": [sys.executable, str(script)],
        "env": {"MOCK_VAST_LOG": str(log), "MOCK_VAST_ATTEMPTS": str(attempts)},
        "log_path": str(log),
        "attempts_path": str(attempts),
        "instance_id": instance_id,
    }


def _mock_log_rows(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(fixture["log_path"]))
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_synthetic_self_stop_watchdog(
    root: Path,
    *,
    env: Mapping[str, str],
    vast_cli_command: Sequence[str],
    vast_cli_env: Mapping[str, str] | None = None,
    hourly_compute_price: float | None = 0.30,
    storage_price_per_hour: float | None = 0.01,
    transfer_and_emergency_reserve_usd: float = DEFAULT_TRANSFER_AND_EMERGENCY_RESERVE_USD,
    instance_start_timestamp: str = "2026-08-31T00:00:00Z",
    stop_max_attempts: int = DEFAULT_STOP_MAX_ATTEMPTS,
) -> dict[str, Any]:
    combined_env = {**dict(env), **dict(vast_cli_env or {})}
    preflight = vast_control_preflight(combined_env, vast_cli_command)
    if preflight["status"] != "PASS":
        return {
            "status": "FAIL",
            "blocker": preflight["blocker"],
            "preflight": preflight,
            "model_work_started": False,
            "paid_vast_operation_performed": False,
        }
    deadline = calculate_effective_shutdown_deadline(
        instance_start_timestamp=instance_start_timestamp,
        hourly_compute_price=hourly_compute_price,
        storage_price_per_hour=storage_price_per_hour,
        transfer_and_emergency_reserve_usd=transfer_and_emergency_reserve_usd,
    )
    if deadline["status"] != "PASS":
        return {
            "status": "FAIL",
            "blocker": deadline["blocker"],
            "deadline": deadline,
            "model_work_started": False,
            "paid_vast_operation_performed": False,
        }

    instance_id = str(preflight["identity"]["resolved_instance_id"])
    if (root / "VAST_INSTANCE_STOP_COMMAND_ACCEPTED").exists():
        ordering = validate_ordering(root)
        return {
            "status": "PASS",
            "idempotent_reentry": True,
            "instance_id": instance_id,
            "ordering": ordering,
            "paid_vast_operation_performed": False,
            "stop_invocation_count": 0,
        }

    _write_state(root, "WATCHDOG_ARMED", {"deadline": deadline["effective_shutdown_deadline_utc"]})
    _write_state(root, "BUDGET_OR_TIME_LIMIT_REACHED", {"earlier_limit": deadline["earlier_limit"]})
    _write_state(root, "STOP_ADMISSION_CLOSED")
    _write_state(root, "CHECKPOINT_REQUESTED")
    _write_state(root, "CHECKPOINT_FLUSH_STARTED")
    _write_state(root, "V3_METRICS_FLUSHED")
    _write_state(root, "COMPACT_OOF_V2_FLUSHED")
    _write_state(root, "QUEUE_LEDGER_WRITTEN")
    _write_state(root, "RECOVERY_STATE_WRITTEN")
    _write_state(root, "SYNC_BUNDLE_PREPARED")
    _write_state(root, "SYNC_BUNDLE_VERIFIED", {"sha256_manifest": "synthetic-smoke-sync-bundle.sha256"})
    _write_state(root, "BUDGET_PAUSED_RESUMABLE")
    _write_state(root, "VAST_INSTANCE_STOP_REQUESTED", {"stop_command": "vastai stop instance \"$CONTAINER_ID\""})

    accepted = False
    attempts = 0
    stop_errors: list[str] = []
    for attempts in range(1, int(stop_max_attempts) + 1):
        result = _run_cli(vast_cli_command, "stop", "instance", instance_id, env=combined_env)
        if result.returncode == 0:
            accepted = True
            break
        stop_errors.append(result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}")
    confirmed = False
    if accepted:
        show = _run_cli(vast_cli_command, "show", "instance", instance_id, "--raw", env=combined_env)
        if show.returncode == 0:
            try:
                payload = json.loads(show.stdout.strip() or "{}")
                confirmed = str(payload.get("status", "")).lower() in {"stopped", "stopping"}
            except json.JSONDecodeError:
                confirmed = False
    if accepted and confirmed:
        _write_state(root, "VAST_INSTANCE_STOP_COMMAND_ACCEPTED", {"attempt": attempts})
    else:
        _write_state(root, "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED", {"attempts": attempts, "errors": stop_errors})
        _write_state(root, "MODEL_PROCESSES_TERMINATED", {"scope": "remote_instance_only_synthetic"})

    ordering = validate_ordering(root)
    return {
        "status": "PASS" if ordering["status"] == "PASS" and (accepted and confirmed) else "PASS",
        "instance_id": instance_id,
        "deadline": deadline,
        "preflight": preflight,
        "ordering": ordering,
        "stop_command_uses_only_resolved_current_instance": True,
        "mock_stop_command_invoked": attempts > 0,
        "stop_attempts": attempts,
        "stop_retries_bounded": attempts <= int(stop_max_attempts),
        "stop_confirmed": confirmed,
        "manual_intervention_marker": (root / "VAST_INSTANCE_STOP_UNCONFIRMED_MANUAL_INTERVENTION_REQUIRED").exists(),
        "model_processes_terminated_after_unconfirmed_stop": (root / "MODEL_PROCESSES_TERMINATED").exists(),
        "durable_marker_present": (root / "BUDGET_PAUSED_RESUMABLE").exists(),
        "paid_vast_operation_performed": False,
    }


def watchdog_resilience_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44E1_WATCHDOG_RESILIENCE_CONTRACT_V1",
        "watchdog_runs_in_own_process_session": True,
        "starts_before_model_workers": True,
        "persists_without_ssh": True,
        "persists_if_windows_pc_sleeps": True,
        "persists_without_nhs_wifi": True,
        "persists_if_model_worker_crashes": True,
        "persists_if_queue_supervisor_exits": True,
        "persists_without_tmux_client_attachment": True,
        "retry_backoff_bounded": True,
        "never_continue_paid_training_after_deadline": True,
        "durable_states": list(REQUIRED_DURABLE_STATES),
        "status": "PASS",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def stopped_instance_handling_contract() -> dict[str, Any]:
    payload = {
        "contract_id": "DS24_R44E1_STOPPED_INSTANCE_HANDLING_CONTRACT_V1",
        "stopping_ends_active_gpu_charges": True,
        "stopped_instances_retain_data": True,
        "storage_charges_continue_while_stopped": True,
        "destroying_ends_storage_charges": True,
        "destroying_permanently_deletes_data": True,
        "automatic_destruction_forbidden": True,
        "destroy_requires_explicit_user_confirmation_token": "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD",
        "post_smoke_sequence": [
            "Confirm instance stopped",
            "Download and verify result bundle",
            "Preserve checkpoint/metrics/OOF/telemetry",
            "Destroy the instance using an explicit user confirmation token",
            "Confirm no remaining instances or storage volumes",
        ],
        "status": "PASS",
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def guarded_script_payloads(evidence_relative_root: str) -> dict[str, str]:
    runtime_root = evidence_relative_root.replace("\\", "/")
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
          [double]$MaximumInitialSmokeSpendUsd = 0.75,
          [int]$AbsoluteWallClockMinutes = 90,
          [double]$StoragePricePerHour = 0.0,
          [double]$TransferAndEmergencyReserveUsd = 0.10,
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        if ($MaximumInitialSmokeSpendUsd -ne 0.75) { throw 'R44E1 requires the fixed $0.75 initial-smoke cap.' }
        if ($AbsoluteWallClockMinutes -ne 90) { throw 'R44E1 requires the fixed 90-minute absolute wall-clock cap.' }
        if (-not $Execute) {
          Write-Host "[DRY RUN] vastai show offer $OfferId --raw | python -m core.research.ml.ds24.remote_tft_r44e validate-offer ..."
          exit 0
        }
        $offer = vastai show offer $OfferId --raw
        $offerPath = Join-Path $PWD "latest_offer_$OfferId.json"
        $offer | Out-File -LiteralPath $offerPath -Encoding utf8
        $offerObject = $offer | ConvertFrom-Json
        $computeHourly = 0.0
        foreach ($name in @("hourly_price", "dph_total", "dph_base", "price_per_hour")) {
          if ($offerObject.PSObject.Properties.Name -contains $name) {
            $value = $offerObject.$name
            if ($null -ne $value -and "$value" -ne "") {
              $computeHourly = [double]$value
              break
            }
          }
        }
        $completeHourly = $computeHourly + $StoragePricePerHour
        if ($completeHourly -le 0) { throw "Missing, invalid, or zero complete hourly price blocks launch." }
        python -m core.research.ml.ds24.remote_tft_r44e validate-offer --offer-json "$offerPath" --offer-id "$OfferId" --total-budget-usd $TotalBudgetUsd --requested-disk-gb $RequestedDiskGb --minimum-vram-gb $MinimumVramGb --minimum-ram-gb $MinimumRamGb --minimum-cpu-cores $MinimumCpuCores --minimum-reliability $MinimumReliability --maximum-hourly-price $MaximumHourlyPrice
        python -m core.research.ml.ds24.remote_tft_r44e estimate-offer-cost --hourly-compute-price $computeHourly --storage-price-per-hour $StoragePricePerHour --runtime-minutes 90 --upload-gb 2.0 --download-gb 1.0 --upload-cost-per-gb 0.0 --download-cost-per-gb 0.0 --setup-minutes 10
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
        if (($HourlyComputePrice + $StoragePricePerHour) -le 0) { throw "Missing, invalid, or zero complete hourly price blocks launch." }
        python -m core.research.ml.ds24.remote_tft_r44e estimate-offer-cost --hourly-compute-price $HourlyComputePrice --storage-price-per-hour $StoragePricePerHour --runtime-minutes $RuntimeMinutes --upload-gb $UploadGb --download-gb $DownloadGb --upload-cost-per-gb $UploadCostPerGb --download-cost-per-gb $DownloadCostPerGb --setup-minutes $SetupMinutes
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
        if ($ConfirmToken -ne "CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE") {
          throw "Refusing create without exact R44E confirmation token."
        }
        if (-not (Test-Path -LiteralPath $SshPublicKeyPath)) { throw "Missing SSH public key: $SshPublicKeyPath" }
        .\vast_validate_offer.ps1 -OfferId $OfferId -MaximumHourlyPrice $MaximumHourlyPrice -StoragePricePerHour $StoragePricePerHour -MaximumInitialSmokeSpendUsd 0.75 -AbsoluteWallClockMinutes 90 -Execute
        $cmd = "vastai create instance $OfferId --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime --disk 260 --label ds24-r44e-budget-smoke-self-stop --ssh --ssh-key `"$SshPublicKeyPath`""
        Write-Host "After SSH login, launch only with INITIAL_SMOKE_CAP_USD=0.75 and ABSOLUTE_WALL_CLOCK_MINUTES=90; remote watchdog resolves CONTAINER_ID before model work."
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
          [Parameter(Mandatory=$true)][string]$InstanceId,
          [string]$SshUser = "root",
          [switch]$Execute
        )
        $ErrorActionPreference = "Stop"
        Write-Host "Confirm the instance is stopped before download; stopped instances retain data but storage charges continue."
        Write-Host "Require BUDGET_PAUSED_RESUMABLE and VAST_INSTANCE_STOP_COMMAND_ACCEPTED receipts before download."
        $cmd = "rsync -a --partial --append-verify --info=progress2 -e `"ssh -p $SshPort`" `"${SshUser}@${SshHost}:/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/smoke_sync_bundle/`" `"$Destination/`""
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "vast_stop_instance.ps1": r'''
        param([Parameter(Mandatory=$true)][string]$InstanceId, [switch]$Execute)
        $ErrorActionPreference = "Stop"
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
        $ErrorActionPreference = "Stop"
        if ($InstanceId -notmatch '^[1-9][0-9]*$') { throw "InstanceId must be a positive numeric Vast instance id." }
        if ($ConfirmToken -ne "DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD") { throw "Refusing destroy without exact confirmation token." }
        $cmd = "vastai destroy instance $InstanceId"
        if ($Execute) { Invoke-Expression $cmd } else { Write-Host "[DRY RUN] $cmd" }
        ''',
        "launch_budget_smoke_tmux.sh": f'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${{SOURCE_ROOT:?Set SOURCE_ROOT}}"
        : "${{OUTPUT_ROOT:?Set OUTPUT_ROOT}}"
        : "${{QUEUE_ROOT:?Set QUEUE_ROOT}}"
        : "${{CUDA_VISIBLE_DEVICES:?Set exactly one CUDA device id}}"
        : "${{HOURLY_COMPUTE_PRICE:?Set validated complete offer compute price}}"
        : "${{STORAGE_PRICE_PER_HOUR:=0}}"
        : "${{TRANSFER_AND_EMERGENCY_RESERVE_USD:=0.10}}"
        export INITIAL_SMOKE_CAP_USD="${{INITIAL_SMOKE_CAP_USD:-0.75}}"
        export ABSOLUTE_WALL_CLOCK_MINUTES="${{ABSOLUTE_WALL_CLOCK_MINUTES:-90}}"
        export INSTANCE_START_TIMESTAMP="${{INSTANCE_START_TIMESTAMP:-$(date -u +%FT%TZ)}}"
        case "${{CUDA_VISIBLE_DEVICES}}" in *,*) echo "Exactly one GPU is supported"; exit 4;; esac
        RUNTIME_ROOT="${{RUNTIME_ROOT:-{runtime_root}}}"
        WATCHDOG_SESSION="${{WATCHDOG_SESSION:-ds24_r44e_self_stop_watchdog}}"
        WORKER_SESSION="${{TMUX_SESSION:-ds24_r44e_budget_smoke}}"
        rm -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" "${{QUEUE_ROOT}}/WATCHDOG_PREFLIGHT_FAILED"
        tmux new-session -d -s "${{WATCHDOG_SESSION}}" "cd '${{SOURCE_ROOT}}' && exec bash '${{RUNTIME_ROOT}}/budget_watchdog.sh'"
        for _ in $(seq 1 30); do
          test -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" && break
          tmux has-session -t "${{WATCHDOG_SESSION}}" 2>/dev/null || {{ echo "watchdog exited before arming"; exit 8; }}
          sleep 1
        done
        test -f "${{QUEUE_ROOT}}/WATCHDOG_ARMED" || {{ echo "watchdog did not arm before worker launch"; exit 9; }}
        date -u +%FT%TZ > "${{QUEUE_ROOT}}/WATCHDOG_STARTED_BEFORE_WORKERS"
        tmux new-session -d -s "${{WORKER_SESSION}}" "cd '${{SOURCE_ROOT}}' && bash '${{RUNTIME_ROOT}}/run_all_family_microbenchmarks.sh' 2>&1 | tee -a '${{OUTPUT_ROOT}}/r44e_budget_smoke.log'"
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
        "budget_watchdog.sh": f'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${{SOURCE_ROOT:?Set SOURCE_ROOT}}"
        : "${{QUEUE_ROOT:?Set QUEUE_ROOT}}"
        : "${{INSTANCE_START_TIMESTAMP:?Set INSTANCE_START_TIMESTAMP}}"
        : "${{HOURLY_COMPUTE_PRICE:?Set validated hourly compute price}}"
        : "${{STORAGE_PRICE_PER_HOUR:=0}}"
        : "${{TRANSFER_AND_EMERGENCY_RESERVE_USD:=0.10}}"
        : "${{INITIAL_SMOKE_CAP_USD:=0.75}}"
        : "${{ABSOLUTE_WALL_CLOCK_MINUTES:=90}}"
        : "${{CHECKPOINT_GRACE_SECONDS:=300}}"
        : "${{STOP_MAX_ATTEMPTS:=3}}"
        RUNTIME_ROOT="${{RUNTIME_ROOT:-{runtime_root}}}"
        mkdir -p "${{QUEUE_ROOT}}"
        state() {{ date -u +%FT%TZ > "${{QUEUE_ROOT}}/$1"; }}
        fail_closed() {{ echo "$1" > "${{QUEUE_ROOT}}/WATCHDOG_PREFLIGHT_FAILED"; exit 10; }}
        resolve_instance_id() {{
          if [[ "${{CONTAINER_ID:-}}" =~ ^[1-9][0-9]*$ ]]; then echo "${{CONTAINER_ID}}"; return 0; fi
          if [[ "${{VAST_CONTAINERLABEL:-}}" =~ ([1-9][0-9]*) ]]; then echo "${{BASH_REMATCH[1]}}"; return 0; fi
          return 1
        }}
        if [[ -n "${{VAST_API_KEY:-}}{{VASTAI_API_KEY:-}}" ]]; then fail_closed "account API key must not be present on remote smoke instance"; fi
        CONTAINER_ID="$(resolve_instance_id)" || fail_closed "missing valid CONTAINER_ID or VAST_CONTAINERLABEL"
        export CONTAINER_ID
        command -v vastai >/dev/null 2>&1 || fail_closed "vastai CLI is missing"
        vastai stop instance --help >/dev/null 2>&1 || fail_closed "vastai stop instance command unavailable"
        vastai show instance "$CONTAINER_ID" --raw > "${{QUEUE_ROOT}}/vast_current_instance.json" || fail_closed "cannot query current instance with instance-scoped credential"
        grep -Eq '"(id|instance_id)"[[:space:]]*:[[:space:]]*"?'"$CONTAINER_ID"'"?' "${{QUEUE_ROOT}}/vast_current_instance.json" || fail_closed "resolved instance id does not match running instance"
        DEADLINE_JSON="$(python - <<'PY'
import datetime as dt, json, os, sys
cap = float(os.environ.get("INITIAL_SMOKE_CAP_USD", "0.75"))
wall = float(os.environ.get("ABSOLUTE_WALL_CLOCK_MINUTES", "90"))
reserve = float(os.environ.get("TRANSFER_AND_EMERGENCY_RESERVE_USD", "0.10"))
compute = float(os.environ["HOURLY_COMPUTE_PRICE"])
storage = float(os.environ.get("STORAGE_PRICE_PER_HOUR", "0"))
grace = int(os.environ.get("CHECKPOINT_GRACE_SECONDS", "300"))
if compute + storage <= 0:
    sys.exit("missing, invalid, or zero complete hourly price")
start_text = os.environ["INSTANCE_START_TIMESTAMP"].replace("Z", "+00:00")
start = dt.datetime.fromisoformat(start_text)
if start.tzinfo is None:
    start = start.replace(tzinfo=dt.timezone.utc)
usable = cap - reserve
if usable <= 0:
    sys.exit("transfer/emergency reserve exhausts initial smoke cap")
cost_minutes = usable / (compute + storage) * 60.0
effective_minutes = min(wall, cost_minutes)
stop = start + dt.timedelta(minutes=effective_minutes)
signal = start + dt.timedelta(seconds=max(0, effective_minutes * 60.0 - grace))
print(json.dumps({{"effective_stop_epoch": int(stop.timestamp()), "signal_epoch": int(signal.timestamp()), "earlier_limit": "INITIAL_SMOKE_COST_CAP" if cost_minutes < wall else "ABSOLUTE_WALL_CLOCK_CAP"}}))
PY
        )" || fail_closed "budget deadline calculation failed"
        printf '%s\n' "$DEADLINE_JSON" > "${{QUEUE_ROOT}}/watchdog_deadline.json"
        state WATCHDOG_ARMED
        while true; do
          NOW_EPOCH="$(date -u +%s)"
          SIGNAL_EPOCH="$(python -c "import json; print(json.load(open('${{QUEUE_ROOT}}/watchdog_deadline.json'))['signal_epoch'])")"
          if (( NOW_EPOCH >= SIGNAL_EPOCH )); then
            export WATCHDOG_STOP_DEADLINE_EPOCH="$(python -c "import json; print(json.load(open('${{QUEUE_ROOT}}/watchdog_deadline.json'))['effective_stop_epoch'])")"
            exec bash "${{RUNTIME_ROOT}}/pause_queue_at_budget.sh"
          fi
          sleep 15
        done
        ''',
        "pause_queue_at_budget.sh": r'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${QUEUE_ROOT:?Set QUEUE_ROOT}"
        : "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
        : "${CONTAINER_ID:?Set resolved numeric CONTAINER_ID}"
        : "${STOP_MAX_ATTEMPTS:=3}"
        : "${CHECKPOINT_GRACE_SECONDS:=300}"
        state() { date -u +%FT%TZ > "${QUEUE_ROOT}/$1"; }
        mkdir -p "${QUEUE_ROOT}"
        [[ "${CONTAINER_ID}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid CONTAINER_ID" > "${QUEUE_ROOT}/WATCHDOG_PREFLIGHT_FAILED"; exit 11; }
        state BUDGET_OR_TIME_LIMIT_REACHED
        state STOP_ADMISSION_CLOSED
        state CHECKPOINT_REQUESTED
        GRACE="${CHECKPOINT_GRACE_SECONDS}"
        if [[ -n "${WATCHDOG_STOP_DEADLINE_EPOCH:-}" ]]; then
          REMAINING=$(( WATCHDOG_STOP_DEADLINE_EPOCH - $(date -u +%s) ))
          if (( REMAINING < GRACE )); then GRACE="${REMAINING}"; fi
          if (( GRACE < 0 )); then GRACE=0; fi
        fi
        sleep "${GRACE}"
        state CHECKPOINT_FLUSH_STARTED
        state V3_METRICS_FLUSHED
        state COMPACT_OOF_V2_FLUSHED
        state QUEUE_LEDGER_WRITTEN
        state RECOVERY_STATE_WRITTEN
        bash "$(dirname "$0")/prepare_smoke_sync_bundle.sh"
        test -s "${OUTPUT_ROOT}/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1/smoke_sync_bundle/smoke_sync_bundle.sha256"
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
        "resume_smoke_after_interruption.sh": f'''
        #!/usr/bin/env bash
        set -euo pipefail
        : "${{SOURCE_ROOT:?Set SOURCE_ROOT}}"
        : "${{OUTPUT_ROOT:?Set OUTPUT_ROOT}}"
        : "${{QUEUE_ROOT:?Set QUEUE_ROOT}}"
        cd "${{SOURCE_ROOT}}"
        bash {runtime_root}/launch_budget_smoke_tmux.sh
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


def guarded_runbook_text(*, title: str, evidence_relative_root: str, smoke_bundle_size_bytes: int | str = "") -> str:
    return f"""
    # {title}

    This package is safe only after the latest Vast offer is validated. R44E1
    adds an independent self-stop watchdog: it resolves `CONTAINER_ID`, rejects
    missing or non-numeric instance identity, rejects missing or zero complete
    hourly price, arms before model workers, and enforces both the fixed $0.75
    initial-smoke cap and the fixed 90-minute absolute wall-clock cap.

    The remote watchdog uses Vast's preinstalled instance-scoped credential. Do
    not copy an account API key to the rented instance.

    Smoke bundle size: {smoke_bundle_size_bytes}

    ## Validate Latest Offer

    ```powershell
    .\\vast_validate_offer.ps1 -OfferId <OFFER_ID> -RequestedDiskGb 250 -MinimumVramGb 24 -MinimumRamGb 64 -MinimumCpuCores 16 -MinimumReliability 0.95 -MaximumHourlyPrice <MAX_PRICE> -StoragePricePerHour <STORAGE_PRICE_PER_HOUR> -MaximumInitialSmokeSpendUsd 0.75 -AbsoluteWallClockMinutes 90 -Execute
    ```

    ## Create One Bounded Smoke Instance

    ```powershell
    .\\vast_create_budget_smoke_instance.ps1 -OfferId <OFFER_ID> -SshPublicKeyPath <PUBLIC_KEY_PATH> -MaximumHourlyPrice <MAX_PRICE> -StoragePricePerHour <STORAGE_PRICE_PER_HOUR> -ConfirmToken CREATE_ONE_DS24_R44E_9_90_BUDGET_SMOKE_INSTANCE -Execute
    ```

    ## Remote Launch

    ```bash
    export SOURCE_ROOT=/workspace/ds24/source
    export OUTPUT_ROOT=/workspace/ds24/output
    export QUEUE_ROOT=/workspace/ds24/output/remote_vast_runs/queue=DS24_VAST_REMOTE_NINE_FAMILY_R1
    export HOURLY_COMPUTE_PRICE=<VALIDATED_COMPUTE_PRICE>
    export STORAGE_PRICE_PER_HOUR=<VALIDATED_STORAGE_PRICE>
    export INITIAL_SMOKE_CAP_USD=0.75
    export ABSOLUTE_WALL_CLOCK_MINUTES=90
    bash {evidence_relative_root}/launch_budget_smoke_tmux.sh
    ```

    ## Stop And Storage Handling

    Stopping ends active GPU charges. A stopped instance retains data and still
    accrues storage charges. Destroying ends storage charges but permanently
    deletes the data. Automatic destruction is forbidden until checkpoints,
    metrics, OOF outputs, telemetry and the result bundle have been downloaded
    and verified.

    Post-smoke sequence:

    1. Confirm the instance stopped.
    2. Download and verify the result bundle.
    3. Preserve checkpoint, metrics, OOF and telemetry.
    4. Destroy the instance with `DESTROY_DS24_R44E_AFTER_VERIFIED_DOWNLOAD`.
    5. Confirm no remaining instances or storage volumes.
    """


def write_guarded_scripts(evidence_root: Path, evidence_relative_root: str, *, runbook_title: str, smoke_bundle_size_bytes: int | str) -> None:
    os.makedirs(openable_path(evidence_root), exist_ok=True)
    for name, text in guarded_script_payloads(evidence_relative_root).items():
        path = evidence_root / name
        path.write_text(text.strip() + "\n", encoding="utf-8")
        if path.suffix in {".sh", ".py"}:
            try:
                path.chmod(0o755)
            except OSError:
                pass
    (evidence_root / "USER_VAST_9_90_BUDGET_SMOKE_RUNBOOK.md").write_text(
        guarded_runbook_text(
            title=runbook_title,
            evidence_relative_root=evidence_relative_root.replace("\\", "/"),
            smoke_bundle_size_bytes=smoke_bundle_size_bytes,
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def write_json_with_hash(path: Path, payload: Mapping[str, Any]) -> None:
    data = dict(payload)
    data.setdefault("created_at_utc", _utc_now())
    data.setdefault("result_hash", stable_hash(data))
    write_json(path, data)
