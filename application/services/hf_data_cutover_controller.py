"""Durable local controller for the finite HF-DATA-CUTOVER-R1 stage graph.

This module owns operational sequencing only. Scientific transformations remain
in their existing application services and are invoked through the established
CLI composition root.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil

from application.services.hf_data_cutover_iex_backfill import (
    BACKFILL_PACING_REQUESTS_PER_MINUTE,
    BACKFILL_REQUESTS_PER_MINUTE,
    BACKFILL_WORKERS,
    INITIAL_VALIDATED_BACKFILL_WORKERS,
    MINIMUM_FINAL_RESERVE_BYTES,
    RawBackfillDisposition,
    TRANSPORT_PROBE_PARTITIONS,
    TRANSPORT_PROBE_WORKERS,
    assess_raw_backfill_manifests,
)
from application.services.hf_data_cutover_controller_validation import (
    StageGateError,
    validate_adjusted_daily,
    validate_feature_bank,
    validate_feature_quality,
    validate_final_closeout,
    validate_hourly,
    validate_raw_authority,
    validate_targets,
)
from application.services.hf_data_cutover_controller_status import transport_progress
from application.services.hf_data_cutover_service import CutoverAuditError, atomic_json, utc_now
from infrastructure.data.iex_market_data_store import sha256_file


CONTROLLER_SCHEMA = "hf_data_cutover_r1.terminal_controller.v1"
EXPECTED_INITIAL_BACKFILL_WORKERS = 32
EXPECTED_RECOVERY_BACKFILL_WORKERS = 8
EXPECTED_REQUESTS_PER_MINUTE = 150
HEARTBEAT_SECONDS = 10.0
MAX_PROVIDER_STAGE_RETRIES = 3
RAW_ZERO_PROGRESS_LIMIT = 3


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    create_time: float

    @classmethod
    def current(cls) -> "ProcessIdentity":
        process = psutil.Process()
        return cls(process.pid, process.create_time())

    @classmethod
    def from_pid(cls, pid: int) -> "ProcessIdentity":
        process = psutil.Process(pid)
        return cls(pid, process.create_time())

    def is_live(self) -> bool:
        try:
            process = psutil.Process(self.pid)
            return process.is_running() and abs(process.create_time() - self.create_time) < 0.01
        except (psutil.Error, OSError, ValueError):
            return False


class ControllerLease:
    """Exclusive PID-reuse-safe ownership of one cutover controller."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.identity = ProcessIdentity.current()

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                identity = _identity_from_file(self.path)
                if identity is not None and identity.is_live():
                    raise RuntimeError(f"HF-DATA-CUTOVER-R1 controller already active: PID {identity.pid}")
                if identity is None and time.time() - self.path.stat().st_mtime < 2.0:
                    raise RuntimeError(f"HF-DATA-CUTOVER-R1 controller lease is being initialized: {self.path}")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({**asdict(self.identity), "acquired_at_utc": utc_now()}, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            return
        raise RuntimeError(f"Unable to acquire controller lease: {self.path}")

    def release(self) -> None:
        identity = _identity_from_file(self.path)
        if identity == self.identity:
            self.path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class Stage:
    stage_id: str
    command: str
    validator: Callable[[Path, Path], dict[str, Any]]
    provider_retryable: bool = False
    prerequisites: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage("raw_iex", "backfill", lambda repo, audit: validate_raw_authority(repo, audit), True),
    Stage("adjusted_daily", "adjust-daily", lambda repo, audit: validate_adjusted_daily(repo, audit), True, ("raw_iex",)),
    Stage("derived_1h", "derive-1h", lambda repo, audit: validate_hourly(repo, audit), False, ("raw_iex", "adjusted_daily")),
    Stage("feature_bank_v1", "features", lambda repo, audit: validate_feature_bank(repo, audit), False, ("derived_1h",)),
    Stage("feature_quality", "feature-validate", lambda repo, audit: validate_feature_quality(repo, audit), False, ("feature_bank_v1",)),
    Stage(
        "targets_h30_h60_h5_h10",
        "targets",
        lambda repo, audit: validate_targets(repo, audit),
        False,
        ("raw_iex", "adjusted_daily", "feature_quality"),
    ),
    Stage("final_validation_closeout", "validate", lambda repo, audit: validate_final_closeout(repo, audit), False, ("targets_h30_h60_h5_h10",)),
)
STAGE_BY_ID = {stage.stage_id: stage for stage in STAGES}


class CutoverController:
    """Run, resume, and validate the fixed remaining cutover graph."""

    def __init__(self, repository_root: Path, audit_root: Path) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.audit_root = audit_root.resolve(strict=True)
        self.control_root = self.audit_root / "controller"
        self.state_path = self.control_root / "CONTROLLER_STATE.json"
        self.events_path = self.control_root / "CONTROLLER_EVENTS.jsonl"
        self.logs_root = self.control_root / "stage_logs"
        self.markers_root = self.control_root / "stage_markers"
        self.attempts_root = self.control_root / "attempts"
        self.raw_progress_path = self.control_root / "RAW_PROGRESS_STATE.json"
        self.pause_path = self.control_root / "pause.request.json"
        self.stop_path = self.control_root / "stop.request.json"
        self.lease = ControllerLease(self.control_root / "controller.lock")
        self.identity = ProcessIdentity.current()
        self.started = time.monotonic()

    def run(self) -> int:
        self.lease.acquire()
        try:
            self.control_root.mkdir(parents=True, exist_ok=True)
            freeze_operational_configuration(self.repository_root, self.audit_root)
            self._event("CONTROLLER_STARTED", controller=asdict(self.identity))
            for stage in STAGES:
                requested = self._control_request()
                if requested is not None:
                    self._publish_safe_stop(requested, stage.stage_id)
                    return 0
                self._assert_reserve()
                for prerequisite in stage.prerequisites:
                    STAGE_BY_ID[prerequisite].validator(self.repository_root, self.audit_root)
                try:
                    evidence = stage.validator(self.repository_root, self.audit_root)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, StageGateError):
                    evidence = self._execute_stage(stage)
                self._mark_complete(stage, evidence)
            self._state("COMPLETE", current_stage=None, run_state="IEX_DATA_AND_FEATURE_V1_READY")
            self._event("CONTROLLER_COMPLETE", terminal_state="IEX_DATA_AND_FEATURE_V1_READY")
            return 0
        except _SafeControlExit:
            return 0
        except _TransportConnectivityBlocked as exc:
            self._state(
                "IEX_TRANSPORT_CONNECTIVITY_BLOCKED",
                current_stage="raw_iex",
                run_state=self._run_state(),
                error=str(exc),
                probe_evidence=str(self.audit_root / "IEX_TRANSPORT_CONNECTIVITY_PROBE.json"),
            )
            self._event(
                "IEX_TRANSPORT_CONNECTIVITY_BLOCKED",
                stage="raw_iex",
                error=str(exc),
                probe_evidence=str(self.audit_root / "IEX_TRANSPORT_CONNECTIVITY_PROBE.json"),
            )
            return 0
        except Exception as exc:
            status = "BLOCKED" if isinstance(exc, (CutoverAuditError, StageGateError)) else "FAILED"
            self._state(status, error=f"{type(exc).__name__}: {exc}")
            self._event("CONTROLLER_BLOCKED" if status == "BLOCKED" else "CONTROLLER_FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.lease.release()

    def _execute_stage(self, stage: Stage) -> dict[str, Any]:
        active = self._discover_stage_process(stage)
        if active is not None:
            self._event("STAGE_PROCESS_ADOPTED", stage=stage.stage_id, child=asdict(active))
            self._wait_for_process(stage, active, None, adopted=True)
            try:
                return stage.validator(self.repository_root, self.audit_root)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, StageGateError):
                if not stage.provider_retryable:
                    raise
                if stage.stage_id == "raw_iex":
                    assessment = raw_backfill_assessment(self.audit_root)
                    if not self._raw_stage_is_resumable(assessment):
                        raise StageGateError(_raw_irrecoverable_message(assessment))
                    self._record_raw_progress(assessment, source="adopted_backfill")

        if stage.stage_id == "raw_iex" and self._raw_recovery_required():
            self._run_transport_probe()
        if (
            stage.stage_id == "raw_iex"
            and (self.audit_root / "PERF1_START_STATE.json").is_file()
            and not (self.audit_root / "PERF1_TRANSPORT_SELECTION.json").is_file()
        ):
            assessment = raw_backfill_assessment(self.audit_root)
            if assessment.get("disposition") == RawBackfillDisposition.IRRECOVERABLE_FAILURE.value:
                raise StageGateError(_raw_irrecoverable_message(assessment))
            self._run_perf1()

        attempts = self._attempt_count(stage.stage_id)
        while True:
            if (
                attempts >= MAX_PROVIDER_STAGE_RETRIES
                and stage.provider_retryable
                and stage.stage_id != "raw_iex"
            ):
                raise StageGateError(
                    f"{stage.stage_id} exhausted {MAX_PROVIDER_STAGE_RETRIES} bounded controller retries"
                )
            self._assert_reserve()
            requested = self._control_request()
            if requested is not None:
                self._publish_safe_stop(requested, stage.stage_id)
                raise _SafeControlExit(requested)
            command = self._stage_command(stage)
            stdout_path = self.logs_root / f"{stage.stage_id}.stdout.log"
            stderr_path = self.logs_root / f"{stage.stage_id}.stderr.log"
            self.logs_root.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=self.repository_root,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                identity = ProcessIdentity.from_pid(process.pid)
                self._event("STAGE_PROCESS_STARTED", stage=stage.stage_id, child=asdict(identity), command=command)
                return_code = self._wait_for_process(stage, identity, process, adopted=False)
            if return_code == 0:
                try:
                    evidence = stage.validator(self.repository_root, self.audit_root)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, StageGateError):
                    if stage.stage_id != "raw_iex":
                        raise
                    assessment = raw_backfill_assessment(self.audit_root)
                    if not self._raw_stage_is_resumable(assessment):
                        raise StageGateError(_raw_irrecoverable_message(assessment))
                    attempts += 1
                    self._write_attempt_count(stage.stage_id, attempts, return_code, stderr_path)
                    progress = self._record_raw_progress(assessment, source="backfill")
                    if progress["circuit_open"]:
                        self._run_transport_probe()
                        continue
                    delay = min(30 * (2 ** min(attempts - 1, 2)), 120)
                    self._event(
                        "RAW_STAGE_RESUME_CYCLE",
                        stage=stage.stage_id,
                        attempt=attempts,
                        delay_seconds=delay,
                        assessment=assessment,
                    )
                    self._interruptible_wait(stage.stage_id, delay)
                    continue
                self._event("STAGE_VALIDATED", stage=stage.stage_id, evidence=evidence)
                return evidence
            attempts += 1
            self._write_attempt_count(stage.stage_id, attempts, return_code, stderr_path)
            if stage.stage_id == "raw_iex":
                assessment = raw_backfill_assessment(self.audit_root)
                if self._raw_stage_is_resumable(assessment):
                    progress = self._record_raw_progress(assessment, source="backfill")
                    if progress["circuit_open"]:
                        self._run_transport_probe()
                        continue
                    delay = min(30 * (2 ** min(attempts - 1, 2)), 120)
                    self._event(
                        "RAW_STAGE_RESUME_CYCLE",
                        stage=stage.stage_id,
                        attempt=attempts,
                        delay_seconds=delay,
                        assessment=assessment,
                    )
                    self._interruptible_wait(stage.stage_id, delay)
                    continue
                raise StageGateError(_raw_irrecoverable_message(assessment))
            if not stage.provider_retryable or not self._is_transient_failure(stage, stderr_path):
                raise StageGateError(
                    f"{stage.stage_id} exited {return_code}; fail-closed evidence: {stderr_path}"
                )
            delay = min(30 * (2 ** (attempts - 1)), 120)
            self._event("TRANSIENT_STAGE_RETRY", stage=stage.stage_id, attempt=attempts, delay_seconds=delay)
            self._interruptible_wait(stage.stage_id, delay)

    def _run_transport_probe(self) -> dict[str, Any]:
        probe_stage = Stage(
            stage_id="raw_iex",
            command="transport-probe",
            validator=lambda _repo, _audit: {},
            provider_retryable=False,
        )
        stdout_path = self.logs_root / "raw_transport_probe.stdout.log"
        stderr_path = self.logs_root / "raw_transport_probe.stderr.log"
        self.logs_root.mkdir(parents=True, exist_ok=True)
        command = self._stage_command(probe_stage)
        active = self._discover_command_process("transport-probe", stage_id="raw_iex")
        if active is not None:
            self._event(
                "IEX_TRANSPORT_PROBE_ADOPTED",
                stage="raw_iex",
                child=asdict(active),
            )
            return_code = self._wait_for_process(probe_stage, active, None, adopted=True)
        else:
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=self.repository_root,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                identity = ProcessIdentity.from_pid(process.pid)
                self._event(
                    "IEX_TRANSPORT_PROBE_STARTED",
                    stage="raw_iex",
                    child=asdict(identity),
                    command=command,
                )
                return_code = self._wait_for_process(probe_stage, identity, process, adopted=False)
        report_path = self.audit_root / "IEX_TRANSPORT_CONNECTIVITY_PROBE.json"
        report = _optional_json(report_path)
        status = report.get("status")
        if status == "IEX_TRANSPORT_CONNECTIVITY_BLOCKED":
            raise _TransportConnectivityBlocked(
                f"sequential probe failed; return_code={return_code}; evidence={report_path}"
            )
        if return_code not in {0, None} or status != "PASSED":
            raise StageGateError(
                "IEX transport probe failed closed: "
                f"return_code={return_code}, status={status}, evidence={report_path}, stderr={stderr_path}"
            )
        completed = int(report["completed_partition_count_after"])
        self._reset_raw_progress(completed, source="successful_transport_probe")
        self._event(
            "IEX_TRANSPORT_PROBE_PASSED",
            stage="raw_iex",
            completed_partition_count=completed,
            successful_partition_count=report.get("successful_partition_count"),
            evidence=str(report_path),
        )
        return report

    def _run_perf1(self) -> dict[str, Any]:
        perf_stage = Stage(
            stage_id="raw_iex",
            command="perf1",
            validator=lambda _repo, _audit: {},
            provider_retryable=False,
        )
        stdout_path = self.logs_root / "perf1.stdout.log"
        stderr_path = self.logs_root / "perf1.stderr.log"
        self.logs_root.mkdir(parents=True, exist_ok=True)
        active = self._discover_command_process("perf1", stage_id="raw_iex")
        if active is not None:
            self._event("PERF1_PROCESS_ADOPTED", stage="raw_iex", child=asdict(active))
            return_code = self._wait_for_process(perf_stage, active, None, adopted=True)
        else:
            command = self._stage_command(perf_stage)
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=self.repository_root,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                identity = ProcessIdentity.from_pid(process.pid)
                self._event("PERF1_PROCESS_STARTED", stage="raw_iex", child=asdict(identity), command=command)
                return_code = self._wait_for_process(perf_stage, identity, process, adopted=False)
        receipt_path = self.audit_root / "PERF1_DEPLOYMENT_RECEIPT.json"
        receipt = _optional_json(receipt_path)
        if return_code not in {0, None} or receipt.get("status") != "READY_FOR_CONTROLLER_DEPLOYMENT":
            raise StageGateError(
                f"PERF1 failed closed: return_code={return_code}, receipt={receipt_path}, stderr={stderr_path}"
            )
        self._event(
            "PERF1_READY_FOR_DEPLOYMENT",
            stage="raw_iex",
            selected_variant=receipt.get("selected_variant"),
            selected_workers=receipt.get("selected_workers"),
            evidence=str(receipt_path),
        )
        return receipt

    def _raw_recovery_required(self) -> bool:
        state = self._raw_progress_state()
        return int(state.get("consecutive_zero_progress_invocations", 0)) >= RAW_ZERO_PROGRESS_LIMIT

    def _record_raw_progress(self, assessment: Mapping[str, Any], *, source: str) -> dict[str, Any]:
        completed = int(assessment["completed_partition_count"])
        live_progress = _optional_json(self.audit_root / "PERF1_LIVE_PROGRESS.json")
        completed = max(completed, int(live_progress.get("completed_partition_count", 0)))
        state = self._raw_progress_state()
        previous = int(state.get("last_completed_partition_count", completed))
        if completed < previous:
            raise StageGateError(
                f"Raw completed partition count regressed from {previous} to {completed}; refusing recovery"
            )
        delta = completed - previous
        zero_cycles = int(state.get("consecutive_zero_progress_invocations", 0)) + 1 if delta == 0 else 0
        observation = {
            "at_utc": utc_now(),
            "source": source,
            "completed_partition_count": completed,
            "delta": delta,
        }
        payload = {
            "schema_version": "hf_data_cutover_r1.raw_progress.v1",
            "last_completed_partition_count": completed,
            "consecutive_zero_progress_invocations": zero_cycles,
            "zero_progress_limit": RAW_ZERO_PROGRESS_LIMIT,
            "circuit_open": zero_cycles >= RAW_ZERO_PROGRESS_LIMIT,
            "observations": [*list(state.get("observations") or ()), observation][-20:],
            "updated_at_utc": utc_now(),
        }
        atomic_json(self.raw_progress_path, payload)
        return payload

    def _reset_raw_progress(self, completed: int, *, source: str) -> None:
        atomic_json(
            self.raw_progress_path,
            {
                "schema_version": "hf_data_cutover_r1.raw_progress.v1",
                "last_completed_partition_count": completed,
                "consecutive_zero_progress_invocations": 0,
                "zero_progress_limit": RAW_ZERO_PROGRESS_LIMIT,
                "circuit_open": False,
                "observations": [
                    {
                        "at_utc": utc_now(),
                        "source": source,
                        "completed_partition_count": completed,
                        "delta": None,
                    }
                ],
                "updated_at_utc": utc_now(),
            },
        )

    def _raw_progress_state(self) -> dict[str, Any]:
        existing = _optional_json(self.raw_progress_path)
        if existing:
            return existing
        completed_counts: list[int] = []
        try:
            records = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            records = []
        for line in records:
            try:
                record = json.loads(line)
                if record.get("event") != "RAW_STAGE_RESUME_CYCLE":
                    continue
                completed_counts.append(int(record["assessment"]["completed_partition_count"]))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        if completed_counts:
            last = completed_counts[-1]
            trailing_same = 0
            for value in reversed(completed_counts):
                if value != last:
                    break
                trailing_same += 1
        else:
            assessment = raw_backfill_assessment(self.audit_root)
            last = int(assessment["completed_partition_count"])
            trailing_same = 0
        payload = {
            "schema_version": "hf_data_cutover_r1.raw_progress.v1",
            "last_completed_partition_count": last,
            "consecutive_zero_progress_invocations": trailing_same,
            "zero_progress_limit": RAW_ZERO_PROGRESS_LIMIT,
            "circuit_open": trailing_same >= RAW_ZERO_PROGRESS_LIMIT,
            "observations": [],
            "bootstrapped_from_event_log": bool(completed_counts),
            "updated_at_utc": utc_now(),
        }
        atomic_json(self.raw_progress_path, payload)
        return payload

    def _wait_for_process(
        self,
        stage: Stage,
        identity: ProcessIdentity,
        process: subprocess.Popen[bytes] | None,
        *,
        adopted: bool,
    ) -> int | None:
        while identity.is_live():
            requested = self._control_request()
            if requested is not None:
                self._terminate_verified(identity, stage)
                self._publish_safe_stop(requested, stage.stage_id)
                raise _SafeControlExit(requested)
            self._assert_reserve(child=identity, stage=stage)
            self._state(
                "RUNNING",
                current_stage=stage.stage_id,
                child={**asdict(identity), "adopted": adopted},
                run_state=self._run_state(),
            )
            self._publish_perf1_final_snapshot_once(stage, identity)
            time.sleep(HEARTBEAT_SECONDS)
        return process.wait() if process is not None else None

    def _publish_perf1_final_snapshot_once(self, stage: Stage, child: ProcessIdentity) -> None:
        if stage.stage_id != "raw_iex" or stage.command != "backfill":
            return
        receipt = _optional_json(self.audit_root / "PERF1_DEPLOYMENT_RECEIPT.json")
        if receipt.get("status") != "READY_FOR_CONTROLLER_DEPLOYMENT":
            return
        snapshot_path = self.audit_root / "PERF1_FINAL_VERIFICATION_SNAPSHOT.json"
        if snapshot_path.exists():
            return
        selection = _read_json(self.audit_root / "PERF1_TRANSPORT_SELECTION.json")
        hash_audit = _read_json(self.audit_root / "PERF1_COMPLETED_PARTITION_HASH_AUDIT.json")
        progress = transport_progress(self.audit_root)
        raw_owners: list[dict[str, Any]] = []
        protected_workers: list[dict[str, Any]] = []
        for process in psutil.process_iter(("pid", "ppid", "create_time", "name", "cmdline")):
            try:
                command = " ".join(process.info.get("cmdline") or ())
                folded = command.casefold()
                identity = {
                    "pid": int(process.info["pid"]),
                    "parent_pid": int(process.info["ppid"]),
                    "create_time": float(process.info["create_time"]),
                    "name": process.info.get("name"),
                    "command_line": command,
                }
                if (
                    "hf_data_cutover_r1.py" in folded
                    and any(f" {name} " in f" {folded} " for name in ("backfill", "perf1", "transport-probe"))
                    and self.audit_root.name.casefold() in folded
                ):
                    raw_owners.append(identity)
                if any(marker in folded for marker in ("ds26", "paper", "parity")):
                    protected_workers.append(identity)
            except (psutil.Error, OSError, TypeError, ValueError):
                continue
        start_state = _read_json(self.audit_root / "PERF1_START_STATE.json")
        completed = int(progress["completed_partitions"])
        payload = {
            "schema_version": "hf_data_cutover_r1.perf1_final_verification.v1",
            "captured_at_utc": utc_now(),
            "controller_live": self.identity.is_live(),
            "controller": asdict(self.identity),
            "stage": "raw_iex",
            "raw_stage_owner_count": len(raw_owners),
            "raw_stage_owners": raw_owners,
            "expected_child": asdict(child),
            "completed_partition_count": completed,
            "completed_partition_count_at_perf1_start": int(start_state["completed_partitions"]),
            "completed_count_not_decreased": completed >= int(start_state["completed_partitions"]),
            "completed_pre_perf1_hashes_match": hash_audit.get("post_benchmark_status") == "MATCH",
            "selected_transport": selection,
            "selected_transport_active": True,
            "free_bytes": shutil.disk_usage(self.repository_root).free,
            "minimum_free_bytes": MINIMUM_FINAL_RESERVE_BYTES,
            "disk_reserve_passed": shutil.disk_usage(self.repository_root).free >= MINIMUM_FINAL_RESERVE_BYTES,
            "protected_workers_observed_only": protected_workers,
            "controller_process_control_scope": "HF_DATA_CUTOVER_CHILDREN_ONLY",
        }
        atomic_json(snapshot_path, payload)
        self._event("PERF1_FINAL_VERIFICATION_SNAPSHOT", stage="raw_iex", evidence=str(snapshot_path))

    def _stage_command(self, stage: Stage) -> list[str]:
        return [
            sys.executable,
            str(self.repository_root / "scripts/local/hf_data_cutover_r1.py"),
            stage.command,
            "--repository-root",
            str(self.repository_root),
            "--output-root",
            str(self.audit_root),
        ]

    def _discover_stage_process(self, stage: Stage) -> ProcessIdentity | None:
        return self._discover_command_process(stage.command, stage_id=stage.stage_id)

    def _discover_command_process(self, command_name: str, *, stage_id: str) -> ProcessIdentity | None:
        matches: list[ProcessIdentity] = []
        audit_marker = self.audit_root.name.casefold()
        for process in psutil.process_iter(("pid", "create_time", "name", "cmdline")):
            try:
                if int(process.info["pid"]) == os.getpid():
                    continue
                if not str(process.info.get("name") or "").casefold().startswith("python"):
                    continue
                command = " ".join(process.info.get("cmdline") or ()).casefold()
                if (
                    "hf_data_cutover_r1.py" in command
                    and f" {command_name.casefold()} " in f" {command} "
                    and audit_marker in command
                ):
                    matches.append(ProcessIdentity(int(process.info["pid"]), float(process.info["create_time"])))
            except (psutil.Error, OSError, TypeError, ValueError):
                continue
        if len(matches) > 1:
            raise StageGateError(f"Multiple active processes claim stage {stage_id}: {matches}")
        return matches[0] if matches else None

    def _terminate_verified(self, identity: ProcessIdentity, stage: Stage) -> None:
        if not identity.is_live():
            return
        process = psutil.Process(identity.pid)
        command = " ".join(process.cmdline()).casefold()
        if "hf_data_cutover_r1.py" not in command or stage.command.casefold() not in command or self.audit_root.name.casefold() not in command:
            raise StageGateError(f"Refusing to stop unverified process {identity.pid}")
        process.terminate()
        try:
            process.wait(timeout=30)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        self._event("STAGE_PROCESS_STOPPED_AT_ATOMIC_BOUNDARY", stage=stage.stage_id, child=asdict(identity))

    def _assert_reserve(self, child: ProcessIdentity | None = None, stage: Stage | None = None) -> None:
        free = shutil.disk_usage(self.repository_root).free
        if free >= MINIMUM_FINAL_RESERVE_BYTES:
            return
        if child is not None and stage is not None:
            self._terminate_verified(child, stage)
        raise StageGateError(
            f"15 GiB reserve breached: free={free}, required={MINIMUM_FINAL_RESERVE_BYTES}"
        )

    def _is_transient_failure(self, stage: Stage, stderr_path: Path) -> bool:
        if stage.stage_id == "raw_iex" and self._run_state() == "IEX_RAW_DATA_INCOMPLETE":
            manifest = _optional_json(self.audit_root / "IEX_RAW_5M_MANIFEST.json")
            failures = list(manifest.get("failures") or [])
            classifications = {str(row.get("classification", "")).casefold() for row in failures}
            if classifications and classifications <= {
                "transient_provider_failure",
                "rate_limit_failure",
                "timeout",
                "connection_failure",
                "missing",
            }:
                return True
        tail = _tail_text(stderr_path, 64_000).casefold()
        transient_markers = (
            "retry-after",
            "too many requests",
            "429",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "remote end closed",
            "service unavailable",
        )
        forbidden_markers = (
            "hash mismatch",
            "identity mismatch",
            "schema",
            "reserve gate",
            "authentication",
            "entitlement",
            "off-grid",
        )
        return any(marker in tail for marker in transient_markers) and not any(
            marker in tail for marker in forbidden_markers
        )

    def _raw_stage_is_resumable(self, assessment: Mapping[str, Any]) -> bool:
        if assessment.get("disposition") != RawBackfillDisposition.RESUMABLE_INCOMPLETE.value:
            return False
        run_manifest = _optional_json(self.audit_root / "RUN_MANIFEST.json")
        if run_manifest.get("state") != "IEX_RAW_DATA_INCOMPLETE":
            return False
        recorded = (run_manifest.get("iex_raw_backfill") or {}).get("completion_disposition")
        return recorded in {None, RawBackfillDisposition.RESUMABLE_INCOMPLETE.value}

    def _interruptible_wait(self, stage_id: str, seconds: int) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            requested = self._control_request()
            if requested is not None:
                self._publish_safe_stop(requested, stage_id)
                raise _SafeControlExit(requested)
            self._state("RETRY_WAIT", current_stage=stage_id, retry_after_seconds=max(0, int(deadline - time.monotonic())))
            time.sleep(min(HEARTBEAT_SECONDS, max(0.1, deadline - time.monotonic())))

    def _control_request(self) -> str | None:
        if self.stop_path.exists():
            return "STOPPED_SAFE"
        if self.pause_path.exists():
            return "PAUSED_SAFE"
        return None

    def _publish_safe_stop(self, status: str, stage_id: str) -> None:
        self._state(status, current_stage=stage_id, run_state=self._run_state())
        self._event(status, stage=stage_id)

    def _mark_complete(self, stage: Stage, evidence: Mapping[str, Any]) -> None:
        self.markers_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "hf_data_cutover_r1.controller_stage.v1",
            "stage": stage.stage_id,
            "command": stage.command,
            "status": "COMPLETE",
            "validated_at_utc": utc_now(),
            "evidence": dict(evidence),
        }
        atomic_json(self.markers_root / f"{stage.stage_id}.json", payload)
        self._event("STAGE_COMPLETE", stage=stage.stage_id, evidence=evidence)

    def _state(self, status: str, **extra: Any) -> None:
        payload = {
            "schema_version": CONTROLLER_SCHEMA,
            "run_id": self.audit_root.name,
            "status": status,
            "controller": asdict(self.identity),
            "heartbeat_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "audit_root": str(self.audit_root),
            "event_log": str(self.events_path),
            "stage_log_root": str(self.logs_root),
            "minimum_free_bytes": MINIMUM_FINAL_RESERVE_BYTES,
            "free_bytes": shutil.disk_usage(self.repository_root).free,
            **extra,
        }
        atomic_json(self.state_path, payload)

    def _event(self, event: str, **fields: Any) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"at_utc": utc_now(), "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _run_state(self) -> str | None:
        return _optional_json(self.audit_root / "RUN_MANIFEST.json").get("state")

    def _attempt_count(self, stage_id: str) -> int:
        return int(_optional_json(self.attempts_root / f"{stage_id}.json").get("attempts", 0))

    def _write_attempt_count(self, stage_id: str, attempts: int, return_code: int | None, stderr: Path) -> None:
        atomic_json(
            self.attempts_root / f"{stage_id}.json",
            {
                "stage": stage_id,
                "attempts": attempts,
                "last_return_code": return_code,
                "last_stderr": str(stderr),
                "updated_at_utc": utc_now(),
            },
        )


class _SafeControlExit(RuntimeError):
    pass


class _TransportConnectivityBlocked(RuntimeError):
    pass


def freeze_operational_configuration(repository_root: Path, audit_root: Path) -> dict[str, Any]:
    run_manifest = _read_json(audit_root / "RUN_MANIFEST.json")
    acquisition = run_manifest.get("iex_raw_backfill") or {}
    if INITIAL_VALIDATED_BACKFILL_WORKERS != EXPECTED_INITIAL_BACKFILL_WORKERS:
        raise StageGateError("Initial validated IEX concurrency drifted from 32")
    if BACKFILL_WORKERS != EXPECTED_RECOVERY_BACKFILL_WORKERS:
        raise StageGateError("IEX transport-recovery concurrency drifted from 8")
    if acquisition.get("max_in_flight_requests") not in {
        EXPECTED_INITIAL_BACKFILL_WORKERS,
        EXPECTED_RECOVERY_BACKFILL_WORKERS,
    }:
        raise StageGateError("Recorded IEX concurrency is neither the initial 32 nor recovery 8")
    if BACKFILL_REQUESTS_PER_MINUTE != EXPECTED_REQUESTS_PER_MINUTE or acquisition.get("requests_per_minute") != EXPECTED_REQUESTS_PER_MINUTE:
        raise StageGateError("Validated shared request limiter is not the frozen value 150 requests/minute")
    if acquisition.get("minimum_final_reserve_bytes") != MINIMUM_FINAL_RESERVE_BYTES:
        raise StageGateError("Acquisition reserve contract drifted from 15 GiB")
    authority_names = (
        "IEX_HISTORICAL_COVERAGE_PROBE.json",
        "IEX_ACQUISITION_ROSTER.json",
        "IEX_SOURCE_USE_AUTHORITY.json",
        "IEX_FEATURE_BANK_V1_SPEC.json",
    )
    missing_authorities = [name for name in authority_names if not (audit_root / name).is_file()]
    if missing_authorities:
        raise StageGateError(f"Frozen controller authority evidence is missing: {missing_authorities}")
    payload = {
        "schema_version": "hf_data_cutover_r1.controller_configuration.v1",
        "run_id": audit_root.name,
        "repository_root": str(repository_root.resolve()),
        "audit_root": str(audit_root.resolve()),
        "operational_configuration_only": True,
        "scientific_authorities_refrozen": False,
        "backfill": {
            "max_in_flight_requests": EXPECTED_INITIAL_BACKFILL_WORKERS,
            "shared_requests_per_minute": EXPECTED_REQUESTS_PER_MINUTE,
            "minimum_final_reserve_bytes": MINIMUM_FINAL_RESERVE_BYTES,
            "plan_identity": acquisition.get("plan_identity"),
            "planned_5m_partitions": acquisition.get("planned_5m_partitions"),
            "planned_daily_partitions": acquisition.get("planned_daily_partitions"),
        },
        "authority_sha256": {
            name: sha256_file(audit_root / name) for name in authority_names
        },
        "stage_graph": [{"stage": stage.stage_id, "command": stage.command} for stage in STAGES],
    }
    path = audit_root / "CONTROLLER_CONFIGURATION.json"
    if path.exists():
        if _read_json(path) != payload:
            raise StageGateError("Frozen controller operational configuration drifted")
    else:
        atomic_json(path, payload)
    recovery_payload = {
        "schema_version": "hf_data_cutover_r1.transport_recovery_configuration.v1",
        "run_id": audit_root.name,
        "operational_configuration_only": True,
        "scientific_authorities_refrozen": False,
        "initial_validated_max_in_flight_requests": EXPECTED_INITIAL_BACKFILL_WORKERS,
        "recovery_max_in_flight_requests": EXPECTED_RECOVERY_BACKFILL_WORKERS,
        "transport_probe_workers": TRANSPORT_PROBE_WORKERS,
        "transport_probe_partition_count": TRANSPORT_PROBE_PARTITIONS,
        "shared_requests_per_minute_ceiling": EXPECTED_REQUESTS_PER_MINUTE,
        "pacing_requests_per_minute": BACKFILL_PACING_REQUESTS_PER_MINUTE,
        "zero_progress_invocation_limit": RAW_ZERO_PROGRESS_LIMIT,
        "minimum_final_reserve_bytes": MINIMUM_FINAL_RESERVE_BYTES,
        "authority_sha256": payload["authority_sha256"],
    }
    recovery_path = audit_root / "IEX_TRANSPORT_RECOVERY_CONFIGURATION.json"
    if recovery_path.exists():
        if _read_json(recovery_path) != recovery_payload:
            raise StageGateError("Frozen IEX transport-recovery configuration drifted")
    else:
        atomic_json(recovery_path, recovery_payload)
    return payload


def controller_status(audit_root: Path) -> dict[str, Any]:
    """Read status without mutating controller or worker state."""
    control_root = audit_root.resolve() / "controller"
    state = _optional_json(control_root / "CONTROLLER_STATE.json")
    lock_identity = _identity_from_file(control_root / "controller.lock")
    payload = dict(state) if state else {"schema_version": CONTROLLER_SCHEMA, "run_id": audit_root.name, "status": "NOT_STARTED"}
    payload["controller_live"] = bool(lock_identity and lock_identity.is_live())
    payload["run_state"] = _optional_json(audit_root / "RUN_MANIFEST.json").get("state")
    payload["transport_progress"] = transport_progress(audit_root)
    child = payload.get("child")
    if isinstance(child, dict):
        try:
            command = " ".join(psutil.Process(int(child["pid"])).cmdline()).casefold()
            if " perf1 " in f" {command} ":
                payload["transport_progress"]["acquisition_mode"] = "PERF1_BENCHMARK"
            elif " backfill " in f" {command} ":
                payload["transport_progress"]["acquisition_mode"] = (
                    "PERF1_SHARDED_BACKLOG"
                    if (audit_root / "PERF1_DEPLOYMENT_RECEIPT.json").is_file()
                    else "RECOVERY_BACKLOG"
                )
            elif " transport-probe " in f" {command} ":
                payload["transport_progress"]["acquisition_mode"] = "TRANSPORT_PROBE"
        except (psutil.Error, OSError, KeyError, TypeError, ValueError):
            pass
    return payload


def raw_backfill_assessment(audit_root: Path) -> dict[str, Any]:
    """Read the raw manifests and classify the handoff without mutating evidence."""
    five_path = audit_root / "IEX_RAW_5M_MANIFEST.json"
    daily_path = audit_root / "IEX_DAILY_MANIFEST.json"
    if not five_path.is_file() or not daily_path.is_file():
        raise StageGateError("Raw command exited without publishing both dataset manifests")
    return assess_raw_backfill_manifests(_read_json(five_path), _read_json(daily_path))


def _raw_irrecoverable_message(assessment: Mapping[str, Any]) -> str:
    return (
        "raw_iex cannot resume automatically: "
        f"disposition={assessment.get('disposition')} "
        f"unresolved={assessment.get('unresolved_partition_count')} "
        f"irrecoverable={assessment.get('irrecoverable_failure_partition_count')}"
    )


def request_control(audit_root: Path, action: str) -> Path:
    if action not in {"pause", "stop"}:
        raise ValueError(f"Unsupported control action: {action}")
    path = audit_root.resolve() / "controller" / f"{action}.request.json"
    atomic_json(path, {"action": action.upper(), "requested_at_utc": utc_now(), "requested_by_pid": os.getpid()})
    return path


def clear_control_requests(audit_root: Path) -> bool:
    control_root = audit_root.resolve() / "controller"
    lock_path = control_root / "controller.lock"
    identity = _identity_from_file(lock_path)
    if identity is not None and identity.is_live():
        return False
    if lock_path.exists() and identity is None and time.time() - lock_path.stat().st_mtime < 2.0:
        return False
    (control_root / "pause.request.json").unlink(missing_ok=True)
    (control_root / "stop.request.json").unlink(missing_ok=True)
    return True


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageGateError(f"Expected JSON object: {path}")
    return payload


def _optional_json(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError, StageGateError):
        return {}


def _identity_from_file(path: Path) -> ProcessIdentity | None:
    try:
        payload = _read_json(path)
        return ProcessIdentity(int(payload["pid"]), float(payload["create_time"]))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, StageGateError):
        return None


def _tail_text(path: Path, maximum_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - maximum_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = [
    "CutoverController",
    "ControllerLease",
    "ProcessIdentity",
    "StageGateError",
    "clear_control_requests",
    "controller_status",
    "freeze_operational_configuration",
    "raw_backfill_assessment",
    "request_control",
    "validate_adjusted_daily",
    "validate_feature_bank",
    "validate_feature_quality",
    "validate_final_closeout",
    "validate_hourly",
    "validate_raw_authority",
    "validate_targets",
]
