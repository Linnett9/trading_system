from __future__ import annotations

import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import copy
import errno
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, BinaryIO, Callable, Iterator, Mapping, Protocol, Sequence, TextIO
from uuid import uuid4


GIB = 1024**3
DELL_WORKER_THREAD_ENVIRONMENT: Mapping[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class CleanV2ResourcePolicy:
    """One versioned operational policy for every clean-V2 runtime surface."""

    policy_id: str = "DS24_CLEAN_V2_OPERATIONAL_MEMORY_POLICY_V4"
    target_available_physical_gib: int = 5
    warning_available_physical_gib: int = 6
    emergency_available_physical_gib: int = 4
    readmission_available_physical_gib: int = 7
    system_commit_headroom_gib: int = 12
    windows_job_aggregate_commit_limit_gib: int = 24
    allocation_safety_factor: float = 1.5
    pressure_checkpoint_grace_seconds: int = 30
    maximum_dell_model_workers: int = 3
    dell_canary_active_worker_limit: int = 2
    maximum_mac_model_workers: int = 1
    maximum_admission_bypasses: int = 12
    allocation_request_poll_seconds: float = 0.25
    allocation_request_timeout_seconds: int = 35
    resource_deferral_cooldown_seconds: int = 30
    material_capacity_change_gib: int = 1
    automatic_retry_allowed: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "resource_policy_id": self.policy_id,
            "units": "GiB (1 GiB = 1024^3 bytes)",
            "target_available_physical_gib": self.target_available_physical_gib,
            "warning_available_physical_gib": self.warning_available_physical_gib,
            "emergency_available_physical_gib": self.emergency_available_physical_gib,
            "readmission_available_physical_gib": self.readmission_available_physical_gib,
            "system_commit_headroom_gib": self.system_commit_headroom_gib,
            "windows_job_aggregate_commit_limit_gib": self.windows_job_aggregate_commit_limit_gib,
            "allocation_safety_factor": self.allocation_safety_factor,
            "pressure_checkpoint_grace_seconds": self.pressure_checkpoint_grace_seconds,
            "maximum_dell_model_workers": self.maximum_dell_model_workers,
            "dell_canary_active_worker_limit": (
                self.dell_canary_active_worker_limit
            ),
            "maximum_mac_model_workers": self.maximum_mac_model_workers,
            "maximum_admission_bypasses": self.maximum_admission_bypasses,
            "allocation_request_poll_seconds": self.allocation_request_poll_seconds,
            "allocation_request_timeout_seconds": self.allocation_request_timeout_seconds,
            "resource_deferral_cooldown_seconds": (
                self.resource_deferral_cooldown_seconds
            ),
            "material_capacity_change_gib": self.material_capacity_change_gib,
            "automatic_retry_allowed": self.automatic_retry_allowed,
            "dell_worker_thread_environment": dict(
                DELL_WORKER_THREAD_ENVIRONMENT
            ),
            "family_resource_profiles": resource_profile_registry_payload(),
            "family_scheduling_profiles": scheduling_profile_registry_payload(),
            "supervisor_reserved_allocation_stages": [
                "panel_assembly",
                "preprocessing",
                "sequence_materialization",
                "fit",
                "score",
                "cache_growth",
            ],
            "windows_job_limit_scope": (
                "ONE_AGGREGATE_COMMITTED_MEMORY_LIMIT_FOR_ALL_OWNED_WORKERS_"
                "AND_DESCENDANTS"
            ),
            "unsafe_pressure_action": (
                "STOP_ADMISSIONS_CHECKPOINT_AT_SAFE_BOUNDARY_THEN_TERMINATE_ONLY_"
                "IDENTITY_VERIFIED_OWNED_WORKERS_IF_EMERGENCY_OR_GRACE_EXPIRES"
            ),
        }


RESOURCE_POLICY = CleanV2ResourcePolicy()
RECOVERY_RESOURCE_POLICY_ID = RESOURCE_POLICY.policy_id
RECOVERY_MIN_AVAILABLE_PHYSICAL_BYTES = RESOURCE_POLICY.target_available_physical_gib * GIB
RECOVERY_MIN_COMMIT_HEADROOM_BYTES = RESOURCE_POLICY.system_commit_headroom_gib * GIB
RECOVERY_ALLOCATION_SAFETY_FACTOR = RESOURCE_POLICY.allocation_safety_factor

RUNTIME_JSON_READ_TIMEOUT_SECONDS = 0.5
RUNTIME_JSON_RETRY_INITIAL_SECONDS = 0.01
RUNTIME_JSON_RETRY_MAX_SECONDS = 0.1


class RuntimeJsonAccessError(RuntimeError):
    """Base class for exhausted transient runtime JSON access."""


class RuntimeJsonReadError(RuntimeJsonAccessError):
    """Raised after bounded retries cannot read an atomically published JSON file."""


class RuntimeJsonWriteError(RuntimeJsonAccessError):
    """Raised after bounded retries cannot publish an atomic JSON file."""


def _is_transient_file_access_error(error: OSError) -> bool:
    return bool(
        isinstance(error, PermissionError)
        or getattr(error, "winerror", None) in {5, 32, 33}
        or error.errno in {errno.EACCES, errno.EBUSY}
    )


def _retry_event(
    *,
    operation: str,
    path: Path,
    attempt: int,
    delay_seconds: float,
    error: OSError,
) -> dict[str, Any]:
    return {
        "classification": "DS24_CLEAN_V2_TRANSIENT_RUNTIME_FILE_RETRY",
        "operation": operation,
        "path": str(path),
        "attempt": attempt,
        "delay_seconds": delay_seconds,
        "error_type": type(error).__name__,
        "error": str(error),
        "winerror": getattr(error, "winerror", None),
    }


def read_json_object_with_retry(
    path: Path,
    *,
    timeout_seconds: float = RUNTIME_JSON_READ_TIMEOUT_SECONDS,
    on_retry: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read an atomic JSON object, retrying only transient access denials."""

    source = Path(path)
    started = time.monotonic()
    attempt = 0
    delay = RUNTIME_JSON_RETRY_INITIAL_SECONDS
    while True:
        attempt += 1
        try:
            text = source.read_text(encoding="utf-8")
            break
        except OSError as error:
            if not _is_transient_file_access_error(error):
                raise
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeJsonReadError(
                    "Transient runtime JSON access did not recover within "
                    f"{timeout_seconds:.3f}s after {attempt} attempts: {source}"
                ) from error
            sleep_seconds = min(delay, remaining)
            if on_retry is not None:
                on_retry(
                    _retry_event(
                        operation="read",
                        path=source,
                        attempt=attempt,
                        delay_seconds=sleep_seconds,
                        error=error,
                    )
                )
            time.sleep(sleep_seconds)
            delay = min(delay * 2, RUNTIME_JSON_RETRY_MAX_SECONDS)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return payload


def write_json_object_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float = RUNTIME_JSON_READ_TIMEOUT_SECONDS,
    on_retry: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    """Durably publish one JSON object through a unique same-directory file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the unique temporary basename short: CLEAN V2 evidence paths can
    # already approach the legacy Windows MAX_PATH boundary.
    temporary = destination.parent / f".{os.getpid()}.{uuid4().hex[:8]}.tmp"
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        started = time.monotonic()
        attempt = 0
        delay = RUNTIME_JSON_RETRY_INITIAL_SECONDS
        while True:
            attempt += 1
            try:
                os.replace(temporary, destination)
                break
            except OSError as error:
                if not _is_transient_file_access_error(error):
                    raise
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise RuntimeJsonWriteError(
                        "Atomic runtime JSON replace did not recover within "
                        f"{timeout_seconds:.3f}s after {attempt} attempts: "
                        f"{destination}"
                    ) from error
                sleep_seconds = min(delay, remaining)
                if on_retry is not None:
                    on_retry(
                        _retry_event(
                            operation="replace",
                            path=destination,
                            attempt=attempt,
                            delay_seconds=sleep_seconds,
                            error=error,
                        )
                    )
                time.sleep(sleep_seconds)
                delay = min(delay * 2, RUNTIME_JSON_RETRY_MAX_SECONDS)
    finally:
        temporary.unlink(missing_ok=True)


WINDOWS_JOB_COMMIT_LIMIT_BYTES = RESOURCE_POLICY.windows_job_aggregate_commit_limit_gib * GIB
RESOURCE_PRESSURE_EXIT_CODE = 75
RESOURCE_CAPACITY_DEFERRED_EXIT_CODE = 76
LOCAL_CAPACITY_DEFERRAL = "LOCAL_CAPACITY_DEFERRAL"
HOST_RESOURCE_PRESSURE = "HOST_RESOURCE_PRESSURE"
SINGLE_WORKER_CAPACITY_BLOCK = "SINGLE_WORKER_CAPACITY_BLOCK"


class CleanV2ResourcePressure(RuntimeError):
    """Raised before an allocation when the memory policy fails closed."""

    def __init__(self, decision: "MemoryDecision") -> None:
        self.decision = decision
        super().__init__(f"{decision.stage}:" + ",".join(decision.blocking_reasons))


class WorkerContainmentError(RuntimeError):
    """Raised when aggregate worker containment cannot be proved."""


class ResourceReservationUnavailable(RuntimeError):
    """Raised when the supervisor does not grant a large allocation in time."""

    def __init__(self, stage: str, response: Mapping[str, Any]) -> None:
        self.stage = stage
        self.response = dict(response)
        reasons = self.response.get("blocking_reasons") or ["RESERVATION_TIMEOUT"]
        super().__init__(f"{stage}:" + ",".join(map(str, reasons)))


@dataclass(frozen=True)
class MemorySnapshot:
    available_physical_bytes: int | None
    total_physical_bytes: int | None
    commit_headroom_bytes: int | None
    commit_limit_bytes: int | None
    committed_bytes: int | None
    source: str
    checked_at_utc: str
    peak_committed_bytes: int | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryDecision:
    stage: str
    purpose: str
    safe: bool
    admission_allowed: bool
    emergency: bool
    pressure_level: str
    estimated_allocation_bytes: int
    guarded_allocation_bytes: int
    projected_available_physical_bytes: int | None
    projected_commit_headroom_bytes: int | None
    required_available_physical_bytes: int
    required_commit_headroom_bytes: int
    blocking_reasons: tuple[str, ...]
    warning_reasons: tuple[str, ...]
    snapshot: MemorySnapshot
    estimated_physical_bytes: int | None = None
    estimated_commit_bytes: int | None = None
    guarded_physical_bytes: int | None = None
    guarded_commit_bytes: int | None = None

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot"] = self.snapshot.payload()
        result["resource_policy"] = recovery_policy_payload()
        return result


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    alive: bool
    creation_time_utc: str
    command_line: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    pid: int
    resident_bytes: int | None
    peak_resident_bytes: int | None
    private_commit_bytes: int | None
    peak_private_commit_bytes: int | None
    source: str
    checked_at_utc: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobMemorySnapshot:
    committed_bytes: int
    peak_committed_bytes: int
    commit_limit_bytes: int
    active_processes: int
    total_processes: int
    terminated_processes: int
    limit_violation_detected: bool
    installation_verified: bool
    checked_at_utc: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AllocationEstimate:
    """Separate incremental physical and committed-memory estimates."""

    physical_bytes: int
    commit_bytes: int

    def __post_init__(self) -> None:
        if self.physical_bytes < 0 or self.commit_bytes < 0:
            raise ValueError("Allocation estimates must be non-negative")

    @classmethod
    def same(cls, byte_count: int) -> "AllocationEstimate":
        return cls(int(byte_count), int(byte_count))

    def guarded(self) -> "AllocationEstimate":
        factor = RESOURCE_POLICY.allocation_safety_factor
        return AllocationEstimate(
            physical_bytes=int(math.ceil(self.physical_bytes * factor)),
            commit_bytes=int(math.ceil(self.commit_bytes * factor)),
        )

    def payload(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerReservationOwner:
    family: str
    attempt_generation: int
    pid: int = 0
    process_creation_time_utc: str = ""

    @property
    def logical_key(self) -> tuple[str, int]:
        return (self.family, self.attempt_generation)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReservationDecision:
    granted: bool
    classification: str
    blocking_reasons: tuple[str, ...]
    worker_reservation_id: str | None
    stage_reservation_id: str | None
    stage: str
    requested: AllocationEstimate
    guarded_requested: AllocationEstimate
    outstanding: AllocationEstimate
    projected_available_physical_bytes: int | None
    projected_system_commit_headroom_bytes: int | None
    projected_job_commit_bytes: int | None
    memory_snapshot: MemorySnapshot
    job_snapshot: JobMemorySnapshot | None

    def payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "requested": self.requested.payload(),
            "guarded_requested": self.guarded_requested.payload(),
            "outstanding": self.outstanding.payload(),
            "memory_snapshot": self.memory_snapshot.payload(),
            "job_snapshot": (
                self.job_snapshot.payload() if self.job_snapshot is not None else None
            ),
            "resource_policy": recovery_policy_payload(),
        }


@dataclass(frozen=True)
class CapacitySnapshot:
    """Comparable capacity state used to gate resource-deferred readmission."""

    available_physical_bytes: int | None
    system_commit_headroom_bytes: int | None
    job_committed_bytes: int | None
    outstanding_physical_bytes: int
    outstanding_commit_bytes: int
    active_worker_reservations: int
    active_families: tuple[str, ...]
    checked_at_utc: str

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["active_families"] = list(self.active_families)
        return result


def recovery_policy_payload() -> dict[str, Any]:
    return RESOURCE_POLICY.payload()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


def _windows_memory_snapshot() -> MemorySnapshot | None:
    """Read true system commit charge/limit, not process commit availability."""

    if os.name != "nt":
        return None
    performance = _PERFORMANCE_INFORMATION()
    performance.cb = ctypes.sizeof(performance)
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_performance_info = psapi.GetPerformanceInfo
        get_performance_info.argtypes = [ctypes.POINTER(_PERFORMANCE_INFORMATION), wintypes.DWORD]
        get_performance_info.restype = wintypes.BOOL
        succeeded = get_performance_info(ctypes.byref(performance), ctypes.sizeof(performance))
    except (AttributeError, OSError):
        return None
    if not succeeded or not performance.PageSize:
        return None
    page_size = int(performance.PageSize)
    commit_limit = int(performance.CommitLimit) * page_size
    committed = int(performance.CommitTotal) * page_size
    return MemorySnapshot(
        available_physical_bytes=int(performance.PhysicalAvailable) * page_size,
        total_physical_bytes=int(performance.PhysicalTotal) * page_size,
        commit_headroom_bytes=max(0, commit_limit - committed),
        commit_limit_bytes=commit_limit,
        committed_bytes=committed,
        peak_committed_bytes=int(performance.CommitPeak) * page_size,
        source="psapi.GetPerformanceInfo.system_commit",
        checked_at_utc=_utc_now(),
    )


def system_memory_snapshot() -> MemorySnapshot:
    """Measure physical availability and true system-wide commit headroom."""

    windows = _windows_memory_snapshot()
    if windows is not None:
        return windows
    try:
        import psutil

        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        commit_limit = int(virtual.total) + int(swap.total)
        committed = int(virtual.total - virtual.available) + int(swap.used)
        return MemorySnapshot(
            available_physical_bytes=int(virtual.available),
            total_physical_bytes=int(virtual.total),
            commit_headroom_bytes=max(0, commit_limit - committed),
            commit_limit_bytes=commit_limit,
            committed_bytes=committed,
            peak_committed_bytes=None,
            source="psutil.system_memory_plus_swap",
            checked_at_utc=_utc_now(),
        )
    except Exception:
        return MemorySnapshot(
            available_physical_bytes=None,
            total_physical_bytes=None,
            commit_headroom_bytes=None,
            commit_limit_bytes=None,
            committed_bytes=None,
            peak_committed_bytes=None,
            source="unavailable",
            checked_at_utc=_utc_now(),
        )


def _pressure_level(snapshot: MemorySnapshot) -> str:
    available = snapshot.available_physical_bytes
    if available is None:
        return "UNMEASURABLE"
    if available < RESOURCE_POLICY.emergency_available_physical_gib * GIB:
        return "EMERGENCY"
    if available < RESOURCE_POLICY.target_available_physical_gib * GIB:
        return "ALLOCATION_BLOCKED"
    if available < RESOURCE_POLICY.warning_available_physical_gib * GIB:
        return "WARNING"
    if available < RESOURCE_POLICY.readmission_available_physical_gib * GIB:
        return "READMISSION_BLOCKED"
    return "NORMAL"


def evaluate_admission_blockers(
    *,
    available_physical_bytes: int | None,
    projected_available_physical_bytes: int | None,
    projected_commit_headroom_bytes: int | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if available_physical_bytes is None:
        reasons.append("AVAILABLE_PHYSICAL_MEMORY_UNMEASURABLE")
    elif available_physical_bytes < RESOURCE_POLICY.readmission_available_physical_gib * GIB:
        reasons.append("AVAILABLE_PHYSICAL_MEMORY_BELOW_7_GIB_READMISSION_FLOOR")
    if projected_available_physical_bytes is None:
        reasons.append("PROJECTED_AVAILABLE_PHYSICAL_MEMORY_UNMEASURABLE")
    elif projected_available_physical_bytes < RESOURCE_POLICY.target_available_physical_gib * GIB:
        reasons.append("PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE")
    if projected_commit_headroom_bytes is None:
        reasons.append("SYSTEM_COMMIT_HEADROOM_UNMEASURABLE")
    elif projected_commit_headroom_bytes < RESOURCE_POLICY.system_commit_headroom_gib * GIB:
        reasons.append("PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_12_GIB_GUARD")
    return tuple(reasons)


def actual_host_pressure_reasons(
    snapshot: MemorySnapshot,
    job_snapshot: JobMemorySnapshot | None = None,
) -> tuple[str, ...]:
    """Return blockers caused by measured host state, never future reservations."""

    reasons: list[str] = []
    available = snapshot.available_physical_bytes
    if available is None:
        reasons.append("AVAILABLE_PHYSICAL_MEMORY_UNMEASURABLE")
    elif available < RESOURCE_POLICY.emergency_available_physical_gib * GIB:
        reasons.append("AVAILABLE_PHYSICAL_MEMORY_BELOW_4_GIB_EMERGENCY")
    elif available < RESOURCE_POLICY.warning_available_physical_gib * GIB:
        reasons.append("AVAILABLE_PHYSICAL_MEMORY_BELOW_6_GIB_WARNING")
    commit_headroom = snapshot.commit_headroom_bytes
    if commit_headroom is None:
        reasons.append("SYSTEM_COMMIT_HEADROOM_UNMEASURABLE")
    elif commit_headroom < RESOURCE_POLICY.system_commit_headroom_gib * GIB:
        reasons.append("SYSTEM_COMMIT_HEADROOM_BELOW_12_GIB_GUARD")
    if job_snapshot is not None:
        if not job_snapshot.installation_verified:
            reasons.append("WINDOWS_JOB_COMMIT_LIMIT_NOT_VERIFIED")
        if job_snapshot.limit_violation_detected:
            reasons.append("WINDOWS_JOB_AGGREGATE_COMMIT_LIMIT_REACHED")
        elif job_snapshot.committed_bytes >= job_snapshot.commit_limit_bytes:
            reasons.append("WINDOWS_JOB_AGGREGATE_COMMIT_LIMIT_REACHED")
    return tuple(dict.fromkeys(reasons))


def reservation_denial_classification(
    *,
    snapshot: MemorySnapshot,
    job_snapshot: JobMemorySnapshot | None,
    blocking_reasons: Sequence[str],
) -> str:
    """Separate scheduling contention from unsafe measured host pressure."""

    if not blocking_reasons:
        return "GRANTED"
    if actual_host_pressure_reasons(snapshot, job_snapshot):
        return HOST_RESOURCE_PRESSURE
    capacity_only_reasons = {
        "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE",
        "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_12_GIB_GUARD",
        "PROJECTED_WINDOWS_JOB_COMMIT_ABOVE_24_GIB_LIMIT",
        "AVAILABLE_PHYSICAL_MEMORY_BELOW_7_GIB_READMISSION_FLOOR",
        "WORKER_SLOT_LIMIT_REACHED",
    }
    if set(blocking_reasons).issubset(capacity_only_reasons):
        return LOCAL_CAPACITY_DEFERRAL
    return "WAIT_RESOURCE"


def reservation_payload_is_local_capacity_deferral(
    payload: Mapping[str, Any],
) -> bool:
    """Recognise current and legacy V3 capacity-only denial evidence."""

    classification = str(payload.get("classification") or "")
    if classification == LOCAL_CAPACITY_DEFERRAL:
        return True
    if classification != "WAIT_RESOURCE":
        return False
    reasons = tuple(str(value) for value in payload.get("blocking_reasons") or ())
    memory_payload = payload.get("memory_snapshot") or {}
    if not isinstance(memory_payload, Mapping):
        return False
    snapshot = MemorySnapshot(
        available_physical_bytes=memory_payload.get("available_physical_bytes"),
        total_physical_bytes=memory_payload.get("total_physical_bytes"),
        commit_headroom_bytes=memory_payload.get("commit_headroom_bytes"),
        commit_limit_bytes=memory_payload.get("commit_limit_bytes"),
        committed_bytes=memory_payload.get("committed_bytes"),
        peak_committed_bytes=memory_payload.get("peak_committed_bytes"),
        source=str(memory_payload.get("source") or "legacy-reservation-evidence"),
        checked_at_utc=str(memory_payload.get("checked_at_utc") or ""),
    )
    job_payload = payload.get("job_snapshot")
    job = None
    if isinstance(job_payload, Mapping) and job_payload:
        try:
            job = JobMemorySnapshot(
                committed_bytes=int(job_payload.get("committed_bytes", 0) or 0),
                peak_committed_bytes=int(
                    job_payload.get("peak_committed_bytes", 0) or 0
                ),
                commit_limit_bytes=int(
                    job_payload.get("commit_limit_bytes", 0) or 0
                ),
                active_processes=int(job_payload.get("active_processes", 0) or 0),
                total_processes=int(job_payload.get("total_processes", 0) or 0),
                terminated_processes=int(
                    job_payload.get("terminated_processes", 0) or 0
                ),
                limit_violation_detected=bool(
                    job_payload.get("limit_violation_detected", False)
                ),
                installation_verified=bool(
                    job_payload.get("installation_verified", False)
                ),
                checked_at_utc=str(job_payload.get("checked_at_utc") or ""),
            )
        except (TypeError, ValueError):
            return False
    return bool(
        reasons
        and reservation_denial_classification(
            snapshot=snapshot,
            job_snapshot=job,
            blocking_reasons=reasons,
        )
        == LOCAL_CAPACITY_DEFERRAL
    )


def capacity_change_is_material(
    before: CapacitySnapshot,
    after: CapacitySnapshot,
    *,
    elapsed_seconds: float,
) -> bool:
    """Gate readmission until cooldown plus a meaningful capacity release."""

    if elapsed_seconds < RESOURCE_POLICY.resource_deferral_cooldown_seconds:
        return False
    if after.active_worker_reservations < before.active_worker_reservations:
        return True
    material_bytes = RESOURCE_POLICY.material_capacity_change_gib * GIB
    if (
        before.outstanding_physical_bytes - after.outstanding_physical_bytes
        >= material_bytes
        or before.outstanding_commit_bytes - after.outstanding_commit_bytes
        >= material_bytes
    ):
        return True
    if (
        before.available_physical_bytes is not None
        and after.available_physical_bytes is not None
        and before.available_physical_bytes
        < RESOURCE_POLICY.readmission_available_physical_gib * GIB
        <= after.available_physical_bytes
    ):
        return True
    if (
        before.available_physical_bytes is not None
        and after.available_physical_bytes is not None
        and after.available_physical_bytes
        >= RESOURCE_POLICY.readmission_available_physical_gib * GIB
        and after.available_physical_bytes - before.available_physical_bytes
        >= material_bytes
    ):
        return True
    if (
        before.system_commit_headroom_bytes is not None
        and after.system_commit_headroom_bytes is not None
        and after.system_commit_headroom_bytes
        - before.system_commit_headroom_bytes
        >= material_bytes
    ):
        return True
    if (
        before.job_committed_bytes is not None
        and after.job_committed_bytes is not None
        and before.job_committed_bytes - after.job_committed_bytes >= material_bytes
    ):
        return True
    return False


def evaluate_memory(
    *,
    stage: str,
    estimated_allocation_bytes: int | None = None,
    estimated_physical_bytes: int | None = None,
    estimated_commit_bytes: int | None = None,
    snapshot: MemorySnapshot | None = None,
    purpose: str = "allocation",
) -> MemoryDecision:
    if estimated_allocation_bytes is not None:
        if estimated_physical_bytes is not None or estimated_commit_bytes is not None:
            raise ValueError("Use either a common or separate allocation estimate")
        estimated_physical_bytes = estimated_allocation_bytes
        estimated_commit_bytes = estimated_allocation_bytes
    if estimated_physical_bytes is None or estimated_commit_bytes is None:
        raise ValueError("Both physical and commit estimates are required")
    if estimated_physical_bytes < 0 or estimated_commit_bytes < 0:
        raise ValueError("Estimated allocation must be non-negative")
    if purpose not in {"admission", "allocation", "observation"}:
        raise ValueError(f"Unsupported memory-check purpose: {purpose}")
    observed = snapshot or system_memory_snapshot()
    guarded_physical = int(
        math.ceil(estimated_physical_bytes * RESOURCE_POLICY.allocation_safety_factor)
    )
    guarded_commit = int(
        math.ceil(estimated_commit_bytes * RESOURCE_POLICY.allocation_safety_factor)
    )
    available = observed.available_physical_bytes
    commit_headroom = observed.commit_headroom_bytes
    projected_available = None if available is None else available - guarded_physical
    projected_commit = None if commit_headroom is None else commit_headroom - guarded_commit
    required_physical = (
        RESOURCE_POLICY.target_available_physical_gib * GIB + guarded_physical
    )
    required_commit = (
        RESOURCE_POLICY.system_commit_headroom_gib * GIB + guarded_commit
    )
    reasons: list[str] = []
    warnings: list[str] = []
    if available is None:
        reasons.append("AVAILABLE_PHYSICAL_MEMORY_UNMEASURABLE")
    else:
        if available < RESOURCE_POLICY.warning_available_physical_gib * GIB:
            warnings.append("AVAILABLE_PHYSICAL_MEMORY_BELOW_WARNING_THRESHOLD")
        if projected_available < RESOURCE_POLICY.target_available_physical_gib * GIB:
            reasons.append("PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE")
        if (
            purpose == "admission"
            and available
            < RESOURCE_POLICY.readmission_available_physical_gib * GIB
        ):
            reasons.append("AVAILABLE_PHYSICAL_MEMORY_BELOW_7_GIB_READMISSION_FLOOR")
    if commit_headroom is None:
        reasons.append("SYSTEM_COMMIT_HEADROOM_UNMEASURABLE")
    elif projected_commit < RESOURCE_POLICY.system_commit_headroom_gib * GIB:
        reasons.append("PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_12_GIB_GUARD")
    level = _pressure_level(observed)
    admission_allowed = not evaluate_admission_blockers(
        available_physical_bytes=available,
        projected_available_physical_bytes=projected_available,
        projected_commit_headroom_bytes=projected_commit,
    )
    return MemoryDecision(
        stage=stage,
        purpose=purpose,
        safe=not reasons,
        admission_allowed=admission_allowed,
        emergency=level == "EMERGENCY",
        pressure_level=level,
        estimated_allocation_bytes=max(
            int(estimated_physical_bytes), int(estimated_commit_bytes)
        ),
        guarded_allocation_bytes=max(guarded_physical, guarded_commit),
        projected_available_physical_bytes=projected_available,
        projected_commit_headroom_bytes=projected_commit,
        required_available_physical_bytes=required_physical,
        required_commit_headroom_bytes=required_commit,
        blocking_reasons=tuple(dict.fromkeys(reasons)),
        warning_reasons=tuple(warnings),
        snapshot=observed,
        estimated_physical_bytes=int(estimated_physical_bytes),
        estimated_commit_bytes=int(estimated_commit_bytes),
        guarded_physical_bytes=guarded_physical,
        guarded_commit_bytes=guarded_commit,
    )


def require_memory(
    *,
    stage: str,
    estimated_allocation_bytes: int,
    snapshot_provider: Callable[[], MemorySnapshot] = system_memory_snapshot,
) -> MemoryDecision:
    decision = evaluate_memory(
        stage=stage,
        estimated_allocation_bytes=estimated_allocation_bytes,
        snapshot=snapshot_provider(),
        purpose="allocation",
    )
    if not decision.safe:
        raise CleanV2ResourcePressure(decision)
    return decision


def allocation_failure_decision(
    *, stage: str, reason: str = "RESOURCE_LIMIT_ALLOCATION_FAILED"
) -> MemoryDecision:
    decision = evaluate_memory(stage=stage, estimated_allocation_bytes=0, purpose="allocation")
    return replace(
        decision,
        safe=False,
        blocking_reasons=tuple((*decision.blocking_reasons, reason)),
    )


def is_resource_allocation_failure(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {
        8,  # ERROR_NOT_ENOUGH_MEMORY
        14,  # ERROR_OUTOFMEMORY
        1455,  # ERROR_COMMITMENT_LIMIT
    }:
        return True
    return type(exc).__name__ in {"ArrayMemoryError", "OutOfMemoryError"}


def estimate_panel_allocation_bytes(*, row_count: int, column_count: int) -> int:
    """Conservative peak for pandas reads, joins, concat, and panel slicing."""

    cells = max(0, int(row_count)) * max(1, int(column_count))
    return cells * 8 * 6 + max(0, int(row_count)) * 256


def estimate_tabular_fit_allocation_bytes(*, row_count: int, predictor_count: int) -> int:
    """Conservative matrix/imputation/estimator committed-memory estimate."""

    cells = max(0, int(row_count)) * max(1, int(predictor_count))
    return cells * 8 * 5 + max(0, int(row_count)) * 64


def estimate_sequence_allocation_bytes(
    *, example_count: int, sequence_length: int, predictor_count: int
) -> int:
    """Estimate bounded Python-window plus tensor/preprocessing allocations."""

    scalar_count = (
        max(0, int(example_count))
        * max(1, int(sequence_length))
        * max(1, int(predictor_count))
    )
    return scalar_count * 52 + max(0, int(example_count)) * 1024


def estimate_worker_startup_allocation_bytes(family: str) -> int:
    if family in {
        "transformer",
        "momentum_transformer",
        "market_context_encoder",
        "temporal_fusion_transformer",
    }:
        return 6 * GIB
    if family == "random_forest":
        return 6 * GIB
    if family in {"momentum", "equal_weight_no_model"}:
        return 2 * GIB
    return 4 * GIB


@dataclass(frozen=True)
class FamilyResourceProfile:
    """Auditable family-specific worker envelope inputs.

    ``requested_floor`` is the minimum unguarded reservation.  When empirical
    evidence is present, the requested value is increased so that the guarded
    reservation can cover the measured high-water plus the explicit cushion.
    The global allocation safety factor is still applied exactly once by the
    reservation ledger.
    """

    profile_id: str
    fit_required: bool
    requested_floor: AllocationEstimate
    measured_high_water: AllocationEstimate | None = None
    high_water_cushion: AllocationEstimate = AllocationEstimate(0, 0)
    evidence: str = "STATIC_FAMILY_FLOOR"

    def requested_peak(self) -> AllocationEstimate:
        if self.measured_high_water is None:
            return self.requested_floor
        factor = RESOURCE_POLICY.allocation_safety_factor
        guarded_target = AllocationEstimate(
            self.measured_high_water.physical_bytes
            + self.high_water_cushion.physical_bytes,
            self.measured_high_water.commit_bytes
            + self.high_water_cushion.commit_bytes,
        )
        return AllocationEstimate(
            max(
                self.requested_floor.physical_bytes,
                int(math.ceil(guarded_target.physical_bytes / factor)),
            ),
            max(
                self.requested_floor.commit_bytes,
                int(math.ceil(guarded_target.commit_bytes / factor)),
            ),
        )

    def payload(self) -> dict[str, Any]:
        requested = self.requested_peak()
        return {
            "profile_id": self.profile_id,
            "fit_required": self.fit_required,
            "requested_floor": self.requested_floor.payload(),
            "measured_high_water": (
                self.measured_high_water.payload()
                if self.measured_high_water is not None
                else None
            ),
            "high_water_cushion": self.high_water_cushion.payload(),
            "requested_peak": requested.payload(),
            "guarded_peak": requested.guarded().payload(),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FamilySchedulingProfile:
    """Measured scheduling traits; admission safety remains ledger-owned."""

    workload_class: str
    bounded_fit_wall_seconds: float
    evidence: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_TABULAR_PROFILE = FamilyResourceProfile(
    profile_id="ORDINARY_TABULAR_STATIC_V1",
    fit_required=True,
    requested_floor=AllocationEstimate(4 * GIB, 6 * GIB),
)
_SEQUENCE_PROFILE = FamilyResourceProfile(
    profile_id="SEQUENCE_STATIC_V1",
    fit_required=True,
    requested_floor=AllocationEstimate(6 * GIB, 10 * GIB),
)
FAMILY_RESOURCE_PROFILES: Mapping[str, FamilyResourceProfile] = {
    "random_forest": FamilyResourceProfile(
        profile_id="RANDOM_FOREST_DELL_EMPIRICAL_V1",
        fit_required=True,
        requested_floor=AllocationEstimate(3 * GIB, 3 * GIB),
        measured_high_water=AllocationEstimate(
            physical_bytes=3_894_980_608,
            commit_bytes=3_889_950_720,
        ),
        high_water_cushion=AllocationEstimate(GIB, GIB),
        evidence=(
            "Dell attempt=1790635165879799300 family=random_forest pid=5804;"
            "aggregate process-tree peak observed 2026-09-29 before source change"
        ),
    ),
    "equal_weight_no_model": FamilyResourceProfile(
        profile_id="NO_MODEL_CONTROL_STATIC_V1",
        fit_required=False,
        requested_floor=AllocationEstimate(2 * GIB, 3 * GIB),
        evidence="NO_ESTIMATOR_FIT;PANEL_AND_SCORE_STAGES_REMAIN_SEPARATELY_GUARDED",
    ),
    "momentum": FamilyResourceProfile(
        profile_id="MOMENTUM_CONTROL_STATIC_V1",
        fit_required=False,
        requested_floor=AllocationEstimate(2 * GIB, 3 * GIB),
        evidence="NO_ESTIMATOR_FIT;PANEL_AND_SCORE_STAGES_REMAIN_SEPARATELY_GUARDED",
    ),
}


_FAMILY_SCHEDULING_PROFILES: Mapping[str, FamilySchedulingProfile] = {
    "ridge_C5": FamilySchedulingProfile(
        "LIGHT_PREDICTIVE",
        0.093,
        "20260929 bounded real-authority benchmark; 9,332 training rows",
    ),
    "elastic_net_C5": FamilySchedulingProfile(
        "LIGHT_PREDICTIVE",
        0.085,
        "20260929 bounded real-authority benchmark; 8,325 training rows",
    ),
    "elastic_net_C6": FamilySchedulingProfile(
        "LIGHT_PREDICTIVE",
        0.093,
        "20260929 bounded real-authority benchmark; 8,325 training rows",
    ),
    "huber": FamilySchedulingProfile(
        "LIGHT_PREDICTIVE",
        0.290,
        "20260929 bounded real-authority benchmark; convergence warning retained",
    ),
    "gradient_boosting_C0_W20": FamilySchedulingProfile(
        "TREE_PREDICTIVE", 6.892, "20260929 bounded real-authority benchmark"
    ),
    "gradient_boosting_C0": FamilySchedulingProfile(
        "TREE_PREDICTIVE", 7.522, "20260929 bounded real-authority benchmark"
    ),
    "gradient_boosting_C0_W40": FamilySchedulingProfile(
        "TREE_PREDICTIVE", 14.240, "20260929 bounded real-authority benchmark"
    ),
    "gradient_boosting_C0_W80": FamilySchedulingProfile(
        "TREE_PREDICTIVE", 28.661, "20260929 bounded real-authority benchmark"
    ),
    "random_forest": FamilySchedulingProfile(
        "TREE_PREDICTIVE", 21.639, "20260929 bounded benchmark; progressed worker pinned"
    ),
    "transformer": FamilySchedulingProfile(
        "SEQUENCE_PREDICTIVE",
        5.287,
        "20260929 one-asset checkpoint smoke; not cross-family throughput comparable",
    ),
    "momentum": FamilySchedulingProfile(
        "CONTROL", 0.0, "NO_ESTIMATOR_FIT"
    ),
    "equal_weight_no_model": FamilySchedulingProfile(
        "CONTROL", 0.0, "NO_ESTIMATOR_FIT"
    ),
}


def family_resource_profile(family: str) -> FamilyResourceProfile:
    if family in FAMILY_RESOURCE_PROFILES:
        return FAMILY_RESOURCE_PROFILES[family]
    if family in {
        "transformer",
        "momentum_transformer",
        "market_context_encoder",
        "temporal_fusion_transformer",
    }:
        return _SEQUENCE_PROFILE
    return _DEFAULT_TABULAR_PROFILE


def family_scheduling_profile(family: str) -> FamilySchedulingProfile:
    if family in _FAMILY_SCHEDULING_PROFILES:
        return _FAMILY_SCHEDULING_PROFILES[family]
    profile = family_resource_profile(family)
    return FamilySchedulingProfile(
        workload_class=(
            "SEQUENCE_PREDICTIVE"
            if profile.profile_id.startswith("SEQUENCE")
            else ("PREDICTIVE" if profile.fit_required else "CONTROL")
        ),
        bounded_fit_wall_seconds=1.0e12,
        evidence="NO_COMPARABLE_BOUNDED_BENCHMARK",
    )


def resource_profile_registry_payload() -> dict[str, Any]:
    return {
        "default_tabular": _DEFAULT_TABULAR_PROFILE.payload(),
        "default_sequence": _SEQUENCE_PROFILE.payload(),
        **{
            family: profile.payload()
            for family, profile in sorted(FAMILY_RESOURCE_PROFILES.items())
        },
    }


def scheduling_profile_registry_payload() -> dict[str, Any]:
    return {
        family: profile.payload()
        for family, profile in sorted(_FAMILY_SCHEDULING_PROFILES.items())
    }


def estimate_worker_peak_allocation(family: str) -> AllocationEstimate:
    """Reserve an honest future peak for a worker before it is launched."""

    return family_resource_profile(family).requested_peak()


def panel_allocation_estimate(*, row_count: int, column_count: int) -> AllocationEstimate:
    physical = estimate_panel_allocation_bytes(
        row_count=row_count, column_count=column_count
    )
    return AllocationEstimate(physical, int(math.ceil(physical * 1.15)))


def panel_slice_allocation_estimate(
    *, row_count: int, column_count: int
) -> AllocationEstimate:
    """Estimate incremental memory for boolean-indexed panel populations.

    The source panel already exists at the stage baseline and must not be
    reserved again.  Training and scoring boolean selections together cover
    no more than the source rows, while pandas can transiently materialize
    value blocks and indexes.  Three cell-widths plus per-row overhead keeps a
    conservative incremental bound without applying the six-copy read/join/
    concat estimate used for initial panel assembly.
    """

    rows = max(0, int(row_count))
    columns = max(1, int(column_count))
    physical = rows * columns * 8 * 3 + rows * 256
    return AllocationEstimate(physical, int(math.ceil(physical * 1.15)))


def tabular_fit_allocation_estimate(
    *, row_count: int, predictor_count: int
) -> AllocationEstimate:
    physical = estimate_tabular_fit_allocation_bytes(
        row_count=row_count, predictor_count=predictor_count
    )
    return AllocationEstimate(physical, int(math.ceil(physical * 1.25)))


def sequence_allocation_estimate(
    *, example_count: int, sequence_length: int, predictor_count: int
) -> AllocationEstimate:
    physical = estimate_sequence_allocation_bytes(
        example_count=example_count,
        sequence_length=sequence_length,
        predictor_count=predictor_count,
    )
    return AllocationEstimate(physical, int(math.ceil(physical * 1.20)))


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory_snapshot(pid: int) -> ProcessMemorySnapshot:
    if os.name == "nt" and int(pid or 0) > 0:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, int(pid))
        if handle:
            try:
                counters = _PROCESS_MEMORY_COUNTERS_EX()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), ctypes.sizeof(counters)
                ):
                    return ProcessMemorySnapshot(
                        pid=int(pid),
                        resident_bytes=int(counters.WorkingSetSize),
                        peak_resident_bytes=int(counters.PeakWorkingSetSize),
                        private_commit_bytes=int(counters.PrivateUsage),
                        peak_private_commit_bytes=int(counters.PeakPagefileUsage),
                        source="psapi.GetProcessMemoryInfo",
                        checked_at_utc=_utc_now(),
                    )
            finally:
                kernel32.CloseHandle(handle)
    try:
        import psutil

        info = psutil.Process(int(pid)).memory_info()
        private = getattr(info, "private", getattr(info, "vms", None))
        return ProcessMemorySnapshot(
            pid=int(pid),
            resident_bytes=int(info.rss),
            peak_resident_bytes=getattr(info, "peak_wset", None),
            private_commit_bytes=None if private is None else int(private),
            peak_private_commit_bytes=getattr(info, "peak_pagefile", None),
            source="psutil.Process.memory_info",
            checked_at_utc=_utc_now(),
        )
    except Exception:
        return ProcessMemorySnapshot(
            pid=int(pid),
            resident_bytes=None,
            peak_resident_bytes=None,
            private_commit_bytes=None,
            peak_private_commit_bytes=None,
            source="unavailable",
            checked_at_utc=_utc_now(),
        )


def process_tree_memory_snapshot(pid: int) -> ProcessMemorySnapshot:
    """Aggregate resident/private memory for one worker and its descendants."""

    try:
        import psutil

        root = psutil.Process(int(pid))
        processes = [root, *root.children(recursive=True)]
    except Exception:
        return process_memory_snapshot(pid)
    snapshots = [process_memory_snapshot(process.pid) for process in processes]

    def total(attribute: str) -> int | None:
        values = [getattr(snapshot, attribute) for snapshot in snapshots]
        measured = [int(value) for value in values if value is not None]
        return sum(measured) if measured else None

    return ProcessMemorySnapshot(
        pid=int(pid),
        resident_bytes=total("resident_bytes"),
        peak_resident_bytes=total("peak_resident_bytes"),
        private_commit_bytes=total("private_commit_bytes"),
        peak_private_commit_bytes=total("peak_private_commit_bytes"),
        source="aggregate_process_tree:" + ",".join(
            sorted({snapshot.source for snapshot in snapshots})
        ),
        checked_at_utc=_utc_now(),
    )


@dataclass
class _WorkerBudget:
    reservation_id: str
    owner: WorkerReservationOwner
    guarded_peak: AllocationEstimate
    baseline: ProcessMemorySnapshot | None = None
    current: ProcessMemorySnapshot | None = None
    peak_resident_bytes: int = 0
    peak_private_commit_bytes: int = 0
    stage: str = ""
    stage_reservation_id: str = ""
    guarded_stage: AllocationEstimate = field(
        default_factory=lambda: AllocationEstimate(0, 0)
    )
    stage_baseline: ProcessMemorySnapshot | None = None


class ResourceReservationLedger:
    """Supervisor-owned, atomic accounting for all model-worker allocations."""

    def __init__(
        self,
        *,
        maximum_workers: int,
        memory_snapshot_provider: Callable[[], MemorySnapshot] = system_memory_snapshot,
        job_snapshot_provider: Callable[[], JobMemorySnapshot | None] = lambda: None,
        process_tree_snapshot_provider: Callable[[int], ProcessMemorySnapshot] = (
            process_tree_memory_snapshot
        ),
    ) -> None:
        if maximum_workers <= 0:
            raise ValueError("maximum_workers must be positive")
        self.maximum_workers = int(maximum_workers)
        self._memory_snapshot_provider = memory_snapshot_provider
        self._job_snapshot_provider = job_snapshot_provider
        self._process_tree_snapshot_provider = process_tree_snapshot_provider
        self._workers: dict[str, _WorkerBudget] = {}
        self._logical_owners: dict[tuple[str, int], str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _consumed(budget: _WorkerBudget) -> AllocationEstimate:
        if budget.baseline is None or budget.current is None:
            return AllocationEstimate(0, 0)
        baseline_resident = int(budget.baseline.resident_bytes or 0)
        current_resident = int(budget.current.resident_bytes or baseline_resident)
        baseline_commit = int(budget.baseline.private_commit_bytes or 0)
        current_commit = int(budget.current.private_commit_bytes or baseline_commit)
        return AllocationEstimate(
            max(0, current_resident - baseline_resident),
            max(0, current_commit - baseline_commit),
        )

    @classmethod
    def _outstanding_for(cls, budget: _WorkerBudget) -> AllocationEstimate:
        consumed = cls._consumed(budget)
        peak_remaining = AllocationEstimate(
            max(0, budget.guarded_peak.physical_bytes - consumed.physical_bytes),
            max(0, budget.guarded_peak.commit_bytes - consumed.commit_bytes),
        )
        stage_consumed = AllocationEstimate(0, 0)
        if budget.stage_baseline is not None and budget.current is not None:
            stage_consumed = AllocationEstimate(
                max(
                    0,
                    int(budget.current.resident_bytes or 0)
                    - int(budget.stage_baseline.resident_bytes or 0),
                ),
                max(
                    0,
                    int(budget.current.private_commit_bytes or 0)
                    - int(budget.stage_baseline.private_commit_bytes or 0),
                ),
            )
        stage_remaining = AllocationEstimate(
            max(
                0,
                budget.guarded_stage.physical_bytes
                - stage_consumed.physical_bytes,
            ),
            max(
                0,
                budget.guarded_stage.commit_bytes
                - stage_consumed.commit_bytes,
            ),
        )
        outstanding = AllocationEstimate(
            max(peak_remaining.physical_bytes, stage_remaining.physical_bytes),
            max(peak_remaining.commit_bytes, stage_remaining.commit_bytes),
        )
        if outstanding.physical_bytes < peak_remaining.physical_bytes:
            raise AssertionError("WORKER_PHYSICAL_HIGH_WATER_DOUBLE_COUNT_INVARIANT")
        if outstanding.commit_bytes < peak_remaining.commit_bytes:
            raise AssertionError("WORKER_COMMIT_HIGH_WATER_DOUBLE_COUNT_INVARIANT")
        return outstanding

    def _reconcile_locked(self) -> None:
        for budget in self._workers.values():
            if budget.owner.pid <= 0:
                continue
            current = self._process_tree_snapshot_provider(budget.owner.pid)
            budget.current = current
            budget.peak_resident_bytes = max(
                budget.peak_resident_bytes,
                int(current.peak_resident_bytes or current.resident_bytes or 0),
            )
            budget.peak_private_commit_bytes = max(
                budget.peak_private_commit_bytes,
                int(
                    current.peak_private_commit_bytes
                    or current.private_commit_bytes
                    or 0
                ),
            )

    def _outstanding_locked(self) -> AllocationEstimate:
        estimates = [self._outstanding_for(budget) for budget in self._workers.values()]
        return AllocationEstimate(
            sum(estimate.physical_bytes for estimate in estimates),
            sum(estimate.commit_bytes for estimate in estimates),
        )

    def _decision_locked(
        self,
        *,
        stage: str,
        purpose: str,
        requested: AllocationEstimate,
        worker_reservation_id: str | None,
        stage_reservation_id: str | None,
        extra_outstanding: AllocationEstimate = AllocationEstimate(0, 0),
        classification_if_blocked: str = "WAIT_RESOURCE",
    ) -> ReservationDecision:
        self._reconcile_locked()
        memory = self._memory_snapshot_provider()
        job = self._job_snapshot_provider()
        current_outstanding = self._outstanding_locked()
        outstanding = AllocationEstimate(
            current_outstanding.physical_bytes + extra_outstanding.physical_bytes,
            current_outstanding.commit_bytes + extra_outstanding.commit_bytes,
        )
        available = memory.available_physical_bytes
        commit_headroom = memory.commit_headroom_bytes
        projected_available = (
            None if available is None else available - outstanding.physical_bytes
        )
        projected_commit = (
            None
            if commit_headroom is None
            else commit_headroom - outstanding.commit_bytes
        )
        projected_job = (
            None
            if job is None
            else int(job.committed_bytes) + outstanding.commit_bytes
        )
        reasons = list(
            evaluate_admission_blockers(
                available_physical_bytes=(
                    available
                    if purpose == "admission" or available is None
                    else max(
                        int(available),
                        RESOURCE_POLICY.readmission_available_physical_gib * GIB,
                    )
                ),
                projected_available_physical_bytes=projected_available,
                projected_commit_headroom_bytes=projected_commit,
            )
        )
        if (
            available is not None
            and available < RESOURCE_POLICY.emergency_available_physical_gib * GIB
        ):
            reasons.append("AVAILABLE_PHYSICAL_MEMORY_BELOW_4_GIB_EMERGENCY")
        if job is not None:
            if not job.installation_verified:
                reasons.append("WINDOWS_JOB_COMMIT_LIMIT_NOT_VERIFIED")
            if projected_job is not None and projected_job > job.commit_limit_bytes:
                reasons.append("PROJECTED_WINDOWS_JOB_COMMIT_ABOVE_24_GIB_LIMIT")
            if job.limit_violation_detected:
                reasons.append("WINDOWS_JOB_AGGREGATE_COMMIT_LIMIT_REACHED")
        guarded = requested.guarded()
        classification = "GRANTED"
        if reasons:
            classification = classification_if_blocked
            if classification_if_blocked == "WAIT_RESOURCE":
                classification = reservation_denial_classification(
                    snapshot=memory,
                    job_snapshot=job,
                    blocking_reasons=reasons,
                )
        return ReservationDecision(
            granted=not reasons,
            classification=classification,
            blocking_reasons=tuple(dict.fromkeys(reasons)),
            worker_reservation_id=worker_reservation_id,
            stage_reservation_id=stage_reservation_id,
            stage=stage,
            requested=requested,
            guarded_requested=guarded,
            outstanding=outstanding,
            projected_available_physical_bytes=projected_available,
            projected_system_commit_headroom_bytes=projected_commit,
            projected_job_commit_bytes=projected_job,
            memory_snapshot=memory,
            job_snapshot=job,
        )

    def try_admit(
        self, owner: WorkerReservationOwner, estimate: AllocationEstimate
    ) -> ReservationDecision:
        with self._lock:
            if owner.logical_key in self._logical_owners:
                raise WorkerContainmentError("DUPLICATE_WORKER_RESERVATION_OWNER")
            if len(self._workers) >= self.maximum_workers:
                decision = self._decision_locked(
                    stage=f"worker_admission:{owner.family}",
                    purpose="admission",
                    requested=estimate,
                    worker_reservation_id=None,
                    stage_reservation_id=None,
                )
                return replace(
                    decision,
                    granted=False,
                    classification="WORKER_SLOT_LIMIT_REACHED",
                    blocking_reasons=("WORKER_SLOT_LIMIT_REACHED",),
                )
            guarded = estimate.guarded()
            decision = self._decision_locked(
                stage=f"worker_admission:{owner.family}",
                purpose="admission",
                requested=estimate,
                worker_reservation_id=None,
                stage_reservation_id=None,
                extra_outstanding=guarded,
            )
            if not decision.granted:
                return decision
            reservation_id = uuid4().hex
            self._workers[reservation_id] = _WorkerBudget(
                reservation_id=reservation_id,
                owner=owner,
                guarded_peak=guarded,
            )
            self._logical_owners[owner.logical_key] = reservation_id
            return replace(decision, worker_reservation_id=reservation_id)

    def bind_worker(
        self, reservation_id: str, owner: WorkerReservationOwner
    ) -> None:
        with self._lock:
            budget = self._workers.get(reservation_id)
            if budget is None or budget.owner.logical_key != owner.logical_key:
                raise WorkerContainmentError("WORKER_RESERVATION_BINDING_MISMATCH")
            if owner.pid <= 0 or not owner.process_creation_time_utc:
                raise WorkerContainmentError("WORKER_RESERVATION_IDENTITY_INCOMPLETE")
            baseline = self._process_tree_snapshot_provider(owner.pid)
            if baseline.resident_bytes is None or baseline.private_commit_bytes is None:
                raise WorkerContainmentError("WORKER_BASELINE_MEMORY_UNMEASURABLE")
            budget.owner = owner
            budget.baseline = baseline
            budget.current = baseline

    def request_stage(
        self,
        owner: WorkerReservationOwner,
        *,
        stage: str,
        estimate: AllocationEstimate,
    ) -> ReservationDecision:
        with self._lock:
            # Stage requests are incremental from the worker's current measured
            # footprint. Refresh first so the high-water delta uses one coherent
            # baseline; _decision_locked refreshes again immediately before grant.
            self._reconcile_locked()
            reservation_id = self._logical_owners.get(owner.logical_key)
            budget = self._workers.get(reservation_id or "")
            if budget is None or not self._owner_matches(budget.owner, owner):
                raise WorkerContainmentError("STALE_OR_UNKNOWN_WORKER_RESERVATION_OWNER")
            if budget.stage_reservation_id:
                raise WorkerContainmentError("WORKER_STAGE_RESERVATION_ALREADY_ACTIVE")
            guarded = estimate.guarded()
            before = self._outstanding_for(budget)
            candidate = AllocationEstimate(
                max(before.physical_bytes, guarded.physical_bytes),
                max(before.commit_bytes, guarded.commit_bytes),
            )
            extra = AllocationEstimate(
                candidate.physical_bytes - before.physical_bytes,
                candidate.commit_bytes - before.commit_bytes,
            )
            stage_reservation_id = uuid4().hex
            decision = self._decision_locked(
                stage=stage,
                purpose="allocation",
                requested=estimate,
                worker_reservation_id=reservation_id,
                stage_reservation_id=stage_reservation_id,
                extra_outstanding=extra,
            )
            if decision.granted:
                budget.stage = stage
                budget.stage_reservation_id = stage_reservation_id
                budget.guarded_stage = guarded
                budget.stage_baseline = budget.current
            return decision

    @staticmethod
    def _owner_matches(
        actual: WorkerReservationOwner, expected: WorkerReservationOwner
    ) -> bool:
        return bool(
            actual.logical_key == expected.logical_key
            and actual.pid == expected.pid
            and same_process_creation_time(
                actual.process_creation_time_utc,
                expected.process_creation_time_utc,
            )
        )

    def release_stage(
        self, owner: WorkerReservationOwner, stage_reservation_id: str
    ) -> bool:
        with self._lock:
            reservation_id = self._logical_owners.get(owner.logical_key)
            budget = self._workers.get(reservation_id or "")
            if (
                budget is None
                or not self._owner_matches(budget.owner, owner)
                or budget.stage_reservation_id != stage_reservation_id
            ):
                return False
            budget.stage = ""
            budget.stage_reservation_id = ""
            budget.guarded_stage = AllocationEstimate(0, 0)
            budget.stage_baseline = None
            return True

    def release_worker(
        self, owner: WorkerReservationOwner, *, allow_unbound: bool = False
    ) -> bool:
        with self._lock:
            reservation_id = self._logical_owners.get(owner.logical_key)
            budget = self._workers.get(reservation_id or "")
            if budget is None:
                return False
            matches = self._owner_matches(budget.owner, owner)
            if not matches and not (
                allow_unbound
                and budget.owner.pid == 0
                and budget.owner.logical_key == owner.logical_key
            ):
                return False
            self._workers.pop(budget.reservation_id, None)
            self._logical_owners.pop(budget.owner.logical_key, None)
            return True

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            self._reconcile_locked()
            outstanding = self._outstanding_locked()
            workers = []
            for budget in sorted(
                self._workers.values(), key=lambda item: item.owner.family
            ):
                workers.append(
                    {
                        "worker_reservation_id": budget.reservation_id,
                        "owner": budget.owner.payload(),
                        "guarded_peak": budget.guarded_peak.payload(),
                        "current_memory": (
                            budget.current.payload() if budget.current else None
                        ),
                        "peak_resident_bytes": budget.peak_resident_bytes,
                        "peak_private_commit_bytes": budget.peak_private_commit_bytes,
                        "outstanding": self._outstanding_for(budget).payload(),
                        "accounting_semantics": (
                            "MAX_OF_WORKER_PEAK_REMAINING_AND_INCREMENTAL_STAGE_"
                            "HIGH_WATER;MEASURED_PROCESS_TREE_GROWTH_SUBTRACTED_ONCE"
                        ),
                        "active_stage": budget.stage or None,
                        "stage_reservation_id": budget.stage_reservation_id or None,
                        "guarded_stage": budget.guarded_stage.payload(),
                        "stage_baseline_memory": (
                            budget.stage_baseline.payload()
                            if budget.stage_baseline is not None
                            else None
                        ),
                    }
                )
            return {
                "maximum_model_workers": self.maximum_workers,
                "active_worker_reservations": len(self._workers),
                "outstanding": outstanding.payload(),
                "accounting_semantics": (
                    "PER_WORKER_HIGH_WATER_NOT_WORKER_PLUS_STAGE;PHYSICAL_AND_"
                    "COMMIT_TRACKED_SEPARATELY"
                ),
                "workers": workers,
            }

    def capacity_snapshot(self) -> CapacitySnapshot:
        with self._lock:
            self._reconcile_locked()
            memory = self._memory_snapshot_provider()
            job = self._job_snapshot_provider()
            outstanding = self._outstanding_locked()
            return CapacitySnapshot(
                available_physical_bytes=memory.available_physical_bytes,
                system_commit_headroom_bytes=memory.commit_headroom_bytes,
                job_committed_bytes=(job.committed_bytes if job is not None else None),
                outstanding_physical_bytes=outstanding.physical_bytes,
                outstanding_commit_bytes=outstanding.commit_bytes,
                active_worker_reservations=len(self._workers),
                active_families=tuple(
                    sorted(budget.owner.family for budget in self._workers.values())
                ),
                checked_at_utc=memory.checked_at_utc,
            )

    def reconcile_dead_workers(
        self,
        identity_provider: Callable[[int], ProcessIdentity] | None = None,
    ) -> tuple[WorkerReservationOwner, ...]:
        """Release only reservations whose exact PID/creation identity is gone."""

        provider = identity_provider or process_identity
        released: list[WorkerReservationOwner] = []
        with self._lock:
            for reservation_id, budget in list(self._workers.items()):
                owner = budget.owner
                if owner.pid <= 0:
                    continue
                identity = provider(owner.pid)
                if process_identity_matches(
                    identity,
                    expected_creation_time=owner.process_creation_time_utc,
                    required_command_fragments=(
                        "ds24_clean_v2_family_worker.py",
                        "--family",
                        owner.family,
                    ),
                ):
                    continue
                self._workers.pop(reservation_id, None)
                self._logical_owners.pop(owner.logical_key, None)
                released.append(owner)
        return tuple(released)

    def observe(self, *, stage: str, purpose: str = "allocation") -> ReservationDecision:
        with self._lock:
            return self._decision_locked(
                stage=stage,
                purpose=purpose,
                requested=AllocationEstimate(0, 0),
                worker_reservation_id=None,
                stage_reservation_id=None,
            )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_object_atomic(path, payload)


class FileReservationClient:
    """Worker-side client for the supervisor-owned reservation ledger."""

    def __init__(
        self,
        root: Path,
        owner: WorkerReservationOwner,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.owner = owner
        self.timeout_seconds = float(
            timeout_seconds or RESOURCE_POLICY.allocation_request_timeout_seconds
        )
        self.poll_seconds = float(
            poll_seconds or RESOURCE_POLICY.allocation_request_poll_seconds
        )

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_id = uuid4().hex
        requests = self.root / "requests"
        responses = self.root / "responses"
        request_path = requests / f"{request_id}.json"
        response_path = responses / f"{request_id}.json"
        _write_json_atomic(
            request_path,
            {
                "request_id": request_id,
                "resource_policy_id": RESOURCE_POLICY.policy_id,
                "requested_at_utc": _utc_now(),
                "owner": self.owner.payload(),
                **dict(payload),
            },
        )
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if response_path.is_file():
                try:
                    result = read_json_object_with_retry(response_path)
                finally:
                    response_path.unlink(missing_ok=True)
                    request_path.unlink(missing_ok=True)
                return result if isinstance(result, dict) else {}
            time.sleep(self.poll_seconds)
        request_path.unlink(missing_ok=True)
        return {
            "granted": False,
            "classification": "RESERVATION_RESPONSE_TIMEOUT",
            "blocking_reasons": ["SUPERVISOR_RESERVATION_RESPONSE_TIMEOUT"],
        }

    @contextmanager
    def reserve(
        self, *, stage: str, estimate: AllocationEstimate
    ) -> Iterator[ReservationDecision | None]:
        response = self._request(
            {
                "action": "reserve",
                "stage": stage,
                "estimate": estimate.payload(),
            }
        )
        if response.get("granted") is not True:
            raise ResourceReservationUnavailable(stage, response)
        stage_reservation_id = str(response.get("stage_reservation_id", ""))
        if not stage_reservation_id:
            raise ResourceReservationUnavailable(
                stage,
                {
                    "blocking_reasons": ["MISSING_STAGE_RESERVATION_ID"],
                    **response,
                },
            )
        try:
            yield None
        finally:
            release = self._request(
                {
                    "action": "release",
                    "stage": stage,
                    "stage_reservation_id": stage_reservation_id,
                }
            )
            if release.get("released") is not True:
                raise WorkerContainmentError(
                    f"STAGE_RESERVATION_RELEASE_REJECTED:{stage}"
                )


def service_file_reservation_requests(
    root: Path, ledger: ResourceReservationLedger
) -> list[dict[str, Any]]:
    """Service each immutable request once; the supervisor is the only writer."""

    requests = Path(root) / "requests"
    responses = Path(root) / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    handled: list[dict[str, Any]] = []
    for request_path in sorted(requests.glob("*.json")):
        response_path = responses / request_path.name
        if response_path.exists():
            continue
        try:
            request = read_json_object_with_retry(request_path)
            if request.get("resource_policy_id") != RESOURCE_POLICY.policy_id:
                raise ValueError("RESOURCE_POLICY_ID_MISMATCH")
            owner_payload = request["owner"]
            owner = WorkerReservationOwner(
                family=str(owner_payload["family"]),
                attempt_generation=int(owner_payload["attempt_generation"]),
                pid=int(owner_payload["pid"]),
                process_creation_time_utc=str(
                    owner_payload["process_creation_time_utc"]
                ),
            )
            action = str(request.get("action", ""))
            if action == "reserve":
                estimate_payload = request["estimate"]
                decision = ledger.request_stage(
                    owner,
                    stage=str(request["stage"]),
                    estimate=AllocationEstimate(
                        physical_bytes=int(estimate_payload["physical_bytes"]),
                        commit_bytes=int(estimate_payload["commit_bytes"]),
                    ),
                )
                response = decision.payload()
            elif action == "release":
                released = ledger.release_stage(
                    owner, str(request["stage_reservation_id"])
                )
                response = {
                    "released": released,
                    "classification": (
                        "RELEASED" if released else "STALE_RELEASE_REJECTED"
                    ),
                }
            else:
                raise ValueError("UNSUPPORTED_RESERVATION_ACTION")
        except Exception as exc:
            response = {
                "granted": False,
                "released": False,
                "classification": "INVALID_OR_STALE_RESERVATION_REQUEST",
                "blocking_reasons": [f"{type(exc).__name__}:{exc}"],
            }
        response = {
            "request_id": request_path.stem,
            "responded_at_utc": _utc_now(),
            **response,
        }
        _write_json_atomic(response_path, response)
        handled.append(response)
    return handled


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", wintypes.LARGE_INTEGER),
        ("TotalKernelTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _JOBOBJECT_MEMORY_USAGE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("JobMemory", ctypes.c_ulonglong),
        ("PeakJobMemoryUsed", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_ASSOCIATE_COMPLETION_PORT(ctypes.Structure):
    _fields_ = [
        ("CompletionKey", ctypes.c_void_p),
        ("CompletionPort", wintypes.HANDLE),
    ]


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class WorkerContainment(Protocol):
    installation_verified: bool

    def launch(
        self,
        command: Sequence[str],
        *,
        cwd: os.PathLike[str] | str,
        stdout: int | TextIO | BinaryIO | None,
        stderr: int | TextIO | BinaryIO | None,
        env: Mapping[str, str],
        before_resume: Callable[[int], None] | None = None,
    ) -> subprocess.Popen[Any]: ...

    def snapshot(self) -> JobMemorySnapshot | None: ...

    def contains_pid(self, pid: int) -> bool: ...

    def close(self) -> None: ...


class WindowsWorkerJob:
    """Supervisor-owned hard aggregate commit limit for workers and descendants."""

    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION_CLASS = 7
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1
    _JOB_OBJECT_MEMORY_USAGE_INFORMATION_CLASS = 28
    _JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT = 9
    _JOB_OBJECT_MSG_JOB_MEMORY_LIMIT = 10
    _CREATE_SUSPENDED = 0x00000004

    def __init__(self, *, commit_limit_bytes: int = WINDOWS_JOB_COMMIT_LIMIT_BYTES) -> None:
        if os.name != "nt":
            raise WorkerContainmentError("WINDOWS_JOB_OBJECT_UNAVAILABLE_ON_THIS_HOST")
        if commit_limit_bytes <= 0:
            raise ValueError("Job commit limit must be positive")
        self.commit_limit_bytes = int(commit_limit_bytes)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()
        self._job_handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._job_handle:
            raise WorkerContainmentError(
                f"CREATE_JOB_OBJECT_FAILED:{ctypes.get_last_error()}"
            )
        self._completion_port = wintypes.HANDLE()
        self._closed = False
        self._limit_violation_detected = False
        try:
            if not self._kernel32.SetHandleInformation(self._job_handle, 0x00000001, 0):
                self._raise_last_error("JOB_HANDLE_INHERITANCE_DISABLE_FAILED")
            self._install_limits()
            self._install_completion_port()
            self.installation_verified = self._verify_installed_limit()
            if not self.installation_verified:
                raise WorkerContainmentError("JOB_COMMIT_LIMIT_VERIFICATION_FAILED")
            self.snapshot()
        except Exception:
            self.close()
            raise

    def _configure_signatures(self) -> None:
        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.SetHandleInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        kernel32.CreateIoCompletionPort.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
        kernel32.GetQueuedCompletionStatus.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.DWORD,
        ]
        kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
        kernel32.Thread32First.restype = wintypes.BOOL
        kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
        kernel32.Thread32Next.restype = wintypes.BOOL
        kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenThread.restype = wintypes.HANDLE
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE

    def _raise_last_error(self, operation: str) -> None:
        raise WorkerContainmentError(f"{operation}:{ctypes.get_last_error()}")

    def _install_limits(self) -> None:
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_JOB_MEMORY | self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        limits.JobMemoryLimit = self.commit_limit_bytes
        if not self._kernel32.SetInformationJobObject(
            self._job_handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self._raise_last_error("SET_JOB_COMMIT_LIMIT_FAILED")

    def _install_completion_port(self) -> None:
        port = self._kernel32.CreateIoCompletionPort(wintypes.HANDLE(-1), None, 0, 1)
        if not port:
            self._raise_last_error("CREATE_JOB_COMPLETION_PORT_FAILED")
        self._completion_port = port
        association = _JOBOBJECT_ASSOCIATE_COMPLETION_PORT(ctypes.c_void_p(id(self)), port)
        if not self._kernel32.SetInformationJobObject(
            self._job_handle,
            self._JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION_CLASS,
            ctypes.byref(association),
            ctypes.sizeof(association),
        ):
            self._raise_last_error("ASSOCIATE_JOB_COMPLETION_PORT_FAILED")

    def _query(self, information_class: int, value: ctypes.Structure) -> None:
        returned = wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self._job_handle,
            information_class,
            ctypes.byref(value),
            ctypes.sizeof(value),
            ctypes.byref(returned),
        ):
            self._raise_last_error(f"QUERY_JOB_INFORMATION_FAILED:{information_class}")

    def _verify_installed_limit(self) -> bool:
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        self._query(self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, limits)
        required_flags = (
            self._JOB_OBJECT_LIMIT_JOB_MEMORY
            | self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        return bool(
            limits.BasicLimitInformation.LimitFlags & required_flags == required_flags
            and int(limits.JobMemoryLimit) == self.commit_limit_bytes
            and not (limits.BasicLimitInformation.LimitFlags & 0x00000800)
        )

    @staticmethod
    def _process_handle(process: subprocess.Popen[Any]) -> wintypes.HANDLE:
        handle = getattr(process, "_handle", None)
        if handle is None:
            raise WorkerContainmentError("SUBPROCESS_HANDLE_UNAVAILABLE")
        return wintypes.HANDLE(int(handle))

    def _contains_handle(self, process_handle: wintypes.HANDLE) -> bool:
        result = wintypes.BOOL()
        if not self._kernel32.IsProcessInJob(
            process_handle, self._job_handle, ctypes.byref(result)
        ):
            self._raise_last_error("VERIFY_JOB_MEMBERSHIP_FAILED")
        return bool(result.value)

    def contains_pid(self, pid: int) -> bool:
        process_handle = self._kernel32.OpenProcess(0x1000, False, int(pid))
        if not process_handle:
            return False
        try:
            return self._contains_handle(process_handle)
        finally:
            self._kernel32.CloseHandle(process_handle)

    def _resume_suspended_process(self, pid: int) -> None:
        snapshot = self._kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
        if ctypes.c_void_p(snapshot).value == ctypes.c_void_p(-1).value:
            self._raise_last_error("THREAD_SNAPSHOT_FAILED")
        resumed = 0
        try:
            entry = _THREADENTRY32()
            entry.dwSize = ctypes.sizeof(entry)
            has_entry = self._kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while has_entry:
                if int(entry.th32OwnerProcessID) == int(pid):
                    thread = self._kernel32.OpenThread(0x0002, False, entry.th32ThreadID)
                    if not thread:
                        self._raise_last_error("OPEN_SUSPENDED_WORKER_THREAD_FAILED")
                    try:
                        if self._kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                            self._raise_last_error("RESUME_WORKER_THREAD_FAILED")
                        resumed += 1
                    finally:
                        self._kernel32.CloseHandle(thread)
                has_entry = self._kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self._kernel32.CloseHandle(snapshot)
        if resumed != 1:
            raise WorkerContainmentError(
                f"SUSPENDED_WORKER_PRIMARY_THREAD_COUNT_INVALID:{resumed}"
            )

    def launch(
        self,
        command: Sequence[str],
        *,
        cwd: os.PathLike[str] | str,
        stdout: int | TextIO | BinaryIO | None,
        stderr: int | TextIO | BinaryIO | None,
        env: Mapping[str, str],
        before_resume: Callable[[int], None] | None = None,
    ) -> subprocess.Popen[Any]:
        if self._closed or not self.installation_verified:
            raise WorkerContainmentError("WORKER_JOB_NOT_AVAILABLE")
        process = subprocess.Popen(  # noqa: S603 - caller supplies frozen authority
            list(command),
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            env=dict(env),
            creationflags=self._CREATE_SUSPENDED,
        )
        process_handle = self._process_handle(process)
        try:
            if not self._kernel32.AssignProcessToJobObject(self._job_handle, process_handle):
                self._raise_last_error("ASSIGN_WORKER_TO_JOB_FAILED")
            if not self._contains_handle(process_handle):
                raise WorkerContainmentError("WORKER_JOB_MEMBERSHIP_NOT_ESTABLISHED")
            if before_resume is not None:
                before_resume(process.pid)
            self._resume_suspended_process(process.pid)
        except Exception:
            process.kill()
            process.wait(timeout=10.0)
            raise
        return process

    def _drain_notifications(self) -> None:
        while self._completion_port and not self._closed:
            message = wintypes.DWORD()
            completion_key = ctypes.c_size_t()
            overlapped = ctypes.c_void_p()
            succeeded = self._kernel32.GetQueuedCompletionStatus(
                self._completion_port,
                ctypes.byref(message),
                ctypes.byref(completion_key),
                ctypes.byref(overlapped),
                0,
            )
            if not succeeded:
                if ctypes.get_last_error() == 258:
                    return
                self._raise_last_error("READ_JOB_COMPLETION_PORT_FAILED")
            if int(message.value) in {
                self._JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT,
                self._JOB_OBJECT_MSG_JOB_MEMORY_LIMIT,
            }:
                self._limit_violation_detected = True

    def snapshot(self) -> JobMemorySnapshot:
        if self._closed:
            raise WorkerContainmentError("WORKER_JOB_ALREADY_CLOSED")
        self._drain_notifications()
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        accounting = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        usage = _JOBOBJECT_MEMORY_USAGE_INFORMATION()
        self._query(self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, limits)
        self._query(self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS, accounting)
        self._query(self._JOB_OBJECT_MEMORY_USAGE_INFORMATION_CLASS, usage)
        return JobMemorySnapshot(
            committed_bytes=int(usage.JobMemory),
            peak_committed_bytes=max(
                int(usage.PeakJobMemoryUsed), int(limits.PeakJobMemoryUsed)
            ),
            commit_limit_bytes=int(limits.JobMemoryLimit),
            active_processes=int(accounting.ActiveProcesses),
            total_processes=int(accounting.TotalProcesses),
            terminated_processes=int(accounting.TotalTerminatedProcesses),
            limit_violation_detected=self._limit_violation_detected,
            installation_verified=self._verify_installed_limit(),
            checked_at_utc=_utc_now(),
        )

    def terminate(self, exit_code: int = RESOURCE_PRESSURE_EXIT_CODE) -> None:
        if not self._closed and not self._kernel32.TerminateJobObject(
            self._job_handle, int(exit_code)
        ):
            self._raise_last_error("TERMINATE_WORKER_JOB_FAILED")

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        job = getattr(self, "_job_handle", None)
        port = getattr(self, "_completion_port", None)
        if job:
            self._kernel32.CloseHandle(job)
            self._job_handle = wintypes.HANDLE()
        if port:
            self._kernel32.CloseHandle(port)
            self._completion_port = wintypes.HANDLE()

    def __enter__(self) -> "WindowsWorkerJob":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


class UnboundedNonWindowsWorkerContainment:
    """Non-Windows adapter; the hard Job Object contract is Windows-only."""

    installation_verified = True

    def launch(
        self,
        command: Sequence[str],
        *,
        cwd: os.PathLike[str] | str,
        stdout: int | TextIO | BinaryIO | None,
        stderr: int | TextIO | BinaryIO | None,
        env: Mapping[str, str],
        before_resume: Callable[[int], None] | None = None,
    ) -> subprocess.Popen[Any]:
        process = subprocess.Popen(  # noqa: S603 - caller supplies frozen authority
            list(command), cwd=cwd, stdout=stdout, stderr=stderr, env=dict(env)
        )
        if before_resume is not None:
            before_resume(process.pid)
        return process

    def snapshot(self) -> None:
        return None

    def contains_pid(self, pid: int) -> bool:
        return process_identity(pid).alive

    def close(self) -> None:
        return None


def create_worker_containment(*, host: str) -> WorkerContainment:
    if os.name == "nt":
        return WindowsWorkerJob()
    if host == "dell":
        raise WorkerContainmentError(
            "DELL_WORKER_CONTAINMENT_REQUIRES_WINDOWS_JOB_OBJECT"
        )
    return UnboundedNonWindowsWorkerContainment()


def _normalise_creation_time(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = datetime.fromtimestamp(float(value), timezone.utc)
        return timestamp.isoformat()
    except (TypeError, ValueError, OSError):
        try:
            import pandas as pd

            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC").isoformat()
        except Exception:
            return str(value)


def same_process_creation_time(left: Any, right: Any) -> bool:
    left_text = _normalise_creation_time(left)
    right_text = _normalise_creation_time(right)
    if not left_text or not right_text:
        return False
    try:
        import pandas as pd

        delta = abs(
            (
                pd.Timestamp(left_text).tz_convert("UTC")
                - pd.Timestamp(right_text).tz_convert("UTC")
            ).total_seconds()
        )
        return delta < 1.0
    except Exception:
        return left_text == right_text


def process_identity(pid: int) -> ProcessIdentity:
    if int(pid or 0) <= 0:
        return ProcessIdentity(int(pid or 0), False, "", "")
    try:
        import psutil

        process = psutil.Process(int(pid))
        return ProcessIdentity(
            pid=int(pid),
            alive=bool(process.is_running()),
            creation_time_utc=_normalise_creation_time(process.create_time()),
            command_line=" ".join(process.cmdline()),
        )
    except Exception:
        return ProcessIdentity(int(pid), False, "", "")


def process_identity_matches(
    identity: ProcessIdentity,
    *,
    expected_creation_time: Any,
    required_command_fragments: tuple[str, ...],
) -> bool:
    command = identity.command_line.lower()
    return bool(
        identity.alive
        and same_process_creation_time(identity.creation_time_utc, expected_creation_time)
        and all(fragment.lower() in command for fragment in required_command_fragments)
    )


def status_worker_identity_matches(row: Mapping[str, Any]) -> bool:
    try:
        pid = int(row.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return False
    family = str(row.get("family", ""))
    expected_creation = row.get("process_creation_time_utc")
    if not family or not expected_creation:
        return False
    return process_identity_matches(
        process_identity(pid),
        expected_creation_time=expected_creation,
        required_command_fragments=("ds24_clean_v2_family_worker.py", "--family", family),
    )


_EXPECTED_STOPPED_SUPERVISOR_CLASSIFICATIONS = frozenset(
    {
        "DS24_CLEAN_V2_TOURNAMENT_BLOCKED_WORKER_CONTAINMENT",
        "DS24_CLEAN_V2_TOURNAMENT_COMPLETE",
        "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED",
        "DS24_CLEAN_V2_TOURNAMENT_PAUSED_RESOURCE_PRESSURE",
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE",
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_CONTROL_RECONCILED",
        "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_EXIT_FAILED_CLOSED",
    }
)


def reconcile_runtime_status_view(
    status: Mapping[str, Any],
    *,
    host: str,
) -> dict[str, Any]:
    """Return a truthful status view without trusting stale PID-only records."""

    reconciled = copy.deepcopy(dict(status))
    supervisor_pid = int(reconciled.get("supervisor_pid", 0) or 0)
    supervisor_creation = reconciled.get("supervisor_process_creation_time_utc")
    supervisor_verified = bool(
        supervisor_pid
        and supervisor_creation
        and process_identity_matches(
            process_identity(supervisor_pid),
            expected_creation_time=supervisor_creation,
            required_command_fragments=(
                "ds24_clean_v2_supervisor.py",
                "--run-queue",
                "--host",
                host,
            ),
        )
    )
    reported_workers = [
        row
        for row in reconciled.get("active_workers", [])
        if isinstance(row, Mapping)
    ]
    verified_workers = [
        dict(row) for row in reported_workers if status_worker_identity_matches(row)
    ]
    stale_workers = [
        dict(row) for row in reported_workers if not status_worker_identity_matches(row)
    ]
    prior_stale_workers = [
        dict(row)
        for row in reconciled.get("stale_worker_records_reconciled", [])
        if isinstance(row, Mapping)
    ]
    reservations = reconciled.get("worker_reservations")
    reservation_rows = []
    if isinstance(reservations, Mapping):
        reservation_rows = [
            row
            for row in reservations.get("workers", [])
            if isinstance(row, Mapping)
        ]
    live_reservations: list[dict[str, Any]] = []
    reclaimed_reservations: list[dict[str, Any]] = []
    for row in reservation_rows:
        owner = row.get("owner") or {}
        if isinstance(owner, Mapping) and status_worker_identity_matches(owner):
            live_reservations.append(dict(row))
        else:
            reclaimed_reservations.append(dict(row))
    prior_reclaimed_reservations = [
        dict(row)
        for row in reconciled.get("stale_reservations_reclaimed", [])
        if isinstance(row, Mapping)
    ]

    runtime_recorded = bool(
        supervisor_pid or reported_workers or reservation_rows
    )
    reported_classification = str(reconciled.get("classification") or "")
    was_already_reconciled = reported_classification in {
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_STALE_RUNTIME_RECONCILED",
        "DS24_CLEAN_V2_ORPHANED_VERIFIED_WORKERS_DETECTED",
    } or reported_classification in _EXPECTED_STOPPED_SUPERVISOR_CLASSIFICATIONS
    if not supervisor_verified and runtime_recorded:
        reconciled["active_workers"] = []
        if verified_workers:
            reconciled["classification"] = (
                "DS24_CLEAN_V2_ORPHANED_VERIFIED_WORKERS_DETECTED"
            )
            reconciled["orphaned_verified_workers"] = verified_workers
        elif reported_classification not in (
            _EXPECTED_STOPPED_SUPERVISOR_CLASSIFICATIONS
        ):
            reconciled["classification"] = (
                "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_"
                "STALE_RUNTIME_RECONCILED"
            )
        live_reservations = []
        reclaimed_reservations = [dict(row) for row in reservation_rows]
    else:
        reconciled["active_workers"] = verified_workers

    if isinstance(reservations, Mapping):
        outstanding_physical = sum(
            int((row.get("outstanding") or {}).get("physical_bytes", 0) or 0)
            for row in live_reservations
        )
        outstanding_commit = sum(
            int((row.get("outstanding") or {}).get("commit_bytes", 0) or 0)
            for row in live_reservations
        )
        reconciled["worker_reservations"] = {
            **dict(reservations),
            "active_worker_reservations": len(live_reservations),
            "outstanding": {
                "physical_bytes": outstanding_physical,
                "commit_bytes": outstanding_commit,
            },
            "workers": live_reservations,
        }
    reconciled["supervisor_identity_verified"] = supervisor_verified
    reconciled["stale_worker_records_reconciled"] = [
        *prior_stale_workers,
        *stale_workers,
    ]
    reconciled["stale_reservations_reclaimed"] = [
        *prior_reclaimed_reservations,
        *reclaimed_reservations,
    ]
    reconciled["runtime_reconciliation_required"] = bool(
        stale_workers
        or reclaimed_reservations
        or (
            runtime_recorded
            and not supervisor_verified
            and not was_already_reconciled
        )
    )
    return reconciled
