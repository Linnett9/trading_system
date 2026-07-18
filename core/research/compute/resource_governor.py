from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterable

from .machine_profile import MachineProfile

REQUEST_CONTRACT = "compute_resource_request.v1"
LEASE_CONTRACT = "compute_resource_lease.v1"
VALID_CLASSES = {"SMALL", "MEDIUM", "LARGE", "XLARGE"}
VALID_ESTIMATE_SOURCES = {
    "CONSERVATIVE_DEFAULT", "BOUNDED_SMOKE", "HISTORICAL_MEASUREMENT",
}
DEEP_GROUPS = {"SELECTOR_DEEP", "EXPOSURE_DEEP", "NEWS_TRANSFORMER"}


def _checksum(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ResourceRequest:
    pipeline: str
    stage: str
    job_id: str
    run_id: str
    resource_class: str
    estimated_peak_ram_bytes: int
    cpu_weight: int
    inner_threads: int
    gpu_required: bool
    concurrency_group: str
    estimate_source: str
    estimate_evidence_identity: str
    lightweight: bool = False
    safe_to_colocate: bool = True
    attempt_identity: str | None = None
    estimated_gpu_ram_bytes: int | None = None
    temporary_disk_bytes: int | None = None
    contract_version: str = REQUEST_CONTRACT
    logical_checksum: str = ""

    def __post_init__(self) -> None:
        if not all((self.pipeline, self.stage, self.job_id, self.run_id,
                    self.concurrency_group, self.estimate_evidence_identity)):
            raise ValueError("Resource request identity fields are required")
        if self.contract_version != REQUEST_CONTRACT:
            raise ValueError("Unsupported resource request contract")
        if self.resource_class not in VALID_CLASSES:
            raise ValueError("Unsupported resource class")
        if self.estimated_peak_ram_bytes <= 0:
            raise ValueError("Production resource request requires a RAM estimate")
        if self.cpu_weight <= 0 or self.inner_threads <= 0:
            raise ValueError("CPU weight and inner threads must be positive")
        if self.estimate_source not in VALID_ESTIMATE_SOURCES:
            raise ValueError("Unsupported estimate source")
        expected = _checksum(self.logical_payload())
        if self.logical_checksum and self.logical_checksum != expected:
            raise ValueError("Resource request logical checksum mismatch")
        if not self.logical_checksum:
            object.__setattr__(self, "logical_checksum", expected)

    def logical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("logical_checksum", None)
        return payload

    @property
    def immutable_job_identity(self) -> str:
        return f"{self.pipeline}:{self.stage}:{self.job_id}"


class LeaseStatus(str, Enum):
    REQUESTED = "REQUESTED"
    WAITING = "WAITING"
    GRANTED = "GRANTED"
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    FAILED_TO_START = "FAILED_TO_START"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    STALE_CANDIDATE = "STALE_CANDIDATE"
    STALE_CONFIRMED = "STALE_CONFIRMED"
    UNKNOWN_OWNER = "UNKNOWN_OWNER"


@dataclass
class ResourceLease:
    request: ResourceRequest
    machine_profile_identity: str = ""
    status: LeaseStatus = LeaseStatus.REQUESTED
    requested_at: str = ""
    granted_at: str | None = None
    process_started_at: str | None = None
    released_at: str | None = None
    release_reason: str | None = None
    blocked_reasons: tuple[str, ...] = ()
    process_id: int | None = None
    process_start_timestamp: str | None = None
    heartbeat_timestamp: str | None = None
    current_phase: str | None = None
    telemetry_identity: str | None = None
    command_identity: str | None = None
    process_classification: str | None = None
    stale_candidate_at: str | None = None
    contract_version: str = LEASE_CONTRACT
    logical_identity: str = field(init=False)

    def __post_init__(self) -> None:
        self.logical_identity = _checksum({
            "contract_version": self.contract_version,
            "request_identity": self.request.logical_checksum,
            "run_id": self.request.run_id,
            "job_id": self.request.job_id,
            "attempt_identity": self.request.attempt_identity,
            "machine_profile_identity": self.machine_profile_identity,
        })

    @property
    def reserved_ram_bytes(self) -> int:
        return self.request.estimated_peak_ram_bytes

    @property
    def reserved_cpu_weight(self) -> int:
        return self.request.cpu_weight


class ResourceGovernor:
    """Thread-safe, deterministic in-memory admission for one orchestrator."""

    def __init__(
        self,
        profile: MachineProfile,
        *,
        available_memory: Callable[[], int | None],
        available_gpus: Callable[[], Iterable[str]] = lambda: (),
        incompatible_groups: Iterable[tuple[str, str]] = (),
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.profile = profile
        self._available_memory = available_memory
        self._available_gpus = available_gpus
        self._clock = clock
        self._lock = threading.RLock()
        self._active: list[ResourceLease] = []
        self._waiting: list[ResourceLease] = []
        self._released: list[ResourceLease] = []
        self._incompatible = {
            frozenset((left, right)) for left, right in incompatible_groups
        }

    def restore_state(
        self,
        *,
        active: Iterable[ResourceLease] = (),
        waiting: Iterable[ResourceLease] = (),
        released: Iterable[ResourceLease] = (),
    ) -> None:
        """Restore validated persisted state before a locked policy decision."""
        with self._lock:
            active_rows = list(active)
            waiting_rows = list(waiting)
            identities = [
                row.request.immutable_job_identity
                for row in active_rows + waiting_rows
            ]
            attempts = [
                row.request.attempt_identity
                for row in active_rows + waiting_rows
            ]
            if len(identities) != len(set(identities)):
                raise ValueError("Duplicate active or waiting immutable job identity")
            if None in attempts or len(attempts) != len(set(attempts)):
                raise ValueError("Persisted leases require unique attempt identities")
            reserving = {
                LeaseStatus.GRANTED, LeaseStatus.ACTIVE,
                LeaseStatus.STALE_CANDIDATE, LeaseStatus.UNKNOWN_OWNER,
            }
            if any(row.status not in reserving for row in active_rows):
                raise ValueError("Invalid persisted active lease state")
            if any(row.status != LeaseStatus.WAITING for row in waiting_rows):
                raise ValueError("Invalid persisted waiting lease state")
            if any(
                row.machine_profile_identity != self.profile.logical_checksum
                for row in active_rows + waiting_rows
            ):
                raise ValueError("Persisted lease machine profile mismatch")
            self._active = active_rows
            self._waiting = waiting_rows
            self._released = list(released)

    def request_lease(self, request: ResourceRequest) -> ResourceLease:
        with self._lock:
            duplicate = next((
                lease for lease in self._active + self._waiting
                if lease.request.immutable_job_identity
                == request.immutable_job_identity
                and (
                    not request.attempt_identity
                    or lease.request.attempt_identity == request.attempt_identity
                )
            ), None)
            if duplicate:
                raise ValueError("Duplicate active or queued immutable job identity")
            lease = ResourceLease(
                request=request,
                machine_profile_identity=self.profile.logical_checksum,
                requested_at=self._now(),
                status=LeaseStatus.WAITING,
            )
            self._waiting.append(lease)
            self._admit_in_order()
            return lease

    def reevaluate_waiting(self) -> None:
        with self._lock:
            self._admit_in_order()

    def snapshot_leases(
        self,
    ) -> tuple[list[ResourceLease], list[ResourceLease], list[ResourceLease]]:
        with self._lock:
            return (
                list(self._active), list(self._waiting), list(self._released)
            )

    def mark_process_started(self, lease: ResourceLease, process_id: int) -> None:
        with self._lock:
            if lease not in self._active or lease.status != LeaseStatus.GRANTED:
                raise ValueError("Process may start only after lease grant")
            lease.status = LeaseStatus.ACTIVE
            lease.process_id = process_id
            lease.process_started_at = self._now()

    def record_phase(self, lease: ResourceLease, phase: str) -> dict[str, str]:
        if not phase:
            raise ValueError("Execution phase is required")
        return {"lease_identity": lease.logical_identity, "phase": phase, "timestamp": self._now()}

    def release_lease(self, lease: ResourceLease, reason: str = "SUCCESS") -> None:
        terminal = {
            "STARTUP_FAILURE": LeaseStatus.FAILED_TO_START,
            "CANCELLED": LeaseStatus.CANCELLED,
        }.get(reason, LeaseStatus.RELEASED)
        with self._lock:
            if lease not in self._active:
                raise ValueError("Only an active reservation can be released")
            self._active.remove(lease)
            lease.status = terminal
            lease.release_reason = reason
            lease.released_at = self._now()
            self._released.append(lease)
            self._admit_in_order()

    def status(self) -> dict[str, object]:
        with self._lock:
            actual = self._available_memory()
            reserved_ram = sum(row.reserved_ram_bytes for row in self._active)
            reserved_cpu = sum(row.reserved_cpu_weight for row in self._active)
            return {
                "machine_profile_identity": self.profile.logical_checksum,
                "detected_available_ram_bytes": actual,
                "schedulable_ram_bytes": self.profile.schedulable_ram_bytes,
                "active_reserved_ram_bytes": reserved_ram,
                "remaining_reservable_ram_bytes": max(
                    self.profile.schedulable_ram_bytes - reserved_ram, 0
                ),
                "active_cpu_weight": reserved_cpu,
                "remaining_cpu_capacity": max(
                    self.profile.cpu_capacity - reserved_cpu, 0
                ),
                "active_leases": [self._lease_summary(row) for row in self._active],
                "waiting_requests": [self._lease_summary(row) for row in self._waiting],
                "latest_released_jobs": [
                    self._lease_summary(row) for row in self._released[-10:]
                ],
                "telemetry_health": "NOT_CONNECTED",
            }

    def _admit_in_order(self) -> None:
        while self._waiting:
            lease = self._waiting[0]
            reasons = self._blocked_reasons(lease.request)
            lease.blocked_reasons = tuple(reasons)
            if reasons:
                for successor in self._waiting[1:]:
                    successor.blocked_reasons = ("QUEUE_PREDECESSOR_BLOCKED",)
                break
            self._waiting.pop(0)
            lease.status = LeaseStatus.GRANTED
            lease.granted_at = self._now()
            self._active.append(lease)

    def _blocked_reasons(self, request: ResourceRequest) -> list[str]:
        actual = self._available_memory()
        if actual is None:
            return ["AVAILABLE_MEMORY_UNKNOWN"]
        if actual < self.profile.critical_reserve_bytes:
            return ["CRITICAL_MEMORY_RESERVE"]
        reasons: list[str] = []
        reserved_ram = sum(row.reserved_ram_bytes for row in self._active)
        if reserved_ram + request.estimated_peak_ram_bytes > self.profile.schedulable_ram_bytes:
            reasons.append("SCHEDULABLE_RAM_CAPACITY")
        if actual - request.estimated_peak_ram_bytes < self.profile.normal_reserve_bytes:
            reasons.append("NORMAL_MEMORY_RESERVE")
        if sum(row.reserved_cpu_weight for row in self._active) + request.cpu_weight > self.profile.cpu_capacity:
            reasons.append("CPU_WEIGHT_CAPACITY")
        if request.lightweight and sum(row.request.lightweight for row in self._active) >= self.profile.max_lightweight_jobs:
            reasons.append("LIGHTWEIGHT_CONCURRENCY_LIMIT")
        if request.concurrency_group in DEEP_GROUPS and sum(
            row.request.concurrency_group in DEEP_GROUPS for row in self._active
        ) >= self.profile.deep_model_limit:
            reasons.append("DEEP_MODEL_CONCURRENCY_LIMIT")
        available_gpus = tuple(self._available_gpus())
        if request.gpu_required and not available_gpus:
            reasons.append("REQUIRED_GPU_UNAVAILABLE")
        elif request.gpu_required and sum(
            row.request.gpu_required for row in self._active
        ) >= len(available_gpus):
            reasons.append("GPU_CAPACITY")
        for row in self._active:
            pair = frozenset((request.concurrency_group, row.request.concurrency_group))
            if pair in self._incompatible or (
                (not request.safe_to_colocate or not row.request.safe_to_colocate) and
                request.concurrency_group != row.request.concurrency_group
            ):
                reasons.append("CONCURRENCY_GROUP_INCOMPATIBLE")
                break
        return reasons

    def _now(self) -> str:
        return self._clock().isoformat()

    @staticmethod
    def _lease_summary(lease: ResourceLease) -> dict[str, object]:
        return {
            "lease_identity": lease.logical_identity,
            "run_id": lease.request.run_id,
            "job_id": lease.request.job_id,
            "status": lease.status.value,
            "reserved_ram_bytes": lease.reserved_ram_bytes,
            "reserved_cpu_weight": lease.reserved_cpu_weight,
            "blocked_reasons": list(lease.blocked_reasons),
            "process_id": lease.process_id,
        }
