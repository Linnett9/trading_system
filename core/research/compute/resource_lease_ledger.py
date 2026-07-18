from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

from .lease_storage import (
    LedgerLockTimeout,
    atomic_write_json,
    exclusive_file_lock,
    quarantine_copy,
)
from .machine_profile import MachineProfile
from .resource_governor import (
    LeaseStatus,
    ResourceGovernor,
    ResourceLease,
    ResourceRequest,
)

LEDGER_CONTRACT = "compute_resource_lease_ledger.v1"
DEFAULT_LEDGER_PATH = Path("reports/compute/resource_leases/resource_lease_ledger.json")
RESERVING_STATES = {
    LeaseStatus.GRANTED, LeaseStatus.ACTIVE,
    LeaseStatus.STALE_CANDIDATE, LeaseStatus.UNKNOWN_OWNER,
}


class LedgerCorrupt(RuntimeError):
    pass


class StaleLedgerRevision(RuntimeError):
    pass


class ProcessClassification(str, Enum):
    LIVE_MATCH = "LIVE_MATCH"
    PID_REUSED = "PID_REUSED"
    PROCESS_MISSING = "PROCESS_MISSING"
    PROCESS_UNKNOWN = "PROCESS_UNKNOWN"
    ACCESS_DENIED = "ACCESS_DENIED"
    METRICS_UNAVAILABLE = "METRICS_UNAVAILABLE"


@dataclass(frozen=True)
class ProcessObservation:
    classification: ProcessClassification
    process_start_timestamp: str | None = None


class ProcessProvider(Protocol):
    def inspect(
        self, process_id: int, expected_start_timestamp: str | None
    ) -> ProcessObservation: ...


class OptionalPsutilProcessProvider:
    def inspect(
        self, process_id: int, expected_start_timestamp: str | None
    ) -> ProcessObservation:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return ProcessObservation(ProcessClassification.METRICS_UNAVAILABLE)
        try:
            actual = datetime.fromtimestamp(
                psutil.Process(process_id).create_time(), timezone.utc
            ).isoformat()
        except psutil.NoSuchProcess:
            return ProcessObservation(ProcessClassification.PROCESS_MISSING)
        except psutil.AccessDenied:
            return ProcessObservation(ProcessClassification.ACCESS_DENIED)
        except (OSError, RuntimeError):
            return ProcessObservation(ProcessClassification.PROCESS_UNKNOWN)
        if expected_start_timestamp and not _same_process_time(
            actual, expected_start_timestamp
        ):
            return ProcessObservation(ProcessClassification.PID_REUSED, actual)
        return ProcessObservation(ProcessClassification.LIVE_MATCH, actual)


class ResourceLeaseLedger:
    def __init__(
        self,
        *,
        profile: MachineProfile,
        path: Path = DEFAULT_LEDGER_PATH,
        available_memory: Callable[[], int | None],
        available_gpus: Callable[[], Iterable[str]] = lambda: (),
        process_provider: ProcessProvider | None = None,
        incompatible_groups: Iterable[tuple[str, str]] = (),
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        lock_timeout_seconds: float = 10.0,
        history_limit: int = 100,
        minimum_heartbeat_seconds: float = 1.0,
    ) -> None:
        if history_limit < 1 or minimum_heartbeat_seconds < 1:
            raise ValueError("History and heartbeat limits must be positive")
        self.profile = profile
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(".lock")
        self._available_memory = available_memory
        self._available_gpus = available_gpus
        self._process_provider = process_provider or OptionalPsutilProcessProvider()
        self._incompatible_groups = tuple(incompatible_groups)
        self._clock = clock
        self._lock_timeout = lock_timeout_seconds
        self._history_limit = history_limit
        self._minimum_heartbeat_seconds = minimum_heartbeat_seconds

    def initialise_ledger(self) -> dict[str, object]:
        with self._locked():
            if self.path.exists():
                return self._load()
            payload = self._empty()
            self._publish(payload, previous_revision=None)
            return payload

    def request_persisted_lease(
        self,
        request: ResourceRequest,
        *,
        expected_revision: int | None = None,
    ) -> tuple[ResourceLease, int]:
        if not request.attempt_identity:
            raise ValueError("Persisted requests require an attempt identity")
        result: ResourceLease | None = None

        def mutation(payload: dict[str, object]) -> None:
            nonlocal result
            governor = self._governor(payload)
            result = governor.request_lease(request)
            self._store_governor(payload, governor)

        payload = self._mutate(mutation, expected_revision=expected_revision)
        assert result is not None
        return result, int(payload["revision"])

    def activate_persisted_lease(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        process_id: int,
        process_start_timestamp: str,
        command_identity: str | None = None,
        expected_revision: int | None = None,
    ) -> int:
        def mutation(payload: dict[str, object]) -> None:
            lease = self._find(payload, lease_identity, active_only=True)
            self._expect_attempt(lease, attempt_identity)
            if lease.status != LeaseStatus.GRANTED:
                raise ValueError("Only a granted lease may become active")
            lease.status = LeaseStatus.ACTIVE
            lease.process_id = process_id
            lease.process_start_timestamp = process_start_timestamp
            lease.process_started_at = self._now()
            lease.heartbeat_timestamp = self._now()
            lease.command_identity = command_identity
            lease.process_classification = ProcessClassification.LIVE_MATCH.value
            self._replace(payload, lease)

        return int(self._mutate(mutation, expected_revision=expected_revision)["revision"])

    def heartbeat_persisted_lease(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        process_id: int,
        process_start_timestamp: str,
        phase: str | None = None,
        telemetry_identity: str | None = None,
        measured_memory_summary: Mapping[str, object] | None = None,
        expected_revision: int | None = None,
    ) -> int:
        def mutation(payload: dict[str, object]) -> None:
            lease = self._find(payload, lease_identity, active_only=True)
            self._expect_attempt(lease, attempt_identity)
            if lease.status not in {
                LeaseStatus.ACTIVE, LeaseStatus.UNKNOWN_OWNER,
                LeaseStatus.STALE_CANDIDATE,
            }:
                raise ValueError("Lease state does not permit heartbeat")
            if (
                lease.process_id != process_id
                or lease.process_start_timestamp != process_start_timestamp
            ):
                raise ValueError("Heartbeat process identity mismatch")
            if self._age(lease.heartbeat_timestamp) < self._minimum_heartbeat_seconds:
                raise ValueError("Heartbeat update is too frequent")
            lease.status = LeaseStatus.ACTIVE
            lease.heartbeat_timestamp = self._now()
            lease.current_phase = phase
            lease.telemetry_identity = telemetry_identity
            lease.process_classification = ProcessClassification.LIVE_MATCH.value
            lease.stale_candidate_at = None
            row = _lease_to_dict(lease)
            row["measured_memory_summary"] = (
                dict(measured_memory_summary) if measured_memory_summary else None
            )
            self._replace_row(payload, row)

        return int(self._mutate(mutation, expected_revision=expected_revision)["revision"])

    def release_persisted_lease(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        reason: str,
        expected_revision: int | None = None,
    ) -> int:
        return self._terminal_transition(
            lease_identity, attempt_identity=attempt_identity,
            status=LeaseStatus.RELEASED, reason=reason,
            expected_revision=expected_revision,
        )

    def fail_persisted_lease(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        reason: str,
        startup_failure: bool = False,
        expected_revision: int | None = None,
    ) -> int:
        return self._terminal_transition(
            lease_identity, attempt_identity=attempt_identity,
            status=(
                LeaseStatus.FAILED_TO_START
                if startup_failure else LeaseStatus.FAILED
            ),
            reason=reason, expected_revision=expected_revision,
        )

    def cancel_persisted_lease(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        reason: str,
        expected_revision: int | None = None,
    ) -> int:
        return self._terminal_transition(
            lease_identity, attempt_identity=attempt_identity,
            status=LeaseStatus.CANCELLED, reason=reason,
            expected_revision=expected_revision,
        )

    def reconcile_ledger(
        self,
        *,
        stale_threshold_seconds: float,
        confirmation_grace_seconds: float,
        expected_revision: int | None = None,
    ) -> dict[str, object]:
        if stale_threshold_seconds <= 0 or confirmation_grace_seconds <= 0:
            raise ValueError("Reconciliation thresholds must be positive")
        observations: list[dict[str, object]] = []

        def mutation(payload: dict[str, object]) -> None:
            active = [_lease_from_dict(row) for row in payload["active_leases"]]  # type: ignore[index]
            retained: list[ResourceLease] = []
            for lease in active:
                observation = self._inspect(lease)
                lease.process_classification = observation.classification.value
                age = self._heartbeat_age(lease)
                event: dict[str, object] = {
                    "lease_identity": lease.logical_identity,
                    "classification": observation.classification.value,
                    "heartbeat_age_seconds": age,
                    "timestamp": self._now(),
                }
                if observation.classification == ProcessClassification.LIVE_MATCH:
                    lease.status = LeaseStatus.ACTIVE
                    lease.stale_candidate_at = None
                elif observation.classification in {
                    ProcessClassification.PROCESS_MISSING,
                    ProcessClassification.PID_REUSED,
                } and age is not None and age >= stale_threshold_seconds:
                    if lease.status == LeaseStatus.STALE_CANDIDATE and (
                        self._age(lease.stale_candidate_at) >= confirmation_grace_seconds
                    ):
                        lease.status = LeaseStatus.STALE_CONFIRMED
                        lease.released_at = self._now()
                        lease.release_reason = "SECOND_RECONCILIATION_CONFIRMED_STALE"
                        self._append_history(payload, lease)
                        event["action"] = "STALE_CONFIRMED"
                        observations.append(event)
                        continue
                    lease.status = LeaseStatus.STALE_CANDIDATE
                    lease.stale_candidate_at = lease.stale_candidate_at or self._now()
                    event["action"] = "STALE_CANDIDATE"
                elif observation.classification in {
                    ProcessClassification.PROCESS_UNKNOWN,
                    ProcessClassification.ACCESS_DENIED,
                    ProcessClassification.METRICS_UNAVAILABLE,
                }:
                    lease.status = LeaseStatus.UNKNOWN_OWNER
                    event["action"] = "RESERVATION_RETAINED"
                retained.append(lease)
                observations.append(event)
            payload["active_leases"] = [_lease_to_dict(row) for row in retained]
            payload["reconciliation_history"] = (
                list(payload.get("reconciliation_history", [])) + observations
            )[-self._history_limit:]
            payload["last_reconciliation_timestamp"] = self._now()
            self._promote_waiting(payload)

        payload = self._mutate(mutation, expected_revision=expected_revision)
        return {"revision": payload["revision"], "observations": observations}

    def confirm_stale_release(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        operator_reason: str,
        expected_revision: int | None = None,
    ) -> int:
        if not operator_reason.strip():
            raise ValueError("Operator reason is required")

        def mutation(payload: dict[str, object]) -> None:
            lease = self._find(payload, lease_identity, active_only=True)
            self._expect_attempt(lease, attempt_identity)
            if lease.status != LeaseStatus.STALE_CANDIDATE:
                raise ValueError("Only a stale candidate may be manually confirmed")
            lease.status = LeaseStatus.STALE_CONFIRMED
            lease.released_at = self._now()
            lease.release_reason = f"OPERATOR_CONFIRMED: {operator_reason}"
            self._remove_active(payload, lease.logical_identity)
            self._append_history(payload, lease)
            payload["reconciliation_history"] = (
                list(payload.get("reconciliation_history", [])) + [{
                    "lease_identity": lease.logical_identity,
                    "action": "FORCED_STALE_RELEASE",
                    "reason": operator_reason,
                    "previous_process_classification": lease.process_classification,
                    "timestamp": self._now(),
                }]
            )[-self._history_limit:]
            self._promote_waiting(payload)

        return int(self._mutate(mutation, expected_revision=expected_revision)["revision"])

    def quarantine_corrupt_ledger(self, *, operator_reason: str) -> Path:
        if not operator_reason.strip():
            raise ValueError("Operator reason is required")
        with self._locked():
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            try:
                self._load()
            except LedgerCorrupt:
                suffix = self._clock().strftime("%Y%m%dT%H%M%S%fZ")
                return quarantine_copy(
                    self.path, self.path.parent / "quarantine",
                    suffix=f"{suffix}.{hashlib.sha256(operator_reason.encode()).hexdigest()[:12]}",
                )
            raise ValueError("Healthy ledger cannot be quarantined as corrupt")

    def read_ledger_status(self) -> dict[str, object]:
        try:
            with self._locked():
                payload = self._load()
        except LedgerLockTimeout:
            return {
                "ledger_path": str(self.path),
                "ledger_health": "LOCK_UNAVAILABLE",
            }
        except LedgerCorrupt as exc:
            return {
                "ledger_path": str(self.path),
                "ledger_health": "CORRUPT",
                "error": str(exc),
            }
        active = [_lease_from_dict(row) for row in payload["active_leases"]]  # type: ignore[index]
        reserved_ram = sum(row.reserved_ram_bytes for row in active)
        reserved_cpu = sum(row.reserved_cpu_weight for row in active)
        stale = [row for row in active if row.status == LeaseStatus.STALE_CANDIDATE]
        unknown = [row for row in active if row.status == LeaseStatus.UNKNOWN_OWNER]
        health = (
            "DEGRADED_UNKNOWN_OWNER" if unknown
            else "STALE_CANDIDATES" if stale else "HEALTHY"
        )
        return {
            "ledger_contract_version": LEDGER_CONTRACT,
            "ledger_path": str(self.path),
            "ledger_revision": payload["revision"],
            "machine_profile_identity": self.profile.logical_checksum,
            "lock_status": "AVAILABLE",
            "detected_available_ram_bytes": self._available_memory(),
            "active_reserved_ram_bytes": reserved_ram,
            "remaining_reservable_ram_bytes": max(
                self.profile.schedulable_ram_bytes - reserved_ram, 0
            ),
            "active_cpu_weight": reserved_cpu,
            "remaining_cpu_capacity": max(
                self.profile.cpu_capacity - reserved_cpu, 0
            ),
            "active_leases": payload["active_leases"],
            "waiting_requests": payload["waiting_requests"],
            "stale_candidates": [_lease_to_dict(row) for row in stale],
            "unknown_owners": [_lease_to_dict(row) for row in unknown],
            "recent_releases": payload["recent_history"],
            "last_reconciliation_time": payload.get("last_reconciliation_timestamp"),
            "ledger_health": health,
        }

    def _terminal_transition(
        self,
        lease_identity: str,
        *,
        attempt_identity: str,
        status: LeaseStatus,
        reason: str,
        expected_revision: int | None,
    ) -> int:
        if not reason.strip():
            raise ValueError("Terminal transition reason is required")

        def mutation(payload: dict[str, object]) -> None:
            lease = self._find(payload, lease_identity, active_only=False)
            self._expect_attempt(lease, attempt_identity)
            if lease.status not in RESERVING_STATES | {
                LeaseStatus.WAITING, LeaseStatus.GRANTED
            }:
                raise ValueError("Invalid terminal lease transition")
            self._remove_any(payload, lease.logical_identity)
            lease.status = status
            lease.released_at = self._now()
            lease.release_reason = reason
            self._append_history(payload, lease)
            self._promote_waiting(payload)

        return int(self._mutate(mutation, expected_revision=expected_revision)["revision"])

    def _mutate(
        self,
        operation: Callable[[dict[str, object]], None],
        *,
        expected_revision: int | None,
    ) -> dict[str, object]:
        with self._locked():
            payload = self._load() if self.path.exists() else self._empty()
            revision = int(payload["revision"])
            if expected_revision is not None and expected_revision != revision:
                raise StaleLedgerRevision(
                    f"Expected ledger revision {expected_revision}, found {revision}"
                )
            operation(payload)
            self._publish(payload, previous_revision=revision)
            return payload

    def _publish(
        self, payload: dict[str, object], *, previous_revision: int | None
    ) -> None:
        payload["revision"] = 0 if previous_revision is None else previous_revision + 1
        payload["last_successful_update_timestamp"] = self._now()
        payload["writer_process_identity"] = {
            "process_id": os.getpid(),
            "process_start_timestamp": None,
        }
        payload["logical_checksum"] = _ledger_checksum(payload)
        self._validate(payload)
        atomic_write_json(self.path, payload)

    def _load(self) -> dict[str, object]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerCorrupt(f"LEDGER_CORRUPT: {exc}") from exc
        if not isinstance(payload, dict):
            raise LedgerCorrupt("LEDGER_CORRUPT: root must be an object")
        self._validate(payload)
        return payload

    def _validate(self, payload: Mapping[str, object]) -> None:
        try:
            if payload["contract_version"] != LEDGER_CONTRACT:
                raise ValueError("unsupported contract")
            if payload["machine_profile_identity"] != self.profile.logical_checksum:
                raise ValueError("wrong machine profile")
            if int(payload["revision"]) < 0:
                raise ValueError("negative revision")
            if payload["logical_checksum"] != _ledger_checksum(payload):
                raise ValueError("invalid checksum")
            active = [_lease_from_dict(row) for row in payload["active_leases"]]  # type: ignore[index]
            waiting = [_lease_from_dict(row) for row in payload["waiting_requests"]]  # type: ignore[index]
            governor = self._new_governor()
            governor.restore_state(active=active, waiting=waiting)
            reserved_ram = sum(row.reserved_ram_bytes for row in active)
            reserved_cpu = sum(row.reserved_cpu_weight for row in active)
            if reserved_ram > self.profile.schedulable_ram_bytes:
                raise ValueError("excessive reserved RAM")
            if reserved_cpu > self.profile.cpu_capacity:
                raise ValueError("excessive reserved CPU")
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerCorrupt(f"LEDGER_CORRUPT: {exc}") from exc

    def _empty(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_version": LEDGER_CONTRACT,
            "machine_profile_identity": self.profile.logical_checksum,
            "machine_profile_checksum": self.profile.logical_checksum,
            "revision": 0,
            "logical_checksum": "",
            "last_successful_update_timestamp": self._now(),
            "writer_process_identity": {"process_id": os.getpid()},
            "active_leases": [],
            "waiting_requests": [],
            "recent_history": [],
            "reconciliation_history": [],
            "corrupt_record_evidence": sorted(
                str(path)
                for path in (self.path.parent / "quarantine").glob("*.corrupt")
            ),
            "last_reconciliation_timestamp": None,
            "source_git_commit": self.profile.source_git_commit,
            "no_workload_execution_claim": True,
        }
        payload["logical_checksum"] = _ledger_checksum(payload)
        return payload

    def _governor(self, payload: Mapping[str, object]) -> ResourceGovernor:
        governor = self._new_governor()
        governor.restore_state(
            active=[_lease_from_dict(row) for row in payload["active_leases"]],  # type: ignore[index]
            waiting=[_lease_from_dict(row) for row in payload["waiting_requests"]],  # type: ignore[index]
        )
        return governor

    def _new_governor(self) -> ResourceGovernor:
        return ResourceGovernor(
            self.profile,
            available_memory=self._available_memory,
            available_gpus=self._available_gpus,
            incompatible_groups=self._incompatible_groups,
            clock=self._clock,
        )

    @staticmethod
    def _store_governor(
        payload: dict[str, object], governor: ResourceGovernor
    ) -> None:
        active, waiting, _ = governor.snapshot_leases()
        payload["active_leases"] = [_lease_to_dict(row) for row in active]
        payload["waiting_requests"] = [_lease_to_dict(row) for row in waiting]

    def _promote_waiting(self, payload: dict[str, object]) -> None:
        governor = self._governor(payload)
        governor.reevaluate_waiting()
        self._store_governor(payload, governor)

    def _find(
        self, payload: Mapping[str, object], identity: str, *, active_only: bool
    ) -> ResourceLease:
        collections = ["active_leases"] if active_only else [
            "active_leases", "waiting_requests"
        ]
        matches = [
            _lease_from_dict(row)
            for name in collections
            for row in payload[name]  # type: ignore[index]
            if row.get("logical_identity") == identity
        ]
        if len(matches) != 1:
            raise ValueError("Lease identity not found or not unique")
        return matches[0]

    @staticmethod
    def _expect_attempt(lease: ResourceLease, attempt_identity: str) -> None:
        if not attempt_identity or lease.request.attempt_identity != attempt_identity:
            raise ValueError("Lease attempt identity mismatch")

    @staticmethod
    def _replace(payload: dict[str, object], lease: ResourceLease) -> None:
        ResourceLeaseLedger._replace_row(payload, _lease_to_dict(lease))

    @staticmethod
    def _replace_row(payload: dict[str, object], row: dict[str, object]) -> None:
        for name in ("active_leases", "waiting_requests"):
            rows = list(payload[name])  # type: ignore[arg-type]
            for index, existing in enumerate(rows):
                if existing["logical_identity"] == row["logical_identity"]:
                    rows[index] = row
                    payload[name] = rows
                    return
        raise ValueError("Lease identity not found")

    @staticmethod
    def _remove_active(payload: dict[str, object], identity: str) -> None:
        rows = list(payload["active_leases"])  # type: ignore[arg-type]
        payload["active_leases"] = [
            row for row in rows if row["logical_identity"] != identity
        ]

    @staticmethod
    def _remove_any(payload: dict[str, object], identity: str) -> None:
        for name in ("active_leases", "waiting_requests"):
            rows = list(payload[name])  # type: ignore[arg-type]
            payload[name] = [
                row for row in rows if row["logical_identity"] != identity
            ]

    def _append_history(
        self, payload: dict[str, object], lease: ResourceLease
    ) -> None:
        payload["recent_history"] = (
            list(payload["recent_history"]) + [_lease_to_dict(lease)]  # type: ignore[arg-type]
        )[-self._history_limit:]

    def _inspect(self, lease: ResourceLease) -> ProcessObservation:
        if lease.process_id is None:
            return ProcessObservation(ProcessClassification.PROCESS_UNKNOWN)
        return self._process_provider.inspect(
            lease.process_id, lease.process_start_timestamp
        )

    def _heartbeat_age(self, lease: ResourceLease) -> float | None:
        return self._age(lease.heartbeat_timestamp or lease.granted_at)

    def _age(self, timestamp: str | None) -> float:
        if not timestamp:
            return float("inf")
        return max((self._clock() - _parse_time(timestamp)).total_seconds(), 0.0)

    def _now(self) -> str:
        return self._clock().isoformat()

    def _locked(self):
        return exclusive_file_lock(
            self.lock_path, timeout_seconds=self._lock_timeout
        )


def _lease_to_dict(lease: ResourceLease) -> dict[str, object]:
    payload = asdict(lease)
    payload["status"] = lease.status.value
    payload["blocked_reasons"] = list(lease.blocked_reasons)
    payload["logical_identity"] = lease.logical_identity
    return payload


def _lease_from_dict(payload: Mapping[str, object]) -> ResourceLease:
    request_payload = dict(payload["request"])  # type: ignore[arg-type]
    request = ResourceRequest(**request_payload)
    fields = {
        name: payload.get(name)
        for name in ResourceLease.__dataclass_fields__
        if name not in {"request", "logical_identity"}
    }
    fields["status"] = LeaseStatus(str(fields["status"]))
    fields["blocked_reasons"] = tuple(fields.get("blocked_reasons") or ())
    lease = ResourceLease(request=request, **fields)
    if payload.get("logical_identity") != lease.logical_identity:
        raise ValueError("Lease logical identity mismatch")
    return lease


def _ledger_checksum(payload: Mapping[str, object]) -> str:
    logical = dict(payload)
    for transient in (
        "logical_checksum", "last_successful_update_timestamp",
        "writer_process_identity",
    ):
        logical.pop(transient, None)
    return hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def _same_process_time(left: str, right: str) -> bool:
    return abs((_parse_time(left) - _parse_time(right)).total_seconds()) < 1.0
