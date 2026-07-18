from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.research.compute.lease_storage import (
    LedgerLockTimeout,
    exclusive_file_lock,
)
from core.research.compute.machine_profile import GIB, dell_i5_10500_profile
from core.research.compute.resource_governor import LeaseStatus, ResourceRequest
from core.research.compute.resource_lease_ledger import (
    LEDGER_CONTRACT,
    LedgerCorrupt,
    ProcessClassification,
    ProcessObservation,
    ResourceLeaseLedger,
    StaleLedgerRevision,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 18, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class Processes:
    def __init__(self, classification: ProcessClassification) -> None:
        self.classification = classification

    def inspect(
        self, process_id: int, expected_start_timestamp: str | None
    ) -> ProcessObservation:
        return ProcessObservation(self.classification, expected_start_timestamp)


def request(
    job: str,
    attempt: str,
    *,
    ram: int = 12 * GIB,
    cpu: int = 2,
) -> ResourceRequest:
    return ResourceRequest(
        pipeline="test", stage="stage", job_id=job, run_id="run",
        resource_class="LARGE", estimated_peak_ram_bytes=ram,
        cpu_weight=cpu, inner_threads=1, gpu_required=False,
        concurrency_group="SELECTOR_TREE",
        estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity="fixture", attempt_identity=attempt,
    )


def ledger(
    path: Path,
    *,
    clock: Clock | None = None,
    processes: Processes | None = None,
    timeout: float = 1.0,
) -> ResourceLeaseLedger:
    return ResourceLeaseLedger(
        profile=dell_i5_10500_profile(
            source_git_commit="abc", generated_at="frozen"
        ),
        path=path,
        available_memory=lambda: 32 * GIB,
        available_gpus=lambda: ("NVIDIA Quadro P620",),
        process_provider=processes,
        clock=clock or Clock(),
        lock_timeout_seconds=timeout,
    )


def test_initialise_validate_corruption_profile_and_revision(tmp_path: Path) -> None:
    path = tmp_path / "resource_lease_ledger.json"
    service = ledger(path)
    initial = service.initialise_ledger()
    assert initial["contract_version"] == LEDGER_CONTRACT
    assert initial["revision"] == 0
    assert service.initialise_ledger()["logical_checksum"] == initial["logical_checksum"]

    lease, revision = service.request_persisted_lease(request("a", "attempt-a"))
    assert lease.status == LeaseStatus.GRANTED
    assert revision == 1
    with pytest.raises(StaleLedgerRevision):
        service.request_persisted_lease(
            request("b", "attempt-b"), expected_revision=0
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["logical_checksum"] = "bad"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LedgerCorrupt, match="invalid checksum"):
        service.initialise_ledger()
    assert service.read_ledger_status()["ledger_health"] == "CORRUPT"
    quarantined = service.quarantine_corrupt_ledger(operator_reason="test")
    assert quarantined.exists()
    assert not path.exists()
    recovered = service.initialise_ledger()
    assert str(quarantined) in recovered["corrupt_record_evidence"]

    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(LedgerCorrupt):
        service.initialise_ledger()

    wrong = ledger(tmp_path / "wrong.json")
    wrong.initialise_ledger()
    wrong_payload = json.loads((tmp_path / "wrong.json").read_text())
    wrong_payload["machine_profile_identity"] = "another-profile"
    # Recalculate is deliberately omitted: either binding or checksum must fail closed.
    (tmp_path / "wrong.json").write_text(json.dumps(wrong_payload))
    assert wrong.read_ledger_status()["ledger_health"] == "CORRUPT"


def test_locking_concurrent_admission_lifecycle_and_status(tmp_path: Path) -> None:
    path = tmp_path / "resource_lease_ledger.json"
    clock = Clock()
    first_service = ledger(path, clock=clock)
    second_service = ledger(path, clock=clock)
    first_service.initialise_ledger()

    barrier = threading.Barrier(2)

    def admit(job: str, attempt: str):
        barrier.wait()
        return ledger(path).request_persisted_lease(request(job, attempt))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda args: admit(*args),
            [("one", "attempt-one"), ("two", "attempt-two")],
        ))
    assert sorted(row[1] for row in results) == [1, 2]
    statuses = sorted(
        (row[0].status for row in results), key=lambda row: row.value
    )
    assert statuses == [LeaseStatus.GRANTED, LeaseStatus.WAITING]
    persisted = first_service.read_ledger_status()
    assert persisted["active_reserved_ram_bytes"] == 12 * GIB
    assert persisted["waiting_requests"][0]["blocked_reasons"] == [
        "CPU_WEIGHT_CAPACITY"
    ] or persisted["waiting_requests"][0]["blocked_reasons"] == [
        "SCHEDULABLE_RAM_CAPACITY"
    ]

    active = persisted["active_leases"][0]
    waiting = persisted["waiting_requests"][0]
    revision = first_service.activate_persisted_lease(
        active["logical_identity"],
        attempt_identity=active["request"]["attempt_identity"],
        process_id=44,
        process_start_timestamp="2026-07-18T00:00:00+00:00",
    )
    clock.advance(15)
    immutable = active["logical_identity"]
    revision = first_service.heartbeat_persisted_lease(
        immutable,
        attempt_identity=active["request"]["attempt_identity"],
        process_id=44,
        process_start_timestamp="2026-07-18T00:00:00+00:00",
        phase="fit",
        expected_revision=revision,
    )
    assert first_service.read_ledger_status()["active_leases"][0][
        "logical_identity"
    ] == immutable
    first_service.release_persisted_lease(
        immutable,
        attempt_identity=active["request"]["attempt_identity"],
        reason="SUCCESS",
        expected_revision=revision,
    )
    status = second_service.read_ledger_status()
    assert status["active_leases"][0]["logical_identity"] == waiting["logical_identity"]
    assert status["waiting_requests"] == []
    assert status["ledger_health"] == "HEALTHY"

    with pytest.raises(ValueError, match="Duplicate"):
        second_service.request_persisted_lease(
            request(
                status["active_leases"][0]["request"]["job_id"],
                status["active_leases"][0]["request"]["attempt_identity"],
            )
        )

    ram_path = tmp_path / "ram_ledger.json"
    ledger(ram_path).initialise_ledger()
    ram_barrier = threading.Barrier(2)

    def admit_ram(job: str):
        ram_barrier.wait()
        return ledger(ram_path).request_persisted_lease(
            request(job, f"attempt-{job}", ram=13 * GIB, cpu=1)
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        ram_results = list(pool.map(admit_ram, ("ram-one", "ram-two")))
    assert sorted(
        (row.status for row in ram_results), key=lambda row: row.value
    ) == [LeaseStatus.GRANTED, LeaseStatus.WAITING]
    assert ledger(ram_path).read_ledger_status()["active_reserved_ram_bytes"] == 13 * GIB


def test_lock_timeout_is_explicit(tmp_path: Path) -> None:
    lock_path = tmp_path / "held.lock"
    acquired = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with exclusive_file_lock(lock_path, timeout_seconds=1):
            acquired.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(1)
    try:
        with pytest.raises(LedgerLockTimeout):
            with exclusive_file_lock(lock_path, timeout_seconds=0.05):
                pass
    finally:
        release.set()
        thread.join()


def test_failure_cancel_and_invalid_transitions_restore_capacity(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    service = ledger(path)
    first, _ = service.request_persisted_lease(request("one", "one"))
    second, _ = service.request_persisted_lease(request("two", "two"))
    assert second.status == LeaseStatus.WAITING
    service.fail_persisted_lease(
        first.logical_identity, attempt_identity="one",
        reason="could not spawn", startup_failure=True,
    )
    status = service.read_ledger_status()
    assert status["active_leases"][0]["logical_identity"] == second.logical_identity
    service.cancel_persisted_lease(
        second.logical_identity, attempt_identity="two", reason="operator"
    )
    assert service.read_ledger_status()["active_reserved_ram_bytes"] == 0
    with pytest.raises(ValueError, match="not found"):
        service.release_persisted_lease(
            second.logical_identity, attempt_identity="two", reason="again"
        )


def test_reconciliation_pid_reuse_unknown_and_manual_confirmation(tmp_path: Path) -> None:
    clock = Clock()
    processes = Processes(ProcessClassification.PROCESS_MISSING)
    path = tmp_path / "ledger.json"
    service = ledger(path, clock=clock, processes=processes)
    lease, _ = service.request_persisted_lease(request("missing", "attempt"))
    service.activate_persisted_lease(
        lease.logical_identity, attempt_identity="attempt", process_id=10,
        process_start_timestamp=clock().isoformat(),
    )
    clock.advance(30)
    fresh = service.reconcile_ledger(
        stale_threshold_seconds=60, confirmation_grace_seconds=30
    )
    assert "action" not in fresh["observations"][0]
    assert service.read_ledger_status()["active_reserved_ram_bytes"] == 12 * GIB
    processes.classification = ProcessClassification.LIVE_MATCH
    live = service.reconcile_ledger(
        stale_threshold_seconds=60, confirmation_grace_seconds=30
    )
    assert live["observations"][0]["classification"] == "LIVE_MATCH"
    processes.classification = ProcessClassification.PROCESS_MISSING
    clock.advance(90)
    first = service.reconcile_ledger(
        stale_threshold_seconds=60, confirmation_grace_seconds=30
    )
    assert first["observations"][0]["action"] == "STALE_CANDIDATE"
    assert service.read_ledger_status()["ledger_health"] == "STALE_CANDIDATES"
    clock.advance(31)
    second = service.reconcile_ledger(
        stale_threshold_seconds=60, confirmation_grace_seconds=30
    )
    assert second["observations"][0]["action"] == "STALE_CONFIRMED"
    assert service.read_ledger_status()["active_reserved_ram_bytes"] == 0

    processes.classification = ProcessClassification.ACCESS_DENIED
    unknown, _ = service.request_persisted_lease(request("unknown", "unknown"))
    service.activate_persisted_lease(
        unknown.logical_identity, attempt_identity="unknown", process_id=11,
        process_start_timestamp=clock().isoformat(),
    )
    clock.advance(120)
    service.reconcile_ledger(
        stale_threshold_seconds=60, confirmation_grace_seconds=30
    )
    status = service.read_ledger_status()
    assert status["ledger_health"] == "DEGRADED_UNKNOWN_OWNER"
    assert status["active_reserved_ram_bytes"] == 12 * GIB

    processes.classification = ProcessClassification.PID_REUSED
    service.reconcile_ledger(
        stale_threshold_seconds=60, confirmation_grace_seconds=30
    )
    assert service.read_ledger_status()["ledger_health"] == "STALE_CANDIDATES"
    revision = service.confirm_stale_release(
        unknown.logical_identity, attempt_identity="unknown",
        operator_reason="verified in Task Manager",
    )
    final = service.read_ledger_status()
    assert final["ledger_revision"] == revision
    assert final["recent_releases"][-1]["release_reason"].startswith(
        "OPERATOR_CONFIRMED"
    )
