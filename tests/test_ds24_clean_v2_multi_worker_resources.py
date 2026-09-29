from __future__ import annotations

import json
import threading
from pathlib import Path

from core.research.ml.ds24.clean_v2_resources import (
    GIB,
    LOCAL_CAPACITY_DEFERRAL,
    RESOURCE_POLICY,
    AllocationEstimate,
    JobMemorySnapshot,
    MemorySnapshot,
    ProcessMemorySnapshot,
    ResourceReservationLedger,
    WorkerReservationOwner,
    capacity_change_is_material,
    estimate_worker_peak_allocation,
    evaluate_memory,
    family_resource_profile,
    panel_allocation_estimate,
    panel_slice_allocation_estimate,
    recovery_policy_payload,
    service_file_reservation_requests,
)


CHECKED_AT = "2026-09-27T00:00:00+00:00"


def _memory(
    *, available_gib: float = 40, commit_headroom_gib: float = 40
) -> MemorySnapshot:
    return MemorySnapshot(
        available_physical_bytes=int(available_gib * GIB),
        total_physical_bytes=64 * GIB,
        commit_headroom_bytes=int(commit_headroom_gib * GIB),
        commit_limit_bytes=80 * GIB,
        committed_bytes=int((80 - commit_headroom_gib) * GIB),
        source="bounded-test",
        checked_at_utc=CHECKED_AT,
    )


def _job(*, committed_gib: int = 0) -> JobMemorySnapshot:
    return JobMemorySnapshot(
        committed_bytes=committed_gib * GIB,
        peak_committed_bytes=committed_gib * GIB,
        commit_limit_bytes=24 * GIB,
        active_processes=0,
        total_processes=0,
        terminated_processes=0,
        limit_violation_detected=False,
        installation_verified=True,
        checked_at_utc=CHECKED_AT,
    )


def _tree(pid: int) -> ProcessMemorySnapshot:
    return ProcessMemorySnapshot(
        pid=pid,
        resident_bytes=100 * 1024**2,
        peak_resident_bytes=100 * 1024**2,
        private_commit_bytes=120 * 1024**2,
        peak_private_commit_bytes=120 * 1024**2,
        source="bounded-test-tree",
        checked_at_utc=CHECKED_AT,
    )


def test_separate_physical_and_commit_estimates_preserve_both_reserves() -> None:
    decision = evaluate_memory(
        stage="separate-accounting",
        estimated_physical_bytes=2 * GIB,
        estimated_commit_bytes=4 * GIB,
        snapshot=_memory(available_gib=8, commit_headroom_gib=18),
        purpose="allocation",
    )

    assert decision.guarded_physical_bytes == 3 * GIB
    assert decision.guarded_commit_bytes == 6 * GIB
    assert decision.projected_available_physical_bytes == 5 * GIB
    assert decision.projected_commit_headroom_bytes == 12 * GIB
    assert decision.safe is True


def test_three_workers_can_be_reserved_and_fourth_is_rejected_by_slot_limit() -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=_memory,
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    estimate = AllocationEstimate(GIB // 2, GIB // 2)

    decisions = [
        ledger.try_admit(WorkerReservationOwner(f"family_{index}", 7), estimate)
        for index in range(4)
    ]

    assert [decision.granted for decision in decisions] == [True, True, True, False]
    assert decisions[3].classification == "WORKER_SLOT_LIMIT_REACHED"
    status = ledger.status_payload()
    assert status["active_worker_reservations"] == 3
    assert status["outstanding"] == {
        "physical_bytes": 9 * GIB // 4,
        "commit_bytes": 9 * GIB // 4,
    }


def test_concurrent_stage_reservations_cannot_overcommit_physical_reserve() -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=20, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    owners: list[WorkerReservationOwner] = []
    for index in range(2):
        unbound = WorkerReservationOwner(f"family_{index}", 9)
        admission = ledger.try_admit(
            unbound, AllocationEstimate(GIB, GIB)
        )
        assert admission.granted and admission.worker_reservation_id
        owner = WorkerReservationOwner(
            f"family_{index}",
            9,
            pid=10_000 + index,
            process_creation_time_utc=f"2026-09-27T00:00:0{index}+00:00",
        )
        ledger.bind_worker(admission.worker_reservation_id, owner)
        owners.append(owner)

    barrier = threading.Barrier(3)
    results = []

    def request(owner: WorkerReservationOwner) -> None:
        barrier.wait(timeout=5)
        results.append(
            ledger.request_stage(
                owner,
                stage="fit",
                estimate=AllocationEstimate(8 * GIB, 2 * GIB),
            )
        )

    threads = [threading.Thread(target=request, args=(owner,)) for owner in owners]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert sum(result.granted for result in results) == 1
    rejected = next(result for result in results if not result.granted)
    assert (
        "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE"
        in rejected.blocking_reasons
    )


def test_release_is_identity_aware_and_preserves_other_worker_budget() -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=_memory,
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    unbound = WorkerReservationOwner("random_forest", 11)
    decision = ledger.try_admit(unbound, AllocationEstimate(GIB, 2 * GIB))
    assert decision.worker_reservation_id
    owner = WorkerReservationOwner(
        "random_forest",
        11,
        pid=12_345,
        process_creation_time_utc="2026-09-27T00:00:00+00:00",
    )
    ledger.bind_worker(decision.worker_reservation_id, owner)

    stale = WorkerReservationOwner(
        "random_forest",
        11,
        pid=12_345,
        process_creation_time_utc="2026-09-27T00:01:00+00:00",
    )
    assert ledger.release_worker(stale) is False
    assert ledger.status_payload()["active_worker_reservations"] == 1
    assert ledger.release_worker(owner) is True
    assert ledger.status_payload()["active_worker_reservations"] == 0


def test_file_reservation_request_propagates_policy_and_owner(
    tmp_path: Path,
) -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=_memory,
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    unbound = WorkerReservationOwner("huber", 12)
    admission = ledger.try_admit(unbound, AllocationEstimate(GIB, GIB))
    assert admission.worker_reservation_id
    owner = WorkerReservationOwner(
        "huber",
        12,
        pid=54_321,
        process_creation_time_utc="2026-09-27T00:00:00+00:00",
    )
    ledger.bind_worker(admission.worker_reservation_id, owner)
    request_path = tmp_path / "requests" / "request-1.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(
            {
                "request_id": "request-1",
                "resource_policy_id": RESOURCE_POLICY.policy_id,
                "action": "reserve",
                "stage": "huber_fit",
                "owner": owner.payload(),
                "estimate": AllocationEstimate(GIB, GIB).payload(),
            }
        ),
        encoding="utf-8",
    )

    handled = service_file_reservation_requests(tmp_path, ledger)
    response = json.loads(
        (tmp_path / "responses" / "request-1.json").read_text(encoding="utf-8")
    )

    assert handled[0]["granted"] is True
    assert response["resource_policy"]["resource_policy_id"] == (
        RESOURCE_POLICY.policy_id
    )
    assert response["worker_reservation_id"] == admission.worker_reservation_id


def test_policy_payload_declares_dell_three_mac_one_and_no_retry() -> None:
    policy = recovery_policy_payload()

    assert policy["maximum_dell_model_workers"] == 3
    assert policy["dell_canary_active_worker_limit"] == 2
    assert policy["maximum_mac_model_workers"] == 1
    assert policy["windows_job_aggregate_commit_limit_gib"] == 24
    assert policy["automatic_retry_allowed"] is False


def test_panel_slice_estimate_excludes_existing_source_panel_high_water() -> None:
    assembly = panel_allocation_estimate(row_count=1_000, column_count=70)
    slicing = panel_slice_allocation_estimate(row_count=1_000, column_count=70)

    assert slicing.physical_bytes == 1_000 * 70 * 8 * 3 + 1_000 * 256
    assert slicing.commit_bytes >= slicing.physical_bytes
    assert slicing.physical_bytes < assembly.physical_bytes


def test_stage_denial_is_local_deferral_and_other_worker_keeps_budget() -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=20, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    owners = []
    for index, family in enumerate(("random_forest", "momentum")):
        unbound = WorkerReservationOwner(family, 14)
        admitted = ledger.try_admit(unbound, AllocationEstimate(GIB, GIB))
        assert admitted.worker_reservation_id
        owner = WorkerReservationOwner(
            family,
            14,
            pid=20_000 + index,
            process_creation_time_utc=f"2026-09-27T00:00:0{index}+00:00",
        )
        ledger.bind_worker(admitted.worker_reservation_id, owner)
        owners.append(owner)

    denied = ledger.request_stage(
        owners[1],
        stage="momentum_panel_assembly",
        estimate=AllocationEstimate(10 * GIB, 2 * GIB),
    )

    assert denied.granted is False
    assert denied.classification == LOCAL_CAPACITY_DEFERRAL
    assert ledger.release_worker(owners[1]) is True
    status = ledger.status_payload()
    assert status["active_worker_reservations"] == 1
    assert status["workers"][0]["owner"]["family"] == "random_forest"


def test_deferred_family_requires_cooldown_and_material_capacity_release() -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=_memory,
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    unbound = WorkerReservationOwner("random_forest", 15)
    admitted = ledger.try_admit(unbound, AllocationEstimate(2 * GIB, 2 * GIB))
    assert admitted.worker_reservation_id
    owner = WorkerReservationOwner(
        "random_forest",
        15,
        pid=30_000,
        process_creation_time_utc="2026-09-27T00:00:00+00:00",
    )
    ledger.bind_worker(admitted.worker_reservation_id, owner)
    before = ledger.capacity_snapshot()

    assert not capacity_change_is_material(before, before, elapsed_seconds=10_000)
    assert ledger.release_worker(owner)
    after = ledger.capacity_snapshot()
    assert not capacity_change_is_material(before, after, elapsed_seconds=0)
    assert capacity_change_is_material(before, after, elapsed_seconds=30)

    next_family = ledger.try_admit(
        WorkerReservationOwner("huber", 15), AllocationEstimate(GIB, GIB)
    )
    assert next_family.granted is True


def test_stage_high_water_replaces_worker_peak_without_double_counting() -> None:
    baseline = ProcessMemorySnapshot(
        pid=40_000,
        resident_bytes=GIB,
        peak_resident_bytes=GIB,
        private_commit_bytes=GIB,
        peak_private_commit_bytes=GIB,
        source="baseline",
        checked_at_utc=CHECKED_AT,
    )
    current = ProcessMemorySnapshot(
        pid=40_000,
        resident_bytes=3 * GIB,
        peak_resident_bytes=3 * GIB,
        private_commit_bytes=3 * GIB,
        peak_private_commit_bytes=3 * GIB,
        source="current",
        checked_at_utc=CHECKED_AT,
    )
    snapshots = iter((baseline, current, current, current))
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=_memory,
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=lambda _pid: next(snapshots),
    )
    unbound = WorkerReservationOwner("random_forest", 16)
    admitted = ledger.try_admit(unbound, AllocationEstimate(4 * GIB, 4 * GIB))
    assert admitted.worker_reservation_id
    owner = WorkerReservationOwner(
        "random_forest",
        16,
        pid=40_000,
        process_creation_time_utc="2026-09-27T00:00:00+00:00",
    )
    ledger.bind_worker(admitted.worker_reservation_id, owner)

    stage = ledger.request_stage(
        owner,
        stage="panel_assembly",
        estimate=AllocationEstimate(3 * GIB, 3 * GIB),
    )

    # Guarded worker peak is 6 GiB; 2 GiB measured growth leaves 4 GiB.
    # The guarded 4.5 GiB stage replaces that remainder, rather than adding to it.
    assert stage.granted is True
    assert stage.outstanding.physical_bytes == 9 * GIB // 2
    status = ledger.status_payload()
    assert status["outstanding"]["physical_bytes"] == 9 * GIB // 2
    assert "HIGH_WATER" in status["accounting_semantics"]


def test_active_stage_reservation_shrinks_as_stage_allocation_is_observed() -> None:
    baseline = ProcessMemorySnapshot(
        pid=41_000,
        resident_bytes=GIB,
        peak_resident_bytes=GIB,
        private_commit_bytes=GIB,
        peak_private_commit_bytes=GIB,
        source="baseline",
        checked_at_utc=CHECKED_AT,
    )
    current = {"snapshot": baseline}
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=_memory,
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=lambda _pid: current["snapshot"],
    )
    unbound = WorkerReservationOwner("random_forest", 17)
    admitted = ledger.try_admit(unbound, AllocationEstimate(4 * GIB, 4 * GIB))
    assert admitted.worker_reservation_id
    owner = WorkerReservationOwner(
        "random_forest",
        17,
        pid=41_000,
        process_creation_time_utc="2026-09-27T00:00:00+00:00",
    )
    ledger.bind_worker(admitted.worker_reservation_id, owner)
    stage = ledger.request_stage(
        owner,
        stage="panel_assembly",
        estimate=AllocationEstimate(4 * GIB, 4 * GIB),
    )
    assert stage.granted is True

    current["snapshot"] = ProcessMemorySnapshot(
        pid=41_000,
        resident_bytes=3 * GIB,
        peak_resident_bytes=3 * GIB,
        private_commit_bytes=3 * GIB,
        peak_private_commit_bytes=3 * GIB,
        source="stage-growth",
        checked_at_utc=CHECKED_AT,
    )
    status = ledger.status_payload()

    # Both the worker envelope and the active stage have consumed 2 GiB of
    # their guarded 6 GiB high-water.  The ledger retains 4 GiB, not 6 GiB on
    # top of the memory already present in host and Job counters.
    assert status["outstanding"] == {
        "physical_bytes": 4 * GIB,
        "commit_bytes": 4 * GIB,
    }
    assert status["workers"][0]["stage_baseline_memory"]["resident_bytes"] == GIB


def test_random_forest_empirical_profile_retains_cushion_without_old_envelope() -> None:
    profile = family_resource_profile("random_forest")
    requested = estimate_worker_peak_allocation("random_forest")
    guarded = requested.guarded()

    assert profile.measured_high_water is not None
    assert guarded.physical_bytes >= (
        profile.measured_high_water.physical_bytes
        + profile.high_water_cushion.physical_bytes
    )
    assert guarded.commit_bytes >= (
        profile.measured_high_water.commit_bytes
        + profile.high_water_cushion.commit_bytes
    )
    assert guarded.physical_bytes < 9 * GIB
    assert guarded.commit_bytes < 12 * GIB
    assert family_resource_profile("equal_weight_no_model").fit_required is False


def test_readmission_floor_wait_is_capacity_only_but_warning_is_host_pressure() -> None:
    readmission_wait = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=6.5, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    ).try_admit(
        WorkerReservationOwner("huber", 18), AllocationEstimate(0, 0)
    )
    warning_pressure = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=5.5, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    ).try_admit(
        WorkerReservationOwner("huber", 18), AllocationEstimate(0, 0)
    )

    assert readmission_wait.classification == LOCAL_CAPACITY_DEFERRAL
    assert warning_pressure.classification == "HOST_RESOURCE_PRESSURE"
