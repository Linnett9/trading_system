from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import core.research.ml.ds24.clean_v2_resources as resources
from core.research.ml.ds24.clean_v2_resources import (
    GIB,
    LOCAL_CAPACITY_DEFERRAL,
    OPPORTUNISTIC_CAPACITY,
    PAUSE_REQUESTED_RESOURCE_PRESSURE,
    PROTECTED_CAPACITY,
    PROTECTED_CAPACITY_RECONSIDERATION,
    PROTECTED_LANE_YIELD_REQUESTED,
    RESOURCE_POLICY,
    AllocationEstimate,
    JobMemorySnapshot,
    MemorySnapshot,
    ProcessMemorySnapshot,
    PressureReadmissionGate,
    ResourceReservationLedger,
    WorkerReservationOwner,
    actual_host_pressure_reasons,
    capacity_change_is_material,
    clear_resource_pause_intent,
    constrain_stage_allocation_to_empirical_worker_envelope,
    estimate_worker_peak_allocation,
    evaluate_memory,
    family_resource_profile,
    merge_family_high_water_registry,
    panel_allocation_estimate,
    panel_slice_allocation_estimate,
    publish_resource_pause_intent,
    read_resource_pause_intent,
    recovery_policy_payload,
    service_file_reservation_requests,
)


CHECKED_AT = "2026-09-27T00:00:00+00:00"


def _configure_eight_gib_commit_guard(monkeypatch) -> None:
    policy = replace(resources.RESOURCE_POLICY, system_commit_headroom_gib=8)
    monkeypatch.setattr(resources, "RESOURCE_POLICY", policy)
    monkeypatch.setattr(
        resources,
        "PROJECTED_SYSTEM_COMMIT_GUARD_REASON",
        "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_8_GIB_GUARD",
    )
    monkeypatch.setattr(
        resources,
        "SYSTEM_COMMIT_GUARD_REASON",
        "SYSTEM_COMMIT_HEADROOM_BELOW_8_GIB_GUARD",
    )
    monkeypatch.setattr(resources, "RECOVERY_MIN_COMMIT_HEADROOM_BYTES", 8 * GIB)


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


def test_concurrent_protected_stages_cannot_overcommit_hard_commit_guards() -> None:
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
                estimate=AllocationEstimate(8 * GIB, 14 * GIB),
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
        "PROJECTED_WINDOWS_JOB_COMMIT_ABOVE_24_GIB_LIMIT"
        in rejected.blocking_reasons
        or "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_12_GIB_GUARD"
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


def test_policy_payload_declares_five_gib_fair_recovery_policy() -> None:
    policy = recovery_policy_payload()

    assert policy["maximum_dell_model_workers"] == 3
    assert policy["resource_policy_id"] == (
        "DS24_CLEAN_V2_FAIR_RECOVERY_POLICY_V5"
    )
    assert policy["dell_canary_active_worker_limit"] == 3
    assert policy["maximum_mac_model_workers"] == 1
    assert policy["windows_job_aggregate_commit_limit_gib"] == 24
    assert policy["automatic_retry_allowed"] is True
    assert policy["system_commit_headroom_gib"] == 5
    assert policy["previous_resource_policy"]["system_commit_headroom_gib"] == 12
    assert policy["fair_turn_wait_seconds"] == 60
    assert policy["automatic_pressure_readmission_allowed"] is True
    assert policy["previous_resource_policy"]["status"] == (
        "HISTORICAL_REPRODUCIBILITY_AUTHORITY"
    )


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
        estimate=AllocationEstimate(10 * GIB, 20 * GIB),
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
    lower_fresh_sample = estimate_worker_peak_allocation(
        "random_forest", AllocationEstimate(GIB, GIB)
    ).guarded()
    assert lower_fresh_sample.physical_bytes == guarded.physical_bytes
    assert lower_fresh_sample.commit_bytes == guarded.commit_bytes


def test_light_family_empirical_profiles_cover_full_worker_high_water() -> None:
    expected = {
        "huber": (4_467_032_064, 4_599_554_048, 39, 12_857),
        "ridge_C5": (5_167_022_080, 5_268_992_000, 7, 2_363),
        "elastic_net_C5": (4_768_890_880, 4_841_054_208, 17, 5_610),
        "elastic_net_C6": (4_768_890_880, 4_841_054_208, 17, 5_610),
    }

    for family, (resident, commit, refits, scores) in expected.items():
        profile = family_resource_profile(family)
        guarded = estimate_worker_peak_allocation(family).guarded()

        assert profile.measured_high_water == AllocationEstimate(resident, commit)
        assert profile.high_water_cushion == AllocationEstimate(GIB, GIB)
        assert profile.observed_refits == refits
        assert profile.observed_score_timestamps == scores
        assert profile.full_worker_high_water_authoritative is True
        assert guarded.physical_bytes >= resident + GIB
        assert guarded.commit_bytes >= commit + GIB
        assert guarded.physical_bytes <= resident + GIB + 1
        assert guarded.commit_bytes <= commit + GIB + 1

    assert family_resource_profile("elastic_net_C6").bounded_training_rows == 8_325
    assert "SIBLING_BOUND" in family_resource_profile("elastic_net_C6").profile_id


def test_demonstrable_stage_demand_is_not_capped_by_stale_worker_peak() -> None:
    generic_panel = AllocationEstimate(4_817_208_000, 5_539_789_200)
    assert constrain_stage_allocation_to_empirical_worker_envelope(
        "huber", generic_panel
    ) == generic_panel
    assert constrain_stage_allocation_to_empirical_worker_envelope(
        "unprofiled_tabular_family", generic_panel
    ) == generic_panel


def test_six_gib_admission_floor_and_four_gib_actual_pressure_are_distinct() -> None:
    admitted = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=6.5, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    ).try_admit(
        WorkerReservationOwner("huber", 18), AllocationEstimate(0, 0)
    )
    admission_wait = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=5.5, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    ).try_admit(
        WorkerReservationOwner("huber", 18), AllocationEstimate(0, 0)
    )

    soft_pressure = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=3.8, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    ).try_admit(
        WorkerReservationOwner("huber", 18), AllocationEstimate(0, 0)
    )

    assert admitted.granted is True
    assert admission_wait.classification == LOCAL_CAPACITY_DEFERRAL
    assert soft_pressure.classification == "HOST_RESOURCE_PRESSURE"


def _ledger_with_two_protected_workers(
    *, available_gib: float = 8, commit_headroom_gib: float = 40,
    job_committed_gib: int = 0,
) -> ResourceReservationLedger:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=available_gib,
            commit_headroom_gib=commit_headroom_gib,
        ),
        job_snapshot_provider=lambda: _job(committed_gib=job_committed_gib),
        process_tree_snapshot_provider=_tree,
    )
    for index, family in enumerate(("random_forest", "momentum")):
        admitted = ledger.try_admit(
            WorkerReservationOwner(family, 20),
            AllocationEstimate(GIB, GIB),
            capacity_class=PROTECTED_CAPACITY,
        )
        assert admitted.worker_reservation_id
        ledger.bind_worker(
            admitted.worker_reservation_id,
            WorkerReservationOwner(
                family,
                20,
                pid=50_000 + index,
                process_creation_time_utc=f"2026-09-30T00:00:0{index}+00:00",
            ),
        )
    return ledger


def test_empirical_huber_third_lane_uses_five_gib_commit_guard() -> None:
    ledger = _ledger_with_two_protected_workers(
        available_gib=9.5,
        commit_headroom_gib=17,
        job_committed_gib=12,
    )
    estimate = estimate_worker_peak_allocation("huber")

    decision = ledger.try_admit(
        WorkerReservationOwner("huber", 20),
        estimate,
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )

    assert RESOURCE_POLICY.system_commit_headroom_gib == 5
    assert decision.granted is True
    assert decision.projected_system_commit_headroom_bytes is not None
    assert decision.projected_system_commit_headroom_bytes >= 5 * GIB
    assert decision.projected_job_commit_bytes is not None
    assert decision.projected_job_commit_bytes <= 24 * GIB


def test_authorized_eight_gib_policy_would_admit_empirical_huber(
    monkeypatch,
) -> None:
    _configure_eight_gib_commit_guard(monkeypatch)
    ledger = _ledger_with_two_protected_workers(
        available_gib=9.5,
        commit_headroom_gib=17,
        job_committed_gib=12,
    )

    decision = ledger.try_admit(
        WorkerReservationOwner("huber", 20),
        estimate_worker_peak_allocation("huber"),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )

    assert resources.RESOURCE_POLICY.system_commit_headroom_gib == 8
    assert decision.granted is True
    assert decision.projected_system_commit_headroom_bytes is not None
    assert decision.projected_system_commit_headroom_bytes >= 8 * GIB
    assert decision.projected_job_commit_bytes is not None
    assert decision.projected_job_commit_bytes <= 24 * GIB


def test_third_worker_may_overbook_only_physical_projection() -> None:
    protected = _ledger_with_two_protected_workers()
    protected_denial = protected.try_admit(
        WorkerReservationOwner("ridge_C5", 20),
        AllocationEstimate(4 * GIB, 4 * GIB),
        capacity_class=PROTECTED_CAPACITY,
    )
    assert not protected_denial.granted
    assert "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE" in (
        protected_denial.blocking_reasons
    )

    opportunistic = _ledger_with_two_protected_workers()
    admitted = opportunistic.try_admit(
        WorkerReservationOwner("ridge_C5", 20),
        AllocationEstimate(4 * GIB, 4 * GIB),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )
    assert admitted.granted is True
    assert admitted.capacity_class == OPPORTUNISTIC_CAPACITY
    assert admitted.projected_available_physical_bytes is not None
    assert admitted.projected_available_physical_bytes < 5 * GIB
    assert opportunistic.status_payload()["active_worker_reservations"] == 3


def test_opportunistic_lane_preserves_job_and_system_commit_guards() -> None:
    job_blocked = _ledger_with_two_protected_workers(job_committed_gib=20)
    job_decision = job_blocked.try_admit(
        WorkerReservationOwner("ridge_C5", 20),
        AllocationEstimate(4 * GIB, 4 * GIB),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )
    assert not job_decision.granted
    assert "PROJECTED_WINDOWS_JOB_COMMIT_ABOVE_24_GIB_LIMIT" in (
        job_decision.blocking_reasons
    )

    commit_blocked = _ledger_with_two_protected_workers(commit_headroom_gib=12)
    commit_decision = commit_blocked.try_admit(
        WorkerReservationOwner("ridge_C5", 20),
        AllocationEstimate(4 * GIB, 4 * GIB),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )
    assert not commit_decision.granted
    assert "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_5_GIB_GUARD" in (
        commit_decision.blocking_reasons
    )


def test_eight_gib_system_commit_guard_exact_boundary_and_below(monkeypatch) -> None:
    _configure_eight_gib_commit_guard(monkeypatch)
    exact = evaluate_memory(
        stage="commit-boundary",
        estimated_physical_bytes=0,
        estimated_commit_bytes=2 * GIB,
        snapshot=_memory(available_gib=20, commit_headroom_gib=11),
    )
    below = evaluate_memory(
        stage="commit-below-boundary",
        estimated_physical_bytes=0,
        estimated_commit_bytes=2 * GIB,
        snapshot=_memory(
            available_gib=20,
            commit_headroom_gib=11 - (1 / GIB),
        ),
    )

    assert exact.projected_commit_headroom_bytes == 8 * GIB
    assert exact.safe is True
    assert below.projected_commit_headroom_bytes == 8 * GIB - 1
    assert below.safe is False
    assert below.blocking_reasons == (
        "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_8_GIB_GUARD",
    )


def test_opportunistic_reservation_does_not_displace_protected_stage_headroom() -> None:
    ledger = _ledger_with_two_protected_workers(available_gib=16)
    opportunistic_owner = WorkerReservationOwner("ridge_C5", 20)
    opportunistic = ledger.try_admit(
        opportunistic_owner,
        AllocationEstimate(4 * GIB, 4 * GIB),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )
    assert opportunistic.worker_reservation_id
    ledger.bind_worker(
        opportunistic.worker_reservation_id,
        WorkerReservationOwner(
            "ridge_C5",
            20,
            pid=50_002,
            process_creation_time_utc="2026-09-30T00:00:02+00:00",
        ),
    )

    protected_stage = ledger.request_stage(
        WorkerReservationOwner(
            "random_forest",
            20,
            pid=50_000,
            process_creation_time_utc="2026-09-30T00:00:00+00:00",
        ),
        stage="random_forest_panel_assembly",
        estimate=AllocationEstimate(6 * GIB, 6 * GIB),
    )

    assert protected_stage.granted is True
    assert protected_stage.capacity_class == PROTECTED_CAPACITY
    assert protected_stage.projected_available_physical_bytes == int(5.5 * GIB)

    assert ledger.release_worker(
        WorkerReservationOwner(
            "momentum",
            20,
            pid=50_001,
            process_creation_time_utc="2026-09-30T00:00:01+00:00",
        )
    )
    second_opportunistic = ledger.try_admit(
        WorkerReservationOwner("huber", 20),
        AllocationEstimate(GIB, GIB),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )
    assert second_opportunistic.granted is False
    assert second_opportunistic.classification == (
        "OPPORTUNISTIC_LANE_ALREADY_OCCUPIED"
    )


def test_pressure_threshold_sequence_and_readmission_hysteresis() -> None:
    observed_levels = [
        evaluate_memory(
            stage="pressure-sequence",
            estimated_allocation_bytes=0,
            snapshot=_memory(available_gib=value, commit_headroom_gib=40),
            purpose="observation",
        )
        for value in (8, 5, 3.8, 2.8, 5, 7.5)
    ]
    assert [decision.pressure_level for decision in observed_levels] == [
        "NORMAL",
        "ADMISSION_BLOCKED",
        "SOFT_PRESSURE",
        "EMERGENCY",
        "ADMISSION_BLOCKED",
        "NORMAL",
    ]
    assert [decision.safe for decision in observed_levels] == [
        True,
        True,
        False,
        False,
        True,
        True,
    ]

    gate = PressureReadmissionGate(paused_at_monotonic=0.0)
    assert not gate.observe(available_physical_bytes=5 * GIB, now=60.0)
    assert not gate.observe(available_physical_bytes=int(7.5 * GIB), now=61.0)
    assert not gate.observe(available_physical_bytes=int(7.5 * GIB), now=90.9)
    assert gate.observe(available_physical_bytes=int(7.5 * GIB), now=91.0)


def test_pause_intent_is_exact_identity_bound_and_clear_is_safe(
    tmp_path: Path,
) -> None:
    owner = WorkerReservationOwner(
        "ridge_C5",
        21,
        pid=60_001,
        process_creation_time_utc="2026-09-30T00:00:00+00:00",
    )
    intent = publish_resource_pause_intent(
        tmp_path,
        owner=owner,
        classification=PAUSE_REQUESTED_RESOURCE_PRESSURE,
        emergency=False,
        resource_decision={"pressure_level": "SOFT_PRESSURE"},
    )
    assert read_resource_pause_intent(tmp_path, owner=owner) == intent
    wrong_identity = WorkerReservationOwner(
        owner.family,
        owner.attempt_generation,
        pid=owner.pid + 1,
        process_creation_time_utc=owner.process_creation_time_utc,
    )
    assert read_resource_pause_intent(tmp_path, owner=wrong_identity) is None
    assert clear_resource_pause_intent(tmp_path, owner=wrong_identity) is False
    assert clear_resource_pause_intent(tmp_path, owner=owner) is True


def test_family_high_water_never_regresses_and_retains_stage_evidence() -> None:
    first = merge_family_high_water_registry(
        {},
        {
            "workers": [
                {
                    "owner": {"family": "ridge_C5"},
                    "capacity_class": OPPORTUNISTIC_CAPACITY,
                    "current_memory": {"resident_bytes": GIB},
                    "peak_resident_bytes": 2 * GIB,
                    "peak_private_commit_bytes": 3 * GIB,
                    "active_stage": "ridge_C5_fit",
                    "stage_baseline_memory": {"resident_bytes": GIB // 2},
                    "guarded_stage": {"physical_bytes": GIB},
                    "outstanding": {"physical_bytes": GIB},
                }
            ]
        },
        recorded_at_utc=CHECKED_AT,
    )
    merged = merge_family_high_water_registry(
        first,
        {
            "workers": [
                {
                    "owner": {"family": "ridge_C5"},
                    "capacity_class": OPPORTUNISTIC_CAPACITY,
                    "current_memory": {"resident_bytes": GIB // 2},
                    "peak_resident_bytes": GIB,
                    "peak_private_commit_bytes": GIB,
                    "active_stage": "ridge_C5_score_matrix",
                }
            ]
        },
        recorded_at_utc=CHECKED_AT,
    )
    row = merged["families"]["ridge_C5"]
    assert row["peak_resident_bytes"] == 2 * GIB
    assert row["peak_private_commit_bytes"] == 3 * GIB
    assert row["active_stage"] == "ridge_C5_score_matrix"


def _protected_rf_stage_ledger() -> tuple[
    ResourceReservationLedger,
    WorkerReservationOwner,
    dict[str, object],
]:
    snapshots: dict[str, object] = {
        "memory": _memory(available_gib=20, commit_headroom_gib=40),
        "job": _job(),
    }
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: snapshots["memory"],  # type: ignore[return-value]
        job_snapshot_provider=lambda: snapshots["job"],  # type: ignore[return-value]
        process_tree_snapshot_provider=_tree,
    )
    unbound = WorkerReservationOwner("random_forest", 30)
    admission = ledger.try_admit(
        unbound,
        AllocationEstimate(
            physical_bytes=3_393_734_028,
            commit_bytes=3_393_734_028,
        ),
        capacity_class=PROTECTED_CAPACITY,
    )
    assert admission.worker_reservation_id
    owner = WorkerReservationOwner(
        "random_forest",
        30,
        pid=70_001,
        process_creation_time_utc="2026-09-30T18:20:00+00:00",
    )
    ledger.bind_worker(admission.worker_reservation_id, owner)
    return ledger, owner, snapshots


def test_a_protected_rf_stage_staggers_when_physical_projection_is_unsafe() -> None:
    ledger, owner, snapshots = _protected_rf_stage_ledger()
    snapshots["memory"] = _memory(available_gib=6.59, commit_headroom_gib=40)

    decision = ledger.request_stage(
        owner,
        stage="random_forest_score_matrix",
        estimate=AllocationEstimate(131_553_720, 131_553_720),
    )

    assert decision.granted is False
    assert decision.capacity_class == PROTECTED_CAPACITY
    assert decision.projected_available_physical_bytes is not None
    assert decision.projected_available_physical_bytes < 5 * GIB
    assert decision.guarded_requested.physical_bytes == 197_330_580
    assert "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE" in (
        decision.blocking_reasons
    )


def test_b_protected_rf_soft_pressure_remains_cooperative(tmp_path: Path) -> None:
    snapshot = _memory(available_gib=3.8, commit_headroom_gib=40)
    reasons = actual_host_pressure_reasons(snapshot, _job())
    owner = WorkerReservationOwner(
        "random_forest",
        31,
        pid=70_002,
        process_creation_time_utc="2026-09-30T18:21:00+00:00",
    )
    intent = publish_resource_pause_intent(
        tmp_path,
        owner=owner,
        classification=PAUSE_REQUESTED_RESOURCE_PRESSURE,
        emergency=False,
        resource_decision={"blocking_reasons": list(reasons)},
    )

    assert reasons == ("AVAILABLE_PHYSICAL_MEMORY_BELOW_4_GIB_SOFT_PAUSE",)
    assert intent.emergency is False
    assert intent.owner.family == "random_forest"


def test_c_protected_rf_emergency_semantics_remain_unchanged() -> None:
    decision = evaluate_memory(
        stage="random_forest_runtime",
        estimated_allocation_bytes=0,
        snapshot=_memory(available_gib=2.8, commit_headroom_gib=40),
        purpose="observation",
    )

    assert decision.safe is False
    assert decision.emergency is True
    assert decision.blocking_reasons == (
        "AVAILABLE_PHYSICAL_MEMORY_BELOW_3_GIB_EMERGENCY",
    )


def test_d_protected_rf_reconsideration_reclaims_free_slot() -> None:
    ledger = ResourceReservationLedger(
        maximum_workers=3,
        memory_snapshot_provider=lambda: _memory(
            available_gib=12, commit_headroom_gib=40
        ),
        job_snapshot_provider=_job,
        process_tree_snapshot_provider=_tree,
    )
    owners: dict[str, WorkerReservationOwner] = {}
    for index, family in enumerate(("random_forest", "momentum")):
        unbound = WorkerReservationOwner(family, 32)
        admitted = ledger.try_admit(
            unbound,
            AllocationEstimate(GIB, GIB),
            capacity_class=PROTECTED_CAPACITY,
        )
        assert admitted.worker_reservation_id
        owner = WorkerReservationOwner(
            family,
            32,
            pid=71_000 + index,
            process_creation_time_utc=f"2026-09-30T18:22:0{index}+00:00",
        )
        ledger.bind_worker(admitted.worker_reservation_id, owner)
        owners[family] = owner
    elastic_unbound = WorkerReservationOwner("elastic_net_C5", 32)
    elastic = ledger.try_admit(
        elastic_unbound,
        AllocationEstimate(GIB, GIB),
        capacity_class=OPPORTUNISTIC_CAPACITY,
    )
    assert elastic.worker_reservation_id
    ledger.bind_worker(
        elastic.worker_reservation_id,
        WorkerReservationOwner(
            "elastic_net_C5",
            32,
            pid=71_002,
            process_creation_time_utc="2026-09-30T18:22:02+00:00",
        ),
    )
    assert ledger.release_worker(owners["random_forest"])

    reconsidered = ledger.try_admit(
        WorkerReservationOwner("random_forest", 32),
        AllocationEstimate(4 * GIB, 4 * GIB),
        capacity_class=PROTECTED_CAPACITY,
        protected_reconsideration=True,
    )

    assert reconsidered.granted is True
    assert reconsidered.classification == PROTECTED_CAPACITY_RECONSIDERATION
    assert reconsidered.capacity_class == PROTECTED_CAPACITY


def test_e_lane_yield_intent_is_identity_bound_and_non_emergency(
    tmp_path: Path,
) -> None:
    owner = WorkerReservationOwner(
        "elastic_net_C5",
        33,
        pid=72_001,
        process_creation_time_utc="2026-09-30T18:23:00+00:00",
    )
    intent = publish_resource_pause_intent(
        tmp_path,
        owner=owner,
        classification=PROTECTED_LANE_YIELD_REQUESTED,
        emergency=False,
        resource_decision={
            "classification": PROTECTED_CAPACITY_RECONSIDERATION,
            "protected_family": "random_forest",
        },
    )

    assert read_resource_pause_intent(tmp_path, owner=owner) == intent
    assert intent.classification == PROTECTED_LANE_YIELD_REQUESTED
    assert intent.emergency is False


def test_f_protected_rf_stage_does_not_bypass_job_cap() -> None:
    ledger, owner, snapshots = _protected_rf_stage_ledger()
    snapshots["memory"] = _memory(available_gib=12, commit_headroom_gib=40)
    snapshots["job"] = _job(committed_gib=20)

    decision = ledger.request_stage(
        owner,
        stage="random_forest_score_matrix",
        estimate=AllocationEstimate(131_553_720, 131_553_720),
    )

    assert decision.granted is False
    assert "PROJECTED_WINDOWS_JOB_COMMIT_ABOVE_24_GIB_LIMIT" in (
        decision.blocking_reasons
    )


def test_g_protected_rf_stage_does_not_bypass_system_commit_guard() -> None:
    ledger, owner, snapshots = _protected_rf_stage_ledger()
    snapshots["memory"] = _memory(available_gib=12, commit_headroom_gib=5)

    decision = ledger.request_stage(
        owner,
        stage="random_forest_score_matrix",
        estimate=AllocationEstimate(131_553_720, 131_553_720),
    )

    assert decision.granted is False
    assert "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_5_GIB_GUARD" in (
        decision.blocking_reasons
    )
