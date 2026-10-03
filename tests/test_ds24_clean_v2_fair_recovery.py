from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.research.ml.ds24.clean_v2_resources import (
    GIB,
    PROJECTED_SYSTEM_COMMIT_GUARD_REASON,
    RESOURCE_POLICY,
    evaluate_admission_blockers,
    recovery_policy_payload,
)
from scripts.local.ds24_clean_v2_supervisor import (
    CONCURRENT_RECOVERY_MODE,
    SERIAL_RECOVERY_MODE,
    FairRecoveryState,
    _serial_recovery_launch_blocker,
)


def test_five_gib_commit_boundary_is_exact_and_historical_projection_passes() -> None:
    exact = evaluate_admission_blockers(
        available_physical_bytes=20 * GIB,
        projected_available_physical_bytes=10 * GIB,
        projected_commit_headroom_bytes=5 * GIB,
    )
    below = evaluate_admission_blockers(
        available_physical_bytes=20 * GIB,
        projected_available_physical_bytes=10 * GIB,
        projected_commit_headroom_bytes=5 * GIB - 1,
    )
    historical = evaluate_admission_blockers(
        available_physical_bytes=8_272_076_800,
        projected_available_physical_bytes=-3_448_353_824,
        projected_commit_headroom_bytes=12_436_490_664,
    )

    assert PROJECTED_SYSTEM_COMMIT_GUARD_REASON not in exact
    assert below == (PROJECTED_SYSTEM_COMMIT_GUARD_REASON,)
    assert PROJECTED_SYSTEM_COMMIT_GUARD_REASON not in historical
    assert historical == (
        "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE",
    )


def test_effective_policy_keeps_unrelated_resource_guards() -> None:
    policy = recovery_policy_payload()

    assert RESOURCE_POLICY.system_commit_headroom_gib == 5
    assert policy["windows_job_aggregate_commit_limit_gib"] == 24
    assert policy["soft_pause_available_physical_gib"] == 4
    assert policy["emergency_available_physical_gib"] == 3
    assert policy["admission_available_physical_gib"] == 6
    assert policy["readmission_available_physical_gib"] == 7
    assert policy["fair_turn_wait_seconds"] == 60
    assert policy["protected_reconsideration_seconds"] == 30


def test_capacity_denial_becomes_a_durable_fair_serial_turn() -> None:
    now = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)
    state = FairRecoveryState(run_mode=CONCURRENT_RECOVERY_MODE)
    decision = {
        "classification": "LOCAL_CAPACITY_DEFERRAL",
        "blocking_reasons": [
            "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE"
        ],
    }

    state.record_capacity_wait(
        family="random_forest",
        decision=decision,
        now=now,
        conflicting_live_family="momentum",
    )

    assert state.run_mode == SERIAL_RECOVERY_MODE
    assert state.pending_successor == "random_forest"
    assert state.wait_age_seconds(now + timedelta(seconds=59)) == 59
    assert state.wait_age_seconds(now + timedelta(seconds=60)) == 60
    restored = FairRecoveryState.from_payload(
        state.payload(now=now + timedelta(seconds=60))
    )
    assert restored.pending_successor == "random_forest"
    assert restored.wait_started_at_utc == state.wait_started_at_utc


def test_serial_successor_cannot_steal_a_slot_before_live_owner_yields() -> None:
    state = FairRecoveryState(
        run_mode=SERIAL_RECOVERY_MODE,
        current_turn="momentum",
        pending_successor="random_forest",
        wait_started_at_utc="2026-10-03T12:00:00+00:00",
    )

    assert _serial_recovery_launch_blocker(
        state,
        family="random_forest",
        active_families={"momentum"},
    ) == "momentum"
    assert _serial_recovery_launch_blocker(
        state,
        family="random_forest",
        active_families=set(),
    ) is None


def test_concurrent_mode_does_not_apply_the_serial_turn_blocker() -> None:
    state = FairRecoveryState(
        run_mode=CONCURRENT_RECOVERY_MODE,
        current_turn="momentum",
        pending_successor="random_forest",
    )

    assert _serial_recovery_launch_blocker(
        state,
        family="random_forest",
        active_families={"momentum"},
    ) is None


def test_serial_recovery_rotates_without_starving_either_family() -> None:
    now = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)
    state = FairRecoveryState(run_mode=SERIAL_RECOVERY_MODE)
    active = "momentum"
    turns = {"random_forest": 0, "momentum": 0}

    for ordinal in range(20):
        successor = (
            "random_forest" if active == "momentum" else "momentum"
        )
        decision = {
            "classification": "LOCAL_CAPACITY_DEFERRAL",
            "blocking_reasons": [
                "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE"
            ],
            "simulation_ordinal": ordinal,
        }
        state.record_capacity_wait(
            family=successor,
            decision=decision,
            now=now,
            conflicting_live_family=active,
        )
        handoff_at = now + timedelta(seconds=60)
        assert state.wait_age_seconds(handoff_at) == 60
        state.record_handoff_request(now=handoff_at)
        state.record_handoff_completion(
            yielded_family=active,
            successor=successor,
            now=handoff_at + timedelta(seconds=1),
        )
        active = successor
        turns[active] += 1
        now = handoff_at + timedelta(seconds=2)

    assert turns == {"random_forest": 10, "momentum": 10}
    assert state.current_turn == "momentum"
    assert state.pending_successor is None
