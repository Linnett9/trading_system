from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.research.ml.ds24 import remote_family_queue
from core.research.ml.ds24 import vast_reverse_queue_r1 as queue


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-09-03T12:00:00Z"


def definition(**kwargs: object) -> dict:
    return queue.build_queue_definition(ROOT, **kwargs)


def neutral_snapshots(queue_definition: dict) -> list[dict]:
    return queue.synthetic_external_status_fixture(queue_definition, now_utc=NOW)["snapshots"]


def update_snapshot_row(snapshot: dict, family_id: str, **updates: object) -> dict:
    updated = copy.deepcopy(snapshot)
    for row in updated["families"]:
        if row["family_id"] == family_id:
            row.update(updates)
    updated["snapshot_hash"] = queue.snapshot_hash(updated)
    return updated


def test_queue_definition_is_exact_reverse_family_contract() -> None:
    first = definition()
    second = definition()

    assert first["queue_id"] == queue.QUEUE_ID
    assert first["canonical_family_order"] == list(remote_family_queue.REMOTE_QUEUE_ORDER)
    assert first["mac_top_down_order"] == list(reversed(remote_family_queue.REMOTE_QUEUE_ORDER))
    assert first["canonical_family_count"] == 9
    assert first["queue_definition_hash"] == second["queue_definition_hash"]
    assert first["vast_order_is_reverse_of_mac_order"] is True
    assert first["mac_aux_queue_module_present_in_checkout"] == (ROOT / queue.MAC_AUX_QUEUE_REL).exists()
    assert str(ROOT) not in json.dumps(first, sort_keys=True)
    assert "\\" not in first["source_registry_path"]
    assert all(family == family.lower() for family in first["canonical_family_order"])
    assert queue.validate_queue_definition(first)["status"] == "PASS"


def test_queue_definition_rejects_duplicate_missing_and_unexpected_membership() -> None:
    order = list(remote_family_queue.REMOTE_QUEUE_ORDER)
    with pytest.raises(queue.VastReverseQueueError, match="DUPLICATE_FAMILY"):
        definition(accepted_order=[*order[:-1], order[0]])
    with pytest.raises(queue.VastReverseQueueError, match="COUNT_MISMATCH"):
        definition(accepted_order=order[:-1])
    with pytest.raises(queue.VastReverseQueueError, match="SET_MISMATCH"):
        definition(accepted_order=[*order[:-1], "unknown_family"])


def test_neutral_snapshots_make_temporal_fusion_transformer_first_claimable(tmp_path: Path) -> None:
    queue_definition = definition()
    plan = queue.dry_run_plan(
        tmp_path,
        queue_definition,
        neutral_snapshots(queue_definition),
        now_utc=NOW,
    )

    assert plan["status"] == "READY"
    assert plan["admission_status"] == "CLAIMABLE"
    assert plan["next_vast_eligible_family"] == "temporal_fusion_transformer"
    assert plan["model_executor_invoked"] is False
    assert plan["guards"]["holdout_accessed"] is False
    assert plan["guards"]["full_prediction_output_written"] is False


def test_missing_external_snapshots_fail_closed_for_claim_cli(tmp_path: Path) -> None:
    with pytest.raises(queue.VastReverseQueueError, match="EXTERNAL_COORDINATION_FAIL_CLOSED"):
        queue.main(
            [
                "claim",
                "--repo-root",
                str(ROOT),
                "--queue-root",
                str(tmp_path),
                "--test-only",
                "--now-utc",
                NOW,
            ]
        )


def test_compatible_dell_running_blocks_admission_without_skip(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    dell = update_snapshot_row(
        dell,
        "temporal_fusion_transformer",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="dell",
        pid=1234,
        pid_alive=True,
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "PAUSED"
    assert plan["admission_status"] == "BLOCKED_EXTERNAL_OWNER"
    assert plan["next_vast_eligible_family"] == ""
    assert plan["blocked_by_live_external"] == ["temporal_fusion_transformer"]


def test_compatible_mac_running_marks_meeting_boundary(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    mac = update_snapshot_row(
        mac,
        "temporal_fusion_transformer",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="mac",
        pid=2222,
        pid_alive=True,
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "PAUSED"
    assert plan["admission_status"] == "BOUNDARY_REACHED_EXTERNAL_OWNER"
    assert plan["nearest_externally_owned_boundary"] == "temporal_fusion_transformer"
    assert plan["queues_met"] is True


def test_verified_external_completion_is_skipped(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    dell = update_snapshot_row(
        dell,
        "temporal_fusion_transformer",
        family_state="COMPLETE",
        ownership_state="COMPLETE",
        result_artifact_manifest_hash="verified-result-manifest-hash",
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "READY"
    assert plan["admission_status"] == "CLAIMABLE"
    assert plan["skipped_external_verified"] == ["temporal_fusion_transformer"]
    assert plan["next_vast_eligible_family"] == "market_context_encoder"


def test_incompatible_external_completion_blocks_reuse(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    dell = update_snapshot_row(
        dell,
        "temporal_fusion_transformer",
        family_state="COMPLETE",
        ownership_state="COMPLETE",
        model_configuration_hash="different-model-configuration",
        result_artifact_manifest_hash="unverified-result-manifest-hash",
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "PAUSED"
    assert plan["admission_status"] == "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED"
    assert plan["next_vast_eligible_family"] == ""


def test_stale_external_status_fails_closed(tmp_path: Path) -> None:
    queue_definition = definition()
    stale = queue.synthetic_external_status_fixture(
        queue_definition,
        now_utc="2026-09-03T11:00:00Z",
    )["snapshots"]

    plan = queue.dry_run_plan(tmp_path, queue_definition, stale, now_utc=NOW)

    assert plan["status"] == "FAIL_CLOSED"
    assert plan["admission_status"] == "EXTERNAL_COORDINATION_FAIL_CLOSED"
    assert "SNAPSHOT_STALE_FAIL_CLOSED" in plan["diagnostics"]["failures"][0]["errors"]


def test_malformed_external_status_fails_closed(tmp_path: Path) -> None:
    queue_definition = definition()
    bad = neutral_snapshots(queue_definition)
    del bad[0]["families"][0]["target_contract_hash"]
    bad[0]["snapshot_hash"] = queue.snapshot_hash(bad[0])

    plan = queue.dry_run_plan(tmp_path, queue_definition, bad, now_utc=NOW)

    assert plan["status"] == "FAIL_CLOSED"
    assert plan["admission_status"] == "EXTERNAL_COORDINATION_FAIL_CLOSED"
    assert plan["diagnostics"]["failures"][0]["classification"] == "SNAPSHOT_MALFORMED_OR_STALE_FAIL_CLOSED"


def test_contradictory_external_active_ownership_fails_closed(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    dell = update_snapshot_row(
        dell,
        "temporal_fusion_transformer",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="dell",
        pid=3001,
        pid_alive=True,
    )
    mac = update_snapshot_row(
        mac,
        "temporal_fusion_transformer",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="mac",
        pid=3002,
        pid_alive=True,
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "FAIL_CLOSED"
    assert plan["diagnostics"]["classification"] == "EXTERNAL_COORDINATION_FAIL_CLOSED"
    assert plan["diagnostics"]["contradictions"][0]["family_id"] == "temporal_fusion_transformer"


def test_dead_external_pid_requires_ambiguous_recovery(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    dell = update_snapshot_row(
        dell,
        "temporal_fusion_transformer",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="dell",
        pid=7777,
        pid_alive=False,
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "PAUSED"
    assert plan["admission_status"] == "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED"
    assert plan["next_vast_eligible_family"] == ""


def test_local_claim_is_atomic_and_resume_deterministic(tmp_path: Path) -> None:
    queue_definition = definition()
    snapshots = neutral_snapshots(queue_definition)

    first = queue.record_local_vast_claim(tmp_path, queue_definition, snapshots, now_utc=NOW, test_only=True)
    second = queue.record_local_vast_claim(tmp_path, queue_definition, snapshots, now_utc=NOW, test_only=True)
    state = queue.load_queue_state(tmp_path, queue_definition)

    assert first["status"] == "CLAIM_RECORDED_TEST_ONLY"
    assert second["status"] == "CLAIM_ALREADY_ACTIVE"
    assert second["claim"]["claim_id"] == first["claim"]["claim_id"]
    assert len(state["vast_claims"]) == 1
    assert state["ledger"][0]["status"] == "RESERVED_FOR_VAST"
    assert first["model_executor_invoked"] is False


def test_corrupted_queue_state_is_refused(tmp_path: Path) -> None:
    queue_definition = definition()
    state = queue.load_queue_state(tmp_path, queue_definition)
    state["generation"] = 99
    queue.queue_state_path(tmp_path).write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(queue.VastReverseQueueError, match="STATE_HASH_MISMATCH"):
        queue.load_queue_state(tmp_path, queue_definition)


def test_queue_definition_mismatch_is_refused_on_resume(tmp_path: Path) -> None:
    original = definition()
    queue.load_queue_state(tmp_path, original)
    incompatible = definition(configuration_overrides={"temporal_fusion_transformer": "IMPLEMENTATION_BLOCKED"})

    with pytest.raises(queue.VastReverseQueueError, match="QUEUE_DEFINITION_MISMATCH"):
        queue.load_queue_state(tmp_path, incompatible)


def test_vast_does_not_traverse_future_mac_boundary(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    mac = update_snapshot_row(
        mac,
        "market_context_encoder",
        family_state="RUNNING",
        ownership_state="RUNNING",
        active_owner="mac",
        pid=9090,
        pid_alive=True,
    )

    plan = queue.dry_run_plan(tmp_path, queue_definition, [dell, mac], now_utc=NOW)

    assert plan["status"] == "READY"
    assert plan["next_vast_eligible_family"] == "temporal_fusion_transformer"
    assert plan["nearest_externally_owned_boundary"] == "market_context_encoder"
    assert plan["queues_met"] is False


def test_blocked_tft_configuration_can_pause_or_skip_without_invented_authority(tmp_path: Path) -> None:
    queue_definition = definition(
        configuration_overrides={"temporal_fusion_transformer": "CONFIGURATION_AUTHORITY_REQUIRED"}
    )
    snapshots = neutral_snapshots(queue_definition)
    skip_plan = queue.dry_run_plan(tmp_path / "skip", queue_definition, snapshots, now_utc=NOW)
    pause_plan = queue.dry_run_plan(
        tmp_path / "pause",
        queue_definition,
        snapshots,
        now_utc=NOW,
        skip_configuration_blockers=False,
    )

    assert skip_plan["next_vast_eligible_family"] == "market_context_encoder"
    assert skip_plan["blocked_families"] == [
        {"family_id": "temporal_fusion_transformer", "reason": "CONFIGURATION_AUTHORITY_REQUIRED"}
    ]
    assert pause_plan["status"] == "PAUSED"
    assert pause_plan["admission_status"] == "CONFIGURATION_AUTHORITY_REQUIRED"
    assert pause_plan["next_vast_eligible_family"] == ""


def test_utc_timestamp_helpers_are_normalized() -> None:
    parsed = queue.parse_utc("2026-09-03T12:00:00")

    assert parsed is not None
    assert queue.format_utc(parsed) == NOW
    assert queue.parse_utc("not-a-date") is None


def test_status_payload_exposes_reserved_family_and_queue_only_guards(tmp_path: Path) -> None:
    queue_definition = definition()
    dell, mac = neutral_snapshots(queue_definition)
    dell = update_snapshot_row(
        dell,
        "temporal_fusion_transformer",
        family_state="COMPLETE",
        ownership_state="COMPLETE",
        result_artifact_manifest_hash="verified-result-manifest-hash",
    )
    snapshots = [dell, mac]
    queue.record_local_vast_claim(
        tmp_path,
        queue_definition,
        snapshots,
        now_utc=NOW,
        lease_seconds=24 * 60 * 60,
        test_only=True,
    )

    status = queue.status_payload(tmp_path, queue_definition)

    assert status["state_validation"]["status"] == "PASS"
    assert status["reserved_for_vast"] == ["market_context_encoder"]
    assert status["skipped_families"][0]["family_id"] == "temporal_fusion_transformer"
    assert status["ledger"][0]["status"] == "SKIPPED_EXTERNAL_VERIFIED"
    assert status["ledger"][0]["skip_reason"] == "compatible external completion verified"
    assert status["model_executor_invoked"] is False
    assert status["guards"]["vast_cloud_operation_performed"] is False
    assert status["guards"]["live_dell_mac_queue_state_mutated"] is False


def test_authority_package_writes_required_queue_only_artifacts(tmp_path: Path) -> None:
    manifest = queue.write_authority_package(ROOT, tmp_path / "authority")
    files = {row["path"] for row in manifest["artifact_inventory"]}

    assert manifest["queue_id"] == queue.QUEUE_ID
    assert manifest["terminal_classification"] == queue.TERMINAL_CLASSIFICATION
    assert manifest["dry_run_next_family"] == "temporal_fusion_transformer"
    assert manifest["safety"]["queue_only"] is True
    assert manifest["cloud_or_vast_operation_occurred"] is False
    assert (tmp_path / "authority" / "manifest.json").is_file()
    assert {
        "vast_reverse_queue_definition.json",
        "external_family_status.schema.json",
        "synthetic_external_status_fixture.json",
        "vast_family_claim.schema.json",
        "dry_run_plan.json",
        "queue_validation.json",
        "test_evidence.json",
        "README.md",
        "limitations.json",
        "process_snapshot_before.json",
        "process_snapshot_after.json",
        "queue_state/queue_state.json",
    } - files == set()
