from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from core.research.ml.ds24 import vast_b2_bootstrap_r1 as b2
from core.research.ml.ds24 import vast_reverse_queue_r1 as reverse_queue


ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-09-03T12:00:00Z"


def qdef() -> dict:
    return b2.read_json(ROOT / b2.DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")


def neutral_pair(queue_definition: dict) -> tuple[dict, dict]:
    dell, mac = reverse_queue.synthetic_external_status_fixture(queue_definition, now_utc=NOW)["snapshots"]
    return dell, mac


def update_snapshot_row(snapshot: dict, family_id: str, **updates: object) -> dict:
    out = copy.deepcopy(snapshot)
    for row in out["families"]:
        if row["family_id"] == family_id:
            row.update(updates)
    out["snapshot_hash"] = reverse_queue.snapshot_hash(out)
    return out


def fake_dataset(tmp_path: Path, payloads: dict[str, bytes] | None = None) -> tuple[Path, Path, list[b2.DatasetManifestRow], list[dict]]:
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    payloads = payloads or {
        "data/a.bin": b"alpha",
        "data/b.bin": b"bravo-bravo",
        "data/c.bin": b"charlie-charlie-charlie",
    }
    for rel, payload in payloads.items():
        path = b2.safe_join(source, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    rows = b2.manifest_rows_from_files(source, list(payloads))
    for row in rows:
        src = b2.safe_join(source, row.relative_path)
        dst = b2.safe_join(remote, row.object_key())
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    remote_objects = [
        {"key": row.object_key(), "size_bytes": row.size_bytes, "sha256": row.sha256, "sha1": row.sha1}
        for row in rows
    ]
    return source, remote, rows, remote_objects


def finalizer_for(rows: list[b2.DatasetManifestRow], repo_root: Path = ROOT) -> b2.DatasetPublisherFinalizer:
    return b2.DatasetPublisherFinalizer(
        repo_root=repo_root,
        expected_count=len(rows),
        expected_bytes=sum(row.size_bytes for row in rows),
    )


def acknowledged_plan(tmp_path: Path, queue_definition: dict, *, dell: dict | None = None, mac: dict | None = None, created_at: str = NOW) -> Path:
    default_dell, default_mac = neutral_pair(queue_definition)
    plan = b2.create_ownership_plan(
        queue_definition,
        dell or default_dell,
        mac or default_mac,
        created_at_utc=created_at,
    )
    plan["acknowledgements"]["dell"] = b2.build_acknowledgement(machine="dell", plan_hash=plan["plan_hash"], now_utc=created_at)
    plan["acknowledgements"]["mac"] = b2.build_acknowledgement(machine="mac", plan_hash=plan["plan_hash"], now_utc=created_at)
    path = tmp_path / "ownership_plan.json"
    b2.write_json_atomic(path, plan)
    return path


def bootstrap_context(tmp_path: Path, *, write_marker: bool = True, corrupt_marker: bool = False) -> tuple[b2.VastBootstrapController, list[b2.DatasetManifestRow], Path, Path]:
    _source, remote, rows, remote_objects = fake_dataset(tmp_path)
    if write_marker:
        authority = finalizer_for(rows).finalize(rows, remote_objects, output_root=remote / b2.B2_PREFIX, now_utc=NOW)
        if corrupt_marker:
            marker = b2.read_json(remote / b2.DATASET_COMPLETE_MARKER_KEY)
            marker["verification_result"] = "FAIL"
            b2.write_json_atomic(remote / b2.DATASET_COMPLETE_MARKER_KEY, marker)
    queue_definition = qdef()
    plan_path = acknowledged_plan(tmp_path, queue_definition)
    output_root = tmp_path / "vast_output"
    config = b2.build_bootstrap_config(
        repo_root=ROOT,
        run_id="run-a",
        dataset_root=tmp_path / "vast_dataset",
        output_root=output_root,
        ownership_plan_path=plan_path,
        queue_definition_hash=queue_definition["queue_definition_hash"],
        dataset_authority_hash="dataset-authority",
    )
    controller = b2.VastBootstrapController(
        repo_root=ROOT,
        config=config,
        dataset_rows=rows,
        remote_root=remote,
        resource_snapshot={
            "cpu_cores": 32,
            "ram_bytes": 128 * 1024**3,
            "gpu_present": True,
            "vram_bytes": 24 * 1024**3,
            "cuda_present": True,
            "disk_free_bytes": 500 * 1024**3,
            "time_synchronized": True,
            "workspace_writable": True,
        },
        env={"B2_APPLICATION_KEY_ID": "id", "B2_APPLICATION_KEY": "key"},
    )
    return controller, rows, remote, output_root


def make_local_run(root: Path) -> Path:
    local = root / "local_run"
    files = {
        "queue_state/queue_state.json": b"{}",
        "ownership/ownership_plan.json": b"{}",
        "metrics_only_v3/family=temporal_fusion_transformer/metrics.json": b"{}",
        "logs/stdout.log": b"log",
        "telemetry/gpu.json": b"{}",
        "checkpoints/family=temporal_fusion_transformer/latest.ckpt": b"checkpoint",
        "prediction_partitions/full_predictions.parquet": b"forbidden",
    }
    for rel, payload in files.items():
        path = b2.safe_join(local, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return local


def published_remote(tmp_path: Path, *, run_id: str = "run-a") -> Path:
    queue_definition = qdef()
    local = make_local_run(tmp_path)
    remote = tmp_path / "remote_runs" / f"queue={b2.QUEUE_ID}" / f"run={run_id}"
    publisher = b2.VastDurableArtifactPublisher(
        local_run_root=local,
        remote_run_root=remote,
        policy=b2.artifact_retention_policy(queue_definition),
        run_id=run_id,
        dataset_authority_hash="dataset-hash",
        queue_definition_hash=queue_definition["queue_definition_hash"],
    )
    assert publisher.publish_once(now_utc=NOW)["status"] == "PASS"
    return tmp_path / "remote_runs"


def test_01_prerequisite_queue_authority_is_valid() -> None:
    result = b2.load_prerequisite_queue_authority(ROOT)
    assert result["status"] == "PASS"
    assert result["queue_id"] == b2.QUEUE_ID
    assert result["queue_definition_hash"] == qdef()["queue_definition_hash"]


def test_02_missing_prerequisite_queue_authority_blocks_dependent_package(tmp_path: Path) -> None:
    result = b2.load_prerequisite_queue_authority(ROOT, tmp_path / "missing")
    assert result["status"] == "FAIL"
    assert result["terminal_if_failed"] == b2.BLOCKED_QUEUE_AUTHORITY


def test_03_manifest_validation_rejects_absolute_traversal_duplicate_and_forbidden_paths(tmp_path: Path) -> None:
    rows = [
        b2.DatasetManifestRow("C:/absolute.bin", 1),
        b2.DatasetManifestRow("../escape.bin", 1),
        b2.DatasetManifestRow("data/a.bin", 1),
        b2.DatasetManifestRow("data/a.bin", 1),
        b2.DatasetManifestRow("prediction_partitions/full_predictions.parquet", 1),
    ]
    result = b2.validate_input_manifest(tmp_path, rows, expected_count=5, expected_bytes=5)
    errors = json.dumps(result["errors"])
    assert result["status"] == "FAIL"
    assert "DRIVE_PATH_REFUSED" in errors
    assert "TRAVERSAL_PATH_REFUSED" in errors
    assert "DUPLICATE_ENTRY" in errors
    assert "FORBIDDEN_ARTIFACT_PATH" in errors


def test_04_dataset_count_bytes_and_marker_authority_pass(tmp_path: Path) -> None:
    _source, remote, rows, remote_objects = fake_dataset(tmp_path)
    authority = finalizer_for(rows).finalize(rows, remote_objects, output_root=remote / b2.B2_PREFIX, now_utc=NOW)
    marker = b2.read_json(remote / b2.DATASET_COMPLETE_MARKER_KEY)
    assert authority["status"] == "PASS"
    assert authority["expected_object_count"] == len(rows)
    assert authority["expected_bytes"] == sum(row.size_bytes for row in rows)
    assert marker["completion_marker_predecessor_hash"] == authority["authority_hash"]
    assert b2.validate_dataset_marker(marker)["status"] == "PASS"


def test_05_missing_b2_completion_marker_blocks_preflight(tmp_path: Path) -> None:
    controller, _rows, _remote, _out = bootstrap_context(tmp_path, write_marker=False)
    result = controller.run(mode="preflight-only", now_utc=NOW)
    assert result["status"] == "FAIL"
    assert result["checks"]["dataset_completion_marker"] is False
    assert result["model_started"] is False


def test_06_invalid_b2_completion_marker_blocks_preflight(tmp_path: Path) -> None:
    controller, _rows, _remote, _out = bootstrap_context(tmp_path, corrupt_marker=True)
    result = controller.run(mode="preflight-only", now_utc=NOW)
    assert result["status"] == "FAIL"
    assert "DATASET_MARKER_VERIFICATION_NOT_PASS" in result["dataset_marker_validation"]["errors"]


def test_07_resumable_interrupted_download_appends_remaining_bytes(tmp_path: Path) -> None:
    source, remote, rows, _objects = fake_dataset(tmp_path, {"data/a.bin": b"abcdefghij"})
    partial = b2.safe_join(tmp_path / "dataset", "data/a.bin")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"abc")
    result = b2.download_dataset_from_fake_b2(remote, tmp_path / "dataset", rows)
    assert result["status"] == "PASS"
    assert result["resumed_partial_count"] == 1
    assert partial.read_bytes() == (source / "data/a.bin").read_bytes()


def test_08_matching_downloaded_file_is_skipped(tmp_path: Path) -> None:
    _source, remote, rows, _objects = fake_dataset(tmp_path)
    local = tmp_path / "dataset"
    for row in rows:
        src = b2.safe_join(remote, row.object_key())
        dst = b2.safe_join(local, row.relative_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    result = b2.download_dataset_from_fake_b2(remote, local, rows)
    assert result["status"] == "PASS"
    assert result["skipped_matching_count"] == len(rows)


def test_09_corrupt_downloaded_file_is_detected(tmp_path: Path) -> None:
    _source, _remote, rows, _objects = fake_dataset(tmp_path, {"data/a.bin": b"abc"})
    local = tmp_path / "dataset"
    bad = b2.safe_join(local, "data/a.bin")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"abd")
    result = b2.verify_downloaded_dataset(local, rows)
    assert result["status"] == "FAIL"
    assert result["failures"][0]["reason"] == "SHA256_MISMATCH"


def test_10_preflight_only_does_not_autostart_or_write_supervisor(tmp_path: Path) -> None:
    controller, _rows, _remote, out = bootstrap_context(tmp_path)
    result = controller.run(mode="preflight-only", now_utc=NOW)
    assert result["status"] == "PASS"
    assert result["model_started"] is False
    assert not (out / "supervisor.lease.json").exists()


def test_11_autostart_after_all_gates_passes_with_stub_executor(tmp_path: Path) -> None:
    controller, _rows, _remote, out = bootstrap_context(tmp_path)
    result = controller.run(mode="start-after-verify", now_utc=NOW, test_stub_executor=True)
    assert result["status"] == "PASS"
    assert result["classification"] == "VAST_REVERSE_QUEUE_AUTOSTARTED_STUB"
    assert result["stub_executor_started"] is True
    assert result["live_model_started"] is False
    assert (out / "supervisor.lease.json").is_file()


def test_12_duplicate_supervisor_prevents_second_launch(tmp_path: Path) -> None:
    controller, _rows, _remote, out = bootstrap_context(tmp_path)
    assert controller.run(mode="start-after-verify", now_utc=NOW, test_stub_executor=True)["status"] == "PASS"
    duplicate = controller.run(mode="start-after-verify", now_utc="2026-09-03T12:01:00Z", test_stub_executor=True)
    assert duplicate["status"] == "FAIL"
    assert duplicate["classification"] == "BOOTSTRAP_LEASE_ALREADY_ACTIVE"
    direct = b2.launch_reverse_queue_stub(out, run_id="run-a", now_utc="2026-09-03T12:01:00Z")
    assert direct["classification"] == "DUPLICATE_SUPERVISOR_PREVENTED"


def test_13_resume_preserves_existing_queue_cursor(tmp_path: Path) -> None:
    controller, _rows, _remote, out = bootstrap_context(tmp_path)
    state_root = out / "queue_state"
    state = reverse_queue.initial_queue_state(qdef(), now_utc=NOW)
    state["current_cursor"] = "market_context_encoder"
    state = reverse_queue.write_queue_state(state_root, state)
    result = controller.run(mode="resume", now_utc=NOW, test_stub_executor=True)
    assert result["status"] == "PASS"
    assert result["launch"]["existing_queue_cursor_preserved"] == "market_context_encoder"
    assert reverse_queue.load_queue_state(state_root, qdef())["state_hash"] == state["state_hash"]


def test_14_ownership_plan_consumes_exact_reverse_queue_first_candidate() -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc=NOW, mac_unavailable_reason="synthetic")
    assert plan["vast_static_partition"] == ["temporal_fusion_transformer"]
    assert plan["meeting_boundary"]["next_vast_eligible_family"] == "temporal_fusion_transformer"


def test_15_compatible_external_running_family_is_excluded_from_vast_partition() -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    dell = update_snapshot_row(dell, "temporal_fusion_transformer", family_state="RUNNING", ownership_state="RUNNING", active_owner="dell", pid=44, pid_alive=True)
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc=NOW, mac_unavailable_reason="synthetic")
    assert "temporal_fusion_transformer" in plan["currently_running_families"]
    assert plan["vast_static_partition"] == []


def test_16_compatible_external_completion_is_skipped_in_ownership_plan() -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    dell = update_snapshot_row(dell, "temporal_fusion_transformer", family_state="COMPLETE", ownership_state="COMPLETE", result_artifact_manifest_hash="result")
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc=NOW, mac_unavailable_reason="synthetic")
    assert plan["completed_compatible_families"] == ["temporal_fusion_transformer"]
    assert plan["vast_static_partition"] == ["market_context_encoder"]


def test_17_stale_ownership_plan_is_rejected(tmp_path: Path) -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc="2026-09-03T11:00:00Z", mac_unavailable_reason="synthetic")
    plan["acknowledgements"]["dell"] = b2.build_acknowledgement(machine="dell", plan_hash=plan["plan_hash"], now_utc=NOW)
    assert b2.validate_ownership_plan(plan, now_utc=NOW)["status"] == "FAIL"


def test_18_missing_acknowledgement_is_rejected() -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc=NOW)
    result = b2.validate_ownership_plan(plan, now_utc=NOW)
    assert result["status"] == "FAIL"
    assert "DELL_ACKNOWLEDGEMENT_MISSING_OR_MISMATCHED" in result["errors"]
    assert "MAC_ACKNOWLEDGEMENT_OR_REASON_MISSING" in result["errors"]


def test_19_static_partition_overlap_with_mac_fails_validation() -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc=NOW, mac_unavailable_reason="synthetic")
    plan["mac_owned_families"] = ["temporal_fusion_transformer"]
    plan["vast_static_partition"] = ["temporal_fusion_transformer"]
    plan["plan_hash"] = b2.stable_hash(b2.ownership_plan_hash_payload(plan))
    plan["acknowledgements"]["dell"] = b2.build_acknowledgement(machine="dell", plan_hash=plan["plan_hash"], now_utc=NOW)
    result = b2.validate_ownership_plan(plan, now_utc=NOW)
    assert result["status"] == "FAIL"
    assert "VAST_MAC_FAMILY_OVERLAP" in result["errors"]


def test_20_meeting_boundary_with_mac_claim_is_explicit() -> None:
    queue_definition = qdef()
    dell, mac = neutral_pair(queue_definition)
    mac = update_snapshot_row(mac, "temporal_fusion_transformer", family_state="RUNNING", ownership_state="RUNNING", active_owner="mac", pid=55, pid_alive=True)
    plan = b2.create_ownership_plan(queue_definition, dell, mac, created_at_utc=NOW, mac_unavailable_reason="synthetic")
    assert plan["meeting_boundary"]["admission_status"] == "BOUNDARY_REACHED_EXTERNAL_OWNER"
    assert plan["meeting_boundary"]["queues_met"] is True
    assert plan["vast_static_partition"] == []


def test_21_no_mac_vast_family_overlap_in_valid_plan(tmp_path: Path) -> None:
    queue_definition = qdef()
    plan_path = acknowledged_plan(tmp_path, queue_definition)
    plan = b2.read_json(plan_path)
    assert b2.validate_ownership_plan(plan, now_utc=NOW)["status"] == "PASS"
    assert set(plan["mac_owned_families"]) & set(plan["vast_static_partition"]) == set()


def test_22_checkpoint_publication_preserves_hash_and_commit_marker(tmp_path: Path) -> None:
    queue_definition = qdef()
    local = make_local_run(tmp_path)
    remote = tmp_path / "remote" / "run=run-a"
    result = b2.VastDurableArtifactPublisher(
        local_run_root=local,
        remote_run_root=remote,
        policy=b2.artifact_retention_policy(queue_definition),
        run_id="run-a",
        dataset_authority_hash="dataset",
        queue_definition_hash=queue_definition["queue_definition_hash"],
    ).publish_once(now_utc=NOW)
    checkpoint = "checkpoints/family=temporal_fusion_transformer/latest.ckpt"
    assert result["status"] == "PASS"
    assert b2.file_sha256(b2.safe_join(remote, checkpoint)) == b2.file_sha256(b2.safe_join(local, checkpoint))
    assert (remote / "COMMITTED.json").is_file()


def test_23_interrupted_publication_does_not_write_committed_marker(tmp_path: Path) -> None:
    queue_definition = qdef()
    remote = tmp_path / "remote" / "run=run-a"
    result = b2.VastDurableArtifactPublisher(
        local_run_root=make_local_run(tmp_path),
        remote_run_root=remote,
        policy=b2.artifact_retention_policy(queue_definition),
        run_id="run-a",
        dataset_authority_hash="dataset",
        queue_definition_hash=queue_definition["queue_definition_hash"],
    ).publish_once(interrupt_after_files=1, now_utc=NOW)
    assert result["status"] == "FAIL"
    assert result["backlog_count"] > 0
    assert not (remote / "COMMITTED.json").exists()


def test_24_publisher_retry_and_backlog_then_success(tmp_path: Path) -> None:
    queue_definition = qdef()
    publisher = b2.VastDurableArtifactPublisher(
        local_run_root=make_local_run(tmp_path),
        remote_run_root=tmp_path / "remote" / "run=run-a",
        policy=b2.artifact_retention_policy(queue_definition),
        run_id="run-a",
        dataset_authority_hash="dataset",
        queue_definition_hash=queue_definition["queue_definition_hash"],
    )
    interrupted = publisher.publish_once(interrupt_after_files=2, now_utc=NOW)
    completed = publisher.publish_once(now_utc=NOW)
    assert interrupted["classification"] == "VAST_OUTPUT_PUBLICATION_INTERRUPTED_RETRYABLE"
    assert completed["classification"] == "VAST_OUTPUTS_DURABLY_PUBLISHED"


def test_25_stale_backup_blocks_new_family_admission_without_killing_fit(tmp_path: Path) -> None:
    queue_definition = qdef()
    publisher = b2.VastDurableArtifactPublisher(
        local_run_root=make_local_run(tmp_path),
        remote_run_root=tmp_path / "remote" / "run=run-a",
        policy=b2.artifact_retention_policy(queue_definition),
        run_id="run-a",
        dataset_authority_hash="dataset",
        queue_definition_hash=queue_definition["queue_definition_hash"],
    )
    publisher.publish_once(now_utc=NOW)
    status = publisher.status(now_utc="2026-09-03T12:45:00Z", max_backup_age_seconds=60)
    assert status["classification"] == "BACKUP_STALE_BLOCK_NEW_FAMILY_ADMISSION"
    assert status["active_fit_killed"] is False


def test_26_whitelist_enforcement_rejects_unapproved_roots() -> None:
    policy = b2.artifact_retention_policy(qdef())
    assert b2.classify_artifact("tmp/cache.bin", policy)["status"] == "REJECTED"
    assert b2.classify_artifact("metrics_only_v3/family=x/metrics.json", policy)["status"] == "ALLOWED"


def test_27_forbidden_full_prediction_artifacts_are_not_inventoried(tmp_path: Path) -> None:
    policy = b2.artifact_retention_policy(qdef())
    rows = b2.inventory_allowed_artifacts(make_local_run(tmp_path), policy)
    assert all("prediction_partitions" not in row["relative_path"] for row in rows)
    assert b2.classify_artifact("prediction_partitions/full_predictions.parquet", policy)["status"] == "FORBIDDEN"


def test_28_family_specific_weight_retention_policy_is_explicit() -> None:
    policy = b2.artifact_retention_policy(qdef())
    rows = {row["family_id"]: row for row in policy["family_weight_retention"]}
    assert rows["temporal_fusion_transformer"]["weights_required_for_resume"] is True
    assert rows["lightgbm_rank_xendcg"]["final_deployable_weight_acceptance"] == "not claimed by transport ticket"
    assert all(row["per_window_weights_retained"] is False for row in rows.values())


def test_29_compact_dell_retrieval_defers_tier2_checkpoints(tmp_path: Path) -> None:
    remote_runs = published_remote(tmp_path)
    client = b2.DellArtifactRepatriationClient(remote_runs_root=remote_runs, local_import_root=tmp_path / "import", free_bytes=100 * 1024**3)
    result = client.retrieve(run_id="run-a", tier="compact", now_utc=NOW)
    assert result["status"] == "PASS"
    assert "tier1" in result["artifact_tiers_present"]
    assert "tier2" in result["artifact_tiers_deferred"]


def test_30_full_tier_dell_retrieval_includes_checkpoints(tmp_path: Path) -> None:
    remote_runs = published_remote(tmp_path)
    client = b2.DellArtifactRepatriationClient(remote_runs_root=remote_runs, local_import_root=tmp_path / "import", free_bytes=100 * 1024**3)
    result = client.retrieve(run_id="run-a", tier="all", now_utc=NOW)
    assert result["status"] == "PASS"
    assert "tier2" in result["artifact_tiers_present"]
    assert result["artifact_tiers_deferred"] == []


def test_31_dell_disk_capacity_defers_without_deleting_b2(tmp_path: Path) -> None:
    remote_runs = published_remote(tmp_path)
    client = b2.DellArtifactRepatriationClient(remote_runs_root=remote_runs, local_import_root=tmp_path / "import", free_bytes=1)
    result = client.retrieve(run_id="run-a", tier="all", now_utc=NOW)
    assert result["classification"] == "DEFERRED_LOCAL_CAPACITY_REMOTE_COPY_DURABLE"
    assert result["b2_copy_remains_durable"] is True


def test_32_atomic_dell_staging_promotion_writes_receipt(tmp_path: Path) -> None:
    remote_runs = published_remote(tmp_path)
    client = b2.DellArtifactRepatriationClient(remote_runs_root=remote_runs, local_import_root=tmp_path / "import", free_bytes=100 * 1024**3)
    result = client.retrieve(run_id="run-a", tier="compact", now_utc=NOW)
    assert result["status"] == "PASS"
    assert (tmp_path / "import" / "run=run-a" / "dell_import_receipt.json").is_file()
    assert not (tmp_path / "import" / ".staging" / "run-a").exists()


def test_33_hash_conflict_quarantines_existing_local_artifact(tmp_path: Path) -> None:
    remote_runs = published_remote(tmp_path)
    client = b2.DellArtifactRepatriationClient(remote_runs_root=remote_runs, local_import_root=tmp_path / "import", free_bytes=100 * 1024**3)
    assert client.retrieve(run_id="run-a", tier="compact", now_utc=NOW)["status"] == "PASS"
    existing = tmp_path / "import" / "run=run-a" / "queue_state" / "queue_state.json"
    existing.write_text("conflict", encoding="utf-8")
    result = client.retrieve(run_id="run-a", tier="compact", now_utc=NOW)
    assert result["status"] == "PASS"
    assert result["conflicts"]
    assert (tmp_path / "import" / "quarantine").exists()


def test_34_identical_artifact_retrieval_is_idempotent(tmp_path: Path) -> None:
    remote_runs = published_remote(tmp_path)
    client = b2.DellArtifactRepatriationClient(remote_runs_root=remote_runs, local_import_root=tmp_path / "import", free_bytes=100 * 1024**3)
    first = client.retrieve(run_id="run-a", tier="compact", now_utc=NOW)
    second = client.retrieve(run_id="run-a", tier="compact", now_utc=NOW)
    assert first["status"] == "PASS"
    assert second["status"] == "PASS"
    assert second["conflicts"] == []


def test_35_windows_linux_path_handling_is_normalized_and_safe() -> None:
    assert b2.normalize_relative_path("data\\x/./y.bin") == "data/x/y.bin"
    with pytest.raises(b2.VastB2BootstrapError, match="DRIVE_PATH_REFUSED"):
        b2.normalize_relative_path("C:\\x\\y.bin")


def test_36_secret_redaction_never_prints_values() -> None:
    result = b2.secret_preflight({"B2_APPLICATION_KEY_ID": "id-123", "B2_APPLICATION_KEY": "super-secret"}, ["B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY"])
    text = json.dumps(result)
    assert result["status"] == "PASS"
    assert "super-secret" not in text
    assert "id-123" not in text
    assert result["secret_values_printed"] is False


def test_37_command_contracts_do_not_persist_credentials() -> None:
    command = b2.build_rclone_command_contract(action="copy", source="b2:<BUCKET>/<PREFIX>", destination="<DEST>", files_from="<FILES>")
    text = json.dumps(command)
    assert command["uses_copy_semantics"] is True
    assert command["credentials_in_command"] is False
    assert "APPLICATION_KEY" not in text


def test_38_launch_contract_has_zero_holdout_and_order_access(tmp_path: Path) -> None:
    result = b2.launch_reverse_queue_stub(tmp_path, run_id="run-a", now_utc=NOW)
    assert result["outer_holdout_access"] is False
    assert result["paper_orders"] == 0
    assert result["live_orders"] == 0


def test_39_start_after_verify_without_live_authorization_refuses_real_model_start(tmp_path: Path) -> None:
    controller, _rows, _remote, _out = bootstrap_context(tmp_path)
    result = controller.run(mode="start-after-verify", now_utc=NOW)
    assert result["status"] == "FAIL"
    assert result["classification"] == b2.READY_LIVE_PREFLIGHT_CLASSIFICATION
    assert result["live_model_started"] is False


def test_40_synthetic_authority_package_evidence_is_green() -> None:
    evidence = b2.synthetic_end_to_end_evidence(ROOT)
    assert evidence["status"] == "PASS"
    assert evidence["cloud_operation_performed"] is False
    assert evidence["vast_instance_rented"] is False


def test_41_repository_deployment_authority_excludes_dataset_bundle() -> None:
    authority = b2.repository_deployment_authority(ROOT, b2.load_prerequisite_queue_authority(ROOT))
    assert authority["dataset_packaged_in_code_bundle"] is False
    assert authority["credentials_packaged"] is False
    assert authority["queue_definition_hash"] == qdef()["queue_definition_hash"]


def test_42_deployment_bundle_rejects_dataset_payload(tmp_path: Path) -> None:
    with pytest.raises(b2.VastB2BootstrapError, match="REFUSES_DATASET_PAYLOAD"):
        b2.deployment_bundle_manifest(tmp_path, ["data/a.bin"])


def test_43_cli_command_contract_modes_are_present(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "bootstrap.json"
    cfg.write_text("{}", encoding="utf-8")
    assert b2.main(["bootstrap", "--config", str(cfg), "--preflight-only", "--vast-instance-id", "<VAST_INSTANCE_ID>"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "BOOTSTRAP_COMMAND_CONTRACT_READY_NOT_EXECUTED"
    assert payload["live_model_started"] is False


def test_44_dell_repatriation_rejects_incomplete_runs_without_marker(tmp_path: Path) -> None:
    remote = tmp_path / "remote" / "run=run-a"
    remote.mkdir(parents=True)
    b2.write_json_atomic(remote / "vast_output_manifest.json", {"run_id": "run-a", "files": []})
    discovery = b2.DellArtifactRepatriationClient(remote_runs_root=tmp_path / "remote", local_import_root=tmp_path / "import").discover()
    assert discovery["eligible_run_count"] == 0


def test_45_runbook_commands_use_placeholders_not_real_credentials() -> None:
    commands = b2.operational_command_catalog(Path("<AUTHORITY_ROOT>"))["commands"]
    text = json.dumps(commands)
    assert "<RUN_ID>" in text
    assert "<VAST_INSTANCE_ID>" in text
    assert "B2_APPLICATION_KEY=" not in text
    assert b2.LIVE_BOOTSTRAP_CONFIRM_TOKEN in text
