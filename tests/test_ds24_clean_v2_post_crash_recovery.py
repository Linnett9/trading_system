from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from core.research.ml import ds24_metrics_only_evaluator as evaluator
from core.research.ml.ds24 import clean_v2_resources as resources
from core.research.ml.ds24.clean_v2_resources import (
    GIB,
    LOCAL_CAPACITY_DEFERRAL,
    RESOURCE_CAPACITY_DEFERRED_EXIT_CODE,
    RESOURCE_PRESSURE_EXIT_CODE,
    CleanV2ResourcePressure,
    JobMemorySnapshot,
    MemorySnapshot,
    ProcessIdentity,
    ProcessMemorySnapshot,
    ResourceReservationLedger,
    ResourceReservationUnavailable,
    RuntimeJsonReadError,
    WindowsWorkerJob,
    estimate_sequence_allocation_bytes,
    evaluate_memory,
    process_identity_matches,
    read_json_object_with_retry,
    recovery_policy_payload,
    reconcile_runtime_status_view,
    write_json_object_atomic,
)
from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.ds24.clean_v2_runtime import MAX_DELL_MODEL_WORKERS
from scripts.local import ds24_clean_v2_family_worker as worker
from scripts.local import ds24_clean_v2_monitor as monitor
from scripts.local import ds24_clean_v2_supervisor as supervisor


def _snapshot(*, physical_gib: int, commit_gib: int) -> MemorySnapshot:
    return MemorySnapshot(
        available_physical_bytes=physical_gib * GIB,
        total_physical_bytes=32 * GIB,
        commit_headroom_bytes=commit_gib * GIB,
        commit_limit_bytes=64 * GIB,
        committed_bytes=(64 - commit_gib) * GIB,
        source="test",
        checked_at_utc="2026-09-26T00:00:00+00:00",
    )


def test_runtime_json_read_retries_only_transient_access_denials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resume_state.json"
    path.write_text('{"metrics_cursor": 375}', encoding="utf-8")
    original_read_text = Path.read_text
    calls = 0

    def transient_read_text(self: Path, *args: object, **kwargs: object) -> str:
        nonlocal calls
        if self == path and calls < 2:
            calls += 1
            raise PermissionError(13, "synthetic sharing denial", str(self))
        return original_read_text(self, *args, **kwargs)

    events: list[dict[str, object]] = []
    monkeypatch.setattr(Path, "read_text", transient_read_text)

    payload = read_json_object_with_retry(path, on_retry=events.append)

    assert payload == {"metrics_cursor": 375}
    assert calls == 2
    assert [event["operation"] for event in events] == ["read", "read"]


def test_runtime_json_read_does_not_retry_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resume_state.json"
    path.write_text('{"metrics_cursor":', encoding="utf-8")
    events: list[dict[str, object]] = []

    with pytest.raises(json.JSONDecodeError):
        read_json_object_with_retry(path, on_retry=events.append)

    assert events == []


def test_runtime_json_read_timeout_is_bounded_and_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resume_state.json"
    path.write_text("{}", encoding="utf-8")
    elapsed = 0.0
    original_read_text = Path.read_text

    def denied_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == path:
            raise PermissionError(13, "synthetic persistent denial", str(self))
        return original_read_text(self, *args, **kwargs)

    def monotonic() -> float:
        return elapsed

    def sleep(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    events: list[dict[str, object]] = []
    monkeypatch.setattr(Path, "read_text", denied_read_text)
    monkeypatch.setattr(resources.time, "monotonic", monotonic)
    monkeypatch.setattr(resources.time, "sleep", sleep)

    with pytest.raises(RuntimeJsonReadError, match="did not recover"):
        read_json_object_with_retry(
            path, timeout_seconds=0.025, on_retry=events.append
        )

    assert elapsed == pytest.approx(0.025)
    assert len(events) == 2


def test_atomic_runtime_json_writer_retries_replace_after_closing_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resume_state.json"
    original_replace = resources.os.replace
    replace_calls = 0
    events: list[dict[str, object]] = []

    def transient_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise PermissionError(13, "synthetic destination sharing denial")
        original_replace(source, destination)

    monkeypatch.setattr(resources.os, "replace", transient_replace)

    write_json_object_atomic(
        path, {"metrics_cursor": 375}, on_retry=events.append
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "metrics_cursor": 375
    }
    assert replace_calls == 2
    assert [event["operation"] for event in events] == ["replace"]
    assert list(tmp_path.glob("*.tmp")) == []


def _pending_score_replay_fixture() -> pd.DataFrame:
    timestamp = "2024-01-02T14:35:00+00:00"
    predictions = pd.DataFrame(
        {
            "family": ["random_forest", "random_forest"],
            "decision_timestamp": [timestamp, timestamp],
            "asset_id": ["A", "B"],
            "prediction": [0.2, 0.1],
        }
    )
    return evaluator.build_pending_score_frame_v3(
        predictions,
        family="random_forest",
        metadata={
            "model_hash": "model",
            "model_vintage_id": "vintage",
            "preprocessing_hash": "preprocessing",
            "policy_hash": "policy",
            "training_cutoff": "2024-01-02T14:30:00+00:00",
            "prediction_timestamp": "2024-01-02T14:35:01+00:00",
        },
    )


def test_pending_score_replay_normalises_timestamp_representation() -> None:
    existing = _pending_score_replay_fixture()
    existing["decision_timestamp"] = pd.to_datetime(
        existing["decision_timestamp"], utc=True
    )
    replay = _pending_score_replay_fixture()

    deduplicated = evaluator.deduplicate_pending_scores(
        pd.concat([existing, replay], ignore_index=True)
    )

    assert len(deduplicated) == len(existing)
    assert not deduplicated.duplicated(
        ["family", "decision_timestamp", "asset_id"]
    ).any()


def test_pending_score_replay_conflict_still_fails_closed() -> None:
    existing = _pending_score_replay_fixture()
    replay = _pending_score_replay_fixture()
    replay.loc[0, "score"] = 99.0

    with pytest.raises(ValueError, match="CONFLICTING_DUPLICATE_PENDING"):
        evaluator.deduplicate_pending_scores(
            pd.concat([existing, replay], ignore_index=True)
        )


def test_random_forest_checkpoint_replay_preserves_preprocessing_identity() -> None:
    predictors = ["feature_a", "feature_b"]
    config = {"preprocessing": ["SimpleImputer(strategy='median')"]}
    fitted_identity = stable_hash(
        {
            "family": "random_forest",
            "predictors": predictors,
            "config": config["preprocessing"],
            "kind": "tabular",
        }
    )

    resumed_identity = worker._preprocessing_identity_hash(
        family="random_forest",
        predictors=predictors,
        config=config,
    )

    assert resumed_identity == fitted_identity
    assert resumed_identity != stable_hash(
        {
            "family": "random_forest",
            "predictors": predictors,
            "config": config["preprocessing"],
            "kind": "resumed_model",
        }
    )


def test_pending_score_exact_replay_collapses_once() -> None:
    existing = _pending_score_replay_fixture()

    deduplicated = evaluator.deduplicate_pending_scores(
        pd.concat([existing, existing.copy()], ignore_index=True)
    )

    assert len(deduplicated) == len(existing)


def test_pending_score_utc_and_scalar_representations_canonicalise() -> None:
    existing = _pending_score_replay_fixture()
    replay = _pending_score_replay_fixture()
    replay["decision_timestamp"] = pd.to_datetime(
        replay["decision_timestamp"], utc=True
    ).dt.tz_convert("America/New_York")
    replay["training_cutoff"] = pd.to_datetime(
        replay["training_cutoff"], utc=True
    )
    replay["expected_target_available_timestamp"] = pd.to_datetime(
        replay["expected_target_available_timestamp"], utc=True
    )
    replay["score"] = replay["score"].map(str)
    replay["score_rank"] = replay["score_rank"].map(str)
    replay["eligible"] = "1"
    replay["target_horizon_minutes"] = "60"

    deduplicated = evaluator.deduplicate_pending_scores(
        pd.concat([existing, replay], ignore_index=True)
    )

    assert len(deduplicated) == len(existing)


@pytest.mark.parametrize(
    "field",
    [
        "model_vintage_id",
        "model_hash",
        "preprocessing_hash",
        "policy_hash",
        "target_id",
        "evaluation_contract_hash",
    ],
)
def test_pending_score_conflicting_scientific_provenance_fails_closed(
    field: str,
) -> None:
    existing = _pending_score_replay_fixture()
    replay = _pending_score_replay_fixture()
    replay.loc[0, field] = "conflicting-provenance"

    with pytest.raises(ValueError, match="CONFLICTING_DUPLICATE_PENDING"):
        evaluator.deduplicate_pending_scores(
            pd.concat([existing, replay], ignore_index=True)
        )


def test_previous_attempt_cannot_override_identical_current_replay() -> None:
    previous = _pending_score_replay_fixture()
    previous["attempt_generation"] = "100"
    previous["clean_source_hash"] = "previous-source"
    current = _pending_score_replay_fixture()
    current["attempt_generation"] = "101"
    current["clean_source_hash"] = "current-source"
    current["prediction_timestamp"] = "2024-01-02T14:36:00+00:00"

    deduplicated = evaluator.deduplicate_pending_scores(
        pd.concat([current, previous], ignore_index=True)
    )

    assert set(deduplicated["attempt_generation"]) == {"101"}
    assert set(deduplicated["clean_source_hash"]) == {"current-source"}


def test_pending_score_malformed_canonical_payload_fails_closed() -> None:
    existing = _pending_score_replay_fixture()
    replay = _pending_score_replay_fixture()
    replay["eligible"] = replay["eligible"].astype(object)
    replay.loc[0, "eligible"] = "not-a-boolean"

    with pytest.raises(ValueError, match="INVALID_PENDING_SCORE_BOOLEAN"):
        evaluator.deduplicate_pending_scores(
            pd.concat([existing, replay], ignore_index=True)
        )


def test_completed_refit_package_is_not_replayed() -> None:
    timestamps = [
        pd.Timestamp("2016-02-18T15:10:00Z"),
        pd.Timestamp("2016-02-18T15:15:00Z"),
    ]

    assert worker._package_score_is_complete(
        timestamps, {timestamp.isoformat() for timestamp in timestamps}
    )
    assert not worker._package_score_is_complete(
        timestamps, {timestamps[0].isoformat()}
    )


def test_recovery_policy_requires_three_dell_workers_and_both_memory_reserves() -> None:
    assert MAX_DELL_MODEL_WORKERS == 3
    policy = recovery_policy_payload()
    assert policy["automatic_retry_allowed"] is False
    assert policy["target_available_physical_gib"] == 5
    assert policy["warning_available_physical_gib"] == 6
    assert policy["emergency_available_physical_gib"] == 4
    assert policy["readmission_available_physical_gib"] == 7
    assert policy["system_commit_headroom_gib"] == 12
    assert policy["windows_job_aggregate_commit_limit_gib"] == 24
    assert policy["allocation_safety_factor"] == 1.5

    physical_pressure = evaluate_memory(
        stage="panel",
        estimated_allocation_bytes=2 * GIB,
        snapshot=_snapshot(physical_gib=7, commit_gib=30),
    )
    commit_pressure = evaluate_memory(
        stage="tensor",
        estimated_allocation_bytes=2 * GIB,
        snapshot=_snapshot(physical_gib=30, commit_gib=14),
    )

    assert physical_pressure.safe is False
    assert physical_pressure.blocking_reasons == (
        "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE",
    )
    assert commit_pressure.safe is False
    assert commit_pressure.blocking_reasons == (
        "PROJECTED_SYSTEM_COMMIT_HEADROOM_BELOW_12_GIB_GUARD",
    )


def test_admission_warning_emergency_and_readmission_thresholds() -> None:
    warning = evaluate_memory(
        stage="observe",
        estimated_allocation_bytes=0,
        snapshot=_snapshot(physical_gib=5, commit_gib=30),
        purpose="observation",
    )
    readmission = evaluate_memory(
        stage="admit",
        estimated_allocation_bytes=0,
        snapshot=_snapshot(physical_gib=6, commit_gib=30),
        purpose="admission",
    )
    emergency = evaluate_memory(
        stage="observe",
        estimated_allocation_bytes=0,
        snapshot=_snapshot(physical_gib=3, commit_gib=30),
        purpose="observation",
    )

    assert warning.safe is True
    assert warning.warning_reasons == (
        "AVAILABLE_PHYSICAL_MEMORY_BELOW_WARNING_THRESHOLD",
    )
    assert readmission.safe is False
    assert readmission.admission_allowed is False
    assert "AVAILABLE_PHYSICAL_MEMORY_BELOW_7_GIB_READMISSION_FLOOR" in (
        readmission.blocking_reasons
    )
    assert emergency.emergency is True
    assert emergency.pressure_level == "EMERGENCY"


def test_sequence_cap_is_applied_before_window_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = pd.date_range("2024-01-02T14:35:00Z", periods=4, freq="5min")
    panel = pd.DataFrame(
        {
            "asset_id": ["AAA"] * 4 + ["ZZZ"] * 4,
            "decision_timestamp": [*timestamps, *timestamps],
            "feature": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
            "target_value": list(range(8)),
        }
    )
    guarded: list[tuple[str, int]] = []
    monkeypatch.setattr(
        worker,
        "require_memory",
        lambda *, stage, estimated_allocation_bytes: guarded.append(
            (stage, estimated_allocation_bytes)
        ),
    )

    examples, targets, metadata = worker._sequence_examples(
        panel,
        ["feature"],
        sequence_length=2,
        eligible_mask=pd.Series(True, index=panel.index),
        include_targets=True,
        maximum_examples=2,
    )

    assert examples == [[[11.0], [12.0]], [[12.0], [13.0]]]
    assert targets == [6.0, 7.0]
    assert metadata["asset_id"].tolist() == ["ZZZ", "ZZZ"]
    assert guarded[-1] == (
        "sequence_window_and_tensor_materialization",
        estimate_sequence_allocation_bytes(
            example_count=2, sequence_length=2, predictor_count=1
        ),
    )


def test_resume_accepts_new_operational_source_only_on_new_attempt() -> None:
    prior = {
        "attempt_generation": 10,
        "clean_source_hash": "old",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "model_config_hash": "model",
        "completed_refits": ["one"],
        "metrics_cursor": 5,
    }
    authority = {
        "clean_source_hash": "new",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "model_config_hash": "model",
    }

    worker._validate_resume_authority(
        prior=prior, authority=authority, attempt_generation=11
    )
    with pytest.raises(worker.CleanV2WorkerError, match="newer attempt"):
        worker._validate_resume_authority(
            prior=prior, authority=authority, attempt_generation=10
        )
    with pytest.raises(worker.CleanV2WorkerError, match="feature_authority_hash"):
        worker._validate_resume_authority(
            prior=prior,
            authority={**authority, "feature_authority_hash": "changed"},
            attempt_generation=11,
        )


def test_process_identity_requires_creation_time_as_well_as_pid_and_command() -> None:
    identity = ProcessIdentity(
        pid=123,
        alive=True,
        creation_time_utc="2026-09-26T10:00:00+00:00",
        command_line=(
            "python scripts/local/ds24_clean_v2_family_worker.py "
            "--family random_forest"
        ),
    )
    fragments = (
        "ds24_clean_v2_family_worker.py",
        "--family",
        "random_forest",
    )

    assert process_identity_matches(
        identity,
        expected_creation_time="2026-09-26T10:00:00+00:00",
        required_command_fragments=fragments,
    )
    assert not process_identity_matches(
        identity,
        expected_creation_time="2026-09-26T09:00:00+00:00",
        required_command_fragments=fragments,
    )
    assert not process_identity_matches(
        identity,
        expected_creation_time="",
        required_command_fragments=fragments,
    )


def test_stale_supervisor_lease_is_archived_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    lease_path = tmp_path / "supervisor_lease_dell.json"
    lease = {
        "pid": 123,
        "process_creation_time_utc": "2026-09-26T10:00:00+00:00",
        "generation": 9,
        "evidence": "preserve",
    }
    lease_path.write_text(json.dumps(lease), encoding="utf-8")

    archived = supervisor._archive_stale_lease(lease_path, lease)

    assert not lease_path.exists()
    assert json.loads(archived.read_text(encoding="utf-8")) == lease
    assert archived.parent == tmp_path / "stale_leases"


def test_release_does_not_remove_a_replacement_process_lease(
    tmp_path: Path,
) -> None:
    lease_path = tmp_path / "supervisor_lease_dell.json"
    replacement = {
        "pid": os.getpid(),
        "process_creation_time_utc": "2026-09-26T11:00:00+00:00",
    }
    lease_path.write_text(json.dumps(replacement), encoding="utf-8")
    old_identity = ProcessIdentity(
        pid=os.getpid(),
        alive=True,
        creation_time_utc="2026-09-26T10:00:00+00:00",
        command_line="supervisor",
    )

    supervisor._release_owned_lease(lease_path, old_identity)

    assert lease_path.is_file()


def test_signed_admission_rejects_a_display_only_worker_limit_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    report = {
        "ready": True,
        "blocking_reasons": [],
        "run_id": "run",
        "host_role": "dell",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "target_contract_hash": "target-contract",
        "static_authority_bundle_sha256": "bundle",
        "clean_source_hash": "source",
        "refit_policy": "refit",
        "launch_families": ["random_forest"],
        "maximum_model_workers": 3,
        "resource_policy": recovery_policy_payload(),
    }
    monkeypatch.setattr(supervisor, "preflight", lambda _host: report)
    admission_core = {
        key: report[key]
        for key in (
            "run_id",
            "host_role",
            "feature_authority_hash",
            "target_authority_hash",
            "target_contract_hash",
            "static_authority_bundle_sha256",
            "clean_source_hash",
            "refit_policy",
            "launch_families",
            "maximum_model_workers",
            "resource_policy",
        )
    }
    admission_core["maximum_model_workers"] = 1
    admission = {
        **admission_core,
        "admission_token": stable_hash(admission_core),
    }
    path = tmp_path / "manual_admission_dell.json"
    path.write_text(json.dumps(admission), encoding="utf-8")

    with pytest.raises(RuntimeError, match="maximum_model_workers"):
        supervisor._validated_admission("dell")


def test_live_child_admission_validation_never_calls_stopped_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    report = {
        "ready": True,
        "blocking_reasons": [],
        "run_id": "run",
        "host_role": "dell",
        "feature_authority_hash": "feature",
        "target_authority_hash": "target",
        "target_contract_hash": "target-contract",
        "static_authority_bundle_sha256": "bundle",
        "clean_source_hash": "source",
        "refit_policy": "refit",
        "launch_families": ["random_forest"],
        "maximum_model_workers": 3,
        "resource_policy": recovery_policy_payload(),
    }
    admission_core = {
        key: report[key]
        for key in (
            "run_id",
            "host_role",
            "feature_authority_hash",
            "target_authority_hash",
            "target_contract_hash",
            "static_authority_bundle_sha256",
            "clean_source_hash",
            "refit_policy",
            "launch_families",
            "maximum_model_workers",
            "resource_policy",
        )
    }
    admission = {
        **admission_core,
        "admission_token": stable_hash(admission_core),
    }
    admission_path = tmp_path / "manual_admission_dell.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    before = admission_path.read_bytes()
    monkeypatch.setattr(supervisor, "_runtime_preflight", lambda _host: report)

    def stopped_preflight_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stopped-state preflight called by live child")

    monkeypatch.setattr(supervisor, "preflight", stopped_preflight_must_not_run)

    validated = supervisor._validated_runtime_admission("dell")

    assert validated["admission_token"] == admission["admission_token"]
    assert admission_path.read_bytes() == before


def test_stopped_preflight_archives_dead_lease_before_control_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "RUN_ROOT", run_root)
    lease_path = run_root / "supervisor_lease_dell.json"
    lease_path.write_text(
        json.dumps(
            {
                "pid": 99_991,
                "generation": 41,
                "process_creation_time_utc": "2026-09-28T12:32:16+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "supervisor_status_dell.json").write_text(
        json.dumps(
            {
                "classification": "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_STARTING",
                "attempt_generation": 41,
                "supervisor_pid": 99_991,
                "supervisor_process_creation_time_utc": (
                    "2026-09-28T12:32:16+00:00"
                ),
                "active_workers": [],
                "failed_families": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: ProcessIdentity(
            pid=pid,
            alive=False,
            creation_time_utc=None,
            command_line="",
        ),
    )

    result = supervisor._prepare_stopped_runtime_for_preflight("dell")

    assert result["stopped"] is True
    assert result["classification"] == "STALE_SUPERVISOR_LEASE_ARCHIVED"
    assert not lease_path.exists()
    archive = tmp_path / str(result["archived_lease_path"])
    assert archive.is_file()
    reconciled = json.loads(
        (run_root / "supervisor_status_dell.json").read_text(encoding="utf-8")
    )
    assert reconciled["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_STALE_RUNTIME_RECONCILED"
    )


def test_stopped_preflight_never_archives_a_live_supervisor_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "RUN_ROOT", run_root)
    lease_path = run_root / "supervisor_lease_dell.json"
    creation = "2026-09-28T12:32:16+00:00"
    lease_path.write_text(
        json.dumps(
            {
                "pid": 42,
                "generation": 43,
                "process_creation_time_utc": creation,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: ProcessIdentity(
            pid=pid,
            alive=True,
            creation_time_utc=creation,
            command_line=(
                "python ds24_clean_v2_supervisor.py --run-queue --host dell"
            ),
        ),
    )

    result = supervisor._prepare_stopped_runtime_for_preflight("dell")

    assert result["stopped"] is False
    assert result["identity_verified"] is True
    assert lease_path.is_file()


def test_detached_child_records_early_exit_and_releases_only_its_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    generation = 44
    creation = "2026-09-28T12:32:16+00:00"
    identity = ProcessIdentity(
        pid=os.getpid(),
        alive=True,
        creation_time_utc=creation,
        command_line="python ds24_clean_v2_supervisor.py --run-queue --host dell",
    )
    status_path = tmp_path / "supervisor_status_dell.json"
    status_path.write_text(
        json.dumps(
            {
                "attempt_generation": generation,
                "supervisor_pid": identity.pid,
                "supervisor_process_creation_time_utc": creation,
                "classification": "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_STARTING",
                "failed_families": {},
            }
        ),
        encoding="utf-8",
    )
    lease_path = tmp_path / "supervisor_lease_dell.json"
    lease_path.write_text(
        json.dumps(
            {
                "pid": identity.pid,
                "process_creation_time_utc": creation,
                "generation": generation,
                "admission_token": "token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "process_identity", lambda _pid: identity)
    monkeypatch.setattr(
        supervisor,
        "_await_parent_startup_record",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        supervisor,
        "run_queue",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic early startup failure")
        ),
    )

    exit_code = supervisor._run_queue_entrypoint(
        "dell", "token", attempt_generation=generation
    )

    recorded = json.loads(status_path.read_text(encoding="utf-8"))
    assert exit_code == 5
    assert recorded["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_EXIT_FAILED_CLOSED"
    )
    assert recorded["supervisor_exit_error"] == "synthetic early startup failure"
    assert recorded["failed_families"] == {}
    assert recorded["automatic_retry"] is False
    assert not lease_path.exists()


def test_explicit_supervisor_exit_failure_is_not_hidden_as_stale_runtime() -> None:
    reconciled = reconcile_runtime_status_view(
        {
            "classification": (
                "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_EXIT_FAILED_CLOSED"
            ),
            "supervisor_pid": 999_999_970,
            "supervisor_process_creation_time_utc": (
                "2026-09-28T12:32:16+00:00"
            ),
            "active_workers": [],
            "failed_families": {},
        },
        host="dell",
    )

    assert reconciled["supervisor_identity_verified"] is False
    assert reconciled["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_EXIT_FAILED_CLOSED"
    )


def test_preflight_archives_terminal_attempt_status_before_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path / "run")
    status_path = supervisor.RUN_ROOT / "supervisor_status_dell.json"
    status_path.parent.mkdir(parents=True)
    terminal_status = {
        "run_id": supervisor.RUN_ID,
        "host_role": "dell",
        "attempt_generation": 20,
        "classification": "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED",
        "failed_families": {"momentum": 3221225786},
    }
    status_path.write_text(json.dumps(terminal_status), encoding="utf-8")

    supervisor._publish_preflight_status(
        "dell",
        {
            "run_id": supervisor.RUN_ID,
            "host_role": "dell",
            "classification": "DS24_CLEAN_V2_TOURNAMENT_PREFLIGHT_PASS",
            "failed_families": {},
            "ready": True,
        },
    )

    published = json.loads(status_path.read_text(encoding="utf-8"))
    evidence_path = tmp_path / published[
        "pre_preflight_terminal_status_evidence_path"
    ]
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == terminal_status
    assert published["failed_families"] == {}
    assert published["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_PREFLIGHT_PASS"
    )


def test_resource_limit_pause_preserves_last_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    family_root = (
        tmp_path
        / "research_runs"
        / "ds24_clean_v2"
        / worker.RUN_ID
        / "family=random_forest"
    )
    family_root.mkdir(parents=True)
    prior = {
        "run_id": worker.RUN_ID,
        "family": "random_forest",
        "owner_host": "dell",
        "attempt_generation": 5,
        "completed_refits": ["2026-01-01T00:00:00+00:00"],
        "metrics_cursor": 11,
        "terminal_state": "RUNNING",
    }
    (family_root / "resume_state.json").write_text(
        json.dumps(prior), encoding="utf-8"
    )
    decision = evaluate_memory(
        stage="fit",
        estimated_allocation_bytes=0,
        snapshot=_snapshot(physical_gib=3, commit_gib=30),
    )

    evidence = worker._record_resource_pause(
        SimpleNamespace(
            family="random_forest", host="dell", resume_generation=5
        ),
        CleanV2ResourcePressure(decision),
        classification="DS24_CLEAN_V2_PAUSED_RESOURCE_LIMIT",
    )

    committed = json.loads(
        (family_root / "resume_state.json").read_text(encoding="utf-8")
    )
    assert evidence["automatic_retry"] is False
    assert committed["completed_refits"] == prior["completed_refits"]
    assert committed["metrics_cursor"] == 11
    assert committed["terminal_state"] == "PAUSED_RESOURCE_PRESSURE"


def test_worker_exit_disposition_never_requests_automatic_relaunch() -> None:
    normal_job = JobMemorySnapshot(
        committed_bytes=0,
        peak_committed_bytes=0,
        commit_limit_bytes=24 * GIB,
        active_processes=0,
        total_processes=1,
        terminated_processes=0,
        limit_violation_detected=False,
        installation_verified=True,
        checked_at_utc="2026-09-27T00:00:00+00:00",
    )
    limited_job = JobMemorySnapshot(
        **{
            **normal_job.payload(),
            "limit_violation_detected": True,
        }
    )

    assert supervisor._worker_exit_disposition(0, normal_job) == "COMPLETE"
    assert supervisor._worker_exit_disposition(
        RESOURCE_PRESSURE_EXIT_CODE, normal_job
    ) == "RESOURCE_LIMIT_NO_AUTOMATIC_RETRY"
    assert supervisor._worker_exit_disposition(
        1, limited_job
    ) == "RESOURCE_LIMIT_NO_AUTOMATIC_RETRY"
    assert supervisor._worker_exit_disposition(
        1, normal_job
    ) == "FAILED_CLOSED_NO_AUTOMATIC_RETRY"


def test_unexpected_worker_exit_records_current_attempt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    family_root = tmp_path / "family=momentum"
    family_root.mkdir(parents=True)
    (family_root / "resume_state.json").write_text(
        json.dumps(
            {
                "family": "momentum",
                "attempt_generation": 19,
                "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
            }
        ),
        encoding="utf-8",
    )
    (family_root / "worker_failure.json").write_text(
        json.dumps(
            {
                "family": "momentum",
                "host": "dell",
                "attempt_generation": 10,
                "error": "historical scientific failure",
            }
        ),
        encoding="utf-8",
    )
    job_snapshot = JobMemorySnapshot(
        committed_bytes=1,
        peak_committed_bytes=2,
        commit_limit_bytes=24 * GIB,
        active_processes=0,
        total_processes=1,
        terminated_processes=1,
        limit_violation_detected=False,
        installation_verified=True,
        checked_at_utc="2026-09-28T14:55:20+00:00",
    )

    evidence = supervisor._record_unexpected_worker_exit(
        family="momentum",
        host="dell",
        attempt_generation=20,
        worker=supervisor.OwnedWorker(
            process=SimpleNamespace(pid=1234),
            process_creation_time_utc="2026-09-28T14:54:55+00:00",
        ),
        exit_code=0xC000013A,
        job_snapshot=job_snapshot,
    )

    immutable = list(family_root.glob("worker_exit_attempt=20_event=*.json"))
    current = json.loads(
        (family_root / "worker_exit.json").read_text(encoding="utf-8")
    )
    assert len(immutable) == 1
    assert current == evidence
    assert evidence["exit_code_hex"] == "0xC000013A"
    assert evidence["exit_reason"] == "EXTERNAL_CONTROL_EVENT"
    assert evidence["current_worker_failure_artifact_recorded"] is False
    assert evidence["worker_failure_artifact_attempt_generation"] == 10
    assert evidence["automatic_retry"] is False


def test_capacity_deferred_exit_is_scheduler_state_not_global_pressure() -> None:
    assert supervisor._worker_exit_disposition(
        RESOURCE_CAPACITY_DEFERRED_EXIT_CODE, None
    ) == "DEFERRED_RESOURCE_CAPACITY"


def test_worker_records_capacity_deferral_without_losing_committed_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    family_root = (
        tmp_path
        / "research_runs"
        / "ds24_clean_v2"
        / worker.RUN_ID
        / "family=random_forest"
    )
    family_root.mkdir(parents=True)
    prior = {
        "run_id": worker.RUN_ID,
        "family": "random_forest",
        "owner_host": "dell",
        "attempt_generation": 17,
        "completed_refits": ["2016-02-02T14:35:00+00:00"],
        "metrics_cursor": 322,
        "terminal_state": "RUNNING",
    }
    (family_root / "resume_state.json").write_text(
        json.dumps(prior), encoding="utf-8"
    )
    response = {
        "classification": LOCAL_CAPACITY_DEFERRAL,
        "blocking_reasons": [
            "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE"
        ],
    }

    evidence = worker._record_capacity_deferral(
        SimpleNamespace(
            family="random_forest", host="dell", resume_generation=17
        ),
        ResourceReservationUnavailable("panel_assembly", response),
    )

    state = json.loads(
        (family_root / "resume_state.json").read_text(encoding="utf-8")
    )
    assert evidence["automatic_retry"] is False
    assert state["terminal_state"] == "DEFERRED_RESOURCE_CAPACITY"
    assert state["completed_refits"] == prior["completed_refits"]
    assert state["metrics_cursor"] == 322


def test_actual_emergency_pressure_remains_a_containment_condition() -> None:
    decision = supervisor._actual_pressure_decision(
        stage="emergency-test",
        memory_snapshot=_snapshot(physical_gib=3, commit_gib=30),
        job_snapshot=None,
    )

    assert decision.safe is False
    assert decision.emergency is True
    assert "AVAILABLE_PHYSICAL_MEMORY_BELOW_4_GIB_EMERGENCY" in (
        decision.blocking_reasons
    )


def test_dead_runtime_status_reclaims_stale_worker_and_reservation() -> None:
    status = {
        "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
        "supervisor_pid": 999_999_991,
        "supervisor_process_creation_time_utc": "2026-09-27T10:00:00+00:00",
        "active_workers": [
            {
                "family": "random_forest",
                "pid": 999_999_992,
                "process_creation_time_utc": "2026-09-27T10:01:00+00:00",
            }
        ],
        "worker_reservations": {
            "active_worker_reservations": 1,
            "outstanding": {"physical_bytes": GIB, "commit_bytes": GIB},
            "workers": [
                {
                    "owner": {
                        "family": "random_forest",
                        "attempt_generation": 1,
                        "pid": 999_999_992,
                        "process_creation_time_utc": (
                            "2026-09-27T10:01:00+00:00"
                        ),
                    },
                    "outstanding": {
                        "physical_bytes": GIB,
                        "commit_bytes": GIB,
                    },
                }
            ],
        },
    }

    reconciled = reconcile_runtime_status_view(status, host="dell")

    assert reconciled["supervisor_identity_verified"] is False
    assert reconciled["active_workers"] == []
    assert len(reconciled["stale_worker_records_reconciled"]) == 1
    assert len(reconciled["stale_reservations_reclaimed"]) == 1
    assert reconciled["worker_reservations"]["active_worker_reservations"] == 0
    assert reconciled["worker_reservations"]["outstanding"] == {
        "physical_bytes": 0,
        "commit_bytes": 0,
    }


def test_dead_supervisor_preserves_intentional_stopped_resumable_status() -> None:
    status = {
        "classification": "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE",
        "supervisor_pid": 999_999_991,
        "supervisor_process_creation_time_utc": "2026-09-27T10:00:00+00:00",
        "active_workers": [],
        "worker_reservations": {
            "active_worker_reservations": 0,
            "outstanding": {"physical_bytes": 0, "commit_bytes": 0},
            "workers": [],
        },
    }

    reconciled = reconcile_runtime_status_view(status, host="dell")

    assert reconciled["supervisor_identity_verified"] is False
    assert reconciled["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE"
    )
    assert reconciled["active_workers"] == []
    assert reconciled["runtime_reconciliation_required"] is False


def test_monitor_never_reports_running_for_nonexistent_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    family_root = run_root / "family=random_forest"
    family_root.mkdir(parents=True)
    (run_root / "supervisor_status_dell.json").write_text(
        json.dumps(
            {
                "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
                "attempt_generation": 21,
                "supervisor_pid": 999_999_981,
                "supervisor_process_creation_time_utc": (
                    "2026-09-27T10:00:00+00:00"
                ),
                "heartbeat_utc": "2026-09-27T10:02:00+00:00",
                "active_workers": [
                    {
                        "family": "random_forest",
                        "attempt_generation": 21,
                        "pid": 999_999_982,
                        "process_creation_time_utc": (
                            "2026-09-27T10:01:00+00:00"
                        ),
                    }
                ],
                "worker_reservations": {
                    "active_worker_reservations": 1,
                    "workers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (family_root / "resume_state.json").write_text(
        json.dumps(
            {
                "run_id": worker.RUN_ID,
                "family": "random_forest",
                "owner_host": "dell",
                "attempt_generation": 21,
                "terminal_state": "RUNNING",
                "completed_refits": ["2016-02-02T14:35:00+00:00"],
                "metrics_cursor": 322,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        monitor,
        "load_contract",
        lambda _name: {"hosts": {"dell": ["random_forest"]}},
    )

    report = monitor.build_report(
        "dell", repository_root=tmp_path, run_root=run_root
    )

    assert report["supervisor_process_alive_verified"] is False
    assert report["active_model_worker_count"] == 0
    assert report["families"][0]["state"] == "RESUMABLE_STALE_PROCESS"
    assert report["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_STALE_RUNTIME_RECONCILED"
    )


def test_local_worker_deferral_keeps_other_worker_and_supervisor_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(physical_gib=40, commit_gib=40)
    job_snapshot = JobMemorySnapshot(
        committed_bytes=0,
        peak_committed_bytes=0,
        commit_limit_bytes=24 * GIB,
        active_processes=2,
        total_processes=2,
        terminated_processes=0,
        limit_violation_detected=False,
        installation_verified=True,
        checked_at_utc="2026-09-27T00:00:00+00:00",
    )

    class FakeProcess:
        def __init__(self, pid: int, exit_code: int | None) -> None:
            self.pid = pid
            self.exit_code = exit_code

        def poll(self) -> int | None:
            return self.exit_code

    class FakeContainment:
        installation_verified = True

        def __init__(self) -> None:
            self.launches = 0
            self.closed = False

        def launch(self, command: list[str], **kwargs: object) -> FakeProcess:
            self.launches += 1
            process = FakeProcess(
                50_000 + self.launches,
                (
                    RESOURCE_CAPACITY_DEFERRED_EXIT_CODE
                    if command[0] == "worker-b"
                    else None
                ),
            )
            before_resume = kwargs.get("before_resume")
            if callable(before_resume):
                before_resume(process.pid)
            return process

        @staticmethod
        def snapshot() -> JobMemorySnapshot:
            return job_snapshot

        @staticmethod
        def contains_pid(_pid: int) -> bool:
            return False

        def close(self) -> None:
            self.closed = True

    containment = FakeContainment()
    tree = ProcessMemorySnapshot(
        pid=0,
        resident_bytes=100 * 1024**2,
        peak_resident_bytes=100 * 1024**2,
        private_commit_bytes=120 * 1024**2,
        peak_private_commit_bytes=120 * 1024**2,
        source="bounded-test",
        checked_at_utc="2026-09-27T00:00:00+00:00",
    )
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    def live_child_must_not_reconcile_stale_runtime(
        *_args: object, **_kwargs: object
    ) -> None:
        raise AssertionError("live child invoked stopped-runtime reconciliation")

    monkeypatch.setattr(
        supervisor,
        "_reconcile_persisted_runtime",
        live_child_must_not_reconcile_stale_runtime,
    )
    monkeypatch.setattr(
        supervisor,
        "_validated_runtime_admission",
        lambda _host: {"admission_token": "token", "maximum_model_workers": 3},
    )
    monkeypatch.setattr(supervisor, "_launch_families", lambda _host: ["a", "b"])
    monkeypatch.setattr(
        supervisor,
        "load_contract",
        lambda _name: {
            "worker_commands": {"dell": {"a": ["worker-a"], "b": ["worker-b"]}}
        },
    )
    monkeypatch.setattr(
        supervisor, "_resource_snapshot", lambda: (100 * GIB, snapshot)
    )
    monkeypatch.setattr(
        supervisor, "create_worker_containment", lambda *, host: containment
    )
    monkeypatch.setattr(
        supervisor,
        "ResourceReservationLedger",
        lambda *, maximum_workers, job_snapshot_provider: ResourceReservationLedger(
            maximum_workers=maximum_workers,
            memory_snapshot_provider=lambda: snapshot,
            job_snapshot_provider=job_snapshot_provider,
            process_tree_snapshot_provider=lambda pid: ProcessMemorySnapshot(
                **{**tree.payload(), "pid": pid}
            ),
        ),
    )
    monkeypatch.setattr(
        ResourceReservationLedger, "reconcile_dead_workers", lambda self: ()
    )
    monkeypatch.setattr(
        supervisor,
        "process_identity",
        lambda pid: ProcessIdentity(
            pid=pid,
            alive=True,
            creation_time_utc="2026-09-27T10:00:00+00:00",
            command_line="ds24_clean_v2_family_worker.py --family",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_worker_identity",
        lambda _family, process: supervisor.OwnedWorker(
            process=process,
            process_creation_time_utc="2026-09-27T10:00:00+00:00",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_family_state",
        lambda family: {
            "terminal_state": (
                "DEFERRED_RESOURCE_CAPACITY" if family == "b" else "RUNNING"
            ),
            "resource_decision": {
                "classification": LOCAL_CAPACITY_DEFERRAL,
                "blocking_reasons": [
                    "PROJECTED_AVAILABLE_PHYSICAL_MEMORY_BELOW_5_GIB_RESERVE"
                ],
            },
        },
    )
    terminated: list[list[str]] = []

    def terminate(active: dict[str, object], **_kwargs: object) -> dict[str, dict[str, object]]:
        terminated.append(sorted(active))
        return {}

    monkeypatch.setattr(supervisor, "_terminate_owned_workers", terminate)
    written_statuses: list[dict[str, object]] = []
    original_write = supervisor._write_json_atomic

    def capture_write(path: Path, payload: object) -> None:
        if path.name == "supervisor_status_dell.json" and isinstance(payload, dict):
            written_statuses.append(payload)
        original_write(path, payload)

    monkeypatch.setattr(supervisor, "_write_json_atomic", capture_write)
    sleep_calls = 0

    def bounded_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            (tmp_path / "stop_request_dell.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(supervisor.time, "sleep", bounded_sleep)

    exit_code = supervisor.run_queue("dell", "token", attempt_generation=22)

    active_after_deferral = [
        row
        for row in written_statuses
        if row.get("classification") == "DS24_CLEAN_V2_TOURNAMENT_RUNNING"
        and [worker_row["family"] for worker_row in row.get("active_workers", [])]
        == ["a"]
        and "b" in row.get("deferred_resource_families", {})
    ]
    assert exit_code == 0
    assert written_statuses[0]["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_SUPERVISOR_RUNTIME_INITIALIZED"
    )
    assert written_statuses[0]["active_workers"] == []
    assert active_after_deferral
    assert terminated == [["a"]]
    assert containment.launches == 2
    assert containment.closed is True
    assert not (tmp_path / "stop_request_dell.json").exists()


def test_persisted_reconciliation_archives_evidence_and_preserves_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    family_root = run_root / "family=random_forest"
    family_root.mkdir(parents=True)
    status_path = run_root / "supervisor_status_dell.json"
    state_path = family_root / "resume_state.json"
    status_path.write_text(
        json.dumps(
            {
                "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
                "attempt_generation": 30,
                "supervisor_pid": 999_999_971,
                "supervisor_process_creation_time_utc": (
                    "2026-09-27T10:00:00+00:00"
                ),
                "active_workers": [
                    {
                        "family": "random_forest",
                        "pid": 999_999_972,
                        "attempt_generation": 30,
                        "process_creation_time_utc": (
                            "2026-09-27T10:01:00+00:00"
                        ),
                    }
                ],
                "worker_reservations": {
                    "active_worker_reservations": 1,
                    "outstanding": {"physical_bytes": GIB, "commit_bytes": GIB},
                    "workers": [],
                },
                "queued_families": [],
            }
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "run_id": worker.RUN_ID,
                "family": "random_forest",
                "owner_host": "dell",
                "attempt_generation": 30,
                "terminal_state": "RUNNING",
                "completed_refits": ["2016-02-02T14:35:00+00:00"],
                "metrics_cursor": 322,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "RUN_ROOT", run_root)

    report = supervisor._reconcile_persisted_runtime("dell", persist=True)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert report["worker_reservations"]["active_worker_reservations"] == 0
    assert state["terminal_state"] == "RESUMABLE_STALE_PROCESS"
    assert state["completed_refits"] == ["2016-02-02T14:35:00+00:00"]
    assert state["metrics_cursor"] == 322
    assert list((run_root / "runtime_reconciliation").glob("*.before.json"))


def test_supervisor_resource_exit_does_not_relaunch_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(physical_gib=30, commit_gib=30)
    job_snapshot = JobMemorySnapshot(
        committed_bytes=0,
        peak_committed_bytes=0,
        commit_limit_bytes=24 * GIB,
        active_processes=0,
        total_processes=1,
        terminated_processes=0,
        limit_violation_detected=False,
        installation_verified=True,
        checked_at_utc="2026-09-27T00:00:00+00:00",
    )

    class FakeProcess:
        pid = os.getpid()

        @staticmethod
        def poll() -> int:
            return RESOURCE_PRESSURE_EXIT_CODE

    class FakeContainment:
        installation_verified = True

        def __init__(self) -> None:
            self.launches = 0
            self.closed = False

        def launch(self, *_args: object, **_kwargs: object) -> FakeProcess:
            self.launches += 1
            process = FakeProcess()
            before_resume = _kwargs.get("before_resume")
            if callable(before_resume):
                before_resume(process.pid)
            return process

        @staticmethod
        def snapshot() -> JobMemorySnapshot:
            return job_snapshot

        @staticmethod
        def contains_pid(_pid: int) -> bool:
            return False

        def close(self) -> None:
            self.closed = True

    containment = FakeContainment()
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(
        supervisor,
        "_validated_runtime_admission",
        lambda _host: {
            "admission_token": "token",
            "maximum_model_workers": 3,
        },
    )
    monkeypatch.setattr(supervisor, "_launch_families", lambda _host: ["random_forest"])
    monkeypatch.setattr(
        supervisor,
        "load_contract",
        lambda _name: {
            "worker_commands": {"dell": {"random_forest": ["worker"]}}
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_resource_snapshot",
        lambda: (100 * GIB, snapshot),
    )
    monkeypatch.setattr(
        supervisor,
        "create_worker_containment",
        lambda *, host: containment,
    )
    monkeypatch.setattr(
        supervisor,
        "_worker_identity",
        lambda _family, process: supervisor.OwnedWorker(
            process=process,
            process_creation_time_utc=supervisor.process_identity(
                process.pid
            ).creation_time_utc,
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_family_state",
        lambda _family: {
            "terminal_state": "PAUSED_RESOURCE_PRESSURE",
            "automatic_retry": False,
        },
    )
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)

    exit_code = supervisor.run_queue(
        "dell", "token", attempt_generation=123
    )

    assert exit_code == 3
    assert containment.launches == 1
    assert containment.closed is True


FIXTURE = Path(__file__).parent / "fixtures" / "ds24_job_memory_child.py"


def _wait_for_pid_file(path: Path, *, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return int(path.read_text(encoding="ascii"))
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_job_contains_worker_descendants_before_they_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_file = tmp_path / "descendant.pid"
    job = WindowsWorkerJob(commit_limit_bytes=256 * 1024**2)
    original_resume = job._resume_suspended_process
    membership_before_resume: list[bool] = []

    def resume_after_asserting_membership(pid: int) -> None:
        membership_before_resume.append(job.contains_pid(pid))
        original_resume(pid)

    monkeypatch.setattr(job, "_resume_suspended_process", resume_after_asserting_membership)
    parent = job.launch(
        [
            sys.executable,
            str(FIXTURE),
            "--mode",
            "spawn-descendant",
            "--pid-file",
            str(pid_file),
        ],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    try:
        descendant_pid = _wait_for_pid_file(pid_file)
        assert membership_before_resume == [True]
        assert job.installation_verified is True
        assert job.contains_pid(parent.pid)
        assert job.contains_pid(descendant_pid)
        assert job.snapshot().active_processes >= 2
    finally:
        job.close()
        parent.wait(timeout=10.0)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_job_enforces_small_aggregate_commit_limit() -> None:
    limit_bytes = 64 * 1024**2
    job = WindowsWorkerJob(commit_limit_bytes=limit_bytes)
    child = job.launch(
        [sys.executable, str(FIXTURE), "--mode", "allocate", "--chunk-mib", "4"],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    try:
        assert child.wait(timeout=20.0) == RESOURCE_PRESSURE_EXIT_CODE
        snapshot = job.snapshot()
        assert snapshot.limit_violation_detected is True
        assert snapshot.commit_limit_bytes == limit_bytes
        assert snapshot.peak_committed_bytes >= limit_bytes - 4 * 1024**2
    finally:
        job.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_supervisor_owned_job_kills_worker_when_owner_exits() -> None:
    import psutil

    owner = subprocess.Popen(  # noqa: S603 - bounded test fixture
        [sys.executable, str(FIXTURE), "--mode", "own-job"],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    child_pid = 0
    try:
        assert owner.stdout is not None
        child_pid = int(json.loads(owner.stdout.readline())["child_pid"])
        assert psutil.pid_exists(child_pid)
        owner.terminate()
        owner.wait(timeout=10.0)
        deadline = time.monotonic() + 10.0
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not psutil.pid_exists(child_pid)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=10.0)
        if child_pid and psutil.pid_exists(child_pid):
            psutil.Process(child_pid).kill()
