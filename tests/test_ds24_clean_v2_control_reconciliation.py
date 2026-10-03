from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.ds24.clean_v2_checkpoint_compatibility import (
    CLEAN_V2_FEATURE_AUTHORITY_HASH,
    CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH,
    CLEAN_V2_TARGET_AUTHORITY_HASH,
)
from core.research.ml.ds24.clean_v2_contracts import load_contract
from core.research.ml.ds24.clean_v2_runtime import RUN_ID
from scripts.local import ds24_clean_v2_family_worker as worker
from scripts.local import ds24_clean_v2_supervisor as supervisor
from scripts.local.ds24_clean_v2_reconcile_failures import (
    FailureReconciliationError,
    ensure_zero_progress_control_resume_state,
    reconcile_registered_control_resume_states,
    reconcile_registered_predictor_resume_states,
    reconcile_legacy_zero_progress_predictor_resume_state,
    reconcile_zero_progress_no_model_control,
    validate_registered_control_resume_states,
)


FAMILY = "equal_weight_no_model"
HOST = "dell"
OLD_GENERATION = 1790595972810150400


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(
    root: Path,
    *,
    family: str = FAMILY,
    feature_hash: str | None = "v1-feature-authority",
    completed_refits: list[str] | None = None,
    metrics_cursor: int = 0,
    other_failures: dict[str, int] | None = None,
    error: str = (
        "CleanV2WorkerError:Resume state authority mismatch: "
        "feature_authority_hash"
    ),
) -> tuple[Path, Path, dict[str, object], bytes]:
    run_root = root / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / f"family={family}"
    state: dict[str, object] = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": HOST,
        "attempt_generation": OLD_GENERATION,
        "attempt_id": "failed-control-attempt",
        "terminal_state": "FAILED_CLOSED",
        "completed_refits": completed_refits or [],
        "metrics_cursor": metrics_cursor,
        "error": error,
        "heartbeat_utc": "2026-09-28T11:47:10+00:00",
    }
    if feature_hash is not None:
        state["feature_authority_hash"] = feature_hash
    failure = {
        "run_id": RUN_ID,
        "family": family,
        "host": HOST,
        "attempt_generation": OLD_GENERATION,
        "attempt_id": "failed-control-attempt",
        "terminal_state": "FAILED_CLOSED",
        "classification": "DS24_CLEAN_V2_WORKER_FAILED_CLOSED",
        "completed_refits": completed_refits or [],
        "metrics_cursor": metrics_cursor,
        "error": error,
        "failed_at_utc": "2026-09-28T11:47:10+00:00",
    }
    _write_json(family_root / "resume_state.json", state)
    _write_json(family_root / "worker_failure.json", failure)
    _write_json(
        family_root / f"worker_failure_attempt={OLD_GENERATION}.json", failure
    )
    _write_json(
        family_root / "resource_deferral.json",
        {
            "run_id": RUN_ID,
            "family": family,
            "attempt_generation": OLD_GENERATION,
            "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
        },
    )
    deferred = {
        "transformer": {
            "family": "transformer",
            "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
        },
        "huber": {
            "family": "huber",
            "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
        },
    }
    failed = {family: 1, **(other_failures or {})}
    _write_json(
        run_root / f"supervisor_status_{HOST}.json",
        {
            "run_id": RUN_ID,
            "host_role": HOST,
            "classification": "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED",
            "attempt_generation": OLD_GENERATION,
            "supervisor_pid": 0,
            "active_workers": [],
            "queued_families": ["random_forest", "transformer", "huber"],
            "failed_families": failed,
            "deferred_resource_families": deferred,
            "resource_paused_families": {},
            "blocking_reasons": ["WORKER_EXIT_NONZERO"],
            "worker_reservations": {"active_worker_reservations": 0},
            "worker_job": {"active_processes": 0},
            "ready": False,
        },
    )
    _write_json(
        run_root / f"manual_admission_{HOST}.json",
        {"run_id": RUN_ID, "clean_source_hash": "stale"},
    )
    random_forest = {
        "run_id": RUN_ID,
        "family": "random_forest",
        "owner_host": HOST,
        "terminal_state": "RUNNING",
        "completed_refits": ["2016-02-02T14:35:00+00:00"],
        "metrics_cursor": 322,
        "latest_scored_decision": "2016-02-09T14:55:00+00:00",
    }
    random_forest_path = run_root / "family=random_forest" / "resume_state.json"
    _write_json(random_forest_path, random_forest)
    return family_root, run_root, deferred, random_forest_path.read_bytes()


def _predictor_fixture(
    root: Path,
    *,
    completed_refits: list[str] | None = None,
    metrics_cursor: int = 0,
    error: str = (
        "CleanV2WorkerError:Resume state authority mismatch: "
        "feature_authority_hash"
    ),
) -> tuple[Path, Path, bytes]:
    family_root, run_root, _, random_forest_before = _fixture(
        root,
        family="ridge_C5",
        feature_hash=None,
        completed_refits=completed_refits,
        metrics_cursor=metrics_cursor,
        error=error,
    )
    state_path = family_root / "resume_state.json"
    state = json.loads(state_path.read_text())
    state["reconciled_from_legacy_resource_pause"] = True
    _write_json(state_path, state)
    return family_root, run_root, random_forest_before


def test_zero_progress_v1_control_is_rejected_then_archived_and_reinitialized(
    tmp_path: Path,
) -> None:
    family_root, _, _, _ = _fixture(tmp_path)
    prior = json.loads((family_root / "resume_state.json").read_text())
    with pytest.raises(worker.CleanV2WorkerError, match="feature_authority_hash"):
        worker._validate_resume_authority(
            prior=prior,
            authority={"feature_authority_hash": CLEAN_V2_FEATURE_AUTHORITY_HASH},
            attempt_generation=OLD_GENERATION + 1,
        )

    report = reconcile_zero_progress_no_model_control(
        family=FAMILY,
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="new-clean-source",
    )
    repaired = json.loads((family_root / "resume_state.json").read_text())
    archive = tmp_path / report["archive_path"]

    assert report["old_feature_authority_hash"] == "v1-feature-authority"
    assert report["committed_work_found"] is False
    assert report["old_state_classification"] == (
        "NON_CLEAN_V2_ZERO_PROGRESS_RESUME_AUTHORITY"
    )
    assert (archive / "resume.json").is_file()
    assert (archive / "failure.json").is_file()
    assert (archive / "manifest.json").is_file()
    assert not (family_root / "worker_failure.json").exists()
    assert (family_root / f"worker_failure_attempt={OLD_GENERATION}.json").is_file()
    assert repaired["terminal_state"] == "READY_FOR_MANUAL_ADMISSION"
    assert repaired["feature_authority_hash"] == CLEAN_V2_FEATURE_AUTHORITY_HASH
    assert repaired["target_authority_hash"] == CLEAN_V2_TARGET_AUTHORITY_HASH
    assert repaired["static_authority_bundle_sha256"] == (
        CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH
    )
    assert repaired["completed_refits"] == []
    assert repaired["completed_scoring_packages"] == []
    assert repaired["metrics_cursor"] == 0


def test_zero_progress_legacy_predictor_placeholder_is_archived_and_removed(
    tmp_path: Path,
) -> None:
    family_root, run_root, random_forest_before = _predictor_fixture(tmp_path)

    report = reconcile_legacy_zero_progress_predictor_resume_state(
        family="ridge_C5",
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    archive = tmp_path / str(report["archive_path"])
    status = json.loads(
        (run_root / f"supervisor_status_{HOST}.json").read_text()
    )

    assert report["changed"] is True
    assert report["committed_work_found"] is False
    assert report["resume_state_removed"] is True
    assert not (family_root / "resume_state.json").exists()
    assert not (family_root / "worker_failure.json").exists()
    assert (family_root / f"worker_failure_attempt={OLD_GENERATION}.json").is_file()
    assert (archive / "resume.json").is_file()
    assert (archive / "failure.json").is_file()
    assert (archive / "manifest.json").is_file()
    assert status["failed_families"] == {}
    assert status["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_PREDICTOR_RECONCILED"
    )
    assert (
        run_root / "family=random_forest" / "resume_state.json"
    ).read_bytes() == random_forest_before


@pytest.mark.parametrize(
    ("completed_refits", "metrics_cursor"),
    [(["2016-02-02T14:35:00+00:00"], 0), ([], 1)],
)
def test_legacy_predictor_placeholder_with_progress_is_never_removed(
    tmp_path: Path,
    completed_refits: list[str],
    metrics_cursor: int,
) -> None:
    family_root, _, _ = _predictor_fixture(
        tmp_path,
        completed_refits=completed_refits,
        metrics_cursor=metrics_cursor,
    )
    before = (family_root / "resume_state.json").read_bytes()

    with pytest.raises(
        FailureReconciliationError,
        match="NONZERO_SCIENTIFIC_PROGRESS",
    ):
        reconcile_legacy_zero_progress_predictor_resume_state(
            family="ridge_C5",
            host=HOST,
            repository_root=tmp_path,
            current_source_hash="current-source",
        )

    assert (family_root / "resume_state.json").read_bytes() == before


def test_genuine_predictor_failure_is_not_treated_as_legacy_migration(
    tmp_path: Path,
) -> None:
    family_root, _, _ = _predictor_fixture(
        tmp_path,
        error="RuntimeError:model fit failed",
    )
    before = (family_root / "resume_state.json").read_bytes()

    with pytest.raises(
        FailureReconciliationError,
        match="GENUINE_PREDICTOR_FAILURE_IS_NOT_REINITIALIZABLE",
    ):
        reconcile_legacy_zero_progress_predictor_resume_state(
            family="ridge_C5",
            host=HOST,
            repository_root=tmp_path,
            current_source_hash="current-source",
        )

    assert (family_root / "resume_state.json").read_bytes() == before


def test_registered_predictor_migration_leaves_authoritative_state_unchanged(
    tmp_path: Path,
) -> None:
    family_root, run_root, _ = _predictor_fixture(tmp_path)
    random_forest_path = run_root / "family=random_forest" / "resume_state.json"
    random_forest = json.loads(random_forest_path.read_text())
    random_forest.update(
        {
            key: f"value-{key}"
            for key in (
                "feature_authority_hash",
                "target_authority_hash",
                "target_contract_hash",
                "model_config_hash",
                "clean_source_hash",
                "static_authority_bundle_sha256",
                "refit_policy_hash",
            )
        }
    )
    _write_json(random_forest_path, random_forest)
    transformer_root = (
        tmp_path
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / "family=transformer"
    )
    transformer_state = {
        "run_id": RUN_ID,
        "family": "transformer",
        "owner_host": HOST,
        "terminal_state": "DEFERRED_RESOURCE_CAPACITY",
        **{key: f"value-{key}" for key in (
            "feature_authority_hash",
            "target_authority_hash",
            "target_contract_hash",
            "model_config_hash",
            "clean_source_hash",
            "static_authority_bundle_sha256",
            "refit_policy_hash",
        )},
    }
    _write_json(transformer_root / "resume_state.json", transformer_state)
    transformer_before = (transformer_root / "resume_state.json").read_bytes()

    reports = reconcile_registered_predictor_resume_states(
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    by_family = {str(report["family"]): report for report in reports}

    assert by_family["ridge_C5"]["changed"] is True
    assert by_family["transformer"]["changed"] is False
    assert not (family_root / "resume_state.json").exists()
    assert (
        transformer_root / "resume_state.json"
    ).read_bytes() == transformer_before


@pytest.mark.parametrize(
    ("completed_refits", "metrics_cursor"),
    [(["2016-02-02T14:35:00+00:00"], 0), ([], 1)],
)
def test_nonzero_scientific_progress_is_never_silently_reset(
    tmp_path: Path,
    completed_refits: list[str],
    metrics_cursor: int,
) -> None:
    family_root, _, _, _ = _fixture(
        tmp_path,
        completed_refits=completed_refits,
        metrics_cursor=metrics_cursor,
    )
    before = (family_root / "resume_state.json").read_bytes()

    with pytest.raises(
        FailureReconciliationError,
        match="NONZERO_SCIENTIFIC_PROGRESS",
    ):
        reconcile_zero_progress_no_model_control(
            family=FAMILY,
            host=HOST,
            repository_root=tmp_path,
            current_source_hash="new-clean-source",
        )

    assert (family_root / "resume_state.json").read_bytes() == before
    assert not (family_root / "reconciliation_archive").exists()


@pytest.mark.parametrize(
    "changed_key",
    [
        "feature_authority_hash",
        "target_authority_hash",
        "static_authority_bundle_sha256",
    ],
)
def test_changed_clean_v2_scientific_authority_still_fails_closed(
    changed_key: str,
) -> None:
    authority = {
        "feature_authority_hash": CLEAN_V2_FEATURE_AUTHORITY_HASH,
        "target_authority_hash": CLEAN_V2_TARGET_AUTHORITY_HASH,
        "static_authority_bundle_sha256": CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH,
    }
    prior = {**authority, "attempt_generation": 1, "metrics_cursor": 0}
    authority[changed_key] = "changed-current-authority"

    with pytest.raises(
        worker.CleanV2WorkerError,
        match=f"Resume state authority mismatch: {changed_key}",
    ):
        worker._validate_resume_authority(
            prior=prior,
            authority=authority,
            attempt_generation=2,
        )


def test_no_model_control_does_not_require_a_fitted_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model_checkpoint.json"
    checkpoint.write_text("not a model checkpoint", encoding="utf-8")
    monkeypatch.setattr(
        worker,
        "_load_resume_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loaded")),
    )

    resumed = worker._resume_model_for_family(
        FAMILY,
        checkpoint,
        expected_scientific_identity={},
        current_operational_identity={},
        completed_refits=[],
        compatibility_evidence_path=tmp_path / "compatibility.json",
    )

    assert resumed is None


def test_reconciliation_preserves_random_forest_and_resource_deferrals(
    tmp_path: Path,
) -> None:
    _, run_root, deferred, random_forest_before = _fixture(
        tmp_path, feature_hash=None
    )

    report = reconcile_zero_progress_no_model_control(
        family=FAMILY,
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="new-clean-source",
    )
    status = json.loads(
        (run_root / f"supervisor_status_{HOST}.json").read_text()
    )
    random_forest_path = run_root / "family=random_forest" / "resume_state.json"

    assert random_forest_path.read_bytes() == random_forest_before
    assert report["random_forest_preservation"] == {
        "completed_refits": 1,
        "score_timestamps": 322,
        "latest_scored_decision": "2016-02-09T14:55:00+00:00",
    }
    assert status["deferred_resource_families"] == deferred
    assert status["classification"] == (
        "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_CONTROL_RECONCILED"
    )
    assert status["failed_families"] == {}
    assert status["ready"] is False
    assert not (run_root / f"manual_admission_{HOST}.json").exists()


def test_genuine_family_failures_remain_failed_closed(tmp_path: Path) -> None:
    _, run_root, _, _ = _fixture(
        tmp_path,
        other_failures={"random_forest": 7},
    )

    report = reconcile_zero_progress_no_model_control(
        family=FAMILY,
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="new-clean-source",
    )
    status = json.loads(
        (run_root / f"supervisor_status_{HOST}.json").read_text()
    )

    assert report["other_failed_families_preserved"] == {"random_forest": 7}
    assert status["failed_families"] == {"random_forest": 7}
    assert status["classification"] == "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED"
    assert "WORKER_EXIT_NONZERO" in status["blocking_reasons"]


def test_supervisor_deferral_does_not_create_a_scientific_resume_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)

    evidence = supervisor._record_supervisor_capacity_deferral(
        family=FAMILY,
        host=HOST,
        attempt_generation=5,
        resource_decision={"classification": "LOCAL_CAPACITY_DEFERRAL"},
    )

    assert evidence["scientific_resume_state_updated"] is False
    assert not (tmp_path / f"family={FAMILY}" / "resume_state.json").exists()
    assert (tmp_path / f"family={FAMILY}" / "resource_deferral.json").is_file()


def test_supervisor_deferral_preserves_worker_owned_source_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "RUN_ROOT", tmp_path)
    state_path = tmp_path / f"family={FAMILY}" / "resume_state.json"
    prior = {
        "run_id": RUN_ID,
        "family": FAMILY,
        "owner_host": HOST,
        "attempt_generation": 4,
        "clean_source_hash": "old-source",
        "terminal_state": "READY_FOR_MANUAL_ADMISSION",
        "completed_refits": [],
        "metrics_cursor": 0,
    }
    _write_json(state_path, prior)

    evidence = supervisor._record_supervisor_capacity_deferral(
        family=FAMILY,
        host=HOST,
        attempt_generation=5,
        resource_decision={"classification": "LOCAL_CAPACITY_DEFERRAL"},
    )

    deferred = json.loads(state_path.read_text())
    assert deferred["attempt_generation"] == 4
    assert deferred["clean_source_hash"] == "old-source"
    assert deferred["terminal_state"] == "DEFERRED_RESOURCE_CAPACITY"
    assert evidence["attempt_generation"] == 5
    assert evidence["scientific_resume_state_updated"] is False
    assert evidence["resume_state_annotation_updated"] is True
    assert evidence["resume_attempt_generation_preserved"] == 4
    worker._validate_resume_authority(
        prior=deferred,
        authority={"clean_source_hash": "new-source"},
        attempt_generation=5,
    )


def test_zero_progress_momentum_missing_identity_is_archived_and_reinitialized(
    tmp_path: Path,
) -> None:
    family_root, _, _, _ = _fixture(
        tmp_path,
        family="momentum",
        feature_hash=CLEAN_V2_FEATURE_AUTHORITY_HASH,
        error="CleanV2WorkerError:Resume control scientific identity is missing",
    )

    report = ensure_zero_progress_control_resume_state(
        family="momentum",
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    repaired = json.loads((family_root / "resume_state.json").read_text())
    archive = tmp_path / str(report["archive_path"])

    assert report["committed_work_found"] is False
    assert report["old_control_scientific_identity"] is None
    assert report["repaired_terminal_state"] == "READY_FOR_MANUAL_ADMISSION"
    assert repaired["family"] == "momentum"
    assert repaired["completed_refits"] == []
    assert repaired["completed_scoring_packages"] == []
    assert repaired["metrics_cursor"] == 0
    assert repaired["control_scientific_identity"]["family"] == "momentum"
    assert (archive / "resume.json").is_file()
    assert (archive / "failure.json").is_file()


def test_current_equal_weight_control_state_remains_unchanged(tmp_path: Path) -> None:
    family_root, _, _, _ = _fixture(tmp_path)
    reconcile_zero_progress_no_model_control(
        family=FAMILY,
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    before = (family_root / "resume_state.json").read_bytes()

    report = ensure_zero_progress_control_resume_state(
        family=FAMILY,
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="newer-operational-source",
    )

    assert report["classification"] == (
        "DS24_CLEAN_V2_CONTROL_RESUME_IDENTITY_CURRENT"
    )
    assert report["changed"] is False
    assert (family_root / "resume_state.json").read_bytes() == before


def test_generic_initialization_covers_every_registered_control(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "research_runs" / "ds24_clean_v2" / RUN_ID
    _write_json(
        run_root / f"supervisor_status_{HOST}.json",
        {
            "run_id": RUN_ID,
            "host_role": HOST,
            "active_workers": [],
            "failed_families": {},
            "queued_families": [],
            "deferred_resource_families": {},
            "resource_paused_families": {},
            "worker_reservations": {"active_worker_reservations": 0},
        },
    )

    reports = reconcile_registered_control_resume_states(
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    registered = load_contract("model_registry.json")["controls"]

    assert [report["family"] for report in reports] == registered
    for family in registered:
        state = json.loads(
            (
                run_root / f"family={family}" / "resume_state.json"
            ).read_text()
        )
        assert state["control_scientific_identity"]["family"] == family
        assert state["expected_refits"] == 0
        assert state["metrics_cursor"] == 0
        assert state["checkpoint_compatibility"]["checkpoint_required"] is False


def test_preflight_reconciles_controls_before_worker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    momentum_root, _, _, _ = _fixture(
        tmp_path,
        family="momentum",
        feature_hash=CLEAN_V2_FEATURE_AUTHORITY_HASH,
        error="CleanV2WorkerError:Resume control scientific identity is missing",
    )
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "clean_source_hash", lambda: "preflight-source")

    reports = supervisor._reconcile_control_resume_states_for_preflight(HOST)

    assert {report["family"] for report in reports} == {
        "momentum",
        "equal_weight_no_model",
    }
    momentum = json.loads((momentum_root / "resume_state.json").read_text())
    assert momentum["terminal_state"] == "READY_FOR_MANUAL_ADMISSION"
    assert momentum["control_scientific_identity"]["family"] == "momentum"


def test_live_runtime_control_validation_is_read_only_and_never_reconciles(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "research_runs" / "ds24_clean_v2" / RUN_ID
    _write_json(
        run_root / f"supervisor_status_{HOST}.json",
        {
            "run_id": RUN_ID,
            "host_role": HOST,
            "active_workers": [],
            "failed_families": {},
            "queued_families": [],
            "deferred_resource_families": {},
            "worker_reservations": {"active_worker_reservations": 0},
        },
    )
    reconcile_registered_control_resume_states(
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    _write_json(
        run_root / f"supervisor_lease_{HOST}.json",
        {"pid": 123, "generation": 456},
    )
    protected_paths = [
        run_root / "family=momentum" / "resume_state.json",
        run_root / "family=equal_weight_no_model" / "resume_state.json",
        run_root / f"supervisor_status_{HOST}.json",
        run_root / f"supervisor_lease_{HOST}.json",
    ]
    before = {path: path.read_bytes() for path in protected_paths}

    reports = validate_registered_control_resume_states(
        host=HOST,
        repository_root=tmp_path,
    )

    assert {report["family"] for report in reports} == {
        "momentum",
        "equal_weight_no_model",
    }
    assert all(report["changed"] is False for report in reports)
    assert all(
        report["validation_mode"] == "LIVE_RUNTIME_READ_ONLY"
        for report in reports
    )
    assert {path: path.read_bytes() for path in protected_paths} == before
    with pytest.raises(
        FailureReconciliationError,
        match="CONTROL_RECONCILIATION_REQUIRES_NO_SUPERVISOR_LEASE",
    ):
        reconcile_registered_control_resume_states(
            host=HOST,
            repository_root=tmp_path,
            current_source_hash="current-source",
        )


def test_genuine_current_control_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "research_runs" / "ds24_clean_v2" / RUN_ID
    _write_json(
        run_root / f"supervisor_status_{HOST}.json",
        {
            "run_id": RUN_ID,
            "host_role": HOST,
            "active_workers": [],
            "failed_families": {},
            "queued_families": [],
            "deferred_resource_families": {},
            "worker_reservations": {"active_worker_reservations": 0},
        },
    )
    ensure_zero_progress_control_resume_state(
        family="momentum",
        host=HOST,
        repository_root=tmp_path,
        current_source_hash="current-source",
    )
    state_path = run_root / "family=momentum" / "resume_state.json"
    state = json.loads(state_path.read_text())
    state["control_scientific_identity"]["feature_authority_hash"] = "changed"
    _write_json(state_path, state)
    before = state_path.read_bytes()

    with pytest.raises(
        FailureReconciliationError,
        match="CONTROL_SCIENTIFIC_IDENTITY_MISMATCH",
    ):
        ensure_zero_progress_control_resume_state(
            family="momentum",
            host=HOST,
            repository_root=tmp_path,
            current_source_hash="current-source",
        )

    assert state_path.read_bytes() == before
