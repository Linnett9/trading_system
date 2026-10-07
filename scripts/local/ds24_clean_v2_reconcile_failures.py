from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (  # noqa: E402
    authority_bundle,
    clean_source_hash,
    file_sha256,
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_checkpoint_compatibility import (  # noqa: E402
    CLEAN_V2_FEATURE_AUTHORITY_HASH,
    CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH,
    CLEAN_V2_TARGET_AUTHORITY_HASH,
    CheckpointCompatibilityError,
    make_no_model_control_resume_identity,
    no_model_control_config,
    validate_no_model_control_resume_identity,
)
from core.research.ml.ds24.clean_v2_resources import process_identity  # noqa: E402
from core.research.ml.ds24.clean_v2_runtime import (  # noqa: E402
    REFIT_POLICY_ID,
    RUN_ID,
)
from core.research.ml.ds24_metrics_only_evaluator import (  # noqa: E402
    resolved_performance_contract_v3_hash,
)


class FailureReconciliationError(RuntimeError):
    """Raised when a recorded failure cannot safely become current state."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FailureReconciliationError(f"Required failure-state file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FailureReconciliationError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise FailureReconciliationError(f"Invalid {label}: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise FailureReconciliationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _control_progress_inventory(
    family_root: Path,
    state: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Inventory every persisted signal that could represent scientific work."""

    progress_records: list[tuple[str, Mapping[str, Any]]] = [
        ("resume_state.json", state),
        ("worker_failure.json", failure),
    ]
    for path in sorted(family_root.glob("worker_failure_attempt=*.json")):
        progress_records.append((path.name, _read_json(path)))
    completed_refits = list(
        dict.fromkeys(
            str(value)
            for _, record in progress_records
            for value in list(record.get("completed_refits") or [])
        )
    )
    completed_scoring_packages = list(
        dict.fromkeys(
            str(value)
            for _, record in progress_records
            for value in list(record.get("completed_scoring_packages") or [])
        )
    )
    metrics_cursor = max(
        (int(record.get("metrics_cursor", 0) or 0) for _, record in progress_records),
        default=0,
    )
    scientific_files: list[str] = []
    for path in sorted(family_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(family_root)
        if relative.parts and relative.parts[0] == "reconciliation_archive":
            continue
        name = path.name.lower()
        if (
            relative.parts[0].lower()
            in {"metrics", "models", "predictions", "scores"}
            or name == "model_checkpoint.json"
            or name.startswith("checkpoint_compatibility")
            or path.suffix.lower() in {".parquet", ".pkl", ".pickle"}
        ):
            scientific_files.append(relative.as_posix())
    latest_scored_values = [
        str(record["latest_scored_decision"])
        for _, record in progress_records
        if record.get("latest_scored_decision")
    ]
    latest_scored_decision = max(latest_scored_values, default=None)
    committed_work = bool(
        completed_refits
        or completed_scoring_packages
        or metrics_cursor
        or latest_scored_decision
        or scientific_files
    )
    return {
        "completed_refits": completed_refits,
        "completed_scoring_packages": completed_scoring_packages,
        "metrics_cursor": metrics_cursor,
        "latest_scored_decision": latest_scored_decision,
        "scientific_files": scientific_files,
        "progress_records_checked": [name for name, _ in progress_records],
        "committed_work": committed_work,
    }


_PREDICTOR_RESUME_AUTHORITY_KEYS = (
    "feature_authority_hash",
    "target_authority_hash",
    "target_contract_hash",
    "model_config_hash",
    "clean_source_hash",
    "static_authority_bundle_sha256",
    "refit_policy_hash",
)


def _legacy_predictor_resume_placeholder(state: Mapping[str, Any]) -> bool:
    """Return whether state is the known pre-authority resource placeholder."""

    return bool(
        state.get("reconciled_from_legacy_resource_pause") is True
        and not any(state.get(key) for key in _PREDICTOR_RESUME_AUTHORITY_KEYS)
    )


def _expected_control_state_authority(family: str) -> dict[str, Any]:
    model_registry = load_contract("model_registry.json")
    tournament = load_contract("tournament_contract.json")
    target_contract = load_contract("target_contract.json")
    config = no_model_control_config(
        family=family,
        model_registry=model_registry,
        refit_policy_id=REFIT_POLICY_ID,
    )
    bundle_hash = str(authority_bundle()["bundle_sha256"])
    if bundle_hash != CLEAN_V2_STATIC_AUTHORITY_BUNDLE_HASH:
        raise FailureReconciliationError(
            "Current static authority bundle is not the admitted CLEAN V2 bundle"
        )
    target_contract_hash = str(target_contract["resolved_contract_sha256"])
    control_identity = make_no_model_control_resume_identity(
        family=family,
        model_config=config,
        feature_authority_hash=CLEAN_V2_FEATURE_AUTHORITY_HASH,
        target_authority_hash=CLEAN_V2_TARGET_AUTHORITY_HASH,
        target_contract_hash=target_contract_hash,
        static_authority_bundle_sha256=bundle_hash,
        predictor_manifest=load_contract("predictor_manifest.json"),
        eligibility_contract=load_contract("eligibility_contract.json"),
        tournament_contract=tournament,
        evaluation_contract_hash=resolved_performance_contract_v3_hash(),
    )
    candidates = [
        dict(candidate)
        for candidate in tournament["execution_budget"]["candidates"]
        if candidate.get("family") == family
    ]
    if len(candidates) != 1:
        raise FailureReconciliationError(
            "No unique control schedule exists in tournament authority"
        )
    schedule = candidates[0]
    scoring_sessions = int(schedule["scoring_session_count"])
    score_sessions_per_package = int(
        tournament["refit_policy"]["score_sessions_per_refit"]
    )
    expected_scoring_packages = (
        scoring_sessions + score_sessions_per_package - 1
    ) // score_sessions_per_package
    return {
        "feature_authority_hash": CLEAN_V2_FEATURE_AUTHORITY_HASH,
        "target_authority_hash": CLEAN_V2_TARGET_AUTHORITY_HASH,
        "target_contract_hash": target_contract_hash,
        "model_config_hash": stable_hash(config),
        "static_authority_bundle_sha256": bundle_hash,
        "refit_policy_hash": stable_hash(
            {"id": REFIT_POLICY_ID, "lookback_sessions": 20}
        ),
        "control_scientific_identity": control_identity,
        "control_scientific_identity_hash": stable_hash(control_identity),
        "expected_scoring_packages": expected_scoring_packages,
    }


def _copy_forensic_file(source: Path, archive_root: Path) -> dict[str, Any]:
    name = source.name
    if name == "resume_state.json":
        archived_name = "resume.json"
    elif name == "worker_failure.json":
        archived_name = "failure.json"
    elif name.startswith("worker_failure_attempt="):
        archived_name = "failure_attempt.json"
    elif name == "resource_deferral.json":
        archived_name = "deferral.json"
    elif name.startswith("resource_deferral_attempt="):
        archived_name = "deferral_attempt.json"
    elif name.startswith("supervisor_status_"):
        archived_name = "status.json"
    elif name.startswith("manual_admission_"):
        archived_name = "admission.json"
    else:
        archived_name = name
    destination = archive_root / archived_name
    shutil.copy2(source, destination)
    source_hash = file_sha256(source)
    if file_sha256(destination) != source_hash:
        raise FailureReconciliationError(
            f"Forensic archive verification failed: {source}"
        )
    return {
        "source": source.as_posix(),
        "archived_name": destination.name,
        "sha256": source_hash,
        "bytes": source.stat().st_size,
    }


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _validate_stopped_control_reconciliation(
    *,
    status: Mapping[str, Any],
    run_root: Path,
    host: str,
) -> None:
    if status.get("active_workers"):
        raise FailureReconciliationError(
            "CONTROL_RECONCILIATION_REQUIRES_ZERO_ACTIVE_WORKERS"
        )
    if status.get("orphaned_verified_workers"):
        raise FailureReconciliationError(
            "CONTROL_RECONCILIATION_REQUIRES_ZERO_ORPHANED_WORKERS"
        )
    reservations = status.get("worker_reservations") or {}
    if int(reservations.get("active_worker_reservations", 0) or 0) != 0:
        raise FailureReconciliationError(
            "CONTROL_RECONCILIATION_REQUIRES_ZERO_ACTIVE_RESERVATIONS"
        )
    if (run_root / f"supervisor_lease_{host}.json").exists():
        raise FailureReconciliationError(
            "CONTROL_RECONCILIATION_REQUIRES_NO_SUPERVISOR_LEASE"
        )
    supervisor_pid = int(status.get("supervisor_pid", 0) or 0)
    if supervisor_pid:
        identity = process_identity(supervisor_pid)
        if identity.alive and "ds24_clean_v2_supervisor.py" in identity.command_line:
            raise FailureReconciliationError(
                "CONTROL_RECONCILIATION_REQUIRES_STOPPED_SUPERVISOR"
            )


def _current_control_state(
    state: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> bool:
    observed_identity = state.get("control_scientific_identity")
    if not isinstance(observed_identity, Mapping):
        return False
    expected_identity = authority["control_scientific_identity"]
    try:
        validate_no_model_control_resume_identity(
            observed_identity, expected_identity
        )
    except CheckpointCompatibilityError as exc:
        raise FailureReconciliationError(
            f"CONTROL_SCIENTIFIC_IDENTITY_MISMATCH:{exc}"
        ) from exc
    scientific_keys = (
        "feature_authority_hash",
        "target_authority_hash",
        "target_contract_hash",
        "model_config_hash",
        "static_authority_bundle_sha256",
        "refit_policy_hash",
        "control_scientific_identity_hash",
        "expected_scoring_packages",
    )
    mismatches = [
        key for key in scientific_keys if state.get(key) != authority.get(key)
    ]
    if state.get("refit_policy_id") != REFIT_POLICY_ID:
        mismatches.append("refit_policy_id")
    if mismatches:
        raise FailureReconciliationError(
            "CONTROL_RESUME_SCIENTIFIC_AUTHORITY_MISMATCH:"
            + ",".join(sorted(mismatches))
        )
    return True


def ensure_zero_progress_control_resume_state(
    *,
    family: str,
    host: str,
    repository_root: Path = ROOT,
    current_source_hash: str | None = None,
) -> dict[str, Any]:
    """Ensure one registered no-fit control has current resume authority."""

    model_registry = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    if family not in model_registry.get("controls", []):
        raise FailureReconciliationError(
            f"Family {family!r} is not a registered CLEAN V2 control"
        )
    if family not in ownership.get("hosts", {}).get(host, []):
        raise FailureReconciliationError(
            f"Control {family!r} is not owned by host {host!r}"
        )
    run_root = repository_root / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / f"family={family}"
    state_path = family_root / "resume_state.json"
    failure_path = family_root / "worker_failure.json"
    status_path = run_root / f"supervisor_status_{host}.json"
    state = _optional_json(state_path)
    failure = _optional_json(failure_path)
    status = _optional_json(status_path) or {
        "run_id": RUN_ID,
        "host_role": host,
        "active_workers": [],
        "failed_families": {},
        "queued_families": [],
        "deferred_resource_families": {},
        "resource_paused_families": {},
        "worker_reservations": {"active_worker_reservations": 0},
    }
    if status.get("run_id") != RUN_ID or status.get("host_role") != host:
        raise FailureReconciliationError("Host-specific supervisor status is invalid")
    _validate_stopped_control_reconciliation(
        status=status,
        run_root=run_root,
        host=host,
    )
    authority = _expected_control_state_authority(family)
    if state:
        expected_state_identity = {
            "run_id": RUN_ID,
            "family": family,
            "owner_host": host,
        }
        for key, expected in expected_state_identity.items():
            if state.get(key) != expected:
                raise FailureReconciliationError(
                    f"Resume-state identity mismatch for {key}: {state.get(key)!r}"
                )
        if _current_control_state(state, authority):
            return {
                "classification": "DS24_CLEAN_V2_CONTROL_RESUME_IDENTITY_CURRENT",
                "run_id": RUN_ID,
                "family": family,
                "host": host,
                "changed": False,
                "terminal_state": state.get("terminal_state"),
                "control_scientific_identity_hash": state.get(
                    "control_scientific_identity_hash"
                ),
            }
        if state.get("terminal_state") == "COMPLETE":
            raise FailureReconciliationError(
                "COMPLETED_CONTROL_STATE_MUST_NOT_BE_REINITIALIZED"
            )
    if failure:
        expected_failure_identity = {
            "run_id": RUN_ID,
            "family": family,
            "host": host,
        }
        for key, expected in expected_failure_identity.items():
            if failure.get(key) != expected:
                raise FailureReconciliationError(
                    f"Worker-failure identity mismatch for {key}: "
                    f"{failure.get(key)!r}"
                )
    progress = _control_progress_inventory(family_root, state, failure)
    if progress["committed_work"]:
        raise FailureReconciliationError(
            "NONZERO_SCIENTIFIC_PROGRESS_REQUIRES_NORMAL_CHECKPOINT_RECONCILIATION"
        )
    observed_identity = state.get("control_scientific_identity")
    if observed_identity not in (None, {}):
        raise FailureReconciliationError(
            "MALFORMED_CONTROL_SCIENTIFIC_IDENTITY_FAILS_CLOSED"
        )
    recorded_error = str(failure.get("error") or state.get("error") or "")
    allowed_stale_identity_errors = (
        "Resume control scientific identity is missing",
        "Resume state authority mismatch:",
    )
    if recorded_error and not any(
        marker in recorded_error for marker in allowed_stale_identity_errors
    ):
        raise FailureReconciliationError(
            "GENUINE_CONTROL_FAILURE_IS_NOT_REINITIALIZABLE:"
            + recorded_error
        )
    failed_families = dict(status.get("failed_families") or {})
    if failure and family not in failed_families:
        raise FailureReconciliationError(
            "Current failure is absent from host supervisor status"
        )
    source_hash = current_source_hash or clean_source_hash(
        repository_root=repository_root
    )
    old_generation = int(
        failure.get("attempt_generation")
        or state.get("attempt_generation")
        or 0
    )
    reconciliation_generation = time.time_ns()
    reconciled_at = _utc_now()
    deferred_before = dict(status.get("deferred_resource_families") or {})
    stored_feature_hash = state.get("feature_authority_hash")
    top_level_current = bool(
        state
        and all(
            state.get(key) == authority.get(key)
            for key in (
                "feature_authority_hash",
                "target_authority_hash",
                "target_contract_hash",
                "model_config_hash",
                "static_authority_bundle_sha256",
                "refit_policy_hash",
            )
        )
    )
    if not state:
        old_state_classification = "UNINITIALIZED_ZERO_PROGRESS_CONTROL"
    elif top_level_current:
        old_state_classification = (
            "CURRENT_TOP_LEVEL_AUTHORITY_MISSING_CONTROL_SCIENTIFIC_IDENTITY"
        )
    elif stored_feature_hash is None:
        old_state_classification = (
            "CONTAMINATED_OPERATIONAL_PLACEHOLDER_MISSING_SCIENTIFIC_AUTHORITY"
        )
    else:
        old_state_classification = "NON_CLEAN_V2_ZERO_PROGRESS_RESUME_AUTHORITY"

    archive_sources = [
        path
        for path in (
            state_path,
            failure_path,
            family_root / f"worker_failure_attempt={old_generation}.json",
            family_root / "resource_deferral.json",
            status_path,
            run_root / f"manual_admission_{host}.json",
        )
        if path.is_file()
    ]
    archive_sources.extend(
        sorted(family_root.glob(f"resource_deferral_attempt={old_generation}_*.json"))
    )
    archive_relative: str | None = None
    if archive_sources:
        family_archive_id = stable_hash(family)[:8]
        archive_root = (
            run_root
            / "_a"
            / "c"
            / family_archive_id
            / f"{reconciliation_generation:x}"
        )
        archive_root.mkdir(parents=True, exist_ok=False)
        archived_files = [
            _copy_forensic_file(path, archive_root) for path in archive_sources
        ]
        archive_relative = archive_root.relative_to(repository_root).as_posix()
        _write_json_atomic(
            archive_root / "manifest.json",
            {
                "classification": (
                    "DS24_CLEAN_V2_OBSOLETE_ZERO_PROGRESS_CONTROL_STATE_ARCHIVE"
                ),
                "run_id": RUN_ID,
                "family": family,
                "host": host,
                "archived_at_utc": reconciled_at,
                "old_attempt_generation": old_generation,
                "stored_control_scientific_identity": observed_identity,
                "stored_feature_authority_hash": stored_feature_hash,
                "expected_feature_authority_hash": (
                    CLEAN_V2_FEATURE_AUTHORITY_HASH
                ),
                "obsolete_state_classification": old_state_classification,
                "progress_inventory": progress,
                "files": archived_files,
            },
        )

    attempt_id = stable_hash(
        {
            "run_id": RUN_ID,
            "family": family,
            "host": host,
            "reconciliation_generation": reconciliation_generation,
            "clean_source_hash": source_hash,
        }
    )
    is_deferred = family in deferred_before
    deferred_evidence = deferred_before.get(family) or {}
    terminal_state = (
        str(deferred_evidence.get("terminal_state") or "DEFERRED_RESOURCE_CAPACITY")
        if is_deferred
        else "READY_FOR_MANUAL_ADMISSION"
    )
    repaired_state = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
        **authority,
        "clean_source_hash": source_hash,
        "refit_policy_id": REFIT_POLICY_ID,
        "expected_refits": 0,
        "completed_refits": [],
        "latest_completed_refit": None,
        "expected_scoring_packages": authority["expected_scoring_packages"],
        "completed_scoring_packages": [],
        "latest_completed_scoring_package": None,
        "latest_scored_decision": None,
        "metrics_cursor": 0,
        "attempt_generation": reconciliation_generation,
        "attempt_id": attempt_id,
        "current_operational_identity": {},
        "checkpoint_compatibility": {
            "classification": (
                "NO_MODEL_CONTROL_FITTED_CHECKPOINT_NOT_APPLICABLE"
            ),
            "fitted_model_required": False,
            "checkpoint_required": False,
        },
        "terminal_state": terminal_state,
        "error": None,
        "heartbeat_utc": reconciled_at,
        "control_reconciliation": {
            "classification": (
                "ZERO_PROGRESS_OBSOLETE_CONTROL_STATE_ARCHIVED_AND_REINITIALIZED"
            ),
            "archive_path": archive_relative,
            "old_attempt_generation": old_generation,
            "old_feature_authority_hash": stored_feature_hash,
        },
        "paper_orders": 0,
        "live_orders": 0,
    }
    if is_deferred:
        repaired_state.update(
            {
                "resource_deferral_classification": deferred_evidence.get(
                    "classification"
                ),
                "resource_deferral_timestamp": deferred_evidence.get(
                    "deferred_at_utc"
                ),
                "resource_decision": deferred_evidence.get("resource_decision"),
            }
        )
    family_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(state_path, repaired_state)

    other_failures = dict(failed_families)
    other_failures.pop(family, None)
    queued = list(status.get("queued_families") or [])
    if family not in queued:
        queued.append(family)
    blocking_reasons = [
        reason
        for reason in list(status.get("blocking_reasons") or [])
        if reason != "WORKER_EXIT_NONZERO" or other_failures
    ]
    prior_reconciliations = list(status.get("control_reconciliations") or [])
    prior_reconciliations.append(
        {
            "family": family,
            "classification": (
                "ZERO_PROGRESS_OBSOLETE_CONTROL_STATE_ARCHIVED_AND_REINITIALIZED"
            ),
            "archive_path": archive_relative,
        }
    )
    repaired_status = {
        **status,
        "classification": (
            "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED"
            if other_failures
            else "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_CONTROL_RECONCILED"
        ),
        "attempt_generation": reconciliation_generation,
        "supervisor_pid": None,
        "supervisor_process_creation_time_utc": None,
        "supervisor_identity_verified": False,
        "supervisor_memory": None,
        "active_workers": [],
        "queued_families": queued,
        "failed_families": other_failures,
        "blocking_reasons": blocking_reasons,
        "ready": False,
        "worker_job": None,
        "last_closed_worker_job_snapshot": status.get("worker_job"),
        "heartbeat_utc": reconciled_at,
        "control_reconciliations": prior_reconciliations,
    }
    if dict(repaired_status.get("deferred_resource_families") or {}) != deferred_before:
        raise FailureReconciliationError(
            "Resource-deferred family authority changed during reconciliation"
        )
    _write_json_atomic(status_path, repaired_status)

    stale_pointers = [
        failure_path,
        run_root / f"manual_admission_{host}.json",
    ]
    if not is_deferred:
        stale_pointers.append(family_root / "resource_deferral.json")
    for stale_pointer in stale_pointers:
        if stale_pointer.is_file():
            stale_pointer.unlink()

    random_forest_path = run_root / "family=random_forest" / "resume_state.json"
    random_forest = _optional_json(random_forest_path)
    report = {
        "classification": (
            "DS24_CLEAN_V2_ZERO_PROGRESS_CONTROL_REINITIALIZED_FOR_MANUAL_ADMISSION"
        ),
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "changed": True,
        "old_attempt_generation": old_generation,
        "reconciliation_generation": reconciliation_generation,
        "old_control_scientific_identity": observed_identity,
        "old_feature_authority_hash": stored_feature_hash,
        "expected_feature_authority_hash": CLEAN_V2_FEATURE_AUTHORITY_HASH,
        "old_state_classification": old_state_classification,
        "committed_work_found": progress["committed_work"],
        "progress_inventory": progress,
        "archive_path": archive_relative,
        "repaired_state_path": state_path.relative_to(repository_root).as_posix(),
        "repaired_terminal_state": repaired_state["terminal_state"],
        "control_scientific_identity_hash": authority[
            "control_scientific_identity_hash"
        ],
        "clean_source_hash": source_hash,
        "random_forest_preservation": {
            "completed_refits": len(random_forest.get("completed_refits") or []),
            "score_timestamps": int(random_forest.get("metrics_cursor", 0) or 0),
            "latest_scored_decision": random_forest.get(
                "latest_scored_decision"
            ),
        },
        "resource_deferred_families": sorted(deferred_before),
        "other_failed_families_preserved": other_failures,
        "tournament_classification": repaired_status["classification"],
        "manual_admission_invalidated": True,
        "reconciled_at_utc": reconciled_at,
    }
    report_path = family_root / "control_reconciliation.json"
    _write_json_atomic(report_path, report)
    _write_json_atomic(
        family_root
        / f"control_reconciliation_generation={reconciliation_generation}.json",
        report,
    )
    return report


def validate_current_control_resume_state(
    *,
    family: str,
    host: str,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Read-only validation for a control while its supervisor is live."""

    model_registry = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    if family not in model_registry.get("controls", []):
        raise FailureReconciliationError(
            f"Family {family!r} is not a registered CLEAN V2 control"
        )
    if family not in ownership.get("hosts", {}).get(host, []):
        raise FailureReconciliationError(
            f"Control {family!r} is not owned by host {host!r}"
        )
    state_path = (
        repository_root
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / f"family={family}"
        / "resume_state.json"
    )
    if not state_path.is_file():
        raise FailureReconciliationError(
            f"CONTROL_RESUME_STATE_MISSING:{family}"
        )
    state = _read_json(state_path)
    expected_state_identity = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
    }
    for key, expected in expected_state_identity.items():
        if state.get(key) != expected:
            raise FailureReconciliationError(
                f"Resume-state identity mismatch for {key}: {state.get(key)!r}"
            )
    authority = _expected_control_state_authority(family)
    if not _current_control_state(state, authority):
        raise FailureReconciliationError(
            f"CONTROL_SCIENTIFIC_IDENTITY_MISSING:{family}"
        )
    return {
        "classification": "DS24_CLEAN_V2_CONTROL_RESUME_IDENTITY_CURRENT",
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "changed": False,
        "validation_mode": "LIVE_RUNTIME_READ_ONLY",
        "terminal_state": state.get("terminal_state"),
        "control_scientific_identity_hash": state.get(
            "control_scientific_identity_hash"
        ),
    }


def reconcile_zero_progress_no_model_control(
    *,
    family: str,
    host: str,
    repository_root: Path = ROOT,
    current_source_hash: str | None = None,
) -> dict[str, Any]:
    """Compatibility name for the generic registered-control reconciler."""

    return ensure_zero_progress_control_resume_state(
        family=family,
        host=host,
        repository_root=repository_root,
        current_source_hash=current_source_hash,
    )


def reconcile_registered_control_resume_states(
    *,
    host: str,
    repository_root: Path = ROOT,
    current_source_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Validate or initialize every registered control owned by one host."""

    model_registry = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    owned = set(ownership.get("hosts", {}).get(host, []))
    source_hash = current_source_hash or clean_source_hash(
        repository_root=repository_root
    )
    return [
        ensure_zero_progress_control_resume_state(
            family=str(family),
            host=host,
            repository_root=repository_root,
            current_source_hash=source_hash,
        )
        for family in model_registry.get("controls", [])
        if family in owned
    ]


def validate_registered_control_resume_states(
    *,
    host: str,
    repository_root: Path = ROOT,
) -> list[dict[str, Any]]:
    """Validate every owned control without performing stopped-state writes."""

    model_registry = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    owned = set(ownership.get("hosts", {}).get(host, []))
    return [
        validate_current_control_resume_state(
            family=str(family),
            host=host,
            repository_root=repository_root,
        )
        for family in model_registry.get("controls", [])
        if family in owned
    ]


def reconcile_legacy_zero_progress_predictor_resume_state(
    *,
    family: str,
    host: str,
    repository_root: Path = ROOT,
    current_source_hash: str | None = None,
) -> dict[str, Any]:
    """Archive one obsolete zero-progress predictor placeholder.

    Historical supervisor recovery wrote operational resource-deferral fields into
    ``resume_state.json`` before a predictor worker had established scientific
    authority.  A current worker must reject that file.  While the runtime is
    stopped, this migration preserves the placeholder as forensic evidence and
    removes only its current pointer so the worker can initialize authoritative
    state on its next admitted attempt.
    """

    model_registry = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    if family not in model_registry.get("families", {}):
        raise FailureReconciliationError(
            f"Family {family!r} is not a registered CLEAN V2 predictor"
        )
    if family not in ownership.get("hosts", {}).get(host, []):
        raise FailureReconciliationError(
            f"Predictor {family!r} is not owned by host {host!r}"
        )

    run_root = repository_root / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / f"family={family}"
    state_path = family_root / "resume_state.json"
    failure_path = family_root / "worker_failure.json"
    status_path = run_root / f"supervisor_status_{host}.json"
    state = _optional_json(state_path)
    failure = _optional_json(failure_path)
    status = _optional_json(status_path) or {
        "run_id": RUN_ID,
        "host_role": host,
        "active_workers": [],
        "failed_families": {},
        "queued_families": [],
        "deferred_resource_families": {},
        "resource_paused_families": {},
        "worker_reservations": {"active_worker_reservations": 0},
    }
    if status.get("run_id") != RUN_ID or status.get("host_role") != host:
        raise FailureReconciliationError("Host-specific supervisor status is invalid")
    _validate_stopped_control_reconciliation(
        status=status,
        run_root=run_root,
        host=host,
    )

    if not state:
        return {
            "classification": "DS24_CLEAN_V2_PREDICTOR_RESUME_STATE_ABSENT",
            "run_id": RUN_ID,
            "family": family,
            "host": host,
            "changed": False,
        }
    expected_state_identity = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
    }
    for key, expected in expected_state_identity.items():
        if state.get(key) != expected:
            raise FailureReconciliationError(
                f"Resume-state identity mismatch for {key}: {state.get(key)!r}"
            )
    missing_authority = [
        key for key in _PREDICTOR_RESUME_AUTHORITY_KEYS if not state.get(key)
    ]
    if not missing_authority:
        return {
            "classification": "DS24_CLEAN_V2_PREDICTOR_RESUME_AUTHORITY_PRESENT",
            "run_id": RUN_ID,
            "family": family,
            "host": host,
            "changed": False,
            "terminal_state": state.get("terminal_state"),
        }
    if not _legacy_predictor_resume_placeholder(state):
        raise FailureReconciliationError(
            "MALFORMED_PREDICTOR_RESUME_AUTHORITY_FAILS_CLOSED:"
            + ",".join(sorted(missing_authority))
        )
    if state.get("terminal_state") == "COMPLETE":
        raise FailureReconciliationError(
            "COMPLETED_PREDICTOR_STATE_MUST_NOT_BE_REINITIALIZED"
        )
    if failure:
        expected_failure_identity = {
            "run_id": RUN_ID,
            "family": family,
            "host": host,
        }
        for key, expected in expected_failure_identity.items():
            if failure.get(key) != expected:
                raise FailureReconciliationError(
                    f"Worker-failure identity mismatch for {key}: "
                    f"{failure.get(key)!r}"
                )

    progress = _control_progress_inventory(family_root, state, failure)
    if progress["committed_work"]:
        raise FailureReconciliationError(
            "NONZERO_SCIENTIFIC_PROGRESS_REQUIRES_NORMAL_CHECKPOINT_RECONCILIATION"
        )
    recorded_error = str(failure.get("error") or state.get("error") or "")
    if recorded_error and "Resume state authority mismatch:" not in recorded_error:
        raise FailureReconciliationError(
            "GENUINE_PREDICTOR_FAILURE_IS_NOT_REINITIALIZABLE:" + recorded_error
        )
    failed_families = dict(status.get("failed_families") or {})
    if failure and family not in failed_families:
        raise FailureReconciliationError(
            "Current failure is absent from host supervisor status"
        )

    source_hash = current_source_hash or clean_source_hash(
        repository_root=repository_root
    )
    old_generation = int(
        failure.get("attempt_generation")
        or state.get("attempt_generation")
        or 0
    )
    reconciliation_generation = time.time_ns()
    reconciled_at = _utc_now()
    archive_sources = [
        path
        for path in (
            state_path,
            failure_path,
            family_root / f"worker_failure_attempt={old_generation}.json",
            family_root / "resource_deferral.json",
            status_path,
            run_root / f"manual_admission_{host}.json",
        )
        if path.is_file()
    ]
    archive_sources.extend(
        sorted(family_root.glob(f"resource_deferral_attempt={old_generation}_*.json"))
    )
    archive_root = (
        run_root
        / "_a"
        / "p"
        / stable_hash(family)[:8]
        / f"{reconciliation_generation:x}"
    )
    archive_root.mkdir(parents=True, exist_ok=False)
    archived_files = [
        _copy_forensic_file(path, archive_root) for path in archive_sources
    ]
    archive_relative = archive_root.relative_to(repository_root).as_posix()
    _write_json_atomic(
        archive_root / "manifest.json",
        {
            "classification": (
                "DS24_CLEAN_V2_OBSOLETE_ZERO_PROGRESS_PREDICTOR_STATE_ARCHIVE"
            ),
            "run_id": RUN_ID,
            "family": family,
            "host": host,
            "archived_at_utc": reconciled_at,
            "old_attempt_generation": old_generation,
            "missing_authority_keys": sorted(missing_authority),
            "progress_inventory": progress,
            "clean_source_hash": source_hash,
            "files": archived_files,
        },
    )

    other_failures = dict(failed_families)
    other_failures.pop(family, None)
    queued = list(status.get("queued_families") or [])
    if family not in queued:
        queued.append(family)
    blocking_reasons = [
        reason
        for reason in list(status.get("blocking_reasons") or [])
        if reason != "WORKER_EXIT_NONZERO" or other_failures
    ]
    prior_reconciliations = list(
        status.get("predictor_resume_reconciliations") or []
    )
    prior_reconciliations.append(
        {
            "family": family,
            "classification": (
                "ZERO_PROGRESS_LEGACY_PREDICTOR_PLACEHOLDER_ARCHIVED"
            ),
            "archive_path": archive_relative,
        }
    )
    repaired_status = {
        **status,
        "classification": (
            "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED"
            if other_failures
            else "DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE_PREDICTOR_RECONCILED"
        ),
        "attempt_generation": reconciliation_generation,
        "supervisor_pid": None,
        "supervisor_process_creation_time_utc": None,
        "supervisor_identity_verified": False,
        "supervisor_memory": None,
        "active_workers": [],
        "queued_families": queued,
        "failed_families": other_failures,
        "blocking_reasons": blocking_reasons,
        "ready": False,
        "worker_job": None,
        "last_closed_worker_job_snapshot": status.get("worker_job"),
        "heartbeat_utc": reconciled_at,
        "predictor_resume_reconciliations": prior_reconciliations,
    }
    _write_json_atomic(status_path, repaired_status)

    state_path.unlink()
    if failure_path.is_file():
        failure_path.unlink()
    admission_path = run_root / f"manual_admission_{host}.json"
    if admission_path.is_file():
        admission_path.unlink()
    deferred_before = dict(status.get("deferred_resource_families") or {})
    if family not in deferred_before:
        resource_deferral_path = family_root / "resource_deferral.json"
        if resource_deferral_path.is_file():
            resource_deferral_path.unlink()

    random_forest = _optional_json(
        run_root / "family=random_forest" / "resume_state.json"
    )
    report = {
        "classification": (
            "DS24_CLEAN_V2_ZERO_PROGRESS_PREDICTOR_PLACEHOLDER_ARCHIVED"
        ),
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "changed": True,
        "old_attempt_generation": old_generation,
        "reconciliation_generation": reconciliation_generation,
        "missing_authority_keys": sorted(missing_authority),
        "committed_work_found": progress["committed_work"],
        "progress_inventory": progress,
        "archive_path": archive_relative,
        "resume_state_removed": True,
        "clean_source_hash": source_hash,
        "random_forest_preservation": {
            "completed_refits": len(random_forest.get("completed_refits") or []),
            "score_timestamps": int(random_forest.get("metrics_cursor", 0) or 0),
            "latest_scored_decision": random_forest.get(
                "latest_scored_decision"
            ),
        },
        "other_failed_families_preserved": other_failures,
        "tournament_classification": repaired_status["classification"],
        "manual_admission_invalidated": True,
        "reconciled_at_utc": reconciled_at,
    }
    report_path = family_root / "predictor_resume_reconciliation.json"
    _write_json_atomic(report_path, report)
    _write_json_atomic(
        family_root
        / f"predictor_resume_reconciliation_generation={reconciliation_generation}.json",
        report,
    )
    return report


def reconcile_registered_predictor_resume_states(
    *,
    host: str,
    repository_root: Path = ROOT,
    current_source_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Migrate obsolete zero-progress placeholders for owned predictors."""

    model_registry = load_contract("model_registry.json")
    ownership = load_contract("cross_host_ownership.json")
    owned = set(ownership.get("hosts", {}).get(host, []))
    source_hash = current_source_hash or clean_source_hash(
        repository_root=repository_root
    )
    return [
        reconcile_legacy_zero_progress_predictor_resume_state(
            family=str(family),
            host=host,
            repository_root=repository_root,
            current_source_hash=source_hash,
        )
        for family in model_registry.get("families", {})
        if family in owned
    ]


def reconcile_failure_state(
    *,
    family: str,
    host: str,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Reconcile one preserved pre-attempt-identity failure, failing closed."""

    ownership = load_contract("cross_host_ownership.json")
    if family not in ownership["hosts"][host]:
        raise FailureReconciliationError(
            f"Family {family!r} is not in the current {host!r} ownership authority"
        )
    run_root = repository_root / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / f"family={family}"
    state_path = family_root / "resume_state.json"
    failure_path = family_root / "worker_failure.json"
    state = _read_json(state_path)
    failure = _read_json(failure_path)

    expected_state_identity = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
    }
    for key, expected in expected_state_identity.items():
        if state.get(key) != expected:
            raise FailureReconciliationError(
                f"Resume-state identity mismatch for {key}: {state.get(key)!r}"
            )
    expected_failure_identity = {
        "run_id": RUN_ID,
        "family": family,
        "host": host,
    }
    for key, expected in expected_failure_identity.items():
        if failure.get(key) != expected:
            raise FailureReconciliationError(
                f"Worker-failure identity mismatch for {key}: {failure.get(key)!r}"
            )
    if state.get("terminal_state") == "COMPLETE":
        raise FailureReconciliationError("Refusing to override a completed attempt")

    failed_at = _timestamp(failure.get("failed_at_utc"), label="failed_at_utc")
    heartbeat = _timestamp(state.get("heartbeat_utc"), label="heartbeat_utc")
    if failed_at < heartbeat:
        raise FailureReconciliationError(
            "Recorded failure predates the current resume-state heartbeat"
        )
    status_path = run_root / f"supervisor_status_{host}.json"
    status = _read_json(status_path)
    if status.get("run_id") != RUN_ID or status.get("host_role") != host:
        raise FailureReconciliationError("Host-specific supervisor status is out of scope")
    active_families = {
        str(row.get("family"))
        for row in status.get("active_workers", [])
        if isinstance(row, dict)
    }
    if family in active_families:
        raise FailureReconciliationError(
            "Refusing to reconcile state while this family has an active worker"
        )
    if family not in status.get("failed_families", {}):
        raise FailureReconciliationError(
            "Host-specific supervisor status does not record this family as failed"
        )

    state_generation = int(state.get("attempt_generation", 0) or 0)
    failure_generation = int(failure.get("attempt_generation", 0) or 0)
    if state_generation and failure_generation and state_generation != failure_generation:
        raise FailureReconciliationError("Attempt generation mismatch")
    attempt_generation = failure_generation or state_generation or int(failed_at.value)
    existing_attempt_id = failure.get("attempt_id") or state.get("attempt_id")
    attempt_id = str(
        existing_attempt_id
        or stable_hash(
            {
                "run_id": RUN_ID,
                "family": family,
                "host": host,
                "attempt_generation": attempt_generation,
                "failed_at_utc": failed_at.isoformat(),
                "error": failure.get("error"),
            }
        )
    )
    log_path = failure.get("log_path") or (
        Path("research_runs")
        / "ds24_clean_v2"
        / RUN_ID
        / "logs"
        / host
        / f"{family}.log"
    ).as_posix()
    if not (repository_root / str(log_path)).is_file():
        raise FailureReconciliationError(f"Preserved worker log is missing: {log_path}")

    completed_refits = list(state.get("completed_refits", []))
    metrics_cursor = int(state.get("metrics_cursor", 0) or 0)
    classification = str(
        failure.get("classification") or "DS24_CLEAN_V2_WORKER_FAILED_CLOSED"
    )
    updated_failure = {
        **failure,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "classification": classification,
        "terminal_state": "FAILED_CLOSED",
        "log_path": str(log_path),
        "completed_refits": completed_refits,
        "metrics_cursor": metrics_cursor,
    }
    updated_state = {
        **state,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "terminal_state": "FAILED_CLOSED",
        "terminal_failure_classification": classification,
        "error": failure.get("error"),
        "failure_timestamp": failed_at.isoformat(),
        "failure_log_path": str(log_path),
        "completed_refits": completed_refits,
        "metrics_cursor": metrics_cursor,
        "heartbeat_utc": failed_at.isoformat(),
    }
    _write_json_atomic(
        family_root / f"worker_failure_attempt={attempt_generation}.json",
        updated_failure,
    )
    _write_json_atomic(failure_path, updated_failure)
    _write_json_atomic(state_path, updated_state)
    return {
        "classification": "DS24_CLEAN_V2_FAILURE_STATE_RECONCILED",
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "terminal_state": "FAILED_CLOSED",
        "error": updated_state["error"],
        "failure_timestamp": updated_state["failure_timestamp"],
        "failure_log_path": updated_state["failure_log_path"],
        "completed_refits": completed_refits,
        "metrics_cursor": metrics_cursor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile one preserved CLEAN V2 worker failure, or explicitly "
            "archive and reinitialize a proven-zero-progress no-model control."
        )
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--host", choices=("dell", "mac"), required=True)
    parser.add_argument(
        "--reinitialize-zero-progress-control",
        action="store_true",
        help=(
            "Archive and reinitialize a stale, zero-progress registered "
            "no-model control resume identity."
        ),
    )
    args = parser.parse_args()
    if args.reinitialize_zero_progress_control:
        report = reconcile_zero_progress_no_model_control(
            family=args.family,
            host=args.host,
        )
    else:
        report = reconcile_failure_state(family=args.family, host=args.host)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
