from __future__ import annotations

import json
import os
import platform
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.registries.io import canonical_hash
from core.research.ml.registries.types import ARTIFACT_KINDS
from core.research.ml.stock_level.prediction_artifacts.rows import _lock_file, _unlock_file


DEFAULT_LEDGER_PATH = Path("reports/ml/experiments/experiment_ledger.jsonl")
EVENT_STATUSES = frozenset({
    "PLANNED", "STARTED", "COMPLETED", "FAILED", "REJECTED",
    "SKIPPED_COMPLETE", "CANCELLED", "INVALIDATED",
})
SELECTOR_LEDGER_CONTRACT = "selector_experiment_ledger.v1"
SELECTOR_STATUSES = frozenset({
    "PLANNED", "RUNNING", "SUCCEEDED", "FAILED", "BLOCKED",
    "REJECTED", "ELIGIBLE_FOR_PORTFOLIO_REPLAY",
})
SELECTOR_TRANSITIONS = {
    "PLANNED": {"RUNNING", "BLOCKED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "BLOCKED"},
    "SUCCEEDED": {"REJECTED", "ELIGIBLE_FOR_PORTFOLIO_REPLAY"},
    "FAILED": set(), "BLOCKED": set(), "REJECTED": set(),
    "ELIGIBLE_FOR_PORTFOLIO_REPLAY": set(),
}
SELECTOR_REQUIRED_IDENTITY_FIELDS = (
    "experiment_id", "component_id", "campaign_id", "model_id", "decision_date",
    "dataset_id", "dataset_manifest_checksum", "daily_spine_id",
    "symbol_registry_id", "feature_schema_hash", "target_contract_hash",
    "target_provenance_contract_version", "ranking_contract_id", "fold_id",
    "purge_sessions", "embargo_sessions", "maximum_label_available_timestamp",
    "hyperparameters", "random_seed", "training_start", "training_end",
    "source_commit", "planned_output_root",
)


def register_selector_plan(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically register an immutable fitted-component plan as PLANNED."""
    components = list(plan.get("components") or [])
    if len(components) != 15:
        raise ValueError("Selector plan must contain exactly 15 fitted components")
    ledger = read_selector_ledger(path, missing_ok=True)
    rows = {row["experiment_id"]: row for row in ledger["experiments"]}
    changed = False
    for component in components:
        definition = selector_experiment_definition(component)
        experiment_id = definition["experiment_id"]
        existing = rows.get(experiment_id)
        if existing:
            if existing["material_trial_identity"] != definition["material_trial_identity"]:
                raise ValueError(f"Duplicate immutable experiment identity: {experiment_id}")
            continue
        timestamp = _selector_timestamp()
        row = {
            **definition, "parent_experiment_id": None,
            "attempt_id": f"{experiment_id}:attempt-1",
            "hypothesis": "Strict-OOS fitted selector adds ranking information",
            "status": "PLANNED", "status_timestamp": timestamp,
            "metrics_path": None, "component_manifest_path": None,
            "failure_reason": None, "blocker_reason": None,
            "continuation_or_rejection_reason": None,
            "attempt_history": [{
                "attempt_id": f"{experiment_id}:attempt-1",
                "status": "PLANNED", "status_timestamp": timestamp,
            }],
        }
        rows[experiment_id] = row
        changed = True
    result = _selector_document(rows.values())
    if changed or not path.exists():
        _atomic_selector_write(path, result)
    return result


def selector_experiment_definition(component: Mapping[str, Any]) -> dict[str, Any]:
    mapped = {
        "experiment_id": component.get("experiment_id"),
        "component_id": component.get("component_id"),
        "campaign_id": component.get("campaign_id"),
        "model_id": component.get("model_id"),
        "decision_date": component.get("decision_date"),
        "dataset_id": component.get("dataset_id"),
        "dataset_manifest_checksum": component.get("dataset_checksum"),
        "daily_spine_id": component.get("daily_spine_id"),
        "symbol_registry_id": component.get("symbol_registry_id"),
        "feature_schema_hash": component.get("feature_schema_hash"),
        "target_contract_hash": component.get("target_contract_hash"),
        "target_provenance_contract_version": component.get("target_provenance_contract_version"),
        "ranking_contract_id": component.get("ranking_contract_id"),
        "fold_id": component.get("fold_id"),
        "purge_sessions": component.get("purge_sessions"),
        "embargo_sessions": component.get("embargo_sessions"),
        "maximum_label_available_timestamp": component.get("maximum_label_available_timestamp"),
        "hyperparameters": component.get("hyperparameters", {}),
        "random_seed": component.get("random_seed"),
        "training_start": component.get("training_start"),
        "training_end": component.get("training_end"),
        "source_commit": component.get("source_commit"),
        "planned_output_root": component.get("planned_output_root"),
    }
    missing = [
        key for key in SELECTOR_REQUIRED_IDENTITY_FIELDS
        if mapped.get(key) is None or mapped.get(key) == ""
    ]
    if missing:
        raise ValueError(f"Selector experiment identity missing: {','.join(missing)}")
    if mapped["target_provenance_contract_version"] != "stock_level_target_provenance_v2":
        raise ValueError("Target provenance v2 is required")
    material = canonical_hash(mapped)
    return {**mapped, "material_trial_identity": material}


def transition_selector_experiment(
    path: Path, *, experiment_id: str, to_status: str,
    component: Mapping[str, Any], attempt_id: str | None = None,
    metrics_path: str | None = None, component_manifest_path: str | None = None,
    failure_reason: str | None = None, blocker_reason: str | None = None,
    continuation_or_rejection_reason: str | None = None,
) -> dict[str, Any]:
    if to_status not in SELECTOR_STATUSES or to_status == "PLANNED":
        raise ValueError(f"Invalid selector transition status: {to_status}")
    ledger = read_selector_ledger(path)
    rows = {row["experiment_id"]: row for row in ledger["experiments"]}
    row = rows.get(experiment_id)
    if not row:
        raise ValueError("Selector experiment is not registered")
    definition = selector_experiment_definition(component)
    if definition["material_trial_identity"] != row["material_trial_identity"]:
        raise ValueError("Immutable selector experiment definition mismatch")
    current = row["status"]
    if to_status == current:
        return ledger
    if to_status not in SELECTOR_TRANSITIONS[current]:
        raise ValueError(f"Invalid selector transition: {current} -> {to_status}")
    chosen_attempt = attempt_id or row["attempt_id"]
    if chosen_attempt != row["attempt_id"] and current != "PLANNED":
        raise ValueError("Retries require a newly registered experiment after terminal failure")
    timestamp = _selector_timestamp()
    updated = {
        **row, "status": to_status, "status_timestamp": timestamp,
        "attempt_id": chosen_attempt,
        "metrics_path": metrics_path or row.get("metrics_path"),
        "component_manifest_path": component_manifest_path or row.get("component_manifest_path"),
        "failure_reason": failure_reason or row.get("failure_reason"),
        "blocker_reason": blocker_reason or row.get("blocker_reason"),
        "continuation_or_rejection_reason": (
            continuation_or_rejection_reason or row.get("continuation_or_rejection_reason")
        ),
        "attempt_history": [*row["attempt_history"], {
            "attempt_id": chosen_attempt, "status": to_status,
            "status_timestamp": timestamp, "failure_reason": failure_reason,
            "blocker_reason": blocker_reason,
        }],
    }
    rows[experiment_id] = updated
    result = _selector_document(rows.values())
    _atomic_selector_write(path, result)
    return result


def selector_trial_counts(ledger: Mapping[str, Any]) -> dict[str, int]:
    rows = list(ledger.get("experiments") or [])
    return {
        "fitted_model_count": len({row["model_id"] for row in rows}),
        "decision_date_count": len({row["decision_date"] for row in rows}),
        "seed_count": len({row["random_seed"] for row in rows}),
        "hyperparameter_configuration_count": len({
            canonical_hash(row["hyperparameters"]) for row in rows
        }),
        "planned_material_trials": len(rows),
        "executed_material_trials": sum(row["status"] != "PLANNED" for row in rows),
        "failed_material_trials": sum(row["status"] == "FAILED" for row in rows),
        "rejected_material_trials": sum(row["status"] == "REJECTED" for row in rows),
    }


def require_selector_experiment(
    path: Path, component: Mapping[str, Any], *, required_status: str,
) -> dict[str, Any]:
    ledger = read_selector_ledger(path)
    definition = selector_experiment_definition(component)
    matches = [
        row for row in ledger["experiments"]
        if row["experiment_id"] == definition["experiment_id"]
    ]
    if len(matches) != 1:
        raise ValueError("Selector experiment ledger evidence is missing")
    row = matches[0]
    if row["material_trial_identity"] != definition["material_trial_identity"]:
        raise ValueError("Selector experiment ledger identity mismatch")
    if row["status"] != required_status:
        raise ValueError(
            f"Selector experiment status must be {required_status}, found {row['status']}"
        )
    return row


def read_selector_ledger(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if not path.exists():
        if missing_ok:
            return _selector_document(())
        raise ValueError("Selector experiment ledger is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Selector experiment ledger is corrupt") from exc
    checksum = payload.get("ledger_checksum")
    expected = canonical_hash({key: value for key, value in payload.items() if key != "ledger_checksum"})
    if payload.get("ledger_contract_version") != SELECTOR_LEDGER_CONTRACT or checksum != expected:
        raise ValueError("Selector experiment ledger checksum is invalid")
    return payload


def _selector_document(rows) -> dict[str, Any]:
    payload = {
        "ledger_contract_version": SELECTOR_LEDGER_CONTRACT,
        "authoritative_representation": "atomic_json",
        "experiments": sorted((dict(row) for row in rows), key=lambda row: row["experiment_id"]),
    }
    payload["trial_counts"] = selector_trial_counts(payload)
    payload["ledger_checksum"] = canonical_hash(payload)
    return payload


def _atomic_selector_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _selector_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_spec_hash(specification: Mapping[str, Any]) -> str:
    return canonical_hash({"contract_version": "experiment_spec_v1", "specification": dict(specification)})


def new_experiment_run_id(spec_hash: str) -> str:
    return f"run-{spec_hash[:12].lower()}-{uuid.uuid4().hex[:12]}"


def append_ledger_event(
    path: Path = DEFAULT_LEDGER_PATH, *, experiment_spec_hash_value: str,
    experiment_run_id: str, event_status: str, artifact_kind: str,
    canonical_model_id: str | None, requested_model_id: str | None,
    registry_hashes: Mapping[str, Any], source_commit: str | None,
    artifact_paths: Sequence[str] = (), error_summary: str | None = None,
    rejection_summary: str | None = None, parent_experiment: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if event_status not in EVENT_STATUSES:
        raise ValueError(f"Invalid ledger event status: {event_status}")
    if artifact_kind not in ARTIFACT_KINDS:
        raise ValueError(f"Invalid artifact kind: {artifact_kind}")
    if event_status == "COMPLETED" and artifact_kind == "RESEARCH_DIAGNOSTIC":
        raise ValueError("A research diagnostic cannot be a completed selector experiment")
    event = {
        "ledger_contract_version": "experiment_ledger_event_v1",
        "event_id": f"event-{uuid.uuid4().hex}",
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_spec_hash": experiment_spec_hash_value,
        "experiment_run_id": experiment_run_id,
        "event_status": event_status, "artifact_kind": artifact_kind,
        "canonical_model_id": canonical_model_id,
        "requested_model_id": requested_model_id,
        "registry_hashes": dict(registry_hashes), "source_commit": source_commit,
        "runtime_identity": {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(), "pid": os.getpid()},
        "artifact_paths": list(artifact_paths), "error_summary": error_summary,
        "rejection_summary": rejection_summary, "parent_experiment": parent_experiment,
        "metadata": dict(metadata or {}),
    }
    line = canonical_json_line(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        _lock_file(handle)
        try:
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)
    return event


def canonical_json_line(event: Mapping[str, Any]) -> bytes:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def read_ledger(path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"Malformed empty ledger line {line_number}: {path}")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed ledger line {line_number}: {path}: {exc}") from exc
        if not isinstance(event, dict) or event.get("event_status") not in EVENT_STATUSES:
            raise ValueError(f"Invalid committed ledger event on line {line_number}: {path}")
        events.append(event)
    return events


def latest_run_states(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = str(event.get("experiment_run_id", ""))
        if not run_id:
            raise ValueError("Ledger event is missing experiment_run_id")
        latest[run_id] = dict(event)
    return latest
