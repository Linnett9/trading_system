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
