from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.selector_component_readiness import audit_component_commands


RUN_STATE_VERSION = "selector_parent_publication_run_state_v2"


@dataclass(frozen=True)
class StageDefinition:
    stage_number: int
    name: str
    mutating: bool
    resumable: bool
    skippable: bool
    expected: str
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    exit_semantics: str = ""


STAGES = (
    StageDefinition(1, "canonical registry preflight", False, True, False, "READY"),
    StageDefinition(2, "canonical registry publication", True, True, True, "complete"),
    StageDefinition(3, "canonical registry verification", False, True, False, "READY"),
    StageDefinition(4, "daily-spine preflight", False, True, False, "READY"),
    StageDefinition(5, "daily-spine publication", True, True, True, "READY"),
    StageDefinition(6, "daily-spine verification", False, True, False, "READY"),
    StageDefinition(7, "selector dataset preflight", False, True, False, "READY"),
    StageDefinition(8, "selector dataset rebuild", True, True, True, "VERIFIED"),
    StageDefinition(9, "selector dataset validation", False, True, False, "READY"),
    StageDefinition(10, "component readiness preflight", False, True, False, "READY"),
    StageDefinition(11, "guarded component production", True, True, False, "complete"),
    StageDefinition(12, "monitoring", False, True, False, "complete"),
    StageDefinition(13, "resume", False, True, False, "complete"),
    StageDefinition(14, "component validation", False, True, False, "VERIFIED_STRICT_OOS"),
    StageDefinition(15, "registry verification", False, True, False, "VERIFIED"),
    StageDefinition(16, "panel refreezing", True, True, True, "READY"),
)


def validate_stage_request(from_stage: int = 1, through_stage: int = 10, *, allow_selector_fits: bool = False) -> None:
    if not 1 <= from_stage <= 16 or not 1 <= through_stage <= 16 or from_stage > through_stage:
        raise ValueError("Invalid stage range")
    if through_stage >= 11 and not allow_selector_fits:
        raise PermissionError("Stage 11+ requires explicit selector-fit approval")


def validate_fresh_component_preflight(
    payload: Mapping[str, Any], *, expected_path: Path, actual_path: Path,
    stage_started_at: datetime, expected_dataset_root: Path,
) -> dict[str, Any]:
    reasons = []
    if actual_path.resolve() != expected_path.resolve(): reasons.append("AUTHORITATIVE_PREFLIGHT_PATH_MISMATCH")
    if not actual_path.exists() or datetime.fromtimestamp(actual_path.stat().st_mtime, timezone.utc) < stage_started_at: reasons.append("STALE_COMPONENT_PREFLIGHT")
    if payload.get("preflight_schema_version") != "selector_component_production_preflight_v1": reasons.append("PREFLIGHT_SCHEMA_MISMATCH")
    if payload.get("component_count") != 15: reasons.append("COMPONENT_COUNT_NOT_15")
    if payload.get("fitting_performed") is not False: reasons.append("PREFLIGHT_PERFORMED_FITTING")
    if payload.get("prediction_performed") is not False: reasons.append("PREFLIGHT_PERFORMED_PREDICTION")
    jobs = payload.get("jobs", []) if isinstance(payload.get("jobs"), list) else []
    command_audit = audit_component_commands(jobs, expected_dataset_root=expected_dataset_root)
    reasons.extend(command_audit["blocking_reasons"])
    for field in ("dataset_identity", "daily_spine_identity", "symbol_registry_identity", "daily_feature_store_identity"):
        if not payload.get(field): reasons.append(f"MISSING_{field.upper()}")
    if payload.get("status") != "READY" or payload.get("blocking_reasons"): reasons.extend(payload.get("blocking_reasons") or ["PREFLIGHT_NOT_READY"])
    return {"status": "READY" if not reasons else "BLOCKED", "blocking_reasons": sorted(set(reasons)), "command_binding_audit": command_audit["audit"]}


def _new_stage_state(stage: StageDefinition) -> dict[str, Any]:
    definition = asdict(stage)
    definition["expected_inputs"] = list(stage.expected_inputs)
    definition["expected_outputs"] = list(stage.expected_outputs)
    return {
        **definition,
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "exit_code": None,
        "error": None,
        "command": None,
        "command_history": [],
        "observed_inputs": [],
        "produced_outputs": [],
        "reused_artifact_identities": [],
        "attempt_count": 0,
        "freshness_metadata": {
            "run_id": None,
            "source_commit": None,
            "parent_artifact_identities": {},
            "stage_started_file_time_utc": None,
        },
    }


def new_run_state(*, run_id: str, repository_path: Path, source_commit: str, from_stage: int, through_stage: int, allow_selector_fits: bool) -> dict[str, Any]:
    stages = [_new_stage_state(stage) for stage in STAGES]
    for stage in stages:
        stage["freshness_metadata"]["run_id"] = run_id
        stage["freshness_metadata"]["source_commit"] = source_commit
    return {"run_state_version": RUN_STATE_VERSION, "run_id": run_id, "start_timestamp": datetime.now(timezone.utc).isoformat(), "repository_path": str(repository_path.resolve()), "source_commit": source_commit, "requested_stage_range": {"from": from_stage, "through": through_stage}, "allow_selector_fits": allow_selector_fits, "stages": stages, "artifacts": {}}


def write_run_state_atomic(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def load_run_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "stage_number", "name", "mutating", "resumable", "skippable", "expected",
        "expected_inputs", "expected_outputs", "exit_semantics", "status", "started_at", "completed_at",
        "exit_code", "error", "command", "command_history", "observed_inputs",
        "produced_outputs", "reused_artifact_identities", "attempt_count",
        "freshness_metadata",
    }
    stages = payload.get("stages")
    if payload.get("run_state_version") != RUN_STATE_VERSION or not isinstance(stages, list) or len(stages) != 16:
        raise ValueError("INCOMPATIBLE_RUN_STATE_SCHEMA: create a new RunId")
    if any(not required.issubset(stage) for stage in stages) or [stage["stage_number"] for stage in stages] != list(range(1, 17)):
        raise ValueError("INCOMPATIBLE_RUN_STATE_SCHEMA: create a new RunId")
    return payload
