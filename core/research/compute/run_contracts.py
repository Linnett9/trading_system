from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

RUN_MANIFEST_CONTRACT = "compute_run_manifest.v1"
RUN_STATUS_CONTRACT = "compute_run_status.v1"
RUN_ITEM_CONTRACT = "compute_run_item_status.v1"
RESULT_RECORD_CONTRACT = "compute_result_record.v1"
METRIC_CONTRACT = "compute_metric_value.v1"
FAILURE_RECORD_CONTRACT = "compute_failure_record.v1"
BLOCKER_RECORD_CONTRACT = "compute_blocker_record.v1"


class RunStatus(str, Enum):
    PLANNED = "PLANNED"
    INPUTS_READY = "INPUTS_READY"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"
    RUNNING = "RUNNING"
    PARTIALLY_COMPLETE = "PARTIALLY_COMPLETE"
    COMPONENTS_COMPLETE = "COMPONENTS_COMPLETE"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ITEM_STATUSES = {
    "PLANNED", "INPUTS_READY", "WAITING_FOR_RESOURCES", "RUNNING",
    "COMPLETE", "SKIPPED_COMPATIBLE", "BLOCKED", "FAILED", "CANCELLED",
    "INCOMPLETE", "CORRUPT",
}
RESULT_KINDS = {
    "MODEL_COMPONENT", "MODEL_CAMPAIGN", "NEWS_SCORING", "FEATURE_STORE",
    "DATASET_BUILD", "REPLAY", "POLICY_SWEEP", "EVALUATION", "SAFEGUARD",
    "DATA_STAGE", "GENERIC_STAGE",
}
METRIC_DIRECTIONS = {
    "HIGHER_IS_BETTER", "LOWER_IS_BETTER", "TARGET_VALUE", "INFORMATIONAL",
}
METRIC_AVAILABILITY = {
    "AVAILABLE", "NOT_APPLICABLE", "NOT_COMPUTED", "BLOCKED", "INVALID",
}


def checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def semantic_payload(payload: Mapping[str, Any], *, mutable: bool = False) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    excluded = {
        "logical_checksum", "creation_timestamp", "latest_update_timestamp",
        "started_timestamp", "completed_timestamp",
    }
    if mutable:
        excluded.add("state_revision")
    for field in excluded:
        result.pop(field, None)
    return result


def build_run_manifest(
    *,
    run_id: str,
    pipeline: str,
    stage: str,
    run_purpose: str,
    source_git_commit: str,
    configuration_identity: str,
    configuration_checksum: str,
    machine_profile_identity: str,
    requested_resource_profile_identity: str,
    parent_input_artifacts: Sequence[Mapping[str, Any]],
    expected_inventory: Sequence[Mapping[str, Any]],
    reports_root: str = "reports/runs",
    campaign_identity: str | None = None,
    source_policy_identity: str = "source_commit_required",
    creation_timestamp: str | None = None,
) -> dict[str, Any]:
    ordered = sorted(
        (dict(row) for row in expected_inventory),
        key=lambda row: (int(row["ordered_position"]), str(row["item_id"])),
    )
    if len({row["item_id"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate run item identity")
    relative = f"{pipeline}/{stage}/{run_id}"
    manifest = {
        "contract_version": RUN_MANIFEST_CONTRACT,
        "run_id": run_id,
        "pipeline": pipeline,
        "stage": stage,
        "campaign_identity": campaign_identity,
        "run_purpose": run_purpose,
        "source_git_commit": source_git_commit,
        "source_policy_identity": source_policy_identity,
        "configuration_identity": configuration_identity,
        "configuration_checksum": configuration_checksum,
        "machine_profile_identity": machine_profile_identity,
        "requested_resource_profile_identity": requested_resource_profile_identity,
        "parent_input_artifacts": [dict(row) for row in parent_input_artifacts],
        "expected_inventory": ordered,
        "deterministic_expected_ordering": [row["item_id"] for row in ordered],
        "expected_job_count": len(ordered),
        "run_root_relative_path": relative,
        "creation_timestamp": creation_timestamp or datetime.now(timezone.utc).isoformat(),
        "execution_claims": {
            "execution_started": False, "components_complete": False,
            "evaluation_complete": False,
        },
        "promotion_claims": {"promoted": False, "winner_selected": False},
    }
    required = (
        run_id, pipeline, stage, run_purpose, source_git_commit,
        configuration_identity, configuration_checksum, machine_profile_identity,
        requested_resource_profile_identity,
    )
    if not all(required) or not parent_input_artifacts or not ordered:
        raise ValueError("Run manifest identity, ancestry, and inventory are required")
    manifest["run_identity"] = checksum(semantic_payload(manifest))
    manifest["compatibility_identity"] = manifest["run_identity"]
    manifest["logical_checksum"] = checksum(semantic_payload(manifest))
    validate_run_manifest(manifest, reports_root=reports_root)
    return manifest


def validate_run_manifest(
    manifest: Mapping[str, Any], *, reports_root: str = "reports/runs"
) -> None:
    if manifest.get("contract_version") != RUN_MANIFEST_CONTRACT:
        raise ValueError("Run manifest contract mismatch")
    expected_root = f"{manifest.get('pipeline')}/{manifest.get('stage')}/{manifest.get('run_id')}"
    if manifest.get("run_root_relative_path") != expected_root:
        raise ValueError("Run identity and root ownership disagree")
    inventory = manifest.get("expected_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Expected run inventory is required")
    ids = [row.get("item_id") for row in inventory]
    if len(ids) != len(set(ids)) or manifest.get("expected_job_count") != len(ids):
        raise ValueError("Run inventory count or identity mismatch")
    if manifest.get("deterministic_expected_ordering") != ids:
        raise ValueError("Run inventory ordering mismatch")
    if manifest.get("run_identity") != manifest.get("compatibility_identity"):
        raise ValueError("Run compatibility identity mismatch")
    if manifest.get("logical_checksum") != checksum(semantic_payload(manifest)):
        raise ValueError("Run manifest logical checksum mismatch")


def build_item_status(
    *, run_identity: str, item_id: str, ordered_position: int,
    pipeline: str, stage: str, attempt_identity: str, status: str,
    dependency_status: str = "READY", **evidence: Any,
) -> dict[str, Any]:
    if status not in ITEM_STATUSES:
        raise ValueError("Unsupported run item status")
    row = {
        "contract_version": RUN_ITEM_CONTRACT,
        "run_identity": run_identity, "item_id": item_id,
        "ordered_position": ordered_position, "pipeline": pipeline,
        "stage": stage, "attempt_identity": attempt_identity, "status": status,
        "dependency_status": dependency_status,
        "model_id": None, "date_identity": None, "horizon_identity": None,
        "fold_identity": None, "lease_identity": None,
        "resource_request_identity": None, "telemetry_identity": None,
        "resource_summary_identity": None, "fitted_model_artifact_identity": None,
        "prediction_artifact_identity": None, "stage_artifact_identity": None,
        "result_record_identity": None, "started_timestamp": None,
        "completed_timestamp": None, "exit_code": None, "blocker_code": None,
        "blocker_reason": None, "failure_code": None, "failure_reason": None,
        "retryable": False, "compatible_skip_evidence": None,
        "artifact_validation": {},
        **evidence,
    }
    _validate_complete_item(row)
    row["logical_checksum"] = checksum(semantic_payload(row))
    return row


def validate_item_status(row: Mapping[str, Any]) -> None:
    if row.get("contract_version") != RUN_ITEM_CONTRACT:
        raise ValueError("Run item contract mismatch")
    if row.get("status") not in ITEM_STATUSES:
        raise ValueError("Unsupported run item status")
    _validate_complete_item(row)
    if row.get("logical_checksum") != checksum(semantic_payload(row)):
        raise ValueError("Run item checksum mismatch")


def metric_value(
    metric_id: str, value: float | None, *, unit: str,
    population_identity: str, direction: str, availability: str,
    source_artifact_identity: str | None,
) -> dict[str, Any]:
    if direction not in METRIC_DIRECTIONS or availability not in METRIC_AVAILABILITY:
        raise ValueError("Metric direction or availability is invalid")
    if availability == "AVAILABLE" and value is None:
        raise ValueError("Available metric requires a value")
    if availability != "AVAILABLE" and value is not None:
        raise ValueError("Unavailable metric value must remain null")
    return {
        "contract_version": METRIC_CONTRACT, "metric_id": metric_id,
        "value": value, "unit": unit, "population_identity": population_identity,
        "direction": direction, "availability": availability,
        "source_artifact_identity": source_artifact_identity,
    }


def build_result_record(
    *, result_identity: str, run_identity: str, item_identity: str,
    result_kind: str, pipeline: str, stage: str, status: str,
    artifact_identities: Sequence[str], metrics: Mapping[str, Mapping[str, Any]],
    dimensions: Mapping[str, Any] | None = None, **optional: Any,
) -> dict[str, Any]:
    if result_kind not in RESULT_KINDS:
        raise ValueError("Unsupported result kind")
    for key, metric in metrics.items():
        if metric.get("metric_id") != key:
            raise ValueError("Result metric identity mismatch")
        metric_value(
            key, metric.get("value"), unit=str(metric.get("unit") or ""),
            population_identity=str(metric.get("population_identity") or ""),
            direction=str(metric.get("direction") or ""),
            availability=str(metric.get("availability") or ""),
            source_artifact_identity=metric.get("source_artifact_identity"),
        )
    row = {
        "contract_version": RESULT_RECORD_CONTRACT,
        "result_identity": result_identity, "run_identity": run_identity,
        "item_identity": item_identity, "result_kind": result_kind,
        "pipeline": pipeline, "stage": stage, "status": status,
        "artifact_identities": list(artifact_identities),
        "model_id": None, "horizon_identity": None, "fold_identity": None,
        "date_identity": None, "counts": {}, "resource_summary": {},
        "primary_metric_identity": None, "metrics": dict(metrics),
        "dimensions": dict(dimensions or {}), "warnings": [], "blockers": [],
        "failures": [], "eligibility_state": "NOT_EVALUATED",
        "promotion_state": "NOT_PROMOTED", **optional,
    }
    row["logical_checksum"] = checksum(semantic_payload(row))
    return row


def failure_record(
    *, run_identity: str, item_identity: str, failure_code: str,
    failure_category: str, message: str, phase: str, retryable: bool,
    exit_code: int | None = None, stderr_log_reference: str | None = None,
    first_occurrence: str, last_occurrence: str,
    related_artifact_identity: str | None = None,
    related_lease_identity: str | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": FAILURE_RECORD_CONTRACT,
        "run_identity": run_identity, "item_identity": item_identity,
        "failure_code": failure_code, "failure_category": failure_category,
        "message": message, "phase": phase, "retryable": retryable,
        "exit_code": exit_code, "stderr_log_reference": stderr_log_reference,
        "first_occurrence": first_occurrence, "last_occurrence": last_occurrence,
        "related_artifact_identity": related_artifact_identity,
        "related_lease_identity": related_lease_identity,
    }


def blocker_record(
    *, run_identity: str, blocker_code: str, blocker_category: str,
    dependency: str, affected_jobs: Sequence[str],
    operator_action_required: str | None, automatically_resolvable: bool,
    evidence_artifact_identity: str | None,
) -> dict[str, Any]:
    return {
        "contract_version": BLOCKER_RECORD_CONTRACT,
        "run_identity": run_identity, "blocker_code": blocker_code,
        "blocker_category": blocker_category, "dependency": dependency,
        "affected_jobs": list(affected_jobs),
        "operator_action_required": operator_action_required,
        "automatically_resolvable": automatically_resolvable,
        "evidence_artifact_identity": evidence_artifact_identity,
    }


def derive_run_status(
    items: Sequence[Mapping[str, Any]], *, inputs_valid: bool,
    evaluation_required: bool = False, evaluation_artifacts_valid: bool = False,
    cancelled: bool = False, fail_run_on_required_failure: bool = True,
) -> str:
    statuses = [str(row["status"]) for row in items]
    if cancelled:
        return RunStatus.CANCELLED.value
    if any(status == "FAILED" for status in statuses) and fail_run_on_required_failure:
        return RunStatus.FAILED.value
    if any(status in {"BLOCKED", "CORRUPT"} for status in statuses):
        return RunStatus.BLOCKED.value
    finished = {"COMPLETE", "SKIPPED_COMPATIBLE"}
    if statuses and all(status in finished for status in statuses):
        return (
            RunStatus.EVALUATION_COMPLETE.value
            if evaluation_required and evaluation_artifacts_valid
            else RunStatus.COMPONENTS_COMPLETE.value
        )
    if any(status in finished for status in statuses):
        return RunStatus.PARTIALLY_COMPLETE.value
    if any(status == "RUNNING" for status in statuses):
        return RunStatus.RUNNING.value
    if any(status == "WAITING_FOR_RESOURCES" for status in statuses):
        return RunStatus.WAITING_FOR_RESOURCES.value
    if inputs_valid and all(status in {"PLANNED", "INPUTS_READY"} for status in statuses):
        return RunStatus.INPUTS_READY.value
    return RunStatus.PLANNED.value


def _validate_complete_item(row: Mapping[str, Any]) -> None:
    if row.get("status") not in {"COMPLETE", "SKIPPED_COMPATIBLE"}:
        return
    validation = row.get("artifact_validation")
    kind = row.get("required_artifact_kind", "NONE")
    if not isinstance(validation, Mapping):
        raise ValueError("Completed item requires artifact validation")
    if kind == "MODEL" and not (
        validation.get("fitted_model_valid")
        and (
            not row.get("predictions_required")
            or (
                validation.get("prediction_valid")
                and validation.get("prediction_model_binding_valid")
            )
        )
    ):
        raise ValueError("Complete model item lacks valid model/prediction evidence")
    if kind == "STAGE" and not validation.get("stage_artifact_valid"):
        raise ValueError("Complete stage item lacks valid stage artifact")
