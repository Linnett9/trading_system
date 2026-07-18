from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.registries import load_registry_bundle
from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_component_rows import (
    EVALUATION_OUTCOMES_CONTRACT,
    PREDICTION_ROWS_CONTRACT,
    TRAINING_ROWS_CONTRACT,
)
from core.research.ml.selector_component_scheduler import WEIGHTS
from core.research.ml.selector_research_campaign import (
    BASELINE_CAMPAIGN_ID,
    RESEARCH_CAMPAIGN_ID,
    validate_selector_campaign,
)
from core.research.ml.selector_research_protocol import (
    validate_selector_research_protocol,
)


PLAN_CONTRACT = "selector_operational_component_plan.v2"
ORDINARY_RUNNER = (
    "core.research.ml.stock_level_benchmark_models:"
    "TabularModelFactory/SequenceModelFactory"
)
WAVE4_RUNNER = (
    "core.research.ml.stock_level.wave4_selector_integration:"
    "publish_wave4_component"
)
ORDINARY_PUBLICATION_OWNER = (
    "core.research.ml.stock_level.ordinary_selector_publication:"
    "publish_planned_ordinary_component"
)
PROFILE_BY_MODEL = {
    "ridge": "ORDINARY_TABULAR",
    "elastic_net": "ORDINARY_TABULAR",
    "ordered_logit_ranker": "ORDINARY_TABULAR",
    "huber": "WAVE4_TABULAR",
    "contextual_elastic_net": "WAVE4_CONTEXTUAL",
    "multi_horizon_ridge": "WAVE4_MULTI_HORIZON",
    "multi_horizon_elastic_net": "WAVE4_MULTI_HORIZON",
    "lightgbm_rank_xendcg": "WAVE4_GROUPED_RANKING",
    "lightgbm_lambdarank": "WAVE4_GROUPED_RANKING",
}
BASE_SOURCE_GUARANTEES = {
    "selector_row_identity", "symbol_identity", "decision_date",
    "label_availability_timestamp", "forward_return_10d",
    "ordinary_registered_feature_order", "decision_date_group_ownership",
}
PARENT_IDENTITIES = (
    "parent_gate_identity", "parent_gate_checksum",
    "selector_dataset_identity", "selector_dataset_checksum",
    "symbol_registry_identity", "symbol_registry_checksum",
    "daily_spine_identity", "daily_spine_checksum",
)


def build_selector_operational_plan(
    *,
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any],
    campaign_selection: str,
    parent_identities: Mapping[str, str],
    source_git_commit: str,
    source_schema_guarantees: Sequence[str] = (),
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create a plan only; this owner never reads rows or publishes packages."""
    validate_selector_campaign(campaign)
    validate_selector_research_protocol(protocol)
    _validate_selection(campaign, campaign_selection)
    if campaign.get("campaign_version") == "v2" and (
        campaign.get("protocol_identity") != protocol.get("protocol_identity")
        or campaign.get("protocol_logical_checksum")
        != protocol.get("logical_checksum")
    ):
        raise ValueError("Campaign and protocol identities differ")
    if not str(source_git_commit or ""):
        raise ValueError("Source Git commit is required")

    matrix = list(campaign["fitted_component_matrix"])
    guarantees = BASE_SOURCE_GUARANTEES | {
        str(value) for value in source_schema_guarantees
    }
    jobs = [
        _plan_job(
            row=row, campaign=campaign, guarantees=guarantees,
            source_git_commit=source_git_commit,
        )
        for row in matrix
    ]
    _validate_jobs(jobs, matrix, campaign)
    missing_parents = [
        field for field in PARENT_IDENTITIES
        if not parent_identities.get(field)
    ]
    blocked_jobs = [
        {"job_id": row["job_id"], "blockers": row["blockers"]}
        for row in jobs if row["blockers"]
    ]
    if missing_parents:
        readiness = "BLOCKED_PARENT_IDENTITIES"
    elif blocked_jobs:
        readiness = "PARTIALLY_BLOCKED_INPUT_SCHEMA"
    else:
        readiness = "READY_FOR_PACKAGE_PUBLICATION"
    bundle = load_registry_bundle()
    logical = {
        "plan_contract_version": PLAN_CONTRACT,
        "selected_campaign": campaign_selection,
        "campaign_id": campaign["campaign_id"],
        "campaign_version": campaign["campaign_version"],
        "campaign_identity": campaign["campaign_identity"],
        "campaign_logical_checksum": campaign["logical_checksum"],
        "protocol_identity": protocol["protocol_identity"],
        "protocol_logical_checksum": protocol["logical_checksum"],
        "expected_component_count": len(matrix),
        "deterministic_component_ordering": campaign[
            "deterministic_ordering"
        ],
        "parent_identity_requirements": dict(parent_identities),
        "missing_parent_identities": missing_parents,
        "selector_model_registry_set_checksum": bundle.registry_set_hash,
        "source_schema_guarantees": sorted(guarantees),
        "source_git_commit": source_git_commit,
        "jobs": jobs,
        "blocked_jobs": blocked_jobs,
        "readiness_status": readiness,
        "campaign_membership_valid": True,
        "source_schema_readiness_complete": not blocked_jobs,
        "artifact_inventory_complete": False,
        "production_rows_read": False,
        "packages_published": False,
        "wave4_fit_inputs_published": False,
        "fitting_performed": False,
        "predictions_published": False,
        "evaluation_performed": False,
    }
    logical["plan_identity"] = canonical_hash({
        "campaign_identity": campaign["campaign_identity"],
        "protocol_identity": protocol["protocol_identity"],
        "parent_identities": dict(parent_identities),
        "jobs": jobs,
    })
    logical["logical_checksum"] = canonical_hash(logical)
    publication = (
        _publish_plan(output_path, logical)
        if output_path is not None else "NOT_REQUESTED"
    )
    return {**logical, "publication_result": publication}


def validate_selector_operational_plan(
    plan: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
) -> None:
    validate_selector_campaign(campaign)
    payload = {
        key: value for key, value in plan.items()
        if key not in {"logical_checksum", "publication_result"}
    }
    if plan.get("plan_contract_version") != PLAN_CONTRACT:
        raise ValueError("Operational-plan contract mismatch")
    if plan.get("logical_checksum") != canonical_hash(payload):
        raise ValueError("Operational-plan checksum mismatch")
    _validate_jobs(
        list(plan.get("jobs") or ()),
        list(campaign["fitted_component_matrix"]),
        campaign,
    )


def _plan_job(
    *,
    row: Mapping[str, Any],
    campaign: Mapping[str, Any],
    guarantees: set[str],
    source_git_commit: str,
) -> dict[str, Any]:
    model_id = str(row["model_id"])
    if model_id not in PROFILE_BY_MODEL:
        raise ValueError(f"No operational input profile for {model_id}")
    adapter = selector_model_adapter(model_id, runner="ordinary")
    declared_runner = str(
        row.get("component_runner")
        or (ORDINARY_RUNNER if campaign["campaign_version"] == "v1" else "")
    )
    if declared_runner != adapter.constructor_owner:
        raise ValueError(
            f"Campaign runner disagrees with registry: {row['job_id']}"
        )
    profile = PROFILE_BY_MODEL[model_id]
    requirements = _requirements(model_id, profile, adapter)
    missing = sorted(
        set(requirements["required_source_guarantees"]) - guarantees
    )
    readiness = (
        "SOURCE_SCHEMA_REQUIRED" if missing
        else (
            "DEPENDENCY_REQUIRED"
            if adapter.dependency_requirements else "CONTRACT_READY"
        )
    )
    blockers = [f"SOURCE_CONTRACT_MISSING:{field}" for field in missing]
    component_identity = canonical_hash({
        "campaign_identity": campaign["campaign_identity"],
        "job_id": row["job_id"], "model_id": model_id,
        "prediction_date": row["prediction_date"],
        "horizon_id": row.get("horizon_id"),
    })
    plan_job_identity = canonical_hash({
        "component_identity": component_identity,
        "component_runner": declared_runner,
        "operational_input_profile": profile,
        "requirements": requirements,
    })
    return {
        "job_id": row["job_id"],
        "campaign_identity": campaign["campaign_identity"],
        "campaign_phase": row.get("phase_id") or "historical",
        "plan_job_identity": plan_job_identity,
        "component_identity": component_identity,
        "model_id": model_id,
        "prediction_date": row["prediction_date"],
        "horizon_id": row.get("horizon_id"),
        "target_contract": row.get("target_contract") or adapter.target_contract,
        "component_runner": declared_runner,
        "runtime_publication_owner": (
            ORDINARY_PUBLICATION_OWNER
            if declared_runner == ORDINARY_RUNNER else WAVE4_RUNNER
        ),
        "operational_input_profile": profile,
        "training_row_contract": TRAINING_ROWS_CONTRACT,
        "prediction_row_contract": PREDICTION_ROWS_CONTRACT,
        "fit_validation_contract": None,
        "evaluation_outcome_contract": EVALUATION_OUTCOMES_CONTRACT,
        **requirements,
        "dependency_requirements": list(adapter.dependency_requirements),
        "model_registry_entry_checksum": adapter.entry_hash,
        "model_configuration_identity": (
            adapter.fitting_configuration_checksum or adapter.entry_hash
        ),
        "seed": 1729 if model_id.startswith("lightgbm_") else 42,
        "inner_thread_count": 1,
        "scheduler_weight": WEIGHTS[model_id],
        "package_publication_requirements": {
            "atomic": True, "logical_checksum": True,
            "artifact_sha256": True,
            "wave4_fit_input_required": declared_runner == WAVE4_RUNNER,
        },
        "source_git_commit": source_git_commit,
        "readiness_status": readiness,
        "blockers": blockers,
    }


def _requirements(model_id: str, profile: str, adapter) -> dict[str, Any]:
    common = {
        "required_feature_profile": adapter.feature_schema,
        "required_target_fields": ["actual_forward_return_10d"],
        "required_context_fields": [],
        "required_grouping_fields": [],
        "required_relevance_label_fields": [],
        "required_source_guarantees": [
            "selector_row_identity", "symbol_identity", "decision_date",
            "label_availability_timestamp", "forward_return_10d",
            "ordinary_registered_feature_order",
        ],
        "prediction_prohibited_fields": [
            "target_value", "target_values", "actual_forward_return_*",
            "relevance_label", "label", "evaluation_outcome",
        ],
    }
    if profile == "WAVE4_CONTEXTUAL":
        common.update(
            required_context_fields=[
                "registered_stock_feature_order",
                "registered_market_context_order",
                "point_in_time_context_evidence",
                "registered_interaction_identity",
            ],
            required_source_guarantees=[
                *common["required_source_guarantees"],
                "contextual_stock_features", "point_in_time_market_context",
            ],
        )
    elif profile == "WAVE4_MULTI_HORIZON":
        common.update(
            required_target_fields=[
                "actual_forward_return_1d", "actual_forward_return_5d",
                "actual_forward_return_10d", "actual_forward_return_20d",
                "horizon_target_maturity_timestamps",
                "horizon_target_availability_states",
            ],
            required_source_guarantees=[
                *common["required_source_guarantees"],
                "forward_return_1d", "forward_return_5d",
                "forward_return_20d", "horizon_maturity_evidence",
            ],
        )
    elif profile == "WAVE4_GROUPED_RANKING":
        common.update(
            required_grouping_fields=[
                "decision_date_group", "canonical_within_group_order",
                "group_size_vector", "split_role",
            ],
            required_relevance_label_fields=[
                "training_integer_relevance_label",
                str(adapter.relevance_contract),
            ],
            required_source_guarantees=[
                *common["required_source_guarantees"],
                "decision_date_group_ownership",
                "tree_cross_sectional_features",
            ],
        )
        if model_id == "lightgbm_lambdarank":
            common["label_gain_policy"] = (
                "exponential_gain_quintile_0_4_v1"
            )
        common.update(
            objective_identity=adapter.objective_identity,
            grouped_query_contract=adapter.grouped_query_contract,
            dependency_preflight_identity=adapter.dependency_preflight_identity,
            metric_identity="ndcg",
        )
    elif model_id == "ordered_logit_ranker":
        common.update(
            required_relevance_label_fields=[
                "training_integer_relevance_label",
                str(adapter.relevance_contract),
            ],
            required_grouping_fields=["decision_date_group"],
        )
    return common


def _validate_selection(
    campaign: Mapping[str, Any], selection: str
) -> None:
    expected = {
        "historical": (BASELINE_CAMPAIGN_ID, "v1", 15),
        "research": (RESEARCH_CAMPAIGN_ID, "v2", 75),
    }
    if selection not in expected:
        raise ValueError("Unknown campaign selection")
    campaign_id, version, count = expected[selection]
    if (
        campaign.get("campaign_id") != campaign_id
        or campaign.get("campaign_version") != version
        or campaign.get("expected_component_count") != count
    ):
        raise ValueError("Campaign selection identity/cardinality mismatch")


def _validate_jobs(
    jobs: Sequence[Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
    campaign: Mapping[str, Any],
) -> None:
    if len(jobs) != len(matrix):
        raise ValueError("Operational jobs differ from campaign cardinality")
    expected_ids = [str(row["job_id"]) for row in matrix]
    actual_ids = [str(row.get("job_id") or "") for row in jobs]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Operational job inventory/order differs from campaign")
    owners = [
        (
            row.get("model_id"), row.get("prediction_date"),
            row.get("horizon_id"),
        )
        for row in jobs
    ]
    if len(owners) != len(set(owners)):
        raise ValueError("Duplicate operational component ownership")
    for job, expected in zip(jobs, matrix):
        for field in ("model_id", "prediction_date", "horizon_id"):
            if (job.get(field) or None) != (expected.get(field) or None):
                raise ValueError(
                    f"Operational job {field} differs from campaign"
                )
        if job.get("campaign_identity") != campaign["campaign_identity"]:
            raise ValueError("Operational job campaign identity mismatch")


def _publish_plan(path: Path, plan: Mapping[str, Any]) -> str:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("logical_checksum") == plan["logical_checksum"]
            and existing.get("logical_checksum") == canonical_hash({
                key: value for key, value in existing.items()
                if key != "logical_checksum"
            })
        ):
            return "SKIPPED_COMPATIBLE"
        raise FileExistsError(f"Incompatible operational plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)
    return "PUBLISHED"
