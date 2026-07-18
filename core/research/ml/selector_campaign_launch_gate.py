from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.selector_research_campaign import (
    BASELINE_CAMPAIGN_ID,
    DATES,
    RESEARCH_CAMPAIGN_ID,
    validate_selector_campaign,
)
from core.research.ml.selector_research_protocol import (
    validate_selector_research_protocol,
)


READINESS_CONTRACT = "selector_campaign_launch_readiness.v1"
REQUIRED_OPERATIONAL_INPUTS = (
    "selector_dataset_identity",
    "selector_dataset_checksum",
    "parent_gate_identity",
    "parent_gate_checksum",
    "operational_input_identity",
    "operational_input_checksum",
    "training_boundary_identity",
    "training_boundary_checksum",
)
PHASE_C_MODELS = ("lightgbm_rank_xendcg", "lightgbm_lambdarank")


def build_selector_campaign_launch_readiness(
    *,
    protocol: Mapping[str, Any],
    campaign: Mapping[str, Any],
    campaign_selection: str,
    operational_inputs: Mapping[str, str] | None = None,
    source_commit: str,
    max_component_workers: int = 3,
    weighted_capacity: int = 4,
) -> dict[str, Any]:
    """Validate launch structure without reading datasets or fitting models."""
    try:
        validate_selector_research_protocol(protocol)
        validate_selector_campaign(campaign)
        _validate_selection(campaign, campaign_selection)
    except ValueError as exc:
        return _report(
            protocol, campaign, source_commit, max_component_workers,
            weighted_capacity, status="INVALID_CAMPAIGN",
            blocked=[], missing=[], errors=[str(exc)],
        )
    if max_component_workers < 1 or weighted_capacity < 1:
        return _report(
            protocol, campaign, source_commit, max_component_workers,
            weighted_capacity, status="INVALID_CAMPAIGN",
            blocked=[], missing=[], errors=["Invalid scheduler bounds"],
        )

    matrix = list(campaign["fitted_component_matrix"])
    adapter_errors = []
    for model_id in sorted({str(row["model_id"]) for row in matrix}):
        try:
            selector_model_adapter(model_id, runner="ordinary")
        except (KeyError, ValueError) as exc:
            adapter_errors.append(f"{model_id}: {exc}")
    if adapter_errors:
        status = "BLOCKED_REGISTRY"
        blocked = [str(row["job_id"]) for row in matrix]
    else:
        blocked = _blocked_phase_c(campaign)
        status = "BLOCKED_MODEL_OWNER" if blocked else ""

    inputs = dict(operational_inputs or {})
    missing = [name for name in REQUIRED_OPERATIONAL_INPUTS if not inputs.get(name)]
    if not status:
        if not operational_inputs:
            status = "READY_FOR_OPERATIONAL_INPUTS"
        elif missing:
            status = "BLOCKED_INPUTS"
        elif all(_immutable(value) for value in inputs.values()):
            status = "READY_TO_LAUNCH"
        else:
            status = "READY_FOR_OPERATIONAL_INPUTS"
    return _report(
        protocol, campaign, source_commit, max_component_workers,
        weighted_capacity, status=status, blocked=blocked, missing=missing,
        errors=adapter_errors, operational_inputs=inputs,
    )


def _validate_selection(campaign: Mapping[str, Any], selection: str) -> None:
    expected = {
        "historical": (BASELINE_CAMPAIGN_ID, 15),
        "research": (RESEARCH_CAMPAIGN_ID, 75),
    }
    if selection not in expected:
        raise ValueError("Unknown selector campaign selection")
    campaign_id, count = expected[selection]
    if campaign.get("campaign_id") != campaign_id:
        raise ValueError("Selector campaign identity and selection differ")
    if campaign.get("expected_component_count") != count:
        raise ValueError("Selector campaign selection count differs")


def _blocked_phase_c(campaign: Mapping[str, Any]) -> list[str]:
    if campaign.get("campaign_id") != RESEARCH_CAMPAIGN_ID:
        return []
    readiness = {
        row["model_id"]: row for row in campaign.get("model_readiness", ())
    }
    if all(
        readiness.get(model, {}).get("campaign") == "CAMPAIGN_READY"
        for model in PHASE_C_MODELS
    ):
        return []
    return [
        f"selector:{date}:{model}"
        for date in DATES for model in PHASE_C_MODELS
    ]


def _immutable(value: str) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _report(
    protocol: Mapping[str, Any],
    campaign: Mapping[str, Any],
    source_commit: str,
    workers: int,
    capacity: int,
    *,
    status: str,
    blocked: list[str],
    missing: list[str],
    errors: list[str],
    operational_inputs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    matrix = list(campaign.get("fitted_component_matrix") or [])
    report = {
        "readiness_contract": READINESS_CONTRACT,
        "protocol_identity": protocol.get("protocol_identity"),
        "campaign_id": campaign.get("campaign_id"),
        "campaign_identity": campaign.get("campaign_identity"),
        "campaign_phase_identities": protocol.get("campaign_phase_identities"),
        "expected_component_count": campaign.get("expected_component_count"),
        "validated_component_count": len(matrix),
        "model_readiness": campaign.get("model_readiness", []),
        "blocked_components": blocked,
        "missing_operational_inputs": missing,
        "operational_inputs": dict(operational_inputs or {}),
        "production_command_template": (
            "python scripts/run_selector_component_batch.py "
            "--campaign-selection <historical|research> "
            "--campaign-manifest <campaign.json> --readiness <readiness.json> "
            "--input-inventory <inputs.json> --parent-gate <gate.json> "
            "--experiment-ledger <ledger.jsonl> --output-root <output>"
        ),
        "max_component_workers": workers,
        "weighted_scheduler_capacity": capacity,
        "inner_model_threads": 1,
        "source_git_commit": source_commit,
        "errors": errors,
        "readiness_status": status,
        "training_performed": False,
        "publication_performed": False,
        "dataset_read_performed": False,
    }
    report["logical_checksum"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()
    return report
