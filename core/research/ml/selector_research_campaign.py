from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.selector_research_protocol import (
    validate_selector_research_protocol,
)


CAMPAIGN_CONTRACT = "selector_research_campaign.v1"
BASELINE_CAMPAIGN_ID = "selector_stage10_historical_baseline.v1"
RESEARCH_CAMPAIGN_ID = "selector_research_backed_campaign.v2"
DATES = (
    "2024-03-15", "2024-09-16", "2025-03-17",
    "2025-09-15", "2026-03-16",
)
HORIZONS = ("return_1s", "return_5s", "return_10s", "return_20s")


def build_selector_research_campaign(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    validate_selector_research_protocol(protocol)
    phases = [
        {
            "phase_id": "phase_a",
            "role": "BASELINE_CONTROL",
            "models": ["ridge", "elastic_net", "ordered_logit_ranker"],
            "horizon_policy": "single_registered_10_session_target",
            "status": "CAMPAIGN_READY",
        },
        {
            "phase_id": "phase_b",
            "role": "ROBUST_CONTEXTUAL_CHALLENGER",
            "models": [
                "huber", "contextual_elastic_net",
                "multi_horizon_ridge", "multi_horizon_elastic_net",
            ],
            "horizon_policy": "explicit_for_multi_horizon_models",
            "status": "CAMPAIGN_READY",
        },
        {
            "phase_id": "phase_c",
            "role": "RANKING_TREE_CHALLENGER",
            "models": [
                "lightgbm_rank_xendcg", "lightgbm_lambdarank"
            ],
            "horizon_policy": "grouped_integer_relevance_10_session_target",
            "status": "CAMPAIGN_READY",
        },
    ]
    fitted = []
    for phase in phases:
        for model in phase["models"]:
            adapter = selector_model_adapter(model, runner="ordinary")
            horizons = HORIZONS if model.startswith("multi_horizon_") else (None,)
            for date in DATES:
                for horizon in horizons:
                    fitted.append(
                        {
                            "job_id": _job_id(model, date, horizon),
                            "phase_id": phase["phase_id"],
                            "component_role": "FITTED_MODEL",
                            "model_id": model,
                            "prediction_date": date,
                            "horizon_id": horizon,
                            "model_registry_entry_checksum": adapter.entry_hash,
                            "component_runner": adapter.constructor_owner,
                            "feature_schema": adapter.feature_schema,
                            "target_contract": (
                                _target(horizon)
                                if horizon else adapter.target_contract
                            ),
                            "ranking_contract": adapter.ranking_problem_contract,
                            "ranking_objective_identity": (
                                adapter.objective_identity
                            ),
                            "grouped_query_contract": (
                                adapter.grouped_query_contract
                            ),
                            "fitting_configuration_checksum": (
                                adapter.fitting_configuration_checksum
                            ),
                            "dependency_preflight_identity": (
                                adapter.dependency_preflight_identity
                            ),
                            "dependency_requirements": list(
                                adapter.dependency_requirements
                            ),
                        }
                    )
    fitted.sort(
        key=lambda row: (
            row["phase_id"], row["prediction_date"], row["model_id"],
            row["horizon_id"] or "",
        )
    )
    diagnostics = [
        {
            "component_role": "DIAGNOSTIC_NON_FITTED",
            "model_id": model,
            "runner": "deterministic_baseline_scores",
        }
        for model in ("momentum_120d", "risk_adjusted_momentum")
    ]
    readiness = [
        *[
            {
                "model_id": model,
                "implementation": "IMPLEMENTED",
                "registration": "REGISTERED",
                "adapter": "ADAPTER_READY",
                "campaign": "CAMPAIGN_READY",
                "blockers": [],
            }
            for model in (
                "ridge", "elastic_net", "ordered_logit_ranker", "huber",
                "contextual_elastic_net", "multi_horizon_ridge",
                "multi_horizon_elastic_net",
            )
        ],
        {
            "model_id": "multi_horizon_ordered_logit",
            "implementation": "IMPLEMENTED",
            "registration": "REGISTERED",
            "adapter": "ADAPTER_READY",
            "campaign": "DEFERRED",
            "blockers": [
                "first campaign restricts multi-horizon challengers to linear families"
            ],
        },
        *[
            {
                "model_id": model,
                "implementation": "AUTHORITATIVE_PRODUCTION_OWNER",
                "registration": "REGISTERED",
                "adapter": "ADAPTER_READY",
                "campaign": "CAMPAIGN_READY",
                "execution_readiness": "CAMPAIGN_READY",
                "blockers": [],
                "required_ranking_contract": (
                    "daily_cross_sectional_ranking_problem_v1"
                ),
                "required_relevance_representation": "integer",
                "required_objective": (
                    "rank_xendcg" if model == "lightgbm_rank_xendcg"
                    else "lambdarank"
                ),
            }
            for model in ("lightgbm_rank_xendcg", "lightgbm_lambdarank")
        ],
    ]
    campaign = {
        "campaign_contract": CAMPAIGN_CONTRACT,
        "campaign_version": "v2",
        "campaign_id": RESEARCH_CAMPAIGN_ID,
        "protocol_identity": protocol["protocol_identity"],
        "protocol_logical_checksum": protocol["logical_checksum"],
        "deterministic_ordering": (
            "phase_id,prediction_date,model_id,horizon_id ascending"
        ),
        "phases": phases,
        "fitted_component_matrix": fitted,
        "diagnostic_components": diagnostics,
        "expected_component_count": len(fitted),
        "model_readiness": readiness,
        "component_weighting_owned_by_scheduler": True,
        "training_performed": False,
        "evaluation_performed": False,
    }
    campaign["campaign_identity"] = _hash(
        {
            "campaign_id": campaign["campaign_id"],
            "protocol_identity": campaign["protocol_identity"],
            "matrix": fitted,
        }
    )
    campaign["logical_checksum"] = _hash(campaign)
    return campaign


def historical_stage10_baseline_campaign() -> dict[str, Any]:
    matrix = [
        {
            "job_id": _job_id(model, date, None),
            "model_id": model,
            "prediction_date": date,
            "horizon_id": None,
            "component_role": "FITTED_MODEL",
        }
        for date in DATES
        for model in ("ridge", "elastic_net", "ordered_logit_ranker")
    ]
    result = {
        "campaign_contract": CAMPAIGN_CONTRACT,
        "campaign_version": "v1",
        "campaign_id": BASELINE_CAMPAIGN_ID,
        "historical_identity_preserved": True,
        "fitted_component_matrix": matrix,
        "expected_component_count": len(matrix),
        "deterministic_ordering": "prediction_date then historical model roster",
    }
    result["campaign_identity"] = _hash(result)
    result["logical_checksum"] = _hash(result)
    return result


def validate_selector_campaign(campaign: Mapping[str, Any]) -> None:
    if (
        campaign.get("campaign_contract") != CAMPAIGN_CONTRACT
        or campaign.get("campaign_version") not in {"v1", "v2"}
    ):
        raise ValueError("Invalid selector campaign contract")
    payload = {
        key: value for key, value in campaign.items()
        if key != "logical_checksum"
    }
    if campaign.get("logical_checksum") != _hash(payload):
        raise ValueError("Selector campaign checksum mismatch")
    matrix = list(campaign.get("fitted_component_matrix") or [])
    if campaign.get("expected_component_count") != len(matrix):
        raise ValueError("Selector campaign component count mismatch")
    job_ids = [str(row.get("job_id") or "") for row in matrix]
    owners = [
        (
            str(row.get("model_id") or ""),
            str(row.get("prediction_date") or ""),
            str(row.get("horizon_id") or ""),
        )
        for row in matrix
    ]
    if len(job_ids) != len(set(job_ids)) or len(owners) != len(set(owners)):
        raise ValueError("Duplicate selector campaign component ownership")


def _job_id(model: str, date: str, horizon: str | None) -> str:
    base = f"selector:{date}:{model}"
    return f"{base}:{horizon}" if horizon else base


def _target(horizon: str) -> str:
    sessions = horizon.removeprefix("return_").removesuffix("s")
    return f"forward_return_{sessions}d"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()
