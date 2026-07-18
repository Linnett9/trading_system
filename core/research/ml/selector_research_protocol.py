from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any, Mapping


PROTOCOL_CONTRACT = "selector_research_protocol.v1"
REQUIRED_IDENTITIES = (
    "selector_model_registry",
    "indicator_registry",
    "equation_registry",
    "ranking_contract",
    "target_contract",
    "symbol_registry",
    "canonical_daily_spine",
    "frozen_selector_dataset",
)


def freeze_selector_research_protocol(
    *,
    campaign_identity: str,
    frozen_identities: Mapping[str, Mapping[str, str]],
    source_commit: str | None = None,
) -> dict[str, Any]:
    identities = {
        name: _identity(name, frozen_identities.get(name))
        for name in REQUIRED_IDENTITIES
    }
    protocol = {
        "protocol_contract": PROTOCOL_CONTRACT,
        "protocol_version": "v1",
        "campaign_identity": campaign_identity,
        "campaign_phase_identities": {
            "phase_a": "baseline_established_linear_controls.v1",
            "phase_b": "robust_contextual_multi_horizon_linear_challengers.v1",
            "phase_c": "grouped_ranking_tree_challengers_deferred.v1",
        },
        "required_model_ids": [
            "ridge", "elastic_net", "ordered_logit_ranker",
            "huber", "contextual_elastic_net",
            "multi_horizon_ridge", "multi_horizon_elastic_net",
            "lightgbm_rank_xendcg", "lightgbm_lambdarank",
        ],
        "required_dates": [
            "2024-03-15", "2024-09-16", "2025-03-17",
            "2025-09-15", "2026-03-16",
        ],
        "required_horizons": [
            "return_1s", "return_5s", "return_10s", "return_20s"
        ],
        "component_roles": {
            "fitted": [
                "ridge", "elastic_net", "ordered_logit_ranker", "huber",
                "contextual_elastic_net", "multi_horizon_ridge",
                "multi_horizon_elastic_net",
                "lightgbm_rank_xendcg", "lightgbm_lambdarank",
            ],
            "diagnostic_non_fitted": [
                "momentum_120d", "risk_adjusted_momentum"
            ],
        },
        "frozen_identities": identities,
        "outcome_maturity_cutoff_contract": (
            "selector_outcome_maturity_cutoff.v1"
        ),
        "component_training_boundary_policy": (
            "component cutoff precedes prediction date; labels must mature by "
            "the component cutoff; multi-horizon components apply the "
            "longest relevant maturity boundary"
        ),
        "purge_sessions": 20,
        "embargo_sessions": 5,
        "development_period": "pre-2024-03-15",
        "validation_periods": [
            "2024-03-15..2025-09-15",
            "rolling five-date component panel",
        ],
        "final_untouched_audit_period": "2026-03-16",
        "search_budgets": {
            "baseline_controls": 0,
            "robust_contextual_challengers": 20,
            "multi_horizon_linear_per_family": 20,
            "ranking_tree_per_objective": 20,
        },
        "seed_policy": {
            "deterministic_models": "no stochastic seed",
            "stochastic_models": [0, 17, 41],
            "publication_seed": 0,
        },
        "required_evaluation_metrics": [
            "rank_ic", "ndcg", "top_bottom_spread", "turnover",
            "net_return", "maximum_drawdown", "coverage",
        ],
        "statistical_safeguards": [
            "matched prediction populations",
            "date-clustered uncertainty",
            "multiple-testing correction",
            "minimum effective date count",
            "untouched final audit period",
        ],
        "transaction_cost_bps_panel": [5, 10, 25, 50],
        "adv_capacity_fraction_panel": [0.01, 0.025, 0.05],
        "promotion_criteria": [
            "strict OOS evidence valid",
            "positive net utility at 25 bps",
            "stable rank evidence across validation dates",
            "capacity evidence passes at 2.5 percent ADV",
        ],
        "rejection_criteria": [
            "temporal or population mismatch",
            "nonpositive net utility at 10 bps",
            "unstable or nonfinite ranking evidence",
        ],
        "defer_criteria": [
            "missing registry or authoritative adapter",
            "missing target, ranking, dependency, or publication contract",
            "final audit period accessed during development",
        ],
        "source_git_commit": source_commit or _git_commit(),
        "training_performed": False,
        "evaluation_performed": False,
    }
    protocol["protocol_identity"] = _hash(
        {
            "contract": PROTOCOL_CONTRACT,
            "campaign_identity": campaign_identity,
            "frozen_identities": identities,
            "source_git_commit": protocol["source_git_commit"],
        }
    )
    protocol["logical_checksum"] = _hash(protocol)
    return protocol


def validate_selector_research_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("protocol_contract") != PROTOCOL_CONTRACT
        or protocol.get("protocol_version") != "v1"
        or not protocol.get("protocol_identity")
    ):
        raise ValueError("Invalid selector research protocol")
    payload = {
        key: value for key, value in protocol.items()
        if key != "logical_checksum"
    }
    if protocol.get("logical_checksum") != _hash(payload):
        raise ValueError("Selector research protocol checksum mismatch")
    identities = protocol.get("frozen_identities") or {}
    if set(identities) != set(REQUIRED_IDENTITIES):
        raise ValueError("Selector research protocol identity roster mismatch")


def _identity(
    name: str, value: Mapping[str, str] | None
) -> dict[str, str]:
    payload = dict(value or {})
    identity = str(payload.get("identity") or "").strip()
    checksum = str(payload.get("checksum") or "").strip()
    if not identity or not checksum:
        raise ValueError(f"Frozen selector identity required: {name}")
    return {"identity": identity, "checksum": checksum}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest().upper()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
