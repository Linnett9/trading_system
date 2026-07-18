from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.types import RegistryValidationError


LIGHTGBM_PRODUCTION_OWNER = (
    "core.research.ml.stock_level.lightgbm_production_selector:"
    "fit_production_lightgbm_selector"
)
LIGHTGBM_PUBLICATION_OWNER = (
    "core.research.ml.stock_level.wave4_selector_integration:"
    "publish_wave4_component"
)
LIGHTGBM_MODELS = {
    "lightgbm_rank_xendcg": {
        "objective": "rank_xendcg",
        "configuration_owner": (
            "core.research.ml.stock_level.lightgbm_rank_xendcg_selector:"
            "fixed_rank_xendcg_configuration"
        ),
    },
    "lightgbm_lambdarank": {
        "objective": "lambdarank",
        "configuration_owner": (
            "core.research.ml.stock_level.lightgbm_lambdarank_selector:"
            "fixed_lambdarank_configuration"
        ),
    },
}


@dataclass(frozen=True)
class SelectorModelAdapter:
    requested_model_id: str
    canonical_model_id: str
    model_family: str
    implementation_owner: str
    bounded_runner_support: bool
    ordinary_runner_support: bool
    feature_schema: str | None
    target_contract: str | None
    worker_support: Any
    seed_support: Any
    checkpoint_support: Any
    dependency_requirements: tuple[str, ...]
    constructor_owner: str
    entry_hash: str
    ranking_problem_contract: str | None
    relevance_contract: str | None
    objective_identity: str | None
    grouped_query_contract: str | None
    fitting_configuration_checksum: str | None
    dependency_preflight_identity: str | None


def selector_model_adapter(requested_model_id: str, *, runner: str, allow_blocked: bool = False) -> SelectorModelAdapter:
    resolution = RegistryResolver(load_registry_bundle()).resolve("selector_models", requested_model_id, role="selector")
    payload = resolution.entry.payload
    supported = payload["bounded_runner_support"] if runner == "bounded" else payload["ordinary_runner_support"] if runner == "ordinary" else False
    if not supported:
        raise RegistryValidationError(f"Selector model {requested_model_id} does not support runner {runner}")
    if payload["implementation_status"] == "BLOCKED_BY_DATA" and not allow_blocked:
        raise RegistryValidationError(f"Selector model {requested_model_id} is blocked by data requirements")
    if resolution.canonical_id in LIGHTGBM_MODELS:
        _validate_lightgbm_production_owner(resolution.canonical_id, payload)
    if runner == "bounded":
        constructor = "core.research.ml.stock_level.bounded_selector_runner:_bounded_model"
    elif resolution.canonical_id in {
        "huber", "contextual_elastic_net", "multi_horizon_ridge",
        "multi_horizon_elastic_net", "multi_horizon_ordered_logit",
        *LIGHTGBM_MODELS,
    }:
        constructor = "core.research.ml.stock_level.wave4_selector_integration:publish_wave4_component"
    else:
        constructor = "core.research.ml.stock_level_benchmark_models:TabularModelFactory/SequenceModelFactory"
    return SelectorModelAdapter(
        requested_model_id, resolution.canonical_id, str(payload["category"]),
        str(payload["implementation_owner"]), bool(payload["bounded_runner_support"]),
        bool(payload["ordinary_runner_support"]), payload.get("feature_schema"),
        payload.get("target_contract"), payload.get("worker_support"), payload.get("seed_support"),
        payload.get("checkpoint_support"), tuple(payload.get("dependency_requirements", ())), constructor,
        resolution.entry.entry_hash, payload.get("ranking_problem_contract"), payload.get("relevance_contract"),
        payload.get("objective_identity"), payload.get("grouped_query_contract"),
        payload.get("fitting_configuration_checksum"),
        payload.get("dependency_preflight_identity"),
    )


def _verify_selector_capabilities(bundle, bounded_models, ordinary_models) -> dict[str, int]:
    resolver = RegistryResolver(bundle)
    bounded = ordinary = blocked = 0
    for entry in bundle.documents["selector_models"].entries:
        payload = entry.payload
        if payload["bounded_runner_support"]:
            bounded += 1
            if entry.canonical_id not in bounded_models and payload["category"] != "BASELINE":
                raise RegistryValidationError(
                    f"Registry claims bounded support but the bounded runner has no constructor for {entry.canonical_id}"
                )
        if payload["ordinary_runner_support"]:
            ordinary += 1
            if entry.canonical_id not in ordinary_models and payload["category"] != "BASELINE":
                raise RegistryValidationError(
                    f"Registry claims ordinary support but the ordinary runner has no constructor for {entry.canonical_id}"
                )
        if payload["implementation_status"] == "BLOCKED_BY_DATA":
            blocked += 1
        resolver.resolve("selector_models", entry.canonical_id, role="selector")
    return {"bounded": bounded, "ordinary": ordinary, "blocked": blocked}


def verify_registry_capabilities() -> dict[str, int]:
    # Runner imports stay deferred so registry consumers remain lightweight.
    from core.research.ml.stock_level.bounded_selector_runner import SUPPORTED_MODELS
    from core.research.ml.stock_level_benchmark_types import MODEL_NAMES
    wave4 = {
        "huber", "contextual_elastic_net", "multi_horizon_ridge",
        "multi_horizon_elastic_net", "multi_horizon_ordered_logit",
        *LIGHTGBM_MODELS,
    }

    return _verify_selector_capabilities(
        load_registry_bundle(), set(SUPPORTED_MODELS), set(MODEL_NAMES) | wave4
    )


def _validate_lightgbm_production_owner(
    model_id: str, payload: Any
) -> None:
    expected = LIGHTGBM_MODELS[model_id]
    required = {
        "implementation_owner": LIGHTGBM_PRODUCTION_OWNER,
        "strict_oos_publication_adapter": LIGHTGBM_PUBLICATION_OWNER,
        "objective": expected["objective"],
        "objective_configuration_owner": expected["configuration_owner"],
        "dependency_preflight_owner": (
            "core.research.ml.stock_level.wave4_selector_integration:"
            "assess_lightgbm_ranking_dependency"
        ),
        "production_owner": True,
        "synthetic_only": False,
        "strict_oos_capable": True,
        "campaign_execution_eligible": True,
        "promotion_evidence": False,
        "promoted": False,
    }
    mismatches = [
        key for key, value in required.items() if payload.get(key) != value
    ]
    synthetic = str(payload.get("synthetic_fixture_owner") or "")
    if not synthetic.endswith(
        (
            "fit_synthetic_rank_xendcg_selector"
            if model_id == "lightgbm_rank_xendcg"
            else "fit_synthetic_lambdarank_selector"
        )
    ):
        mismatches.append("synthetic_fixture_owner")
    if mismatches:
        raise RegistryValidationError(
            f"LightGBM production provenance invalid for {model_id}: "
            + ",".join(sorted(mismatches))
        )
    module_name, callable_name = LIGHTGBM_PRODUCTION_OWNER.split(":", 1)
    if not callable(getattr(import_module(module_name), callable_name, None)):
        raise RegistryValidationError(
            f"LightGBM production owner cannot resolve for {model_id}"
        )
