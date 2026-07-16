from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.types import RegistryValidationError


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


def selector_model_adapter(requested_model_id: str, *, runner: str, allow_blocked: bool = False) -> SelectorModelAdapter:
    resolution = RegistryResolver(load_registry_bundle()).resolve("selector_models", requested_model_id, role="selector")
    payload = resolution.entry.payload
    supported = payload["bounded_runner_support"] if runner == "bounded" else payload["ordinary_runner_support"] if runner == "ordinary" else False
    if not supported:
        raise RegistryValidationError(f"Selector model {requested_model_id} does not support runner {runner}")
    if payload["implementation_status"] == "BLOCKED_BY_DATA" and not allow_blocked:
        raise RegistryValidationError(f"Selector model {requested_model_id} is blocked by data requirements")
    if runner == "bounded":
        constructor = "core.research.ml.stock_level.bounded_selector_runner:_bounded_model"
    elif resolution.canonical_id in {
        "huber", "contextual_elastic_net", "multi_horizon_ridge",
        "multi_horizon_elastic_net", "multi_horizon_ordered_logit",
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
    }

    return _verify_selector_capabilities(
        load_registry_bundle(), set(SUPPORTED_MODELS), set(MODEL_NAMES) | wave4
    )
