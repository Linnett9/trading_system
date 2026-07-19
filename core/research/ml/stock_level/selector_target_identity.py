from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.research.ml.registries import (
    RegistryResolver,
    load_registry_bundle,
    resolve_target_identity,
)


@dataclass(frozen=True)
class SelectorTargetIdentity:
    economic_target_id: str
    target_provenance_contract_version: str
    economic_target_entry_hash: str

    def as_dict(self) -> dict[str, str]:
        return {
            "economic_target_id": self.economic_target_id,
            "target_provenance_contract_version": (
                self.target_provenance_contract_version
            ),
        }


def validate_selector_target_identity(
    *,
    economic_target_id: Any,
    target_provenance_contract_version: Any,
    expected: Mapping[str, Any] | None = None,
) -> SelectorTargetIdentity:
    economic = str(economic_target_id or "").strip()
    provenance = str(target_provenance_contract_version or "").strip()
    if not economic:
        raise ValueError("ECONOMIC_TARGET_ID_MISSING")
    if not provenance:
        raise ValueError("TARGET_PROVENANCE_IDENTITY_MISSING")
    resolution = resolve_target_identity(
        economic_target_id=economic,
        target_provenance_contract_version=provenance,
    )
    if not resolution.supported:
        raise ValueError(f"UNSUPPORTED_TARGET_IDENTITY:{resolution.status.value}")
    if expected is not None:
        expected_economic = str(expected.get("economic_target_id") or "").strip()
        expected_provenance = str(
            expected.get("target_provenance_contract_version") or ""
        ).strip()
        if not expected_economic or not expected_provenance:
            raise ValueError("MODEL_EXPLICIT_TARGET_IDENTITY_MISSING")
        if economic != expected_economic:
            raise ValueError("ECONOMIC_TARGET_IDENTITY_MISMATCH")
        if provenance != expected_provenance:
            raise ValueError("TARGET_PROVENANCE_IDENTITY_MISMATCH")
    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", economic, role="selector"
    )
    return SelectorTargetIdentity(economic, provenance, target.entry.entry_hash)


def validate_selector_prediction_target_binding(
    binding: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
) -> None:
    pair = validate_selector_target_identity(
        economic_target_id=binding.get("economic_target_id"),
        target_provenance_contract_version=binding.get(
            "target_provenance_contract_version"
        ),
        expected=model_metadata,
    )
    if pair.economic_target_id != model_metadata.get("economic_target_id"):
        raise ValueError("PREDICTION_ECONOMIC_TARGET_MISMATCH")
    if (
        pair.target_provenance_contract_version
        != model_metadata.get("target_provenance_contract_version")
    ):
        raise ValueError("PREDICTION_TARGET_PROVENANCE_MISMATCH")
