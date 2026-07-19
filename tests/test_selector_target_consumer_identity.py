from __future__ import annotations

import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.stock_level.selector_target_identity import (
    validate_selector_prediction_target_binding,
    validate_selector_target_identity,
)


VALID = {
    "economic_target_id": "forward_return_10d",
    "target_provenance_contract_version": "stock_level_target_provenance_v2",
}


def test_explicit_selector_target_identity_resolves_authoritatively() -> None:
    identity = validate_selector_target_identity(**VALID)
    assert identity.economic_target_id == "forward_return_10d"
    assert identity.target_provenance_contract_version == (
        "stock_level_target_provenance_v2"
    )
    assert identity.economic_target_entry_hash


def test_deprecated_v4_remains_unregistered() -> None:
    with pytest.raises(KeyError):
        RegistryResolver(load_registry_bundle()).resolve(
            "target_contracts", "stock_level_target_provenance_v4"
        )


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"economic_target_id": ""}, "ECONOMIC_TARGET_ID_MISSING"),
        (
            {"target_provenance_contract_version": ""},
            "TARGET_PROVENANCE_IDENTITY_MISSING",
        ),
        (
            {"economic_target_id": "unknown_return"},
            "UNSUPPORTED_TARGET_IDENTITY",
        ),
        (
            {"target_provenance_contract_version": "unknown_provenance"},
            "UNSUPPORTED_TARGET_IDENTITY",
        ),
    ],
)
def test_explicit_selector_target_identity_fails_closed(changed, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_selector_target_identity(**{**VALID, **changed})


@pytest.mark.parametrize(
    "changed",
    [
        {"economic_target_id": "unknown_return"},
        {"target_provenance_contract_version": "unknown_provenance"},
    ],
)
def test_prediction_binding_rejects_explicit_identity_mismatch(changed) -> None:
    with pytest.raises(ValueError):
        validate_selector_prediction_target_binding(
            {**VALID, **changed},
            VALID,
        )


def test_prediction_binding_accepts_matching_explicit_identities() -> None:
    validate_selector_prediction_target_binding(VALID, VALID)
