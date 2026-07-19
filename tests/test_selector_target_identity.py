from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.registries import (
    CURRENT_ECONOMIC_TARGET_ID,
    CURRENT_TARGET_PROVENANCE_VERSION,
    TargetIdentityStatus,
    load_registry_bundle,
    resolve_target_identity,
    validate_target_identity_manifest,
)
from core.research.ml.registries.io import RegistryResolver


def test_current_selector_target_pair_is_supported() -> None:
    result = resolve_target_identity(
        economic_target_id=CURRENT_ECONOMIC_TARGET_ID,
        target_provenance_contract_version=CURRENT_TARGET_PROVENANCE_VERSION,
    )
    assert result.status is TargetIdentityStatus.SUPPORTED
    assert result.supported


@pytest.mark.parametrize(
    ("economic", "provenance", "status"),
    [
        (
            "forward_return_10d",
            "stock_level_target_provenance_v1",
            TargetIdentityStatus.LEGACY_INCOMPATIBLE_PROVENANCE,
        ),
        (
            "forward_return_10d",
            "stock_level_target_provenance_v4",
            TargetIdentityStatus.DEPRECATED_ERRONEOUS_IDENTIFIER,
        ),
        (
            "unknown_return",
            "stock_level_target_provenance_v2",
            TargetIdentityStatus.UNKNOWN_ECONOMIC_TARGET,
        ),
        (
            "forward_return_10d",
            "stock_level_target_provenance_v99",
            TargetIdentityStatus.UNKNOWN_PROVENANCE_VERSION,
        ),
    ],
)
def test_target_identity_statuses(economic, provenance, status) -> None:
    assert resolve_target_identity(
        economic_target_id=economic,
        target_provenance_contract_version=provenance,
    ).status is status


@pytest.mark.parametrize(
    ("economic", "provenance", "namespace"),
    [
        (
            "stock_level_target_provenance_v2",
            "stock_level_target_provenance_v2",
            "economic_target_id",
        ),
        (
            "stock_level_target_provenance_v4",
            "stock_level_target_provenance_v2",
            "economic_target_id",
        ),
        (
            "forward_return_10d",
            "forward_return_10d",
            "target_provenance_contract_version",
        ),
    ],
)
def test_target_identity_namespaces_fail_closed(
    economic, provenance, namespace
) -> None:
    with pytest.raises(ValueError, match=namespace):
        resolve_target_identity(
            economic_target_id=economic,
            target_provenance_contract_version=provenance,
        )


def test_target_registry_separates_aliases_from_provenance() -> None:
    bundle = load_registry_bundle()
    target = next(
        entry
        for entry in bundle.documents["target_contracts"].entries
        if entry.canonical_id == "forward_return_10d"
    )
    assert "stock_level_target_provenance_v4" not in target.aliases
    assert target.payload["target_provenance_contract_versions"] == [
        "stock_level_target_provenance_v2"
    ]
    with pytest.raises(KeyError):
        RegistryResolver(bundle).resolve(
            "target_contracts", "stock_level_target_provenance_v4"
        )


def test_manifest_and_rows_require_matching_explicit_identities() -> None:
    manifest = {
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
    }
    rows = [dict(manifest, row_id="a"), dict(manifest, row_id="b")]
    assert validate_target_identity_manifest(manifest, rows).supported

    for changed in (
        {"economic_target_id": "other"},
        {"target_provenance_contract_version": "other"},
    ):
        with pytest.raises(ValueError, match="manifest and rows"):
            validate_target_identity_manifest({**manifest, **changed}, rows)


@pytest.mark.parametrize(
    "manifest",
    [
        {"target_provenance_contract_version": "stock_level_target_provenance_v2"},
        {"economic_target_id": "forward_return_10d"},
        {
            "target_contract": "forward_return_10d",
            "economic_target_id": "forward_return_10d",
            "target_provenance_contract_version": "stock_level_target_provenance_v2",
        },
    ],
)
def test_manifest_missing_or_ambiguous_identity_fails(manifest) -> None:
    with pytest.raises(ValueError):
        validate_target_identity_manifest(manifest, [])


def test_manifest_rejects_multiple_row_identity_values() -> None:
    manifest = {
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
    }
    with pytest.raises(ValueError, match="exactly one"):
        validate_target_identity_manifest(
            manifest,
            [
                dict(manifest),
                {**manifest, "economic_target_id": "other"},
            ],
        )


def test_registry_json_is_valid_and_v4_is_not_an_alias() -> None:
    path = Path("config/ml_registries/target_contracts.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = next(
        row for row in payload["entries"] if row["canonical_id"] == "forward_return_10d"
    )
    assert "stock_level_target_provenance_v4" not in entry["aliases"]
