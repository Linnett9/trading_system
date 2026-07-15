from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.research.ml.registries import RegistryBundle, RegistryResolver, canonical_hash, load_registry_bundle
from core.research.ml.registries.io import entry_hash, load_registry
from core.research.ml.registries.types import RegistryValidationError


def _selector_payload() -> dict:
    return json.loads(Path("config/ml_registries/selector_models.v1.json").read_text())


def _write(tmp_path: Path, payload: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_all_registries_validate_and_hash_deterministically():
    first = load_registry_bundle(); second = load_registry_bundle()
    assert first.registry_set_hash == second.registry_set_hash
    assert {kind: len(doc.entries) for kind, doc in first.documents.items()} == {
        "equations": 13, "indicators": 29, "selector_models": 15,
        "exposure": 20, "portfolio_policies": 5,
    }
    assert [entry.canonical_id for entry in first.documents["indicators"].entries] == sorted(
        entry.canonical_id for entry in first.documents["indicators"].entries
    )


def test_duplicate_ids_alias_collisions_invalid_status_and_required_fields_fail(tmp_path: Path):
    cases = []
    duplicate = _selector_payload(); duplicate["entries"].append(copy.deepcopy(duplicate["entries"][0])); cases.append(duplicate)
    collision = _selector_payload(); collision["entries"][1]["aliases"] = [collision["entries"][0]["canonical_id"]]; cases.append(collision)
    invalid = _selector_payload(); invalid["entries"][0]["implementation_status"] = "MAGIC"; cases.append(invalid)
    missing = _selector_payload(); del missing["entries"][0]["implementation_owner"]; cases.append(missing)
    for index, payload in enumerate(cases):
        with pytest.raises(RegistryValidationError):
            load_registry(_write(tmp_path / str(index), payload), expected_kind="selector_models")


def test_tft_alias_requested_and_canonical_are_preserved_and_unknown_fails():
    resolver = RegistryResolver(load_registry_bundle())
    result = resolver.resolve("selector_models", "tft", role="selector")
    assert result.requested_id == "tft"
    assert result.canonical_id == "temporal_fusion_transformer"
    with pytest.raises(KeyError, match="Unknown selector_models registry ID"):
        resolver.resolve("selector_models", "not_a_model")


def test_sequence_capabilities_are_truthful_and_news_remains_blocked():
    resolver = RegistryResolver(load_registry_bundle())
    sequence = resolver.resolve("selector_models", "transformer").entry.payload
    assert sequence["capabilities"]["training"] is True
    assert sequence["checkpoint_support"]["load"] is False
    assert sequence["bounded_runner_support"] is False
    assert sequence["ordinary_runner_support"] is True
    assert resolver.resolve("selector_models", "news_analysis_transformer").entry.payload["implementation_status"] == "BLOCKED_BY_DATA"


def test_strict_hash_rejects_non_serializable_and_alias_order_does_not_change_entry_hash():
    with pytest.raises(TypeError):
        canonical_hash({"path": Path("x")})
    payload = _selector_payload()["entries"][3]
    reversed_aliases = {**payload, "aliases": list(reversed(payload["aliases"]))}
    assert entry_hash(payload) == entry_hash(reversed_aliases)


def test_invalid_feature_and_equation_references_fail(tmp_path: Path):
    resolver = RegistryResolver(load_registry_bundle())
    with pytest.raises(RegistryValidationError, match="Unknown feature schema"):
        resolver.verify_feature_schema("missing/schema.json")
    payload = json.loads(Path("config/ml_registries/indicators.v1.json").read_text())
    payload["entries"][0]["calculation_equation"] = "unknown_equation"
    path = _write(tmp_path, payload)
    document = load_registry(path, expected_kind="indicators")
    bundle = load_registry_bundle()
    broken = RegistryBundle({**bundle.documents, "indicators": document}, bundle.registry_set_hash)
    with pytest.raises(RegistryValidationError, match="Unknown equation"):
        RegistryResolver(broken).validate_references()
