from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.selector_dataset_lineage import logical_manifest_checksum
from core.research.ml.selector_publication_gates import (
    GATE_CONTRACT_VERSION, evaluate_selector_parent_publication_gate,
)


DATES = ["2024-03-15", "2024-09-16", "2025-03-17"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", "forward_return_10d", role="selector"
    )
    registry = {
        "status": "READY", "publication_status": "complete",
        "validation_status": "VERIFIED",
        "dataset_type": "canonical_asset_registry_audit",
        "dataset_id": "registry", "symbol_registry_version": "registry-v1",
        "registry_content_checksum": "REGISTRY-CONTENT",
    }
    rp = _write(tmp_path / "registry.json", registry)
    spine = {
        "status": "READY", "publication_status": "complete",
        "validation_status": "READY",
        "dataset_type": "canonical_daily_stock_spine",
        "dataset_id": "spine", "schema_version": "spine-v1",
        "spine_artifact_checksum": "SPINE-ARTIFACT",
        "row_population_checksum": "SPINE-POPULATION",
        "canonical_symbol_registry_identity": "registry",
        "canonical_symbol_registry_version": "registry-v1",
        "canonical_symbol_registry_manifest_checksum": _sha(rp),
    }
    sp = _write(tmp_path / "spine.json", spine)
    feature = {
        "status": "READY", "publication_status": "complete",
        "validation_status": "VERIFIED",
        "dataset_type": "daily_price_features",
        "dataset_id": "features", "schema_version": "features-v1",
        "source_dataset_ids": ["spine"],
        "source_spine_manifest_checksum": _sha(sp),
    }
    fp = _write(tmp_path / "features.json", feature)
    dataset = {
        "status": "READY", "publication_status": "complete",
        "validation_status": "VERIFIED",
        "manifest_schema_version": "authoritative_frozen_selector_dataset_v2",
        "frozen_dataset_version": "v2", "dataset_id": "selector",
        "dataset_checksum": "ROWS", "row_population_checksum": "SELECTOR-POPULATION",
        "symbol_registry_identity": "registry",
        "symbol_registry_version": "registry-v1",
        "symbol_registry_checksum": _sha(rp),
        "daily_stock_spine_identity": "spine",
        "daily_stock_spine_version": "spine-v1",
        "daily_stock_spine_checksum": _sha(sp),
        "daily_feature_store_identity": "features",
        "daily_feature_store_version": "features-v1",
        "daily_feature_store_checksum": _sha(fp),
        "target_contract": target.canonical_id,
        "target_contract_checksum": target.entry.entry_hash,
        "feature_schema_checksum": "FEATURE-SCHEMA",
        "target_schema_checksum": "TARGET-SCHEMA",
        "checksums": {
            "rows.parquet": "ROWS",
            "feature_schema.json": "FEATURE-SCHEMA",
            "target_schema.json": "TARGET-SCHEMA",
        },
        "git_commit": "abc123", "creation_timestamp": "first",
    }
    dataset["logical_checksum"] = logical_manifest_checksum(dataset)
    dp = _write(tmp_path / "dataset.json", dataset)
    dates = {
        "status": "READY", "publication_status": "complete",
        "validation_status": "VERIFIED",
        "selector_dataset_id": "selector",
        "selector_dataset_manifest_checksum": _sha(dp),
        "row_population_checksum": "SELECTOR-POPULATION",
        "available_operational_dates": DATES,
    }
    op = _write(tmp_path / "dates.json", dates)
    return {"registry": rp, "spine": sp, "feature": fp, "dataset": dp, "dates": op}


def _run(paths, root, required=DATES):
    return evaluate_selector_parent_publication_gate(
        registry_manifest=paths["registry"], spine_manifest=paths["spine"],
        feature_manifest=paths["feature"], dataset_manifest=paths["dataset"],
        operational_dates_manifest=paths["dates"],
        required_operational_dates=required, approved_root=root,
    )


def _mutate(paths, key, field, value):
    payload = json.loads(paths[key].read_text())
    payload[field] = value
    if key == "dataset":
        payload["logical_checksum"] = logical_manifest_checksum(payload)
    _write(paths[key], payload)


def test_all_parents_valid_and_identity_stable(tmp_path):
    paths = _fixture(tmp_path)
    first = _run(paths, tmp_path); second = _run(paths, tmp_path)
    assert first == second
    assert first["status"] == "READY"
    assert first["gate_contract_version"] == GATE_CONTRACT_VERSION


def test_timestamps_and_report_paths_do_not_change_gate_identity(tmp_path):
    paths = _fixture(tmp_path); first = _run(paths, tmp_path)
    dataset = json.loads(paths["dataset"].read_text())
    dataset["creation_timestamp"] = "second"
    dataset["report_path"] = "different"
    dataset["logical_checksum"] = logical_manifest_checksum(dataset)
    _write(paths["dataset"], dataset)
    dates = json.loads(paths["dates"].read_text())
    dates["selector_dataset_manifest_checksum"] = _sha(paths["dataset"])
    _write(paths["dates"], dates)
    second = _run(paths, tmp_path)
    assert first["logical_checksum"] == second["logical_checksum"]


@pytest.mark.parametrize(
    "key,field,value,blocker",
    [
        ("registry", "status", "BLOCKED", "PARENT_NOT_READY"),
        ("registry", "registry_content_checksum", None, "REGISTRY_MISMATCH"),
        ("spine", "dataset_id", "changed", "SPINE_MISMATCH"),
        ("spine", "spine_artifact_checksum", None, "SPINE_MISMATCH"),
        ("spine", "row_population_checksum", None, "SPINE_MISMATCH"),
        ("feature", "source_dataset_ids", ["wrong"], "FEATURE_STORE_MISMATCH"),
        ("feature", "source_dataset_ids", ["spine", "arbitrary"], "FEATURE_STORE_MISMATCH"),
        ("dataset", "target_contract", "wrong", "TARGET_CONTRACT_MISMATCH"),
        ("dataset", "target_contract_checksum", "wrong", "TARGET_CONTRACT_MISMATCH"),
        ("dataset", "dataset_id", "changed", "DATASET_MISMATCH"),
        ("dataset", "feature_schema_checksum", "wrong", "CHECKSUM_MISMATCH"),
        ("dataset", "target_schema_checksum", "wrong", "CHECKSUM_MISMATCH"),
        ("dates", "row_population_checksum", "wrong", "POPULATION_MISMATCH"),
    ],
)
def test_substituted_parent_or_contract_fails_closed(tmp_path, key, field, value, blocker):
    paths = _fixture(tmp_path)
    _mutate(paths, key, field, value)
    result = _run(paths, tmp_path)
    assert result["status"] == "BLOCKED"
    assert blocker in result["blockers"]


def test_missing_operational_date(tmp_path):
    paths = _fixture(tmp_path)
    assert "DATE_COVERAGE_INCOMPLETE" in _run(
        paths, tmp_path, required=[*DATES, "2026-03-16"]
    )["blockers"]


def test_missing_and_malformed_manifest(tmp_path):
    paths = _fixture(tmp_path)
    paths["registry"].unlink()
    assert "MISSING_PARENT" in _run(paths, tmp_path)["blockers"]
    paths = _fixture(tmp_path / "second")
    paths["spine"].write_text("{", encoding="utf-8")
    assert "MALFORMED_MANIFEST" in _run(paths, tmp_path / "second")["blockers"]


def test_changed_dataset_logical_checksum(tmp_path):
    paths = _fixture(tmp_path)
    payload = json.loads(paths["dataset"].read_text())
    payload["logical_checksum"] = "wrong"
    _write(paths["dataset"], payload)
    assert "LOGICAL_CHECKSUM_MISMATCH" in _run(paths, tmp_path)["blockers"]


def test_arbitrary_root_is_rejected(tmp_path):
    approved = tmp_path / "approved"; approved.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    paths = _fixture(outside)
    assert "LEGACY_OR_ARBITRARY_ROOT" in _run(paths, approved)["blockers"]


def test_no_parquet_reader_is_imported(monkeypatch, tmp_path):
    paths = _fixture(tmp_path)
    real_import = __import__
    def guarded(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise AssertionError("Parquet reader invoked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    assert _run(paths, tmp_path)["status"] == "READY"
