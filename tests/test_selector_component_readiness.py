from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.selector_component_readiness import (
    COMPONENT_SCHEMA, DATES, MODELS, READINESS_CONTRACT,
    VERIFIED_STRICT_OOS, assess_selector_component_readiness,
)


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(tmp_path: Path, *, status="READY") -> Path:
    return _write(tmp_path / "gate.json", {
        "gate_contract_version": "selector_parent_publication_gate.v1",
        "status": status, "logical_checksum": "PARENT-GATE",
        "selector_dataset_id": "dataset",
        "selector_dataset_artifact_checksum": "DATASET-CHECKSUM",
        "canonical_registry_id": "registry",
        "daily_spine_id": "spine",
        "selector_feature_schema_checksum": "FEATURE-HASH",
    })


def _component(root: Path, model: str, date: str, **overrides) -> Path:
    resolver = RegistryResolver(load_registry_bundle())
    resolution = resolver.resolve("selector_models", model, role="selector")
    target = resolver.resolve("target_contracts", "forward_return_10d", role="selector")
    artifact = root / f"model={model}" / f"date={date}" / "predictions.bin"
    artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_bytes(b"predictions")
    import hashlib
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest().upper()
    feature = (
        "canonical_v2_daily_tree_cross_sectional_features_v1"
        if model == "ordered_logit_ranker"
        else "canonical_v2_daily_tabular_features_v1"
    )
    payload = {
        "component_schema_version": COMPONENT_SCHEMA,
        "selector_model_identity": model,
        "selector_model_version": resolution.entry.entry_hash,
        "prediction_date": date,
        "training_start": "2020-01-01",
        "training_cutoff": "2024-01-01",
        "training_label_available_timestamp_max": "2024-01-01",
        "fold_identity": "fold",
        "frozen_selector_dataset_identity": {
            "dataset_id": "dataset", "dataset_checksum": "DATASET-CHECKSUM",
        },
        "symbol_registry_identity": "registry",
        "daily_stock_spine_identity": "spine",
        "feature_contract_version": feature,
        "economic_target_id": resolution.entry.payload["economic_target_id"],
        "target_provenance_contract_version": resolution.entry.payload[
            "target_provenance_contract_version"
        ],
        "legacy_target_contract": resolution.entry.payload.get("target_contract"),
        "ranking_contract_version": resolution.entry.payload.get(
            "ranking_problem_contract"
        ) or "ranking_metric_contract_v1",
        "prediction_row_count": 2,
        "prediction_population_checksum": f"population-{date}",
        "prediction_artifact_path": str(artifact),
        "prediction_checksum": checksum,
        "artifact_link": {
            "artifact_checksum": checksum,
            "feature_schema_hash": "FEATURE-HASH",
            "target_contract_hash": target.entry.entry_hash,
            "verification_status": VERIFIED_STRICT_OOS,
        },
        "publication_status": "complete",
        "validation_status": VERIFIED_STRICT_OOS,
        "git_commit": "abc123",
        "non_production_smoke": False,
    }
    payload.update(overrides)
    return _write(artifact.parent / "manifest.json", payload)


def _run(tmp_path, root, gate=None, approved=None):
    return assess_selector_component_readiness(
        parent_gate_path=gate or _gate(tmp_path),
        authoritative_root=root,
        selector_dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "config.yaml",
        approved_component_roots=tuple(approved or [root]),
    )


def _all(root):
    for date in DATES:
        for model in MODELS:
            _component(root, model, date)


def test_empty_root_produces_fifteen_missing_and_deterministic_plan(tmp_path):
    root = tmp_path / "components"
    first = _run(tmp_path, root); second = _run(tmp_path, root)
    assert first["readiness_contract_version"] == READINESS_CONTRACT
    assert first["expected_component_count"] == first["missing_component_count"] == 15
    assert len(first["production_plan"]) == 15
    assert first["logical_checksum"] == second["logical_checksum"]
    assert [job["job_id"] for job in first["production_plan"]] == [
        f"selector:{date}:{model}" for date in DATES for model in MODELS
    ]


def test_one_valid_component(tmp_path):
    root = tmp_path / "components"; _component(root, "ridge", DATES[0])
    result = _run(tmp_path, root)
    assert result["ready_component_count"] == 1
    assert result["missing_component_count"] == 14


def test_all_fifteen_valid_and_matched(tmp_path):
    root = tmp_path / "components"; _all(root)
    result = _run(tmp_path, root)
    assert result["overall_status"] == "READY"
    assert result["ready_component_count"] == 15
    assert not result["production_plan"]
    assert all(row["status"] == "READY" for row in result["matched_population_results"])


def test_parent_gate_blocked(tmp_path):
    result = _run(tmp_path, tmp_path / "components", _gate(tmp_path, status="BLOCKED"))
    assert result["overall_status"] == "BLOCKED"
    assert all(row["state"] == "BLOCKED_PARENT_GATE" for row in result["component_matrix"])


def test_arbitrary_component_root_is_rejected(tmp_path):
    arbitrary = tmp_path / "arbitrary"
    result = _run(
        tmp_path, arbitrary,
        approved=[tmp_path / "approved-components"],
    )
    assert result["overall_status"] == "BLOCKED"
    assert all(
        row["state"] == "NON_AUTHORITATIVE_ROOT"
        for row in result["component_matrix"]
    )


@pytest.mark.parametrize(
    "overrides,state",
    [
        ({"selector_model_identity": "wrong"}, "MODEL_IDENTITY_MISMATCH"),
        ({"selector_model_version": "wrong"}, "MODEL_IDENTITY_MISMATCH"),
        ({"prediction_date": "wrong"}, "INCOMPLETE"),
        ({"frozen_selector_dataset_identity": {"dataset_id": "wrong", "dataset_checksum": "DATASET-CHECKSUM"}}, "DATASET_IDENTITY_MISMATCH"),
        ({"feature_contract_version": "wrong"}, "FEATURE_CONTRACT_MISMATCH"),
        ({"economic_target_id": "wrong"}, "TARGET_CONTRACT_MISMATCH"),
        ({"ranking_contract_version": "wrong"}, "RANKING_CONTRACT_MISMATCH"),
        ({"fold_identity": None}, "FOLD_IDENTITY_MISSING"),
        ({"training_cutoff": "2027-01-01"}, "TEMPORAL_LEAKAGE"),
        ({"training_label_available_timestamp_max": "2027-01-01"}, "TEMPORAL_LEAKAGE"),
        ({"prediction_row_count": 0}, "PREDICTION_INCOMPLETE"),
        ({"artifact_link": {}}, "ARTIFACT_CHECKSUM_MISMATCH"),
        ({"validation_status": "PENDING"}, "ARTIFACT_LINK_UNVERIFIED"),
        ({"non_production_smoke": True}, "SMOKE_OUTPUT_REJECTED"),
    ],
)
def test_invalid_component_states(tmp_path, overrides, state):
    root = tmp_path / "components"
    model = "ordered_logit_ranker" if state == "RANKING_CONTRACT_MISMATCH" else "ridge"
    _component(root, model, DATES[0], **overrides)
    result = _run(tmp_path, root)
    row = next(
        row for row in result["component_matrix"]
        if row["model_id"] == model and row["prediction_date"] == DATES[0]
    )
    assert state in row["reasons"]


def test_malformed_component_manifest(tmp_path):
    root = tmp_path / "components"
    path = root / "model=ridge" / f"date={DATES[0]}" / "manifest.json"
    path.parent.mkdir(parents=True); path.write_text("{")
    row = _run(tmp_path, root)["component_matrix"][0]
    assert row["state"] == "MALFORMED"


def test_wrong_artifact_checksum(tmp_path):
    root = tmp_path / "components"; manifest = _component(root, "ridge", DATES[0])
    payload = json.loads(manifest.read_text()); payload["prediction_checksum"] = "wrong"
    manifest.write_text(json.dumps(payload))
    row = _run(tmp_path, root)["component_matrix"][0]
    assert "ARTIFACT_CHECKSUM_MISMATCH" in row["reasons"]


def test_population_mismatch_blocks_date_panel(tmp_path):
    root = tmp_path / "components"
    for model in MODELS:
        _component(root, model, DATES[0])
    manifest = root / "model=elastic_net" / f"date={DATES[0]}" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["prediction_population_checksum"] = "different"
    manifest.write_text(json.dumps(payload))
    result = _run(tmp_path, root)
    panel = result["matched_population_results"][0]
    assert panel["status"] == "BLOCKED"
    assert all(
        row["state"] == "READY"
        for row in result["component_matrix"] if row["prediction_date"] == DATES[0]
    )
    assert "POPULATION_MISMATCH" in result["blockers"]


def test_plan_contains_ordered_logit_contracts(tmp_path):
    job = next(
        job for job in _run(tmp_path, tmp_path / "components")["production_plan"]
        if job["model_id"] == "ordered_logit_ranker"
    )
    assert job["feature_schema"].endswith("canonical_v2_daily_tree_cross_sectional_v1.json")
    assert job["ranking_contract"] == "daily_cross_sectional_ranking_problem_v1"
    assert job["relevance_contract"] == "within_date_quintile_relevance_v1"
    assert job["expected_parent_gate_checksum"] == "PARENT-GATE"


def test_no_parquet_or_model_fitting_import(monkeypatch, tmp_path):
    real_import = __import__
    def guarded(name, *args, **kwargs):
        if name.startswith("pyarrow") or "bounded_selector_runner" in name:
            raise AssertionError("forbidden readiness import")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    result = _run(tmp_path, tmp_path / "components")
    assert result["fitting_performed"] is False
    assert result["commands_executed"] is False
