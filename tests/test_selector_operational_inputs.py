from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_operational_inputs import (
    DATES, INVENTORY_CONTRACT, MODELS, build_operational_inputs, validate_inventory,
)


def _parents():
    dataset = {
        "dataset_id": "dataset-v2", "dataset_checksum": "DATASET",
        "publication_status": "complete", "validation_status": "VERIFIED",
    }
    gate = {
        "gate_contract_version": "selector_parent_publication_gate.v1",
        "status": "READY", "logical_checksum": "GATE",
        "selector_dataset_id": "dataset-v2",
        "selector_dataset_artifact_checksum": "DATASET",
    }
    return dataset, gate


def _plan(model_override=None):
    resolver = RegistryResolver(load_registry_bundle())
    jobs = []
    for prediction_date in DATES:
        for original in MODELS:
            model = model_override if model_override and original == "ridge" else original
            payload = resolver.resolve("selector_models", model, role="selector").entry.payload
            job = {
                "job_id": f"selector:{prediction_date}:{model}", "model_id": model,
                "prediction_date": prediction_date, "feature_schema": payload["feature_schema"],
                "target_contract": payload["target_contract"],
                "ranking_contract": payload.get("ranking_problem_contract"),
                "relevance_contract": payload.get("relevance_contract"),
            }
            job["logical_checksum"] = canonical_hash(job)
            jobs.append(job)
    value = {"production_plan": jobs}
    value["logical_checksum"] = canonical_hash(value)
    return value


def _rows():
    resolver = RegistryResolver(load_registry_bundle())
    features = set()
    for model in MODELS:
        schema = json.loads(Path(resolver.resolve("selector_models", model).entry.payload["feature_schema"]).read_text())
        features.update(row["name"] for row in schema["features"])
    history = [(date(2024, 1, 1) + timedelta(days=index)).isoformat() for index in range(70)]
    all_dates = sorted(set(history) | set(DATES))
    rows = []
    for day_index, day in enumerate(all_dates):
        for asset in range(5):
            row = {
                "row_id": f"{day}-{asset}", "asset_id": f"A{asset}", "canonical_symbol": f"S{asset}",
                "decision_session_date": day, "decision_timestamp": day,
                "label_available_timestamp": day,
                "actual_forward_return_10d": float(asset + day_index / 100),
                "actual_benchmark_return_10d": float(asset) / 100,
            }
            row.update({name: float(asset + offset / 1000) for offset, name in enumerate(sorted(features))})
            rows.append(row)
    return rows


def _build(tmp_path, **updates):
    dataset, gate = _parents()
    values = dict(plan=_plan(), dataset_manifest=dataset, parent_gate=gate, rows=_rows(),
                  output_root=tmp_path / "published", evaluation_cutoff="2027-01-01",
                  source_git_commit="abc123")
    values.update(updates)
    return build_operational_inputs(**values)


def test_exact_fifteen_inventory_and_no_challengers(tmp_path):
    result = _build(tmp_path)
    assert result["inventory_contract_version"] == INVENTORY_CONTRACT
    assert len(result["packages"]) == 15
    assert {row["model_id"] for row in result["packages"]} == set(MODELS)


@pytest.mark.parametrize("model", MODELS)
def test_model_packages(tmp_path, model):
    result = _build(tmp_path)
    row = next(item for item in result["packages"] if item["model_id"] == model)
    manifest = json.loads(Path(row["package_manifest_path"]).read_text())
    assert manifest["model_id"] == model
    assert manifest["training_row_count"] > 0 and manifest["prediction_row_count"] == 5


def test_prediction_population_identical_across_models(tmp_path):
    result = _build(tmp_path)
    for day in DATES:
        assert len({row["prediction_ordered_population_checksum"] for row in result["packages"] if row["prediction_date"] == day}) == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(label_available_timestamp="2030-01-01"), "Immature"),
    (lambda rows: rows.append(dict(rows[0])), "Duplicate source"),
    (lambda rows: rows[0].update(**{next(key for key in rows[0] if key.startswith("predicted_")): float("nan")}), "Nonfinite"),
])
def test_invalid_source_rows(tmp_path, mutation, match):
    rows = _rows(); mutation(rows)
    with pytest.raises(ValueError, match=match):
        _build(tmp_path, rows=rows)


def test_prediction_date_removed_from_training(tmp_path):
    result = _build(tmp_path)
    package = result["packages"][0]
    training = json.loads(Path(package["training_rows_path"]).read_text())
    assert all(row["decision_session_date"] != package["prediction_date"] for row in training)


def test_purge_and_embargo_recorded_and_enforced(tmp_path):
    result = _build(tmp_path)
    manifest = json.loads(Path(result["packages"][0]["package_manifest_path"]).read_text())
    assert manifest["purge_sessions"] == manifest["embargo_sessions"] == 10
    training = json.loads(Path(manifest["training_rows_path"]).read_text())
    assert max(row["decision_session_date"] for row in training) < manifest["training_cutoff"]


def test_prediction_targets_are_removed(tmp_path):
    result = _build(tmp_path)
    rows = json.loads(Path(result["packages"][0]["prediction_rows_path"]).read_text())
    assert all("actual_forward_return_10d" not in row for row in rows)


def test_ordered_logit_groups_and_all_classes(tmp_path):
    result = _build(tmp_path)
    row = next(item for item in result["packages"] if item["model_id"] == "ordered_logit_ranker")
    training = json.loads(Path(row["training_rows_path"]).read_text())
    assert all("date_group" in item and "relevance_label" in item for item in training)
    assert {item["relevance_label"] for item in training} == set(range(5))


def test_challenger_plan_rejected(tmp_path):
    with pytest.raises(ValueError, match="15 base jobs"):
        _build(tmp_path, plan=_plan("huber"))


def test_dataset_identity_mismatch_rejected(tmp_path):
    dataset, gate = _parents(); gate["selector_dataset_artifact_checksum"] = "WRONG"
    with pytest.raises(ValueError, match="Dataset identity"):
        _build(tmp_path, dataset_manifest=dataset, parent_gate=gate)


def test_plan_checksum_mismatch_rejected(tmp_path):
    plan = _plan(); plan["production_plan"][0]["logical_checksum"] = "WRONG"
    with pytest.raises(ValueError, match="checksum"):
        _build(tmp_path, plan=plan)


def test_stable_checksums_and_compatible_skip(tmp_path):
    first = _build(tmp_path); second = _build(tmp_path)
    assert first["logical_checksum"] == second["logical_checksum"]


def test_incompatible_immutable_package_fails_closed(tmp_path):
    _build(tmp_path)
    manifest = next((tmp_path / "published/component_inputs").glob("*/manifest.json"))
    payload = json.loads(manifest.read_text()); payload["model_id"] = "wrong"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(FileExistsError, match="Incompatible immutable"):
        _build(tmp_path)


def test_outcomes_cover_exact_five_dates_and_are_mature(tmp_path):
    result = _build(tmp_path)
    manifest = json.loads(Path(result["mature_outcome_manifest_path"]).read_text())
    assert manifest["required_dates"] == list(DATES)
    assert manifest["validation_status"] == "VERIFIED_MATURE"
    assert manifest["evaluation_cutoff"] == "2027-01-01"


def test_immature_outcome_rejected(tmp_path):
    with pytest.raises(ValueError, match="Immature outcome"):
        _build(tmp_path, evaluation_cutoff="2024-01-01")


def test_inventory_resolves_every_job(tmp_path):
    result = _build(tmp_path)
    validated = validate_inventory(Path(result["inventory_path"]))
    assert validated["status"] == "READY"


def test_inventory_rejects_missing_package(tmp_path):
    result = _build(tmp_path)
    inventory = Path(result["inventory_path"]); payload = json.loads(inventory.read_text())
    payload["packages"].pop(); payload["logical_checksum"] = canonical_hash({k: v for k, v in payload.items() if k != "logical_checksum"})
    inventory.write_text(json.dumps(payload))
    assert "INVENTORY_JOB_COVERAGE_MISMATCH" in validate_inventory(inventory)["reasons"]


def test_inventory_rejects_arbitrary_json(tmp_path):
    path = tmp_path / "arbitrary.json"; path.write_text("{}")
    assert validate_inventory(path)["status"] == "BLOCKED"


def test_builders_do_not_import_execution_workflows():
    text = Path("core/research/ml/selector_operational_inputs.py").read_text()
    for forbidden in ("portfolio_replay", "policy_sweep", "exposure", "finalize_alpaca", "evaluate_selector_components"):
        assert forbidden not in text
