from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.experiment_ledger import read_selector_ledger
from core.research.ml.selector_operational_inputs import (
    DATES,
    MODELS,
    PLAN_CONTRACT,
    TARGET_PROVENANCE_CONTRACT_VERSION,
    _validate_component_plan_rows,
    build_selector_component_plan,
    validate_selector_component_plan,
)
from scripts.build_selector_operational_inputs import main as build_inputs_main


def _target_hash() -> str:
    return RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", "forward_return_10d", role="selector"
    ).entry.entry_hash


def _dataset(tmp_path: Path, **updates):
    value = {
        "dataset_contract_version": "canonical_v2_selector_dataset.v1",
        "dataset_id": "canonical-v2-selector-dataset",
        "dataset_checksum": "DATASET",
        "dataset_manifest_path": str(tmp_path / "dataset" / "manifest.json"),
        "daily_spine_id": "daily-spine-v2",
        "daily_spine_manifest_path": str(tmp_path / "spine" / "manifest.json"),
        "symbol_registry_id": "symbol-registry-v2",
        "symbol_registry_manifest_path": str(tmp_path / "registry" / "manifest.json"),
        "target_contract_id": "forward_return_10d",
        "target_contract_hash": _target_hash(),
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "publication_status": "complete",
        "validation_status": "VERIFIED",
    }
    value.update(updates)
    return value


def _gate(**updates):
    value = {
        "gate_contract_version": "selector_parent_publication_gate.v1",
        "status": "READY",
        "logical_checksum": "PARENT-GATE",
        "selector_run_id": "run-1",
        "selector_dataset_id": "canonical-v2-selector-dataset",
        "selector_dataset_artifact_checksum": "DATASET",
        "daily_spine_id": "daily-spine-v2",
        "canonical_registry_id": "symbol-registry-v2",
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "available_operational_dates": list(DATES),
        "strict_oos_folds": {
            date: {
                "fold_id": f"fold-{date}",
                "training_start": "2020-01-01",
                "training_end": "2024-01-31",
                "validation_date": date,
                "purge_sessions": 10,
                "embargo_sessions": 10,
                "maximum_label_available_timestamp": "2024-01-15",
            }
            for date in DATES
        },
    }
    value.update(updates)
    return value


def _plan(tmp_path: Path, **updates):
    values = {
        "dataset_manifest": _dataset(tmp_path),
        "parent_gate": _gate(),
        "output_root": tmp_path / "plan",
        "campaign_id": "campaign-1",
        "selector_run_id": "run-1",
        "source_commit": "abcdef123456",
        "write": True,
    }
    values.update(updates)
    return build_selector_component_plan(**values)


def _component_manifest(row, *, compatible=True):
    prediction = Path(row["planned_output_root"]) / "predictions.csv"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text("row_id,score\n1,0.5\n", encoding="utf-8")
    payload = {
        "component_schema_version": "authoritative_selector_component_v1",
        "component_id": row["component_id"] if compatible else "different",
        "campaign_id": row["campaign_id"],
        "selector_model_identity": row["model_id"],
        "prediction_date": row["decision_date"],
        "frozen_selector_dataset_identity": {
            "dataset_id": row["dataset_id"],
            "dataset_checksum": "DATASET",
        },
        "feature_contract_version": row["feature_schema_id"],
        "target_contract_version": row["target_contract_id"],
        "target_provenance_contract_version": row["target_provenance_contract_version"],
        "fold_identity": row["fold_id"],
        "git_commit": row["source_commit"],
        "prediction_artifact_path": str(prediction),
        "publication_status": "complete",
        "validation_status": "VERIFIED_STRICT_OOS",
    }
    payload["manifest_checksum"] = canonical_hash(payload)
    (prediction.parent / "manifest.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_exact_fifteen_fitted_components_no_momentum_and_order(tmp_path):
    plan = _plan(tmp_path)
    assert plan["plan_contract_version"] == PLAN_CONTRACT
    assert plan["component_count"] == 15
    assert plan["fitted_models"] == list(MODELS)
    assert "momentum" not in plan["fitted_models"]
    assert [(row["decision_date"], row["model_id"]) for row in plan["components"]] == [
        (date, model) for date in DATES for model in MODELS
    ]


def test_planning_optionally_registers_all_fifteen_before_execution(tmp_path):
    ledger_path = tmp_path / "selector_experiment_ledger.json"
    plan = _plan(tmp_path, experiment_ledger_path=ledger_path)
    ledger = read_selector_ledger(ledger_path)
    assert {row["experiment_id"] for row in ledger["experiments"]} == {
        row["experiment_id"] for row in plan["components"]
    }
    assert {row["status"] for row in ledger["experiments"]} == {"PLANNED"}
    assert ledger["trial_counts"]["planned_material_trials"] == 15


def test_repeated_planning_identical_ids_and_checksum(tmp_path):
    first = _plan(tmp_path)
    second = _plan(tmp_path)
    assert first["logical_checksum"] == second["logical_checksum"]
    assert [row["component_id"] for row in first["components"]] == [
        row["component_id"] for row in second["components"]
    ]
    assert validate_selector_component_plan(tmp_path / "plan" / "component_plan.json")["status"] == "READY"


def test_duplicate_component_ids_and_outputs_rejected(tmp_path):
    rows = copy.deepcopy(_plan(tmp_path, write=False)["components"])
    rows[1]["component_id"] = rows[0]["component_id"]
    with pytest.raises(ValueError, match="duplicate component IDs"):
        _validate_component_plan_rows(rows)
    rows = copy.deepcopy(_plan(tmp_path, write=False)["components"])
    rows[1]["planned_output_root"] = rows[0]["planned_output_root"]
    with pytest.raises(ValueError, match="duplicate output ownership"):
        _validate_component_plan_rows(rows)


@pytest.mark.parametrize(
    "dataset_update,gate_update,match",
    [
        ({}, {"status": "BLOCKED"}, "not READY"),
        ({}, {"selector_run_id": "other-run"}, "another run"),
        ({}, {"target_provenance_contract_version": "stock_level_target_provenance_v1"}, "Target provenance"),
        ({"target_provenance_contract_version": None}, {}, "Dataset lineage missing"),
        ({}, {"available_operational_dates": list(DATES[:-1])}, "dates are absent"),
    ],
)
def test_parent_gate_fail_closed_before_outputs(tmp_path, dataset_update, gate_update, match):
    with pytest.raises(ValueError, match=match):
        _plan(
            tmp_path,
            dataset_manifest=_dataset(tmp_path, **dataset_update),
            parent_gate=_gate(**gate_update),
        )
    assert not (tmp_path / "plan").exists()


def test_missing_model_registry_entry_blocks(tmp_path):
    with pytest.raises(KeyError):
        _plan(tmp_path, model_ids=("ridge", "elastic_net", "missing_model"))
    assert not (tmp_path / "plan").exists()


def test_identity_changes_for_material_inputs(tmp_path, monkeypatch):
    base = _plan(tmp_path / "base", write=False)["components"][0]["component_id"]
    feature = _plan(tmp_path / "feature", write=False, random_seed=43)["components"][0]["component_id"]
    fold_gate = _gate()
    fold_gate["strict_oos_folds"][DATES[0]]["fold_id"] = "changed-fold"
    fold = _plan(tmp_path / "fold", write=False, parent_gate=fold_gate)["components"][0]["component_id"]
    commit = _plan(tmp_path / "commit", write=False, source_commit="different")["components"][0]["component_id"]
    monkeypatch.setattr("core.research.ml.selector_operational_inputs._sha", lambda path: "CHANGED-FEATURE")
    changed_feature = _plan(tmp_path / "schema", write=False)["components"][0]["component_id"]
    assert len({base, feature, fold, commit, changed_feature}) == 5


def test_target_contract_change_blocks_before_output(tmp_path):
    with pytest.raises(ValueError, match="Target contract"):
        _plan(tmp_path, dataset_manifest=_dataset(tmp_path, target_contract_hash="wrong"))
    assert not (tmp_path / "plan").exists()


def test_complete_incomplete_and_incompatible_resume_states(tmp_path):
    first = _plan(tmp_path, write=False)
    row = first["components"][0]
    _component_manifest(row)
    resumed = _plan(tmp_path, write=False)
    assert resumed["components"][0]["status"] == "COMPLETE_COMPATIBLE"
    incomplete_row = resumed["components"][1]
    Path(incomplete_row["planned_output_root"]).mkdir(parents=True)
    incomplete = _plan(tmp_path, write=False)
    assert incomplete["components"][1]["status"] == "PLANNED"
    bad_row = resumed["components"][2]
    _component_manifest(bad_row, compatible=False)
    incompatible = _plan(tmp_path, write=False)
    assert incompatible["components"][2]["status"] == "PLANNED"
    assert incompatible["components"][2]["planned_output_root"] != bad_row["planned_output_root"]


def test_failed_planning_preserves_previous_valid_plan(tmp_path):
    first = _plan(tmp_path)
    before = (tmp_path / "plan" / "component_plan.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        _plan(tmp_path, parent_gate=_gate(status="BLOCKED"))
    assert (tmp_path / "plan" / "component_plan.json").read_text(encoding="utf-8") == before
    assert json.loads(before)["logical_checksum"] == first["logical_checksum"]


def test_planning_does_not_execute_factories_or_import_forbidden_owners(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("model factory executed")
    monkeypatch.setattr("core.research.ml.stock_level_benchmark_models._build_tabular_model", forbidden, raising=False)
    _plan(tmp_path)
    source = Path("core/research/ml/selector_operational_inputs.py").read_text(encoding="utf-8")
    for forbidden_owner in ("portfolio_replay", "policy_sweep", "exposure", "news", "finalize_alpaca", "pyarrow"):
        assert forbidden_owner not in source


def test_component_plan_cli_build_and_verify(tmp_path, monkeypatch, capsys):
    dataset_path = tmp_path / "dataset_manifest.json"
    gate_path = tmp_path / "gate.json"
    dataset_path.write_text(json.dumps(_dataset(tmp_path)), encoding="utf-8")
    gate_path.write_text(json.dumps(_gate()), encoding="utf-8")
    argv = [
        "build_selector_operational_inputs.py",
        "--component-plan-only",
        "--selector-dataset", str(dataset_path),
        "--parent-gate", str(gate_path),
        "--output-root", str(tmp_path / "plan"),
        "--evaluation-cutoff", "unused",
        "--selector-run-id", "run-1",
        "--campaign-id", "campaign-1",
        "--source-commit", "abcdef123456",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert build_inputs_main() == 0
    assert "selector-component" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", argv + ["--verify-only"])
    assert build_inputs_main() == 0
