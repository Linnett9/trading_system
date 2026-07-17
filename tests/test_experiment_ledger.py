from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest

from core.research.ml.experiment_ledger import (
    append_ledger_event, experiment_spec_hash, latest_run_states,
    new_experiment_run_id, read_ledger, read_selector_ledger,
    register_selector_plan, selector_experiment_definition,
    transition_selector_experiment,
)


def _append(path: Path, run_id: str, status: str):
    return append_ledger_event(
        path, experiment_spec_hash_value="A" * 64, experiment_run_id=run_id,
        event_status=status, artifact_kind="MODEL_EXPERIMENT",
        canonical_model_id="ridge", requested_model_id="ridge",
        registry_hashes={"entry":"B" * 64}, source_commit="commit",
        error_summary="kept" if status == "FAILED" else None,
        rejection_summary="kept" if status == "REJECTED" else None,
    )


def test_append_history_latest_state_and_distinct_attempt_identity(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"; spec = experiment_spec_hash({"model":"ridge"})
    first, second = new_experiment_run_id(spec), new_experiment_run_id(spec)
    assert first != second
    _append(path, first, "STARTED"); _append(path, first, "FAILED")
    _append(path, second, "STARTED"); _append(path, second, "REJECTED")
    _append(path, "skip", "SKIPPED_COMPLETE")
    events = read_ledger(path)
    assert [event["event_status"] for event in events] == ["STARTED", "FAILED", "STARTED", "REJECTED", "SKIPPED_COMPLETE"]
    states = latest_run_states(events)
    assert states[first]["event_status"] == "FAILED"
    assert states[second]["rejection_summary"] == "kept"


def test_concurrent_appends_are_valid_json_lines(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: _append(path, f"run-{index}", "COMPLETED"), range(20)))
    assert len(read_ledger(path)) == 20


def test_missing_final_newline_is_read_and_malformed_line_fails(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"; _append(path, "run", "COMPLETED")
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(read_ledger(path)) == 1
    path.write_text(path.read_text() + "\nnot-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed ledger line"):
        read_ledger(path)


def test_diagnostic_cannot_masquerade_as_completed_model_experiment(tmp_path: Path):
    with pytest.raises(ValueError, match="diagnostic"):
        append_ledger_event(
            tmp_path / "x.jsonl", experiment_spec_hash_value="x", experiment_run_id="run",
            event_status="COMPLETED", artifact_kind="RESEARCH_DIAGNOSTIC",
            canonical_model_id=None, requested_model_id=None, registry_hashes={}, source_commit=None,
        )


def _component(index=0, **updates):
    model = ("ridge", "elastic_net", "ordered_logit_ranker")[index % 3]
    date = ("2024-03-15", "2024-09-16", "2025-03-17", "2025-09-15", "2026-03-16")[index // 3]
    value = {
        "experiment_id": f"experiment-{index}", "component_id": f"component-{index}",
        "campaign_id": "wave4", "model_id": model, "decision_date": date,
        "dataset_id": "dataset", "dataset_checksum": "dataset-checksum",
        "daily_spine_id": "spine", "symbol_registry_id": "registry",
        "feature_schema_hash": f"schema-{model}", "target_contract_hash": "target",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "ranking_contract_id": "ranking", "fold_id": f"fold-{date}",
        "purge_sessions": 10, "embargo_sessions": 10,
        "maximum_label_available_timestamp": f"{date}T00:00:00Z",
        "hyperparameters": {}, "random_seed": 42,
        "training_start": "2020-01-01", "training_end": "2023-01-01",
        "source_commit": "commit", "planned_output_root": f"components/{index}",
    }
    value.update(updates)
    return value


def _plan():
    return {"components": [_component(index) for index in range(15)]}


def test_selector_plan_registration_is_atomic_idempotent_and_counts_15(tmp_path):
    path = tmp_path / "selector-ledger.json"
    first = register_selector_plan(path, _plan())
    assert first == register_selector_plan(path, _plan()) == read_selector_ledger(path)
    assert first["trial_counts"] == {
        "fitted_model_count": 3, "decision_date_count": 5, "seed_count": 1,
        "hyperparameter_configuration_count": 1, "planned_material_trials": 15,
        "executed_material_trials": 0, "failed_material_trials": 0,
        "rejected_material_trials": 0,
    }
    assert "momentum" not in {row["model_id"] for row in first["experiments"]}


def test_selector_identity_changes_and_provenance_is_required():
    base = selector_experiment_definition(_component())
    for field, value in (
        ("model_id", "elastic_net"), ("decision_date", "2025-01-01"),
        ("feature_schema_hash", "other"), ("target_contract_hash", "other"),
        ("fold_id", "other"), ("random_seed", 7), ("source_commit", "other"),
    ):
        changed = selector_experiment_definition(_component(**{field: value}))
        assert changed["material_trial_identity"] != base["material_trial_identity"]
    with pytest.raises(ValueError, match="identity missing"):
        selector_experiment_definition(_component(target_provenance_contract_version=None))
    with pytest.raises(ValueError, match="provenance v2"):
        selector_experiment_definition(_component(target_provenance_contract_version="stock_level_target_provenance_v1"))


def test_selector_lifecycle_failure_and_invalid_recovery_are_retained(tmp_path):
    path = tmp_path / "selector-ledger.json"; component = _component()
    register_selector_plan(path, _plan())
    transition_selector_experiment(path, experiment_id="experiment-0", to_status="RUNNING", component=component)
    failed = transition_selector_experiment(
        path, experiment_id="experiment-0", to_status="FAILED", component=component,
        failure_reason="synthetic failure",
    )
    row = next(row for row in failed["experiments"] if row["experiment_id"] == "experiment-0")
    assert [event["status"] for event in row["attempt_history"]] == ["PLANNED", "RUNNING", "FAILED"]
    with pytest.raises(ValueError, match="FAILED -> SUCCEEDED"):
        transition_selector_experiment(path, experiment_id="experiment-0", to_status="SUCCEEDED", component=component)


def test_selector_success_rejection_and_corrupt_checksum_fail_closed(tmp_path):
    path = tmp_path / "selector-ledger.json"; component = _component()
    register_selector_plan(path, _plan())
    transition_selector_experiment(path, experiment_id="experiment-0", to_status="RUNNING", component=component)
    transition_selector_experiment(path, experiment_id="experiment-0", to_status="SUCCEEDED",
                                   component=component, component_manifest_path="manifest.json")
    final = transition_selector_experiment(path, experiment_id="experiment-0", to_status="REJECTED",
                                           component=component, continuation_or_rejection_reason="gate")
    assert final["trial_counts"]["rejected_material_trials"] == 1
    payload = json.loads(path.read_text()); payload["experiments"][0]["model_id"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        read_selector_ledger(path)


def test_selector_write_failure_preserves_prior_ledger(tmp_path, monkeypatch):
    path = tmp_path / "selector-ledger.json"; register_selector_plan(path, _plan())
    before = path.read_bytes()
    def fail(*_args):
        raise OSError("synthetic")
    monkeypatch.setattr("core.research.ml.experiment_ledger.os.replace", fail)
    with pytest.raises(OSError, match="synthetic"):
        transition_selector_experiment(path, experiment_id="experiment-0", to_status="RUNNING",
                                       component=_component())
    assert path.read_bytes() == before
