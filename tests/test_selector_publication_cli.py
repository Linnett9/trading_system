from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import application.services.selector_evaluation_commands as commands


def _args(tmp_path, **updates):
    values = {
        "mode": "ml-selector-parent-gate",
        "symbol_registry_manifest": str(tmp_path / "registry.json"),
        "daily_spine_manifest": str(tmp_path / "spine.json"),
        "daily_feature_manifest": str(tmp_path / "feature.json"),
        "selector_dataset_manifest": str(tmp_path / "dataset.json"),
        "operational_dates_manifest": str(tmp_path / "dates.json"),
        "approved_root": str(tmp_path),
        "required_operational_date": ["2024-03-15"],
        "verification_output": str(tmp_path / "result.json"),
        "parent_gate": str(tmp_path / "gate.json"),
        "selector_dataset_root": str(tmp_path / "dataset"),
        "component_output_root": str(tmp_path / "components"),
        "approved_component_root": [str(tmp_path / "components")],
        "config": str(tmp_path / "config.yaml"),
        "production_plan_job": str(tmp_path / "job.json"),
        "training_rows_json": str(tmp_path / "training.json"),
        "prediction_rows_json": str(tmp_path / "prediction.json"),
        "experiment_ledger": str(tmp_path / "ledger.jsonl"),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_parent_gate_command_success_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "evaluate_selector_parent_publication_gate", lambda **kwargs: {
        "status": "READY", "blockers": [], "logical_checksum": "gate",
    })
    result = commands.run_selector_publication_validate({}, _args(tmp_path))
    assert result["status"] == "READY"
    assert json.loads((tmp_path / "result.json").read_text())["logical_checksum"] == "gate"


def test_parent_gate_blocked_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "evaluate_selector_parent_publication_gate", lambda **kwargs: {
        "status": "BLOCKED", "blockers": ["SPINE_MISMATCH"], "logical_checksum": "gate",
    })
    with pytest.raises(SystemExit) as exc:
        commands.run_selector_publication_validate({}, _args(tmp_path))
    assert exc.value.code == 2


def test_component_readiness_command_and_fifteen_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "assess_selector_component_readiness", lambda **kwargs: {
        "overall_status": "PARTIAL", "expected_component_count": 15,
        "ready_component_count": 0, "missing_component_count": 15,
        "blockers": ["MISSING_COMPONENTS"], "production_plan": [{}] * 15,
    })
    args = _args(tmp_path, verification_output=str(tmp_path / "readiness.json"))
    result = commands.run_selector_component_preflight({}, args)
    assert len(result["production_plan"]) == 15
    assert (tmp_path / "readiness.json").exists()


def test_component_readiness_blocked_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "assess_selector_component_readiness", lambda **kwargs: {
        "overall_status": "BLOCKED", "expected_component_count": 15,
        "ready_component_count": 0, "missing_component_count": 0,
        "blockers": ["BLOCKED_PARENT_GATE"],
    })
    with pytest.raises(SystemExit) as exc:
        commands.run_selector_component_preflight({}, _args(tmp_path))
    assert exc.value.code == 2


def test_single_component_publication_command(tmp_path, monkeypatch):
    for name, payload in (
        ("job.json", {"job_id": "one"}), ("training.json", []), ("prediction.json", []),
    ):
        (tmp_path / name).write_text(json.dumps(payload))
    observed = {}
    def publish(**kwargs):
        observed.update(kwargs)
        return {"status": "SKIPPED_COMPATIBLE", "manifest_path": "component/manifest.json"}
    monkeypatch.setattr(commands, "publish_planned_ordinary_component", publish)
    result = commands.run_selector_component_publish({}, _args(
        tmp_path, verification_output=str(tmp_path / "publication.json")
    ))
    assert result["status"] == "SKIPPED_COMPATIBLE"
    assert observed["job"]["job_id"] == "one"
    assert (tmp_path / "publication.json").exists()


def test_cli_layer_imports_no_replay_evaluation_or_exposure(monkeypatch, tmp_path):
    real_import = __import__
    def guarded(name, *args, **kwargs):
        if any(value in name for value in ("portfolio_replay", "policy_sweep", "exposure")):
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    monkeypatch.setattr(commands, "evaluate_selector_parent_publication_gate", lambda **kwargs: {
        "status": "READY", "blockers": [], "logical_checksum": "gate",
    })
    commands.run_selector_publication_validate({}, _args(tmp_path))
