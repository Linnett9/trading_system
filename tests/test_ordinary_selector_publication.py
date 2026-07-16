from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.experiment_ledger import read_ledger
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.adapters import verify_registry_capabilities
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_component_readiness import assess_selector_component_readiness
from core.research.ml.stock_level.ordinary_selector_publication import (
    publish_planned_ordinary_component,
)


DATE = "2024-03-15"


def _gate(tmp_path, status="READY"):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({
        "gate_contract_version": "selector_parent_publication_gate.v1",
        "status": status, "logical_checksum": "GATE",
        "selector_dataset_id": "dataset",
        "selector_dataset_artifact_checksum": "DATASET",
        "canonical_registry_id": "registry", "daily_spine_id": "spine",
        "selector_feature_schema_checksum": "FEATURE-HASH",
    }))
    return path


def _job(tmp_path, model, **updates):
    resolver = RegistryResolver(load_registry_bundle())
    payload = resolver.resolve("selector_models", model, role="selector").entry.payload
    value = {
        "job_id": f"selector:{DATE}:{model}", "model_id": model,
        "prediction_date": DATE,
        "selector_dataset_root": str(tmp_path / "dataset"),
        "authoritative_output_root": str(tmp_path / "components" / f"model={model}" / f"date={DATE}"),
        "feature_schema": payload["feature_schema"],
        "target_contract": payload["target_contract"],
        "ranking_contract": payload.get("ranking_problem_contract"),
        "relevance_contract": payload.get("relevance_contract"),
        "expected_parent_gate_checksum": "GATE",
        "expected_dataset_checksum": "DATASET",
        "dependency_state": "MISSING",
        "command_template": "not executed",
        "overwrite_policy": "never_overwrite_complete_component",
        "resume_policy": "resume_only_incomplete_owned_component",
    }
    value.update(updates)
    value["logical_checksum"] = canonical_hash({
        key: item for key, item in value.items() if key != "logical_checksum"
    })
    return value


def _rows(model):
    schema = json.loads(Path(_job(Path("."), model)["feature_schema"]).read_text())
    features = [row["name"] for row in schema["features"]]
    train = []
    for day_index, day in enumerate(("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")):
        for asset_index in range(5):
            row = {
                "row_id": f"{day}-{asset_index}", "asset_id": f"A{asset_index}",
                "symbol": f"S{asset_index}", "decision_timestamp": day,
                "label_available_timestamp": "2024-02-01",
                "actual_forward_return_10d": asset_index + day_index / 10,
            }
            row.update({name: asset_index + day_index / 10 + offset / 1000 for offset, name in enumerate(features)})
            train.append(row)
    predict = []
    for asset_index in range(5):
        row = {
            "row_id": f"p-{asset_index}", "asset_id": f"A{asset_index}",
            "symbol": f"S{asset_index}", "decision_timestamp": DATE,
        }
        row.update({name: asset_index + offset / 1000 for offset, name in enumerate(features)})
        predict.append(row)
    return train, predict


def _publish(tmp_path, model="ridge", **kwargs):
    train, predict = _rows(model)
    values = {
        "job": _job(tmp_path, model), "parent_gate_path": _gate(tmp_path),
        "training_rows": train, "prediction_rows": predict,
        "ledger_path": tmp_path / "ledger.jsonl",
    }
    values.update(kwargs)
    return publish_planned_ordinary_component(**values), values


@pytest.mark.parametrize("model", ["ridge", "elastic_net", "ordered_logit_ranker"])
def test_model_publication_and_complete_manifest(tmp_path, model):
    result, values = _publish(tmp_path, model)
    assert result["status"] == "COMPLETED"
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["component_schema_version"] == "authoritative_selector_component_v1"
    assert manifest["selector_model_identity"] == model
    assert manifest["validation_status"] == "VERIFIED_STRICT_OOS"
    assert manifest["prediction_row_count"] == 5
    assert manifest["artifact_link"]["verification_status"] == "VERIFIED_STRICT_OOS"
    assert [row["event_status"] for row in read_ledger(values["ledger_path"])] == ["STARTED", "COMPLETED"]


def test_ordered_logit_semantic_outputs(tmp_path):
    result, _ = _publish(tmp_path, "ordered_logit_ranker")
    with (Path(result["manifest_path"]).parent / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        probabilities = [float(row[f"ordered_logit_probability_{index}"]) for index in range(5)]
        assert sum(probabilities) == pytest.approx(1.0)
        assert float(row["ordered_logit_expected_relevance"]) == pytest.approx(float(row["selector_score"]))
    assert sorted(int(row["deterministic_rank"]) for row in rows) == [1, 2, 3, 4, 5]


def test_deterministic_asset_tie_breaking(tmp_path, monkeypatch):
    import core.research.ml.stock_level.ordinary_selector_publication as publication
    class Tied:
        def fit(self, x, y): return self
        def predict(self, x): return [1.0] * len(x)
        def get_params(self): return {}
    monkeypatch.setattr(publication, "_build_tabular_model", lambda *args: Tied())
    result, _ = _publish(tmp_path, "ridge")
    rows = list(csv.DictReader((Path(result["manifest_path"]).parent / "predictions.csv").open()))
    assert [(row["asset_id"], int(row["deterministic_rank"])) for row in rows] == [
        ("A0", 1), ("A1", 2), ("A2", 3), ("A3", 4), ("A4", 5)
    ]


@pytest.mark.parametrize(
    "case,match",
    [
        ("blocked_gate", "Parent gate not ready"),
        ("plan_checksum", "Production-plan checksum mismatch"),
        ("dataset", "Dataset mismatch"),
        ("feature", "Feature-schema mismatch"),
        ("target", "Target contract mismatch"),
        ("ranking", "Ranking-contract mismatch"),
        ("immature", "Immature"),
        ("overlap", "overlap"),
        ("nonfinite", "Nonfinite"),
        ("duplicate", "Duplicate"),
        ("incomplete", "Incomplete"),
    ],
)
def test_fail_closed_inputs(tmp_path, case, match):
    job = _job(tmp_path, "ordered_logit_ranker")
    gate = _gate(tmp_path)
    train, predict = _rows("ordered_logit_ranker")
    if case == "blocked_gate": gate = _gate(tmp_path, "BLOCKED")
    elif case == "plan_checksum": job["logical_checksum"] = "wrong"
    elif case == "dataset":
        job["expected_dataset_checksum"] = "wrong"
        job["logical_checksum"] = canonical_hash({k: v for k, v in job.items() if k != "logical_checksum"})
    elif case == "feature":
        job["feature_schema"] = "wrong"
        job["logical_checksum"] = canonical_hash({k: v for k, v in job.items() if k != "logical_checksum"})
    elif case == "target":
        job["target_contract"] = "future_volatility"
        job["logical_checksum"] = canonical_hash({k: v for k, v in job.items() if k != "logical_checksum"})
    elif case == "ranking":
        job["ranking_contract"] = "wrong"
        job["logical_checksum"] = canonical_hash({k: v for k, v in job.items() if k != "logical_checksum"})
    elif case == "immature": train[0]["label_available_timestamp"] = "2025-01-01"
    elif case == "overlap": train[0]["decision_timestamp"] = DATE
    elif case == "nonfinite": train[0][next(iter(json.loads(Path(job["feature_schema"]).read_text())["features"]))["name"]] = float("nan")
    elif case == "duplicate": predict[1]["row_id"] = predict[0]["row_id"]
    elif case == "incomplete": predict = []
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        publish_planned_ordinary_component(
            job=job, parent_gate_path=gate, training_rows=train,
            prediction_rows=predict, ledger_path=tmp_path / "ledger.jsonl",
        )


def test_compatible_resume_skips_fitting(tmp_path, monkeypatch):
    result, values = _publish(tmp_path, "ridge")
    import core.research.ml.stock_level.ordinary_selector_publication as publication
    monkeypatch.setattr(publication, "_build_tabular_model", lambda *args: (_ for _ in ()).throw(AssertionError("fit")))
    second = publish_planned_ordinary_component(**values)
    assert second["status"] == "SKIPPED_COMPATIBLE"
    assert [row["event_status"] for row in read_ledger(values["ledger_path"])] == ["STARTED", "COMPLETED", "SKIPPED_COMPLETE"]


def test_incomplete_replacement_and_incompatible_complete_rejection(tmp_path):
    job = _job(tmp_path, "ridge"); owner = Path(job["authoritative_output_root"])
    owner.mkdir(parents=True); (owner / "partial").write_text("x")
    train, predict = _rows("ridge")
    result = publish_planned_ordinary_component(
        job=job, parent_gate_path=_gate(tmp_path), training_rows=train,
        prediction_rows=predict, ledger_path=tmp_path / "ledger.jsonl",
    )
    assert result["status"] == "COMPLETED"
    manifest = Path(result["manifest_path"]); payload = json.loads(manifest.read_text())
    payload["production_plan_job_checksum"] = "wrong"; manifest.write_text(json.dumps(payload))
    with pytest.raises(FileExistsError, match="Incompatible complete"):
        publish_planned_ordinary_component(
            job=job, parent_gate_path=_gate(tmp_path), training_rows=train,
            prediction_rows=predict, ledger_path=tmp_path / "ledger.jsonl",
        )


def test_component_readiness_accepts_published_fixture(tmp_path):
    result, _ = _publish(tmp_path, "ridge")
    gate = _gate(tmp_path)
    readiness = assess_selector_component_readiness(
        parent_gate_path=gate,
        authoritative_root=tmp_path / "components",
        selector_dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "config",
        approved_component_roots=(tmp_path / "components",),
    )
    row = next(row for row in readiness["component_matrix"] if row["model_id"] == "ridge" and row["prediction_date"] == DATE)
    assert row["state"] == "READY"


def test_ordinary_capability_is_truthful():
    resolver = RegistryResolver(load_registry_bundle())
    assert resolver.resolve("selector_models", "ordered_logit_ranker").entry.payload["ordinary_runner_support"] is True
    assert verify_registry_capabilities()["ordinary"] >= 1


def test_no_replay_policy_or_exposure_import(monkeypatch, tmp_path):
    real_import = __import__
    def guarded(name, *args, **kwargs):
        if any(term in name for term in ("portfolio_replay", "policy_sweep", "exposure")):
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    _publish(tmp_path, "ridge")


def test_invalid_ordered_logit_probabilities_are_rejected(tmp_path, monkeypatch):
    import core.research.ml.stock_level.ordinary_selector_publication as publication
    class InvalidProbabilities:
        diagnostics = {}
        def fit(self, x, y, **kwargs): return self
        def predict(self, x): return list(range(len(x)))
        def predict_proba(self, x): return [[0.5] * 5 for _ in x]
        def get_params(self): return {}
    monkeypatch.setattr(publication, "_build_tabular_model", lambda *args: InvalidProbabilities())
    with pytest.raises(ValueError, match="Invalid ordered-logit probabilities"):
        _publish(tmp_path, "ordered_logit_ranker")
    assert read_ledger(tmp_path / "ledger.jsonl")[-1]["event_status"] == "REJECTED"


def test_unexpected_fitter_failure_is_recorded(tmp_path, monkeypatch):
    import core.research.ml.stock_level.ordinary_selector_publication as publication
    class Broken:
        def fit(self, x, y): raise RuntimeError("synthetic fitter failure")
    monkeypatch.setattr(publication, "_build_tabular_model", lambda *args: Broken())
    with pytest.raises(RuntimeError, match="synthetic fitter failure"):
        _publish(tmp_path, "ridge")
    assert read_ledger(tmp_path / "ledger.jsonl")[-1]["event_status"] == "FAILED"
