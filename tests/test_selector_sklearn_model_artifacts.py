from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.research.compute.artifact_storage import validate_artifact_package
from core.research.compute.model_artifacts import (
    inspect_model_metadata,
    read_trusted_model_bytes,
)
from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
    load_trusted_selector_model,
    publish_selector_sklearn_model_package,
    reconstruct_contextual_selector_features,
    resolve_selector_model_package,
)
from core.research.ml.selector_compute_execution import (
    GIB,
    SelectorComputeExecution,
)


class FakeEstimator:
    def __init__(self, marker="one"):
        self.marker = marker

    def get_params(self, deep=True):
        return {"marker": self.marker}


class _State:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeOrderedLogit(FakeEstimator):
    def __init__(self):
        super().__init__("ordered")
        self.beta_ = np.asarray([0.1, 0.2])
        self.thresholds_ = np.asarray([-1.0, 0.0, 1.0, 2.0])
        self.classes_ = np.asarray([0, 1, 2, 3, 4])
        self.imputer_ = _State(statistics_=np.asarray([0.0, 1.0]))
        self.scaler_ = _State(
            mean_=np.asarray([0.0, 1.0]),
            scale_=np.asarray([1.0, 2.0]),
        )


def _prediction(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "score"])
        writer.writeheader()
        writer.writerow({"row_id": "one", "score": "0.2"})


def _publish(tmp_path, *, model_id="ridge", estimator=None, **changes):
    prediction = tmp_path / "component" / "predictions.csv"
    if not prediction.exists():
        _prediction(prediction)
    values = {
        "component_root": tmp_path / "component",
        "estimator": estimator or FakeEstimator(),
        "preprocessing": {
            "ordered_feature_ids": ["a", "b"],
            "location": [0.0, 1.0],
            "scale": [1.0, 2.0],
        },
        "feature_order": ["a", "b"],
        "model_id": model_id,
        "model_family": model_id,
        "model_configuration": {"alpha": 1.0},
        "random_seed": 42,
        "training_boundary": {
            "training_cutoff": "2024-12-31",
            "fold_identity": "fold",
        },
        "training_population_checksum": "training-population",
        "target_horizon_identity": "forward_return_10d",
        "prediction_path": prediction,
        "prediction_schema": ["row_id", "score"],
        "prediction_count": 1,
        "input_population_checksum": "input-population",
        "output_population_checksum": "output-population",
        "campaign_identity": "campaign",
        "plan_job_identity": "job",
        "component_identity": "component",
        "component_runner": "runner",
        "runtime_owner": "runtime",
        "implementation_owner": "implementation",
        "decision_date": "2025-01-02",
        "fold_identity": "fold",
        "training_row_artifact_identity": "training-rows",
        "prediction_row_artifact_identity": "prediction-rows",
        "source_schema_guarantee_identity": "schema",
        "input_package_identity": "input-package",
        "source_git_commit": "commit",
    }
    values.update(changes)
    return publish_selector_sklearn_model_package(**values)


@pytest.mark.parametrize(
    "model_id", ["ridge", "elastic_net", "huber"]
)
def test_single_model_families_publish_complete_packages(tmp_path, model_id):
    result = _publish(tmp_path, model_id=model_id)
    assert result["completion_status"] == "COMPLETE"
    manifest = validate_artifact_package(Path(result["model_package_path"]))
    assert manifest["serialization_handler"] == "SKLEARN_PIPELINE"
    assert manifest["promotion_status"] == "NOT_PROMOTED"
    assert manifest["model_metadata"]["feature_order"] == ["a", "b"]
    binding = validate_artifact_package(Path(result["prediction_package_path"]))
    assert (
        binding["prediction_model_binding"]["fitted_model_package_checksum"]
        == manifest["package_checksum"]
    )


def test_ordered_logit_preserves_complete_trusted_state(tmp_path):
    result = _publish(
        tmp_path,
        model_id="ordered_logit_ranker",
        estimator=FakeOrderedLogit(),
        preprocessing=None,
    )
    metadata = inspect_model_metadata(Path(result["model_package_path"]))
    evidence = metadata["model_metadata"]["ordered_logit_evidence"]
    assert evidence["coefficient_values"] == [0.1, 0.2]
    assert evidence["threshold_values"] == [-1.0, 0.0, 1.0, 2.0]
    assert evidence["class_order"] == [0, 1, 2, 3, 4]
    assert evidence["link_identity"] == "cumulative_logit"
    with pytest.raises(PermissionError):
        read_trusted_model_bytes(
            Path(result["model_package_path"]),
            "model/estimator.joblib",
            trusted_artifact=False,
        )


def test_ridge_composite_and_ordered_logit_reload_without_refit(tmp_path):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([[1.0, np.nan], [2.0, 1.0], [3.0, 0.0], [4.0, 2.0]])
    y = np.asarray([0.2, 0.4, 0.7, 1.1])
    ridge = TransformedTargetRegressor(
        regressor=make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge()
        ),
        transformer=StandardScaler(),
    ).fit(x, y)
    expected = ridge.predict(x)
    ridge_result = _publish(
        tmp_path / "ridge",
        estimator=ridge,
        preprocessing=None,
        model_configuration=ridge.get_params(),
    )
    ridge.fit = lambda *args, **kwargs: pytest.fail("reload refitted ridge")
    reloaded_ridge = load_trusted_selector_model(
        Path(ridge_result["model_package_path"]), trusted_artifact=True
    )
    assert reloaded_ridge.predict(x) == pytest.approx(expected)

    from core.research.ml.ranking import OrderedLogitRanker

    ordered = OrderedLogitRanker()
    ordered.imputer_ = SimpleImputer(strategy="median").fit(x)
    imputed = ordered.imputer_.transform(x)
    ordered.scaler_ = StandardScaler().fit(imputed)
    ordered.beta_ = np.asarray([0.3, -0.2])
    ordered.thresholds_ = np.asarray([-1.5, -0.5, 0.5, 1.5])
    ordered.classes_ = np.arange(5)
    ordered_expected = ordered.predict_proba(x)
    ordered_result = _publish(
        tmp_path / "ordered",
        model_id="ordered_logit_ranker",
        estimator=ordered,
        preprocessing=None,
    )
    ordered.fit = lambda *args, **kwargs: pytest.fail(
        "reload refitted ordered logit"
    )
    reloaded_ordered = load_trusted_selector_model(
        Path(ordered_result["model_package_path"]), trusted_artifact=True
    )
    assert reloaded_ordered.predict_proba(x) == pytest.approx(ordered_expected)


def test_contextual_evidence_and_reconstruction_are_ordered(tmp_path):
    context = {
        "ordered_stock_feature_ids": ["stock"],
        "ordered_market_context_ids": ["market"],
        "interaction_specification": [
            {
                "interaction_id": "stock_x_market",
                "stock_feature_id": "stock",
                "market_context_id": "market",
            }
        ],
        "interaction_output_order": [
            "stock:stock", "interaction:stock_x_market"
        ],
        "pit_context_ancestry": {"identity": "pit"},
        "context_source_guarantee_identity": "context-schema",
    }
    result = _publish(
        tmp_path,
        model_id="contextual_elastic_net",
        feature_order=["stock:stock", "interaction:stock_x_market"],
        preprocessing={
            "stock_feature_ids": ["stock"],
            "context_feature_ids": ["market"],
            "stock_location": [1.0],
            "stock_scale": [2.0],
            "context_location": [2.0],
            "context_scale": [4.0],
            "stock_lower": None,
            "stock_upper": None,
            "context_lower": None,
            "context_upper": None,
        },
        contextual_evidence=context,
    )
    metadata = inspect_model_metadata(Path(result["model_package_path"]))
    assert (
        metadata["model_metadata"]["contextual_evidence"][
            "interaction_output_order"
        ]
        == ["stock:stock", "interaction:stock_x_market"]
    )
    assert reconstruct_contextual_selector_features(
        [{"stock": 3.0, "market": 6.0}],
        preprocessing={
            "stock_feature_ids": ["stock"],
            "context_feature_ids": ["market"],
            "stock_location": [1.0],
            "stock_scale": [2.0],
            "context_location": [2.0],
            "context_scale": [4.0],
            "stock_lower": None,
            "stock_upper": None,
            "context_lower": None,
            "context_upper": None,
        },
        contextual_evidence=context,
    ) == [[1.0, 1.0]]


@pytest.mark.parametrize(
    "change",
    [
        {"feature_order": ["b", "a"]},
        {"preprocessing": {
            "ordered_feature_ids": ["a", "b"],
            "location": [9.0, 1.0],
            "scale": [1.0, 2.0],
        }},
        {"estimator": FakeEstimator("changed")},
        {"random_seed": 7},
        {"fold_identity": "changed-fold"},
        {"decision_date": "2025-01-03"},
        {"target_horizon_identity": "forward_return_20d"},
        {"training_population_checksum": "changed-training-population"},
        {"output_population_checksum": "changed-prediction-population"},
        {"input_population_checksum": "changed-input-population"},
        {"input_package_identity": "changed-input-package"},
    ],
)
def test_changed_model_ownership_is_not_overwritten(tmp_path, change):
    _publish(tmp_path)
    with pytest.raises(FileExistsError, match="INCOMPATIBLE_EXISTING"):
        _publish(tmp_path, **change)


def test_exact_existing_package_skips_and_predictions_alone_do_not_qualify(tmp_path):
    first = _publish(tmp_path)
    second = _publish(tmp_path)
    assert first["completion_status"] == "COMPLETE"
    assert second["compatible_skip_status"] == "SKIPPED_COMPATIBLE"
    (Path(second["model_package_path"]) / "completion.json").unlink()
    with pytest.raises(ValueError, match="partial|corrupt"):
        validate_artifact_package(Path(second["model_package_path"]))


def test_serialization_failure_leaves_no_valid_package(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "joblib.dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic serialization failure")
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        _publish(tmp_path)
    root = tmp_path / "component" / "shared_model_artifact" / "model"
    assert not (root / "manifest.json").exists()
    assert not (root / "completion.json").exists()


def test_preprocessing_feature_and_prediction_contracts_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="PREPROCESSING"):
        _publish(tmp_path / "missing-prep", preprocessing=None)
    with pytest.raises(ValueError, match="FEATURE_ORDER"):
        _publish(tmp_path / "missing-features", feature_order=[])
    wrong = tmp_path / "wrong-count"
    with pytest.raises(ValueError, match="count"):
        _publish(wrong, prediction_count=2)
    wrong_schema = tmp_path / "wrong-schema"
    with pytest.raises(ValueError, match="schema"):
        _publish(
            wrong_schema, prediction_schema=["score", "row_id"]
        )


def test_compute_hook_resolves_only_complete_bound_component(tmp_path):
    result = _publish(tmp_path)
    component_root = tmp_path / "component"
    prediction = component_root / "predictions.csv"
    component = {
        "publication_status": "complete",
        "production_plan_job_checksum": "JOB-CHECKSUM",
        "selector_model_identity": "ridge",
        "prediction_date": "2025-01-02",
        "prediction_artifact_path": str(prediction),
        "prediction_checksum": __import__("hashlib").sha256(
            prediction.read_bytes()
        ).hexdigest(),
        "shared_model_artifact": result,
    }
    manifest_path = component_root / "manifest.json"
    manifest_path.write_text(json.dumps(component), encoding="utf-8")
    job = {
        "job_id": "job",
        "logical_checksum": "JOB-CHECKSUM",
        "model_id": "ridge",
        "prediction_date": "2025-01-02",
        "campaign_identity": "campaign",
        "authoritative_output_root": str(component_root),
    }
    resolved = resolve_selector_model_package(
        job=job,
        component_result={"manifest_path": str(manifest_path)},
        run_identity="run",
    )
    assert resolved["completion_status"] == "COMPLETE"
    assert resolved["artifact_identity"] == result["artifact_identity"]

    missing = tmp_path / "prediction-only"
    missing.mkdir()
    (missing / "manifest.json").write_text(
        json.dumps({**component, "prediction_artifact_path": str(prediction)}),
        encoding="utf-8",
    )
    assert resolve_selector_model_package(
        job={**job, "authoritative_output_root": str(missing)},
        component_result=None,
        run_identity="run",
    ) is None


def test_default_compute_hook_publishes_inventory_result_and_summary(tmp_path):
    job_id = "selector:2025-01-02:ridge"
    result = _publish(
        tmp_path,
        plan_job_identity=job_id,
        component_identity=job_id,
    )
    component_root = tmp_path / "component"
    prediction = component_root / "predictions.csv"
    job = {
        "job_id": job_id,
        "model_id": "ridge",
        "prediction_date": "2025-01-02",
        "horizon_id": None,
        "selector_dataset_root": "dataset",
        "authoritative_output_root": str(component_root),
        "feature_schema": "schema",
        "target_contract": "target",
        "expected_parent_gate_checksum": "gate",
        "expected_dataset_checksum": "dataset",
        "dependency_state": "ready",
        "overwrite_policy": "never",
        "resume_policy": "compatible",
        "logical_checksum": "JOB-CHECKSUM",
        "campaign_identity": "campaign",
    }
    component = {
        "publication_status": "complete",
        "production_plan_job_checksum": "JOB-CHECKSUM",
        "selector_model_identity": "ridge",
        "prediction_date": "2025-01-02",
        "prediction_artifact_path": str(prediction),
        "prediction_checksum": hashlib.sha256(
            prediction.read_bytes()
        ).hexdigest(),
        "shared_model_artifact": result,
    }
    (component_root / "manifest.json").write_text(
        json.dumps(component), encoding="utf-8"
    )
    campaign = {
        "campaign_contract": "selector_research_campaign.v1",
        "campaign_version": "v2",
        "campaign_id": "fixture",
        "campaign_identity": "campaign",
        "expected_component_count": 1,
        "fitted_component_matrix": [{
            "job_id": job_id,
            "model_id": "ridge",
            "prediction_date": "2025-01-02",
            "horizon_id": None,
            "component_runner": "fixture:runner",
        }],
    }
    campaign["logical_checksum"] = hashlib.sha256(
        json.dumps(
            campaign, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest().upper()
    adapter = SelectorComputeExecution(
        jobs=[job],
        campaign_manifest=campaign,
        readiness={"logical_checksum": "PLAN"},
        run_id="run",
        source_git_commit="commit",
        runs_root=tmp_path / "runs",
        lease_ledger_path=tmp_path / "ledger.json",
        registry_path=tmp_path / "registry.json",
        available_memory=lambda: 32 * GIB,
    )
    outcome = adapter.execute_component(
        job=job,
        command=["must-not-launch"],
        environment={},
        report_path=tmp_path / "unused.json",
        transcript_path=tmp_path / "unused.txt",
        popen=lambda *args, **kwargs: pytest.fail("compatible skip launched"),
    )
    adapter.close()
    assert outcome["status"] == "SKIPPED_COMPATIBLE"
    assert (adapter.run_root / "artifact_inventory.csv").read_text(
        encoding="utf-8"
    ).count("selector-model:") == 1
    results = json.loads(
        (adapter.run_root / "results.json").read_text(encoding="utf-8")
    )
    assert results["records"][0]["result_kind"] == "MODEL_COMPONENT"
    assert (adapter.run_root / "summary.md").exists()
