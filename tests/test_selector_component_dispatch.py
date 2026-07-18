from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from application.services.selector_evaluation_commands import (
    ORDINARY_RUNTIME_OWNER,
    WAVE4_RUNTIME_OWNER,
    dispatch_selector_component_publication,
)
from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.registries.io import canonical_hash


MODELS = (
    "ridge", "elastic_net", "ordered_logit_ranker", "huber",
    "contextual_elastic_net", "multi_horizon_ridge",
    "multi_horizon_elastic_net", "lightgbm_rank_xendcg",
    "lightgbm_lambdarank",
)


def _fixture(model: str):
    horizon = "return_5s" if model.startswith("multi_horizon_") else None
    job_id = f"selector:2025-03-17:{model}" + (
        f":{horizon}" if horizon else ""
    )
    runner = selector_model_adapter(
        model, runner="ordinary"
    ).constructor_owner
    matrix = [{
        "job_id": job_id, "model_id": model,
        "prediction_date": "2025-03-17", "horizon_id": horizon,
        "component_runner": runner,
    }]
    campaign = {
        "campaign_contract": "selector_research_campaign.v1",
        "campaign_version": "v2", "campaign_id": "dispatch-fixture",
        "campaign_identity": "CAMPAIGN", "fitted_component_matrix": matrix,
        "expected_component_count": 1,
    }
    campaign["logical_checksum"] = _hash(campaign)
    job = {
        **matrix[0], "logical_checksum": "JOB-CHECKSUM",
        "authoritative_output_root": "unused-component-owner",
    }
    gate = {"selector_dataset_artifact_checksum": "DATASET"}
    package = {
        "package_id": f"package:{job_id}",
        "production_plan_job_id": job_id,
        "production_plan_job_checksum": "JOB-CHECKSUM",
        "model_id": model, "prediction_date": "2025-03-17",
        "selector_dataset_checksum": "DATASET",
        "fold_identity": "FOLD", "training_cutoff": "2025-02-01",
        "purge_sessions": 20, "embargo_sessions": 5,
        "source_git_commit": "COMMIT", "publication_status": "complete",
    }
    package["logical_checksum"] = canonical_hash(package)
    return campaign, job, gate, package, runner


def _dispatch(model: str, **changes):
    campaign, job, gate, package, runner = _fixture(model)
    for target, values in changes.items():
        {"campaign": campaign, "job": job, "package": package}[target].update(
            values
        )
    if "campaign" in changes:
        campaign["logical_checksum"] = _logical_hash(campaign)
    if "package" in changes:
        package["logical_checksum"] = canonical_hash({
            key: value for key, value in package.items()
            if key != "logical_checksum"
        })
    calls = []

    def ordinary(**kwargs):
        calls.append(("ordinary", kwargs))
        return {"status": "ORDINARY"}

    def wave4(**kwargs):
        calls.append(("wave4", kwargs))
        return {"status": "WAVE4"}

    result = dispatch_selector_component_publication(
        campaign=campaign, job=job, package=package,
        training_rows=[], prediction_rows=[], parent_gate=gate,
        parent_gate_path=Path("gate.json"), ledger_path=Path("ledger.jsonl"),
        supplied_campaign_identity="CAMPAIGN",
        supplied_plan_job_identity=job["job_id"],
        supplied_component_runner=runner,
        ordinary_publisher=ordinary, wave4_publisher=wave4,
        wave4_adapter=lambda **kwargs: {
            "fit_input": {"logical_input_checksum": "INPUT"}
        },
    )
    return result, calls


@pytest.mark.parametrize("model", MODELS)
def test_campaign_runner_routes_to_exact_registered_publication_owner(model):
    result, calls = _dispatch(model)
    expected = "ordinary" if model in {
        "ridge", "elastic_net", "ordered_logit_ranker"
    } else "wave4"
    assert result["status"] == expected.upper()
    assert [name for name, _ in calls] == [expected]
    evidence = calls[0][1]
    assert evidence["campaign_identity"] == "CAMPAIGN"
    assert evidence["plan_job_identity"].endswith(
        model if not model.startswith("multi_horizon_") else f"{model}:return_5s"
    )
    assert evidence["operational_input_identity"].startswith("package:")
    assert evidence["resolved_runtime_owner"] == (
        ORDINARY_RUNTIME_OWNER if expected == "ordinary"
        else WAVE4_RUNTIME_OWNER
    )


def test_unknown_or_registry_mismatched_runner_fails_before_publication():
    campaign, job, gate, package, _ = _fixture("huber")
    campaign["fitted_component_matrix"][0]["component_runner"] = "unknown:owner"
    campaign["logical_checksum"] = _logical_hash(campaign)
    with pytest.raises(ValueError, match="registry adapter"):
        dispatch_selector_component_publication(
            campaign=campaign, job=job, package=package,
            training_rows=[], prediction_rows=[], parent_gate=gate,
            parent_gate_path=Path("gate"), ledger_path=Path("ledger"),
            supplied_campaign_identity="CAMPAIGN",
            supplied_plan_job_identity=job["job_id"],
            supplied_component_runner="unknown:owner",
            ordinary_publisher=lambda **kwargs: pytest.fail("ordinary called"),
            wave4_publisher=lambda **kwargs: pytest.fail("wave4 called"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("campaign_identity", "WRONG", "Campaign identity mismatch"),
        ("plan_job_identity", "", "Plan-job identity is required"),
        ("model_id", "ridge", "model_id mismatch"),
        ("prediction_date", "2025-03-18", "prediction_date mismatch"),
        ("horizon_id", "return_1s", "horizon_id mismatch"),
    ),
)
def test_identity_mismatches_fail_before_publication(field, value, message):
    campaign, job, gate, package, runner = _fixture("huber")
    supplied = {
        "supplied_campaign_identity": "CAMPAIGN",
        "supplied_plan_job_identity": job["job_id"],
    }
    if field in {"campaign_identity", "plan_job_identity"}:
        supplied[f"supplied_{field}"] = value
    else:
        job[field] = value
    with pytest.raises(ValueError, match=message):
        dispatch_selector_component_publication(
            campaign=campaign, job=job, package=package,
            training_rows=[], prediction_rows=[], parent_gate=gate,
            parent_gate_path=Path("gate"), ledger_path=Path("ledger"),
            supplied_component_runner=runner, **supplied,
            ordinary_publisher=lambda **kwargs: pytest.fail("ordinary called"),
            wave4_publisher=lambda **kwargs: pytest.fail("wave4 called"),
        )


def test_operational_package_identity_is_revalidated():
    with pytest.raises(ValueError, match="package identity mismatch"):
        _dispatch("ridge", package={"model_id": "elastic_net"})


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest().upper()


def _logical_hash(value):
    return _hash({
        key: item for key, item in value.items()
        if key != "logical_checksum"
    })
