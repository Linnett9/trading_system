from __future__ import annotations

import copy

import pytest

from core.research.ml.selector_component_scheduler import (
    run_component_jobs,
    validate_component_plan,
)
from core.research.ml.selector_component_batch import (
    run_stage10_component_batch,
)
from core.research.ml.selector_component_readiness import READINESS_CONTRACT
from core.research.ml.selector_research_campaign import (
    build_selector_research_campaign,
    historical_stage10_baseline_campaign,
)
from core.research.ml.selector_research_protocol import (
    REQUIRED_IDENTITIES,
    freeze_selector_research_protocol,
)


def test_protocol_and_campaign_are_deterministic_and_frozen():
    first_protocol = _protocol()
    second_protocol = _protocol()
    first = build_selector_research_campaign(first_protocol)
    second = build_selector_research_campaign(second_protocol)

    assert first_protocol == second_protocol
    assert first == second
    assert first["expected_component_count"] == 75
    assert len(first["fitted_component_matrix"]) == 75
    assert first["campaign_identity"] == second["campaign_identity"]
    assert first["logical_checksum"] == second["logical_checksum"]
    assert first["training_performed"] is False
    assert first["evaluation_performed"] is False


def test_historical_fifteen_job_campaign_remains_readable():
    campaign = historical_stage10_baseline_campaign()

    assert campaign["historical_identity_preserved"] is True
    assert campaign["expected_component_count"] == 15
    assert len(campaign["fitted_component_matrix"]) == 15
    assert {
        row["model_id"] for row in campaign["fitted_component_matrix"]
    } == {"ridge", "elastic_net", "ordered_logit_ranker"}


def test_campaign_phases_roles_horizons_and_readiness_are_explicit():
    campaign = build_selector_research_campaign(_protocol())
    matrix = campaign["fitted_component_matrix"]
    multi = [row for row in matrix if row["model_id"].startswith("multi_horizon_")]

    assert len(multi) == 40
    assert len({row["job_id"] for row in multi}) == 40
    assert {row["horizon_id"] for row in multi} == {
        "return_1s", "return_5s", "return_10s", "return_20s"
    }
    assert {
        row["model_id"] for row in campaign["diagnostic_components"]
    } == {"momentum_120d", "risk_adjusted_momentum"}
    assert all(
        row["component_role"] == "DIAGNOSTIC_NON_FITTED"
        for row in campaign["diagnostic_components"]
    )
    readiness = {
        row["model_id"]: row for row in campaign["model_readiness"]
    }
    for model in (
        "ridge", "elastic_net", "ordered_logit_ranker", "huber",
        "contextual_elastic_net", "multi_horizon_ridge",
        "multi_horizon_elastic_net",
    ):
        assert readiness[model]["campaign"] == "CAMPAIGN_READY"
    assert readiness["multi_horizon_ordered_logit"]["campaign"] == "DEFERRED"
    assert readiness["lightgbm_rank_xendcg"]["campaign"] == "CAMPAIGN_READY"
    assert readiness["lightgbm_lambdarank"]["campaign"] == "CAMPAIGN_READY"
    assert (
        readiness["lightgbm_rank_xendcg"]["required_ranking_contract"]
        == "daily_cross_sectional_ranking_problem_v1"
    )
    assert readiness["lightgbm_rank_xendcg"][
        "required_relevance_representation"
    ] == "integer"


def test_every_admitted_job_has_registry_and_component_adapter():
    campaign = build_selector_research_campaign(_protocol())

    assert all(
        row["model_registry_entry_checksum"]
        and row["component_runner"]
        and row["target_contract"]
        for row in campaign["fitted_component_matrix"]
    )
    ordered = next(
        row for row in campaign["fitted_component_matrix"]
        if row["model_id"] == "ordered_logit_ranker"
    )
    assert ordered["ranking_contract"] == "daily_cross_sectional_ranking_problem_v1"


@pytest.mark.parametrize("change", ["count", "missing", "unexpected", "duplicate"])
def test_runtime_inventory_must_equal_campaign_manifest(change):
    campaign = build_selector_research_campaign(_protocol())
    jobs = [
        _runtime_job(row, campaign["campaign_identity"])
        for row in campaign["fitted_component_matrix"]
    ]
    if change in {"count", "missing"}:
        jobs.pop()
        message = "count differs"
    elif change == "unexpected":
        jobs[-1] = {
            **jobs[-1],
            "model_id": "unexpected",
            "job_id": f"selector:{jobs[-1]['prediction_date']}:unexpected",
        }
        message = "differ from campaign"
    else:
        jobs[-1] = copy.deepcopy(jobs[0])
        message = "Duplicate"

    with pytest.raises(ValueError, match=message):
        validate_component_plan(jobs, campaign_manifest=campaign)


def test_scheduler_uses_manifest_count_and_preserves_deterministic_reporting():
    campaign = build_selector_research_campaign(_protocol())
    jobs = [
        _runtime_job(row, campaign["campaign_identity"])
        for row in campaign["fitted_component_matrix"]
    ]
    reordered = list(reversed(jobs))

    result = run_component_jobs(
        reordered,
        runner=lambda job: {"status": "COMPLETED", "job": job["job_id"]},
        max_component_workers=3,
        capacity=4,
        campaign_manifest=campaign,
    )

    assert len(result) == campaign["expected_component_count"]
    assert [row["job_id"] for row in result] == [
        row["job_id"] for row in campaign["fitted_component_matrix"]
    ]
    assert all(row["status"] == "COMPLETED" for row in result)


def test_batch_uses_manifest_cardinality_and_identity(tmp_path):
    campaign = build_selector_research_campaign(_protocol())
    jobs = [
        _runtime_job(row, campaign["campaign_identity"])
        for row in campaign["fitted_component_matrix"]
    ]
    inventory = {
        "packages": [
            {
                "job_id": job["job_id"],
                "training_rows_path": f"training/{index}.json",
                "prediction_rows_path": f"prediction/{index}.json",
            }
            for index, job in enumerate(jobs)
        ]
    }
    report = run_stage10_component_batch(
        readiness={
            "readiness_contract_version": READINESS_CONTRACT,
            "logical_checksum": "synthetic-readiness",
            "production_plan": list(reversed(jobs)),
        },
        input_inventory=inventory,
        parent_gate_path=tmp_path / "gate.json",
        ledger_path=tmp_path / "ledger.jsonl",
        output_root=tmp_path / "batch",
        campaign_manifest=campaign,
        runner=lambda job, package: {"status": "COMPLETED"},
    )

    assert report["job_count"] == campaign["expected_component_count"] == 75
    assert report["campaign_identity"] == campaign["campaign_identity"]
    assert [row["job_id"] for row in report["jobs"]] == [
        row["job_id"] for row in campaign["fitted_component_matrix"]
    ]


def _protocol():
    return freeze_selector_research_protocol(
        campaign_identity="synthetic-selector-campaign",
        frozen_identities={
            name: {
                "identity": f"synthetic-{name}",
                "checksum": f"checksum-{name}",
            }
            for name in REQUIRED_IDENTITIES
        },
        source_commit="fixture-commit",
    )


def _runtime_job(row, campaign_identity):
    return {
        "job_id": row["job_id"],
        "model_id": row["model_id"],
        "prediction_date": row["prediction_date"],
        "horizon_id": row["horizon_id"],
        "selector_dataset_root": "dataset",
        "authoritative_output_root": "output",
        "feature_schema": row["feature_schema"],
        "target_contract": row["target_contract"],
        "expected_parent_gate_checksum": "gate",
        "expected_dataset_checksum": "dataset",
        "dependency_state": "READY",
        "overwrite_policy": "FAIL_IF_INCOMPATIBLE",
        "resume_policy": "SKIP_COMPATIBLE",
        "logical_checksum": f"job-{row['job_id']}",
        "campaign_identity": campaign_identity,
    }
