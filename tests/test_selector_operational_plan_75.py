from __future__ import annotations

import copy

import pytest

from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_operational_plan import (
    PLAN_CONTRACT,
    build_selector_operational_plan,
    validate_selector_operational_plan,
)
from core.research.ml.selector_research_campaign import (
    build_selector_research_campaign,
    historical_stage10_baseline_campaign,
)
from core.research.ml.selector_research_protocol import (
    REQUIRED_IDENTITIES,
    freeze_selector_research_protocol,
)


def _protocol():
    return freeze_selector_research_protocol(
        campaign_identity="fixture-campaign",
        frozen_identities={
            name: {"identity": f"id-{name}", "checksum": f"hash-{name}"}
            for name in REQUIRED_IDENTITIES
        },
        source_commit="fixture",
    )


def _parents():
    return {
        "parent_gate_identity": "gate-id",
        "parent_gate_checksum": "gate-hash",
        "selector_dataset_identity": "dataset-id",
        "selector_dataset_checksum": "dataset-hash",
        "symbol_registry_identity": "symbols-id",
        "symbol_registry_checksum": "symbols-hash",
        "daily_spine_identity": "spine-id",
        "daily_spine_checksum": "spine-hash",
    }


def _build(selection, campaign, protocol, **updates):
    values = {
        "campaign": campaign, "protocol": protocol,
        "campaign_selection": selection,
        "parent_identities": _parents(),
        "source_git_commit": "fixture-commit",
    }
    values.update(updates)
    return build_selector_operational_plan(**values)


def test_historical_and_research_plans_derive_exact_campaign_order():
    protocol = _protocol()
    historical = historical_stage10_baseline_campaign()
    research = build_selector_research_campaign(protocol)
    old = _build("historical", historical, protocol)
    new = _build("research", research, protocol)
    assert old["plan_contract_version"] == PLAN_CONTRACT
    assert len(old["jobs"]) == 15
    assert len(new["jobs"]) == 75
    assert [row["job_id"] for row in new["jobs"]] == [
        row["job_id"] for row in research["fitted_component_matrix"]
    ]
    assert len({row["plan_job_identity"] for row in new["jobs"]}) == 75
    assert len({row["component_identity"] for row in new["jobs"]}) == 75
    with pytest.raises(ValueError, match="selection identity"):
        _build("research", historical, protocol)


def test_every_model_receives_explicit_profile_and_runner():
    protocol = _protocol()
    campaign = build_selector_research_campaign(protocol)
    plan = _build("research", campaign, protocol)
    profiles = {
        row["model_id"]: row["operational_input_profile"]
        for row in plan["jobs"]
    }
    assert profiles == {
        "ridge": "ORDINARY_TABULAR",
        "elastic_net": "ORDINARY_TABULAR",
        "ordered_logit_ranker": "ORDINARY_TABULAR",
        "huber": "WAVE4_TABULAR",
        "contextual_elastic_net": "WAVE4_CONTEXTUAL",
        "multi_horizon_ridge": "WAVE4_MULTI_HORIZON",
        "multi_horizon_elastic_net": "WAVE4_MULTI_HORIZON",
        "lightgbm_rank_xendcg": "WAVE4_GROUPED_RANKING",
        "lightgbm_lambdarank": "WAVE4_GROUPED_RANKING",
    }
    expected = {
        row["job_id"]: row["component_runner"]
        for row in campaign["fitted_component_matrix"]
    }
    assert all(
        row["component_runner"] == expected[row["job_id"]]
        for row in plan["jobs"]
    )
    assert all(row["inner_thread_count"] == 1 for row in plan["jobs"])


def test_source_schema_blockers_are_exact_without_changing_membership():
    protocol = _protocol()
    campaign = build_selector_research_campaign(protocol)
    plan = _build("research", campaign, protocol)
    by_model = {}
    for row in plan["jobs"]:
        by_model.setdefault(row["model_id"], row)
        assert "target_value" in row["prediction_prohibited_fields"]
        assert "actual_forward_return_*" in row[
            "prediction_prohibited_fields"
        ]
    assert by_model["huber"]["readiness_status"] == "DEPENDENCY_REQUIRED"
    assert by_model["contextual_elastic_net"]["readiness_status"] == (
        "SOURCE_SCHEMA_REQUIRED"
    )
    assert "SOURCE_CONTRACT_MISSING:point_in_time_market_context" in (
        by_model["contextual_elastic_net"]["blockers"]
    )
    for model in ("multi_horizon_ridge", "multi_horizon_elastic_net"):
        assert set(by_model[model]["required_target_fields"]) >= {
            "actual_forward_return_1d", "actual_forward_return_5d",
            "actual_forward_return_10d", "actual_forward_return_20d",
        }
        assert by_model[model]["horizon_id"] in {
            "return_1s", "return_5s", "return_10s", "return_20s"
        }
    for model in ("lightgbm_rank_xendcg", "lightgbm_lambdarank"):
        assert by_model[model]["readiness_status"] == "SOURCE_SCHEMA_REQUIRED"
        assert by_model[model]["required_grouping_fields"]
        assert by_model[model]["required_relevance_label_fields"]
        assert "relevance_label" in by_model[model][
            "prediction_prohibited_fields"
        ]
    assert len(plan["jobs"]) == 75
    assert plan["readiness_status"] == "PARTIALLY_BLOCKED_INPUT_SCHEMA"


def test_validation_rejects_missing_unexpected_duplicate_and_changed_runner():
    protocol = _protocol()
    campaign = build_selector_research_campaign(protocol)
    plan = _build("research", campaign, protocol)
    for mutation, message in (
        (lambda jobs: jobs.pop(), "cardinality"),
        (
            lambda jobs: jobs.append(copy.deepcopy(jobs[-1])),
            "cardinality",
        ),
        (
            lambda jobs: jobs.__setitem__(1, copy.deepcopy(jobs[0])),
            "inventory/order|Duplicate",
        ),
    ):
        changed = copy.deepcopy(plan)
        changed.pop("publication_result", None)
        mutation(changed["jobs"])
        changed["logical_checksum"] = canonical_hash({
            key: value for key, value in changed.items()
            if key != "logical_checksum"
        })
        with pytest.raises(ValueError, match=message):
            validate_selector_operational_plan(changed, campaign=campaign)

    changed_campaign = copy.deepcopy(campaign)
    changed_campaign["fitted_component_matrix"][0][
        "component_runner"
    ] = "unknown:runner"
    changed_campaign["logical_checksum"] = canonical_hash({
        key: value for key, value in changed_campaign.items()
        if key != "logical_checksum"
    })
    with pytest.raises(ValueError, match="runner disagrees"):
        _build("research", changed_campaign, protocol)


def test_deterministic_atomic_publication_skip_and_incompatible_failure(
    tmp_path,
):
    protocol = _protocol()
    campaign = build_selector_research_campaign(protocol)
    path = tmp_path / "operational-plan.json"
    first = _build(
        "research", campaign, protocol, output_path=path
    )
    second = _build(
        "research", campaign, protocol, output_path=path
    )
    assert first["publication_result"] == "PUBLISHED"
    assert second["publication_result"] == "SKIPPED_COMPATIBLE"
    assert first["logical_checksum"] == second["logical_checksum"]
    with pytest.raises(FileExistsError, match="Incompatible"):
        _build(
            "research", campaign, protocol, output_path=path,
            source_git_commit="different",
        )
    assert first["production_rows_read"] is False
    assert first["packages_published"] is False
    assert first["fitting_performed"] is False
    assert first["evaluation_performed"] is False
