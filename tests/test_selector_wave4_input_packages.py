from __future__ import annotations

import hashlib
import json
from pathlib import Path

from application.services.selector_evaluation_commands import (
    _validate_operational_package,
    adapt_operational_package_to_wave4,
)
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_campaign_launch_gate import (
    build_selector_campaign_launch_readiness,
)
from core.research.ml.selector_component_rows import (
    prediction_row,
    training_row,
)
from core.research.ml.selector_operational_plan import (
    build_selector_operational_plan,
)
from core.research.ml.selector_research_campaign import (
    build_selector_research_campaign,
)
from core.research.ml.selector_research_protocol import (
    REQUIRED_IDENTITIES,
    freeze_selector_research_protocol,
)
from core.research.ml.selector_wave4_input_packages import (
    publish_selector_operational_packages_v2,
    source_schema_guarantee_manifest,
    validate_v2_package,
)


def _protocol():
    return freeze_selector_research_protocol(
        campaign_identity="fixture",
        frozen_identities={
            name: {"identity": f"id-{name}", "checksum": f"sum-{name}"}
            for name in REQUIRED_IDENTITIES
        },
        source_commit="fixture",
    )


def _parents():
    return {
        "parent_gate_identity": "gate", "parent_gate_checksum": "gate-sum",
        "selector_dataset_identity": "dataset",
        "selector_dataset_checksum": "dataset-sum",
        "symbol_registry_identity": "symbols",
        "symbol_registry_checksum": "symbols-sum",
        "daily_spine_identity": "spine",
        "daily_spine_checksum": "spine-sum",
    }


def _realm():
    protocol = _protocol()
    campaign = build_selector_research_campaign(protocol)
    plan = build_selector_operational_plan(
        campaign=campaign, protocol=protocol,
        campaign_selection="research", parent_identities=_parents(),
        source_git_commit="fixture",
    )
    return protocol, campaign, plan


def _guarantees(plan, *, complete=True, remove=()):
    values = {
        value for job in plan["jobs"]
        for value in job["required_source_guarantees"]
    } if complete else {
        "selector_row_identity", "symbol_identity", "decision_date",
        "label_availability_timestamp", "forward_return_10d",
        "ordinary_registered_feature_order",
        "decision_date_group_ownership",
    }
    values -= set(remove)
    return source_schema_guarantee_manifest(
        dataset_identity="dataset", schema_checksum="schema",
        guarantees=sorted(values),
        field_identities={value: f"field:{value}" for value in values},
        source_commit="fixture",
    )


def _job(plan, model, horizon=None):
    return next(
        row for row in plan["jobs"]
        if row["model_id"] == model and row.get("horizon_id") == horizon
    )


def _rows(job):
    feature_ids = ["signal"]
    common = {
        "dataset_identity": "dataset",
        "campaign_identity": job["campaign_identity"],
        "plan_job_identity": job["plan_job_identity"],
        "model_id": job["model_id"], "symbol_identity": "A",
        "target_horizon": job.get("horizon_id") or "return_10s",
        "fold_identity": "fold", "feature_schema_identity": "schema",
        "feature_order_checksum": canonical_hash(feature_ids),
        "ordered_feature_ids": feature_ids,
        "ordered_feature_values": [1.0],
        "feature_availability_timestamp": "2024-01-01T00:00:00Z",
    }
    training = training_row({
        **common, "decision_date": "2024-01-01T00:00:00Z",
        "prediction_date": job["prediction_date"],
        "dataset_row_identity": "train-A",
        "training_boundary_identity": "2024-02-01T00:00:00Z",
        "purge_sessions": 10, "embargo_sessions": 10,
        "target_contract": job["target_contract"],
        "target_availability_timestamp": "2024-01-10T00:00:00Z",
        "target_maturity_timestamp": "2024-01-10T00:00:00Z",
        "target_value": 0.2,
    })
    prediction = prediction_row({
        **common, "decision_date": job["prediction_date"],
        "prediction_date": job["prediction_date"],
        "dataset_row_identity": "predict-A",
        "feature_availability_timestamp": job["prediction_date"],
    })
    profile = job["operational_input_profile"]
    if profile == "WAVE4_CONTEXTUAL":
        extra = {
            "stock_feature_ids": ["momentum"],
            "stock_feature_values": [1.0],
            "market_context_ids": ["market_volatility"],
            "market_context_values": [0.5],
        }
        training.update(extra); prediction.update(extra)
        training = _rehash_training(training)
        prediction = _rehash_prediction(prediction)
    elif profile == "WAVE4_MULTI_HORIZON":
        horizons = ("return_1s", "return_5s", "return_10s", "return_20s")
        training.update(
            horizon_target_values={value: 0.1 for value in horizons},
            horizon_target_maturity_timestamps={
                value: "2024-01-10T00:00:00Z" for value in horizons
            },
            horizon_target_availability_states={
                value: "MATURE" for value in horizons
            },
        )
        training = _rehash_training(training)
    elif profile == "WAVE4_GROUPED_RANKING":
        training["relevance_label"] = 1
        training = _rehash_training(training)
    return {"training_rows": [training], "prediction_rows": [prediction]}


def _publish(tmp_path, plan, campaign, guarantees, rows):
    return publish_selector_operational_packages_v2(
        plan=plan, campaign=campaign, source_guarantees=guarantees,
        parent_identities=_parents(), rows_by_job=rows,
        output_root=tmp_path,
    )


def test_complete_inventory_blocks_missing_schema_without_dropping_jobs(
    tmp_path,
):
    protocol, campaign, plan = _realm()
    result = _publish(tmp_path, plan, campaign, _guarantees(plan, complete=False), {})
    assert result["expected_jobs"] == 75
    assert len(result["per_job_results"]) == 75
    contextual = next(
        row for row in result["per_job_results"]
        if "contextual_elastic_net" in row["job_id"]
    )
    assert contextual["status"] == "BLOCKED_SOURCE_SCHEMA"
    gate = build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=campaign, campaign_selection="research",
        package_publication=result, source_commit="fixture",
    )
    assert gate["readiness_status"] == "BLOCKED_INPUTS"


def test_huber_and_ordinary_packages_publish_atomically_and_skip(tmp_path):
    _, campaign, plan = _realm()
    guarantees = _guarantees(plan)
    huber = _job(plan, "huber")
    ridge = _job(plan, "ridge")
    supplied = {huber["job_id"]: _rows(huber), ridge["job_id"]: _rows(ridge)}
    first = _publish(tmp_path, plan, campaign, guarantees, supplied)
    by_id = {row["job_id"]: row for row in first["per_job_results"]}
    assert by_id[huber["job_id"]]["status"] == "PACKAGE_PUBLISHED"
    assert by_id[ridge["job_id"]]["status"] == "PACKAGE_PUBLISHED"
    huber_manifest = validate_v2_package(
        Path(by_id[huber["job_id"]]["manifest_path"])
    )
    ridge_manifest = validate_v2_package(
        Path(by_id[ridge["job_id"]]["manifest_path"])
    )
    assert huber_manifest["wave4_fit_input_applicable"] is True
    assert ridge_manifest["wave4_fit_input_applicable"] is False
    fit_path = Path(huber_manifest["wave4_fit_input_path"])
    assert hashlib.sha256(fit_path.read_bytes()).hexdigest().upper() == (
        huber_manifest["wave4_fit_input_artifact_sha256"]
    )
    _validate_operational_package(
        huber_manifest,
        {
            "job_id": huber["job_id"], "model_id": "huber",
            "prediction_date": huber["prediction_date"],
        },
        {"selector_dataset_artifact_checksum": "dataset-sum"},
    )
    adapted = adapt_operational_package_to_wave4(
        model_id="huber", job=huber, package=huber_manifest,
        training_rows=[], prediction_rows=[],
    )
    assert adapted["fit_input"]["operational_input_identity"] == (
        huber_manifest["package_id"]
    )
    second = _publish(tmp_path, plan, campaign, guarantees, supplied)
    by_id = {row["job_id"]: row for row in second["per_job_results"]}
    assert by_id[huber["job_id"]]["status"] == "SKIPPED_COMPATIBLE"


def test_contextual_multi_horizon_and_rankers_use_existing_builders(tmp_path):
    _, campaign, plan = _realm()
    guarantees = _guarantees(plan)
    selected = [
        _job(plan, "contextual_elastic_net"),
        _job(plan, "multi_horizon_ridge", "return_5s"),
        _job(plan, "multi_horizon_elastic_net", "return_20s"),
        _job(plan, "lightgbm_rank_xendcg"),
        _job(plan, "lightgbm_lambdarank"),
    ]
    rows = {job["job_id"]: _rows(job) for job in selected}
    rows[selected[0]["job_id"]].update(
        stock_feature_schema_identity="stock-schema",
        market_context_schema_identity="context-schema",
        interactions=[{
            "interaction_id": "momentum_x_market_volatility",
            "stock_feature_id": "momentum",
            "market_context_id": "market_volatility",
            "transformation": "scaled_product",
        }],
    )
    result = _publish(tmp_path, plan, campaign, guarantees, rows)
    by_id = {row["job_id"]: row for row in result["per_job_results"]}
    for job in selected:
        assert by_id[job["job_id"]]["status"] == "PACKAGE_PUBLISHED"
        manifest = validate_v2_package(
            Path(by_id[job["job_id"]]["manifest_path"])
        )
        fit = json.loads(Path(manifest["wave4_fit_input_path"]).read_text())
        assert fit["model_id"] == job["model_id"]
        assert fit["horizon_id"] == job.get("horizon_id")
        assert fit["fitting_performed"] is False
        if job["model_id"] == "lightgbm_lambdarank":
            assert fit["label_gain_policy"] == (
                "exponential_gain_quintile_0_4_v1"
            )


def test_missing_horizon_guarantee_and_tampering_fail_closed(tmp_path):
    _, campaign, plan = _realm()
    multi = _job(plan, "multi_horizon_ridge", "return_1s")
    blocked = _publish(
        tmp_path, plan, campaign,
        _guarantees(plan, remove={"forward_return_20d"}),
        {multi["job_id"]: _rows(multi)},
    )
    row = next(
        value for value in blocked["per_job_results"]
        if value["job_id"] == multi["job_id"]
    )
    assert row["status"] == "BLOCKED_SOURCE_SCHEMA"

    huber = _job(plan, "huber")
    supplied = {huber["job_id"]: _rows(huber)}
    published = _publish(
        tmp_path / "tamper", plan, campaign, _guarantees(plan), supplied
    )
    evidence = next(
        value for value in published["per_job_results"]
        if value["job_id"] == huber["job_id"]
    )
    manifest = json.loads(Path(evidence["manifest_path"]).read_text())
    Path(manifest["wave4_fit_input_path"]).write_text("{}")
    repeated = _publish(
        tmp_path / "tamper", plan, campaign, _guarantees(plan), supplied
    )
    evidence = next(
        value for value in repeated["per_job_results"]
        if value["job_id"] == huber["job_id"]
    )
    assert evidence["status"] == "INVALID_INPUT"
    assert "tampered" in evidence["blockers"][0]


def test_launch_gate_requires_complete_exact_package_inventory(tmp_path):
    protocol, campaign, plan = _realm()
    rows = {job["job_id"]: _rows(job) for job in plan["jobs"]}
    for job in plan["jobs"]:
        if job["operational_input_profile"] == "WAVE4_CONTEXTUAL":
            rows[job["job_id"]].update(
                stock_feature_schema_identity="stock-schema",
                market_context_schema_identity="context-schema",
                interactions=[{
                    "interaction_id": "momentum_x_market_volatility",
                    "stock_feature_id": "momentum",
                    "market_context_id": "market_volatility",
                    "transformation": "scaled_product",
                }],
            )
    complete = _publish(
        tmp_path, plan, campaign, _guarantees(plan), rows
    )
    assert complete["all_packages_complete"] is True
    assert build_selector_campaign_launch_readiness(
        protocol=protocol, campaign=campaign, campaign_selection="research",
        package_publication=complete, source_commit="fixture",
    )["readiness_status"] == "READY_TO_LAUNCH"


def _rehash_training(value):
    payload = {
        key: item for key, item in value.items()
        if key not in {"logical_row_checksum", "contract_version", "role"}
    }
    return training_row(payload)


def _rehash_prediction(value):
    payload = {
        key: item for key, item in value.items()
        if key not in {"logical_row_checksum", "contract_version", "role"}
    }
    return prediction_row(payload)
