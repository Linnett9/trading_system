from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.registries.io import canonical_hash
from core.research.ml.experiment_ledger import (
    register_selector_plan, transition_selector_experiment,
)
from core.research.ml.wave4_gate_campaign import DEFAULT_THRESHOLDS, evaluate_wave4_campaign

DATES = ("2024-03-15", "2024-09-16", "2025-03-17", "2025-09-15", "2026-03-16")
MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path):
    components = []
    for date in DATES:
        for model in MODELS:
            components.append({
                "component_id": f"{model}-{date}", "experiment_id": f"experiment-{model}-{date}",
                "campaign_id": "wave4", "model_id": model, "decision_date": date,
                "dataset_id": "dataset", "dataset_checksum": "dataset-checksum",
                "daily_spine_id": "spine", "symbol_registry_id": "registry",
                "feature_schema_hash": f"feature-{model}", "target_contract_id": "forward_return_10d",
                "target_contract_hash": "target-hash",
                "target_provenance_contract_version": "stock_level_target_provenance_v2",
                "ranking_contract_id": "daily_cross_sectional_ranking_problem_v1",
                "fold_id": f"fold-{date}", "source_commit": "abc123",
                "purge_sessions": 10, "embargo_sessions": 10,
                "maximum_label_available_timestamp": f"{date}T00:00:00Z",
                "hyperparameters": {}, "random_seed": 42,
                "training_start": "2020-01-01", "training_end": "2023-01-01",
                "planned_output_root": f"components/{model}/{date}",
            })
    plan = {
        "plan_contract_version": "selector_operational_component_plan.v1",
        "campaign_id": "wave4", "component_count": 15, "fitted_models": list(MODELS),
        "decision_dates": list(DATES), "dataset_id": "dataset",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "source_commit": "abc123", "components": components,
    }
    plan["logical_checksum"] = canonical_hash(plan)
    plan_path = tmp_path / "component_plan.json"; _write_json(plan_path, plan)
    ledger_path = tmp_path / "selector_experiment_ledger.json"
    register_selector_plan(ledger_path, plan)
    for component in components:
        transition_selector_experiment(
            ledger_path, experiment_id=component["experiment_id"],
            to_status="RUNNING", component=component,
        )
        transition_selector_experiment(
            ledger_path, experiment_id=component["experiment_id"],
            to_status="SUCCEEDED", component=component,
            component_manifest_path=f"{component['component_id']}/manifest.json",
        )
    manifests = []
    for component in components:
        owner = tmp_path / "components" / component["component_id"]
        owner.mkdir(parents=True)
        predictions = owner / "predictions.csv"
        rows = []
        for index in range(25):
            rows.append({
                "row_id": f"row-{index}", "asset_id": f"asset-{index:02}",
                "prediction_date": component["decision_date"],
                "selector_score": index + (0.01 if component["model_id"] == "elastic_net" else 0),
                "actual_forward_return_10d": index / 100,
                "market_return_10d": 0.01, "population_size": 25,
            })
        with predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        import hashlib
        checksum = hashlib.sha256(predictions.read_bytes()).hexdigest().upper()
        manifest = {
            "campaign_id": "wave4", "component_id": component["component_id"],
            "experiment_id": component["experiment_id"],
            "selector_model_identity": component["model_id"], "prediction_date": component["decision_date"],
            "frozen_selector_dataset_identity": {"dataset_id": "dataset", "dataset_checksum": "dataset-checksum"},
            "daily_stock_spine_identity": "spine", "symbol_registry_identity": "registry",
            "feature_schema_hash": component["feature_schema_hash"],
            "target_contract_version": "forward_return_10d", "target_contract_hash": "target-hash",
            "target_provenance_contract_version": "stock_level_target_provenance_v2",
            "ranking_contract_version": "daily_cross_sectional_ranking_problem_v1",
            "fold_identity": component["fold_id"], "git_commit": "abc123",
            "publication_status": "complete", "validation_status": "VERIFIED_STRICT_OOS",
            "prediction_artifact_path": str(predictions), "prediction_checksum": checksum,
            "prediction_row_count": len(rows),
        }
        manifest["manifest_checksum"] = canonical_hash(manifest)
        path = owner / "manifest.json"; _write_json(path, manifest); manifests.append(path)
    return plan_path, manifests


def _run(tmp_path, manifests=None, thresholds=None):
    plan, default = _fixture(tmp_path)
    return evaluate_wave4_campaign(
        component_plan_path=plan, component_manifest_paths=default if manifests is None else manifests,
        output_root=tmp_path / "report", thresholds=thresholds,
    ), default


def _mutate(path, field, value):
    payload = json.loads(path.read_text()); payload[field] = value
    payload["manifest_checksum"] = canonical_hash({k: v for k, v in payload.items() if k != "manifest_checksum"})
    _write_json(path, payload)


def test_exactly_15_valid_components_are_ready_and_outputs_are_atomic(tmp_path):
    report, _ = _run(tmp_path)
    assert report["primary_status"] == "READY_FOR_PORTFOLIO_REPLAY"
    assert report["effective_material_trial_count"] == 15
    assert set(path.name for path in (tmp_path / "report").iterdir()) == {
        "wave4_campaign_report.json", "wave4_campaign_report.md",
    }
    required = {"spearman_rank_ic", "pearson_ic", "market_residual_rank_ic", "ndcg_at_10",
                "ndcg_at_20", "top_10_realised_return", "top_20_realised_return",
                "top_minus_bottom_spread", "rank_turnover", "top_10_continuity",
                "top_20_continuity", "prediction_coverage", "largest_tied_score_group",
                "score_dispersion", "coefficient_stability"}
    assert required <= set(report["per_date_metrics"][0])


def test_14_components_block_incomplete_and_failed_date_is_visible(tmp_path):
    plan, manifests = _fixture(tmp_path)
    report = evaluate_wave4_campaign(component_plan_path=plan, component_manifest_paths=manifests[:-1],
                                     output_root=tmp_path / "report")
    assert report["primary_status"] == "BLOCKED_INCOMPLETE"
    assert any("MISSING_COMPONENT" in reason for reason in report["failure_blocker_reasons"])


def test_extra_and_duplicate_components_block(tmp_path):
    plan, manifests = _fixture(tmp_path)
    report = evaluate_wave4_campaign(component_plan_path=plan,
        component_manifest_paths=[*manifests, manifests[0]], output_root=tmp_path / "report")
    assert report["primary_status"] == "BLOCKED_INCOMPLETE"
    assert "EXTRA_COMPONENT_CLAIMS_CAMPAIGN" in report["failure_blocker_reasons"]
    assert "DUPLICATE_MODEL_DATE_COMPONENT" in report["failure_blocker_reasons"]


@pytest.mark.parametrize("field,value,reason", [
    ("publication_status", "failed", "COMPONENT_INCOMPLETE"),
    ("validation_status", None, "STRICT_OOS_VERIFICATION_ABSENT"),
    ("target_provenance_contract_version", "stock_level_target_provenance_v1", "TARGET_PROVENANCE_V2_REQUIRED"),
    ("daily_stock_spine_identity", "other", "DAILY_SPINE_IDENTITY_MISMATCH"),
    ("symbol_registry_identity", "other", "SYMBOL_REGISTRY_IDENTITY_MISMATCH"),
    ("feature_schema_hash", "other", "FEATURE_SCHEMA_IDENTITY_MISMATCH"),
    ("target_contract_version", "other", "TARGET_CONTRACT_IDENTITY_MISMATCH"),
    ("ranking_contract_version", "other", "RANKING_CONTRACT_IDENTITY_MISMATCH"),
    ("fold_identity", "other", "FOLD_IDENTITY_MISMATCH"),
    ("git_commit", "other", "SOURCE_COMMIT_INCOMPATIBLE"),
    ("experiment_id", None, "EXPERIMENT_ID_MISSING_OR_MISMATCH"),
])
def test_manifest_contracts_fail_closed(tmp_path, field, value, reason):
    plan, manifests = _fixture(tmp_path); _mutate(manifests[0], field, value)
    report = evaluate_wave4_campaign(component_plan_path=plan, component_manifest_paths=manifests,
                                     output_root=tmp_path / "report")
    assert reason in report["failure_blocker_reasons"]
    expected_status = "BLOCKED_INCOMPLETE" if reason == "COMPONENT_INCOMPLETE" else "BLOCKED_LINEAGE"
    assert report["primary_status"] == expected_status


def test_missing_prediction_evidence_blocks(tmp_path):
    plan, manifests = _fixture(tmp_path)
    payload = json.loads(manifests[0].read_text()); Path(payload["prediction_artifact_path"]).unlink()
    report = evaluate_wave4_campaign(component_plan_path=plan, component_manifest_paths=manifests,
                                     output_root=tmp_path / "report")
    assert "PREDICTION_EVIDENCE_ABSENT" in report["failure_blocker_reasons"]


@pytest.mark.parametrize("column,value,reason", [
    ("selector_score", "nan", "NONFINITE_SCORES"),
    ("asset_id", "asset-01", "DUPLICATE_ECONOMIC_ROWS"),
])
def test_unsafe_prediction_rows_block_metrics(tmp_path, column, value, reason):
    plan, manifests = _fixture(tmp_path)
    payload = json.loads(manifests[0].read_text()); artifact = Path(payload["prediction_artifact_path"])
    rows = list(csv.DictReader(artifact.open()))
    if column == "asset_id": rows[0][column] = rows[1][column]
    else: rows[0][column] = value
    with artifact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import hashlib
    payload["prediction_checksum"] = hashlib.sha256(artifact.read_bytes()).hexdigest().upper()
    payload["manifest_checksum"] = canonical_hash({k: v for k, v in payload.items() if k != "manifest_checksum"})
    _write_json(manifests[0], payload)
    report = evaluate_wave4_campaign(component_plan_path=plan, component_manifest_paths=manifests,
                                     output_root=tmp_path / "report")
    assert report["primary_status"] == "BLOCKED_METRICS"
    assert reason in report["failure_blocker_reasons"]


def test_physical_order_is_irrelevant_and_repeat_is_deterministic(tmp_path):
    first, manifests = _run(tmp_path)
    payload = json.loads(manifests[0].read_text()); artifact = Path(payload["prediction_artifact_path"])
    rows = list(csv.DictReader(artifact.open())); rows.reverse()
    with artifact.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import hashlib
    payload["prediction_checksum"] = hashlib.sha256(artifact.read_bytes()).hexdigest().upper()
    payload["manifest_checksum"] = canonical_hash({k: v for k, v in payload.items() if k != "manifest_checksum"})
    _write_json(manifests[0], payload)
    second = evaluate_wave4_campaign(component_plan_path=tmp_path/"component_plan.json",
        component_manifest_paths=manifests, output_root=tmp_path/"report2")
    assert first["per_date_metrics"] == second["per_date_metrics"]
    repeated = evaluate_wave4_campaign(component_plan_path=tmp_path/"component_plan.json",
        component_manifest_paths=manifests, output_root=tmp_path/"report3")
    assert second["campaign_checksum"] == repeated["campaign_checksum"]


def test_threshold_and_component_changes_change_campaign_identity(tmp_path):
    first, manifests = _run(tmp_path)
    changed = dict(DEFAULT_THRESHOLDS); changed["minimum_mean_rank_ic"] = 0.5
    second = evaluate_wave4_campaign(component_plan_path=tmp_path/"component_plan.json",
        component_manifest_paths=manifests, output_root=tmp_path/"report2", thresholds=changed)
    assert first["campaign_checksum"] != second["campaign_checksum"]
    _mutate(manifests[0], "coefficient_checksum", "changed")
    third = evaluate_wave4_campaign(component_plan_path=tmp_path/"component_plan.json",
        component_manifest_paths=manifests, output_root=tmp_path/"report3", thresholds=changed)
    assert second["campaign_checksum"] != third["campaign_checksum"]


def test_missing_momentum_is_explicit_and_rejected_models_remain_visible(tmp_path):
    thresholds = dict(DEFAULT_THRESHOLDS); thresholds["minimum_mean_rank_ic"] = 2.0
    report, _ = _run(tmp_path, thresholds=thresholds)
    assert report["primary_status"] == "REJECTED"
    assert report["models_rejected"] == list(MODELS)
    assert report["momentum_control_comparison"] == {
        "available": False, "reason": "OPTIONAL_MOMENTUM_EVIDENCE_NOT_PROVIDED"
    }


def test_failed_construction_preserves_previous_report_and_path_has_no_forbidden_imports(tmp_path):
    _run(tmp_path)
    before = (tmp_path/"report"/"wave4_campaign_report.json").read_text()
    with pytest.raises(ValueError):
        evaluate_wave4_campaign(component_plan_path=tmp_path/"component_plan.json",
            component_manifest_paths=[], output_root=tmp_path/"report", thresholds={"unknown": 1})
    assert (tmp_path/"report"/"wave4_campaign_report.json").read_text() == before
    source = Path("core/research/ml/wave4_gate_campaign.py").read_text().lower()
    cli = Path("scripts/evaluate_wave4_selector_campaign.py").read_text().lower()
    for forbidden in ("model_factory", "ordinary_selector_publication", "portfolio_replay",
                      "exposure", "news", "five_minute", "pyarrow", "subprocess"):
        assert f"import {forbidden}" not in source + cli
