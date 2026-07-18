from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from core.research.ml.experiment_ledger import read_ledger
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_component_evaluation import (
    BASE_MODELS, EVALUATION_CONTRACT, evaluate_selector_components,
)


DATES = ("2024-03-15", "2024-09-16")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _readiness(tmp_path, status="READY"):
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({
        "readiness_contract_version": "selector_component_readiness.v2",
        "overall_status": status, "logical_checksum": "READINESS",
        "required_dates": list(DATES), "dataset_identity": "dataset",
        "dataset_checksum": "DATASET",
    }))
    return path


def _component(tmp_path, date, model, mutate=None):
    resolver = RegistryResolver(load_registry_bundle())
    resolution = resolver.resolve("selector_models", model, role="selector")
    target = resolver.resolve("target_contracts", "forward_return_10d", role="selector")
    owner = tmp_path / "components" / f"model={model}" / f"date={date}"
    rows = []
    for index in range(40):
        score = float(index if model != "elastic_net" else 39-index)
        row = {
            "row_id": f"{date}-{index:02d}", "asset_id": f"A{index:02d}",
            "symbol": f"S{index:02d}", "prediction_date": date,
            "model_id": model, "selector_score": score,
            "deterministic_rank": 40-index, "dataset_identity": "dataset",
            "feature_contract_identity": "feature",
            "target_contract_identity": "forward_return_10d",
            "fold_identity": f"fold-{date}",
        }
        if model == "momentum_120d":
            row["predicted_momentum_120d"] = score
        if model == "ordered_logit_ranker":
            probabilities = [0.02, 0.03, 0.05, 0.10, 0.80]
            for cls, value in enumerate(probabilities):
                row[f"ordered_logit_probability_{cls}"] = value
            row["ordered_logit_predicted_relevance_class"] = 4
            row["ordered_logit_expected_relevance"] = score
        rows.append(row)
    prediction = owner / "predictions.csv"; _write_csv(prediction, rows)
    population = canonical_hash([row["row_id"] for row in rows])
    manifest = {
        "component_schema_version": "authoritative_selector_component_v1",
        "selector_model_identity": model,
        "selector_model_version": resolution.entry.entry_hash,
        "prediction_date": date,
        "frozen_selector_dataset_identity": {"dataset_id": "dataset", "dataset_checksum": "DATASET"},
        "feature_contract_version": "feature",
        "target_contract_version": "forward_return_10d",
        "ranking_contract_version": "daily_cross_sectional_ranking_problem_v1" if model == "ordered_logit_ranker" else "ranking_metric_contract_v1",
        "relevance_contract_version": "within_date_quintile_relevance_v1" if model == "ordered_logit_ranker" else None,
        "training_start": "2020-01-01", "training_cutoff": "2024-01-01",
        "training_label_available_timestamp_max": "2024-02-01",
        "fold_identity": f"fold-{date}", "symbol_registry_identity": "registry",
        "daily_stock_spine_identity": "spine", "git_commit": "abc123",
        "prediction_row_count": 40,
        "prediction_population_checksum": population,
        "prediction_artifact_path": str(prediction),
        "prediction_checksum": _sha(prediction),
        "artifact_link": {
            "verification_status": "VERIFIED_STRICT_OOS",
            "artifact_checksum": _sha(prediction),
            "target_contract_hash": target.entry.entry_hash,
            "feature_schema_hash": "FEATURE",
        },
        "publication_status": "complete",
        "validation_status": "VERIFIED_STRICT_OOS",
        "non_production_smoke": False,
        "metrics_path": str(owner / "metrics.json"),
    }
    if mutate: mutate(manifest, rows, prediction)
    manifest["manifest_checksum"] = canonical_hash(manifest)
    path = owner / "manifest.json"; path.write_text(json.dumps(manifest))
    return path


def _outcomes(tmp_path, *, mature=True, benchmark=True):
    rows = []
    for date in DATES:
        for index in range(40):
            rows.append({
                "row_id": f"{date}-{index:02d}", "prediction_date": date,
                "asset_id": f"A{index:02d}",
                "outcome_field": "actual_forward_return_10d",
                "actual_forward_return_10d": float(index),
                "benchmark_return": float(index) / 10 if benchmark else "",
                "target_horizon": "10_sessions",
                "label_available_timestamp": "2025-01-01",
                "outcome_source_identity": "synthetic-outcomes-v1",
                "target_contract": "forward_return_10d",
                "maturity_status": "MATURE" if mature else "IMMATURE",
            })
    path = tmp_path / "outcomes.csv"; _write_csv(path, rows); return path


def _panel(tmp_path, mutate=None):
    paths = []
    for date in DATES:
        for model in BASE_MODELS:
            paths.append(_component(
                tmp_path, date, model,
                mutate if date == DATES[0] and model == "ridge" else None,
            ))
    return paths


def _evaluate(tmp_path, **updates):
    values = {
        "readiness_path": _readiness(tmp_path),
        "output_root": tmp_path / "evaluation",
        "ledger_path": tmp_path / "ledger.jsonl",
        "panel_id": "panel-v1", "required_models": BASE_MODELS,
        "required_dates": DATES, "evaluation_cutoff": "2025-01-02",
    }
    values.update(updates)
    if "component_manifests" not in values:
        values["component_manifests"] = _panel(tmp_path)
    if "outcome_path" not in values:
        values["outcome_path"] = _outcomes(tmp_path)
    return evaluate_selector_components(**values), values


def test_valid_matched_panel_metrics_reports_and_ledger(tmp_path):
    result, values = _evaluate(tmp_path)
    assert result["evaluation_contract_version"] == EVALUATION_CONTRACT
    assert result["evaluation_status"] == "READY"
    assert len(result["per_date_metrics"]) == 8
    ridge = next(row for row in result["per_date_metrics"] if row["model_id"] == "ridge")
    assert ridge["spearman_rank_ic"] == pytest.approx(1.0)
    assert ridge["pearson_ic"] == pytest.approx(1.0)
    assert ridge["ndcg_at_10"] is not None
    assert ridge["top_10_mean_return"] == pytest.approx(34.5)
    assert ridge["top_minus_bottom_10"] == pytest.approx(30.0)
    assert result["aggregate_metrics"]["ridge"]["rank_turnover"] == pytest.approx(0.0)
    assert result["aggregate_metrics"]["ridge"]["top_10_continuity"] == pytest.approx(1.0)
    assert result["ordered_logit"]["probability_valid"] is True
    assert result["ordered_logit"]["class_calibration_inputs"]
    assert [row["event_status"] for row in read_ledger(values["ledger_path"])] == ["STARTED", "COMPLETED"]
    for name in ("evaluation.json", "metrics.csv", "report.md"):
        assert (values["output_root"] / name).exists()


def test_momentum_is_deterministic_and_missing_benchmark_is_allowed(tmp_path):
    first, _ = _evaluate(tmp_path, outcome_path=_outcomes(tmp_path, benchmark=False))
    first_report = (tmp_path / "evaluation" / "evaluation.json").read_text()
    second, _ = _evaluate(tmp_path, outcome_path=_outcomes(tmp_path, benchmark=False),
                          ledger_path=tmp_path / "ledger-2.jsonl")
    momentum = next(row for row in first["per_date_metrics"] if row["model_id"] == "momentum_120d")
    assert momentum["residual_spearman_rank_ic"] is None
    assert first["logical_checksum"] == second["logical_checksum"]
    assert second["publication_result"] == "SKIPPED_COMPLETE"
    assert first_report == (tmp_path / "evaluation" / "evaluation.json").read_text()


def test_evaluation_worker_count_does_not_change_logical_checksum(tmp_path):
    first, values = _evaluate(tmp_path, max_workers=1)
    second, _ = _evaluate(
        tmp_path, max_workers=4, ledger_path=tmp_path / "worker-ledger.jsonl"
    )
    assert first["logical_checksum"] == second["logical_checksum"]
    assert first["execution_metadata"]["outcome_load_count"] == 1


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda manifest, rows, path: manifest["frozen_selector_dataset_identity"].update(dataset_id="wrong"), "UNMATCHED_PANEL"),
        (lambda manifest, rows, path: manifest.update(target_contract_version="wrong"), "COMPONENT_VALIDATION_FAILED"),
        (lambda manifest, rows, path: manifest["artifact_link"].update(verification_status="PENDING"), "COMPONENT_VALIDATION_FAILED"),
        (lambda manifest, rows, path: manifest.update(publication_status="incomplete"), "COMPONENT_VALIDATION_FAILED"),
        (lambda manifest, rows, path: manifest.update(non_production_smoke=True), "COMPONENT_VALIDATION_FAILED"),
    ],
)
def test_invalid_components_block(tmp_path, mutation, reason):
    result, _ = _evaluate(tmp_path, component_manifests=_panel(tmp_path, mutation))
    assert result["evaluation_status"] == "BLOCKED"
    assert reason in result["blockers"]
    if reason == "COMPONENT_VALIDATION_FAILED":
        assert result["rejected_components"]


def test_population_duplicate_nonfinite_and_probability_failures(tmp_path):
    def duplicate(manifest, rows, path):
        rows[1]["row_id"] = rows[0]["row_id"]; _write_csv(path, rows)
        manifest["prediction_checksum"] = _sha(path); manifest["artifact_link"]["artifact_checksum"] = _sha(path)
    result, _ = _evaluate(tmp_path, component_manifests=_panel(tmp_path, duplicate))
    assert result["evaluation_status"] == "BLOCKED"

    def nonfinite(manifest, rows, path):
        rows[0]["selector_score"] = "nan"; _write_csv(path, rows)
        manifest["prediction_checksum"] = _sha(path); manifest["artifact_link"]["artifact_checksum"] = _sha(path)
    result, _ = _evaluate(tmp_path, component_manifests=_panel(tmp_path, nonfinite),
                          replacement_policy="replace_incompatible")
    assert result["evaluation_status"] == "BLOCKED"

    def bad_probability(manifest, rows, path):
        # Applied to Ridge by _panel; force it to present as ordered logit to exercise validation.
        manifest["selector_model_identity"] = "ordered_logit_ranker"
    result, _ = _evaluate(tmp_path, component_manifests=_panel(tmp_path, bad_probability),
                          replacement_policy="replace_incompatible")
    assert result["evaluation_status"] == "BLOCKED"


def test_immature_outcomes_are_rejected_and_ledger_records_rejection(tmp_path):
    result, values = _evaluate(tmp_path, outcome_path=_outcomes(tmp_path, mature=False))
    assert result["evaluation_status"] == "BLOCKED"
    assert "IMMATURE_OUTCOME" in result["blockers"]
    assert [row["event_status"] for row in read_ledger(values["ledger_path"])] == ["STARTED", "REJECTED"]


def test_outcome_cutoff_target_and_population_fail_closed(tmp_path):
    result, _ = _evaluate(tmp_path, evaluation_cutoff="2024-12-31")
    assert "IMMATURE_OUTCOME" in result["blockers"]
    rows = list(csv.DictReader(_outcomes(tmp_path).open()))
    rows[0]["target_contract"] = "future_volatility"
    path = tmp_path / "wrong-target.csv"; _write_csv(path, rows)
    assert "IMMATURE_OUTCOME" in _evaluate(
        tmp_path, outcome_path=path, replacement_policy="replace_incompatible"
    )[0]["blockers"]
    rows = rows[1:]; path = tmp_path / "missing-outcome.csv"; _write_csv(path, rows)
    assert "OUTCOME_POPULATION_MISMATCH" in _evaluate(
        tmp_path, outcome_path=path, replacement_policy="replace_incompatible"
    )[0]["blockers"]


def test_incompatible_existing_evaluation_fails_closed(tmp_path):
    result, values = _evaluate(tmp_path)
    path = values["output_root"] / "evaluation.json"
    payload = json.loads(path.read_text()); payload["logical_checksum"] = "wrong"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Incompatible existing"):
        evaluate_selector_components(**values)
    assert read_ledger(values["ledger_path"])[-1]["event_status"] == "REJECTED"


def test_invalid_readiness_and_unexpected_failure_are_ledgered(tmp_path, monkeypatch):
    ledger = tmp_path / "rejected.jsonl"
    with pytest.raises(ValueError, match="not READY"):
        evaluate_selector_components(
            readiness_path=_readiness(tmp_path, "BLOCKED"),
            component_manifests=[], outcome_path=_outcomes(tmp_path),
            output_root=tmp_path / "out", ledger_path=ledger,
            panel_id="x", required_dates=DATES, evaluation_cutoff="2025-01-02",
        )
    assert read_ledger(ledger)[-1]["event_status"] == "REJECTED"

    import core.research.ml.selector_component_evaluation as evaluation
    monkeypatch.setattr(evaluation, "_load_components", lambda paths: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        evaluate_selector_components(
            readiness_path=_readiness(tmp_path),
            component_manifests=[], outcome_path=_outcomes(tmp_path),
            output_root=tmp_path / "out2", ledger_path=tmp_path / "failed.jsonl",
            panel_id="x", required_dates=DATES, evaluation_cutoff="2025-01-02",
        )
    assert read_ledger(tmp_path / "failed.jsonl")[-1]["event_status"] == "FAILED"


def test_no_factory_execution_publication_replay_exposure_or_pyarrow_import(monkeypatch, tmp_path):
    real_import = __import__
    def guarded(name, *args, **kwargs):
        forbidden = ("benchmark_execution", "benchmark_models", "ordinary_selector_publication", "portfolio_replay", "policy_sweep", "exposure", "pyarrow")
        if any(value in name for value in forbidden): raise AssertionError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", guarded)
    assert _evaluate(tmp_path)[0]["evaluation_status"] == "READY"
