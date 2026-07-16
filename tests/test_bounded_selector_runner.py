from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.stock_level.bounded_selector_runner import BoundedSelectorSettings, PredictionQualityError, _prediction_quality, run_bounded_selector
from core.research.ml.stock_level.selector_dataset import DETERMINISTIC_SIGNAL_COLUMNS
from core.research.ml.stock_level.selector_feature_schema import schema_hash
from core.research.ml.experiment_ledger import read_ledger


def _dataset(tmp_path: Path, *, tree_schema: bool = False) -> Path:
    root = tmp_path / "dataset"; root.mkdir()
    rows, scores = [], []
    tree_contract = (
        json.loads(Path("config/selector_features/canonical_v2_daily_tree_cross_sectional_v1.json").read_text())
        if tree_schema else None
    )
    tree_features = [entry["name"] for entry in tree_contract["features"]] if tree_contract else []
    start = date(2024, 1, 1)
    for index in range(14):
        day = (start + timedelta(days=index)).isoformat()
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
            row_id = f"{day}-{symbol}"
            row = {"row_id": row_id, "asset_id": symbol, "symbol": symbol, "rebalance_date": day, "decision_session_date": day, "decision_timestamp": f"{day} 20:05:00+00:00", "label_available_timestamp": f"{(start + timedelta(days=index + 2)).isoformat()} 20:05:00+00:00", "selector_eligible": True, "target_status": "realized", "actual_forward_return_10d": 0.01 * (symbol_index + 1) + 0.0001 * index, "actual_benchmark_return_10d": 0.005}
            row.update({name: symbol_index + 0.01 * index + 0.0001 * offset for offset, name in enumerate(tree_features) if name not in DETERMINISTIC_SIGNAL_COLUMNS})
            rows.append(row)
            score = {"row_id": row_id, "decision_timestamp": f"{day} 20:05:00+00:00"}
            score.update({name: symbol_index + 0.01 * index + 0.001 * offset for offset, name in enumerate(DETERMINISTIC_SIGNAL_COLUMNS)})
            scores.append(score)
    pq.write_table(pa.Table.from_pylist(rows), root / "rows.parquet")
    pq.write_table(pa.Table.from_pylist(scores), root / "baseline_scores.parquet")
    (root / "feature_schema.json").write_text(json.dumps({"features": list(DETERMINISTIC_SIGNAL_COLUMNS)}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"dataset_id": "tiny", "source_sha256": "source-a"}), encoding="utf-8")
    return root


def _tree_schema_path() -> str:
    return "config/selector_features/canonical_v2_daily_tree_cross_sectional_v1.json"


def _config(root: Path, output: Path, **bounded):
    values = {"dataset_root": str(root), "output_root": str(output), "oos_start_date": "2024-01-10", "oos_end_date": "2024-01-12", "model_allowlist": ["ridge", "elastic_net"], "baseline_allowlist": ["momentum_120d", "risk_adjusted_momentum"], "resume": True, "overwrite_incomplete_dates": True}
    values.update(bounded)
    return {"ml": {"stock_selector_bounded": values}}


def test_bounded_mode_requires_hard_bound(tmp_path: Path):
    with pytest.raises(ValueError, match="requires oos_end_date or max_oos_dates"):
        BoundedSelectorSettings.from_config({"ml": {"stock_selector_bounded": {"dataset_root": str(tmp_path), "output_root": str(tmp_path / "out")}}})


def test_explicit_range_and_max_dates_write_atomic_unique_populations(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "out"
    result = run_bounded_selector(_config(root, output, max_oos_dates=2))
    assert [row["decision_date"] for row in result["dates"]] == ["2024-01-10", "2024-01-11"]
    assert not list(output.glob("*.tmp"))
    for day in ("2024-01-10", "2024-01-11"):
        partition = output / f"date={day}"
        manifest = json.loads((partition / "manifest.json").read_text())
        metrics = json.loads((partition / "metrics.json").read_text())
        predictions = pq.read_table(partition / "predictions.parquet")
        assert manifest["completion_status"] == "complete"
        assert manifest["oos_row_count"] == 6
        assert manifest["training_row_count"] < 84
        assert manifest["training_decision_timestamp_max"] < manifest["label_availability_cutoff"]
        assert manifest["training_label_available_timestamp_max"] <= manifest["label_availability_cutoff"]
        assert len(set(predictions["row_id"].to_pylist())) == 6
        assert set(predictions["decision_session_date"].to_pylist()) == {day}
        assert all(not name.startswith("actual_") for name in result["feature_columns"])
        for candidate in ("predicted_momentum_120d", "predicted_risk_adjusted_momentum", "stock_level_predicted_forward_return_10d_ridge", "stock_level_predicted_forward_return_10d_elastic_net"):
            assert predictions[candidate].null_count == 0
        assert metrics["model_details"]["momentum_120d"]["prediction_quality"]["dispersion_requirement_applied"] is False
        assert metrics["model_details"]["risk_adjusted_momentum"]["prediction_quality"]["coverage"] == 1.0


def test_resume_skips_only_valid_complete_date(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "out"; config = _config(root, output, oos_end_date="2024-01-10")
    first = run_bounded_selector(config); second = run_bounded_selector(config)
    assert first["dates"][0]["status"] == "complete"
    assert second["dates"][0]["status"] == "skipped_complete"
    manifest = json.loads((output / "date=2024-01-10" / "manifest.json").read_text())
    assert manifest["prediction_quality_contract"]["contract_version"] == "fitted_candidate_prediction_quality_v1"


def test_changed_explicit_feature_schema_identity_reruns(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "schema-out"
    paths = []
    for index, rule in enumerate(("strictly prior", "strictly prior and published")):
        payload = {"contract_version": "test_v1", "features": [{"name": name, "data_type": "double", "availability_rule": rule} for name in DETERMINISTIC_SIGNAL_COLUMNS]}
        payload["schema_hash"] = schema_hash(payload)
        path = tmp_path / f"schema-{index}.json"; path.write_text(json.dumps(payload), encoding="utf-8"); paths.append(path)
    first = run_bounded_selector(_config(root, output, oos_end_date="2024-01-10", feature_schema_path=str(paths[0])))
    second = run_bounded_selector(_config(root, output, oos_end_date="2024-01-10", feature_schema_path=str(paths[1])))
    assert first["dates"][0]["status"] == "complete"
    assert second["dates"][0]["status"] == "complete"
    manifest = json.loads((output / "date=2024-01-10" / "manifest.json").read_text())
    assert manifest["selected_feature_schema"]["schema_hash"] == json.loads(paths[1].read_text())["schema_hash"]
    assert manifest["feature_selection_mode"] == "explicit_versioned_schema"
    assert manifest["selected_feature_count"] == len(DETERMINISTIC_SIGNAL_COLUMNS)
    assert manifest["legacy_include_engineered_features_flag"] is False


@pytest.mark.parametrize("damage", ["incomplete", "corrupt", "config_mismatch"])
def test_incomplete_corrupt_or_incompatible_date_is_rerun(tmp_path: Path, damage: str):
    root = _dataset(tmp_path); output = tmp_path / "out"; config = _config(root, output, oos_end_date="2024-01-10")
    run_bounded_selector(config); manifest_path = output / "date=2024-01-10" / "manifest.json"
    if damage == "corrupt": manifest_path.write_text("not-json", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["completion_status" if damage == "incomplete" else "config_hash"] = "bad"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_bounded_selector(config)
    assert result["dates"][0]["status"] == "complete"
    assert json.loads(manifest_path.read_text())["completion_status"] == "complete"


def test_no_broker_or_trading_state_owner_is_imported():
    import inspect
    import core.research.ml.stock_level.bounded_selector_runner as runner
    source = inspect.getsource(runner)
    assert "broker" not in source.lower()
    assert "paper_trading" not in source.lower()


@pytest.mark.parametrize(
    "model,overrides,expected",
    [
        ("random_forest", {"random_forest_n_estimators": 3, "random_forest_max_depth": 2, "random_forest_min_samples_leaf": 2}, {"estimator_count": 3, "max_depth": 2, "min_samples_leaf": 2}),
        ("gradient_boosting", {"gradient_boosting_n_estimators": 3, "gradient_boosting_max_depth": 1, "gradient_boosting_learning_rate": 0.1}, {"estimator_count": 3, "max_depth": 1, "learning_rate": 0.1}),
    ],
)
def test_isolated_tree_smoke_completes_and_reports_parameters(tmp_path: Path, model: str, overrides: dict, expected: dict):
    root = _dataset(tmp_path); output = tmp_path / model
    config = _config(root, output, oos_end_date="2024-01-10", model_allowlist=[model], smoke_overrides=overrides)
    result = run_bounded_selector(config)
    manifest = json.loads((output / "date=2024-01-10" / "manifest.json").read_text())
    metrics = json.loads((output / "date=2024-01-10" / "metrics.json").read_text())
    assert result["dates"][0]["status"] == "complete"
    assert manifest["non_production_smoke"] is True
    assert manifest["smoke_overrides"] == overrides
    assert metrics["model_details"][model]["parameters"] | expected == metrics["model_details"][model]["parameters"]
    assert metrics["model_details"][model]["fit_seconds"] >= 0
    assert metrics["model_details"][model]["prediction_seconds"] >= 0


def test_changed_tree_override_reruns_completed_date(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "tree"
    first = _config(root, output, oos_end_date="2024-01-10", model_allowlist=["random_forest"], smoke_overrides={"random_forest_n_estimators": 2, "random_forest_min_samples_leaf": 2})
    second = _config(root, output, oos_end_date="2024-01-10", model_allowlist=["random_forest"], smoke_overrides={"random_forest_n_estimators": 3, "random_forest_min_samples_leaf": 2})
    run_bounded_selector(first)
    result = run_bounded_selector(second)
    manifest = json.loads((output / "date=2024-01-10" / "manifest.json").read_text())
    assert result["dates"][0]["status"] == "complete"
    assert manifest["smoke_overrides"]["random_forest_n_estimators"] == 3


def test_failed_candidate_never_creates_complete_date(tmp_path: Path, monkeypatch):
    import core.research.ml.stock_level.bounded_selector_runner as runner
    root = _dataset(tmp_path); output = tmp_path / "failed"

    class BrokenModel:
        def get_params(self): return {"randomforestregressor__n_estimators": 1, "randomforestregressor__max_depth": 1, "randomforestregressor__min_samples_leaf": 1, "randomforestregressor__max_features": 1.0, "randomforestregressor__bootstrap": True, "randomforestregressor__random_state": 42, "randomforestregressor__n_jobs": 1}
        def fit(self, x, y): raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_bounded_model", lambda *args: BrokenModel())
    with pytest.raises(KeyboardInterrupt):
        run_bounded_selector(_config(root, output, oos_end_date="2024-01-10", model_allowlist=["random_forest"]))
    assert not (output / "date=2024-01-10" / "manifest.json").exists()


@pytest.mark.parametrize("values", [[0.01, 0.01, 0.01], [0.01, 0.01 + 1e-15, 0.01 + 2e-15]])
def test_constant_and_numerically_constant_predictions_are_rejected(values):
    with pytest.raises(PredictionQualityError, match="below_tolerance"):
        _prediction_quality(values, len(values), require_dispersion=True)


def test_genuinely_varying_small_predictions_are_accepted():
    quality = _prediction_quality([-2e-8, 0.0, 2e-8], 3, require_dispersion=True)
    assert quality["status"] == "accepted"
    assert quality["unique_finite_value_count"] == 3


def test_non_finite_and_scalar_predictions_are_rejected():
    with pytest.raises(PredictionQualityError, match="non_finite_predictions"):
        _prediction_quality([0.0, float("nan")], 2, require_dispersion=True)
    with pytest.raises(PredictionQualityError, match="one_dimensional"):
        _prediction_quality(0.01, 2, require_dispersion=True)


def test_degenerate_candidate_records_failure_without_complete_manifest(tmp_path: Path, monkeypatch):
    import core.research.ml.stock_level.bounded_selector_runner as runner
    root = _dataset(tmp_path); output = tmp_path / "degenerate"

    class ConstantModel:
        def get_params(self): return {}
        def fit(self, x, y): return self
        def predict(self, x): return [0.01] * len(x)

    monkeypatch.setattr(runner, "_bounded_model", lambda *args: ConstantModel())
    with pytest.raises(PredictionQualityError, match="unique_finite_prediction_count_below_two"):
        run_bounded_selector(_config(root, output, oos_end_date="2024-01-10", model_allowlist=["ridge"], baseline_allowlist=[]))
    assert not (output / "date=2024-01-10" / "manifest.json").exists()
    failure = json.loads((output / "date=2024-01-10.model=ridge.failure.json").read_text())
    assert failure["status"] == "rejected"
    assert failure["prediction_quality"]["unique_finite_value_count"] == 1


def test_constant_direct_baseline_is_not_subject_to_dispersion_gate():
    quality = _prediction_quality([0.5, 0.5, 0.5], 3, require_dispersion=False)
    assert quality["status"] == "accepted"
    assert quality["dispersion_requirement_applied"] is False


def test_registry_identity_and_ledger_enter_bounded_manifest(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "registry"; ledger = tmp_path / "ledger.jsonl"
    config = _config(root, output, oos_end_date="2024-01-10", model_allowlist=["rf"], baseline_allowlist=[], smoke_overrides={"random_forest_n_estimators": 2, "random_forest_min_samples_leaf": 2}, experiment_ledger_path=str(ledger))
    result = run_bounded_selector(config)
    assert result["dates"][0]["status"] == "complete"
    manifest = json.loads((output / "date=2024-01-10" / "manifest.json").read_text())
    assert manifest["identity_version"] == "bounded_selector_identity_v3_registry"
    assert manifest["model_registry_entries"][0]["requested_model_id"] == "rf"
    assert manifest["model_registry_entries"][0]["canonical_model_id"] == "random_forest"
    assert manifest["model_registry_entries"][0]["model_entry_hash"]
    assert manifest["registry_set_hash"] and manifest["selector_registry_hash"]
    assert manifest["experiments"][0]["experiment_spec_hash"]
    assert manifest["experiments"][0]["experiment_run_id"]
    assert manifest["component_schema_version"] == "authoritative_selector_component_v1"
    assert manifest["selector_model_identity"] == "random_forest"
    assert manifest["selector_model_version"]
    assert manifest["training_start"] < manifest["training_cutoff"] < manifest["label_availability_cutoff"]
    assert manifest["fold_identity"] and manifest["prediction_population_checksum"]
    assert manifest["prediction_row_count"] == manifest["oos_row_count"]
    assert manifest["publication_status"] == "complete"
    assert manifest["validation_status"] == "VERIFIED_STRICT_OOS"
    assert [row["event_status"] for row in read_ledger(ledger)] == ["STARTED", "COMPLETED"]


def test_registry_rejection_is_retained_in_ledger(tmp_path: Path, monkeypatch):
    import core.research.ml.stock_level.bounded_selector_runner as runner
    root = _dataset(tmp_path); output = tmp_path / "rejected"; ledger = tmp_path / "ledger.jsonl"
    class ConstantModel:
        def get_params(self): return {}
        def fit(self, x, y): return self
        def predict(self, x): return [0.01] * len(x)
    monkeypatch.setattr(runner, "_bounded_model", lambda *args: ConstantModel())
    with pytest.raises(PredictionQualityError):
        run_bounded_selector(_config(root, output, oos_end_date="2024-01-10", model_allowlist=["ridge"], baseline_allowlist=[], experiment_ledger_path=str(ledger)))
    assert [row["event_status"] for row in read_ledger(ledger)] == ["STARTED", "REJECTED"]


def test_ordered_logit_bounded_predictions_are_complete_and_ranked(tmp_path: Path):
    root = _dataset(tmp_path, tree_schema=True); output = tmp_path / "ordered"
    result = run_bounded_selector(_config(
        root, output, oos_end_date="2024-01-10",
        model_allowlist=["ordered_logit_ranker"], baseline_allowlist=[],
        feature_schema_path=_tree_schema_path(),
    ))
    assert result["dates"][0]["status"] == "complete"
    predictions = pq.read_table(output / "date=2024-01-10" / "predictions.parquet")
    probabilities = [
        predictions[f"ordered_logit_probability_{index}"].to_pylist()
        for index in range(5)
    ]
    assert all(sum(values) == pytest.approx(1.0) for values in zip(*probabilities))
    assert predictions["stock_level_predicted_forward_return_10d_ordered_logit_ranker"].null_count == 0
    assert sorted(predictions["ordered_logit_cross_sectional_rank"].to_pylist()) == list(range(1, predictions.num_rows + 1))
    metrics = json.loads((output / "date=2024-01-10" / "metrics.json").read_text())
    details = metrics["model_details"]["ordered_logit_ranker"]
    assert details["ordered_logit_diagnostics"]["training_row_count"] > 0
    assert details["ordered_logit_diagnostics"]["coefficient_values"]
    manifest = json.loads((output / "date=2024-01-10" / "manifest.json").read_text())
    identity = manifest["model_registry_entries"][0]
    assert identity["ranking_problem_contract"] == "daily_cross_sectional_ranking_problem_v1"
    assert identity["relevance_contract"] == "within_date_quintile_relevance_v1"
    assert manifest["target_identity"]["target_contract"] == "forward_return_10d"
    assert manifest["target_identity"]["target_entry_hash"]


def test_ordered_logit_requires_frozen_tree_schema(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "wrong-schema"
    with pytest.raises(ValueError, match="explicit 21-feature"):
        run_bounded_selector(_config(
            root, output, oos_end_date="2024-01-10",
            model_allowlist=["ordered_logit_ranker"], baseline_allowlist=[],
        ))


def test_registry_entry_mismatch_prevents_resume_and_skip_is_ledgered(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "identity"; ledger = tmp_path / "ledger.jsonl"
    config = _config(root, output, oos_end_date="2024-01-10", model_allowlist=["ridge"], baseline_allowlist=[], experiment_ledger_path=str(ledger))
    run_bounded_selector(config)
    manifest_path = output / "date=2024-01-10" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_registry_entries"][0]["model_entry_hash"] = "incompatible"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert run_bounded_selector(config)["dates"][0]["status"] == "complete"
    assert run_bounded_selector(config)["dates"][0]["status"] == "skipped_complete"
    assert read_ledger(ledger)[-1]["event_status"] == "SKIPPED_COMPLETE"


def test_registry_failure_is_retained_in_ledger(tmp_path: Path, monkeypatch):
    import core.research.ml.stock_level.bounded_selector_runner as runner
    root = _dataset(tmp_path); output = tmp_path / "failed-ledger"; ledger = tmp_path / "ledger.jsonl"
    class BrokenModel:
        def get_params(self): return {}
        def fit(self, x, y): raise RuntimeError("synthetic failure")
    monkeypatch.setattr(runner, "_bounded_model", lambda *args: BrokenModel())
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_bounded_selector(_config(root, output, oos_end_date="2024-01-10", model_allowlist=["ridge"], baseline_allowlist=[], experiment_ledger_path=str(ledger)))
    assert [row["event_status"] for row in read_ledger(ledger)] == ["STARTED", "FAILED"]
