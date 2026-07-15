from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.research.ml.stock_level.bounded_selector_runner import BoundedSelectorSettings, run_bounded_selector
from core.research.ml.stock_level.selector_dataset import DETERMINISTIC_SIGNAL_COLUMNS


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"; root.mkdir()
    rows, scores = [], []
    start = date(2024, 1, 1)
    for index in range(14):
        day = (start + timedelta(days=index)).isoformat()
        for symbol_index, symbol in enumerate(("AAA", "BBB")):
            row_id = f"{day}-{symbol}"
            rows.append({"row_id": row_id, "asset_id": symbol, "symbol": symbol, "rebalance_date": day, "decision_session_date": day, "decision_timestamp": f"{day} 20:05:00+00:00", "label_available_timestamp": f"{(start + timedelta(days=index + 2)).isoformat()} 20:05:00+00:00", "selector_eligible": True, "target_status": "realized", "actual_forward_return_10d": 0.01 * (index + symbol_index + 1), "actual_benchmark_return_10d": 0.005})
            score = {"row_id": row_id, "decision_timestamp": f"{day} 20:05:00+00:00"}
            score.update({name: 0.01 * (index + symbol_index + offset + 1) for offset, name in enumerate(DETERMINISTIC_SIGNAL_COLUMNS)})
            scores.append(score)
    pq.write_table(pa.Table.from_pylist(rows), root / "rows.parquet")
    pq.write_table(pa.Table.from_pylist(scores), root / "baseline_scores.parquet")
    (root / "feature_schema.json").write_text(json.dumps({"features": list(DETERMINISTIC_SIGNAL_COLUMNS)}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"dataset_id": "tiny", "source_sha256": "source-a"}), encoding="utf-8")
    return root


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
        predictions = pq.read_table(partition / "predictions.parquet")
        assert manifest["completion_status"] == "complete"
        assert manifest["oos_row_count"] == 2
        assert manifest["training_row_count"] < 28
        assert manifest["training_decision_timestamp_max"] < manifest["label_availability_cutoff"]
        assert manifest["training_label_available_timestamp_max"] <= manifest["label_availability_cutoff"]
        assert len(set(predictions["row_id"].to_pylist())) == 2
        assert set(predictions["decision_session_date"].to_pylist()) == {day}
        assert all(not name.startswith("actual_") for name in result["feature_columns"])
        for candidate in ("predicted_momentum_120d", "predicted_risk_adjusted_momentum", "stock_level_predicted_forward_return_10d_ridge", "stock_level_predicted_forward_return_10d_elastic_net"):
            assert predictions[candidate].null_count == 0


def test_resume_skips_only_valid_complete_date(tmp_path: Path):
    root = _dataset(tmp_path); output = tmp_path / "out"; config = _config(root, output, oos_end_date="2024-01-10")
    first = run_bounded_selector(config); second = run_bounded_selector(config)
    assert first["dates"][0]["status"] == "complete"
    assert second["dates"][0]["status"] == "skipped_complete"


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
    first = _config(root, output, oos_end_date="2024-01-10", model_allowlist=["random_forest"], smoke_overrides={"random_forest_n_estimators": 2})
    second = _config(root, output, oos_end_date="2024-01-10", model_allowlist=["random_forest"], smoke_overrides={"random_forest_n_estimators": 3})
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
