from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.immutable_runs import deterministic_run_id, preserve_immutable_run
from core.research.ml.stock_level.final_fitted_selector import (
    ARTIFACT_SCHEMA_VERSION,
    assert_final_fit_reuse_compatible,
    load_final_fitted_stock_selector,
    write_final_fitted_stock_selector,
)
from core.research.ml.stock_level_benchmark_types import (
    FEATURE_COLUMNS,
    PREDICTION_PREFIX,
    TABULAR_MODEL_NAMES,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
)


def test_final_fit_uses_oos_selected_winner_not_in_sample_fit(tmp_path):
    config = _write_selector_inputs(tmp_path, winner="ridge")
    result = write_final_fitted_stock_selector(config)

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["selected_model_name"] == "ridge"
    assert float(audit["source_oos_evidence"]["leaderboard_row"]["mean_spearman_ic"]) == 0.10

    _rewrite_source_targets_to_favour_other_model(tmp_path)
    second = write_final_fitted_stock_selector(config)
    second_audit = json.loads(second.audit_path.read_text(encoding="utf-8"))
    assert second_audit["selected_model_name"] == "ridge"


def test_final_holdout_outcomes_do_not_change_winner(tmp_path):
    config = _write_selector_inputs(tmp_path, winner="elastic_net")
    _rewrite_final_rows(tmp_path, target_value=99.0)

    result = write_final_fitted_stock_selector(config)

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["selected_model_name"] == "elastic_net"
    assert audit["source_oos_evidence"]["identity"]["selected_model_name"] == "elastic_net"


def test_final_fit_excludes_immature_labels_and_later_cutoff_matures_rows(tmp_path):
    early = _write_selector_inputs(tmp_path / "early", winner="ridge", cutoff="2024-01-07")
    early_result = write_final_fitted_stock_selector(early)
    early_audit = json.loads(early_result.audit_path.read_text(encoding="utf-8"))

    late = _write_selector_inputs(tmp_path / "late", winner="ridge", cutoff="2024-01-12")
    late_result = write_final_fitted_stock_selector(late)
    late_audit = json.loads(late_result.audit_path.read_text(encoding="utf-8"))

    assert early_audit["excluded_immature_label_count"] > 0
    assert late_audit["eligible_training_row_count"] > early_audit["eligible_training_row_count"]


def test_final_fit_persists_features_excludes_timing_and_reloads(tmp_path):
    config = _write_selector_inputs(tmp_path, winner="random_forest")
    result = write_final_fitted_stock_selector(config)
    loaded = load_final_fitted_stock_selector(result.output_dir)
    contract = loaded.feature_contract

    assert tuple(contract["ordered_feature_columns"]) == FEATURE_COLUMNS
    assert not set(TARGET_PROVENANCE_COLUMNS) & set(contract["ordered_feature_columns"])
    assert contract["feature_count"] == len(FEATURE_COLUMNS)
    assert json.loads(result.audit_path.read_text())["reload_prediction_equivalence"]["passed"]
    rows = _source_rows()
    predictions = loaded.predict_rows(rows[:2])
    assert list(predictions[0]) == ["rebalance_date", "symbol", f"{PREDICTION_PREFIX}random_forest"]


@pytest.mark.parametrize("model_name", TABULAR_MODEL_NAMES)
def test_all_active_tabular_selector_models_persist_and_reload(tmp_path, model_name):
    config = _write_selector_inputs(tmp_path / model_name, winner=model_name)
    result = write_final_fitted_stock_selector(config)
    loaded = load_final_fitted_stock_selector(result.output_dir)

    assert loaded.feature_contract["selected_model_name"] == model_name
    assert loaded.predict_rows(_source_rows()[:1])


def test_loader_rejects_missing_or_reordered_features(tmp_path):
    config = _write_selector_inputs(tmp_path, winner="gradient_boosting")
    result = write_final_fitted_stock_selector(config)
    loaded = load_final_fitted_stock_selector(result.output_dir)
    row = dict(_source_rows()[0])
    row.pop(FEATURE_COLUMNS[0])

    with pytest.raises(ValueError, match="Missing"):
        loaded.predict_rows([row])
    with pytest.raises(ValueError, match="feature order"):
        loaded.predict_feature_matrix(_source_rows()[:1], feature_columns=list(reversed(FEATURE_COLUMNS)))


@pytest.mark.parametrize(
    ("identity_key", "replacement"),
    [
        ("dataset_hash", "changed-dataset"),
        ("model_input_hash", "changed-input"),
        ("final_fit_decision_timestamp", "2030-01-01"),
        ("source_oos_evidence_identity", {"changed": True}),
        ("target_provenance_contract_version", "changed-contract"),
    ],
)
def test_reuse_identity_changes_invalidate_completed_selector(tmp_path, identity_key, replacement):
    config = _write_selector_inputs(tmp_path, winner="ridge")
    result = write_final_fitted_stock_selector(config)
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
    expected = dict(manifest["identity"])
    expected[identity_key] = replacement

    with pytest.raises(RuntimeError, match="reuse identity mismatch"):
        assert_final_fit_reuse_compatible(result.output_dir, expected)


def test_partial_final_fit_does_not_update_latest_completed(tmp_path, monkeypatch):
    import core.research.ml.stock_level.final_fitted_selector as module

    config = _write_selector_inputs(tmp_path, winner="ridge")

    def fail_preserve(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "preserve_immutable_run", fail_preserve)
    with pytest.raises(RuntimeError, match="boom"):
        write_final_fitted_stock_selector(config)

    assert not (tmp_path / "selector" / "final_fitted_selector" / "latest_completed.json").exists()


def test_complete_final_fit_updates_latest_without_champion_or_deep_news(tmp_path):
    config = _write_selector_inputs(tmp_path, winner="ridge")
    result = write_final_fitted_stock_selector(config)
    audit = json.loads(result.audit_path.read_text())

    assert result.latest_completed_path.exists()
    assert not (result.output_dir / "champion.json").exists()
    assert audit["deep_selector_models_enabled"] is False
    assert audit["news_enabled"] is False
    source = Path("core/research/ml/stock_level/final_fitted_selector.py").read_text()
    assert "broker" not in source
    assert "paper" not in source
    assert "live" not in source


def _write_selector_inputs(
    tmp_path: Path,
    *,
    winner: str,
    cutoff: str = "2024-01-12",
) -> dict:
    selector_dir = tmp_path / "selector"
    selector_dir.mkdir(parents=True)
    source_path = selector_dir / "stock_level_prediction_artifacts.csv"
    _write_csv(source_path, _source_rows())
    benchmark_path = selector_dir / "stock_level_model_ranking_benchmark.json"
    leaderboard_path = selector_dir / "stock_level_model_ranking_benchmark.csv"
    predictions_path = selector_dir / "stock_level_model_oos_predictions.csv"
    leaderboard = _leaderboard(winner)
    _write_csv(leaderboard_path, leaderboard)
    _write_csv(predictions_path, _oos_predictions())
    benchmark = {
        "feature_columns": list(FEATURE_COLUMNS),
        "best_ml_model": next(row for row in leaderboard if row["name"] == winner),
        "walk_forward": {"folds": [{"test_end_date": cutoff, "oos_prediction_date_max": cutoff}]},
        "temporal_policy": {"workflow": "stock_selector_oos_benchmark"},
    }
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    run_id = deterministic_run_id("stock_selector_benchmark", {"winner": winner, "cutoff": cutoff})
    preserve_immutable_run(
        output_dir=selector_dir,
        run_id=run_id,
        kind="stock_selector_benchmark",
        identity={"winner": winner, "cutoff": cutoff},
        artifact_paths=(benchmark_path, leaderboard_path, predictions_path),
    )
    return {
        "ml": {
            "output_dir": str(selector_dir),
            "stock_level_prediction_artifacts_path": str(source_path),
            "stock_level_model_ranking_benchmark_path": str(benchmark_path),
            "stock_level_model_oos_predictions_path": str(predictions_path),
            "stock_selector_final_fit_decision_timestamp": cutoff,
            "stock_ranker_include_sequence_models": False,
            "stock_ranker_model_set": "fast",
            "sklearn_n_jobs": 1,
            "random_seed": 7,
        }
    }


def _source_rows() -> list[dict[str, str]]:
    rows = []
    for date_index in range(10):
        day = date_index + 1
        date = f"2024-01-{day:02d}"
        label_start = f"2024-01-{day + 1:02d}"
        label_end = f"2024-01-{day + 2:02d}"
        label_available = f"2024-01-{day + 3:02d}"
        for symbol_index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
            base = float(symbol_index + 1) / 10.0 + date_index / 100.0
            row = {
                "rebalance_date": date,
                "symbol": symbol,
                "benchmark_symbol": "SPY",
                "actual_forward_return_10d": str(base / 10.0),
                "actual_forward_return_5d": str(base / 20.0),
                "actual_future_volatility": "0.1",
                "actual_future_drawdown": "-0.01",
                "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
                "feature_timestamp": date,
                "decision_timestamp": date,
                "target_horizon": "10d",
                "target_observation_count": "10",
                "target_start_timestamp": label_start,
                "label_start_timestamp": label_start,
                "label_end_timestamp": label_end,
                "label_available_timestamp": label_available,
                "target_price_convention": "close_to_close",
            }
            for offset, column in enumerate(FEATURE_COLUMNS):
                row[column] = str(base + offset / 100.0)
            rows.append(row)
    return rows


def _leaderboard(winner: str) -> list[dict[str, str]]:
    rows = []
    ordered = [winner, *(name for name in TABULAR_MODEL_NAMES if name != winner)]
    for rank, name in enumerate(ordered, start=1):
        rows.append(
            {
                "rank": str(rank),
                "name": name,
                "kind": "ml_model",
                "signal_column": f"{PREDICTION_PREFIX}{name}",
                "mean_spearman_ic": str(0.11 - rank / 100.0),
                "top_minus_bottom_spread": str(0.05 - rank / 1000.0),
            }
        )
    return rows


def _oos_predictions() -> list[dict[str, str]]:
    return [
        {
            "rebalance_date": "2024-01-08",
            "symbol": "AAA",
            "fold_id": "1",
            "actual_forward_return_10d": "0.1",
            **{f"{PREDICTION_PREFIX}{name}": "0.1" for name in TABULAR_MODEL_NAMES},
        }
    ]


def _rewrite_source_targets_to_favour_other_model(tmp_path: Path) -> None:
    path = tmp_path / "selector" / "stock_level_prediction_artifacts.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row["actual_forward_return_10d"] = "10.0" if row["symbol"] == "DDD" else "-10.0"
    _write_csv(path, rows)


def _rewrite_final_rows(tmp_path: Path, *, target_value: float) -> None:
    path = tmp_path / "selector" / "stock_level_prediction_artifacts.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows[-8:]:
        row["actual_forward_return_10d"] = str(target_value)
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
