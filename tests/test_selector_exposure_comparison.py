from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.research.ml.immutable_runs import deterministic_run_id, preserve_immutable_run
from core.research.ml.stock_level.selector_exposure_comparison import (
    DECISION_DOES_NOT_ADD_VALUE,
    DECISION_INSUFFICIENT,
    write_selector_exposure_comparison,
)


def test_selector_exposure_comparison_uses_same_oos_holdings_weights_dates_and_costs(tmp_path):
    config = _write_inputs(tmp_path)

    result = write_selector_exposure_comparison(config)
    matched = list(csv.DictReader(result.matched_periods_csv.open()))
    audit = json.loads(result.audit_json.read_text())
    summary = json.loads(result.comparison_summary_json.read_text())

    assert audit["strict_oos_selector_predictions_used"] is True
    assert audit["final_fitted_selector_used_for_historical_evaluation"] is False
    assert audit["matched_invariants"]["holdings_identical"] is True
    assert audit["matched_invariants"]["base_weights_identical"] is True
    assert audit["matched_invariants"]["benchmark_returns_identical"] is True
    assert audit["matched_invariants"]["starting_capital_identical"] is True
    assert [row["rebalance_date"] for row in matched] == ["2024-01-02", "2024-01-03", "2024-01-04"]
    for row in matched:
        expected = (
            float(row["base_portfolio_return_before_costs"]) * float(row["exposure_multiplier"])
            - float(row["variant_a_cost"])
            - float(row["incremental_overlay_cost"])
        )
        assert float(row["variant_b_net_return"]) == pytest.approx(expected)
    assert summary["research_conclusion"] in {DECISION_DOES_NOT_ADD_VALUE, DECISION_INSUFFICIENT}


def test_exposure_multiplier_one_reproduces_variant_a_before_incremental_costs(tmp_path):
    config = _write_inputs(tmp_path, probabilities=[0.1, 0.1, 0.1])
    result = write_selector_exposure_comparison(config)
    matched = list(csv.DictReader(result.matched_periods_csv.open()))

    assert all(float(row["exposure_multiplier"]) == 1.0 for row in matched)
    assert all(
        float(row["variant_b_net_return"]) == pytest.approx(float(row["variant_a_net_return"]))
        for row in matched
    )


def test_exposure_multiplier_zero_produces_cash_behaviour_and_incremental_cost(tmp_path):
    config = _write_inputs(tmp_path, probabilities=[1.0, 1.0, 1.0], reduced_exposure=0.0)
    result = write_selector_exposure_comparison(config)
    matched = list(csv.DictReader(result.matched_periods_csv.open()))

    assert all(float(row["exposure_multiplier"]) == 0.0 for row in matched)
    assert float(matched[0]["incremental_overlay_cost"]) > 0.0
    assert float(matched[0]["variant_b_net_return"]) < float(matched[0]["variant_a_net_return"])


def test_selector_exposure_comparison_rejects_dual_momentum_dataset(tmp_path):
    config = _write_inputs(tmp_path)
    dataset = Path(config["ml"]["stock_selector_rebalance_dataset_path"])
    rows = list(csv.DictReader(dataset.open()))
    for row in rows:
        row.pop("selector_signal")
        row["strategy_name"] = "dual_momentum"
    _write_csv(dataset, rows)

    with pytest.raises(RuntimeError, match="selector-derived"):
        write_selector_exposure_comparison(config)


def test_selector_exposure_comparison_requires_strict_oos_meta_predictions(tmp_path):
    config = _write_inputs(tmp_path)
    meta = Path(config["ml"]["selector_exposure_meta_output_dir"]) / "prediction_artifacts.csv"
    rows = list(csv.DictReader(meta.open()))
    rows[0]["split"] = "train"
    _write_csv(meta, rows)

    with pytest.raises(RuntimeError, match="strict OOS"):
        write_selector_exposure_comparison(config)


def test_missing_exposure_dates_are_excluded_and_reported(tmp_path):
    config = _write_inputs(tmp_path)
    meta = Path(config["ml"]["selector_exposure_meta_output_dir"]) / "prediction_artifacts.csv"
    rows = list(csv.DictReader(meta.open()))[:2]
    _write_csv(meta, rows)

    result = write_selector_exposure_comparison(config)
    audit = json.loads(result.audit_json.read_text())

    assert audit["date_matching"]["common_eligible_dates"] == 2
    assert any("missing_exposure_prediction" in row["reason"] for row in audit["date_matching"]["excluded_dates"])


def test_partial_selector_exposure_comparison_does_not_update_latest(tmp_path, monkeypatch):
    import core.research.ml.stock_level.selector_exposure_comparison as module

    config = _write_inputs(tmp_path)

    def fail_preserve(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "preserve_immutable_run", fail_preserve)
    with pytest.raises(RuntimeError, match="boom"):
        write_selector_exposure_comparison(config)

    assert not (tmp_path / "comparison" / "latest_completed.json").exists()


def test_complete_selector_exposure_comparison_updates_latest_without_champion_or_news_deep(tmp_path):
    config = _write_inputs(tmp_path)
    result = write_selector_exposure_comparison(config)
    audit = json.loads(result.audit_json.read_text())

    assert result.latest_completed_path.exists()
    assert not (result.output_dir / "champion.json").exists()
    assert audit["news_enabled"] is False
    assert audit["deep_selector_models_enabled"] is False
    source = Path("core/research/ml/stock_level/selector_exposure_comparison.py").read_text()
    assert "paper" not in source
    assert "broker" not in source
    assert "live" not in source


def _write_inputs(
    tmp_path: Path,
    *,
    probabilities: list[float] | None = None,
    reduced_exposure: float = 0.7,
) -> dict:
    root = tmp_path / "run"
    selector_benchmark = root / "selector_benchmark"
    selector_replay = root / "selector_replay"
    meta = root / "meta"
    cache = tmp_path / "cache"
    for path in (selector_benchmark, selector_replay, meta, cache):
        path.mkdir(parents=True)
    signal = "stock_level_predicted_forward_return_10d_ridge"
    policy = "long_only_top_n_equal_weight"
    strategy_id = f"{signal}|{policy}"
    _write_csv(selector_benchmark / "stock_level_model_oos_predictions.csv", [
        {"rebalance_date": f"2024-01-0{i}", "symbol": "AAA", "fold_id": "1", signal: "0.1"}
        for i in range(2, 5)
    ])
    summary = {
        "winners": {"best_ml_model": {"signal_column": signal, "policy": policy, "strategy_id": strategy_id}},
        "signal_columns": [signal],
        "policies": [policy],
    }
    (selector_replay / "stock_level_portfolio_replay_summary.json").write_text(json.dumps(summary))
    equity = [
        {
            "rebalance_date": f"2024-01-0{i}",
            "strategy_id": strategy_id,
            "signal_column": signal,
            "policy": policy,
            "gross_return": str(0.02 * i),
            "transaction_cost_drag": "0.001",
            "net_return": str(0.02 * i - 0.001),
            "turnover": "0.2",
            "equity": "1.0",
            "benchmark_return": "0.01",
        }
        for i in range(2, 5)
    ]
    _write_csv(selector_replay / "stock_level_portfolio_replay_equity_curves.csv", equity)
    holdings = []
    for i in range(2, 5):
        holdings.extend([
            {"rebalance_date": f"2024-01-0{i}", "strategy_id": strategy_id, "signal_column": signal, "policy": policy, "symbol": "AAA", "weight": "0.6", "side": "long"},
            {"rebalance_date": f"2024-01-0{i}", "strategy_id": strategy_id, "signal_column": signal, "policy": policy, "symbol": "BBB", "weight": "0.4", "side": "long"},
        ])
    _write_csv(selector_replay / "stock_level_portfolio_replay_holdings.csv", holdings)
    probabilities = probabilities or [0.1, 0.9, 0.1]
    _write_csv(meta / "prediction_artifacts.csv", [
        {
            "feature_id": f"f{i}",
            "rebalance_date": f"2024-01-0{i}",
            "split": "out_of_fold",
            "fold": "1",
            "actual_label": "0",
            "predicted_probability": str(probabilities[i - 2]),
            "prediction": "0",
            "model_type": "logistic_regression",
            "validation_method": "chronological_meta_walk_forward_strict_oos",
        }
        for i in range(2, 5)
    ])
    dataset = cache / "stock_selector_rebalance_dataset.csv"
    _write_csv(dataset, [
        {
            "feature_id": f"f{i}",
            "feature_date": f"2024-01-0{i}",
            "rebalance_date": f"2024-01-0{i}",
            "label_available_timestamp": f"2024-01-0{i+1}",
            "selector_signal": signal,
            "portfolio_policy": policy,
            "strategy_id": strategy_id,
            "source_predictions_path": str(selector_benchmark / "stock_level_model_oos_predictions.csv"),
            "should_reduce_exposure": "0",
        }
        for i in range(2, 5)
    ])
    for output_dir, kind in [
        (selector_benchmark, "stock_selector_benchmark"),
        (selector_replay, "stock_selector_replay"),
        (meta, "exposure_meta_ensemble"),
    ]:
        artifact = output_dir / "identity.json"
        artifact.write_text(json.dumps({"kind": kind}))
        preserve_immutable_run(
            output_dir=output_dir,
            run_id=deterministic_run_id(kind, {"path": str(output_dir)}),
            kind=kind,
            identity={"path": str(output_dir)},
            artifact_paths=(artifact,),
        )
    return {
        "ml": {
            "selector_exposure_comparison_source_root": str(root),
            "selector_exposure_cache_root": str(cache),
            "selector_exposure_meta_output_dir": str(meta),
            "stock_selector_rebalance_dataset_path": str(dataset),
            "selector_exposure_comparison_output_dir": str(tmp_path / "comparison"),
            "promotion_reduced_exposure": reduced_exposure,
            "decision_threshold": 0.5,
            "stock_portfolio_replay_cost_bps": 10.0,
            "stock_portfolio_replay_slippage_bps": 5.0,
        }
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
