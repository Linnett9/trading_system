from __future__ import annotations

import csv
import inspect
import json
import math
from statistics import mean

import pytest

from core.research.ml import (
    allocation_optimizer,
    allocation_v2,
    allocation_v2_variants,
    meta_auxiliary,
)
from core.research.ml.allocation_optimizer import (
    build_optimizer_sampler,
    optuna_is_available,
    score_optimizer_candidate,
)
from core.research.ml.allocation_v2 import (
    _forecast_values,
    _policy_exposures,
    write_allocation_v2_reports,
)


def test_allocation_policies_use_probability_and_multitask_forecasts():
    rows = [
        _row("2026-01-01", 0.01, return_10d=0.03, drawdown=-0.02, volatility=0.02),
        _row("2026-01-08", -0.01, return_10d=0.01, drawdown=-0.20, volatility=0.10),
    ]

    policies = _policy_exposures(
        rows,
        probabilities=[0.4, 0.9],
        config={"decision_threshold": 0.5, "promotion_reduced_exposure": 0.7},
    )

    assert policies["binary_exposure_overlay"] == [1.0, 0.7]
    assert policies["return_only_allocation"] == [1.0, 0.8]
    assert policies["risk_adjusted_allocation"] == [0.8, 0.0]
    assert policies["meta_ensemble_allocation"] == [0.8, 0.0]


def test_allocation_v2_writes_ranked_research_only_outputs(tmp_path):
    rows = [
        _row("2026-01-01", 0.02, return_10d=0.03),
        _row("2026-01-08", -0.01, return_10d=-0.03),
        _row("2026-01-15", 0.01, return_10d=0.02),
    ]
    diagnostics = {
        "balanced_accuracy": 0.61,
        "brier_score": 0.19,
        "expected_calibration_error": 0.08,
    }

    paths = write_allocation_v2_reports(
        output_dir=tmp_path,
        rows=rows,
        meta_probabilities=[0.2, 0.9, 0.3],
        diagnostics=diagnostics,
        config={"allocation_transaction_cost_bps": 5.0},
    )

    comparison = json.loads(paths.comparison_json.read_text(encoding="utf-8"))
    with paths.comparison_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    shadow = json.loads(paths.shadow_overlay_json.read_text(encoding="utf-8"))

    assert len(comparison["policies"]) == 11
    assert len(csv_rows) == 15
    assert comparison["classification_metrics_role"] == "diagnostics_only"
    assert comparison["research_only"] is True
    assert comparison["trading_impact"] == "none"
    assert comparison["production_validated"] is False
    assert comparison["automatic_promotion"] is False
    assert all(row["estimated_transaction_costs"] >= 0.0 for row in comparison["policies"])
    assert all(row["balanced_accuracy"] == 0.61 for row in comparison["policies"])
    assert [row["total_return"] for row in comparison["ranking"]] == sorted(
        (row["total_return"] for row in comparison["ranking"]),
        reverse=True,
    )
    for row in comparison["policies"]:
        assert row["policy_name"]
        assert row["policy_version"] == "2.0.0"
        assert row["required_prediction_columns"]
        assert 0.0 <= row["exposure_min"] <= row["exposure_max"] <= 1.0
        assert row["research_only"] is True
        assert row["trading_impact"] == "none"
        assert row["production_validated"] is False
        assert 0.0 <= row["min_exposure"] <= row["max_exposure"] <= 1.0
        assert "performance_when_exposure_reduced" in row
        assert "performance_when_exposure_high" in row
        assert "performance_during_worst_drawdown_windows" in row
        assert "median_exposure" in row
        assert "maximum_one_period_exposure_change" in row
        assert "pct_periods_at_100_exposure" in row
        assert "prediction_to_exposure_diagnostics" in row
        assert row["drawdown_impact"]["effect"] in {
            "avoided",
            "worsened",
            "unchanged",
        }
    assert shadow["selection_is_not_promotion"] is True
    assert shadow["research_only"] is True
    assert shadow["trading_impact"] == "none"
    assert shadow["production_validated"] is False
    assert set(shadow["policies"]) == {
        "binary_exposure_overlay",
        "return_only_allocation",
        "risk_adjusted_allocation",
        "meta_ensemble_allocation",
        "return_only_allocation_conservative",
        "return_only_allocation_balanced",
        "return_only_allocation_aggressive",
        "risk_adjusted_allocation_conservative",
        "risk_adjusted_allocation_balanced",
        "risk_adjusted_allocation_aggressive",
        "best_grid_search_diagnostic_policy",
    }
    assert set(shadow["baselines"]) == {
        "champion_baseline",
        "always_full_exposure",
        "always_half_exposure",
        "always_zero_exposure",
    }
    assert "Trading impact: none" in paths.leaderboard_markdown.read_text(
        encoding="utf-8"
    )
    assert all(row["research_only"] == "True" for row in csv_rows)
    assert all(row["trading_impact"] == "none" for row in csv_rows)
    assert all(row["production_validated"] == "False" for row in csv_rows)
    diagnostics = json.loads(paths.diagnostics_json.read_text(encoding="utf-8"))
    assert diagnostics["research_only"] is True
    assert diagnostics["trading_impact"] == "none"
    assert diagnostics["production_validated"] is False
    assert paths.diagnostics_markdown.exists()
    grid = json.loads(paths.grid_search_json.read_text(encoding="utf-8"))
    assert grid["grid_size"] == 32
    assert grid["selected_diagnostic_policy"]["overfit_warning"] == (
        "Research-only diagnostic. Policy selected on holdout is not production-valid."
    )
    assert all(
        math.isfinite(float(row[metric]))
        for row in grid["candidates"]
        for metric in (
            "total_return",
            "max_drawdown",
            "sharpe",
            "sortino",
            "calmar",
            "turnover",
            "estimated_transaction_costs",
            "objective",
        )
    )
    assert paths.grid_search_csv.exists()
    assert paths.grid_search_markdown.exists()
    constant_baselines = {
        row["policy_name"]
        for row in comparison["baselines"]
        if row["robustness_flags"]["exposure_is_constant"]
    }
    assert constant_baselines == {
        "champion_baseline",
        "always_full_exposure",
        "always_half_exposure",
        "always_zero_exposure",
    }


def test_selected_optimizer_exposure_path_is_persisted(tmp_path):
    rows = [
        _row("2024-01-01", 0.02, return_10d=0.03, drawdown=-0.02, volatility=0.10),
        _row("2024-01-08", -0.01, return_10d=-0.02, drawdown=-0.08, volatility=0.20),
        _row("2024-01-15", 0.03, return_10d=0.04, drawdown=-0.01, volatility=0.08),
        _row("2024-01-22", 0.01, return_10d=0.01, drawdown=-0.03, volatility=0.12),
    ]
    selection_rows = [
        _row("2023-12-01", 0.01, return_10d=0.02, drawdown=-0.02, volatility=0.10),
        _row("2023-12-08", -0.02, return_10d=-0.03, drawdown=-0.10, volatility=0.30),
        _row("2023-12-15", 0.02, return_10d=0.03, drawdown=-0.01, volatility=0.08),
        _row("2023-12-22", 0.01, return_10d=0.01, drawdown=-0.03, volatility=0.12),
    ]

    paths = write_allocation_v2_reports(
        output_dir=tmp_path,
        rows=rows,
        meta_probabilities=[0.2, 0.8, 0.3, 0.4],
        diagnostics={},
        config={
            "allocation_optimizer": {
                "enabled": True,
                "sampler": "random",
                "candidate_count": 3,
            }
        },
        selection_rows=selection_rows,
        selection_meta_probabilities=[0.2, 0.8, 0.3, 0.4],
    )

    exposure_path = json.loads(
        paths.selected_optimizer_exposure_path_json.read_text(encoding="utf-8")
    )

    assert paths.selected_optimizer_exposure_path_csv.exists()
    assert exposure_path["mode"] == "selected_optimizer_exposure_path_research_only"
    assert exposure_path["row_count"] == 4
    assert exposure_path["rows"][0]["rebalance_date"] == "2024-01-01"
    assert "net_return" in exposure_path["rows"][0]
    assert "drawdown" in exposure_path["rows"][0]


def test_missing_auxiliary_columns_skip_only_affected_policies(tmp_path):
    rows = [
        {
            "rebalance_date": "2026-01-01",
            "champion_return_next_period": "0.01",
        },
        {
            "rebalance_date": "2026-01-08",
            "champion_return_next_period": "-0.02",
        },
    ]

    paths = write_allocation_v2_reports(
        output_dir=tmp_path,
        rows=rows,
        meta_probabilities=[0.2, 0.8],
        diagnostics={},
        config={},
    )

    comparison = json.loads(paths.comparison_json.read_text(encoding="utf-8"))
    policies = {row["policy_name"]: row for row in comparison["policies"]}

    assert policies["binary_exposure_overlay"]["available"] is True
    assert policies["meta_ensemble_allocation"]["available"] is True
    assert policies["return_only_allocation"]["available"] is False
    assert policies["risk_adjusted_allocation"]["available"] is False
    assert "missing required prediction columns" in policies[
        "return_only_allocation"
    ]["skip_reason"]
    assert comparison["available_policy_count"] == 2
    assert comparison["skipped_policy_count"] == 8


def test_binary_policy_still_works_alone_and_clips_configured_exposure():
    rows = [{
        "rebalance_date": "2026-01-01",
        "champion_return_next_period": "0.01",
    }]

    policies = _policy_exposures(
        rows,
        probabilities=[0.9],
        config={
            "decision_threshold": 0.5,
            "promotion_reduced_exposure": 4.0,
        },
    )

    assert policies == {
        "binary_exposure_overlay": [1.0],
        "meta_ensemble_allocation": [0.0],
    }
    assert all(
        0.0 <= exposure <= 1.0
        for exposures in policies.values()
        for exposure in exposures
    )


def test_higher_expected_return_does_not_reduce_exposure():
    rows = [
        _row("2026-01-01", 0.0, return_10d=-0.01),
        _row("2026-01-08", 0.0, return_10d=0.03),
    ]

    exposures = _policy_exposures(rows, [0.4, 0.4], {})[
        "return_only_allocation"
    ]

    assert exposures[1] >= exposures[0]


def test_higher_predicted_drawdown_risk_reduces_exposure():
    rows = [
        _row("2026-01-01", 0.0, return_10d=0.04, drawdown=-0.01),
        _row("2026-01-08", 0.0, return_10d=0.04, drawdown=-0.06),
    ]

    exposures = _policy_exposures(rows, [0.4, 0.4], {})[
        "risk_adjusted_allocation"
    ]

    assert exposures[1] < exposures[0]


def test_higher_predicted_volatility_reduces_exposure():
    rows = [
        _row("2026-01-01", 0.0, return_10d=0.03, volatility=0.02),
        _row("2026-01-08", 0.0, return_10d=0.03, volatility=0.10),
    ]

    exposures = _policy_exposures(rows, [0.4, 0.4], {})[
        "risk_adjusted_allocation"
    ]

    assert exposures[1] < exposures[0]


def test_allocation_prefers_meta_auxiliary_forecasts(tmp_path):
    rows = [
        {
            **_row("2026-01-01", 0.01, return_10d=-0.20),
            "meta_predicted_forward_return_10d": "0.04",
            "meta_predicted_future_drawdown": "-0.01",
            "meta_predicted_future_volatility": "0.02",
        },
        {
            **_row("2026-01-08", -0.01, return_10d=-0.15),
            "meta_predicted_forward_return_10d": "0.03",
            "meta_predicted_future_drawdown": "-0.02",
            "meta_predicted_future_volatility": "0.03",
        },
    ]

    assert _forecast_values(rows[0], "predicted_forward_return_10d") == [0.04]
    paths = write_allocation_v2_reports(
        output_dir=tmp_path,
        rows=rows,
        meta_probabilities=[0.4, 0.4],
        diagnostics={},
        config={},
    )
    comparison = json.loads(paths.comparison_json.read_text(encoding="utf-8"))
    policies = {row["policy_name"]: row for row in comparison["policies"]}

    assert policies["return_only_allocation"]["forecast_source"] == "meta_auxiliary"
    assert policies["risk_adjusted_allocation"]["forecast_source"] == "meta_auxiliary"
    assert policies["binary_exposure_overlay"]["forecast_source"] == "probability_only"


def test_conservative_variant_has_lower_average_exposure_than_aggressive():
    rows = [
        _row(
            f"2026-01-{index + 1:02d}",
            0.0,
            return_10d=-0.04 + (index * 0.01),
            drawdown=-0.01 - (index * 0.005),
            volatility=0.05 + (index * 0.01),
        )
        for index in range(10)
    ]

    policies = _policy_exposures(rows, [0.4] * len(rows), {})

    assert mean(policies["return_only_allocation_conservative"]) <= mean(
        policies["return_only_allocation_aggressive"]
    )
    assert mean(policies["risk_adjusted_allocation_conservative"]) <= mean(
        policies["risk_adjusted_allocation_aggressive"]
    )


def test_quantile_mapping_creates_ordered_clipped_exposures():
    rows = [
        _row(
            f"2026-02-{index + 1:02d}",
            0.0,
            return_10d=float(index),
        )
        for index in range(10)
    ]

    exposures = _policy_exposures(rows, [0.4] * len(rows), {})[
        "return_only_allocation_balanced"
    ]

    assert exposures == sorted(exposures)
    assert all(0.0 <= exposure <= 1.0 for exposure in exposures)


def test_grid_selection_is_frozen_before_holdout_evaluation(tmp_path):
    selection_rows = [
        _row(
            f"2025-01-{index + 1:02d}",
            0.02 if index >= 4 else -0.01,
            return_10d=-0.04 + (index * 0.01),
            drawdown=-0.08 + (index * 0.005),
            volatility=0.20 - (index * 0.01),
        )
        for index in range(8)
    ]
    holdout_rows = [
        _row(
            f"2026-01-{index + 1:02d}",
            0.10 if index % 2 else -0.10,
            return_10d=-0.02 + (index * 0.02),
            drawdown=-0.05,
            volatility=0.10,
        )
        for index in range(4)
    ]
    reversed_outcomes = [
        {**row, "champion_return_next_period": str(-float(row["champion_return_next_period"]))}
        for row in holdout_rows
    ]

    optimizer_config = {
        "allocation_optimizer": {
            "sampler": "bayesian",
            "candidate_count": 8,
            "bootstrap_iterations": 50,
            "random_seed": 7,
            "bayesian": {
                "n_trials": 8,
                "seed": 7,
                "startup_trials": 2,
                "direction": "maximize",
            },
        }
    }
    first = write_allocation_v2_reports(
        output_dir=tmp_path / "first",
        rows=holdout_rows,
        meta_probabilities=[0.4] * len(holdout_rows),
        diagnostics={},
        config=optimizer_config,
        selection_rows=selection_rows,
        selection_meta_probabilities=[0.4] * len(selection_rows),
    )
    second = write_allocation_v2_reports(
        output_dir=tmp_path / "second",
        rows=reversed_outcomes,
        meta_probabilities=[0.4] * len(reversed_outcomes),
        diagnostics={},
        config=optimizer_config,
        selection_rows=selection_rows,
        selection_meta_probabilities=[0.4] * len(selection_rows),
    )

    first_grid = json.loads(first.grid_search_json.read_text(encoding="utf-8"))
    second_grid = json.loads(second.grid_search_json.read_text(encoding="utf-8"))
    first_selected = first_grid["selected_diagnostic_policy"]
    second_selected = second_grid["selected_diagnostic_policy"]

    assert first_selected["candidate_id"] == second_selected["candidate_id"]
    assert first_grid["selection_protocol"] == (
        "out_of_fold_train_selection_then_frozen_holdout_evaluation"
    )
    assert first_selected["selection_metrics"]["total_return"] == (
        second_selected["selection_metrics"]["total_return"]
    )
    assert first_selected["holdout_metrics"]["total_return"] != (
        second_selected["holdout_metrics"]["total_return"]
    )
    comparison = json.loads(first.comparison_json.read_text(encoding="utf-8"))
    quantile_policy = next(
        row
        for row in comparison["policies"]
        if row["policy_name"] == "return_only_allocation_balanced"
    )
    assert quantile_policy["threshold_fit_scope"] == (
        "out_of_fold_train_predictions"
    )
    first_optimizer = json.loads(
        first.optimizer_results_json.read_text(encoding="utf-8")
    )
    second_optimizer = json.loads(
        second.optimizer_results_json.read_text(encoding="utf-8")
    )
    assert first_optimizer["candidate_count"] == 8
    assert len(first_optimizer["candidates"]) == 8
    assert first_optimizer["selected_policy"]["candidate_id"] == (
        second_optimizer["selected_policy"]["candidate_id"]
    )
    paired = first_optimizer["paired_comparison_vs_binary_overlay"]
    assert paired["available"] is True
    assert all(
        math.isfinite(value)
        for value in paired["compounded_return_delta"]["confidence_interval_95"]
    )
    assert first.optimizer_candidates_csv.exists()
    assert first.optimizer_report_markdown.exists()
    assert first_optimizer["research_only"] is True
    assert first_optimizer["trading_impact"] == "none"
    assert first_optimizer["production_validated"] is False
    assert first_optimizer["sampler_requested"] == "bayesian"
    expected_sampler = "bayesian" if optuna_is_available() else "random"
    assert first_optimizer["sampler_used"] == expected_sampler
    assert first_optimizer["optuna_available"] is optuna_is_available()
    if not optuna_is_available():
        assert "Optuna is not installed" in first_optimizer["fallback_reason"]
    assert first_optimizer["selected_policy"]["selected_params"] == (
        first_optimizer["selected_policy"]["parameters"]
    )
    assert first_optimizer["selected_policy"]["frozen_holdout_metrics"] == (
        first_optimizer["selected_policy"]["holdout_metrics"]
    )
    assert all(
        row["sampler_requested"] == "bayesian"
        and row["sampler_used"] == expected_sampler
        and row["trial_number"] >= 0
        and math.isfinite(row["objective_value"])
        for row in first_optimizer["candidates"]
    )


def test_random_optimizer_sampler_is_deterministic_and_bounded():
    config = {
        "allocation_optimizer": {
            "random_seed": 11,
            "ranges": {
                "min_exposure": [0.10, 0.20],
                "max_exposure": [0.80, 0.90],
                "neutral_exposure": [0.40, 0.60],
            },
        }
    }

    first = build_optimizer_sampler(config).sample(12)
    second = build_optimizer_sampler(config).sample(12)

    assert first == second
    assert len(first) == 12
    metadata = build_optimizer_sampler(config).metadata()
    assert metadata["sampler_requested"] == "random"
    assert metadata["sampler_used"] == "random"
    assert metadata["fallback_reason"] is None
    assert all(
        0.0 <= row["min_exposure"]
        <= row["neutral_exposure"]
        <= row["max_exposure"]
        <= 1.0
        for row in first
    )


def test_bayesian_sampler_falls_back_when_optuna_is_unavailable(monkeypatch):
    monkeypatch.setattr(allocation_optimizer, "optuna_is_available", lambda: False)

    sampler = build_optimizer_sampler({
        "allocation_optimizer": {
            "sampler": "bayesian",
            "bayesian": {"seed": 19, "n_trials": 4},
        }
    })
    metadata = sampler.metadata()

    assert metadata["sampler_requested"] == "bayesian"
    assert metadata["sampler_used"] == "random"
    assert metadata["optuna_available"] is False
    assert "falling back" in metadata["fallback_reason"]
    assert len(sampler.sample(4)) == 4


def test_optimizer_legacy_diagnostic_objective_remains_default():
    result = score_optimizer_candidate(
        diagnostic_objective=1.234,
        exposure_path=[],
        config={},
        candidate_name="legacy",
    )

    assert result["objective_mode"] == "diagnostic_period_grid_return"
    assert result["objective_value"] == pytest.approx(1.234)
    assert result["canonical_non_overlap_return"] is None


def test_canonical_objective_can_select_a_different_candidate():
    fragile = _optimizer_path(
        [0.50, 0.50],
        outcome_end_dates=["2024-02-01", "2024-02-02"],
    )
    stable = _optimizer_path(
        [0.40, 0.40],
        outcome_end_dates=["2024-01-01", "2024-01-02"],
    )
    config = {
        "allocation_optimizer": {
            "objective_mode": "canonical_non_overlap_return",
        }
    }

    fragile_score = score_optimizer_candidate(
        diagnostic_objective=1.25,
        exposure_path=fragile,
        config=config,
        candidate_name="fragile",
    )
    stable_score = score_optimizer_candidate(
        diagnostic_objective=0.96,
        exposure_path=stable,
        config=config,
        candidate_name="stable",
    )

    assert 1.25 > 0.96
    assert stable_score["objective_value"] > fragile_score["objective_value"]


def test_anomaly_sensitive_candidate_is_penalized():
    config = {
        "allocation_optimizer": {
            "objective_mode": "robustness_adjusted_canonical_score",
            "anomaly_policy": "penalize",
            "max_allowed_anomaly_dependency_ratio": 0.25,
        }
    }

    result = score_optimizer_candidate(
        diagnostic_objective=0.80,
        exposure_path=_optimizer_path([0.80, 0.05]),
        config=config,
        candidate_name="anomaly_sensitive",
    )

    assert result["flagged_anomaly_dates"] == ["2024-01-01"]
    assert result["anomaly_adjusted_canonical_return"] < result[
        "canonical_non_overlap_return"
    ]
    assert result["anomaly_dependency_ratio"] > 0.25
    assert result["anomaly_dependency_penalty"] > 0.0


def test_anomaly_adjusted_objective_uses_quarantined_return():
    result = score_optimizer_candidate(
        diagnostic_objective=0.80,
        exposure_path=_optimizer_path([0.80, 0.05]),
        config={
            "allocation_optimizer": {
                "objective_mode": "anomaly_adjusted_canonical_return",
                "anomaly_policy": "penalize",
            }
        },
        candidate_name="anomaly_adjusted",
    )

    assert result["objective_value"] == result[
        "anomaly_adjusted_canonical_return"
    ]
    assert result["objective_value"] < result["canonical_non_overlap_return"]


def test_robust_lower_raw_return_beats_fragile_high_return():
    config = {
        "allocation_optimizer": {
            "objective_mode": "robustness_adjusted_canonical_score",
            "anomaly_policy": "penalize",
            "cost_stress_multiplier": 2.0,
            "max_allowed_anomaly_dependency_ratio": 0.25,
        }
    }
    fragile = score_optimizer_candidate(
        diagnostic_objective=0.80,
        exposure_path=_optimizer_path([0.80]),
        config=config,
        candidate_name="fragile",
    )
    robust = score_optimizer_candidate(
        diagnostic_objective=0.60,
        exposure_path=_optimizer_path([0.10] * 6),
        config=config,
        candidate_name="robust",
    )

    assert fragile["canonical_non_overlap_return"] > robust[
        "canonical_non_overlap_return"
    ]
    assert robust["robustness_adjusted_score"] > fragile[
        "robustness_adjusted_score"
    ]


def test_optimizer_inner_loop_scoring_writes_no_report_files(tmp_path):
    before = set(tmp_path.iterdir())

    score_optimizer_candidate(
        diagnostic_objective=0.10,
        exposure_path=_optimizer_path([0.05, 0.04]),
        config={
            "allocation_optimizer": {
                "objective_mode": "robustness_adjusted_canonical_score",
            }
        },
        candidate_name="no_io",
    )

    assert set(tmp_path.iterdir()) == before


@pytest.mark.skipif(not optuna_is_available(), reason="Optuna is not installed")
def test_optuna_bayesian_sampler_smoke():
    sampler = build_optimizer_sampler({
        "allocation_optimizer": {
            "sampler": "bayesian",
            "bayesian": {
                "n_trials": 4,
                "seed": 5,
                "startup_trials": 2,
                "direction": "maximize",
            },
        }
    })

    for trial_number in range(4):
        candidate = sampler.suggest(trial_number)
        sampler.observe(candidate, float(candidate["return_weight"]))

    metadata = sampler.metadata()
    assert metadata["sampler_used"] == "bayesian"
    assert metadata["best_trial_number"] is not None
    assert metadata["n_trials"] == 4


def test_allocation_v2_does_not_import_operational_code_paths():
    source = inspect.getsource(allocation_v2) + inspect.getsource(
        allocation_v2_variants
    ) + inspect.getsource(meta_auxiliary) + inspect.getsource(allocation_optimizer)

    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "broker" not in source
    assert "production_execution" not in source


def _row(
    date: str,
    period_return: float,
    *,
    return_10d: float,
    drawdown: float = -0.01,
    volatility: float = 0.01,
) -> dict[str, str]:
    return {
        "rebalance_date": date,
        "champion_return_next_period": str(period_return),
        "multitask_transformer_predicted_forward_return_10d": str(return_10d),
        "multitask_transformer_predicted_future_drawdown": str(drawdown),
        "multitask_transformer_predicted_future_volatility": str(volatility),
        "multitask_transformer_predicted_max_adverse_excursion": str(drawdown),
        "multitask_transformer_predicted_max_favourable_excursion": str(
            max(return_10d, 0.0)
        ),
    }


def _optimizer_path(
    returns: list[float],
    *,
    outcome_end_dates: list[str] | None = None,
) -> list[dict[str, object]]:
    dates = [f"2024-01-{index + 1:02d}" for index in range(len(returns))]
    ends = outcome_end_dates or dates
    return [
        {
            "rebalance_date": date,
            "outcome_end_date": end,
            "period_return": period_return,
            "exposure": 1.0,
            "turnover": 0.0,
            "cost": 0.0,
            "net_return": period_return,
            "selected_symbols": ["SPY"],
        }
        for date, end, period_return in zip(dates, ends, returns)
    ]
