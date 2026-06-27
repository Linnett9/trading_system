from __future__ import annotations

import csv
import inspect
import json

from core.research.ml import trading_research_leaderboard
from core.research.ml.trading_research_leaderboard import (
    write_trading_research_leaderboard,
)


def test_trading_leaderboard_combines_reports_and_ranks_trading_outcomes(
    tmp_path,
):
    classification_path = tmp_path / "leaderboard.json"
    allocation_path = tmp_path / "allocation_policy_comparison.json"
    optimizer_path = tmp_path / "allocation_optimizer_results.json"
    auxiliary_path = tmp_path / "meta_auxiliary_metrics.json"
    _write_json(classification_path, {
        "leaderboard": [
            _classification_row("accurate_model", 0.10, -0.10, 0.99),
            _classification_row("profitable_model", 0.20, -0.20, 0.40),
            {
                **_classification_row("meta_ensemble_logistic", 0.15, -0.08, 0.70),
                "selection_role": "configured_meta_model",
            },
            {
                **_classification_row("selected_overlay", 0.99, -0.01, 1.0),
                "selection_role": "selected_overlay",
                "selected_model": "meta_ensemble_lightgbm",
            },
        ]
    })
    _write_json(allocation_path, {
        "policies": [
            _allocation_row("risk_adjusted_balanced", 0.18, 0.05),
            {
                "policy_name": "unavailable_policy",
                "available": False,
                "skip_reason": "missing predictions",
                "total_return": 1.0,
            },
        ],
        "baselines": [
            _allocation_row("champion_baseline", 0.25, 0.12),
        ],
    })
    _write_json(optimizer_path, {
        "sampler_requested": "bayesian",
        "sampler_used": "random",
        "objective_mode": "robustness_adjusted_canonical_score",
        "fallback_reason": "Optuna unavailable",
        "selected_policy": {
            "candidate_id": "optimizer_0042",
            "objective_value": 0.7,
            "selected_by_robustness_objective": True,
            "holdout_objective_metrics": {
                "canonical_non_overlap_return": 0.17,
                "anomaly_adjusted_canonical_return": 0.14,
                "anomaly_dependency_ratio": 0.18,
                "robustness_adjusted_score": 0.09,
            },
            "frozen_holdout_metrics": _allocation_row(
                "selected_random_optimizer_diagnostic_policy", 0.19, 0.04
            ),
        },
    })
    _write_json(auxiliary_path, {
        "available_targets": ["actual_forward_return_10d"],
        "targets": {
            "actual_forward_return_10d": {
                "available": True,
                "mae": 0.01,
                "rmse": 0.02,
                "pearson_correlation": 0.4,
                "spearman_correlation": 0.5,
                "directional_accuracy": 0.6,
            }
        },
    })

    paths = write_trading_research_leaderboard(
        tmp_path,
        classification_path,
        allocation_path,
        optimizer_path,
        auxiliary_path,
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    names = [row["entity_name"] for row in payload["leaderboard"]]
    assert names == [
        "champion_baseline",
        "profitable_model",
        "selected_random_optimizer_diagnostic_policy",
        "risk_adjusted_balanced",
        "meta_ensemble_logistic",
        "accurate_model",
    ]
    assert "selected_overlay" not in names
    assert payload["leaderboard"][1]["max_drawdown"] == 0.20
    assert payload["classification_metrics_role"] == "diagnostics_only"
    assert payload["meta_auxiliary_available_targets"] == [
        "actual_forward_return_10d"
    ]
    assert payload["optimizer_status"]["sampler_used"] == "random"
    optimizer_row = next(
        row for row in payload["leaderboard"]
        if row["entity_type"] == "allocation_optimizer"
    )
    assert optimizer_row["optimizer_objective_mode"] == (
        "robustness_adjusted_canonical_score"
    )
    assert optimizer_row["canonical_non_overlap_return"] == 0.17
    assert optimizer_row["anomaly_adjusted_canonical_return"] == 0.14
    assert optimizer_row["anomaly_dependency_ratio"] == 0.18
    assert optimizer_row["robustness_adjusted_score"] == 0.09
    assert optimizer_row["selected_by_robustness_objective"] is True
    assert payload["research_only"] is True
    assert payload["trading_impact"] == "none"
    assert payload["production_validated"] is False

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "## Classification Diagnostics" in markdown
    assert "## Meta Auxiliary Forecast Diagnostics" in markdown
    assert "Research only. Trading impact: none. Production validated: false." in markdown
    with paths.csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["entity_name"] == "champion_baseline"
    assert all(row["research_only"] == "True" for row in csv_rows)


def test_trading_leaderboard_uses_drawdown_as_first_tie_breaker(tmp_path):
    classification_path = tmp_path / "leaderboard.json"
    allocation_path = tmp_path / "allocation.json"
    optimizer_path = tmp_path / "optimizer.json"
    auxiliary_path = tmp_path / "auxiliary.json"
    _write_json(classification_path, {"leaderboard": []})
    _write_json(allocation_path, {
        "policies": [
            _allocation_row("larger_drawdown", 0.20, 0.20, sharpe=5.0),
            _allocation_row("smaller_drawdown", 0.20, 0.10, sharpe=0.1),
        ],
        "baselines": [],
    })
    _write_json(optimizer_path, {})
    _write_json(auxiliary_path, {})

    paths = write_trading_research_leaderboard(
        tmp_path,
        classification_path,
        allocation_path,
        optimizer_path,
        auxiliary_path,
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert [row["entity_name"] for row in payload["leaderboard"]] == [
        "smaller_drawdown",
        "larger_drawdown",
    ]


def test_trading_leaderboard_handles_reports_not_run_yet(tmp_path):
    paths = write_trading_research_leaderboard(
        tmp_path,
        tmp_path / "missing_leaderboard.json",
        tmp_path / "missing_allocation.json",
        tmp_path / "missing_optimizer.json",
        tmp_path / "missing_auxiliary.json",
    )

    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["leaderboard"] == []
    assert payload["meta_auxiliary_forecast_metrics"] == {}
    assert paths.csv_path.read_text(encoding="utf-8").startswith("rank,")


def test_trading_leaderboard_includes_benchmark_validation_status(tmp_path):
    classification_path = tmp_path / "leaderboard.json"
    allocation_path = tmp_path / "allocation.json"
    optimizer_path = tmp_path / "optimizer.json"
    auxiliary_path = tmp_path / "auxiliary.json"
    _write_json(classification_path, {"leaderboard": []})
    _write_json(allocation_path, {"policies": [], "baselines": []})
    _write_json(optimizer_path, {})
    _write_json(auxiliary_path, {})
    _write_json(tmp_path / "benchmark_relative_validation.json", {
        "candidates": [
            {
                "candidate_name": "spy_buy_and_hold",
                "available": True,
                "canonical_non_overlap_return": 0.12,
                "max_drawdown": 0.08,
                "sharpe": 1.0,
                "sortino": 1.2,
                "turnover": 1.0,
                "benchmark_relative_pass": False,
                "tradability_validation_pass": True,
                "promotion_candidate_status": "blocked",
            }
        ]
    })

    paths = write_trading_research_leaderboard(
        tmp_path,
        classification_path,
        allocation_path,
        optimizer_path,
        auxiliary_path,
    )
    row = json.loads(paths.json_path.read_text(encoding="utf-8"))["leaderboard"][0]

    assert row["entity_name"] == "spy_buy_and_hold"
    assert row["benchmark_relative_pass"] is False
    assert row["tradability_validation_pass"] is True
    assert row["promotion_candidate_status"] == "blocked"


def test_trading_leaderboard_does_not_import_operational_code_paths():
    source = inspect.getsource(trading_research_leaderboard)

    assert "paper_trading" not in source
    assert "live_trading" not in source
    assert "broker" not in source
    assert "production_execution" not in source


def _classification_row(
    model: str,
    total_return: float,
    max_drawdown: float,
    balanced_accuracy: float,
) -> dict[str, object]:
    return {
        "model": model,
        "selection_role": None,
        "overlay_total_return": total_return,
        "overlay_max_drawdown": max_drawdown,
        "turnover": 0.3,
        "holdout_balanced_accuracy": balanced_accuracy,
        "walk_forward_balanced_accuracy": balanced_accuracy - 0.05,
        "brier_score": 0.2,
        "expected_calibration_error": 0.1,
    }


def _allocation_row(
    name: str,
    total_return: float,
    max_drawdown: float,
    *,
    sharpe: float = 1.0,
) -> dict[str, object]:
    return {
        "policy_name": name,
        "available": True,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sharpe + 0.1,
        "calmar": sharpe + 0.2,
        "turnover": 0.4,
        "estimated_transaction_costs": 0.001,
    }


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
