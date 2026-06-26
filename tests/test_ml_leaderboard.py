from __future__ import annotations

import json

from core.research.ml.leaderboard import write_leaderboard, write_source_leaderboard


def test_leaderboard_writes_json_and_markdown(tmp_path):
    source = tmp_path / "dlinear"
    source.mkdir()
    (source / "metrics.json").write_text(json.dumps({
        "model_type": "dlinear",
        "metrics": {"accuracy": 0.6, "balanced_accuracy": 0.61},
    }))
    (source / "probability_calibration.json").write_text(json.dumps({
        "brier_score": 0.2,
        "brier_skill_score": 0.1,
        "expected_calibration_error": 0.08,
    }))
    (source / "calibrated_probability_calibration.json").write_text(json.dumps({
        "best_method_by_brier": "isotonic",
    }))
    (source / "holdout_shadow_overlay.json").write_text(json.dumps({
        "result": {
            "base_total_return": 0.1,
            "overlay_total_return": 0.12,
            "base_max_drawdown": -0.2,
            "overlay_max_drawdown": -0.15,
            "overlay_turnover": 1.0,
            "reduced_exposure_days": 5,
        }
    }))

    write_leaderboard(
        tmp_path / "leaderboard.json",
        tmp_path / "leaderboard.md",
        [source],
        {"accuracy": 0.7, "balanced_accuracy": 0.72},
        {"brier_score": 0.18, "brier_skill_score": 0.2},
        {
            "overlay_start_date": "2024-01-01",
            "overlay_end_date": "2024-03-01",
            "overlay_sample_count": 12,
            "overlay_baseline_return": 0.10,
            "overlay_adjusted_return": 0.13,
            "overlay_total_return": 0.13,
            "return_delta": 0.03,
        },
        meta_selections={
            "selected_classifier": {
                "selection_role": "selected_classifier",
                "selected_model": "meta_ensemble_random_forest",
                "selection_reason": "highest holdout balanced accuracy",
                "metrics": {"balanced_accuracy": 0.8},
                "calibration": {
                    "method": "raw",
                    "brier_score": 0.2,
                    "expected_calibration_error": 0.1,
                },
                "overlay": {"return_delta": 0.01, "overlay_sample_count": 10},
                "walk_forward_summary": {"balanced_accuracy": 0.7},
                "promotion_gates": {
                    "promotion_candidate": False,
                    "checks": {"finite_sanity_check": {"passed": True}},
                },
                "promotion_gate_score": 1.23,
            },
            "selected_calibrated": {
                "selection_role": "selected_calibrated",
                "selected_model": "meta_ensemble_logistic",
                "selection_reason": "lowest Brier score",
                "metrics": {"balanced_accuracy": 0.7},
                "calibration": {
                    "method": "platt",
                    "brier_score": 0.18,
                    "expected_calibration_error": 0.08,
                },
                "overlay": {"return_delta": 0.02, "overlay_sample_count": 10},
                "walk_forward_summary": {"balanced_accuracy": 0.65},
                "promotion_gates": {
                    "promotion_candidate": True,
                    "checks": {"finite_sanity_check": {"passed": True}},
                },
                "promotion_gate_score": 1.1,
            },
            "selected_overlay": {
                "selection_role": "selected_overlay",
                "selected_model": "meta_ensemble_gradient_boosting",
                "selection_reason": "highest promotion-gate utility",
                "metrics": {"balanced_accuracy": 0.75},
                "calibration": {
                    "method": "raw",
                    "brier_score": 0.19,
                    "expected_calibration_error": 0.09,
                },
                "overlay": {"return_delta": 0.04, "overlay_sample_count": 10},
                "walk_forward_summary": {"balanced_accuracy": 0.72},
                "promotion_gates": {
                    "promotion_candidate": True,
                    "checks": {"finite_sanity_check": {"passed": True}},
                },
                "promotion_gate_score": 1.5,
            },
        },
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    markdown = (tmp_path / "leaderboard.md").read_text()

    assert payload["trading_impact"] == "none"
    assert any(row["model"] == "meta_ensemble_logistic" for row in payload["leaderboard"])
    assert payload["leaderboard"][1]["calibration_method"] == "isotonic"
    assert payload["leaderboard"][1]["expected_calibration_error"] == 0.08
    meta_row = next(
        row for row in payload["leaderboard"]
        if row["model"] == "meta_ensemble_logistic"
    )
    assert meta_row["overlay_start_date"] == "2024-01-01"
    assert meta_row["overlay_end_date"] == "2024-03-01"
    assert meta_row["overlay_sample_count"] == 12
    assert meta_row["overlay_baseline_return"] == 0.10
    assert meta_row["overlay_adjusted_return"] == 0.13
    selected = {
        row["selection_role"]: row
        for row in payload["leaderboard"]
        if row.get("selection_role")
    }
    assert selected["selected_classifier"]["selected_model"] == (
        "meta_ensemble_random_forest"
    )
    assert selected["selected_calibrated"]["selection_reason"] == "lowest Brier score"
    assert selected["selected_overlay"]["promotion_gate_score"] == 1.5
    assert "dlinear" in markdown
    assert "expected_calibration_error" in markdown
    assert "overlay_sample_count" in markdown
    assert "selected_overlay" in markdown


def test_source_leaderboard_preserves_existing_meta_rows(tmp_path):
    source = tmp_path / "patchtst"
    source.mkdir()
    _write_source_run(source, "patchtst")
    existing = {
        "leaderboard": [
            {"model": "champion_baseline"},
            {"model": "old_source_model"},
            {
                "model": "meta_ensemble_logistic",
                "selection_role": "configured_meta_model",
                "selected_model": "meta_ensemble_logistic",
            },
            {
                "model": "selected_overlay",
                "selection_role": "selected_overlay",
                "selected_model": "meta_ensemble_lightgbm",
            },
        ]
    }
    (tmp_path / "leaderboard.json").write_text(json.dumps(existing), encoding="utf-8")

    write_source_leaderboard(
        tmp_path / "leaderboard.json",
        tmp_path / "leaderboard.md",
        [source],
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    models = [row["model"] for row in payload["leaderboard"]]
    markdown = (tmp_path / "leaderboard.md").read_text()

    assert "champion_baseline" in models
    assert "patchtst" in models
    assert "old_source_model" not in models
    assert "meta_ensemble_logistic" in models
    assert "selected_overlay" in models
    assert "patchtst" in markdown


def _write_source_run(path, model_type: str) -> None:
    (path / "metrics.json").write_text(json.dumps({
        "model_type": model_type,
        "metrics": {"accuracy": 0.6, "balanced_accuracy": 0.61},
    }))
    (path / "probability_calibration.json").write_text(json.dumps({
        "brier_score": 0.2,
        "brier_skill_score": 0.1,
        "expected_calibration_error": 0.08,
    }))
    (path / "calibrated_probability_calibration.json").write_text(json.dumps({
        "best_method_by_brier": "raw",
    }))
    (path / "holdout_shadow_overlay.json").write_text(json.dumps({
        "result": {
            "base_total_return": 0.1,
            "overlay_total_return": 0.12,
            "base_max_drawdown": -0.2,
            "overlay_max_drawdown": -0.15,
            "overlay_turnover": 1.0,
            "reduced_exposure_days": 5,
        }
    }))
