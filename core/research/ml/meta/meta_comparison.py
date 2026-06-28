from __future__ import annotations

from typing import Any

from core.research.ml.metrics.calibration import compare_calibration_methods
from core.research.ml.metrics.evaluation import classification_metrics
from core.research.ml.meta.meta_dataset import _feature_values
from core.research.ml.meta.meta_evaluation import _walk_forward_meta_evaluation
from core.research.ml.meta.meta_models import (
    _fit_meta_model,
    _meta_ensemble_name,
    _normalize_meta_model_type,
)
from core.research.ml.meta.meta_overlay import (
    _overlay_summary,
    _promotion_gate_report,
)


def _compare_meta_learners(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    model_types: list[str],
    threshold: float,
    reduced_exposure: float,
    reduce_when: str,
    random_seed: int,
    calibration_bin_count: int,
    sklearn_n_jobs: int = 1,
    all_rows: list[dict[str, str]] | None = None,
    walk_forward_folds: int = 3,
    promotion_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_features = [_feature_values(row) for row in train_rows]
    train_labels = [int(row["actual_label"]) for row in train_rows]
    holdout_features = [_feature_values(row) for row in holdout_rows]
    holdout_labels = [int(row["actual_label"]) for row in holdout_rows]
    models = []
    for model_type in model_types:
        normalized = _normalize_meta_model_type(model_type)
        try:
            model = _fit_meta_model(
                normalized,
                train_features,
                train_labels,
                random_seed=random_seed,
                sklearn_n_jobs=sklearn_n_jobs,
            )
            train_probabilities = model.predict_proba(train_features)
            probabilities = model.predict_proba(holdout_features)
            predictions = [int(value >= threshold) for value in probabilities]
            calibration_comparison = compare_calibration_methods(
                train_labels,
                train_probabilities,
                holdout_labels,
                probabilities,
                bin_count=calibration_bin_count,
            )
            calibration = _best_calibration_summary(calibration_comparison)
            overlay = _overlay_summary(
                holdout_rows,
                probabilities,
                threshold,
                reduced_exposure,
                reduce_when=reduce_when,
            )
            walk_forward = _walk_forward_meta_evaluation(
                all_rows or train_rows + holdout_rows,
                model_type=normalized,
                fold_count=walk_forward_folds,
                threshold=threshold,
                reduced_exposure=reduced_exposure,
                reduce_when=reduce_when,
                random_seed=random_seed,
                calibration_bin_count=calibration_bin_count,
                sklearn_n_jobs=sklearn_n_jobs,
            )
            promotion_gates = _promotion_gate_report(
                metrics=classification_metrics(holdout_labels, predictions),
                calibration=calibration,
                overlay=overlay,
                walk_forward=walk_forward,
                config=promotion_config or {},
            )
            models.append({
                "model_type": normalized,
                "leaderboard_model": _meta_ensemble_name(normalized),
                "available": True,
                "metrics": classification_metrics(holdout_labels, predictions),
                "calibration": calibration,
                "best_calibration_method": calibration.get("method"),
                "overlay": overlay,
                "walk_forward_summary": walk_forward.get("summary", {}),
                "promotion_gates": promotion_gates,
                "promotion_gate_score": _promotion_gate_score(
                    metrics=classification_metrics(holdout_labels, predictions),
                    calibration=calibration,
                    overlay=overlay,
                    walk_forward=walk_forward.get("summary", {}),
                    promotion_gates=promotion_gates,
                ),
            })
        except Exception as exc:
            if normalized == "lightgbm":
                models.append({
                    "model_type": normalized,
                    "leaderboard_model": _meta_ensemble_name(normalized),
                    "available": False,
                    "reason": str(exc),
                })
                continue
            raise
    available_models = [row for row in models if row.get("available")]
    classifier_ranking = sorted(
        available_models,
        key=lambda row: _classifier_rank_key(row),
    )
    calibration_ranking = sorted(
        available_models,
        key=lambda row: _calibration_rank_key(row),
    )
    overlay_ranking = sorted(
        available_models,
        key=lambda row: _overlay_rank_key(row),
    )
    selections = _meta_selection_summary(
        classifier_ranking,
        calibration_ranking,
        overlay_ranking,
    )
    return {
        "mode": "meta_learner_comparison_research_only",
        "models": models,
        "ranked_model_types": [row["model_type"] for row in classifier_ranking],
        "classifier_ranking": _selection_ranking_rows(classifier_ranking),
        "calibration_ranking": _selection_ranking_rows(calibration_ranking),
        "promotion_gate_ranking": _selection_ranking_rows(overlay_ranking),
        "selections": selections,
        "research_only": True,
        "trading_impact": "none",
    }


def _classifier_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(row.get("metrics", {}).get("balanced_accuracy") or 0.0),
        float(row.get("calibration", {}).get("brier_score") or float("inf")),
        -float(row.get("overlay", {}).get("return_delta") or 0.0),
    )


def _calibration_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("calibration", {}).get("brier_score") or float("inf")),
        float(row.get("calibration", {}).get("expected_calibration_error") or float("inf")),
        -float(row.get("metrics", {}).get("balanced_accuracy") or 0.0),
    )


def _overlay_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(row.get("promotion_gate_score") or 0.0),
        -float(row.get("walk_forward_summary", {}).get("balanced_accuracy") or 0.0),
        float(row.get("calibration", {}).get("brier_score") or float("inf")),
    )


def _meta_selection_summary(
    classifier_ranking: list[dict[str, Any]],
    calibration_ranking: list[dict[str, Any]],
    overlay_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    selections = {}
    if classifier_ranking:
        selections["selected_classifier"] = _selection_payload(
            classifier_ranking[0],
            role="selected_classifier",
            reason=(
                "highest holdout balanced accuracy; not automatically selected "
                "as trading overlay"
            ),
        )
    if calibration_ranking:
        selections["selected_calibrated"] = _selection_payload(
            calibration_ranking[0],
            role="selected_calibrated",
            reason="lowest Brier score with ECE as tie-breaker",
        )
    if overlay_ranking:
        selections["selected_overlay"] = _selection_payload(
            overlay_ranking[0],
            role="selected_overlay",
            reason=(
                "highest promotion-gate utility balancing walk-forward accuracy, "
                "Brier/ECE, overlay return delta, drawdown impact, turnover, "
                "and reduced-exposure days"
            ),
        )
    return selections


def _selection_payload(
    row: dict[str, Any],
    role: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "selection_role": role,
        "selected_model": row.get("leaderboard_model"),
        "model_type": row.get("model_type"),
        "selection_reason": reason,
        "metrics": row.get("metrics", {}),
        "calibration": row.get("calibration", {}),
        "overlay": row.get("overlay", {}),
        "walk_forward_summary": row.get("walk_forward_summary", {}),
        "promotion_gates": row.get("promotion_gates", {}),
        "promotion_gate_score": row.get("promotion_gate_score"),
    }


def _selection_ranking_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "model_type": row.get("model_type"),
            "leaderboard_model": row.get("leaderboard_model"),
            "holdout_balanced_accuracy": row.get("metrics", {}).get(
                "balanced_accuracy"
            ),
            "walk_forward_balanced_accuracy": row.get(
                "walk_forward_summary", {}
            ).get("balanced_accuracy"),
            "brier_score": row.get("calibration", {}).get("brier_score"),
            "expected_calibration_error": row.get("calibration", {}).get(
                "expected_calibration_error"
            ),
            "overlay_return_delta": row.get("overlay", {}).get("return_delta"),
            "max_drawdown_delta": row.get("overlay", {}).get("max_drawdown_delta"),
            "turnover": row.get("overlay", {}).get("overlay_turnover"),
            "reduced_exposure_days": row.get("overlay", {}).get(
                "reduced_exposure_days"
            ),
            "promotion_gate_score": row.get("promotion_gate_score"),
            "promotion_candidate": row.get("promotion_gates", {}).get(
                "promotion_candidate"
            ),
        }
        for index, row in enumerate(rows)
    ]


def _promotion_gate_score(
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    overlay: dict[str, Any],
    walk_forward: dict[str, Any],
    promotion_gates: dict[str, Any],
) -> float:
    gate_checks = promotion_gates.get("checks", {})
    passed_gate_count = sum(
        1 for check in gate_checks.values()
        if isinstance(check, dict) and check.get("passed")
    )
    finite_bonus = 0.25 if gate_checks.get("finite_sanity_check", {}).get("passed") else -1.0
    return (
        2.0 * float(walk_forward.get("balanced_accuracy") or 0.0)
        + 0.5 * float(metrics.get("balanced_accuracy") or 0.0)
        - 1.5 * float(calibration.get("brier_score") or 1.0)
        - 1.0 * float(calibration.get("expected_calibration_error") or 1.0)
        + 2.0 * float(overlay.get("return_delta") or 0.0)
        + 1.0 * float(overlay.get("max_drawdown_delta") or 0.0)
        - 0.01 * float(overlay.get("overlay_turnover") or 0.0)
        - 0.001 * float(overlay.get("reduced_exposure_days") or 0.0)
        + 0.05 * passed_gate_count
        + finite_bonus
    )


def _best_calibration_summary(calibration_comparison: dict[str, Any]) -> dict[str, Any]:
    method = calibration_comparison.get("best_method_by_brier", "raw")
    payload = calibration_comparison.get("methods", {}).get(method, {})
    calibration = dict(payload.get("calibration", {}))
    calibration["method"] = method
    return calibration
