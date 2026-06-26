from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_leaderboard(
    json_path: Path,
    markdown_path: Path,
    source_dirs: list[Path],
    meta_metrics: dict[str, Any],
    meta_calibration: dict[str, Any],
    meta_overlay: dict[str, Any],
    meta_model_name: str = "meta_ensemble_logistic",
    meta_walk_forward: dict[str, Any] | None = None,
    promotion_gates: dict[str, Any] | None = None,
    meta_selections: dict[str, Any] | None = None,
) -> None:
    meta_walk_forward = meta_walk_forward or {}
    promotion_gates = promotion_gates or {}
    meta_selections = meta_selections or {}
    rows = [_champion_row()]
    for source_dir in source_dirs:
        row = _source_row(source_dir)
        if row is not None:
            rows.append(row)
    rows.append({
        "model": meta_model_name,
        "holdout_accuracy": meta_metrics.get("accuracy"),
        "holdout_balanced_accuracy": meta_metrics.get("balanced_accuracy"),
        "walk_forward_balanced_accuracy": meta_walk_forward.get("balanced_accuracy"),
        "calibration_method": meta_calibration.get("method", "raw"),
        "brier_score": meta_calibration.get("brier_score"),
        "expected_calibration_error": meta_calibration.get(
            "expected_calibration_error"
        ),
        "brier_skill_vs_base_rate": meta_calibration.get("brier_skill_score"),
        "overlay_start_date": meta_overlay.get("overlay_start_date"),
        "overlay_end_date": meta_overlay.get("overlay_end_date"),
        "overlay_sample_count": meta_overlay.get("overlay_sample_count"),
        "overlay_baseline_return": meta_overlay.get("overlay_baseline_return"),
        "overlay_adjusted_return": meta_overlay.get("overlay_adjusted_return"),
        "overlay_total_return": meta_overlay.get("overlay_total_return"),
        "overlay_return_delta": meta_overlay.get("return_delta"),
        "overlay_max_drawdown": meta_overlay.get("overlay_max_drawdown"),
        "overlay_max_drawdown_improvement": meta_overlay.get("max_drawdown_delta"),
        "turnover": meta_overlay.get("overlay_turnover"),
        "reduced_exposure_days": meta_overlay.get("reduced_exposure_days"),
        "promotion_candidate": promotion_gates.get("promotion_candidate"),
        "finite_sanity_check": (
            promotion_gates.get("checks", {})
            .get("finite_sanity_check", {})
            .get("passed")
        ),
        "selection_role": "configured_meta_model",
        "selected_model": meta_model_name,
        "selection_reason": "configured ml.meta_model_type baseline row",
        "promotion_gate_score": promotion_gates.get("promotion_gate_score"),
        "trading_impact": "none",
        "production_validated": False,
    })
    for role in (
        "selected_classifier",
        "selected_calibrated",
        "selected_overlay",
    ):
        selection = meta_selections.get(role)
        if selection:
            rows.append(_meta_selection_row(selection))
    payload = {
        "leaderboard": rows,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(rows), encoding="utf-8")


def _champion_row() -> dict[str, Any]:
    return {
        "model": "champion_baseline",
        "holdout_accuracy": None,
        "holdout_balanced_accuracy": None,
        "walk_forward_balanced_accuracy": None,
        "calibration_method": None,
        "brier_score": None,
        "expected_calibration_error": None,
        "brier_skill_vs_base_rate": None,
        "overlay_start_date": None,
        "overlay_end_date": None,
        "overlay_sample_count": None,
        "overlay_baseline_return": None,
        "overlay_adjusted_return": None,
        "overlay_total_return": None,
        "overlay_return_delta": 0.0,
        "overlay_max_drawdown": None,
        "overlay_max_drawdown_improvement": 0.0,
        "turnover": 0.0,
        "reduced_exposure_days": 0,
        "promotion_candidate": False,
        "finite_sanity_check": True,
        "selection_role": None,
        "selected_model": None,
        "selection_reason": None,
        "promotion_gate_score": None,
        "trading_impact": "none",
        "production_validated": False,
    }


def _source_row(source_dir: Path) -> dict[str, Any] | None:
    metrics_path = source_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = metrics_payload.get("metrics", {})
    calibration = _read_json(source_dir / "probability_calibration.json")
    calibration_summary = _calibration_summary(calibration)
    calibrated_comparison = _read_json(
        source_dir / "calibrated_probability_calibration.json"
    )
    overlay = _read_json(source_dir / "holdout_shadow_overlay.json")
    result = overlay.get("result") or {}
    return {
        "model": metrics_payload.get("model_type", source_dir.name),
        "holdout_accuracy": metrics.get("accuracy"),
        "holdout_balanced_accuracy": metrics.get("balanced_accuracy"),
        "walk_forward_balanced_accuracy": _walk_forward_balanced_accuracy(source_dir),
        "calibration_method": calibrated_comparison.get("best_method_by_brier", "raw"),
        "brier_score": calibration_summary.get("brier_score"),
        "expected_calibration_error": calibration_summary.get(
            "expected_calibration_error"
        ),
        "brier_skill_vs_base_rate": calibration_summary.get("brier_skill_score"),
        "overlay_start_date": result.get("overlay_start_date"),
        "overlay_end_date": result.get("overlay_end_date"),
        "overlay_sample_count": result.get("overlay_sample_count"),
        "overlay_baseline_return": result.get(
            "overlay_baseline_return",
            result.get("base_total_return"),
        ),
        "overlay_adjusted_return": result.get(
            "overlay_adjusted_return",
            result.get("overlay_total_return"),
        ),
        "overlay_total_return": result.get("overlay_total_return"),
        "overlay_return_delta": (
            result.get("overlay_total_return") - result.get("base_total_return")
            if result.get("overlay_total_return") is not None
            and result.get("base_total_return") is not None
            else None
        ),
        "overlay_max_drawdown": result.get("overlay_max_drawdown"),
        "overlay_max_drawdown_improvement": (
            result.get("overlay_max_drawdown") - result.get("base_max_drawdown")
            if result.get("overlay_max_drawdown") is not None
            and result.get("base_max_drawdown") is not None
            else None
        ),
        "turnover": result.get("overlay_turnover"),
        "reduced_exposure_days": result.get("reduced_exposure_days"),
        "promotion_candidate": False,
        "finite_sanity_check": None,
        "selection_role": None,
        "selected_model": None,
        "selection_reason": None,
        "promotion_gate_score": None,
        "trading_impact": "none",
        "production_validated": False,
    }


def _meta_selection_row(selection: dict[str, Any]) -> dict[str, Any]:
    metrics = selection.get("metrics", {})
    calibration = selection.get("calibration", {})
    overlay = selection.get("overlay", {})
    walk_forward = selection.get("walk_forward_summary", {})
    gates = selection.get("promotion_gates", {})
    return {
        "model": selection.get("selection_role"),
        "holdout_accuracy": metrics.get("accuracy"),
        "holdout_balanced_accuracy": metrics.get("balanced_accuracy"),
        "walk_forward_balanced_accuracy": walk_forward.get("balanced_accuracy"),
        "calibration_method": calibration.get("method", "raw"),
        "brier_score": calibration.get("brier_score"),
        "expected_calibration_error": calibration.get(
            "expected_calibration_error"
        ),
        "brier_skill_vs_base_rate": calibration.get("brier_skill_score"),
        "overlay_start_date": overlay.get("overlay_start_date"),
        "overlay_end_date": overlay.get("overlay_end_date"),
        "overlay_sample_count": overlay.get("overlay_sample_count"),
        "overlay_baseline_return": overlay.get("overlay_baseline_return"),
        "overlay_adjusted_return": overlay.get("overlay_adjusted_return"),
        "overlay_total_return": overlay.get("overlay_total_return"),
        "overlay_return_delta": overlay.get("return_delta"),
        "overlay_max_drawdown": overlay.get("overlay_max_drawdown"),
        "overlay_max_drawdown_improvement": overlay.get("max_drawdown_delta"),
        "turnover": overlay.get("overlay_turnover"),
        "reduced_exposure_days": overlay.get("reduced_exposure_days"),
        "promotion_candidate": gates.get("promotion_candidate"),
        "finite_sanity_check": (
            gates.get("checks", {})
            .get("finite_sanity_check", {})
            .get("passed")
        ),
        "selection_role": selection.get("selection_role"),
        "selected_model": selection.get("selected_model"),
        "selection_reason": selection.get("selection_reason"),
        "promotion_gate_score": selection.get("promotion_gate_score"),
        "trading_impact": "none",
        "production_validated": False,
    }


def _walk_forward_balanced_accuracy(source_dir: Path) -> float | None:
    payload = _read_json(source_dir / "walk_forward_metrics.json")
    values = [
        fold.get("metrics", {}).get("balanced_accuracy")
        for fold in payload.get("folds", [])
        if fold.get("metrics", {}).get("balanced_accuracy") is not None
    ]
    return sum(values) / len(values) if values else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if "calibration" in payload and isinstance(payload["calibration"], dict):
        return payload["calibration"]
    return payload


def _markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model",
        "selection_role",
        "selected_model",
        "holdout_balanced_accuracy",
        "walk_forward_balanced_accuracy",
        "calibration_method",
        "brier_score",
        "expected_calibration_error",
        "overlay_start_date",
        "overlay_end_date",
        "overlay_sample_count",
        "overlay_baseline_return",
        "overlay_adjusted_return",
        "overlay_return_delta",
        "overlay_max_drawdown_improvement",
        "turnover",
        "reduced_exposure_days",
        "promotion_gate_score",
        "promotion_candidate",
        "finite_sanity_check",
        "selection_reason",
    ]
    lines = [
        "# Regime Transformer Meta Ensemble Leaderboard",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_format(row.get(header)) for header in headers) + "|")
    lines.append("")
    lines.append("Research only. Trading impact: none. Production validated: false.")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
