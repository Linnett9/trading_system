from __future__ import annotations

from typing import Any

from core.research.ml.trading_research_leaderboard_math import (
    _drawdown_magnitude,
    _first_number,
    _number,
)
from core.research.ml.trading_research_leaderboard_types import (
    RESEARCH_METADATA,
)


def _classification_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("leaderboard", [])
    return [row for row in rows if isinstance(row, dict)]


def _classification_trading_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trading_rows = []
    for row in rows:
        role = row.get("selection_role")
        if role in {"selected_classifier", "selected_calibrated", "selected_overlay"}:
            continue
        total_return = _first_number(
            row,
            "overlay_compounded_return",
            "overlay_total_return",
            "overlay_adjusted_return",
        )
        if total_return is None:
            continue
        model = str(row.get("model") or "unknown_model")
        entity_type = (
            "meta_ensemble"
            if model.startswith("meta_ensemble") or role == "configured_meta_model"
            else "base_model"
        )
        trading_rows.append(_trading_row(
            entity_name=model,
            entity_type=entity_type,
            source="classification_leaderboard",
            metrics={
                "total_return": total_return,
                "max_drawdown": _drawdown_magnitude(
                    row.get("overlay_max_drawdown")
                ),
                "turnover": _number(row.get("turnover")),
            },
            classification=row,
            selection_role=role,
        ))
    return trading_rows


def _allocation_trading_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for collection, entity_type in (
        ("policies", "allocation_policy"),
        ("baselines", "allocation_baseline"),
    ):
        candidates = payload.get(collection, [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("available") is False or candidate.get("skip_reason"):
                continue
            if _number(candidate.get("total_return")) is None:
                continue
            rows.append(_trading_row(
                entity_name=str(candidate.get("policy_name") or "unknown_policy"),
                entity_type=entity_type,
                source="allocation_v2",
                metrics=candidate,
                classification=candidate,
                selection_role=None,
            ))
    return rows


def _optimizer_trading_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    selected = payload.get("selected_policy")
    if not isinstance(selected, dict):
        return None
    metrics = selected.get("frozen_holdout_metrics") or selected.get(
        "holdout_metrics"
    )
    if not isinstance(metrics, dict) or _number(metrics.get("total_return")) is None:
        return None
    objective_metrics = selected.get("holdout_objective_metrics") or selected.get(
        "objective_metrics"
    )
    if not isinstance(objective_metrics, dict):
        objective_metrics = {}
    row = _trading_row(
        entity_name=str(
            metrics.get("policy_name")
            or selected.get("candidate_id")
            or "allocation_optimizer_selected"
        ),
        entity_type="allocation_optimizer",
        source="allocation_optimizer",
        metrics=metrics,
        classification=metrics,
        selection_role="frozen_holdout_evaluation",
        detail={
            "candidate_id": selected.get("candidate_id"),
            "objective_value": _number(
                selected.get("objective_value", selected.get("objective"))
            ),
            "sampler_requested": payload.get("sampler_requested"),
            "sampler_used": payload.get("sampler_used"),
            "optimizer_objective_mode": payload.get("objective_mode"),
            "objective_metrics": objective_metrics,
        },
    )
    row.update({
        "optimizer_objective_mode": payload.get("objective_mode"),
        "canonical_non_overlap_return": _number(
            objective_metrics.get("canonical_non_overlap_return")
        ),
        "anomaly_adjusted_canonical_return": _number(
            objective_metrics.get("anomaly_adjusted_canonical_return")
        ),
        "anomaly_dependency_ratio": _number(
            objective_metrics.get("anomaly_dependency_ratio")
        ),
        "robustness_adjusted_score": _number(
            objective_metrics.get("robustness_adjusted_score")
        ),
        "selected_by_robustness_objective": bool(
            selected.get("selected_by_robustness_objective", False)
        ),
    })
    return row


def _trading_row(
    *,
    entity_name: str,
    entity_type: str,
    source: str,
    metrics: dict[str, Any],
    classification: dict[str, Any],
    selection_role: Any,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "source": source,
        "selection_role": selection_role,
        "total_return": _number(metrics.get("total_return")),
        "max_drawdown": _drawdown_magnitude(metrics.get("max_drawdown")),
        "sharpe": _number(metrics.get("sharpe")),
        "sortino": _number(metrics.get("sortino")),
        "calmar": _number(metrics.get("calmar")),
        "turnover": _number(metrics.get("turnover")),
        "estimated_transaction_costs": _number(
            metrics.get("estimated_transaction_costs")
        ),
        "balanced_accuracy": _first_number(
            classification,
            "balanced_accuracy",
            "holdout_balanced_accuracy",
        ),
        "walk_forward_balanced_accuracy": _number(
            classification.get("walk_forward_balanced_accuracy")
        ),
        "brier_score": _number(classification.get("brier_score")),
        "expected_calibration_error": _number(
            classification.get("expected_calibration_error")
        ),
        "classification_metrics_role": "diagnostics_only",
        "diagnostic_period_grid_return": _number(metrics.get("total_return")),
        "canonical_continuous_return": None,
        "canonical_non_overlap_return": None,
        "canonical_tradable_total_return": None,
        "paper_tradable_equity_return": None,
        "anomaly_adjusted_return": None,
        "anomaly_adjusted_canonical_return": None,
        "anomaly_dependency_ratio": None,
        "robustness_adjusted_score": None,
        "selected_by_robustness_objective": False,
        "optimizer_objective_mode": None,
        "benchmark_relative_pass": None,
        "tradability_validation_pass": None,
        "promotion_candidate_status": None,
        "profit_concentration_ratio": None,
        "turnover_after_hysteresis": None,
        "detail": detail or {},
        **RESEARCH_METADATA,
    }


def _classification_diagnostics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "model": row.get("model"),
            "selection_role": row.get("selection_role"),
            "selected_model": row.get("selected_model"),
            "holdout_accuracy": _number(row.get("holdout_accuracy")),
            "holdout_balanced_accuracy": _number(
                row.get("holdout_balanced_accuracy")
            ),
            "walk_forward_balanced_accuracy": _number(
                row.get("walk_forward_balanced_accuracy")
            ),
            "calibration_method": row.get("calibration_method"),
            "brier_score": _number(row.get("brier_score")),
            "expected_calibration_error": _number(
                row.get("expected_calibration_error")
            ),
        }
        for row in rows
    ]


def _canonical_only_rows(
    canonical: dict[str, Any],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_names = {row.get("entity_name") for row in existing_rows}
    rows = []
    for name, candidate in canonical.get("candidates", {}).items():
        if name in existing_names:
            continue
        diagnostic = candidate.get("diagnostic_period_grid", {})
        canonical_metrics = candidate.get("canonical_continuous_equity", {})
        total_return = _number(diagnostic.get("total_return"))
        if total_return is None:
            total_return = _number(canonical_metrics.get("total_return"))
        if total_return is None:
            continue
        rows.append(_trading_row(
            entity_name=str(name),
            entity_type="canonical_replay",
            source="canonical_continuous_equity_replay",
            metrics={
                "total_return": total_return,
                "max_drawdown": canonical_metrics.get("max_drawdown"),
                "sharpe": canonical_metrics.get("sharpe"),
                "sortino": canonical_metrics.get("sortino"),
                "calmar": canonical_metrics.get("calmar"),
                "turnover": canonical_metrics.get("turnover"),
                "estimated_transaction_costs": canonical_metrics.get(
                    "estimated_transaction_costs"
                ),
            },
            classification={},
            selection_role="canonical_non_overlapping_replay",
        ))
    return rows


def _benchmark_validation_rows(
    validation: dict[str, Any],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_names = {row.get("entity_name") for row in existing_rows}
    rows = []
    for candidate in validation.get("candidates", []) or []:
        name = str(candidate.get("candidate_name", ""))
        if not name or name in existing_names or not candidate.get("available"):
            continue
        row = _trading_row(
            entity_name=name,
            entity_type="tradable_benchmark",
            source="benchmark_relative_validation",
            metrics={
                "total_return": candidate.get("canonical_non_overlap_return"),
                "max_drawdown": candidate.get("max_drawdown"),
                "sharpe": candidate.get("sharpe"),
                "sortino": candidate.get("sortino"),
                "turnover": candidate.get("turnover"),
            },
            classification={},
            selection_role="benchmark_relative_validation",
        )
        row["canonical_continuous_return"] = _number(
            candidate.get("canonical_non_overlap_return")
        )
        row["canonical_non_overlap_return"] = row[
            "canonical_continuous_return"
        ]
        rows.append(row)
    return rows
