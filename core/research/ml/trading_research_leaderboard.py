from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
RANKING_BASIS = [
    "total_return_desc",
    "max_drawdown_asc",
    "sharpe_desc",
    "sortino_desc",
    "calmar_desc",
    "turnover_asc",
    "estimated_transaction_costs_asc",
]
NOTICE = "Research only. Trading impact: none. Production validated: false."


@dataclass(frozen=True)
class TradingResearchLeaderboardPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


def write_trading_research_leaderboard(
    output_dir: Path,
    classification_leaderboard_path: Path,
    allocation_comparison_path: Path,
    optimizer_results_path: Path,
    auxiliary_metrics_path: Path,
) -> TradingResearchLeaderboardPaths:
    """Combine research reports without changing their source selection logic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classification = _read_json(classification_leaderboard_path)
    allocation = _read_json(allocation_comparison_path)
    optimizer = _read_json(optimizer_results_path)
    auxiliary = _read_json(auxiliary_metrics_path)

    classification_rows = _classification_rows(classification)
    trading_rows = [
        *_classification_trading_rows(classification_rows),
        *_allocation_trading_rows(allocation),
    ]
    optimizer_row = _optimizer_trading_row(optimizer)
    if optimizer_row is not None:
        trading_rows.append(optimizer_row)

    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(sorted(trading_rows, key=_ranking_key), start=1)
    ]
    payload = {
        "mode": "trading_research_leaderboard",
        "ranking_basis": RANKING_BASIS,
        "classification_metrics_role": "diagnostics_only",
        "leaderboard": ranked_rows,
        "classification_diagnostics": _classification_diagnostics(
            classification_rows
        ),
        "meta_auxiliary_forecast_metrics": auxiliary.get("targets", {}),
        "meta_auxiliary_available_targets": auxiliary.get(
            "available_targets", []
        ),
        "source_artifacts": {
            "base_and_meta_classification": str(classification_leaderboard_path),
            "allocation_v2": str(allocation_comparison_path),
            "allocation_optimizer": str(optimizer_results_path),
            "meta_auxiliary": str(auxiliary_metrics_path),
        },
        "optimizer_status": {
            "sampler_requested": optimizer.get("sampler_requested"),
            "sampler_used": optimizer.get("sampler_used"),
            "fallback_reason": optimizer.get("fallback_reason"),
            "skip_reason": optimizer.get("skip_reason"),
        },
        **RESEARCH_METADATA,
    }

    paths = TradingResearchLeaderboardPaths(
        csv_path=output_dir / "trading_research_leaderboard.csv",
        json_path=output_dir / "trading_research_leaderboard.json",
        markdown_path=output_dir / "trading_research_leaderboard.md",
    )
    paths.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(paths.csv_path, ranked_rows)
    paths.markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return paths


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
    return _trading_row(
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
        },
    )


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


def _ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        _descending(row.get("total_return")),
        _ascending(row.get("max_drawdown")),
        _descending(row.get("sharpe")),
        _descending(row.get("sortino")),
        _descending(row.get("calmar")),
        _ascending(row.get("turnover")),
        _ascending(row.get("estimated_transaction_costs")),
    )


def _descending(value: Any) -> float:
    number = _number(value)
    return math.inf if number is None else -number


def _ascending(value: Any) -> float:
    number = _number(value)
    return math.inf if number is None else number


def _drawdown_magnitude(value: Any) -> float | None:
    number = _number(value)
    return abs(number) if number is not None else None


def _first_number(payload: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "entity_name",
        "entity_type",
        "source",
        "selection_role",
        "total_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "turnover",
        "estimated_transaction_costs",
        "balanced_accuracy",
        "walk_forward_balanced_accuracy",
        "brier_score",
        "expected_calibration_error",
        "classification_metrics_role",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row.get(name) for name in fieldnames} for row in rows
        )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trading Research Leaderboard",
        "",
        "Trading outcomes determine rank. Classification metrics are diagnostics only.",
        "",
        "|rank|candidate|type|total return|max drawdown|Sharpe|Sortino|Calmar|turnover|costs|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            "|{rank}|{entity_name}|{entity_type}|{total_return}|{max_drawdown}|"
            "{sharpe}|{sortino}|{calmar}|{turnover}|{costs}|".format(
                rank=row["rank"],
                entity_name=row["entity_name"],
                entity_type=row["entity_type"],
                total_return=_format(row.get("total_return")),
                max_drawdown=_format(row.get("max_drawdown")),
                sharpe=_format(row.get("sharpe")),
                sortino=_format(row.get("sortino")),
                calmar=_format(row.get("calmar")),
                turnover=_format(row.get("turnover")),
                costs=_format(row.get("estimated_transaction_costs")),
            )
        )

    lines.extend([
        "",
        "## Classification Diagnostics",
        "",
        "|model|role|balanced accuracy|walk-forward balanced accuracy|Brier|ECE|",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in payload["classification_diagnostics"]:
        lines.append(
            "|{model}|{role}|{balanced}|{walk_forward}|{brier}|{ece}|".format(
                model=row.get("model") or "",
                role=row.get("selection_role") or "",
                balanced=_format(row.get("holdout_balanced_accuracy")),
                walk_forward=_format(
                    row.get("walk_forward_balanced_accuracy")
                ),
                brier=_format(row.get("brier_score")),
                ece=_format(row.get("expected_calibration_error")),
            )
        )

    lines.extend([
        "",
        "## Meta Auxiliary Forecast Diagnostics",
        "",
        "|target|available|MAE|RMSE|Pearson|Spearman|directional accuracy|",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for target, metrics in payload["meta_auxiliary_forecast_metrics"].items():
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "|{target}|{available}|{mae}|{rmse}|{pearson}|{spearman}|{directional}|".format(
                target=target,
                available=metrics.get("available", False),
                mae=_format(metrics.get("mae")),
                rmse=_format(metrics.get("rmse")),
                pearson=_format(metrics.get("pearson_correlation")),
                spearman=_format(metrics.get("spearman_correlation")),
                directional=_format(metrics.get("directional_accuracy")),
            )
        )
    lines.extend(["", NOTICE, ""])
    return "\n".join(lines)


def _format(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.4f}"
