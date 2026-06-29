from __future__ import annotations

from typing import Any

from core.research.framework.ranking import CrossSectionalRankingEvaluator
from core.research.ml.stock_level_benchmark_data import _sort_value
from core.research.ml.stock_level_benchmark_types import (
    BASELINE_COLUMNS,
    PREDICTION_PREFIX,
    TARGET_COLUMN,
)


def _build_leaderboard(
    predictions: list[dict[str, Any]],
    model_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    summaries = [
        _evaluate_signal(
            predictions,
            name,
            f"{PREDICTION_PREFIX}{name}",
            kind="ml_model",
        )
        for name in model_names
    ]
    summaries.extend(
        _evaluate_signal(predictions, name, column, kind="baseline")
        for name, column in BASELINE_COLUMNS.items()
    )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -_sort_value(row["mean_spearman_ic"]),
            -_sort_value(row["top_minus_bottom_spread"]),
            row["name"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _evaluate_signal(
    rows: list[dict[str, Any]],
    name: str,
    signal_column: str,
    *,
    kind: str,
) -> dict[str, Any]:
    return CrossSectionalRankingEvaluator(target_column=TARGET_COLUMN).evaluate(
        rows,
        name=name,
        signal_column=signal_column,
        kind=kind,
    )


def _compare_to_momentum(
    best_ml: dict[str, Any] | None,
    momentum: dict[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "mean_pearson_ic",
        "mean_spearman_ic",
        "top_decile_return",
        "bottom_decile_return",
        "top_minus_bottom_spread",
        "top_decile_hit_rate",
        "risk_adjusted_spread",
        "spread_sharpe",
    )
    deltas = {
        name: (
            None
            if best_ml is None
            or best_ml.get(name) is None
            or momentum.get(name) is None
            else float(best_ml[name]) - float(momentum[name])
        )
        for name in metric_names
    }
    beats = bool(
        best_ml
        and deltas["mean_spearman_ic"] is not None
        and deltas["top_minus_bottom_spread"] is not None
        and deltas["mean_spearman_ic"] > 0.0
        and deltas["top_minus_bottom_spread"] > 0.0
    )
    return {
        "model": best_ml["name"] if best_ml else None,
        "momentum_baseline": momentum["name"],
        "metric_deltas_ml_minus_momentum": deltas,
        "beats_momentum_120d": beats,
    }
