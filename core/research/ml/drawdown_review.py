from __future__ import annotations

import json
from pathlib import Path


COMPARISON_FIELDS = (
    "selection_count",
    "exposure_target",
    "cash_weight",
    "average_rank_score",
    "selected_score_dispersion",
    "largest_weight",
    "selection_weight_herfindahl",
    "selection_overlap_with_prior",
    "selection_average_pairwise_correlation_63d",
    "selection_sector_concentration",
    "selection_sector_coverage",
    "replacements",
    "spy_distance_sma_200",
    "spy_realized_volatility_21d",
    "spy_volatility_ratio_21d_63d",
    "spy_max_drawdown_63d",
    "breadth_above_sma_200",
    "breadth_change_since_last_rebalance",
    "recent_champion_return_2_rebalances",
    "recent_champion_excess_return_2_rebalances",
)


def write_drawdown_event_review(
    path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    path.write_text(
        json.dumps(build_drawdown_event_review(rows), indent=2),
        encoding="utf-8",
    )


def build_drawdown_event_review(rows: list[dict[str, float | str]]) -> dict:
    event_rows = [row for row in rows if int(row["drawdown_event"]) == 1]
    normal_rows = [row for row in rows if int(row["drawdown_event"]) == 0]
    return {
        "outcome": "future champion drawdown <= -10% within the label horizon",
        "sample_count": len(rows),
        "drawdown_event_count": len(event_rows),
        "non_event_count": len(normal_rows),
        "event_cases": [_event_case(row) for row in event_rows],
        "event_vs_non_event_means": {
            field: _cohort_comparison(field, event_rows, normal_rows)
            for field in COMPARISON_FIELDS
        },
        "chronological_event_rate": _chronological_event_rate(rows),
        "research_only": True,
    }


def _event_case(row: dict[str, float | str]) -> dict[str, float | str]:
    fields = (
        "rebalance_date", "outcome_end_date", "selected_symbols", "regime_label",
        "selection_count", "exposure_target", "largest_weight",
        "selection_weight_herfindahl", "selection_overlap_with_prior",
        "selection_average_pairwise_correlation_63d",
        "selection_sector_concentration", "selection_sector_coverage", "replacements",
        "spy_distance_sma_200", "spy_realized_volatility_21d",
        "spy_volatility_ratio_21d_63d", "spy_max_drawdown_63d",
        "breadth_above_sma_200", "breadth_change_since_last_rebalance",
        "recent_champion_excess_return_2_rebalances",
        "champion_return_next_period", "benchmark_return_next_period",
        "champion_excess_return", "future_max_drawdown",
    )
    return {field: row.get(field) for field in fields}


def _cohort_comparison(
    field: str,
    event_rows: list[dict[str, float | str]],
    normal_rows: list[dict[str, float | str]],
) -> dict[str, float | None]:
    event_mean = _mean(field, event_rows)
    normal_mean = _mean(field, normal_rows)
    return {
        "event_mean": event_mean,
        "non_event_mean": normal_mean,
        "difference": (
            event_mean - normal_mean
            if event_mean is not None and normal_mean is not None else None
        ),
    }


def _chronological_event_rate(rows: list[dict[str, float | str]]) -> list[dict]:
    if not rows:
        return []
    window_size = max(1, len(rows) // 3)
    windows = []
    for start in range(0, len(rows), window_size):
        window = rows[start:start + window_size]
        event_count = sum(int(row["drawdown_event"]) for row in window)
        windows.append({
            "start_date": window[0]["rebalance_date"],
            "end_date": window[-1]["rebalance_date"],
            "rebalance_count": len(window),
            "drawdown_event_count": event_count,
            "drawdown_event_rate": event_count / len(window),
        })
    return windows


def _mean(field: str, rows: list[dict[str, float | str]]) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None
