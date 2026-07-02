from __future__ import annotations

from datetime import datetime
from typing import Any

from core.research.ml.replay.canonical_replay_math import _date, _number
from core.research.ml.replay.canonical_replay_metrics import _equity_rows, _summary
from core.research.ml.replay.canonical_replay_types import RESEARCH_METADATA


def _champion_rows(champion_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in (
        champion_audit.get("exact_champion_replay", {}).get("period_rows", [])
        or []
    ):
        if not isinstance(row, dict):
            continue
        period_return = _number(row.get("period_return"))
        if period_return is None:
            continue
        selected_symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
        target_weights = {
            str(symbol): float(weight)
            for symbol, weight in (row.get("target_weights", {}) or {}).items()
            if _number(weight) is not None
        }
        rows.append({
            "candidate_name": "exact_champion_replay",
            "rebalance_date": str(row.get("rebalance_date", "")),
            "outcome_end_date": str(row.get("outcome_end_date", "")),
            "period_return": period_return,
            "exposure": _number(row.get("exposure_target")),
            "turnover": None,
            "cost": None,
            "net_return": period_return,
            "selected_symbols": selected_symbols,
            "target_weights": target_weights,
            "max_position_weight": max(target_weights.values(), default=None),
            "source": "exact_champion_replay",
        })
    return rows
def _selected_optimizer_rows(
    selected_optimizer: dict[str, Any],
    period_by_date: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in selected_optimizer.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        date = str(row.get("rebalance_date", ""))
        period = period_by_date.get(date, {})
        period_return = _number(row.get("period_return"))
        exposure = _number(row.get("exposure"))
        if period_return is None or exposure is None:
            continue
        cost = _number(row.get("cost")) or 0.0
        net_return = _number(row.get("net_return"))
        if net_return is None:
            net_return = period_return * exposure - cost
        target_weights = dict(period.get("target_weights", {}) or {})
        selected_symbols = list(period.get("selected_symbols", []) or [])
        invalid_reason = _optimizer_replay_invalid_reason(
            exposure,
            selected_symbols,
            target_weights,
        )
        rows.append({
            "candidate_name": "selected_bayesian_optimizer_diagnostic_policy",
            "rebalance_date": date,
            "outcome_end_date": str(period.get("outcome_end_date", "")),
            "period_return": period_return,
            "exposure": exposure,
            "turnover": _number(row.get("turnover")),
            "cost": cost,
            "net_return": 0.0 if invalid_reason else net_return,
            "selected_symbols": selected_symbols,
            "target_weights": target_weights,
            "max_position_weight": max(
                (_number(value) or 0.0 for value in target_weights.values()),
                default=None,
            ),
            "replay_valid": invalid_reason is None,
            "replay_invalid_reason": invalid_reason,
            "empty_selection_with_positive_exposure": (
                invalid_reason == "empty_selection_with_positive_exposure"
            ),
            "empty_selection_resolution": "invalidated" if invalid_reason else None,
            "source": "selected_optimizer_exposure_path",
            "score": _number(row.get("score")),
            "predicted_forward_return": _number(
                row.get("predicted_forward_return")
            ),
            "predicted_future_drawdown": _number(
                row.get("predicted_future_drawdown")
            ),
            "predicted_future_volatility": _number(
                row.get("predicted_future_volatility")
            ),
        })
    return rows
def _optimizer_replay_invalid_reason(
    exposure: float,
    selected_symbols: list[Any],
    target_weights: dict[str, Any],
) -> str | None:
    if exposure > 0.0 and not selected_symbols and not target_weights:
        return "empty_selection_with_positive_exposure"
    return None
def _candidate_payload(
    candidate_name: str,
    rows: list[dict[str, Any]],
    *,
    excluded_dates: set[str],
    excluded_symbols: set[str],
    period_return_semantics: str,
    period_cost_semantics: str,
) -> dict[str, Any]:
    filtered_rows = []
    replay_rows = []
    for row in sorted(rows, key=lambda item: str(item["rebalance_date"])):
        reason = _exclusion_reason(row, excluded_dates, excluded_symbols)
        replay_rows.append({**row, "exclusion_reason": reason})
        if reason is None:
            filtered_rows.append(row)
    non_overlap_rows = _non_overlapping_rows(filtered_rows)
    empty_selection_dates = [
        row["rebalance_date"] for row in replay_rows
        if row.get("empty_selection_with_positive_exposure")
    ]
    return {
        "candidate_name": candidate_name,
        "available": bool(rows),
        "period_return_semantics": period_return_semantics,
        "period_cost_semantics": period_cost_semantics,
        "diagnostic_period_grid": _summary(filtered_rows, all_rows=True),
        "canonical_continuous_equity": _summary(non_overlap_rows, all_rows=False),
        "rows": _equity_rows(replay_rows, non_overlap_rows),
        "empty_selection_with_positive_exposure_count": len(empty_selection_dates),
        "empty_selection_with_positive_exposure_dates": empty_selection_dates,
        "empty_selection_resolution": (
            "invalidated" if empty_selection_dates else "unchanged"
        ),
        **RESEARCH_METADATA,
    }
def _exclusion_reason(
    row: dict[str, Any],
    excluded_dates: set[str],
    excluded_symbols: set[str],
) -> str | None:
    invalid_reason = row.get("replay_invalid_reason")
    if invalid_reason:
        return str(invalid_reason)
    date = str(row.get("rebalance_date", ""))
    if date in excluded_dates:
        return "excluded_rebalance_date"
    symbols = {str(symbol) for symbol in row.get("selected_symbols", [])}
    if symbols & excluded_symbols:
        return "excluded_symbol"
    return None
def _non_overlapping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    previous_end: datetime | None = None
    for row in sorted(rows, key=lambda item: str(item["rebalance_date"])):
        start = _date(row.get("rebalance_date"))
        end = _date(row.get("outcome_end_date")) or start
        if start is None:
            continue
        if previous_end is None or start >= previous_end:
            kept.append(row)
            previous_end = end
    return kept
