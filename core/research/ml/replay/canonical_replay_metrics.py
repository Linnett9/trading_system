from __future__ import annotations

from typing import Any

from core.research.ml.replay.canonical_replay_math import (
    _annualized_return,
    _compound_returns,
    _compound_rows,
    _max_drawdown,
    _number,
    _sharpe,
    _sortino,
)
from core.research.ml.replay.canonical_replay_types import RESEARCH_METADATA
from core.research.performance_metrics import calmar_ratio


def _summary(rows: list[dict[str, Any]], *, all_rows: bool) -> dict[str, Any]:
    equity_rows = _compound_rows(rows)
    returns = [float(row["net_return"]) for row in equity_rows]
    total = _compound_returns(returns)
    drawdown = _max_drawdown([1.0] + [float(row["equity"]) for row in equity_rows])
    annualized = _annualized_return(total, equity_rows)
    return {
        "evaluation_mode": (
            "diagnostic_period_grid" if all_rows else "canonical_non_overlapping"
        ),
        "row_count": len(rows),
        "start_date": rows[0]["rebalance_date"] if rows else None,
        "end_date": rows[-1]["rebalance_date"] if rows else None,
        "last_outcome_end_date": rows[-1].get("outcome_end_date") if rows else None,
        "total_return": total,
        "canonical_tradable_total_return": None if all_rows else total,
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "sharpe": _sharpe(returns, equity_rows),
        "sortino": _sortino(returns, equity_rows),
        "calmar": calmar_ratio(annualized if annualized is not None else total, drawdown),
        "turnover": sum(
            float(row["turnover"])
            for row in rows
            if _number(row.get("turnover")) is not None
        ),
        "estimated_transaction_costs": sum(
            float(row["cost"])
            for row in rows
            if _number(row.get("cost")) is not None
        ),
        "largest_positive_period": max(returns, default=None),
        "largest_negative_period": min(returns, default=None),
    }
def _equity_rows(
    rows: list[dict[str, Any]],
    non_overlap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    non_overlap_dates = {row["rebalance_date"] for row in non_overlap_rows}
    compounded = {
        row["rebalance_date"]: row
        for row in _compound_rows([
            row for row in rows
            if row.get("exclusion_reason") is None
            and row["rebalance_date"] in non_overlap_dates
        ])
    }
    output = []
    for row in rows:
        date = row["rebalance_date"]
        compound = compounded.get(date, {})
        output.append({
            "candidate_name": row["candidate_name"],
            "rebalance_date": date,
            "outcome_end_date": row.get("outcome_end_date"),
            "included_in_canonical": (
                date in non_overlap_dates and row.get("exclusion_reason") is None
            ),
            "exclusion_reason": row.get("exclusion_reason"),
            "period_return": row.get("period_return"),
            "exposure": row.get("exposure"),
            "turnover": row.get("turnover"),
            "cost": row.get("cost"),
            "net_return": row.get("net_return"),
            "equity": compound.get("equity"),
            "drawdown": compound.get("drawdown"),
            "selected_symbols": row.get("selected_symbols", []),
            "target_weights": row.get("target_weights", {}),
            "max_position_weight": row.get("max_position_weight"),
            "replay_valid": row.get("replay_valid", True),
            "replay_invalid_reason": row.get("replay_invalid_reason"),
            "empty_selection_with_positive_exposure": bool(
                row.get("empty_selection_with_positive_exposure", False)
            ),
            "empty_selection_resolution": row.get("empty_selection_resolution"),
            "source": row.get("source"),
            **RESEARCH_METADATA,
        })
    return output
