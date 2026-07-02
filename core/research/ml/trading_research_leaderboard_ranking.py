from __future__ import annotations

from typing import Any

from core.research.ml.trading_research_leaderboard_math import (
    _ascending,
    _descending,
    _number,
)


def _ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    canonical_return = _number(row.get("canonical_continuous_return"))
    canonical_required = bool(row.get("canonical_ranking_available"))
    ranking_return = (
        canonical_return
        if canonical_return is not None
        else _number(row.get("total_return"))
    )
    return (
        1.0 if canonical_required and canonical_return is None else 0.0,
        _descending(ranking_return),
        _ascending(row.get("max_drawdown")),
        _descending(row.get("sharpe")),
        _descending(row.get("sortino")),
        _descending(row.get("calmar")),
        _ascending(row.get("turnover")),
        _ascending(row.get("estimated_transaction_costs")),
    )
