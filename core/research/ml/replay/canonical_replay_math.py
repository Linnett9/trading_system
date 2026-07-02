from __future__ import annotations

import math
from datetime import datetime
from statistics import mean
from typing import Any


def _compound_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    equity = 1.0
    peak = 1.0
    output = []
    for row in rows:
        net_return = float(row.get("net_return") or 0.0)
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        output.append({
            **row,
            "equity": equity,
            "drawdown": (peak - equity) / peak if peak else 0.0,
        })
    return output
def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0
def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown
def _annualized_return(total_return: float, rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2 or total_return <= -1.0:
        return None
    start = _date(rows[0].get("rebalance_date"))
    end = _date(rows[-1].get("outcome_end_date")) or _date(rows[-1].get("rebalance_date"))
    if start is None or end is None:
        return None
    elapsed_days = (end - start).days
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0
def _periods_per_year(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    start = _date(rows[0].get("rebalance_date"))
    end = _date(rows[-1].get("outcome_end_date")) or _date(rows[-1].get("rebalance_date"))
    if start is None or end is None:
        return 1.0
    elapsed_days = (end - start).days
    return max(1.0, len(rows) * 365.25 / elapsed_days) if elapsed_days > 0 else 1.0
def _sharpe(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    average = mean(returns)
    variance = mean((value - average) ** 2 for value in returns)
    if variance <= 0:
        return 0.0
    return average / math.sqrt(variance) * math.sqrt(_periods_per_year(rows))
def _sortino(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    downside = [min(0.0, value) for value in returns]
    deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if deviation <= 0:
        return 0.0
    return mean(returns) / deviation * math.sqrt(_periods_per_year(rows))
def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None
def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
