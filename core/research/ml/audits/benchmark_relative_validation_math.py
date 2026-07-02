from __future__ import annotations

import math
from datetime import datetime
from statistics import mean
from typing import Any


def _compound(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0

def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    output = [equity]
    for value in returns:
        equity *= 1.0 + value
        output.append(equity)
    return output

def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown

def _sharpe(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if len(returns) < 2:
        return 0.0
    average = mean(returns)
    variance = sum((value - average) ** 2 for value in returns) / len(returns)
    return average / math.sqrt(variance) * math.sqrt(_periods_per_year(rows)) if variance > 0 else 0.0

def _sortino(returns: list[float], rows: list[dict[str, Any]]) -> float:
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(downside)) if downside else 0.0
    return mean(returns) / downside_deviation * math.sqrt(_periods_per_year(rows)) if downside_deviation > 0 else 0.0

def _periods_per_year(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    start = datetime.fromisoformat(str(rows[0]["rebalance_date"])[:10])
    end = datetime.fromisoformat(str(rows[-1]["outcome_end_date"])[:10])
    years = max((end - start).days / 365.25, 1.0 / 365.25)
    return len(rows) / years

def _return(row: dict[str, Any] | None) -> float | None:
    return _number((row or {}).get("canonical_non_overlap_return"))

def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
