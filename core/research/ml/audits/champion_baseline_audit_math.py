from __future__ import annotations

import math
from datetime import datetime
from statistics import mean
from typing import Any


def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    curve = []
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    return curve


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
    try:
        start = datetime.fromisoformat(rows[0]["rebalance_date"][:10])
        end = datetime.fromisoformat(rows[-1]["outcome_end_date"][:10])
    except ValueError:
        return None
    elapsed_days = (end - start).days
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def _observed_periods_per_year(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 1.0
    try:
        start = datetime.fromisoformat(rows[0]["rebalance_date"][:10])
        end = datetime.fromisoformat(rows[-1]["outcome_end_date"][:10])
    except ValueError:
        return 1.0
    elapsed_days = (end - start).days
    if elapsed_days <= 0:
        return 1.0
    return max(1.0, len(rows) * 365.25 / elapsed_days)


def _sharpe(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    average = mean(returns)
    std = math.sqrt(mean((value - average) ** 2 for value in returns))
    if std == 0.0:
        return 0.0
    return average / std * math.sqrt(_observed_periods_per_year(rows))


def _sortino(returns: list[float], rows: list[dict[str, Any]]) -> float:
    if not returns:
        return 0.0
    downside = [min(0.0, value) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if downside_deviation == 0.0:
        return 0.0
    return mean(returns) / downside_deviation * math.sqrt(
        _observed_periods_per_year(rows)
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
