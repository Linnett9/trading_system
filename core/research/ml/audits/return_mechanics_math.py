from __future__ import annotations

import math
from datetime import datetime
from statistics import mean, median
from typing import Any


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_float(value: Any) -> float:
    number = _number(value)
    if number is None:
        raise ValueError(f"Expected finite numeric value, got {value!r}")
    return number


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


def _geometric_mean_return(total_return: float, periods: int) -> float | None:
    if periods <= 0 or total_return <= -1.0:
        return None
    return (1.0 + total_return) ** (1.0 / periods) - 1.0


def _annualized_return(
    total_return: float,
    records: list[dict[str, float | str]],
) -> float | None:
    if len(records) < 2 or total_return <= -1.0:
        return None
    try:
        dates = [datetime.fromisoformat(str(row["date"])[:10]) for row in records]
    except ValueError:
        return None
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
        if (current - previous).days > 0
    ]
    terminal_days = max(1, round(median(gaps))) if gaps else 0
    elapsed_days = (dates[-1] - dates[0]).days + terminal_days
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0


def _observed_periods_per_year(records: list[dict[str, float | str]]) -> float:
    if len(records) < 2:
        return 1.0
    try:
        dates = [datetime.fromisoformat(str(row["date"])[:10]) for row in records]
    except ValueError:
        return 1.0
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
        if (current - previous).days > 0
    ]
    terminal_days = max(1, round(median(gaps))) if gaps else 0
    elapsed_days = (dates[-1] - dates[0]).days + terminal_days
    if elapsed_days <= 0:
        return 1.0
    return max(1.0, len(records) * 365.25 / elapsed_days)


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(mean((value - average) ** 2 for value in values))


def _sharpe_ratio(
    returns: list[float],
    records: list[dict[str, float | str]],
) -> float:
    if not returns:
        return 0.0
    std = _population_std(returns)
    if std == 0.0:
        return 0.0
    return mean(returns) / std * math.sqrt(_observed_periods_per_year(records))


def _sortino_ratio(
    returns: list[float],
    records: list[dict[str, float | str]],
) -> float:
    if not returns:
        return 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if downside_deviation == 0.0:
        return 0.0
    return mean(returns) / downside_deviation * math.sqrt(
        _observed_periods_per_year(records)
    )


def _metric_delta(left: Any, right: Any) -> float | None:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return False
    return math.isclose(left_number, right_number, abs_tol=tolerance)
