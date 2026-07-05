from __future__ import annotations

import math
from datetime import datetime
from statistics import mean


def _annualized_return(
    total_return: float,
    periods: list[tuple[str, float, float]],
) -> float | None:
    if len(periods) < 2 or total_return <= -1.0:
        return None
    try:
        start = datetime.fromisoformat(periods[0][0][:10])
        end = datetime.fromisoformat(periods[-1][0][:10])
    except ValueError:
        return None
    elapsed_days = (end - start).days + _estimated_terminal_period_days(periods)
    if elapsed_days <= 0:
        return None
    return (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0
def _percentage_at_exposure(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(
        math.isclose(value, target, abs_tol=1e-12) for value in values
    ) / len(values)
def _estimated_terminal_period_days(
    periods: list[tuple[str, float, float]],
) -> int:
    try:
        dates = [datetime.fromisoformat(row[0][:10]) for row in periods]
    except ValueError:
        return 0
    gaps = [
        (current - previous).days
        for previous, current in zip(dates, dates[1:])
        if (current - previous).days > 0
    ]
    return max(1, round(mean(gaps))) if gaps else 0
def _compound_returns(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0
def _current_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = max(values)
    return (values[-1] / peak) - 1.0 if peak else 0.0
def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)
    return drawdown
def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(mean((value - average) ** 2 for value in values))
def _sharpe_ratio(
    returns: list[float],
    periods: list[tuple[str, float, float]],
) -> float:
    if not returns:
        return 0.0
    average = mean(returns)
    standard_deviation = _population_std(returns)
    if standard_deviation == 0.0:
        return 0.0
    return (
        average
        / standard_deviation
        * math.sqrt(_observed_periods_per_year(periods))
    )
def _sortino_ratio(
    returns: list[float],
    periods: list[tuple[str, float, float]],
) -> float:
    if not returns:
        return 0.0
    downside = [min(value, 0.0) for value in returns]
    downside_deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if downside_deviation == 0.0:
        return 0.0
    return (
        mean(returns)
        / downside_deviation
        * math.sqrt(_observed_periods_per_year(periods))
    )
def _observed_periods_per_year(
    periods: list[tuple[str, float, float]],
) -> float:
    if len(periods) < 2:
        return 1.0
    try:
        start = datetime.fromisoformat(periods[0][0][:10])
        end = datetime.fromisoformat(periods[-1][0][:10])
    except ValueError:
        return 1.0
    elapsed_days = (end - start).days + _estimated_terminal_period_days(periods)
    if elapsed_days <= 0:
        return 1.0
    return max(1.0, len(periods) * 365.25 / elapsed_days)
