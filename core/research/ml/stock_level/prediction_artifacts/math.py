from __future__ import annotations

import math
from bisect import bisect_left
from statistics import mean, pstdev
from typing import Any


def _history_values_before(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
) -> list[float]:
    index = bisect_left(dates, rebalance_date)
    return values[:index]


def _trailing_return(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values, rebalance_date)
    if len(history) <= lookback:
        return ""
    end = history[-1]
    start = history[-lookback - 1]
    return (end / start) - 1.0 if start > 0.0 else ""


def _trailing_volatility(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values, rebalance_date)
    if len(history) <= lookback:
        return ""
    prices = history[-lookback - 1 :]
    returns = [
        (prices[index] / prices[index - 1]) - 1.0
        for index in range(1, len(prices))
        if prices[index - 1] > 0.0
    ]
    return pstdev(returns) if len(returns) > 1 else ""


def _trailing_drawdown(
    dates: list[str],
    values: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values, rebalance_date)
    if len(history) < lookback:
        return ""
    values = history[-lookback:]
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, (value / peak) - 1.0 if peak > 0.0 else 0.0)
    return worst


def _trailing_liquidity_score(
    dates: list[str],
    values_by_index: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    history = _history_values_before(dates, values_by_index, rebalance_date)
    if not history:
        return ""
    values = history[-lookback:]
    return math.log1p(mean(values)) if values else ""


def _forward_return(data: dict[str, Any], date: str, horizon: int) -> float | str:
    dates = data.get("close_dates", [])
    closes = data.get("close", {})
    if date not in closes:
        return ""
    index = dates.index(date)
    if index + horizon >= len(dates) or closes[date] <= 0.0:
        return ""
    return closes[dates[index + horizon]] / closes[date] - 1.0


def _average_dollar_volume(
    dates: list[str],
    values_by_index: list[float],
    rebalance_date: str,
    *,
    lookback: int,
) -> float | str:
    index = bisect_left(dates, rebalance_date)
    if index >= len(dates) or dates[index] != rebalance_date:
        return ""
    values = [
        value
        for value in values_by_index[max(0, index - lookback + 1) : index + 1]
        if value > 0.0
    ]
    return mean(values) if values else ""
