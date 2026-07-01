from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any


def _trailing_return(values: list[float], lookback: int) -> float | str:
    if len(values) <= lookback or values[-lookback - 1] <= 0.0:
        return ""
    return values[-1] / values[-lookback - 1] - 1.0
def _momentum_persistence(values: list[float]) -> float | str:
    if len(values) < 140:
        return ""
    endpoints = range(len(values) - 120, len(values))
    outcomes = [values[index] / values[index - 20] - 1.0 for index in endpoints]
    return mean(1.0 if value > 0.0 else 0.0 for value in outcomes)
def _trend_r_squared(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    sample = [math.log(value) for value in values[-lookback:] if value > 0.0]
    if len(sample) != lookback:
        return ""
    x = list(range(lookback))
    x_mean = mean(x)
    y_mean = mean(sample)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, sample)) / denominator
    fitted = [y_mean + slope * (value - x_mean) for value in x]
    total = sum((value - y_mean) ** 2 for value in sample)
    residual = sum((value - estimate) ** 2 for value, estimate in zip(sample, fitted))
    return 1.0 - residual / total if total > 0.0 else 1.0
def _distance_from_high(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    high = max(values[-lookback:])
    return values[-1] / high - 1.0 if high > 0.0 else ""
def _drawdown_recovery_days(values: list[float], lookback: int) -> int | str:
    if len(values) < lookback:
        return ""
    sample = values[-lookback:]
    high = max(sample)
    latest_high_index = max(index for index, value in enumerate(sample) if value == high)
    return len(sample) - 1 - latest_high_index
def _max_drawdown(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    peak = values[-lookback]
    worst = 0.0
    for value in values[-lookback:]:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst
def _ulcer_index(values: list[float], lookback: int) -> float | str:
    if len(values) < lookback:
        return ""
    peak = values[-lookback]
    squared = []
    for value in values[-lookback:]:
        peak = max(peak, value)
        squared.append((value / peak - 1.0) ** 2)
    return math.sqrt(mean(squared))
def _downside_deviation(values: list[float], lookback: int) -> float | str:
    if len(values) <= lookback:
        return ""
    sample = values[-lookback - 1 :]
    returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample))]
    return math.sqrt(mean(min(value, 0.0) ** 2 for value in returns))
def _volatility(values: list[float], lookback: int) -> float | str:
    if len(values) <= lookback:
        return ""
    sample = values[-lookback - 1 :]
    returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample))]
    return pstdev(returns) if len(returns) > 1 else ""
def _volatility_percentile(values: list[float]) -> float | str:
    if len(values) < 272:
        return ""
    series = []
    for end in range(len(values) - 252, len(values)):
        sample = values[end - 20 : end + 1]
        returns = [sample[index] / sample[index - 1] - 1.0 for index in range(1, len(sample))]
        series.append(pstdev(returns))
    return _percentile_rank(series[-1], series)
def _atr_percentile(history: list[dict[str, float | str]]) -> float | str:
    if len(history) < 266:
        return ""
    true_ranges = []
    for index, row in enumerate(history):
        high = float(row["high"])
        low = float(row["low"])
        previous_close = float(history[index - 1]["close"]) if index else float(row["close"])
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    normalized_atr = []
    for end in range(len(history) - 252, len(history)):
        if end < 13:
            continue
        atr = mean(true_ranges[end - 13 : end + 1])
        close = float(history[end]["close"])
        normalized_atr.append(atr / close if close > 0.0 else 0.0)
    return _percentile_rank(normalized_atr[-1], normalized_atr) if normalized_atr else ""
def _slope(x: tuple[float, ...], y: tuple[Any, ...]) -> float | str:
    numbers = [_number(value) for value in y]
    if any(value is None for value in numbers):
        return ""
    x_mean = mean(x)
    y_mean = mean(float(value) for value in numbers if value is not None)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum(
        (a - x_mean) * (float(b) - y_mean)
        for a, b in zip(x, numbers)
        if b is not None
    ) / denominator
def _percentile_rank(value: float | None, values: list[float]) -> float | str:
    if value is None or not values:
        return ""
    less = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    return (less + 0.5 * equal) / len(values)
def _ratio_minus_one(left: Any, right: Any) -> float | str:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None or right_number <= 0.0:
        return ""
    return left_number / right_number - 1.0
def _difference(left: Any, right: Any) -> float | str:
    left_number = _number(left)
    right_number = _number(right)
    return left_number - right_number if left_number is not None and right_number is not None else ""
def _volatility_regime(percentile: Any) -> int | str:
    value = _number(percentile)
    if value is None:
        return ""
    return 0 if value < 1.0 / 3.0 else 2 if value > 2.0 / 3.0 else 1
def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
