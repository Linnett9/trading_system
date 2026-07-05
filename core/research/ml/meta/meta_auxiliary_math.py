from __future__ import annotations

import math
from statistics import mean
from typing import Any


def _regression_metrics(
    rows: list[dict[str, str]],
    actual_name: str,
    prediction_name: str,
) -> dict[str, Any]:
    pairs = [
        (float(row[actual_name]), float(row[prediction_name]))
        for row in rows
        if _finite_value(row.get(actual_name))
        and _finite_value(row.get(prediction_name))
    ]
    if not pairs:
        return {
            "available": True,
            "prediction_column": prediction_name,
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "directional_accuracy": None,
            "residual_quantiles": {},
        }
    actual = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    errors = [estimate - observed for observed, estimate in pairs]
    is_return = actual_name in {
        "actual_forward_return_5d",
        "actual_forward_return_10d",
    }
    return {
        "available": True,
        "prediction_column": prediction_name,
        "sample_count": len(pairs),
        "mae": mean(abs(value) for value in errors),
        "rmse": math.sqrt(mean(value * value for value in errors)),
        "pearson_correlation": _pearson(actual, predicted),
        "spearman_correlation": _pearson(_ranks(actual), _ranks(predicted)),
        "directional_accuracy": (
            mean(
                float((observed >= 0.0) == (estimate >= 0.0))
                for observed, estimate in pairs
            )
            if is_return
            else None
        ),
        "residual_quantiles": {
            "p10": _quantile(errors, 0.10),
            "p50": _quantile(errors, 0.50),
            "p90": _quantile(errors, 0.90),
        },
    }
def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None
def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[ordered[position][0]] = average_rank
        index = end
    return output
def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)
def _finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
