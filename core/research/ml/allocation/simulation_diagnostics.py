from __future__ import annotations

import math
from statistics import mean

from core.research.ml.allocation.exposures import _forecast_values


def _prediction_to_exposure_diagnostics(
    rows: list[dict[str, str]],
    exposures: list[float],
) -> dict[str, float | None]:
    return_values = [
        _mean_forecast_or_none(row, "predicted_forward_return_10d")
        for row in rows
    ]
    volatility_values = [
        _mean_forecast_or_none(row, "predicted_future_volatility")
        for row in rows
    ]
    drawdown_values = [
        _mean_forecast_or_none(row, "predicted_future_drawdown")
        for row in rows
    ]
    drawdown_risk_values = [
        abs(value) if value is not None else None
        for value in drawdown_values
    ]
    return {
        "correlation_predicted_forward_return_10d_to_exposure": (
            _paired_correlation(return_values, exposures)
        ),
        "correlation_predicted_future_volatility_to_exposure": (
            _paired_correlation(volatility_values, exposures)
        ),
        "correlation_predicted_future_drawdown_to_exposure": (
            _paired_correlation(drawdown_values, exposures)
        ),
        "average_exposure_predicted_return_top_quartile": _quartile_exposure(
            return_values,
            exposures,
            top=True,
        ),
        "average_exposure_predicted_return_bottom_quartile": _quartile_exposure(
            return_values,
            exposures,
            top=False,
        ),
        "average_exposure_predicted_drawdown_risk_top_quartile": (
            _quartile_exposure(drawdown_risk_values, exposures, top=True)
        ),
        "average_exposure_predicted_volatility_top_quartile": (
            _quartile_exposure(volatility_values, exposures, top=True)
        ),
    }
def _mean_forecast_or_none(
    row: dict[str, str],
    suffix: str,
) -> float | None:
    try:
        values = _forecast_values(row, suffix)
    except (TypeError, ValueError):
        return None
    return mean(values) if values else None
def _paired_correlation(
    values: list[float | None],
    exposures: list[float],
) -> float | None:
    pairs = [
        (float(value), float(exposure))
        for value, exposure in zip(values, exposures)
        if value is not None
    ]
    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = mean(left)
    right_mean = mean(right)
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0.0:
        return None
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    ) / denominator
def _quartile_exposure(
    values: list[float | None],
    exposures: list[float],
    *,
    top: bool,
) -> float | None:
    pairs = sorted(
        (
            (float(value), float(exposure))
            for value, exposure in zip(values, exposures)
            if value is not None
        ),
        key=lambda pair: pair[0],
        reverse=top,
    )
    if not pairs:
        return None
    count = max(1, math.ceil(len(pairs) * 0.25))
    return mean(exposure for _, exposure in pairs[:count])
