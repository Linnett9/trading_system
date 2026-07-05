from __future__ import annotations

import random
from statistics import mean
from typing import Any


def bootstrap_paired_comparison(
    selected_returns: list[float],
    baseline_returns: list[float],
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    if len(selected_returns) != len(baseline_returns) or not selected_returns:
        return {
            "available": False,
            "reason": "paired return series are empty or have different lengths",
        }
    generator = random.Random(random_seed)
    compounded_deltas = []
    mean_deltas = []
    sample_count = len(selected_returns)
    for _ in range(max(1, iterations)):
        indexes = [generator.randrange(sample_count) for _ in range(sample_count)]
        selected_sample = [selected_returns[index] for index in indexes]
        baseline_sample = [baseline_returns[index] for index in indexes]
        compounded_deltas.append(
            _compound(selected_sample) - _compound(baseline_sample)
        )
        mean_deltas.append(mean(
            selected - baseline
            for selected, baseline in zip(selected_sample, baseline_sample)
        ))
    return {
        "available": True,
        "method": "paired_nonparametric_period_bootstrap",
        "iterations": max(1, iterations),
        "sample_count": sample_count,
        "compounded_return_delta": {
            "mean": mean(compounded_deltas),
            "confidence_interval_95": [
                _quantile(compounded_deltas, 0.025),
                _quantile(compounded_deltas, 0.975),
            ],
            "probability_selected_outperforms": mean(
                float(value > 0.0) for value in compounded_deltas
            ),
        },
        "mean_period_return_delta": {
            "mean": mean(mean_deltas),
            "confidence_interval_95": [
                _quantile(mean_deltas, 0.025),
                _quantile(mean_deltas, 0.975),
            ],
        },
    }

def _compound(returns: list[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
    return equity - 1.0

def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)
