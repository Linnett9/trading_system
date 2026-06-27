from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol


OPTIMIZER_NOTICE = (
    "Research-only optimizer. Parameters selected on out-of-fold data and evaluated "
    "once on frozen holdout; not production-valid."
)


class CandidateSampler(Protocol):
    method: str

    def sample(self, count: int) -> list[dict[str, float | str]]:
        ...


@dataclass(frozen=True)
class OptimizerPaths:
    candidates_csv: Path
    results_json: Path
    report_markdown: Path


@dataclass(frozen=True)
class RandomSearchSampler:
    ranges: dict[str, tuple[float, float]]
    random_seed: int
    method: str = "random_search"

    def sample(self, count: int) -> list[dict[str, float | str]]:
        generator = random.Random(self.random_seed)
        candidates = []
        for index in range(count):
            minimum = _uniform(generator, self.ranges["min_exposure"])
            maximum = _uniform(generator, self.ranges["max_exposure"])
            if maximum < minimum:
                minimum, maximum = maximum, minimum
            neutral_bounds = self.ranges["neutral_exposure"]
            neutral_low = max(minimum, neutral_bounds[0])
            neutral_high = min(maximum, neutral_bounds[1])
            neutral = (
                generator.uniform(neutral_low, neutral_high)
                if neutral_low <= neutral_high
                else (minimum + maximum) / 2.0
            )
            candidates.append({
                "candidate_id": f"random_candidate_{index + 1:04d}",
                "mapping_method": "quantile",
                "return_weight": _uniform(generator, self.ranges["return_weight"]),
                "drawdown_weight": _uniform(
                    generator,
                    self.ranges["drawdown_weight"],
                ),
                "volatility_weight": _uniform(
                    generator,
                    self.ranges["volatility_weight"],
                ),
                "min_exposure": minimum,
                "max_exposure": maximum,
                "neutral_exposure": neutral,
                "max_exposure_change": _uniform(
                    generator,
                    self.ranges["max_exposure_change"],
                ),
                "transaction_cost_bps": _uniform(
                    generator,
                    self.ranges["transaction_cost_bps"],
                ),
            })
        return candidates


def build_optimizer_sampler(config: dict[str, Any]) -> CandidateSampler:
    optimizer = config.get("allocation_optimizer", {})
    method = str(optimizer.get("method", "random_search"))
    if method != "random_search":
        raise ValueError(
            f"Unsupported allocation optimizer method '{method}'; "
            "the sampler interface is ready for future Bayesian strategies"
        )
    configured_ranges = optimizer.get("ranges", {})
    defaults = {
        "return_weight": (0.50, 1.50),
        "drawdown_weight": (0.0, 1.0),
        "volatility_weight": (0.0, 0.50),
        "min_exposure": (0.0, 0.40),
        "max_exposure": (0.70, 1.0),
        "neutral_exposure": (0.30, 0.80),
        "max_exposure_change": (0.10, 1.0),
        "transaction_cost_bps": (5.0, 5.0),
    }
    ranges = {
        name: _range(configured_ranges.get(name), default)
        for name, default in defaults.items()
    }
    return RandomSearchSampler(
        ranges=ranges,
        random_seed=int(optimizer.get("random_seed", 42)),
    )


def optimizer_candidate_count(config: dict[str, Any]) -> int:
    optimizer = config.get("allocation_optimizer", {})
    count = int(optimizer.get("candidate_count", 256))
    maximum = int(optimizer.get("max_candidate_count", 5_000))
    if count < 1 or count > maximum:
        raise ValueError(
            f"allocation_optimizer.candidate_count must be between 1 and {maximum}"
        )
    return count


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


def write_optimizer_reports(
    output_dir: Path,
    report: dict[str, Any],
) -> OptimizerPaths:
    paths = OptimizerPaths(
        candidates_csv=output_dir / "allocation_optimizer_candidates.csv",
        results_json=output_dir / "allocation_optimizer_results.json",
        report_markdown=output_dir / "allocation_optimizer_report.md",
    )
    payload = {
        "mode": "allocation_optimizer_research_only",
        **report,
        "optimizer_notice": OPTIMIZER_NOTICE,
        "automatic_promotion": False,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    paths.results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    candidates = list(payload.get("candidates", []))
    rows = [_csv_row(row) for row in candidates]
    fieldnames = list(rows[0]) if rows else [
        "candidate_id",
        "objective_rank",
        "outcome_rank",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with paths.candidates_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(paths.report_markdown, payload)
    return paths


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Allocation Optimizer",
        "",
        OPTIMIZER_NOTICE,
        "",
        f"Method: {payload.get('method', 'unavailable')}",
        f"Candidates: {payload.get('candidate_count', 0)}",
        "",
    ]
    selected = payload.get("selected_policy")
    if selected:
        lines.extend([
            "## Selected Diagnostic Policy",
            "",
            f"Candidate: {selected.get('candidate_id')}",
            f"Selection objective: {selected.get('objective')}",
            f"Holdout return: {selected.get('holdout_metrics', {}).get('total_return')}",
            f"Holdout max drawdown: {selected.get('holdout_metrics', {}).get('max_drawdown')}",
            "",
        ])
    if payload.get("skip_reason"):
        lines.extend([f"Skipped: {payload['skip_reason']}", ""])
    lines.append(
        "Research only. Trading impact: none. Production validated: false."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Allocation optimizer ranges must contain [minimum, maximum]")
    minimum, maximum = float(value[0]), float(value[1])
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ValueError("Allocation optimizer ranges must be finite and ordered")
    return minimum, maximum


def _uniform(generator: random.Random, bounds: tuple[float, float]) -> float:
    return generator.uniform(bounds[0], bounds[1])


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


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, value in row.items():
        output[name] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    output.update({
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    })
    return output
