from __future__ import annotations

import csv
import importlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Protocol


OPTIMIZER_NOTICE = (
    "Research-only optimizer. Parameters selected on out-of-fold data and evaluated "
    "once on frozen holdout; not production-valid."
)


class CandidateSampler(Protocol):
    method: str
    sampler_requested: str
    sampler_used: str
    optuna_available: bool
    fallback_reason: str | None

    def sample(self, count: int) -> list[dict[str, float | str]]:
        ...

    def suggest(self, trial_number: int) -> dict[str, float | str]:
        ...

    def observe(
        self,
        candidate: dict[str, float | str],
        objective_value: float | None,
    ) -> None:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OptimizerPaths:
    candidates_csv: Path
    results_json: Path
    report_markdown: Path
    selected_exposure_path_csv: Path
    selected_exposure_path_json: Path


@dataclass
class RandomSearchSampler:
    ranges: dict[str, tuple[float, float]]
    random_seed: int
    method: str = "random_search"
    sampler_requested: str = "random"
    sampler_used: str = "random"
    optuna_available: bool = False
    fallback_reason: str | None = None
    _generator: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._generator = random.Random(self.random_seed)

    def sample(self, count: int) -> list[dict[str, float | str]]:
        generator = random.Random(self.random_seed)
        return [
            _random_candidate(generator, index, self.ranges)
            for index in range(count)
        ]

    def suggest(self, trial_number: int) -> dict[str, float | str]:
        return _random_candidate(self._generator, trial_number, self.ranges)

    def observe(
        self,
        candidate: dict[str, float | str],
        objective_value: float | None,
    ) -> None:
        del candidate, objective_value

    def metadata(self) -> dict[str, Any]:
        return {
            "sampler_requested": self.sampler_requested,
            "sampler_used": self.sampler_used,
            "optuna_available": self.optuna_available,
            "fallback_reason": self.fallback_reason,
            "sampler_seed": self.random_seed,
        }


@dataclass
class OptunaBayesianSampler:
    ranges: dict[str, tuple[float, float]]
    n_trials: int
    sampler_seed: int
    startup_trials: int
    study_direction: str
    method: str = "bayesian_search"
    sampler_requested: str = "bayesian"
    sampler_used: str = "bayesian"
    optuna_available: bool = True
    fallback_reason: str | None = None
    _study: Any = field(init=False, repr=False)
    _pending_trials: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        import optuna

        sampler = optuna.samplers.TPESampler(
            seed=self.sampler_seed,
            n_startup_trials=self.startup_trials,
        )
        self._study = optuna.create_study(
            direction=self.study_direction,
            sampler=sampler,
        )

    def sample(self, count: int) -> list[dict[str, float | str]]:
        return [self.suggest(index) for index in range(count)]

    def suggest(self, trial_number: int) -> dict[str, float | str]:
        trial = self._study.ask()
        minimum = _suggest_float(trial, "min_exposure", self.ranges["min_exposure"])
        maximum = _suggest_float(trial, "max_exposure", self.ranges["max_exposure"])
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        neutral_raw = _suggest_float(
            trial,
            "neutral_exposure",
            self.ranges["neutral_exposure"],
        )
        candidate_id = f"bayesian_trial_{trial.number:04d}"
        candidate = {
            "candidate_id": candidate_id,
            "trial_number": trial.number,
            "mapping_method": "quantile",
            "return_weight": _suggest_float(
                trial,
                "return_weight",
                self.ranges["return_weight"],
            ),
            "drawdown_weight": _suggest_float(
                trial,
                "drawdown_weight",
                self.ranges["drawdown_weight"],
            ),
            "volatility_weight": _suggest_float(
                trial,
                "volatility_weight",
                self.ranges["volatility_weight"],
            ),
            "min_exposure": minimum,
            "max_exposure": maximum,
            "neutral_exposure": min(maximum, max(minimum, neutral_raw)),
            "max_exposure_change": _suggest_float(
                trial,
                "max_exposure_change",
                self.ranges["max_exposure_change"],
            ),
            "transaction_cost_bps": _suggest_float(
                trial,
                "transaction_cost_bps",
                self.ranges["transaction_cost_bps"],
            ),
        }
        self._pending_trials[candidate_id] = trial
        return candidate

    def observe(
        self,
        candidate: dict[str, float | str],
        objective_value: float | None,
    ) -> None:
        trial = self._pending_trials.pop(str(candidate["candidate_id"]))
        if objective_value is None:
            import optuna

            self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return
        self._study.tell(trial, float(objective_value))

    def metadata(self) -> dict[str, Any]:
        completed = [trial for trial in self._study.trials if trial.value is not None]
        best_trial = self._study.best_trial.number if completed else None
        return {
            "sampler_requested": self.sampler_requested,
            "sampler_used": self.sampler_used,
            "optuna_available": self.optuna_available,
            "fallback_reason": self.fallback_reason,
            "best_trial_number": best_trial,
            "startup_trials": self.startup_trials,
            "n_trials": self.n_trials,
            "study_direction": self.study_direction,
            "sampler_seed": self.sampler_seed,
        }


def build_optimizer_sampler(config: dict[str, Any]) -> CandidateSampler:
    optimizer = config.get("allocation_optimizer", {})
    requested = str(
        optimizer.get("sampler")
        or _legacy_sampler_name(optimizer.get("method"))
        or "random"
    ).lower()
    if requested not in {"random", "bayesian"}:
        raise ValueError(
            f"Unsupported allocation optimizer sampler '{requested}'"
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
    available = optuna_is_available()
    bayesian = optimizer.get("bayesian", {})
    if requested == "bayesian" and available:
        direction = str(bayesian.get("direction", "maximize")).lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError("allocation_optimizer.bayesian.direction is invalid")
        try:
            return OptunaBayesianSampler(
                ranges=ranges,
                n_trials=int(
                    bayesian.get("n_trials", optimizer.get("candidate_count", 256))
                ),
                sampler_seed=int(
                    bayesian.get("seed", optimizer.get("random_seed", 42))
                ),
                startup_trials=int(bayesian.get("startup_trials", 32)),
                study_direction=direction,
            )
        except Exception as exc:
            fallback_reason = (
                "Optuna initialization failed; falling back to deterministic "
                f"random search: {exc}"
            )
            return RandomSearchSampler(
                ranges=ranges,
                random_seed=int(
                    bayesian.get("seed", optimizer.get("random_seed", 42))
                ),
                sampler_requested=requested,
                optuna_available=True,
                fallback_reason=fallback_reason,
            )
    fallback_reason = (
        "Optuna is not installed; falling back to deterministic random search"
        if requested == "bayesian" and not available
        else None
    )
    return RandomSearchSampler(
        ranges=ranges,
        random_seed=int(
            bayesian.get("seed", optimizer.get("random_seed", 42))
            if requested == "bayesian"
            else optimizer.get("random_seed", 42)
        ),
        sampler_requested=requested,
        optuna_available=available,
        fallback_reason=fallback_reason,
    )


def optimizer_candidate_count(
    config: dict[str, Any],
    sampler: CandidateSampler | None = None,
) -> int:
    optimizer = config.get("allocation_optimizer", {})
    requested = sampler.sampler_requested if sampler else str(
        optimizer.get("sampler", "random")
    )
    count = int(
        optimizer.get("bayesian", {}).get(
            "n_trials",
            optimizer.get("candidate_count", 256),
        )
        if requested == "bayesian"
        else optimizer.get("candidate_count", 256)
    )
    maximum = int(optimizer.get("max_candidate_count", 5_000))
    if count < 1 or count > maximum:
        raise ValueError(
            f"allocation_optimizer.candidate_count must be between 1 and {maximum}"
        )
    return count


def optuna_is_available() -> bool:
    try:
        importlib.import_module("optuna")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


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
        selected_exposure_path_csv=output_dir / "selected_optimizer_exposure_path.csv",
        selected_exposure_path_json=output_dir / "selected_optimizer_exposure_path.json",
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
    _write_selected_exposure_path(paths, payload)
    _write_markdown(paths.report_markdown, payload)
    return paths


def _write_selected_exposure_path(
    paths: OptimizerPaths,
    payload: dict[str, Any],
) -> None:
    rows = list(payload.get("selected_optimizer_exposure_path", []))
    path_payload = {
        "mode": "selected_optimizer_exposure_path_research_only",
        "sampler_requested": payload.get("sampler_requested"),
        "sampler_used": payload.get("sampler_used"),
        "selected_policy": payload.get("selected_policy", {}),
        "row_count": len(rows),
        "rows": rows,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    paths.selected_exposure_path_json.write_text(
        json.dumps(path_payload, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "rebalance_date",
        "source_row_count",
        "period_return",
        "exposure",
        "score",
        "predicted_forward_return",
        "predicted_future_drawdown",
        "predicted_future_volatility",
        "turnover",
        "transaction_cost_bps",
        "cost",
        "net_return",
        "equity",
        "drawdown",
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with paths.selected_exposure_path_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                **{name: row.get(name) for name in fieldnames},
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
            }
            for row in rows
        )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Allocation Optimizer",
        "",
        OPTIMIZER_NOTICE,
        "",
        f"Sampler requested: {payload.get('sampler_requested', 'unknown')}",
        f"Sampler used: {payload.get('sampler_used', 'unknown')}",
        f"Optuna available: {payload.get('optuna_available', False)}",
        f"Fallback reason: {payload.get('fallback_reason') or 'none'}",
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
            f"Selected params: {selected.get('selected_params')}",
            f"Holdout return: {selected.get('frozen_holdout_metrics', {}).get('total_return')}",
            f"Holdout max drawdown: {selected.get('frozen_holdout_metrics', {}).get('max_drawdown')}",
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


def _legacy_sampler_name(value: Any) -> str | None:
    normalized = str(value or "").lower()
    if normalized in {"random", "random_search"}:
        return "random"
    if normalized in {"bayesian", "bayesian_search", "optuna"}:
        return "bayesian"
    return normalized or None


def _random_candidate(
    generator: random.Random,
    trial_number: int,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, float | str]:
    minimum = _uniform(generator, ranges["min_exposure"])
    maximum = _uniform(generator, ranges["max_exposure"])
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    neutral_bounds = ranges["neutral_exposure"]
    neutral_low = max(minimum, neutral_bounds[0])
    neutral_high = min(maximum, neutral_bounds[1])
    neutral = (
        generator.uniform(neutral_low, neutral_high)
        if neutral_low <= neutral_high
        else (minimum + maximum) / 2.0
    )
    return {
        "candidate_id": f"random_candidate_{trial_number + 1:04d}",
        "trial_number": trial_number,
        "mapping_method": "quantile",
        "return_weight": _uniform(generator, ranges["return_weight"]),
        "drawdown_weight": _uniform(generator, ranges["drawdown_weight"]),
        "volatility_weight": _uniform(generator, ranges["volatility_weight"]),
        "min_exposure": minimum,
        "max_exposure": maximum,
        "neutral_exposure": neutral,
        "max_exposure_change": _uniform(generator, ranges["max_exposure_change"]),
        "transaction_cost_bps": _uniform(generator, ranges["transaction_cost_bps"]),
    }


def _suggest_float(
    trial: Any,
    name: str,
    bounds: tuple[float, float],
) -> float:
    if bounds[0] == bounds[1]:
        return bounds[0]
    return float(trial.suggest_float(name, bounds[0], bounds[1]))


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
