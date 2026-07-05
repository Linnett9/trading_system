from __future__ import annotations

import importlib
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from core.research.ml.allocation.allocation_optimizer_types import CandidateSampler


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
    facade = sys.modules.get(
        "core.research.ml.allocation.allocation_optimizer"
    )
    availability_check = getattr(
        facade,
        "optuna_is_available",
        optuna_is_available,
    )
    available = availability_check()
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
