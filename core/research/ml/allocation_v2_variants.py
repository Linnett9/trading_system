from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from statistics import mean
from typing import Any


HOLDOUT_OVERFIT_WARNING = (
    "Research-only diagnostic. Policy selected on holdout is not production-valid."
)
OUT_OF_SAMPLE_SELECTION_NOTICE = (
    "Research-only diagnostic. Policy selected on out-of-fold training predictions "
    "and evaluated on frozen holdout; not production-valid."
)


@dataclass(frozen=True)
class AllocationVariant:
    policy_family: str
    variant_name: str
    mapping_method: str
    return_weight: float
    drawdown_weight: float
    volatility_weight: float
    score_thresholds: dict[str, float]
    quantile_cutoffs: tuple[float, float, float, float]
    min_exposure: float
    max_exposure: float
    neutral_exposure: float
    transaction_cost_bps: float
    max_exposure_change: float | None
    threshold_fit_scope: str
    overfit_warning: str | None

    @property
    def policy_name(self) -> str:
        return f"{self.policy_family}_{self.variant_name}"


def named_policy_variants(config: dict[str, Any]) -> tuple[AllocationVariant, ...]:
    configured = config.get("allocation_policy_variants", {})
    variants = []
    for family, family_defaults in _default_variant_config().items():
        family_overrides = configured.get(family, {})
        for name, defaults in family_defaults.items():
            payload = {**defaults, **family_overrides.get(name, {})}
            variants.append(_variant_from_payload(family, name, payload))
    return tuple(variants)


def map_variant_scores(
    scores: list[float],
    dates: list[str],
    variant: AllocationVariant,
    *,
    fit_scores: list[float] | None = None,
) -> list[float]:
    if variant.mapping_method == "quantile":
        exposures = _quantile_exposures(
            scores,
            variant,
            fit_scores=fit_scores,
        )
    elif variant.mapping_method == "absolute":
        exposures = [_absolute_exposure(score, variant) for score in scores]
    else:
        raise ValueError(
            f"Unsupported allocation mapping method '{variant.mapping_method}'"
        )
    clipped = [_clip(value, variant.min_exposure, variant.max_exposure) for value in exposures]
    if variant.max_exposure_change is None:
        return clipped
    return _smooth_by_rebalance_date(
        clipped,
        dates,
        max_change=float(variant.max_exposure_change),
        minimum=variant.min_exposure,
        maximum=variant.max_exposure,
    )


def grid_candidate_payloads(config: dict[str, Any]) -> list[dict[str, float | str]]:
    grid = config.get("allocation_policy_grid", {})
    dimensions = (
        ("return_weight", grid.get("return_weights", [0.75, 1.0])),
        ("drawdown_weight", grid.get("drawdown_weights", [0.25, 0.50])),
        ("volatility_weight", grid.get("volatility_weights", [0.10, 0.25])),
        ("min_exposure", grid.get("min_exposures", [0.10])),
        ("max_exposure", grid.get("max_exposures", [0.80, 1.0])),
        ("neutral_exposure", grid.get("neutral_exposures", [0.50])),
        ("max_exposure_change", grid.get("max_exposure_changes", [0.20, 0.40])),
    )
    candidates = []
    for index, values in enumerate(
        product(*(dimension[1] for dimension in dimensions)),
        start=1,
    ):
        candidate = {
            name: float(value)
            for (name, _), value in zip(dimensions, values)
        }
        candidate.update({
            "candidate_id": f"grid_candidate_{index:03d}",
            "mapping_method": str(grid.get("mapping_method", "quantile")),
            "transaction_cost_bps": float(
                grid.get(
                    "transaction_cost_bps",
                    config.get("allocation_transaction_cost_bps", 5.0),
                )
            ),
        })
        candidates.append(candidate)
    return candidates


def grid_variant(candidate: dict[str, float | str]) -> AllocationVariant:
    return _variant_from_payload(
        "risk_adjusted_allocation_grid",
        str(candidate["candidate_id"]),
        {
            **candidate,
            "score_thresholds": {
                "strong": 0.02,
                "good": 0.005,
                "neutral": 0.0,
                "weak": -0.02,
            },
            "quantile_cutoffs": [0.10, 0.30, 0.70, 0.90],
        },
    )


def _variant_from_payload(
    family: str,
    name: str,
    payload: dict[str, Any],
) -> AllocationVariant:
    mapping = str(payload.get("mapping_method", "quantile"))
    fit_scope = (
        "holdout_in_sample_diagnostic"
        if mapping == "quantile"
        else "fixed_configured_thresholds"
    )
    cutoffs = tuple(float(value) for value in payload.get(
        "quantile_cutoffs",
        [0.10, 0.30, 0.70, 0.90],
    ))
    if len(cutoffs) != 4 or list(cutoffs) != sorted(cutoffs):
        raise ValueError("Allocation quantile_cutoffs must contain four ordered values")
    minimum = float(payload.get("min_exposure", 0.0))
    maximum = float(payload.get("max_exposure", 1.0))
    neutral = float(payload.get("neutral_exposure", 0.5))
    if not 0.0 <= minimum <= neutral <= maximum <= 1.0:
        raise ValueError(
            "Allocation variant exposures must satisfy 0 <= min <= neutral <= max <= 1"
        )
    max_change = payload.get("max_exposure_change")
    if max_change is not None and not 0.0 <= float(max_change) <= 1.0:
        raise ValueError("max_exposure_change must be between 0 and 1")
    return AllocationVariant(
        policy_family=family,
        variant_name=name,
        mapping_method=mapping,
        return_weight=float(payload.get("return_weight", 1.0)),
        drawdown_weight=float(payload.get("drawdown_weight", 0.0)),
        volatility_weight=float(payload.get("volatility_weight", 0.0)),
        score_thresholds={
            key: float(value)
            for key, value in payload.get("score_thresholds", {}).items()
        },
        quantile_cutoffs=(cutoffs[0], cutoffs[1], cutoffs[2], cutoffs[3]),
        min_exposure=minimum,
        max_exposure=maximum,
        neutral_exposure=neutral,
        transaction_cost_bps=float(payload.get("transaction_cost_bps", 5.0)),
        max_exposure_change=(float(max_change) if max_change is not None else None),
        threshold_fit_scope=fit_scope,
        overfit_warning=HOLDOUT_OVERFIT_WARNING if mapping == "quantile" else None,
    )


def _quantile_exposures(
    scores: list[float],
    variant: AllocationVariant,
    *,
    fit_scores: list[float] | None = None,
) -> list[float]:
    if not scores:
        return []
    threshold_scores = fit_scores if fit_scores else scores
    thresholds = [
        _quantile(threshold_scores, cutoff)
        for cutoff in variant.quantile_cutoffs
    ]
    defensive = (variant.min_exposure + variant.neutral_exposure) / 2.0
    high = (variant.neutral_exposure + variant.max_exposure) / 2.0
    levels = (
        variant.min_exposure,
        defensive,
        variant.neutral_exposure,
        high,
        variant.max_exposure,
    )
    output = []
    for score in scores:
        if score <= thresholds[0]:
            output.append(levels[0])
        elif score <= thresholds[1]:
            output.append(levels[1])
        elif score <= thresholds[2]:
            output.append(levels[2])
        elif score <= thresholds[3]:
            output.append(levels[3])
        else:
            output.append(levels[4])
    return output


def _absolute_exposure(score: float, variant: AllocationVariant) -> float:
    thresholds = variant.score_thresholds
    defensive = (variant.min_exposure + variant.neutral_exposure) / 2.0
    high = (variant.neutral_exposure + variant.max_exposure) / 2.0
    if score >= float(thresholds.get("strong", 0.02)):
        return variant.max_exposure
    if score >= float(thresholds.get("good", 0.005)):
        return high
    if score >= float(thresholds.get("neutral", 0.0)):
        return variant.neutral_exposure
    if score >= float(thresholds.get("weak", -0.02)):
        return defensive
    return variant.min_exposure


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate allocation quantiles without scores")
    probability = min(1.0, max(0.0, float(probability)))
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _smooth_by_rebalance_date(
    exposures: list[float],
    dates: list[str],
    *,
    max_change: float,
    minimum: float,
    maximum: float,
) -> list[float]:
    by_date: dict[str, list[int]] = {}
    for index, date in enumerate(dates):
        by_date.setdefault(date, []).append(index)
    smoothed_by_date: dict[str, float] = {}
    previous = None
    for date in sorted(by_date):
        desired = mean(exposures[index] for index in by_date[date])
        if previous is None:
            smoothed = desired
        else:
            smoothed = min(previous + max_change, max(previous - max_change, desired))
        smoothed = _clip(smoothed, minimum, maximum)
        smoothed_by_date[date] = smoothed
        previous = smoothed
    return [smoothed_by_date[date] for date in dates]


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, float(value)))


def _default_variant_config() -> dict[str, dict[str, dict[str, Any]]]:
    common = {
        "mapping_method": "quantile",
        "return_weight": 1.0,
        "score_thresholds": {
            "strong": 0.02,
            "good": 0.005,
            "neutral": 0.0,
            "weak": -0.02,
        },
        "quantile_cutoffs": [0.10, 0.30, 0.70, 0.90],
        "transaction_cost_bps": 5.0,
    }
    return {
        "return_only_allocation": {
            "conservative": {
                **common,
                "drawdown_weight": 0.0,
                "volatility_weight": 0.0,
                "min_exposure": 0.0,
                "max_exposure": 0.80,
                "neutral_exposure": 0.35,
                "max_exposure_change": 0.20,
            },
            "balanced": {
                **common,
                "drawdown_weight": 0.0,
                "volatility_weight": 0.0,
                "min_exposure": 0.10,
                "max_exposure": 1.0,
                "neutral_exposure": 0.55,
                "max_exposure_change": 0.30,
            },
            "aggressive": {
                **common,
                "drawdown_weight": 0.0,
                "volatility_weight": 0.0,
                "min_exposure": 0.30,
                "max_exposure": 1.0,
                "neutral_exposure": 0.70,
                "max_exposure_change": 0.50,
            },
        },
        "risk_adjusted_allocation": {
            "conservative": {
                **common,
                "drawdown_weight": 0.75,
                "volatility_weight": 0.40,
                "min_exposure": 0.0,
                "max_exposure": 0.80,
                "neutral_exposure": 0.35,
                "max_exposure_change": 0.20,
            },
            "balanced": {
                **common,
                "drawdown_weight": 0.40,
                "volatility_weight": 0.20,
                "min_exposure": 0.10,
                "max_exposure": 1.0,
                "neutral_exposure": 0.55,
                "max_exposure_change": 0.30,
            },
            "aggressive": {
                **common,
                "drawdown_weight": 0.20,
                "volatility_weight": 0.10,
                "min_exposure": 0.30,
                "max_exposure": 1.0,
                "neutral_exposure": 0.70,
                "max_exposure_change": 0.50,
            },
        },
    }
