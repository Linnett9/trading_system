from __future__ import annotations

import math
from functools import partial
from statistics import mean
from typing import Any

from core.research.ml.allocation.allocation_v2_variants import (
    AllocationVariant,
    HOLDOUT_OVERFIT_WARNING,
    OUT_OF_SAMPLE_SELECTION_NOTICE,
    grid_candidate_payloads,
    grid_variant,
    map_variant_scores,
    named_policy_variants,
)
from core.research.ml.allocation.types import AllocationPolicyDefinition


def _policy_definitions(
    config: dict[str, Any] | None = None,
    *,
    selection_rows: list[dict[str, str]] | None = None,
) -> tuple[AllocationPolicyDefinition, ...]:
    base_definitions = (
        AllocationPolicyDefinition(
            policy_name="binary_exposure_overlay",
            required_prediction_columns=("predicted_probability",),
            exposure_builder=_binary_exposures,
        ),
        AllocationPolicyDefinition(
            policy_name="return_only_allocation",
            required_prediction_columns=(
                "predicted_forward_return_10d|predicted_forward_return_5d",
            ),
            exposure_builder=_return_only_exposures,
        ),
        AllocationPolicyDefinition(
            policy_name="risk_adjusted_allocation",
            required_prediction_columns=(
                "predicted_forward_return_10d|predicted_forward_return_5d",
                "predicted_future_drawdown|predicted_max_adverse_excursion",
                "predicted_future_volatility",
            ),
            exposure_builder=_risk_adjusted_exposures,
        ),
        AllocationPolicyDefinition(
            policy_name="meta_ensemble_allocation",
            required_prediction_columns=("predicted_probability",),
            exposure_builder=_meta_exposures,
        ),
    )
    variant_definitions = []
    for variant in named_policy_variants(config or {}):
        uses_frozen_selection = bool(selection_rows) and variant.mapping_method == "quantile"
        required_columns = (
            ("predicted_forward_return_10d|predicted_forward_return_5d",)
            if variant.policy_family == "return_only_allocation"
            else (
                "predicted_forward_return_10d|predicted_forward_return_5d",
                "predicted_future_drawdown|predicted_max_adverse_excursion",
                "predicted_future_volatility",
            )
        )
        variant_definitions.append(AllocationPolicyDefinition(
            policy_name=variant.policy_name,
            required_prediction_columns=required_columns,
            exposure_builder=partial(
                _variant_exposures,
                variant=variant,
                fit_rows=selection_rows,
            ),
            mapping_method=variant.mapping_method,
            threshold_fit_scope=(
                "out_of_fold_train_predictions"
                if uses_frozen_selection
                else variant.threshold_fit_scope
            ),
            overfit_warning=(
                OUT_OF_SAMPLE_SELECTION_NOTICE
                if uses_frozen_selection
                else variant.overfit_warning
            ),
            transaction_cost_bps=variant.transaction_cost_bps,
            exposure_min=variant.min_exposure,
            exposure_max=variant.max_exposure,
        ))
    return base_definitions + tuple(variant_definitions)


def _baseline_definitions() -> tuple[AllocationPolicyDefinition, ...]:
    return (
        AllocationPolicyDefinition(
            policy_name="champion_baseline",
            required_prediction_columns=(),
            exposure_builder=_always_full_exposures,
            policy_kind="diagnostic_baseline",
        ),
        AllocationPolicyDefinition(
            policy_name="always_full_exposure",
            required_prediction_columns=(),
            exposure_builder=_always_full_exposures,
            policy_kind="diagnostic_baseline",
        ),
        AllocationPolicyDefinition(
            policy_name="always_half_exposure",
            required_prediction_columns=(),
            exposure_builder=_always_half_exposures,
            policy_kind="diagnostic_baseline",
        ),
        AllocationPolicyDefinition(
            policy_name="always_zero_exposure",
            required_prediction_columns=(),
            exposure_builder=_always_zero_exposures,
            policy_kind="diagnostic_baseline",
        ),
    )


def _policy_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> dict[str, list[float]]:
    exposures, _ = _evaluate_policy_exposures(rows, probabilities, config)
    return exposures


def _evaluate_policy_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
    *,
    selection_rows: list[dict[str, str]] | None = None,
) -> tuple[dict[str, list[float]], dict[str, str]]:
    exposures: dict[str, list[float]] = {}
    skipped: dict[str, str] = {}
    for definition in _policy_definitions(config, selection_rows=selection_rows):
        missing = _missing_requirements(
            definition.required_prediction_columns,
            rows,
            probabilities,
        )
        if missing:
            skipped[definition.policy_name] = (
                "missing required prediction columns: " + ", ".join(missing)
            )
            continue
        try:
            raw_exposures = definition.exposure_builder(rows, probabilities, config)
            exposures[definition.policy_name] = [
                _clip_exposure(value, definition)
                for value in raw_exposures
            ]
        except (TypeError, ValueError) as exc:
            skipped[definition.policy_name] = f"invalid policy inputs: {exc}"
    return exposures, skipped


def _missing_requirements(
    requirements: tuple[str, ...],
    rows: list[dict[str, str]],
    probabilities: list[float],
) -> list[str]:
    missing = []
    for requirement in requirements:
        alternatives = requirement.split("|")
        if alternatives == ["predicted_probability"]:
            if len(probabilities) != len(rows) or not all(
                _is_probability(value) for value in probabilities
            ):
                missing.append(requirement)
            continue
        if not rows or not all(
            any(_has_forecast(row, name) for name in alternatives)
            for row in rows
        ):
            missing.append(requirement)
    return missing


def _binary_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del rows
    threshold = float(config.get("decision_threshold", 0.5))
    reduced_exposure = float(config.get("promotion_reduced_exposure", 0.7))
    return [
        reduced_exposure if probability >= threshold else 1.0
        for probability in probabilities
    ]


def _return_only_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities
    return [_score_to_exposure(_return_forecast(row), config) for row in rows]


def _risk_adjusted_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities
    return [_risk_adjusted_exposure(row, config) for row in rows]


def _meta_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del rows, config
    return [_meta_probability_exposure(value) for value in probabilities]


def _variant_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
    *,
    variant: AllocationVariant,
    fit_rows: list[dict[str, str]] | None = None,
) -> list[float]:
    del probabilities, config
    scores = _variant_scores(rows, variant)
    fit_scores = _variant_scores(fit_rows, variant) if fit_rows else None
    dates = [
        str(row.get("rebalance_date") or row.get("date") or "")
        for row in rows
    ]
    return map_variant_scores(
        scores,
        dates,
        variant,
        fit_scores=fit_scores,
    )


def _variant_scores(
    rows: list[dict[str, str]],
    variant: AllocationVariant,
) -> list[float]:
    scores = []
    for row in rows:
        expected_return = _return_forecast(row)
        if variant.policy_family == "return_only_allocation":
            score = variant.return_weight * expected_return
        else:
            drawdowns = _forecast_values(row, "predicted_future_drawdown")
            adverse = _forecast_values(row, "predicted_max_adverse_excursion")
            volatility = _forecast_values(row, "predicted_future_volatility")
            drawdown = mean(drawdowns) if drawdowns else mean(adverse)
            score = (
                (variant.return_weight * expected_return)
                - (variant.drawdown_weight * abs(drawdown))
                - (variant.volatility_weight * mean(volatility))
            )
        scores.append(score)
    return scores


def _always_full_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities, config
    return [1.0 for _ in rows]


def _always_half_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities, config
    return [0.5 for _ in rows]


def _always_zero_exposures(
    rows: list[dict[str, str]],
    probabilities: list[float],
    config: dict[str, Any],
) -> list[float]:
    del probabilities, config
    return [0.0 for _ in rows]


def _return_forecast(row: dict[str, str]) -> float:
    ten_day = _forecast_values(row, "predicted_forward_return_10d")
    if ten_day:
        return mean(ten_day)
    five_day = _forecast_values(row, "predicted_forward_return_5d")
    return mean(five_day)


def _risk_adjusted_exposure(row: dict[str, str], config: dict[str, Any]) -> float:
    drawdowns = _forecast_values(row, "predicted_future_drawdown")
    adverse = _forecast_values(row, "predicted_max_adverse_excursion")
    volatility = _forecast_values(row, "predicted_future_volatility")
    drawdown = mean(drawdowns) if drawdowns else mean(adverse)
    risk = mean(volatility)
    severe_drawdown = float(config.get("allocation_severe_drawdown", -0.15))
    severe_volatility = float(config.get("allocation_severe_volatility", 0.40))
    if drawdown <= severe_drawdown or risk >= severe_volatility:
        return 0.0
    score = _return_forecast(row) - (0.5 * abs(drawdown)) - (0.25 * risk)
    return _score_to_exposure(score, config)


def _score_to_exposure(score: float, config: dict[str, Any]) -> float:
    thresholds = config.get("allocation_score_thresholds", {})
    levels = (
        (float(thresholds.get("strong", 0.02)), 1.0),
        (float(thresholds.get("good", 0.005)), 0.8),
        (float(thresholds.get("neutral", 0.0)), 0.5),
        (float(thresholds.get("weak", -0.02)), 0.2),
    )
    for threshold, exposure in levels:
        if score >= threshold:
            return exposure
    return 0.0


def _meta_probability_exposure(probability: float) -> float:
    probability = _finite_float(probability)
    if probability >= 0.80:
        return 0.0
    if probability >= 0.65:
        return 0.2
    if probability >= 0.50:
        return 0.5
    if probability >= 0.35:
        return 0.8
    return 1.0


def _forecast_values(row: dict[str, str], suffix: str) -> list[float]:
    meta_name = f"meta_{suffix}"
    if row.get(meta_name) not in (None, ""):
        return [_finite_float(row[meta_name])]
    values = []
    for name, raw_value in row.items():
        if name == meta_name:
            continue
        if name == suffix or name.endswith(f"_{suffix}"):
            if raw_value in (None, ""):
                continue
            values.append(_finite_float(raw_value))
    return values


def _has_forecast(row: dict[str, str], suffix: str) -> bool:
    try:
        return bool(_forecast_values(row, suffix))
    except (TypeError, ValueError):
        return False


def _forecast_source(
    definition: AllocationPolicyDefinition,
    rows: list[dict[str, str]],
) -> str:
    auxiliary_requirements = [
        requirement
        for requirement in definition.required_prediction_columns
        if requirement != "predicted_probability"
    ]
    if not auxiliary_requirements:
        return "probability_only"
    uses_meta_for_every_requirement = all(
        all(
            any(
                row.get(f"meta_{name}") not in (None, "")
                for name in requirement.split("|")
            )
            for row in rows
        )
        for requirement in auxiliary_requirements
    )
    return (
        "meta_auxiliary"
        if uses_meta_for_every_requirement
        else "source_model_auxiliary"
    )


def _clip_exposure(
    value: float,
    definition: AllocationPolicyDefinition,
) -> float:
    return min(
        definition.exposure_max,
        max(definition.exposure_min, _finite_float(value)),
    )


def _is_probability(value: Any) -> bool:
    try:
        result = _finite_float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= result <= 1.0


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Allocation inputs must be finite")
    return result
