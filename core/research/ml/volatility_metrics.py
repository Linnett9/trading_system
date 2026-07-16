from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "volatility_metric_input_v1"
CONVERSION_CONTRACT = "volatility_representation_conversion_v1"
RESULT_CONTRACT = "volatility_metric_result_v1"
COMPARISON_CONTRACT = "volatility_metric_comparison_v1"
METRIC_VERSION = "1.0"
REPRESENTATIONS = {
    "variance",
    "volatility",
    "annualised_variance",
    "annualised_volatility",
}
STATUSES = {
    "VALID",
    "INSUFFICIENT_DATA",
    "INVALID_INPUT",
    "UNMATCHED_POPULATION",
    "INCOMPATIBLE_REPRESENTATION",
    "UNSUPPORTED_CONFIGURATION",
    "NUMERICAL_FAILURE",
}


class VolatilityMetricInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def volatility_metric_input(
    observation_ids: Sequence[str],
    forecast_values: Sequence[float],
    realised_values: Sequence[float],
    *,
    forecast_representation: str,
    realised_representation: str,
    model_identity: str,
    horizon_identity: str,
    annualisation_factor: float,
    value_unit: str,
    forecast_availability_timestamps: Sequence[str] | None = None,
    realised_maturity_timestamps: Sequence[str] | None = None,
    decision_cutoff_timestamps: Sequence[str] | None = None,
    sample_weights: Sequence[float] | None = None,
    volatility_target: float | Sequence[float] | None = None,
    realised_portfolio_volatility: Sequence[float] | None = None,
    exposure_values: Sequence[float] | None = None,
    benchmark_forecast: Sequence[float] | None = None,
    panel_identity: str = "synthetic_panel",
    fold_identity: str = "synthetic_fold",
    overlapping_horizons: bool = False,
    minimum_observations: int = 1,
) -> dict[str, Any]:
    ids = [str(value) for value in observation_ids]
    n = len(ids)
    if not ids or len(set(ids)) != n:
        raise VolatilityMetricInputError("UNMATCHED_POPULATION", "OBSERVATION_IDENTITIES_NOT_UNIQUE")
    if ids != sorted(ids):
        raise VolatilityMetricInputError("INVALID_INPUT", "OBSERVATIONS_NOT_DETERMINISTICALLY_ORDERED")
    if n < minimum_observations:
        raise VolatilityMetricInputError("INSUFFICIENT_DATA", "OBSERVATION_COUNT_INSUFFICIENT")
    forecasts = _nonnegative_vector(forecast_values, n, "FORECAST")
    realised = _nonnegative_vector(realised_values, n, "REALISED")
    _representation(forecast_representation)
    _representation(realised_representation)
    if not model_identity or not horizon_identity or not value_unit:
        raise VolatilityMetricInputError("INVALID_INPUT", "IDENTITY_OR_UNIT_MISSING")
    factor = float(annualisation_factor)
    if not math.isfinite(factor) or factor <= 0:
        raise VolatilityMetricInputError("INVALID_INPUT", "ANNUALISATION_FACTOR_INVALID")
    weights = _positive_weights(sample_weights, n)
    availability = _timestamps(forecast_availability_timestamps, n, "FORECAST_AVAILABILITY")
    maturity = _timestamps(realised_maturity_timestamps, n, "REALISED_MATURITY")
    cutoffs = _timestamps(decision_cutoff_timestamps, n, "DECISION_CUTOFF")
    for index in range(n):
        if availability and cutoffs and _time(availability[index]) > _time(cutoffs[index]):
            raise VolatilityMetricInputError("INVALID_INPUT", "FORECAST_AVAILABLE_AFTER_DECISION_CUTOFF")
        if availability and maturity and _time(maturity[index]) <= _time(availability[index]):
            raise VolatilityMetricInputError("INVALID_INPUT", "REALISED_VALUE_NOT_MATURE_AFTER_FORECAST")
    target = _optional_scalar_or_vector(volatility_target, n, "VOLATILITY_TARGET")
    portfolio = _optional_vector(realised_portfolio_volatility, n, "REALISED_PORTFOLIO_VOLATILITY", nonnegative=True)
    exposure = _optional_vector(exposure_values, n, "EXPOSURE", nonnegative=False)
    benchmark = _optional_vector(benchmark_forecast, n, "BENCHMARK_FORECAST", nonnegative=True)
    population_checksum = canonical_hash({"contract": INPUT_CONTRACT, "observation_ids": ids})
    timing_checksum = canonical_hash(
        {"availability": availability, "maturity": maturity, "decision_cutoffs": cutoffs}
    )
    value_checksum = canonical_hash(
        {"observation_ids": ids, "forecast_values": forecasts, "realised_values": realised, "sample_weights": weights}
    )
    return {
        "contract_version": INPUT_CONTRACT,
        "observation_ids": ids,
        "forecast_availability_timestamps": availability,
        "realised_maturity_timestamps": maturity,
        "decision_cutoff_timestamps": cutoffs,
        "forecast_values": forecasts,
        "realised_values": realised,
        "forecast_representation": forecast_representation,
        "realised_representation": realised_representation,
        "model_identity": str(model_identity),
        "horizon_identity": str(horizon_identity),
        "annualisation_factor": factor,
        "value_unit": str(value_unit),
        "sample_weights": weights,
        "volatility_target": target,
        "realised_portfolio_volatility": portfolio,
        "exposure_values": exposure,
        "benchmark_forecast": benchmark,
        "panel_identity": str(panel_identity),
        "fold_identity": str(fold_identity),
        "overlapping_horizons": bool(overlapping_horizons),
        "observation_count": n,
        "population_checksum": population_checksum,
        "timing_checksum": timing_checksum,
        "input_value_checksum": value_checksum,
    }


def convert_volatility_representation(
    observation_ids: Sequence[str],
    values: Sequence[float],
    *,
    source_representation: str,
    destination_representation: str,
    annualisation_factor: float,
    source_unit: str,
    destination_unit: str,
) -> dict[str, Any]:
    ids = [str(value) for value in observation_ids]
    if not ids or len(ids) != len(set(ids)) or ids != sorted(ids):
        raise VolatilityMetricInputError("INVALID_INPUT", "CONVERSION_OBSERVATION_IDENTITIES_INVALID")
    source = _representation(source_representation)
    destination = _representation(destination_representation)
    array = np.asarray(_nonnegative_vector(values, len(ids), "CONVERSION_VALUE"), dtype=float)
    factor = float(annualisation_factor)
    if not math.isfinite(factor) or factor <= 0:
        raise VolatilityMetricInputError("INVALID_INPUT", "ANNUALISATION_FACTOR_INVALID")
    daily_variance = _to_daily_variance(array, source, factor)
    converted = _from_daily_variance(daily_variance, destination, factor)
    logical = {
        "contract_version": CONVERSION_CONTRACT,
        "status": "VALID",
        "valid": True,
        "blocking_reasons": [],
        "observation_ids": ids,
        "source_representation": source,
        "destination_representation": destination,
        "source_unit": str(source_unit),
        "destination_unit": str(destination_unit),
        "annualisation_factor": factor,
        "source_values": array.tolist(),
        "converted_values": converted.tolist(),
        "population_checksum": canonical_hash({"contract": INPUT_CONTRACT, "observation_ids": ids}),
        "source_checksum": canonical_hash({"observation_ids": ids, "values": array.tolist()}),
        "conversion_configuration_checksum": canonical_hash(
            {
                "source_representation": source,
                "destination_representation": destination,
                "annualisation_factor": factor,
                "source_unit": source_unit,
                "destination_unit": destination_unit,
            }
        ),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def forecast_error_metrics(
    data: Mapping[str, Any],
    *,
    include_observation_values: bool = True,
    normalised_mse_denominator_policy: str | None = None,
    qlike_forecast_floor: float | None = None,
) -> dict[str, Any]:
    metric_id = "volatility_point_forecast_metrics"
    config = {
        "include_observation_values": bool(include_observation_values),
        "normalised_mse_denominator_policy": normalised_mse_denominator_policy,
        "qlike_forecast_floor": qlike_forecast_floor,
        "qlike_formula": "realised_over_forecast-log(realised_over_forecast)-1",
        "signed_error_orientation": "forecast_minus_realised",
    }
    try:
        base = _validated_input(data)
        if base["forecast_representation"] != base["realised_representation"]:
            raise VolatilityMetricInputError("INCOMPATIBLE_REPRESENTATION", "FORECAST_REALISED_REPRESENTATION_MISMATCH")
        forecast = np.asarray(base["forecast_values"], dtype=float)
        realised = np.asarray(base["realised_values"], dtype=float)
        weights = np.asarray(base["sample_weights"], dtype=float)
        error = forecast - realised
        squared = error**2
        absolute = np.abs(error)
        warnings = ["OVERLAPPING_HORIZON_OUTCOMES"] if base["overlapping_horizons"] else []
        qlike_values = None
        qlike_aggregate = None
        if base["forecast_representation"] in {"variance", "annualised_variance"}:
            qlike_forecast = forecast.copy()
            if np.any(qlike_forecast <= 0):
                if qlike_forecast_floor is None:
                    raise VolatilityMetricInputError("INVALID_INPUT", "QLIKE_FORECAST_VARIANCE_NOT_POSITIVE")
                floor = float(qlike_forecast_floor)
                if not math.isfinite(floor) or floor <= 0:
                    raise VolatilityMetricInputError("INVALID_INPUT", "QLIKE_FORECAST_FLOOR_INVALID")
                qlike_forecast = np.maximum(qlike_forecast, floor)
                warnings.append("QLIKE_FORECAST_FLOOR_APPLIED")
            if np.any(realised == 0):
                raise VolatilityMetricInputError("INVALID_INPUT", "QLIKE_REALISED_VARIANCE_ZERO_UNDEFINED")
            ratio = realised / qlike_forecast
            qlike_values = ratio - np.log(ratio) - 1.0
            qlike_aggregate = _distribution(qlike_values, weights)
        nmse = None
        if normalised_mse_denominator_policy is not None:
            if normalised_mse_denominator_policy != "weighted_mean_realised_squared_v1":
                raise VolatilityMetricInputError("UNSUPPORTED_CONFIGURATION", "NORMALISED_MSE_POLICY_UNSUPPORTED")
            denominator = _weighted_mean(realised**2, weights)
            if denominator <= 0:
                raise VolatilityMetricInputError("INSUFFICIENT_DATA", "NORMALISED_MSE_DENOMINATOR_ZERO")
            nmse = _weighted_mean(squared, weights) / denominator
        calibration = _point_calibration(forecast, realised, weights)
        values = {
            "mse": _weighted_mean(squared, weights),
            "rmse": math.sqrt(_weighted_mean(squared, weights)),
            "mae": _weighted_mean(absolute, weights),
            "median_absolute_error": float(np.median(absolute)),
            "mean_signed_error": _weighted_mean(error, weights),
            "normalised_mse": nmse,
            "under_prediction_count": int(np.sum(error < 0)),
            "over_prediction_count": int(np.sum(error > 0)),
            "exact_prediction_count": int(np.sum(error == 0)),
            "mean_forecast": _weighted_mean(forecast, weights),
            "mean_realised": _weighted_mean(realised, weights),
            "forecast_to_realised_ratio": _safe_ratio(_weighted_mean(forecast, weights), _weighted_mean(realised, weights)),
            "calibration_slope_through_origin": calibration["slope_through_origin"],
            "intercept_slope_regression": calibration["intercept_slope_regression"],
            "calibration_status": calibration["status"],
            "under_forecast_frequency": _weighted_mean((error < 0).astype(float), weights),
            "over_forecast_frequency": _weighted_mean((error > 0).astype(float), weights),
            "forecast_dispersion": math.sqrt(_weighted_mean((forecast - _weighted_mean(forecast, weights)) ** 2, weights)),
            "realised_dispersion": math.sqrt(_weighted_mean((realised - _weighted_mean(realised, weights)) ** 2, weights)),
            "qlike": qlike_aggregate,
            "observation_values": {
                "signed_errors": error.tolist(),
                "absolute_errors": absolute.tolist(),
                "squared_errors": squared.tolist(),
                "qlike_losses": None if qlike_values is None else qlike_values.tolist(),
            } if include_observation_values else None,
        }
        return _result(metric_id, base, config, values, warnings=warnings)
    except (VolatilityMetricInputError, KeyError, TypeError, ValueError, OverflowError) as exc:
        error = exc if isinstance(exc, VolatilityMetricInputError) else VolatilityMetricInputError("INVALID_INPUT", type(exc).__name__)
        return _blocked(metric_id, data, config, error)


def volatility_target_metrics(
    data: Mapping[str, Any],
    *,
    tolerance_bands: Sequence[float] = (0.05, 0.10),
    include_observation_values: bool = True,
) -> dict[str, Any]:
    metric_id = "volatility_target_error_metrics"
    config = {
        "tolerance_bands": [float(value) for value in tolerance_bands],
        "include_observation_values": bool(include_observation_values),
        "target_error_orientation": "realised_minus_target",
    }
    try:
        base = _validated_input(data)
        if not base["realised_portfolio_volatility"] or not base["volatility_target"]:
            raise VolatilityMetricInputError("INVALID_INPUT", "TARGET_OR_REALISED_PORTFOLIO_VOLATILITY_MISSING")
        if base["realised_representation"] not in {"volatility", "annualised_volatility"}:
            raise VolatilityMetricInputError("INCOMPATIBLE_REPRESENTATION", "TARGET_METRICS_REQUIRE_VOLATILITY_REPRESENTATION")
        bands = np.asarray(config["tolerance_bands"], dtype=float)
        if bands.ndim != 1 or np.any(~np.isfinite(bands)) or np.any(bands < 0) or list(bands) != sorted(set(bands.tolist())):
            raise VolatilityMetricInputError("INVALID_INPUT", "TOLERANCE_BANDS_INVALID")
        target = np.asarray(base["volatility_target"], dtype=float)
        realised = np.asarray(base["realised_portfolio_volatility"], dtype=float)
        if np.any(target <= 0):
            raise VolatilityMetricInputError("INVALID_INPUT", "VOLATILITY_TARGET_MUST_BE_POSITIVE")
        weights = np.asarray(base["sample_weights"], dtype=float)
        error = realised - target
        percentage = error / target
        values = {
            "mean_signed_target_error": _weighted_mean(error, weights),
            "mean_absolute_target_error": _weighted_mean(np.abs(error), weights),
            "root_mean_squared_target_error": math.sqrt(_weighted_mean(error**2, weights)),
            "mean_percentage_target_error": _weighted_mean(percentage, weights),
            "mean_absolute_percentage_target_error": _weighted_mean(np.abs(percentage), weights),
            "overshoot_frequency": _weighted_mean((error > 0).astype(float), weights),
            "undershoot_frequency": _weighted_mean((error < 0).astype(float), weights),
            "maximum_overshoot": float(max(np.max(error), 0)),
            "maximum_undershoot": float(max(np.max(-error), 0)),
            "proportion_within_tolerance": {
                str(float(band)): _weighted_mean((np.abs(percentage) <= band + 1e-12).astype(float), weights)
                for band in bands
            },
            "exposure_values_recorded": bool(base["exposure_values"]),
            "exposure_values_used_in_metric": False,
            "observation_values": {
                "signed_target_errors": error.tolist(),
                "absolute_target_errors": np.abs(error).tolist(),
                "squared_target_errors": (error**2).tolist(),
                "percentage_target_errors": percentage.tolist(),
            } if include_observation_values else None,
        }
        warnings = ["OVERLAPPING_HORIZON_OUTCOMES"] if base["overlapping_horizons"] else []
        return _result(metric_id, base, config, values, warnings=warnings)
    except (VolatilityMetricInputError, KeyError, TypeError, ValueError) as exc:
        error = exc if isinstance(exc, VolatilityMetricInputError) else VolatilityMetricInputError("INVALID_INPUT", type(exc).__name__)
        return _blocked(metric_id, data, config, error)


def compare_volatility_results(
    candidate_result: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
    *,
    metric_path: str,
) -> dict[str, Any]:
    config = {"metric_path": metric_path, "lower_is_better": True, "inference_owner": "Ticket 1D-A"}
    try:
        if not candidate_result.get("valid") or not benchmark_result.get("valid"):
            raise VolatilityMetricInputError("INVALID_INPUT", "COMPARISON_RESULT_INVALID")
        compatibility = (
            ("population_checksum", "UNMATCHED_POPULATION", "COMPARISON_POPULATION_MISMATCH"),
            ("horizon_identity", "UNSUPPORTED_CONFIGURATION", "COMPARISON_HORIZON_MISMATCH"),
            ("forecast_representation", "INCOMPATIBLE_REPRESENTATION", "COMPARISON_FORECAST_REPRESENTATION_MISMATCH"),
            ("realised_representation", "INCOMPATIBLE_REPRESENTATION", "COMPARISON_REALISED_REPRESENTATION_MISMATCH"),
            ("annualisation_factor", "UNSUPPORTED_CONFIGURATION", "COMPARISON_ANNUALISATION_MISMATCH"),
            ("timing_checksum", "UNMATCHED_POPULATION", "COMPARISON_TIMING_MISMATCH"),
            ("value_unit", "INCOMPATIBLE_REPRESENTATION", "COMPARISON_UNIT_MISMATCH"),
        )
        for field, status, reason in compatibility:
            if candidate_result.get(field) != benchmark_result.get(field):
                raise VolatilityMetricInputError(status, reason)
        candidate = float(_nested(candidate_result["aggregate_metric_values"], metric_path))
        benchmark = float(_nested(benchmark_result["aggregate_metric_values"], metric_path))
        if not math.isfinite(candidate) or not math.isfinite(benchmark):
            raise VolatilityMetricInputError("INVALID_INPUT", "COMPARISON_METRIC_NON_FINITE")
        difference = candidate - benchmark
        relative = difference / abs(benchmark) if benchmark != 0 else None
        logical = {
            "contract_version": COMPARISON_CONTRACT,
            "metric_id": "matched_volatility_model_comparison",
            "metric_version": METRIC_VERSION,
            "status": "VALID",
            "valid": True,
            "blocking_reasons": [],
            "warnings": [],
            "metric_path": metric_path,
            "candidate_metric": candidate,
            "benchmark_metric": benchmark,
            "absolute_difference_candidate_minus_benchmark": difference,
            "relative_difference": relative,
            "improvement": -difference,
            "improvement_direction": "lower",
            "observation_count": candidate_result["observation_count"],
            "population_checksum": candidate_result["population_checksum"],
            "horizon_identity": candidate_result["horizon_identity"],
            "forecast_representation": candidate_result["forecast_representation"],
            "realised_representation": candidate_result["realised_representation"],
            "value_unit": candidate_result["value_unit"],
            "annualisation_factor": candidate_result["annualisation_factor"],
            "timing_checksum": candidate_result["timing_checksum"],
            "dependency_aware_inference_compatible": True,
            "statistical_significance_performed": False,
            "configuration_checksum": canonical_hash(config),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except (VolatilityMetricInputError, KeyError, TypeError, ValueError) as exc:
        error = exc if isinstance(exc, VolatilityMetricInputError) else VolatilityMetricInputError("INVALID_INPUT", "COMPARISON_METRIC_PATH_INVALID")
        logical = {
            "contract_version": COMPARISON_CONTRACT,
            "metric_id": "matched_volatility_model_comparison",
            "metric_version": METRIC_VERSION,
            "status": error.status,
            "valid": False,
            "blocking_reasons": [error.reason],
            "warnings": [],
            "configuration_checksum": canonical_hash(config),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}


def verify_volatility_result(data: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        config = result.get("metric_configuration")
        if not isinstance(config, Mapping):
            raise VolatilityMetricInputError("INVALID_INPUT", "RESULT_CONFIGURATION_MISSING")
        if result.get("metric_id") == "volatility_point_forecast_metrics":
            expected = forecast_error_metrics(data, **{
                "include_observation_values": config["include_observation_values"],
                "normalised_mse_denominator_policy": config["normalised_mse_denominator_policy"],
                "qlike_forecast_floor": config["qlike_forecast_floor"],
            })
        elif result.get("metric_id") == "volatility_target_error_metrics":
            expected = volatility_target_metrics(data, **{
                "tolerance_bands": config["tolerance_bands"],
                "include_observation_values": config["include_observation_values"],
            })
        else:
            raise VolatilityMetricInputError("UNSUPPORTED_CONFIGURATION", "RESULT_METRIC_ID_UNSUPPORTED")
        for field in (
            "population_checksum", "input_value_checksum", "timing_checksum",
            "horizon_identity", "forecast_representation", "realised_representation",
            "value_unit", "annualisation_factor", "configuration_checksum",
            "aggregate_metric_values", "logical_result_checksum",
        ):
            if result.get(field) != expected.get(field):
                reasons.append(f"{field.upper()}_MISMATCH")
        return {
            "contract_version": "volatility_metric_verification_v1",
            "valid": not reasons,
            "blocking_reasons": sorted(set(reasons)),
        }
    except (VolatilityMetricInputError, KeyError, TypeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, VolatilityMetricInputError) else type(exc).__name__
        return {"contract_version": "volatility_metric_verification_v1", "valid": False, "blocking_reasons": [reason]}


def verify_volatility_conversion(result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        expected = convert_volatility_representation(
            result["observation_ids"],
            result["source_values"],
            source_representation=result["source_representation"],
            destination_representation=result["destination_representation"],
            annualisation_factor=result["annualisation_factor"],
            source_unit=result["source_unit"],
            destination_unit=result["destination_unit"],
        )
        for field in (
            "converted_values", "population_checksum", "source_checksum",
            "conversion_configuration_checksum", "logical_result_checksum",
        ):
            if result.get(field) != expected.get(field):
                reasons.append(f"{field.upper()}_MISMATCH")
        return {
            "contract_version": "volatility_conversion_verification_v1",
            "valid": not reasons,
            "blocking_reasons": sorted(set(reasons)),
        }
    except (VolatilityMetricInputError, KeyError, TypeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, VolatilityMetricInputError) else type(exc).__name__
        return {"contract_version": "volatility_conversion_verification_v1", "valid": False, "blocking_reasons": [reason]}


def verify_volatility_comparison(
    candidate_result: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
    comparison_result: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = []
    try:
        expected = compare_volatility_results(
            candidate_result,
            benchmark_result,
            metric_path=str(comparison_result["metric_path"]),
        )
        for field in (
            "candidate_metric", "benchmark_metric",
            "absolute_difference_candidate_minus_benchmark", "relative_difference",
            "improvement", "population_checksum", "horizon_identity",
            "configuration_checksum", "logical_result_checksum",
        ):
            if comparison_result.get(field) != expected.get(field):
                reasons.append(f"{field.upper()}_MISMATCH")
        return {
            "contract_version": "volatility_comparison_verification_v1",
            "valid": not reasons,
            "blocking_reasons": sorted(set(reasons)),
        }
    except (VolatilityMetricInputError, KeyError, TypeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, VolatilityMetricInputError) else type(exc).__name__
        return {"contract_version": "volatility_comparison_verification_v1", "valid": False, "blocking_reasons": [reason]}


def _validated_input(data: Mapping[str, Any]) -> dict[str, Any]:
    try:
        base = volatility_metric_input(
            data["observation_ids"],
            data["forecast_values"],
            data["realised_values"],
            forecast_representation=str(data["forecast_representation"]),
            realised_representation=str(data["realised_representation"]),
            model_identity=str(data["model_identity"]),
            horizon_identity=str(data["horizon_identity"]),
            annualisation_factor=float(data["annualisation_factor"]),
            value_unit=str(data["value_unit"]),
            forecast_availability_timestamps=data.get("forecast_availability_timestamps") or None,
            realised_maturity_timestamps=data.get("realised_maturity_timestamps") or None,
            decision_cutoff_timestamps=data.get("decision_cutoff_timestamps") or None,
            sample_weights=data.get("sample_weights"),
            volatility_target=data.get("volatility_target") or None,
            realised_portfolio_volatility=data.get("realised_portfolio_volatility") or None,
            exposure_values=data.get("exposure_values") or None,
            benchmark_forecast=data.get("benchmark_forecast") or None,
            panel_identity=str(data.get("panel_identity", "synthetic_panel")),
            fold_identity=str(data.get("fold_identity", "synthetic_fold")),
            overlapping_horizons=bool(data.get("overlapping_horizons", False)),
        )
        supplied = data.get("population_checksum")
        if supplied and supplied != base["population_checksum"]:
            raise VolatilityMetricInputError("UNMATCHED_POPULATION", "POPULATION_CHECKSUM_MISMATCH")
        return base
    except KeyError as exc:
        raise VolatilityMetricInputError("INVALID_INPUT", f"INPUT_FIELD_MISSING:{exc.args[0]}") from exc


def _result(metric_id, base, config, values, *, warnings=()):
    logical = {
        "contract_version": RESULT_CONTRACT,
        "metric_id": metric_id,
        "metric_version": METRIC_VERSION,
        "status": "VALID",
        "valid": True,
        "blocking_reasons": [],
        "warnings": sorted(set(warnings)),
        "model_identity": base["model_identity"],
        "observation_count": base["observation_count"],
        "horizon_identity": base["horizon_identity"],
        "forecast_representation": base["forecast_representation"],
        "realised_representation": base["realised_representation"],
        "value_unit": base["value_unit"],
        "annualisation_factor": base["annualisation_factor"],
        "overlapping_horizon_disclosure": base["overlapping_horizons"],
        "aggregate_metric_values": _jsonable(values),
        "weighting_convention": "positive supplied weights; normalised by weighted mean",
        "population_checksum": base["population_checksum"],
        "input_value_checksum": base["input_value_checksum"],
        "timing_checksum": base["timing_checksum"],
        "metric_configuration": _jsonable(config),
        "configuration_checksum": canonical_hash(config),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked(metric_id, data, config, error):
    logical = {
        "contract_version": RESULT_CONTRACT,
        "metric_id": metric_id,
        "metric_version": METRIC_VERSION,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False,
        "blocking_reasons": [error.reason],
        "warnings": [],
        "model_identity": data.get("model_identity") if isinstance(data, Mapping) else None,
        "observation_count": len(data.get("observation_ids", ())) if isinstance(data, Mapping) else 0,
        "horizon_identity": data.get("horizon_identity") if isinstance(data, Mapping) else None,
        "forecast_representation": data.get("forecast_representation") if isinstance(data, Mapping) else None,
        "realised_representation": data.get("realised_representation") if isinstance(data, Mapping) else None,
        "value_unit": data.get("value_unit") if isinstance(data, Mapping) else None,
        "annualisation_factor": data.get("annualisation_factor") if isinstance(data, Mapping) else None,
        "overlapping_horizon_disclosure": data.get("overlapping_horizons") if isinstance(data, Mapping) else None,
        "aggregate_metric_values": {},
        "weighting_convention": "none",
        "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "input_value_checksum": data.get("input_value_checksum") if isinstance(data, Mapping) else None,
        "timing_checksum": data.get("timing_checksum") if isinstance(data, Mapping) else None,
        "metric_configuration": _jsonable(config),
        "configuration_checksum": canonical_hash(config),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _point_calibration(forecast, realised, weights):
    denominator = float(weights @ (forecast**2))
    slope = None if denominator <= 0 else float(weights @ (forecast * realised) / denominator)
    if len(forecast) < 3:
        regression = None
        status = "INSUFFICIENT_DATA"
    else:
        design = np.column_stack([np.ones(len(forecast)), forecast])
        weighted_design = design * np.sqrt(weights)[:, None]
        if np.linalg.matrix_rank(weighted_design) < 2:
            regression = None
            status = "INSUFFICIENT_DATA"
        else:
            coefficients = np.linalg.lstsq(weighted_design, realised * np.sqrt(weights), rcond=None)[0]
            regression = {"intercept": float(coefficients[0]), "slope": float(coefficients[1])}
            status = "VALID"
    return {"slope_through_origin": slope, "intercept_slope_regression": regression, "status": status}


def _distribution(values, weights):
    return {
        "weighted_mean": _weighted_mean(values, weights),
        "median": float(np.median(values)),
        "quantiles": {
            "0.05": float(np.quantile(values, 0.05)),
            "0.50": float(np.quantile(values, 0.50)),
            "0.95": float(np.quantile(values, 0.95)),
        },
        "lower_is_better": True,
        "zero_realised_policy": "ratio_form_qlike_rejects_zero_realised_variance",
    }


def _to_daily_variance(values, representation, factor):
    if representation == "variance":
        return values
    if representation == "volatility":
        return values**2
    if representation == "annualised_variance":
        return values / factor
    return (values / math.sqrt(factor)) ** 2


def _from_daily_variance(values, representation, factor):
    if representation == "variance":
        return values
    if representation == "volatility":
        return np.sqrt(values)
    if representation == "annualised_variance":
        return values * factor
    return np.sqrt(values) * math.sqrt(factor)


def _representation(value):
    value = str(value)
    if value not in REPRESENTATIONS:
        raise VolatilityMetricInputError("INCOMPATIBLE_REPRESENTATION", "REPRESENTATION_UNSUPPORTED")
    return value


def _nonnegative_vector(values, n, owner):
    result = _optional_vector(values, n, owner, nonnegative=True)
    if result is None:
        raise VolatilityMetricInputError("INVALID_INPUT", f"{owner}_MISSING")
    return result


def _optional_vector(values, n, owner, *, nonnegative):
    if values is None:
        return []
    result = [float(value) for value in values]
    if len(result) != n:
        raise VolatilityMetricInputError("UNMATCHED_POPULATION", f"{owner}_POPULATION_MISMATCH")
    if not all(math.isfinite(value) for value in result):
        raise VolatilityMetricInputError("INVALID_INPUT", f"{owner}_NON_FINITE")
    if nonnegative and any(value < 0 for value in result):
        raise VolatilityMetricInputError("INVALID_INPUT", f"{owner}_NEGATIVE")
    return result


def _optional_scalar_or_vector(values, n, owner):
    if values is None:
        return []
    if isinstance(values, (int, float)):
        result = [float(values)] * n
    else:
        result = [float(value) for value in values]
    if len(result) != n:
        raise VolatilityMetricInputError("UNMATCHED_POPULATION", f"{owner}_POPULATION_MISMATCH")
    if not all(math.isfinite(value) and value >= 0 for value in result):
        raise VolatilityMetricInputError("INVALID_INPUT", f"{owner}_INVALID")
    return result


def _positive_weights(values, n):
    if values is None:
        return [1.0] * n
    result = [float(value) for value in values]
    if len(result) != n:
        raise VolatilityMetricInputError("UNMATCHED_POPULATION", "SAMPLE_WEIGHT_POPULATION_MISMATCH")
    if not all(math.isfinite(value) and value > 0 for value in result) or sum(result) <= 0:
        raise VolatilityMetricInputError("INVALID_INPUT", "SAMPLE_WEIGHTS_INVALID")
    return result


def _timestamps(values, n, owner):
    if values is None:
        return []
    result = [str(value) for value in values]
    if len(result) != n:
        raise VolatilityMetricInputError("UNMATCHED_POPULATION", f"{owner}_POPULATION_MISMATCH")
    for value in result:
        _time(value)
    return result


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise VolatilityMetricInputError("INVALID_INPUT", "TIMESTAMP_INVALID") from exc


def _weighted_mean(values, weights):
    return float(np.average(values, weights=weights))


def _safe_ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _nested(value, path):
    current = value
    for key in path.split("."):
        current = current[key]
    return current


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
