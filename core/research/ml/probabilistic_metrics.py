from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_SCHEMA = "probabilistic_prediction_input_v1"
RESULT_SCHEMA = "probabilistic_metric_result_v1"
PROBABILITY_TOLERANCE = 1e-8
LOG_LOSS_CLIP = 1e-15
CALIBRATION_POLICY = "equal_width_top_confidence_10_bins_v1"
INTERVAL_SCORE_VERSION = "winkler_interval_score_v1"
STATUSES = {
    "VALID", "INSUFFICIENT_DATA", "INVALID_INPUT", "UNMATCHED_POPULATION",
    "UNSUPPORTED_PREDICTION_TYPE", "NUMERICAL_FAILURE",
}


class ProbabilisticInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def probabilistic_input(
    observation_ids: Sequence[str],
    realised_targets: Sequence[float],
    *,
    prediction_type: str,
    model_id: str,
    decision_timestamps: Sequence[str] | None = None,
    target_maturity_timestamps: Sequence[str] | None = None,
    prediction_availability_timestamps: Sequence[str] | None = None,
    sample_weights: Sequence[float] | None = None,
    target_unit: str = "unitless",
    orientation: str = "higher_target_is_better",
    minimum_observations: int = 1,
) -> dict[str, Any]:
    ids = [str(value) for value in observation_ids]
    if len(ids) != len(set(ids)):
        raise ProbabilisticInputError("UNMATCHED_POPULATION", "OBSERVATION_IDENTITIES_NOT_UNIQUE")
    if ids != sorted(ids):
        raise ProbabilisticInputError("INVALID_INPUT", "OBSERVATIONS_NOT_DETERMINISTICALLY_ORDERED")
    targets = _finite_vector(realised_targets, "realised_targets")
    if len(ids) != len(targets):
        raise ProbabilisticInputError("UNMATCHED_POPULATION", "TARGET_POPULATION_MISMATCH")
    if len(ids) < minimum_observations:
        raise ProbabilisticInputError("INSUFFICIENT_DATA", "OBSERVATION_COUNT_INSUFFICIENT")
    if not model_id:
        raise ProbabilisticInputError("INVALID_INPUT", "MODEL_ID_MISSING")
    weights = _weights(sample_weights, len(ids))
    decision = _optional_strings(decision_timestamps, len(ids), "DECISION_TIMESTAMP_POPULATION_MISMATCH")
    maturity = _optional_strings(target_maturity_timestamps, len(ids), "TARGET_MATURITY_POPULATION_MISMATCH")
    availability = _optional_strings(prediction_availability_timestamps, len(ids), "PREDICTION_AVAILABILITY_POPULATION_MISMATCH")
    if maturity and availability:
        for prediction_time, maturity_time in zip(availability, maturity):
            if prediction_time > maturity_time:
                raise ProbabilisticInputError("INVALID_INPUT", "PREDICTION_AFTER_TARGET_MATURITY")
    population_checksum = canonical_hash({"contract": INPUT_SCHEMA, "observation_ids": ids})
    return {
        "contract_schema": INPUT_SCHEMA, "observation_ids": ids,
        "realised_targets": targets, "prediction_type": prediction_type,
        "model_id": model_id, "decision_timestamps": decision,
        "target_maturity_timestamps": maturity,
        "prediction_availability_timestamps": availability,
        "sample_weights": weights, "target_unit": target_unit,
        "orientation": orientation, "observation_count": len(ids),
        "model_count": 1, "population_checksum": population_checksum,
    }


def classification_metrics(
    data: Mapping[str, Any],
    *,
    class_probabilities: Sequence[Sequence[float]],
    class_labels: Sequence[str] | None = None,
    calibration_bins: int = 10,
    clipping_epsilon: float = LOG_LOSS_CLIP,
) -> dict[str, Any]:
    metric_id = "multiclass_probabilistic_metrics"
    config = {
        "calibration_bins": calibration_bins, "calibration_policy": CALIBRATION_POLICY,
        "clipping_epsilon": clipping_epsilon, "probability_tolerance": PROBABILITY_TOLERANCE,
    }
    try:
        base = _validated_base(data, prediction_type="classification")
        probabilities = np.asarray(class_probabilities, dtype=float)
        n = base["observation_count"]
        if probabilities.ndim != 2 or probabilities.shape[0] != n or probabilities.shape[1] < 2:
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "CLASS_PROBABILITY_SHAPE_MISMATCH")
        if not np.isfinite(probabilities).all():
            raise ProbabilisticInputError("INVALID_INPUT", "CLASS_PROBABILITY_NON_FINITE")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ProbabilisticInputError("INVALID_INPUT", "CLASS_PROBABILITY_OUT_OF_BOUNDS")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0):
            raise ProbabilisticInputError("INVALID_INPUT", "CLASS_PROBABILITY_SUM_INVALID")
        if calibration_bins < 2:
            raise ProbabilisticInputError("INVALID_INPUT", "CALIBRATION_BIN_COUNT_INVALID")
        if not 0 < clipping_epsilon < 0.5:
            raise ProbabilisticInputError("INVALID_INPUT", "LOG_LOSS_CLIP_INVALID")
        labels = np.asarray(base["realised_targets"], dtype=int)
        class_count = probabilities.shape[1]
        if np.any(labels < 0) or np.any(labels >= class_count) or not np.all(labels == np.asarray(base["realised_targets"])):
            raise ProbabilisticInputError("INVALID_INPUT", "REALISED_CLASS_LABEL_INVALID")
        names = list(class_labels or [str(index) for index in range(class_count)])
        if len(names) != class_count or len(names) != len(set(names)):
            raise ProbabilisticInputError("INVALID_INPUT", "CLASS_LABELS_INVALID")
        weights = np.asarray(base["sample_weights"], dtype=float)
        clipped = np.clip(probabilities, clipping_epsilon, 1.0)
        nll_rows = -np.log(clipped[np.arange(n), labels])
        one_hot = np.eye(class_count)[labels]
        brier_rows = np.sum((probabilities - one_hot) ** 2, axis=1)
        cumulative_probabilities = np.cumsum(probabilities, axis=1)[:, :-1]
        cumulative_outcomes = np.cumsum(one_hot, axis=1)[:, :-1]
        rps_rows = np.sum((cumulative_probabilities - cumulative_outcomes) ** 2, axis=1) / (class_count - 1)
        expected_relevance = probabilities @ np.arange(class_count, dtype=float)
        top_class = np.argmax(probabilities, axis=1)
        confidence = probabilities[np.arange(n), top_class]
        correct = (top_class == labels).astype(float)
        calibration = _calibration(confidence, correct, weights, calibration_bins)
        classwise = {
            names[index]: _calibration(probabilities[:, index], one_hot[:, index], weights, calibration_bins)
            for index in range(class_count)
        }
        metrics = {
            "negative_log_likelihood": _weighted_mean(nll_rows, weights),
            "multiclass_brier_score": _weighted_mean(brier_rows, weights),
            "ranked_probability_score": _weighted_mean(rps_rows, weights),
            "expected_relevance": expected_relevance.tolist(),
            "mean_expected_relevance": _weighted_mean(expected_relevance, weights),
            "expected_calibration_error": calibration["expected_calibration_error"],
            "calibration_bins": calibration["bins"],
            "classwise_calibration": classwise,
            "mean_probability_concentration": _weighted_mean(np.sum(probabilities**2, axis=1), weights),
            "mean_predictive_entropy": _weighted_mean(-np.sum(np.where(probabilities > 0, probabilities * np.log(np.clip(probabilities, clipping_epsilon, 1)), 0), axis=1), weights),
            "top_class_accuracy_diagnostic": _weighted_mean(correct, weights),
            "top_class_accuracy_is_primary": False,
        }
        return _result(metric_id, "1.0", base, config, metrics, aggregation="weighted_mean", weighting=_weighting_policy(base))
    except ProbabilisticInputError as exc:
        return _blocked(metric_id, "1.0", data, config, exc)


def quantile_metrics(
    data: Mapping[str, Any],
    *,
    quantile_levels: Sequence[float],
    quantile_forecasts: Sequence[Sequence[float]],
    require_non_crossing: bool = False,
) -> dict[str, Any]:
    metric_id = "quantile_prediction_metrics"
    config = {"quantile_levels": list(quantile_levels), "require_non_crossing": require_non_crossing, "equality_policy": "realised_equal_forecast_has_zero_pinball_loss"}
    try:
        base = _validated_base(data, prediction_type="quantile")
        levels = np.asarray(quantile_levels, dtype=float)
        if levels.ndim != 1 or len(levels) < 1 or not np.isfinite(levels).all():
            raise ProbabilisticInputError("INVALID_INPUT", "QUANTILE_LEVELS_INVALID")
        if np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) <= 0):
            raise ProbabilisticInputError("INVALID_INPUT", "QUANTILE_LEVELS_NOT_STRICTLY_INCREASING")
        forecasts = np.asarray(quantile_forecasts, dtype=float)
        if forecasts.shape != (base["observation_count"], len(levels)):
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "QUANTILE_FORECAST_SHAPE_MISMATCH")
        if not np.isfinite(forecasts).all():
            raise ProbabilisticInputError("INVALID_INPUT", "QUANTILE_FORECAST_NON_FINITE")
        crossings = np.maximum(forecasts[:, :-1] - forecasts[:, 1:], 0)
        crossing_count = int(np.sum(crossings > 0))
        crossing_severity = float(crossings.sum())
        if crossing_count and require_non_crossing:
            raise ProbabilisticInputError("INVALID_INPUT", "QUANTILE_CROSSING_FORBIDDEN")
        targets = np.asarray(base["realised_targets"])[:, None]
        errors = targets - forecasts
        losses = np.maximum(levels * errors, (levels - 1) * errors)
        weights = np.asarray(base["sample_weights"])
        per_quantile = {}
        for index, level in enumerate(levels):
            indicator = (targets[:, 0] <= forecasts[:, index]).astype(float)
            empirical = _weighted_mean(indicator, weights)
            per_quantile[str(float(level))] = {
                "pinball_loss": _weighted_mean(losses[:, index], weights),
                "empirical_below_or_equal_rate": empirical,
                "calibration_error": empirical - float(level),
                "exceedance_rate": 1 - empirical,
            }
        lower = [str(float(level)) for level in levels if level < 0.5]
        upper = [str(float(level)) for level in levels if level > 0.5]
        warnings = ["QUANTILE_CROSSING_PRESENT"] if crossing_count else []
        metrics = {
            "per_quantile": per_quantile,
            "mean_pinball_loss": _weighted_mean(losses.mean(axis=1), weights),
            "weighted_pinball_loss": float(np.average(losses, weights=weights, axis=0).mean()),
            "quantile_crossing_count": crossing_count,
            "quantile_crossing_severity": crossing_severity,
            "lower_tail_quantiles": lower, "upper_tail_quantiles": upper,
        }
        return _result(metric_id, "1.0", base, config, metrics, warnings=warnings, aggregation="weighted_mean", weighting=_weighting_policy(base))
    except ProbabilisticInputError as exc:
        return _blocked(metric_id, "1.0", data, config, exc)


def empirical_crps(
    data: Mapping[str, Any],
    *,
    predictive_samples: Sequence[Sequence[float]],
) -> dict[str, Any]:
    metric_id = "empirical_sample_crps"
    config = {"formulation": "E|X-y|-0.5E|X-X_prime|", "minimum_predictive_samples": 2}
    try:
        base = _validated_base(data, prediction_type="samples")
        samples = np.asarray(predictive_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] != base["observation_count"]:
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "PREDICTIVE_SAMPLE_SHAPE_MISMATCH")
        if samples.shape[1] < 2:
            raise ProbabilisticInputError("INSUFFICIENT_DATA", "PREDICTIVE_SAMPLE_COUNT_INSUFFICIENT")
        if not np.isfinite(samples).all():
            raise ProbabilisticInputError("INVALID_INPUT", "PREDICTIVE_SAMPLE_NON_FINITE")
        targets = np.asarray(base["realised_targets"])
        first = np.mean(np.abs(samples - targets[:, None]), axis=1)
        pairwise = np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
        rows = first - 0.5 * pairwise
        metrics = {"observation_crps": rows.tolist(), "mean_crps": _weighted_mean(rows, np.asarray(base["sample_weights"])), "predictive_sample_count": samples.shape[1]}
        return _result(metric_id, "empirical_v1", base, config, metrics, aggregation="weighted_mean", weighting=_weighting_policy(base))
    except ProbabilisticInputError as exc:
        return _blocked(metric_id, "empirical_v1", data, config, exc)


def prediction_interval_metrics(
    data: Mapping[str, Any],
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    nominal_coverage: float,
    target_buckets: Sequence[str] | None = None,
) -> dict[str, Any]:
    metric_id = "prediction_interval_diagnostics"
    config = {"nominal_coverage": nominal_coverage, "interval_score": INTERVAL_SCORE_VERSION}
    try:
        base = _validated_base(data, prediction_type="interval")
        n = base["observation_count"]
        lower = np.asarray(_finite_vector(lower_bounds, "lower_bounds"))
        upper = np.asarray(_finite_vector(upper_bounds, "upper_bounds"))
        if len(lower) != n or len(upper) != n:
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "INTERVAL_POPULATION_MISMATCH")
        if np.any(lower > upper):
            raise ProbabilisticInputError("INVALID_INPUT", "INTERVAL_BOUNDS_CROSSED")
        if not 0 < nominal_coverage < 1:
            raise ProbabilisticInputError("INVALID_INPUT", "NOMINAL_COVERAGE_INVALID")
        targets = np.asarray(base["realised_targets"])
        weights = np.asarray(base["sample_weights"])
        covered = ((targets >= lower) & (targets <= upper)).astype(float)
        width = upper - lower
        alpha = 1 - nominal_coverage
        lower_miss = targets < lower
        upper_miss = targets > upper
        score = width + (2 / alpha) * (lower - targets) * lower_miss + (2 / alpha) * (targets - upper) * upper_miss
        buckets = _optional_strings(target_buckets, n, "TARGET_BUCKET_POPULATION_MISMATCH")
        conditional = {}
        if buckets:
            for bucket in sorted(set(buckets)):
                mask = np.asarray([value == bucket for value in buckets])
                conditional[bucket] = {
                    "count": int(mask.sum()), "coverage": _weighted_mean(covered[mask], weights[mask]),
                    "mean_width": _weighted_mean(width[mask], weights[mask]),
                }
        empirical = _weighted_mean(covered, weights)
        metrics = {
            "empirical_coverage": empirical, "coverage_error": empirical - nominal_coverage,
            "mean_interval_width": _weighted_mean(width, weights),
            "median_interval_width": float(np.median(width)),
            "mean_interval_score": _weighted_mean(score, weights),
            "under_coverage_count": int(np.sum(~covered.astype(bool))),
            "over_coverage_count": int(np.sum(covered)),
            "lower_bound_miss_count": int(lower_miss.sum()),
            "upper_bound_miss_count": int(upper_miss.sum()),
            "conditional_coverage": conditional,
            "calibration_sharpness_summary": {"absolute_coverage_error": abs(empirical - nominal_coverage), "mean_width": _weighted_mean(width, weights)},
        }
        return _result(metric_id, "1.0", base, config, metrics, aggregation="weighted_mean", weighting=_weighting_policy(base))
    except ProbabilisticInputError as exc:
        return _blocked(metric_id, "1.0", data, config, exc)


def distribution_negative_log_likelihood(
    data: Mapping[str, Any],
    *,
    family: str,
    parameters: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    metric_id = "distribution_negative_log_likelihood"
    config = {"family": family, "parameterisation": "gaussian_location_scale" if family == "gaussian" else None}
    try:
        base = _validated_base(data, prediction_type="distribution")
        if family != "gaussian":
            raise ProbabilisticInputError("UNSUPPORTED_PREDICTION_TYPE", f"DISTRIBUTION_FAMILY_UNSUPPORTED:{family}")
        means = np.asarray(_finite_vector(parameters.get("mean", ()), "gaussian_mean"))
        scales = np.asarray(_finite_vector(parameters.get("scale", ()), "gaussian_scale"))
        n = base["observation_count"]
        if len(means) != n or len(scales) != n:
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "DISTRIBUTION_PARAMETER_POPULATION_MISMATCH")
        if np.any(scales <= 0):
            raise ProbabilisticInputError("INVALID_INPUT", "GAUSSIAN_SCALE_NON_POSITIVE")
        targets = np.asarray(base["realised_targets"])
        rows = 0.5 * np.log(2 * math.pi * scales**2) + ((targets - means) ** 2) / (2 * scales**2)
        metrics = {"family": family, "parameterisation": "mean_and_positive_standard_deviation", "observation_negative_log_likelihood": rows.tolist(), "mean_negative_log_likelihood": _weighted_mean(rows, np.asarray(base["sample_weights"]))}
        return _result(metric_id, "gaussian_v1", base, config, metrics, aggregation="weighted_mean", weighting=_weighting_policy(base))
    except ProbabilisticInputError as exc:
        return _blocked(metric_id, "gaussian_v1", data, config, exc)


def matched_metric_comparison(
    candidate_result: Mapping[str, Any],
    benchmark_result: Mapping[str, Any],
    *,
    metric_path: str,
    lower_is_better: bool,
) -> dict[str, Any]:
    metric_id = "matched_probabilistic_metric_comparison"
    config = {"metric_path": metric_path, "lower_is_better": lower_is_better, "significance_owner": "Ticket 1D-A"}
    try:
        if not candidate_result.get("valid") or not benchmark_result.get("valid"):
            raise ProbabilisticInputError("INVALID_INPUT", "COMPARISON_INPUT_RESULT_INVALID")
        if candidate_result.get("population_checksum") != benchmark_result.get("population_checksum"):
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "COMPARISON_POPULATION_MISMATCH")
        candidate_value = float(_nested(candidate_result.get("metric_values", {}), metric_path))
        benchmark_value = float(_nested(benchmark_result.get("metric_values", {}), metric_path))
        if not math.isfinite(candidate_value) or not math.isfinite(benchmark_value):
            raise ProbabilisticInputError("INVALID_INPUT", "COMPARISON_METRIC_NON_FINITE")
        difference = candidate_value - benchmark_value
        improvement = -difference if lower_is_better else difference
        base = {
            "observation_count": candidate_result["observation_count"], "model_count": 2,
            "population_checksum": candidate_result["population_checksum"],
            "prediction_type": candidate_result["prediction_type"],
            "target_unit": candidate_result["target_unit"],
        }
        metrics = {
            "candidate_metric_value": candidate_value, "benchmark_metric_value": benchmark_value,
            "metric_difference_candidate_minus_benchmark": difference,
            "improvement_value": improvement, "direction_of_improvement": "lower" if lower_is_better else "higher",
            "matched_observation_count": base["observation_count"],
            "block_bootstrap_compatible": True, "statistical_significance_performed": False,
        }
        return _result(metric_id, "1.0", base, config, metrics, aggregation="matched_comparison", weighting="inherited")
    except (ProbabilisticInputError, KeyError, TypeError, ValueError) as exc:
        error = exc if isinstance(exc, ProbabilisticInputError) else ProbabilisticInputError("INVALID_INPUT", "COMPARISON_METRIC_PATH_INVALID")
        return _blocked(metric_id, "1.0", {}, config, error)


def _validated_base(data: Mapping[str, Any], *, prediction_type: str) -> dict[str, Any]:
    try:
        base = probabilistic_input(
            data["observation_ids"], data["realised_targets"], prediction_type=prediction_type,
            model_id=str(data.get("model_id") or "model"),
            decision_timestamps=data.get("decision_timestamps") or None,
            target_maturity_timestamps=data.get("target_maturity_timestamps") or None,
            prediction_availability_timestamps=data.get("prediction_availability_timestamps") or None,
            sample_weights=data.get("sample_weights"), target_unit=str(data.get("target_unit", "unitless")),
            orientation=str(data.get("orientation", "higher_target_is_better")),
        )
        supplied = data.get("population_checksum")
        if supplied and supplied != base["population_checksum"]:
            raise ProbabilisticInputError("UNMATCHED_POPULATION", "POPULATION_CHECKSUM_MISMATCH")
        return base
    except KeyError as exc:
        raise ProbabilisticInputError("INVALID_INPUT", f"INPUT_FIELD_MISSING:{exc.args[0]}") from exc


def _calibration(probabilities: np.ndarray, outcomes: np.ndarray, weights: np.ndarray, bin_count: int) -> dict[str, Any]:
    edges = np.linspace(0, 1, bin_count + 1)
    indexes = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bin_count - 1)
    indexes = np.maximum(indexes, 0)
    rows = []
    total = float(weights.sum())
    ece = 0.0
    for index in range(bin_count):
        mask = indexes == index
        weight = float(weights[mask].sum())
        if weight == 0:
            rows.append({"bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]), "weight": 0.0, "mean_probability": None, "empirical_frequency": None})
            continue
        mean_probability = _weighted_mean(probabilities[mask], weights[mask])
        empirical = _weighted_mean(outcomes[mask], weights[mask])
        ece += (weight / total) * abs(mean_probability - empirical)
        rows.append({"bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]), "weight": weight, "mean_probability": mean_probability, "empirical_frequency": empirical})
    return {"expected_calibration_error": ece, "bins": rows}


def _result(metric_id: str, version: str, base: Mapping[str, Any], config: Mapping[str, Any], values: Mapping[str, Any], *, warnings: Sequence[str] = (), aggregation: str, weighting: str) -> dict[str, Any]:
    logical = {
        "schema_version": RESULT_SCHEMA, "metric_id": metric_id, "metric_version": version,
        "prediction_type": base.get("prediction_type"), "status": "VALID", "valid": True,
        "blocking_reasons": [], "warnings": sorted(set(warnings)),
        "observation_count": int(base.get("observation_count", 0)), "model_count": int(base.get("model_count", 1)),
        "population_checksum": base.get("population_checksum"), "configuration_checksum": canonical_hash(_identity_safe(config)),
        "metric_values": _jsonable(values), "aggregation_policy": aggregation, "weighting_policy": weighting,
        "target_unit": base.get("target_unit"),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked(metric_id: str, version: str, data: Mapping[str, Any], config: Mapping[str, Any], error: ProbabilisticInputError) -> dict[str, Any]:
    status = error.status if error.status in STATUSES else "INVALID_INPUT"
    logical = {
        "schema_version": RESULT_SCHEMA, "metric_id": metric_id, "metric_version": version,
        "prediction_type": data.get("prediction_type") if isinstance(data, Mapping) else None,
        "status": status, "valid": False, "blocking_reasons": [error.reason], "warnings": [],
        "observation_count": len(data.get("observation_ids", ())) if isinstance(data, Mapping) else 0,
        "model_count": 1, "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": canonical_hash(_identity_safe(config)), "metric_values": {},
        "aggregation_policy": "none", "weighting_policy": "none",
        "target_unit": data.get("target_unit") if isinstance(data, Mapping) else None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _weights(values: Sequence[float] | None, n: int) -> list[float]:
    if values is None:
        return [1.0] * n
    weights = _finite_vector(values, "sample_weights")
    if len(weights) != n:
        raise ProbabilisticInputError("UNMATCHED_POPULATION", "SAMPLE_WEIGHT_POPULATION_MISMATCH")
    if any(value < 0 for value in weights) or sum(weights) <= 0:
        raise ProbabilisticInputError("INVALID_INPUT", "SAMPLE_WEIGHTS_INVALID")
    return weights


def _finite_vector(values: Sequence[float], owner: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ProbabilisticInputError("INVALID_INPUT", f"NON_FINITE_VALUE:{owner}")
    return result


def _optional_strings(values: Sequence[str] | None, n: int, reason: str) -> list[str]:
    if values is None:
        return []
    result = [str(value) for value in values]
    if len(result) != n:
        raise ProbabilisticInputError("UNMATCHED_POPULATION", reason)
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighting_policy(base: Mapping[str, Any]) -> str:
    return "uniform" if len(set(base["sample_weights"])) == 1 else "explicit_nonnegative_sample_weights"


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return value


def _creation_metadata() -> dict[str, Any]:
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}


def _identity_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _identity_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_identity_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite_float": repr(value)}
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
