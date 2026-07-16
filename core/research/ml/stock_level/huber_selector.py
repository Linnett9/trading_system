from __future__ import annotations

import json
import math
import platform
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "huber_selector_input_v1"
PREPROCESSING_CONTRACT = "huber_selector_preprocessing_v1"
MODEL_CONTRACT = "huber_selector_model_v1"
PREDICTION_CONTRACT = "huber_selector_prediction_v1"
FIT_RESULT_CONTRACT = "huber_selector_fit_result_v1"
STABILITY_CONTRACT = "huber_selector_coefficient_stability_v1"
CONTROL_CONTRACT = "huber_selector_linear_control_comparison_v1"
STATUSES = {
    "READY", "INVALID_INPUT", "INSUFFICIENT_DATA", "TEMPORAL_VIOLATION",
    "FEATURE_SCHEMA_MISMATCH", "PREPROCESSING_FAILURE", "NON_CONVERGENCE",
    "NONFINITE_PREDICTION", "DEGENERATE_RANKING", "DEPENDENCY_UNAVAILABLE",
    "NUMERICAL_FAILURE",
}
DEFAULT_EPSILON = 1.35
DEFAULT_ALPHA = 0.0001
DEFAULT_TOLERANCE = 1e-5
DEFAULT_MAX_ITERATIONS = 100


class HuberSelectorError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def huber_selector_input(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_horizon: str,
    target_contract_identity: str,
    feature_schema_identity: str,
    training_fold_identity: str,
    validation_fold_identity: str,
    dataset_identity: str,
    source_population_checksum: str,
) -> dict[str, Any]:
    normalised = []
    row_ids = []
    expected_features = None
    for raw in rows:
        row_id = str(raw.get("row_id") or "")
        asset_id = str(raw.get("asset_id") or "")
        decision = str(raw.get("decision_timestamp") or "")
        maturity = str(raw.get("target_maturity_timestamp") or "")
        availability = str(raw.get("feature_availability_timestamp") or decision)
        feature_ids = [str(value) for value in raw.get("feature_ids", ())]
        values = [float(value) for value in raw.get("feature_values", ())]
        target = float(raw.get("target_value"))
        weight = float(raw.get("sample_weight", 1.0))
        split = str(raw.get("split") or "")
        if not row_id or not asset_id or split not in {"TRAINING", "VALIDATION"}:
            raise HuberSelectorError("INVALID_INPUT", "ROW_IDENTITY_OR_SPLIT_INVALID")
        if len(feature_ids) != len(values) or not feature_ids or len(feature_ids) != len(set(feature_ids)):
            raise HuberSelectorError("FEATURE_SCHEMA_MISMATCH", "FEATURE_DIMENSION_OR_IDENTITY_INVALID")
        if feature_ids != sorted(feature_ids):
            raise HuberSelectorError("FEATURE_SCHEMA_MISMATCH", "FEATURE_ORDER_NOT_CANONICAL")
        if expected_features is None:
            expected_features = feature_ids
        elif feature_ids != expected_features:
            raise HuberSelectorError("FEATURE_SCHEMA_MISMATCH", "FEATURE_ORDER_MISMATCH")
        if not all(math.isfinite(value) for value in values):
            raise HuberSelectorError("INVALID_INPUT", "FEATURE_VALUE_NON_FINITE")
        if not math.isfinite(target):
            raise HuberSelectorError("INVALID_INPUT", "TARGET_VALUE_NON_FINITE")
        if not math.isfinite(weight) or weight <= 0:
            raise HuberSelectorError("INVALID_INPUT", "SAMPLE_WEIGHT_INVALID")
        if _time(availability) > _time(decision):
            raise HuberSelectorError("TEMPORAL_VIOLATION", "FEATURE_AVAILABLE_AFTER_DECISION")
        _time(maturity)
        row_ids.append(row_id)
        normalised.append({
            "row_id": row_id, "asset_id": asset_id, "decision_timestamp": decision,
            "feature_availability_timestamp": availability, "feature_ids": feature_ids,
            "feature_values": values, "target_value": target,
            "target_maturity_timestamp": maturity, "sample_weight": weight, "split": split,
        })
    if not normalised or len(row_ids) != len(set(row_ids)):
        raise HuberSelectorError("INVALID_INPUT", "ROW_IDENTITIES_NOT_UNIQUE")
    ordered = sorted(normalised, key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))
    if normalised != ordered:
        raise HuberSelectorError("INVALID_INPUT", "ROWS_NOT_DETERMINISTICALLY_ORDERED")
    logical = {
        "contract_version": INPUT_CONTRACT, "rows": normalised,
        "ordered_feature_ids": expected_features, "target_horizon": str(target_horizon),
        "target_contract_identity": str(target_contract_identity),
        "feature_schema_identity": str(feature_schema_identity),
        "training_fold_identity": str(training_fold_identity),
        "validation_fold_identity": str(validation_fold_identity),
        "dataset_identity": str(dataset_identity),
        "source_population_checksum": str(source_population_checksum),
        "row_population_checksum": canonical_hash(row_ids),
        "feature_schema_checksum": canonical_hash({"identity": feature_schema_identity, "feature_ids": expected_features}),
        "target_contract_checksum": canonical_hash({"identity": target_contract_identity, "horizon": target_horizon}),
        "input_value_checksum": canonical_hash(normalised),
    }
    logical["logical_input_checksum"] = canonical_hash(logical)
    return logical


def fit_huber_selector(
    data: Mapping[str, Any],
    *,
    epsilon: float = DEFAULT_EPSILON,
    alpha: float = DEFAULT_ALPHA,
    fit_intercept: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    maximum_iterations: int = DEFAULT_MAX_ITERATIONS,
    warm_start: bool = False,
    winsorisation_limits: tuple[float, float] | None = None,
    minimum_training_rows: int = 5,
    minimum_rank_diversity: int = 2,
    source_commit: str | None = None,
) -> dict[str, Any]:
    config = {
        "epsilon": epsilon, "alpha": alpha, "fit_intercept": fit_intercept,
        "tolerance": tolerance, "maximum_iterations": maximum_iterations,
        "warm_start": warm_start, "winsorisation_limits": winsorisation_limits,
        "minimum_training_rows": minimum_training_rows,
        "minimum_rank_diversity": minimum_rank_diversity,
        "scaler": "training_standardisation_ddof0_v1",
    }
    try:
        base = _validated_input(data)
        if epsilon < 1 or alpha < 0 or tolerance <= 0 or maximum_iterations < 1:
            raise HuberSelectorError("INVALID_INPUT", "HUBER_CONFIGURATION_INVALID")
        training = [row for row in base["rows"] if row["split"] == "TRAINING"]
        validation = [row for row in base["rows"] if row["split"] == "VALIDATION"]
        if len(training) < minimum_training_rows:
            raise HuberSelectorError("INSUFFICIENT_DATA", "TRAINING_SAMPLE_INADEQUATE")
        if not validation:
            raise HuberSelectorError("INSUFFICIENT_DATA", "VALIDATION_SAMPLE_EMPTY")
        training_cutoff = max(row["decision_timestamp"] for row in training)
        validation_start = min(row["decision_timestamp"] for row in validation)
        maximum_maturity = max(row["target_maturity_timestamp"] for row in training)
        if training_cutoff >= validation_start:
            raise HuberSelectorError("TEMPORAL_VIOLATION", "TRAINING_VALIDATION_BOUNDARY_OVERLAP")
        if maximum_maturity > validation_start:
            raise HuberSelectorError("TEMPORAL_VIOLATION", "TRAINING_TARGET_NOT_MATURE_BY_VALIDATION")
        x_train = np.asarray([row["feature_values"] for row in training], dtype=float)
        y_train = np.asarray([row["target_value"] for row in training], dtype=float)
        weights = np.asarray([row["sample_weight"] for row in training], dtype=float)
        x_validation = np.asarray([row["feature_values"] for row in validation], dtype=float)
        preprocessing = _fit_preprocessing(x_train, base["ordered_feature_ids"], winsorisation_limits)
        train_scaled = _transform(x_train, preprocessing)
        validation_scaled = _transform(x_validation, preprocessing)
        try:
            import sklearn
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import HuberRegressor
        except ImportError:
            raise HuberSelectorError("DEPENDENCY_UNAVAILABLE", "SKLEARN_HUBER_UNAVAILABLE")
        estimator = HuberRegressor(
            epsilon=epsilon, alpha=alpha, fit_intercept=fit_intercept,
            tol=tolerance, max_iter=maximum_iterations, warm_start=warm_start,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(train_scaled, y_train, sample_weight=weights)
        convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        if convergence_warnings or int(estimator.n_iter_) >= maximum_iterations:
            raise HuberSelectorError("NON_CONVERGENCE", "HUBER_ESTIMATOR_DID_NOT_CONVERGE")
        predictions = np.asarray(estimator.predict(validation_scaled), dtype=float)
        if not np.isfinite(predictions).all():
            raise HuberSelectorError("NONFINITE_PREDICTION", "VALIDATION_PREDICTION_NON_FINITE")
        prediction_rows = _rank_predictions(validation, predictions)
        diversity = len({row["within_date_rank"] for row in prediction_rows})
        if diversity < minimum_rank_diversity:
            raise HuberSelectorError("DEGENERATE_RANKING", "VALIDATION_RANK_DIVERSITY_INADEQUATE")
        model = _model_contract(
            base, preprocessing, estimator, sklearn.__version__, config,
            training_cutoff, maximum_maturity, source_commit,
        )
        for row in prediction_rows:
            row.update({
                "contract_version": PREDICTION_CONTRACT,
                "model_checksum": model["model_checksum"],
                "dataset_checksum": base["source_population_checksum"],
                "feature_schema_checksum": base["feature_schema_checksum"],
                "target_contract_checksum": base["target_contract_checksum"],
                "fold_identity": base["validation_fold_identity"],
                "training_cutoff": training_cutoff,
                "maximum_label_availability_timestamp": maximum_maturity,
            })
        prediction_checksum = canonical_hash(prediction_rows)
        residuals = y_train - estimator.predict(train_scaled)
        diagnostics = {
            "coefficient_vector": estimator.coef_.tolist(),
            "intercept": float(estimator.intercept_),
            "robust_scale": float(estimator.scale_),
            "number_of_iterations": int(estimator.n_iter_),
            "convergence_status": "CONVERGED",
            "training_residual_summary": _summary(residuals),
            "validation_prediction_summary": _summary(predictions),
            "outlier_count": int(np.sum(estimator.outliers_)),
            "outlier_mask": estimator.outliers_.tolist(),
            "coefficient_norm": float(np.linalg.norm(estimator.coef_)),
            "maximum_absolute_coefficient": float(np.max(np.abs(estimator.coef_))),
            "feature_coefficients": [
                {"feature_id": feature, "coefficient": float(value)}
                for feature, value in zip(base["ordered_feature_ids"], estimator.coef_)
            ],
            "prediction_dispersion": float(np.std(predictions)),
            "rank_diversity": diversity,
            "training_count": len(training), "validation_count": len(validation),
            "diagnostic_is_promotion_evidence": False,
        }
        logical = {
            "contract_version": FIT_RESULT_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "estimator_identity": "sklearn.linear_model.HuberRegressor",
            "dependency_name": "scikit-learn", "dependency_version": sklearn.__version__,
            "observation_counts": {"training": len(training), "validation": len(validation)},
            "feature_count": len(base["ordered_feature_ids"]),
            "training_fold_identity": base["training_fold_identity"],
            "validation_fold_identity": base["validation_fold_identity"],
            "temporal_integrity": "TRAINING_ONLY_PREPROCESSING_AND_FIT",
            "preprocessing": preprocessing,
            "preprocessing_checksum": preprocessing["preprocessing_checksum"],
            "model": model, "model_checksum": model["model_checksum"],
            "predictions": prediction_rows, "prediction_checksum": prediction_checksum,
            "diagnostic_summary": diagnostics,
            "configuration": config, "configuration_checksum": canonical_hash(config),
            "input_checksum": base["logical_input_checksum"],
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except HuberSelectorError as exc:
        return _blocked(data, config, exc)
    except Exception as exc:  # pragma: no cover
        return _blocked(data, config, HuberSelectorError("NUMERICAL_FAILURE", type(exc).__name__))


def coefficient_stability(results: Sequence[Mapping[str, Any]], *, unstable_sign_consistency_threshold: float = 0.75) -> dict[str, Any]:
    try:
        if len(results) < 2 or any(not result.get("valid") for result in results):
            raise HuberSelectorError("INSUFFICIENT_DATA", "STABILITY_REQUIRES_TWO_VALID_FITS")
        schemas = {result["model"]["feature_schema_checksum"] for result in results}
        features = [result["model"]["ordered_feature_ids"] for result in results]
        if len(schemas) != 1 or any(value != features[0] for value in features):
            raise HuberSelectorError("FEATURE_SCHEMA_MISMATCH", "STABILITY_FEATURE_SCHEMA_MISMATCH")
        coefficients = np.asarray([result["diagnostic_summary"]["coefficient_vector"] for result in results])
        intercepts = np.asarray([result["diagnostic_summary"]["intercept"] for result in results])
        scales = np.asarray([result["diagnostic_summary"]["robust_scale"] for result in results])
        sign_consistency = np.maximum(np.mean(coefficients > 0, axis=0), np.mean(coefficients < 0, axis=0))
        correlations = []
        for left in range(len(coefficients)):
            for right in range(left + 1, len(coefficients)):
                correlations.append(_spearman(coefficients[left], coefficients[right]))
        logical = {
            "contract_version": STABILITY_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "feature_schema_checksum": next(iter(schemas)),
            "ordered_feature_ids": features[0],
            "coefficient_mean": np.mean(coefficients, axis=0).tolist(),
            "coefficient_median": np.median(coefficients, axis=0).tolist(),
            "coefficient_standard_deviation": np.std(coefficients, axis=0).tolist(),
            "coefficient_sign_consistency": sign_consistency.tolist(),
            "maximum_coefficient_range": float(np.max(np.ptp(coefficients, axis=0))),
            "coefficient_rank_correlations": correlations,
            "unstable_feature_ids": [
                feature for feature, consistency in zip(features[0], sign_consistency)
                if consistency < unstable_sign_consistency_threshold
            ],
            "intercept_dispersion": float(np.std(intercepts)),
            "robust_scale_dispersion": float(np.std(scales)),
            "convergence_consistency": all(result["diagnostic_summary"]["convergence_status"] == "CONVERGED" for result in results),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return logical
    except HuberSelectorError as exc:
        logical = {"contract_version": STABILITY_CONTRACT, "status": exc.status, "valid": False, "blocking_reasons": [exc.reason]}
        logical["logical_result_checksum"] = canonical_hash(logical)
        return logical


def compare_huber_with_ols(clean_data: Mapping[str, Any], outlier_data: Mapping[str, Any], **fit_options) -> dict[str, Any]:
    clean = fit_huber_selector(clean_data, **fit_options)
    outlier = fit_huber_selector(outlier_data, **fit_options)
    if not clean.get("valid") or not outlier.get("valid"):
        return {"contract_version": CONTROL_CONTRACT, "status": "INVALID_INPUT", "valid": False, "blocking_reasons": ["HUBER_FIT_INVALID"]}
    from sklearn.linear_model import LinearRegression
    base = _validated_input(clean_data)
    changed = _validated_input(outlier_data)
    if base["row_population_checksum"] != changed["row_population_checksum"]:
        return {"contract_version": CONTROL_CONTRACT, "status": "INVALID_INPUT", "valid": False, "blocking_reasons": ["CONTROL_POPULATION_MISMATCH"]}
    training = [row for row in base["rows"] if row["split"] == "TRAINING"]
    validation = [row for row in base["rows"] if row["split"] == "VALIDATION"]
    changed_training = [row for row in changed["rows"] if row["split"] == "TRAINING"]
    preprocessing = clean["preprocessing"]
    x_train = _transform(np.asarray([row["feature_values"] for row in training]), preprocessing)
    x_validation = _transform(np.asarray([row["feature_values"] for row in validation]), preprocessing)
    weights = np.asarray([row["sample_weight"] for row in training])
    ols_clean = LinearRegression().fit(x_train, [row["target_value"] for row in training], sample_weight=weights)
    ols_outlier = LinearRegression().fit(x_train, [row["target_value"] for row in changed_training], sample_weight=weights)
    y_validation = np.asarray([row["target_value"] for row in validation])
    huber_clean_predictions = np.asarray([row["predicted_return"] for row in clean["predictions"]])
    huber_outlier_predictions = np.asarray([row["predicted_return"] for row in outlier["predictions"]])
    ols_clean_predictions = ols_clean.predict(x_validation)
    ols_outlier_predictions = ols_outlier.predict(x_validation)
    logical = {
        "contract_version": CONTROL_CONTRACT, "status": "READY", "valid": True, "blocking_reasons": [],
        "huber": _control_metrics(y_validation, huber_outlier_predictions, outlier["diagnostic_summary"]["coefficient_norm"]),
        "ols": _control_metrics(y_validation, ols_outlier_predictions, float(np.linalg.norm(ols_outlier.coef_))),
        "huber_prediction_change_from_outlier": float(np.mean(np.abs(huber_outlier_predictions - huber_clean_predictions))),
        "ols_prediction_change_from_outlier": float(np.mean(np.abs(ols_outlier_predictions - ols_clean_predictions))),
        "huber_coefficient_change_from_outlier": float(np.linalg.norm(
            np.asarray(outlier["diagnostic_summary"]["coefficient_vector"]) - np.asarray(clean["diagnostic_summary"]["coefficient_vector"])
        )),
        "ols_coefficient_change_from_outlier": float(np.linalg.norm(ols_outlier.coef_ - ols_clean.coef_)),
        "historical_superiority_claimed": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def verify_huber_result(data: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    config = result.get("configuration", {})
    expected = fit_huber_selector(
        data, epsilon=config.get("epsilon", DEFAULT_EPSILON), alpha=config.get("alpha", DEFAULT_ALPHA),
        fit_intercept=config.get("fit_intercept", True), tolerance=config.get("tolerance", DEFAULT_TOLERANCE),
        maximum_iterations=config.get("maximum_iterations", DEFAULT_MAX_ITERATIONS),
        warm_start=config.get("warm_start", False),
        winsorisation_limits=tuple(config["winsorisation_limits"]) if config.get("winsorisation_limits") else None,
        minimum_training_rows=config.get("minimum_training_rows", 5),
        minimum_rank_diversity=config.get("minimum_rank_diversity", 2),
        source_commit=result.get("model", {}).get("source_commit"),
    )
    fields = (
        "status", "preprocessing", "preprocessing_checksum", "model", "model_checksum",
        "predictions", "prediction_checksum", "diagnostic_summary", "configuration_checksum",
        "input_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": "huber_selector_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def _fit_preprocessing(matrix, feature_ids, winsorisation_limits):
    transformed = matrix.copy()
    lower_values = upper_values = None
    if winsorisation_limits is not None:
        lower, upper = winsorisation_limits
        if not 0 <= lower < upper <= 1:
            raise HuberSelectorError("PREPROCESSING_FAILURE", "WINSORISATION_LIMITS_INVALID")
        lower_values = np.quantile(transformed, lower, axis=0)
        upper_values = np.quantile(transformed, upper, axis=0)
        transformed = np.clip(transformed, lower_values, upper_values)
    mean = np.mean(transformed, axis=0)
    raw_scale = np.std(transformed, axis=0)
    constant = raw_scale == 0
    scale = np.where(constant, 1.0, raw_scale)
    logical = {
        "contract_version": PREPROCESSING_CONTRACT,
        "identity": "training_standardisation_ddof0_v1",
        "ordered_feature_ids": list(feature_ids), "location": mean.tolist(), "scale": scale.tolist(),
        "constant_feature_ids": [feature for feature, value in zip(feature_ids, constant) if value],
        "constant_feature_policy": "center_then_unit_scale",
        "winsorisation_limits": winsorisation_limits,
        "winsorisation_lower_values": None if lower_values is None else lower_values.tolist(),
        "winsorisation_upper_values": None if upper_values is None else upper_values.tolist(),
        "fitted_on": "TRAINING_ONLY",
    }
    logical["preprocessing_checksum"] = canonical_hash(logical)
    return logical


def _transform(matrix, preprocessing):
    values = np.asarray(matrix, dtype=float)
    if preprocessing["winsorisation_limits"] is not None:
        values = np.clip(values, preprocessing["winsorisation_lower_values"], preprocessing["winsorisation_upper_values"])
    transformed = (values - np.asarray(preprocessing["location"])) / np.asarray(preprocessing["scale"])
    if not np.isfinite(transformed).all():
        raise HuberSelectorError("PREPROCESSING_FAILURE", "PREPROCESSING_OUTPUT_NON_FINITE")
    return transformed


def _model_contract(base, preprocessing, estimator, dependency_version, config, cutoff, maturity, source_commit):
    logical = {
        "contract_version": MODEL_CONTRACT,
        "estimator_identity": "sklearn.linear_model.HuberRegressor",
        "estimator_dependency": "scikit-learn", "estimator_dependency_version": dependency_version,
        "loss_convention": "quadratic for standardized residual magnitude below epsilon; linear beyond epsilon",
        "regularisation_convention": "L2 alpha penalty on coefficients",
        "scale_estimation": "joint robust scale estimated by HuberRegressor",
        "epsilon": config["epsilon"], "alpha": config["alpha"], "fit_intercept": config["fit_intercept"],
        "tolerance": config["tolerance"], "maximum_iterations": config["maximum_iterations"],
        "warm_start": config["warm_start"], "preprocessing_identity": preprocessing["identity"],
        "preprocessing_checksum": preprocessing["preprocessing_checksum"],
        "ordered_feature_ids": base["ordered_feature_ids"],
        "feature_schema_checksum": base["feature_schema_checksum"],
        "target_contract_checksum": base["target_contract_checksum"],
        "training_population_checksum": canonical_hash([
            row["row_id"] for row in base["rows"] if row["split"] == "TRAINING"
        ]),
        "training_cutoff": cutoff, "maximum_training_label_maturity_timestamp": maturity,
        "random_seed": None, "source_commit": source_commit,
        "coefficient_vector": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
        "robust_scale": float(estimator.scale_), "iteration_count": int(estimator.n_iter_),
    }
    logical["model_checksum"] = canonical_hash(logical)
    return logical


def _rank_predictions(validation, predictions):
    grouped = {}
    for row, prediction in zip(validation, predictions):
        grouped.setdefault(row["decision_timestamp"], []).append((row, float(prediction)))
    output = []
    for decision, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]))
        count = len(ranked)
        for rank, (row, prediction) in enumerate(ranked, 1):
            output.append({
                "row_id": row["row_id"], "asset_id": row["asset_id"],
                "decision_timestamp": decision, "predicted_return": prediction,
                "within_date_rank": rank,
                "within_date_percentile_rank": (count - rank) / max(count - 1, 1),
                "prediction_semantics": "continuous_return_score_not_probability",
            })
    return sorted(output, key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))


def _validated_input(data):
    base = huber_selector_input(
        data["rows"], target_horizon=data["target_horizon"],
        target_contract_identity=data["target_contract_identity"],
        feature_schema_identity=data["feature_schema_identity"],
        training_fold_identity=data["training_fold_identity"],
        validation_fold_identity=data["validation_fold_identity"],
        dataset_identity=data["dataset_identity"],
        source_population_checksum=data["source_population_checksum"],
    )
    if data.get("logical_input_checksum") and data["logical_input_checksum"] != base["logical_input_checksum"]:
        raise HuberSelectorError("INVALID_INPUT", "INPUT_CHECKSUM_MISMATCH")
    return base


def _control_metrics(actual, predictions, coefficient_norm):
    error = predictions - actual
    return {
        "prediction_mse": float(np.mean(error**2)), "prediction_mae": float(np.mean(np.abs(error))),
        "spearman_rank_correlation": _spearman(predictions, actual),
        "coefficient_norm": float(coefficient_norm),
    }


def _spearman(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def _ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)), "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)), "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _blocked(data, config, error):
    logical = {
        "contract_version": FIT_RESULT_CONTRACT,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
        "estimator_identity": "sklearn.linear_model.HuberRegressor",
        "dependency_name": "scikit-learn", "dependency_version": None,
        "observation_counts": {}, "feature_count": None,
        "training_fold_identity": data.get("training_fold_identity") if isinstance(data, Mapping) else None,
        "validation_fold_identity": data.get("validation_fold_identity") if isinstance(data, Mapping) else None,
        "temporal_integrity": "FAILED", "preprocessing": {}, "preprocessing_checksum": None,
        "model": {}, "model_checksum": None, "predictions": [], "prediction_checksum": None,
        "diagnostic_summary": {}, "configuration": config,
        "configuration_checksum": canonical_hash(config),
        "input_checksum": data.get("logical_input_checksum") if isinstance(data, Mapping) else None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HuberSelectorError("TEMPORAL_VIOLATION", "TIMESTAMP_INVALID") from exc


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
