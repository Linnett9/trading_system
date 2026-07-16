from __future__ import annotations

import json
import math
import platform
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "contextual_elastic_net_input_v1"
INTERACTION_CONTRACT = "contextual_interaction_contract_v1"
PREPROCESSING_CONTRACT = "contextual_elastic_net_preprocessing_v1"
MODEL_CONTRACT = "contextual_elastic_net_model_v1"
PREDICTION_CONTRACT = "contextual_elastic_net_prediction_v1"
FIT_RESULT_CONTRACT = "contextual_elastic_net_fit_result_v1"
STABILITY_CONTRACT = "contextual_elastic_net_stability_v1"
COMPARISON_CONTRACT = "contextual_elastic_net_incremental_comparison_v1"
DEFAULT_ALPHA = 0.001
DEFAULT_L1_RATIO = 0.25
DEFAULT_TOLERANCE = 1e-4
DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_INTERACTIONS = (
    ("momentum_x_market_volatility", "momentum", "market_volatility"),
    ("momentum_x_market_trend", "momentum", "market_trend"),
    ("drawdown_recovery_x_market_drawdown", "drawdown_recovery", "market_drawdown"),
    ("risk_adjusted_momentum_x_market_volatility", "risk_adjusted_momentum", "market_volatility"),
    ("liquidity_x_market_volatility", "liquidity", "market_volatility"),
    ("stock_volatility_x_market_volatility", "stock_volatility", "market_volatility"),
)
STATUSES = {
    "READY", "INVALID_INPUT", "INSUFFICIENT_DATA", "TEMPORAL_VIOLATION",
    "FEATURE_SCHEMA_MISMATCH", "CONTEXT_SCHEMA_MISMATCH",
    "INTERACTION_CONTRACT_MISMATCH", "INCONSISTENT_DATE_CONTEXT",
    "PREPROCESSING_FAILURE", "NON_CONVERGENCE", "NONFINITE_PREDICTION",
    "DEGENERATE_RANKING", "DEPENDENCY_UNAVAILABLE", "NUMERICAL_FAILURE",
}


class ContextualElasticNetError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def contextual_interaction_contract(
    stock_feature_ids: Sequence[str],
    market_context_ids: Sequence[str],
    *,
    interactions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    stock = list(stock_feature_ids)
    context = list(market_context_ids)
    entries = list(interactions) if interactions is not None else [
        {
            "interaction_id": interaction_id, "stock_feature_id": stock_id,
            "market_context_id": context_id, "transformation": "scaled_product",
            "point_in_time_availability_rule": "both_inputs_available_by_decision_timestamp",
            "output_unit": "standardised_product", "interaction_order": index,
        }
        for index, (interaction_id, stock_id, context_id) in enumerate(DEFAULT_INTERACTIONS)
        if stock_id in stock and context_id in context
    ]
    if len(entries) == len(stock) * len(context) and len(entries) > len(DEFAULT_INTERACTIONS):
        raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "FULL_ALL_PAIRS_EXPANSION_FORBIDDEN")
    seen = set()
    normalised = []
    for index, raw in enumerate(entries):
        stock_id = str(raw.get("stock_feature_id") or "")
        context_id = str(raw.get("market_context_id") or "")
        identity = str(raw.get("interaction_id") or "")
        key = (stock_id, context_id)
        if stock_id not in stock:
            raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", f"UNKNOWN_STOCK_FEATURE:{stock_id}")
        if context_id not in context:
            raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", f"UNKNOWN_CONTEXT_FEATURE:{context_id}")
        if not identity or key in seen:
            raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "DUPLICATE_OR_AMBIGUOUS_INTERACTION")
        seen.add(key)
        row = {
            "interaction_id": identity, "stock_feature_id": stock_id,
            "market_context_id": context_id,
            "transformation": str(raw.get("transformation", "scaled_product")),
            "point_in_time_availability_rule": str(raw.get(
                "point_in_time_availability_rule", "both_inputs_available_by_decision_timestamp"
            )),
            "output_unit": str(raw.get("output_unit", "standardised_product")),
            "interaction_order": int(raw.get("interaction_order", index)),
        }
        if row["transformation"] != "scaled_product":
            raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "INTERACTION_TRANSFORMATION_UNSUPPORTED")
        row["interaction_checksum"] = canonical_hash(row)
        normalised.append(row)
    normalised.sort(key=lambda row: (row["interaction_order"], row["interaction_id"]))
    if [row["interaction_order"] for row in normalised] != list(range(len(normalised))):
        raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "INTERACTION_ORDER_INVALID")
    logical = {
        "contract_version": INTERACTION_CONTRACT,
        "interaction_policy": "explicit_bounded_registered_list_only",
        "interactions": normalised,
    }
    logical["interaction_contract_checksum"] = canonical_hash(logical)
    return logical


def contextual_elastic_net_input(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_horizon: str,
    stock_feature_schema_identity: str,
    market_context_schema_identity: str,
    interaction_contract_identity: str,
    training_fold_identity: str,
    validation_fold_identity: str,
    dataset_identity: str,
    source_population_checksum: str,
) -> dict[str, Any]:
    normalised, row_ids = [], []
    stock_order = context_order = None
    contexts_by_date = {}
    for raw in rows:
        row_id, asset = str(raw.get("row_id") or ""), str(raw.get("asset_id") or "")
        decision = str(raw.get("decision_timestamp") or "")
        availability = str(raw.get("feature_availability_timestamp") or decision)
        maturity = str(raw.get("target_maturity_timestamp") or "")
        split = str(raw.get("split") or "")
        stock_ids = [str(value) for value in raw.get("stock_feature_ids", ())]
        context_ids = [str(value) for value in raw.get("market_context_ids", ())]
        stock_values = _finite(raw.get("stock_feature_values", ()), "STOCK_FEATURE")
        context_values = _finite(raw.get("market_context_values", ()), "CONTEXT_FEATURE")
        target = float(raw.get("target_value"))
        weight = float(raw.get("sample_weight", 1.0))
        if not row_id or not asset or split not in {"TRAINING", "VALIDATION"}:
            raise ContextualElasticNetError("INVALID_INPUT", "ROW_IDENTITY_OR_SPLIT_INVALID")
        if stock_ids != sorted(stock_ids) or len(stock_ids) != len(stock_values) or len(set(stock_ids)) != len(stock_ids):
            raise ContextualElasticNetError("FEATURE_SCHEMA_MISMATCH", "STOCK_FEATURE_SCHEMA_INVALID")
        if context_ids != sorted(context_ids) or len(context_ids) != len(context_values) or len(set(context_ids)) != len(context_ids):
            raise ContextualElasticNetError("CONTEXT_SCHEMA_MISMATCH", "CONTEXT_FEATURE_SCHEMA_INVALID")
        if stock_order is None:
            stock_order, context_order = stock_ids, context_ids
        elif stock_ids != stock_order:
            raise ContextualElasticNetError("FEATURE_SCHEMA_MISMATCH", "STOCK_FEATURE_ORDER_MISMATCH")
        elif context_ids != context_order:
            raise ContextualElasticNetError("CONTEXT_SCHEMA_MISMATCH", "CONTEXT_FEATURE_ORDER_MISMATCH")
        if not math.isfinite(target):
            raise ContextualElasticNetError("INVALID_INPUT", "TARGET_NON_FINITE")
        if not math.isfinite(weight) or weight <= 0:
            raise ContextualElasticNetError("INVALID_INPUT", "SAMPLE_WEIGHT_INVALID")
        if _time(availability) > _time(decision):
            raise ContextualElasticNetError("TEMPORAL_VIOLATION", "FEATURE_AVAILABLE_AFTER_DECISION")
        _time(maturity)
        context_tuple = tuple(context_values)
        if decision in contexts_by_date and contexts_by_date[decision] != context_tuple:
            raise ContextualElasticNetError("INCONSISTENT_DATE_CONTEXT", "MULTIPLE_CONTEXT_VECTORS_FOR_DATE")
        contexts_by_date[decision] = context_tuple
        row_ids.append(row_id)
        normalised.append({
            "row_id": row_id, "asset_id": asset, "decision_timestamp": decision,
            "feature_availability_timestamp": availability,
            "stock_feature_ids": stock_ids, "stock_feature_values": stock_values,
            "market_context_ids": context_ids, "market_context_values": context_values,
            "target_value": target, "target_maturity_timestamp": maturity,
            "sample_weight": weight, "split": split,
        })
    if not normalised or len(row_ids) != len(set(row_ids)):
        raise ContextualElasticNetError("INVALID_INPUT", "ROW_IDENTITIES_NOT_UNIQUE")
    if normalised != sorted(normalised, key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"])):
        raise ContextualElasticNetError("INVALID_INPUT", "ROWS_NOT_DETERMINISTICALLY_ORDERED")
    logical = {
        "contract_version": INPUT_CONTRACT, "rows": normalised,
        "ordered_stock_feature_ids": stock_order,
        "ordered_market_context_ids": context_order,
        "target_horizon": str(target_horizon),
        "stock_feature_schema_identity": str(stock_feature_schema_identity),
        "market_context_schema_identity": str(market_context_schema_identity),
        "interaction_contract_identity": str(interaction_contract_identity),
        "training_fold_identity": str(training_fold_identity),
        "validation_fold_identity": str(validation_fold_identity),
        "dataset_identity": str(dataset_identity),
        "source_population_checksum": str(source_population_checksum),
        "row_population_checksum": canonical_hash(row_ids),
        "stock_schema_checksum": canonical_hash({"identity": stock_feature_schema_identity, "ids": stock_order}),
        "context_schema_checksum": canonical_hash({"identity": market_context_schema_identity, "ids": context_order}),
        "input_value_checksum": canonical_hash(normalised),
    }
    logical["logical_input_checksum"] = canonical_hash(logical)
    return logical


def fit_contextual_elastic_net(
    data: Mapping[str, Any],
    interaction_contract: Mapping[str, Any],
    *,
    alpha: float = DEFAULT_ALPHA,
    l1_ratio: float = DEFAULT_L1_RATIO,
    fit_intercept: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    maximum_iterations: int = DEFAULT_MAX_ITERATIONS,
    selection: str = "cyclic",
    random_seed: int = 0,
    warm_start: bool = False,
    include_context_main_effects: bool = False,
    winsorisation_limits: tuple[float, float] | None = None,
    minimum_training_rows: int = 8,
    minimum_rank_diversity: int = 2,
    source_commit: str | None = None,
) -> dict[str, Any]:
    config = {
        "alpha": alpha, "l1_ratio": l1_ratio, "fit_intercept": fit_intercept,
        "tolerance": tolerance, "maximum_iterations": maximum_iterations,
        "selection": selection, "random_seed": random_seed, "warm_start": warm_start,
        "include_context_main_effects": include_context_main_effects,
        "winsorisation_limits": winsorisation_limits,
        "minimum_training_rows": minimum_training_rows,
        "minimum_rank_diversity": minimum_rank_diversity,
        "preprocessing_order": "scale_bases_on_training_then_construct_scaled_products_v1",
    }
    try:
        base = _validated_input(data)
        interactions = _validated_interactions(interaction_contract, base)
        if alpha < 0 or not 0 <= l1_ratio <= 1 or tolerance <= 0 or maximum_iterations < 1 or selection not in {"cyclic", "random"}:
            raise ContextualElasticNetError("INVALID_INPUT", "ELASTIC_NET_CONFIGURATION_INVALID")
        training = [row for row in base["rows"] if row["split"] == "TRAINING"]
        validation = [row for row in base["rows"] if row["split"] == "VALIDATION"]
        if len(training) < minimum_training_rows:
            raise ContextualElasticNetError("INSUFFICIENT_DATA", "TRAINING_SAMPLE_INADEQUATE")
        if not validation:
            raise ContextualElasticNetError("INSUFFICIENT_DATA", "VALIDATION_SAMPLE_EMPTY")
        cutoff = max(row["decision_timestamp"] for row in training)
        validation_start = min(row["decision_timestamp"] for row in validation)
        maturity = max(row["target_maturity_timestamp"] for row in training)
        if cutoff >= validation_start:
            raise ContextualElasticNetError("TEMPORAL_VIOLATION", "TRAINING_VALIDATION_BOUNDARY_OVERLAP")
        if maturity > validation_start:
            raise ContextualElasticNetError("TEMPORAL_VIOLATION", "TRAINING_TARGET_NOT_MATURE_BY_VALIDATION")
        stock_train = np.asarray([row["stock_feature_values"] for row in training])
        context_train = np.asarray([row["market_context_values"] for row in training])
        stock_validation = np.asarray([row["stock_feature_values"] for row in validation])
        context_validation = np.asarray([row["market_context_values"] for row in validation])
        preprocessing = _fit_preprocessing(
            stock_train, context_train, base["ordered_stock_feature_ids"],
            base["ordered_market_context_ids"], winsorisation_limits,
        )
        x_train, lineage = _design_matrix(
            stock_train, context_train, preprocessing, interactions,
            include_context_main_effects,
        )
        x_validation, validation_lineage = _design_matrix(
            stock_validation, context_validation, preprocessing, interactions,
            include_context_main_effects,
        )
        if lineage != validation_lineage:
            raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "DESIGN_LINEAGE_CHANGED")
        try:
            import sklearn
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import ElasticNet
        except ImportError:
            raise ContextualElasticNetError("DEPENDENCY_UNAVAILABLE", "SKLEARN_ELASTIC_NET_UNAVAILABLE")
        estimator = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
            tol=tolerance, max_iter=maximum_iterations, selection=selection,
            random_state=random_seed, warm_start=warm_start,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(
                x_train, [row["target_value"] for row in training],
                sample_weight=[row["sample_weight"] for row in training],
            )
        if any(issubclass(item.category, ConvergenceWarning) for item in caught) or int(estimator.n_iter_) >= maximum_iterations:
            raise ContextualElasticNetError("NON_CONVERGENCE", "ELASTIC_NET_DID_NOT_CONVERGE")
        if not np.isfinite(estimator.coef_).all():
            raise ContextualElasticNetError("NUMERICAL_FAILURE", "COEFFICIENT_NON_FINITE")
        predictions = estimator.predict(x_validation)
        if not np.isfinite(predictions).all():
            raise ContextualElasticNetError("NONFINITE_PREDICTION", "VALIDATION_PREDICTION_NON_FINITE")
        prediction_rows = _rank_predictions(validation, predictions)
        diversity = len({row["within_date_rank"] for row in prediction_rows})
        if diversity < minimum_rank_diversity:
            raise ContextualElasticNetError("DEGENERATE_RANKING", "VALIDATION_RANK_DIVERSITY_INADEQUATE")
        design_checksum = canonical_hash({"columns": lineage, "training_matrix": x_train.tolist()})
        model = _model(
            base, interactions, preprocessing, lineage, estimator, sklearn.__version__,
            config, cutoff, maturity, source_commit, design_checksum,
        )
        for row in prediction_rows:
            row.update({
                "contract_version": PREDICTION_CONTRACT, "model_checksum": model["model_checksum"],
                "dataset_checksum": base["source_population_checksum"],
                "stock_schema_checksum": base["stock_schema_checksum"],
                "context_schema_checksum": base["context_schema_checksum"],
                "interaction_checksum": interactions["interaction_contract_checksum"],
                "fold_identity": base["validation_fold_identity"], "training_cutoff": cutoff,
                "maximum_label_availability_timestamp": maturity,
            })
        prediction_checksum = canonical_hash(prediction_rows)
        diagnostics = _coefficient_diagnostics(estimator, lineage, interactions, predictions, diversity)
        logical = {
            "contract_version": FIT_RESULT_CONTRACT, "status": "READY", "valid": True,
            "blocking_reasons": [], "warnings": [],
            "estimator_identity": "sklearn.linear_model.ElasticNet",
            "dependency_name": "scikit-learn", "dependency_version": sklearn.__version__,
            "observation_counts": {"training": len(training), "validation": len(validation)},
            "stock_feature_count": len(base["ordered_stock_feature_ids"]),
            "context_feature_count": len(base["ordered_market_context_ids"]),
            "interaction_count": len(interactions["interactions"]),
            "design_column_count": len(lineage),
            "temporal_integrity": "TRAINING_ONLY_PREPROCESSING_AND_FIT",
            "preprocessing": preprocessing, "preprocessing_checksum": preprocessing["preprocessing_checksum"],
            "interaction_contract": interactions,
            "interaction_checksum": interactions["interaction_contract_checksum"],
            "design_column_lineage": lineage, "design_matrix_checksum": design_checksum,
            "model": model, "model_checksum": model["model_checksum"],
            "predictions": prediction_rows, "prediction_checksum": prediction_checksum,
            "coefficient_diagnostics": diagnostics,
            "configuration": config, "configuration_checksum": canonical_hash(config),
            "input_checksum": base["logical_input_checksum"],
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except ContextualElasticNetError as exc:
        return _blocked(data, interaction_contract, config, exc)
    except Exception as exc:  # pragma: no cover
        return _blocked(data, interaction_contract, config, ContextualElasticNetError("NUMERICAL_FAILURE", type(exc).__name__))


def context_sensitivity(
    result: Mapping[str, Any],
    *,
    stock_feature_values: Sequence[float],
    baseline_context_values: Sequence[float],
    changed_context_values: Sequence[float],
    baseline_context_identity: str,
    changed_context_identity: str,
) -> dict[str, Any]:
    if not result.get("valid"):
        return {"status": "INVALID_INPUT", "valid": False, "blocking_reasons": ["FIT_RESULT_INVALID"]}
    preprocessing = result["preprocessing"]
    interactions = result["interaction_contract"]
    stock = np.asarray([stock_feature_values], dtype=float)
    baseline = np.asarray([baseline_context_values], dtype=float)
    changed = np.asarray([changed_context_values], dtype=float)
    x_base, lineage = _design_matrix(
        stock, baseline, preprocessing, interactions,
        result["configuration"]["include_context_main_effects"],
    )
    x_changed, _ = _design_matrix(
        stock, changed, preprocessing, interactions,
        result["configuration"]["include_context_main_effects"],
    )
    coefficients = np.asarray(result["model"]["coefficient_vector"])
    intercept = result["model"]["intercept"]
    base_score = float(intercept + (x_base @ coefficients)[0])
    changed_score = float(intercept + (x_changed @ coefficients)[0])
    contributions = []
    for index, column in enumerate(lineage):
        if column["column_type"] == "interaction":
            change = float((x_changed[0, index] - x_base[0, index]) * coefficients[index])
            if change != 0:
                contributions.append({"column_id": column["column_id"], "score_change": change})
    main_indexes = [index for index, column in enumerate(lineage) if column["column_type"] == "stock"]
    return {
        "contract_version": "contextual_elastic_net_context_sensitivity_v1",
        "status": "READY", "valid": True, "blocking_reasons": [],
        "baseline_context_identity": baseline_context_identity,
        "changed_context_identity": changed_context_identity,
        "baseline_score": base_score, "changed_context_score": changed_score,
        "total_score_change": changed_score - base_score,
        "affected_interaction_contributions": contributions,
        "unchanged_stock_main_effect_contribution": float(x_base[0, main_indexes] @ coefficients[main_indexes]),
        "diagnostic_is_historical_evidence": False,
    }


def build_contextual_design_matrix(
    data: Mapping[str, Any],
    interaction_contract: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    *,
    split: str,
    include_context_main_effects: bool = False,
) -> dict[str, Any]:
    base = _validated_input(data)
    interactions = _validated_interactions(interaction_contract, base)
    rows = [row for row in base["rows"] if row["split"] == split]
    if not rows:
        raise ContextualElasticNetError("INSUFFICIENT_DATA", "DESIGN_SPLIT_EMPTY")
    matrix, lineage = _design_matrix(
        np.asarray([row["stock_feature_values"] for row in rows]),
        np.asarray([row["market_context_values"] for row in rows]),
        preprocessing, interactions, include_context_main_effects,
    )
    return {
        "contract_version": "contextual_design_matrix_v1",
        "split": split, "row_ids": [row["row_id"] for row in rows],
        "column_lineage": lineage, "matrix": matrix.tolist(),
        "design_matrix_checksum": canonical_hash(
            {"row_ids": [row["row_id"] for row in rows], "lineage": lineage, "matrix": matrix.tolist()}
        ),
    }


def contextual_stability(results: Sequence[Mapping[str, Any]], *, unstable_threshold: float = 0.75) -> dict[str, Any]:
    if len(results) < 2 or any(not result.get("valid") for result in results):
        return _simple_block(STABILITY_CONTRACT, "INSUFFICIENT_DATA", "TWO_VALID_FITS_REQUIRED")
    columns = [result["model"]["ordered_design_column_ids"] for result in results]
    interactions = {result["interaction_checksum"] for result in results}
    if len(interactions) != 1 or any(value != columns[0] for value in columns):
        return _simple_block(STABILITY_CONTRACT, "INTERACTION_CONTRACT_MISMATCH", "STABILITY_DESIGN_IDENTITY_MISMATCH")
    coefficient = np.asarray([result["model"]["coefficient_vector"] for result in results])
    signs = np.maximum(np.mean(coefficient > 0, axis=0), np.mean(coefficient < 0, axis=0))
    nonzero = np.mean(coefficient != 0, axis=0)
    correlations = [
        _spearman(coefficient[left], coefficient[right])
        for left in range(len(coefficient)) for right in range(left + 1, len(coefficient))
    ]
    logical = {
        "contract_version": STABILITY_CONTRACT, "status": "READY", "valid": True,
        "blocking_reasons": [], "ordered_design_column_ids": columns[0],
        "coefficient_mean": np.mean(coefficient, axis=0).tolist(),
        "coefficient_median": np.median(coefficient, axis=0).tolist(),
        "coefficient_standard_deviation": np.std(coefficient, axis=0).tolist(),
        "sign_consistency": signs.tolist(), "nonzero_selection_frequency": nonzero.tolist(),
        "interaction_selection_frequency": {
            column: float(nonzero[index]) for index, column in enumerate(columns[0]) if column.startswith("interaction:")
        },
        "coefficient_vector_rank_correlations": correlations,
        "unstable_column_ids": [column for column, value in zip(columns[0], signs) if value < unstable_threshold],
        "prediction_rank_stability": _prediction_rank_stability(results),
        "convergence_consistency": all(result["coefficient_diagnostics"]["convergence_status"] == "CONVERGED" for result in results),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def compare_contextual_with_stock_only(
    data: Mapping[str, Any],
    interaction_contract: Mapping[str, Any],
    **fit_options,
) -> dict[str, Any]:
    contextual = fit_contextual_elastic_net(data, interaction_contract, **fit_options)
    if not contextual.get("valid"):
        return _simple_block(COMPARISON_CONTRACT, "INVALID_INPUT", "CONTEXTUAL_FIT_INVALID")
    from sklearn.linear_model import ElasticNet
    base = _validated_input(data)
    training = [row for row in base["rows"] if row["split"] == "TRAINING"]
    validation = [row for row in base["rows"] if row["split"] == "VALIDATION"]
    preprocessing = contextual["preprocessing"]
    stock_train = _scale(
        np.asarray([row["stock_feature_values"] for row in training]),
        preprocessing["stock_location"], preprocessing["stock_scale"],
        preprocessing["stock_lower"], preprocessing["stock_upper"],
    )
    stock_validation = _scale(
        np.asarray([row["stock_feature_values"] for row in validation]),
        preprocessing["stock_location"], preprocessing["stock_scale"],
        preprocessing["stock_lower"], preprocessing["stock_upper"],
    )
    config = contextual["configuration"]
    from sklearn.exceptions import ConvergenceWarning
    control = ElasticNet(
        alpha=config["alpha"], l1_ratio=config["l1_ratio"], fit_intercept=config["fit_intercept"],
        tol=config["tolerance"], max_iter=config["maximum_iterations"], selection=config["selection"],
        random_state=config["random_seed"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        control.fit(
            stock_train, [row["target_value"] for row in training],
            sample_weight=[row["sample_weight"] for row in training],
        )
    control_convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    actual = np.asarray([row["target_value"] for row in validation])
    contextual_prediction = np.asarray([row["continuous_score"] for row in contextual["predictions"]])
    control_prediction = control.predict(stock_validation)
    logical = {
        "contract_version": COMPARISON_CONTRACT, "status": "READY", "valid": True,
        "blocking_reasons": [],
        "contextual": _metrics(actual, contextual_prediction, contextual["model"]["coefficient_vector"]),
        "stock_only": _metrics(actual, control_prediction, control.coef_),
        "interaction_recovery": {
            row["column_id"]: row["coefficient"]
            for row in contextual["coefficient_diagnostics"]["interaction_effects"]
            if row["nonzero"]
        },
        "stock_only_control_convergence_warning": control_convergence_warning,
        "contextual_improvement_assumed": False,
        "historical_superiority_claimed": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def verify_contextual_elastic_net_result(
    data: Mapping[str, Any],
    interaction_contract: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    config = result.get("configuration", {})
    expected = fit_contextual_elastic_net(
        data, interaction_contract, alpha=config.get("alpha", DEFAULT_ALPHA),
        l1_ratio=config.get("l1_ratio", DEFAULT_L1_RATIO),
        fit_intercept=config.get("fit_intercept", True),
        tolerance=config.get("tolerance", DEFAULT_TOLERANCE),
        maximum_iterations=config.get("maximum_iterations", DEFAULT_MAX_ITERATIONS),
        selection=config.get("selection", "cyclic"), random_seed=config.get("random_seed", 0),
        warm_start=config.get("warm_start", False),
        include_context_main_effects=config.get("include_context_main_effects", False),
        winsorisation_limits=tuple(config["winsorisation_limits"]) if config.get("winsorisation_limits") else None,
        minimum_training_rows=config.get("minimum_training_rows", 8),
        minimum_rank_diversity=config.get("minimum_rank_diversity", 2),
        source_commit=result.get("model", {}).get("source_commit"),
    )
    fields = (
        "status", "preprocessing", "interaction_contract", "design_column_lineage",
        "design_matrix_checksum", "model", "model_checksum", "predictions",
        "prediction_checksum", "coefficient_diagnostics", "configuration_checksum",
        "input_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": "contextual_elastic_net_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def _validated_interactions(contract, base):
    expected = contextual_interaction_contract(
        base["ordered_stock_feature_ids"], base["ordered_market_context_ids"],
        interactions=contract.get("interactions"),
    )
    if contract.get("interaction_contract_checksum") != expected["interaction_contract_checksum"]:
        raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "INTERACTION_CHECKSUM_MISMATCH")
    if base["interaction_contract_identity"] != contract.get("contract_version"):
        raise ContextualElasticNetError("INTERACTION_CONTRACT_MISMATCH", "INTERACTION_IDENTITY_MISMATCH")
    return expected


def _fit_preprocessing(stock, context, stock_ids, context_ids, winsorisation):
    stock_lower = stock_upper = context_lower = context_upper = None
    if winsorisation is not None:
        low, high = winsorisation
        if not 0 <= low < high <= 1:
            raise ContextualElasticNetError("PREPROCESSING_FAILURE", "WINSORISATION_LIMITS_INVALID")
        stock_lower, stock_upper = np.quantile(stock, low, axis=0), np.quantile(stock, high, axis=0)
        context_lower, context_upper = np.quantile(context, low, axis=0), np.quantile(context, high, axis=0)
        stock, context = np.clip(stock, stock_lower, stock_upper), np.clip(context, context_lower, context_upper)
    stock_mean, context_mean = np.mean(stock, axis=0), np.mean(context, axis=0)
    stock_raw, context_raw = np.std(stock, axis=0), np.std(context, axis=0)
    stock_constant = np.isclose(stock_raw, 0.0, atol=1e-15, rtol=0)
    context_constant = np.isclose(context_raw, 0.0, atol=1e-15, rtol=0)
    stock_scale, context_scale = np.where(stock_constant, 1, stock_raw), np.where(context_constant, 1, context_raw)
    logical = {
        "contract_version": PREPROCESSING_CONTRACT,
        "identity": "scale_bases_on_training_then_construct_scaled_products_v1",
        "stock_feature_ids": list(stock_ids), "context_feature_ids": list(context_ids),
        "stock_location": stock_mean.tolist(), "stock_scale": stock_scale.tolist(),
        "context_location": context_mean.tolist(), "context_scale": context_scale.tolist(),
        "constant_stock_feature_ids": [value for value, constant in zip(stock_ids, stock_constant) if constant],
        "constant_context_feature_ids": [value for value, constant in zip(context_ids, context_constant) if constant],
        "constant_policy": "center_then_unit_scale",
        "winsorisation_limits": winsorisation,
        "stock_lower": None if stock_lower is None else stock_lower.tolist(),
        "stock_upper": None if stock_upper is None else stock_upper.tolist(),
        "context_lower": None if context_lower is None else context_lower.tolist(),
        "context_upper": None if context_upper is None else context_upper.tolist(),
        "fitted_on": "TRAINING_ONLY",
    }
    logical["preprocessing_checksum"] = canonical_hash(logical)
    return logical


def _design_matrix(stock, context, preprocessing, interaction_contract, include_context):
    stock_scaled = _scale(
        stock, preprocessing["stock_location"], preprocessing["stock_scale"],
        preprocessing["stock_lower"], preprocessing["stock_upper"],
    )
    context_scaled = _scale(
        context, preprocessing["context_location"], preprocessing["context_scale"],
        preprocessing["context_lower"], preprocessing["context_upper"],
    )
    columns = [stock_scaled]
    lineage = [
        {"column_id": f"stock:{feature}", "column_type": "stock", "stock_feature_id": feature, "context_feature_id": None}
        for feature in preprocessing["stock_feature_ids"]
    ]
    if include_context:
        columns.append(context_scaled)
        lineage.extend([
            {"column_id": f"context:{feature}", "column_type": "context", "stock_feature_id": None, "context_feature_id": feature}
            for feature in preprocessing["context_feature_ids"]
        ])
    stock_indexes = {value: index for index, value in enumerate(preprocessing["stock_feature_ids"])}
    context_indexes = {value: index for index, value in enumerate(preprocessing["context_feature_ids"])}
    for interaction in interaction_contract["interactions"]:
        values = (
            stock_scaled[:, stock_indexes[interaction["stock_feature_id"]]]
            * context_scaled[:, context_indexes[interaction["market_context_id"]]]
        )[:, None]
        columns.append(values)
        lineage.append({
            "column_id": f"interaction:{interaction['interaction_id']}",
            "column_type": "interaction",
            "stock_feature_id": interaction["stock_feature_id"],
            "context_feature_id": interaction["market_context_id"],
        })
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all() or len({row["column_id"] for row in lineage}) != len(lineage):
        raise ContextualElasticNetError("PREPROCESSING_FAILURE", "DESIGN_MATRIX_INVALID")
    return matrix, lineage


def _scale(values, location, scale, lower, upper):
    matrix = np.asarray(values, dtype=float)
    if lower is not None:
        matrix = np.clip(matrix, lower, upper)
    result = (matrix - np.asarray(location)) / np.asarray(scale)
    if not np.isfinite(result).all():
        raise ContextualElasticNetError("PREPROCESSING_FAILURE", "SCALED_VALUE_NON_FINITE")
    return result


def _model(base, interactions, preprocessing, lineage, estimator, version, config, cutoff, maturity, commit, design_checksum):
    logical = {
        "contract_version": MODEL_CONTRACT, "estimator_identity": "sklearn.linear_model.ElasticNet",
        "dependency_version": version,
        "objective_convention": "(1/(2*n))*||y-Xw||^2 + alpha*l1_ratio*||w||_1 + 0.5*alpha*(1-l1_ratio)*||w||_2^2",
        "alpha": config["alpha"], "l1_ratio": config["l1_ratio"],
        "fit_intercept": config["fit_intercept"], "tolerance": config["tolerance"],
        "maximum_iterations": config["maximum_iterations"], "selection": config["selection"],
        "random_seed": config["random_seed"], "warm_start": config["warm_start"],
        "preprocessing_identity": preprocessing["identity"],
        "preprocessing_checksum": preprocessing["preprocessing_checksum"],
        "interaction_contract_identity": interactions["contract_version"],
        "interaction_contract_checksum": interactions["interaction_contract_checksum"],
        "ordered_design_column_ids": [row["column_id"] for row in lineage],
        "stock_schema_checksum": base["stock_schema_checksum"],
        "context_schema_checksum": base["context_schema_checksum"],
        "training_population_checksum": canonical_hash([row["row_id"] for row in base["rows"] if row["split"] == "TRAINING"]),
        "training_cutoff": cutoff, "maximum_training_label_maturity": maturity,
        "source_commit": commit, "design_matrix_checksum": design_checksum,
        "coefficient_vector": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
        "iteration_count": int(estimator.n_iter_), "dual_gap": float(estimator.dual_gap_),
    }
    logical["model_checksum"] = canonical_hash(logical)
    return logical


def _coefficient_diagnostics(estimator, lineage, interactions, predictions, diversity):
    rows = []
    for column, coefficient in zip(lineage, estimator.coef_):
        rows.append({
            **column, "coefficient": float(coefficient), "absolute_coefficient": abs(float(coefficient)),
            "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
            "nonzero": bool(coefficient != 0),
        })
    maximum = max((row["absolute_coefficient"] for row in rows), default=0)
    for row in rows:
        row["relative_magnitude"] = row["absolute_coefficient"] / maximum if maximum else 0
    by_type = {kind: [row for row in rows if row["column_type"] == kind] for kind in ("stock", "context", "interaction")}
    return {
        "stock_main_effects": by_type["stock"], "context_main_effects": by_type["context"],
        "interaction_effects": by_type["interaction"],
        "nonzero_coefficient_count": int(np.sum(estimator.coef_ != 0)),
        "stock_main_effect_sparsity": _sparsity(by_type["stock"]),
        "context_effect_sparsity": _sparsity(by_type["context"]),
        "interaction_sparsity": _sparsity(by_type["interaction"]),
        "l1_norm": float(np.sum(np.abs(estimator.coef_))),
        "l2_norm": float(np.linalg.norm(estimator.coef_)),
        "maximum_absolute_coefficient": float(maximum),
        "convergence_iterations": int(estimator.n_iter_), "dual_gap": float(estimator.dual_gap_),
        "convergence_status": "CONVERGED",
        "prediction_dispersion": float(np.std(predictions)), "rank_diversity": diversity,
        "interaction_coefficients_are_causal": False,
    }


def _sparsity(rows):
    return None if not rows else 1 - sum(row["nonzero"] for row in rows) / len(rows)


def _rank_predictions(validation, predictions):
    groups = {}
    for row, prediction in zip(validation, predictions):
        groups.setdefault(row["decision_timestamp"], []).append((row, float(prediction)))
    output = []
    for decision, values in sorted(groups.items()):
        ranked = sorted(values, key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]))
        for rank, (row, prediction) in enumerate(ranked, 1):
            output.append({
                "row_id": row["row_id"], "asset_id": row["asset_id"], "decision_timestamp": decision,
                "continuous_score": prediction, "within_date_rank": rank,
                "within_date_percentile_rank": (len(ranked) - rank) / max(len(ranked) - 1, 1),
                "prediction_semantics": "continuous_score_not_probability",
            })
    return sorted(output, key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))


def _validated_input(data):
    base = contextual_elastic_net_input(
        data["rows"], target_horizon=data["target_horizon"],
        stock_feature_schema_identity=data["stock_feature_schema_identity"],
        market_context_schema_identity=data["market_context_schema_identity"],
        interaction_contract_identity=data["interaction_contract_identity"],
        training_fold_identity=data["training_fold_identity"],
        validation_fold_identity=data["validation_fold_identity"],
        dataset_identity=data["dataset_identity"],
        source_population_checksum=data["source_population_checksum"],
    )
    if data.get("logical_input_checksum") and data["logical_input_checksum"] != base["logical_input_checksum"]:
        raise ContextualElasticNetError("INVALID_INPUT", "INPUT_CHECKSUM_MISMATCH")
    return base


def _metrics(actual, prediction, coefficients):
    error = prediction - actual
    return {
        "mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error))),
        "spearman_rank_correlation": _spearman(prediction, actual),
        "prediction_dispersion": float(np.std(prediction)),
        "coefficient_sparsity": float(np.mean(np.asarray(coefficients) == 0)),
    }


def _prediction_rank_stability(results):
    mappings = [{row["row_id"]: row["within_date_rank"] for row in result["predictions"]} for result in results]
    common = sorted(set.intersection(*(set(value) for value in mappings)))
    if len(common) < 2:
        return None
    correlations = []
    for left in range(len(mappings)):
        for right in range(left + 1, len(mappings)):
            correlations.append(_spearman([mappings[left][key] for key in common], [mappings[right][key] for key in common]))
    return float(np.mean(correlations))


def _spearman(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def _ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _finite(values, owner):
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ContextualElasticNetError("INVALID_INPUT", f"{owner}_NON_FINITE_OR_EMPTY")
    return result


def _blocked(data, interaction, config, error):
    logical = {
        "contract_version": FIT_RESULT_CONTRACT,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
        "estimator_identity": "sklearn.linear_model.ElasticNet",
        "dependency_name": "scikit-learn", "dependency_version": None,
        "observation_counts": {}, "stock_feature_count": None, "context_feature_count": None,
        "interaction_count": None, "design_column_count": None, "temporal_integrity": "FAILED",
        "preprocessing": {}, "preprocessing_checksum": None,
        "interaction_contract": dict(interaction) if isinstance(interaction, Mapping) else {},
        "interaction_checksum": interaction.get("interaction_contract_checksum") if isinstance(interaction, Mapping) else None,
        "design_column_lineage": [], "design_matrix_checksum": None,
        "model": {}, "model_checksum": None, "predictions": [], "prediction_checksum": None,
        "coefficient_diagnostics": {}, "configuration": config,
        "configuration_checksum": canonical_hash(config),
        "input_checksum": data.get("logical_input_checksum") if isinstance(data, Mapping) else None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _simple_block(contract, status, reason):
    logical = {"contract_version": contract, "status": status, "valid": False, "blocking_reasons": [reason]}
    logical["logical_result_checksum"] = canonical_hash(logical)
    return logical


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContextualElasticNetError("TEMPORAL_VIOLATION", "TIMESTAMP_INVALID") from exc


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
