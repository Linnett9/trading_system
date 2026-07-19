from __future__ import annotations

import copy
import json
import math
import platform
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence

from core.research.ml.selector_component_rows import validate_model_row_roles

import numpy as np


TARGET_CONTRACT = "multi_horizon_target_contract_v1"
INPUT_CONTRACT = "multi_horizon_linear_input_v1"
PREPROCESSING_CONTRACT = "multi_horizon_preprocessing_v1"
MODEL_CONTRACT = "multi_horizon_linear_model_v1"
ORDERED_ADAPTER_CONTRACT = "multi_horizon_ordered_logit_adapter_v1"
PREDICTION_CONTRACT = "multi_horizon_prediction_v1"
COMBINED_CONTRACT = "multi_horizon_combined_score_v1"
RESULT_CONTRACT = "multi_horizon_linear_result_v1"
HORIZONS = (("return_1s", 1), ("return_5s", 5), ("return_10s", 10), ("return_20s", 20))
HORIZON_IDS = tuple(value[0] for value in HORIZONS)
COMBINATION_WEIGHTS = {
    "short_term": {"return_1s": 0.6, "return_5s": 0.4},
    "medium_term": {"return_5s": 0.5, "return_10s": 0.5},
    "long_term": {"return_10s": 0.3, "return_20s": 0.7},
}
STATUSES = {
    "READY", "PARTIALLY_AVAILABLE", "INVALID_INPUT", "INSUFFICIENT_DATA",
    "TEMPORAL_VIOLATION", "FEATURE_SCHEMA_MISMATCH", "TARGET_CONTRACT_MISMATCH",
    "HORIZON_POPULATION_MISMATCH", "PREPROCESSING_FAILURE", "NON_CONVERGENCE",
    "NONFINITE_PREDICTION", "INSUFFICIENT_HORIZONS", "NUMERICAL_FAILURE",
}


class MultiHorizonError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class FittedMultiHorizonMember:
    estimator: Any
    model_family: str
    horizon_id: str
    horizon_order: int
    family_order: int
    member_order: int
    ordered_feature_ids: tuple[str, ...]
    preprocessing: Mapping[str, Any]
    estimator_configuration: Mapping[str, Any]
    random_state_identity: int | str
    target_identity: str
    training_population: Mapping[str, Any]
    fold_identity: str
    training_cutoff: str
    input_identity: str
    configuration_identity: str


class _FittedMemberCallbackFailure(Exception):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def multi_horizon_target_contract(
    definitions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(definitions) if definitions is not None else [
        {
            "horizon_id": horizon_id, "horizon_sessions": sessions,
            "target_definition": f"forward_total_return_over_{sessions}_sessions",
            "target_unit": "decimal_return",
            "benchmark_adjustment_convention": "none",
            "target_start_rule": "first_session_after_decision",
            "target_end_rule": f"close_of_session_{sessions}_after_decision",
            "maturity_timestamp_rule": f"after_close_session_{sessions}",
            "overlapping_outcomes": sessions > 1,
            "target_type": "regression",
            "minimum_valid_return_observations": sessions,
        }
        for horizon_id, sessions in HORIZONS
    ]
    if [str(row.get("horizon_id")) for row in rows] != list(HORIZON_IDS):
        if len({str(row.get("horizon_id")) for row in rows}) != len(rows):
            raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "DUPLICATE_HORIZON_ID")
        raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "EXACT_HORIZON_PANEL_REQUIRED")
    normalised = []
    definitions_seen = set()
    for raw, (expected_id, expected_sessions) in zip(rows, HORIZONS):
        sessions = int(raw["horizon_sessions"])
        definition = str(raw["target_definition"])
        if sessions != expected_sessions or sessions <= 0:
            raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "HORIZON_SESSION_MISMATCH")
        if definition in definitions_seen:
            raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "DUPLICATE_TARGET_DEFINITION")
        definitions_seen.add(definition)
        row = {
            "horizon_id": expected_id, "horizon_sessions": sessions,
            "target_definition": definition, "target_unit": str(raw["target_unit"]),
            "benchmark_adjustment_convention": str(raw["benchmark_adjustment_convention"]),
            "target_start_rule": str(raw["target_start_rule"]),
            "target_end_rule": str(raw["target_end_rule"]),
            "maturity_timestamp_rule": str(raw["maturity_timestamp_rule"]),
            "overlapping_outcomes": bool(raw["overlapping_outcomes"]),
            "target_type": str(raw["target_type"]),
            "minimum_valid_return_observations": int(raw["minimum_valid_return_observations"]),
        }
        if row["target_type"] not in {"regression", "ordinal"}:
            raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "TARGET_TYPE_UNSUPPORTED")
        row["target_checksum"] = canonical_hash(row)
        normalised.append(row)
    logical = {"contract_version": TARGET_CONTRACT, "horizons": normalised}
    logical["target_panel_checksum"] = canonical_hash(logical)
    return logical


def multi_horizon_linear_input(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_contract: Mapping[str, Any],
    feature_schema_identity: str,
    dataset_identity: str,
    fold_identity: str,
    source_population_checksum: str,
) -> dict[str, Any]:
    panel = multi_horizon_target_contract(target_contract.get("horizons"))
    if target_contract.get("target_panel_checksum") != panel["target_panel_checksum"]:
        raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "TARGET_PANEL_CHECKSUM_MISMATCH")
    normalised, row_ids = [], []
    feature_order = None
    for raw in rows:
        row_id, asset = str(raw.get("row_id") or ""), str(raw.get("asset_id") or "")
        decision = str(raw.get("decision_timestamp") or "")
        availability = str(raw.get("feature_availability_timestamp") or decision)
        split = str(raw.get("split") or "")
        feature_ids = [str(value) for value in raw.get("feature_ids", ())]
        features = [float(value) for value in raw.get("feature_values", ())]
        targets = dict(raw.get("target_values") or {})
        maturities = {str(key): str(value) for key, value in (raw.get("target_maturity_timestamps") or {}).items()}
        states = {str(key): str(value) for key, value in (raw.get("target_availability_state") or {}).items()}
        weight = float(raw.get("sample_weight", 1.0))
        if not row_id or not asset or split not in {
            "TRAINING", "FIT_VALIDATION", "PREDICTION", "VALIDATION"
        }:
            raise MultiHorizonError("INVALID_INPUT", "ROW_IDENTITY_OR_SPLIT_INVALID")
        if feature_ids != sorted(feature_ids) or len(feature_ids) != len(features) or len(set(feature_ids)) != len(feature_ids):
            raise MultiHorizonError("FEATURE_SCHEMA_MISMATCH", "FEATURE_SCHEMA_INVALID")
        if feature_order is None:
            feature_order = feature_ids
        elif feature_ids != feature_order:
            raise MultiHorizonError("FEATURE_SCHEMA_MISMATCH", "FEATURE_ORDER_MISMATCH")
        if not all(math.isfinite(value) for value in features):
            raise MultiHorizonError("INVALID_INPUT", "FEATURE_NON_FINITE")
        labeled = split != "PREDICTION"
        if labeled and (
            set(targets) != set(HORIZON_IDS)
            or set(maturities) != set(HORIZON_IDS)
            or set(states) != set(HORIZON_IDS)
        ):
            raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "ROW_HORIZON_SET_MISMATCH")
        for horizon_id in HORIZON_IDS if labeled else ():
            state = states[horizon_id]
            if state not in {"MATURE", "IMMATURE", "INVALID"}:
                raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "TARGET_AVAILABILITY_STATE_INVALID")
            _time(maturities[horizon_id])
            value = targets[horizon_id]
            if state == "MATURE" and (value is None or not math.isfinite(float(value))):
                raise MultiHorizonError("INVALID_INPUT", "MATURE_TARGET_NON_FINITE")
            if state != "MATURE" and value is not None and not math.isfinite(float(value)):
                raise MultiHorizonError("INVALID_INPUT", "TARGET_NON_FINITE")
            targets[horizon_id] = None if value is None else float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise MultiHorizonError("INVALID_INPUT", "SAMPLE_WEIGHT_INVALID")
        if _time(availability) > _time(decision):
            raise MultiHorizonError("TEMPORAL_VIOLATION", "FEATURE_AVAILABLE_AFTER_DECISION")
        row_ids.append(row_id)
        normalised.append({
            "row_id": row_id, "asset_id": asset, "decision_timestamp": decision,
            "feature_availability_timestamp": availability, "feature_ids": feature_ids,
            "feature_values": features,
            "sample_weight": weight, "split": split,
        })
        if labeled:
            normalised[-1].update(
                target_values=targets,
                target_maturity_timestamps=maturities,
                target_availability_state=states,
            )
    if not normalised or len(row_ids) != len(set(row_ids)):
        raise MultiHorizonError("INVALID_INPUT", "ROW_IDENTITIES_NOT_UNIQUE")
    if normalised != sorted(normalised, key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"])):
        raise MultiHorizonError("INVALID_INPUT", "ROWS_NOT_DETERMINISTICALLY_ORDERED")
    strengthened = not any(row["split"] == "VALIDATION" for row in normalised)
    if strengthened:
        validate_model_row_roles(
            normalised, role_field="split",
            target_fields=(
                "target_values", "target_maturity_timestamps",
                "target_availability_state",
            ),
            require_prediction=False,
        )
    logical = {
        "contract_version": INPUT_CONTRACT, "rows": normalised,
        "target_contract": panel, "ordered_feature_ids": feature_order,
        "feature_schema_identity": str(feature_schema_identity),
        "feature_schema_checksum": canonical_hash({"identity": feature_schema_identity, "ids": feature_order}),
        "dataset_identity": str(dataset_identity), "fold_identity": str(fold_identity),
        "source_population_checksum": str(source_population_checksum),
        "row_population_checksum": canonical_hash(row_ids),
        "input_value_checksum": canonical_hash(normalised),
        "row_contract_version": (
            "selector_component_rows.v2"
            if strengthened else "legacy_wave4_validation_rows.v1"
        ),
    }
    logical["logical_input_checksum"] = canonical_hash(logical)
    return logical


def ordered_logit_adapter(
    *,
    horizon_id: str,
    class_probabilities: Sequence[Sequence[float]],
    expected_relevance: Sequence[float],
    row_ids: Sequence[str],
    model_identity: str,
    fold_identity: str,
    target_checksum: str,
    expected_target_checksum: str,
) -> dict[str, Any]:
    if horizon_id not in HORIZON_IDS or target_checksum != expected_target_checksum:
        raise MultiHorizonError("TARGET_CONTRACT_MISMATCH", "ORDERED_ADAPTER_TARGET_MISMATCH")
    probabilities = np.asarray(class_probabilities, dtype=float)
    expected = np.asarray(expected_relevance, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(row_ids) or expected.shape != (len(row_ids),):
        raise MultiHorizonError("HORIZON_POPULATION_MISMATCH", "ORDERED_ADAPTER_DIMENSION_MISMATCH")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8):
        raise MultiHorizonError("INVALID_INPUT", "ORDERED_ADAPTER_PROBABILITY_INVALID")
    logical = {
        "contract_version": ORDERED_ADAPTER_CONTRACT, "horizon_id": horizon_id,
        "prediction_semantics": "expected_ordinal_relevance",
        "class_probabilities": probabilities.tolist(), "expected_relevance": expected.tolist(),
        "row_ids": list(row_ids), "model_identity": model_identity,
        "fold_identity": fold_identity, "target_checksum": target_checksum,
    }
    logical["adapter_checksum"] = canonical_hash(logical)
    return logical


def fit_multi_horizon_linear_selector(
    data: Mapping[str, Any],
    *,
    training_cutoff: str,
    model_families: Sequence[str] = ("ridge", "elastic_net"),
    ridge_alpha: float = 1.0,
    elastic_alpha: float = 0.001,
    elastic_l1_ratio: float = 0.25,
    elastic_tolerance: float = 1e-4,
    elastic_maximum_iterations: int = 5000,
    minimum_training_rows: int = 5,
    minimum_rank_diversity: int = 2,
    fitted_member_callback: Callable[[FittedMultiHorizonMember], None] | None = None,
) -> dict[str, Any]:
    config = {
        "training_cutoff": training_cutoff, "model_families": list(model_families),
        "ridge_alpha": ridge_alpha, "elastic_alpha": elastic_alpha,
        "elastic_l1_ratio": elastic_l1_ratio, "elastic_tolerance": elastic_tolerance,
        "elastic_maximum_iterations": elastic_maximum_iterations,
        "minimum_training_rows": minimum_training_rows,
        "minimum_rank_diversity": minimum_rank_diversity,
        "preprocessing_policy": "independent_mature_population_per_horizon_v1",
        "horizon_ensemble_weights": {"ridge": 0.5, "elastic_net": 0.5},
        "combined_score_weights": COMBINATION_WEIGHTS,
    }
    try:
        base = _validated_input(data)
        cutoff = _time(training_cutoff)
        families = tuple(model_families)
        if not families or any(value not in {"ridge", "elastic_net"} for value in families):
            raise MultiHorizonError("INVALID_INPUT", "MODEL_FAMILY_PANEL_INVALID")
        validation = [
            row for row in base["rows"]
            if row["split"] in {"PREDICTION", "VALIDATION"}
        ]
        if not validation:
            raise MultiHorizonError("INSUFFICIENT_DATA", "VALIDATION_POPULATION_EMPTY")
        validation_start = min(_time(row["decision_timestamp"]) for row in validation)
        if cutoff >= validation_start:
            raise MultiHorizonError("TEMPORAL_VIOLATION", "TRAINING_CUTOFF_NOT_BEFORE_VALIDATION")
        try:
            import sklearn
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import ElasticNet, Ridge
        except ImportError:
            raise MultiHorizonError("NUMERICAL_FAILURE", "SKLEARN_UNAVAILABLE")
        populations, preprocessing, models, predictions, diagnostics = {}, {}, [], [], {}
        available_horizons = []
        member_order = 0
        for horizon_order, target in enumerate(base["target_contract"]["horizons"]):
            horizon_id = target["horizon_id"]
            candidates = [row for row in base["rows"] if row["split"] == "TRAINING"]
            eligible = [
                row for row in candidates
                if row["target_availability_state"][horizon_id] == "MATURE"
                and _time(row["target_maturity_timestamps"][horizon_id]) <= cutoff
                and _time(row["decision_timestamp"]) < cutoff
            ]
            immature = [
                row for row in candidates
                if row["target_availability_state"][horizon_id] == "IMMATURE"
                or _time(row["target_maturity_timestamps"][horizon_id]) > cutoff
            ]
            invalid = [row for row in candidates if row["target_availability_state"][horizon_id] == "INVALID"]
            populations[horizon_id] = {
                "eligible_count": len(eligible), "immature_count": len(immature),
                "invalid_count": len(invalid), "eligible_row_ids": [row["row_id"] for row in eligible],
                "excluded_immature_row_ids": [row["row_id"] for row in immature],
                "invalid_row_ids": [row["row_id"] for row in invalid],
                "training_checksum": canonical_hash([row["row_id"] for row in eligible]),
                "validation_checksum": canonical_hash([row["row_id"] for row in validation]),
                "maximum_label_maturity_timestamp": max(
                    (row["target_maturity_timestamps"][horizon_id] for row in eligible), default=None
                ),
            }
            if len(eligible) < minimum_training_rows:
                diagnostics[horizon_id] = {"status": "INSUFFICIENT_DATA"}
                continue
            x_train = np.asarray([row["feature_values"] for row in eligible])
            x_validation = np.asarray([row["feature_values"] for row in validation])
            mean, raw_scale = np.mean(x_train, axis=0), np.std(x_train, axis=0)
            scale = np.where(np.isclose(raw_scale, 0, atol=1e-15, rtol=0), 1, raw_scale)
            prep = {
                "contract_version": PREPROCESSING_CONTRACT,
                "policy": "independent_mature_population_per_horizon_v1",
                "horizon_id": horizon_id, "feature_ids": base["ordered_feature_ids"],
                "location": mean.tolist(), "scale": scale.tolist(),
                "constant_feature_ids": [
                    feature for feature, value in zip(base["ordered_feature_ids"], raw_scale)
                    if math.isclose(float(value), 0, abs_tol=1e-15)
                ],
                "training_population_checksum": populations[horizon_id]["training_checksum"],
            }
            prep["preprocessing_checksum"] = canonical_hash(prep)
            preprocessing[horizon_id] = prep
            train_scaled, validation_scaled = (x_train - mean) / scale, (x_validation - mean) / scale
            y_train = np.asarray([row["target_values"][horizon_id] for row in eligible])
            weights = np.asarray([row["sample_weight"] for row in eligible])
            horizon_predictions = {}
            horizon_models = []
            for family_order, family in enumerate(families):
                if family == "ridge":
                    estimator = Ridge(alpha=ridge_alpha, fit_intercept=True, solver="auto", tol=1e-4)
                    estimator.fit(train_scaled, y_train, sample_weight=weights)
                    convergence = "CLOSED_FORM_OR_DETERMINISTIC_SOLVER_COMPLETE"
                    sparsity = None
                    estimator_configuration = {
                        "alpha": ridge_alpha, "fit_intercept": True,
                        "solver": "auto", "tolerance": 1e-4,
                    }
                    random_state_identity: int | str = (
                        "NOT_APPLICABLE_DETERMINISTIC"
                    )
                else:
                    estimator = ElasticNet(
                        alpha=elastic_alpha, l1_ratio=elastic_l1_ratio, fit_intercept=True,
                        tol=elastic_tolerance, max_iter=elastic_maximum_iterations,
                        selection="cyclic", random_state=0,
                    )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(train_scaled, y_train, sample_weight=weights)
                    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or estimator.n_iter_ >= elastic_maximum_iterations:
                        raise MultiHorizonError("NON_CONVERGENCE", f"ELASTIC_NET_NON_CONVERGENCE:{horizon_id}")
                    convergence = "CONVERGED"
                    sparsity = float(np.mean(estimator.coef_ == 0))
                    estimator_configuration = {
                        "alpha": elastic_alpha, "l1_ratio": elastic_l1_ratio,
                        "fit_intercept": True,
                        "maximum_iterations": elastic_maximum_iterations,
                        "tolerance": elastic_tolerance, "selection": "cyclic",
                    }
                    random_state_identity = 0
                if fitted_member_callback is not None:
                    payload = FittedMultiHorizonMember(
                        estimator=estimator,
                        model_family=family,
                        horizon_id=horizon_id,
                        horizon_order=horizon_order,
                        family_order=family_order,
                        member_order=member_order,
                        ordered_feature_ids=tuple(base["ordered_feature_ids"]),
                        preprocessing=copy.deepcopy(prep),
                        estimator_configuration=copy.deepcopy(
                            estimator_configuration
                        ),
                        random_state_identity=random_state_identity,
                        target_identity=target["target_checksum"],
                        training_population=copy.deepcopy(
                            populations[horizon_id]
                        ),
                        fold_identity=base["fold_identity"],
                        training_cutoff=training_cutoff,
                        input_identity=base["logical_input_checksum"],
                        configuration_identity=canonical_hash(config),
                    )
                    try:
                        fitted_member_callback(payload)
                    except Exception as exc:
                        raise _FittedMemberCallbackFailure from exc
                member_order += 1
                values = np.asarray(estimator.predict(validation_scaled))
                if not np.isfinite(values).all():
                    raise MultiHorizonError("NONFINITE_PREDICTION", f"NONFINITE_PREDICTION:{horizon_id}:{family}")
                model = {
                    "contract_version": MODEL_CONTRACT, "model_id": f"{family}_{horizon_id}_v1",
                    "model_family": family, "horizon_id": horizon_id,
                    "estimator_identity": f"sklearn.linear_model.{type(estimator).__name__}",
                    "dependency_version": sklearn.__version__,
                    "configuration": estimator_configuration,
                    "feature_schema_checksum": base["feature_schema_checksum"],
                    "target_checksum": target["target_checksum"],
                    "preprocessing_checksum": prep["preprocessing_checksum"],
                    "training_population_checksum": populations[horizon_id]["training_checksum"],
                    "coefficient_vector": estimator.coef_.tolist(), "intercept": float(estimator.intercept_),
                    "training_cutoff": training_cutoff,
                    "maximum_label_maturity_timestamp": populations[horizon_id]["maximum_label_maturity_timestamp"],
                }
                model["model_checksum"] = canonical_hash(model)
                horizon_models.append(model)
                horizon_predictions[family] = values
                predictions.extend(_prediction_rows(
                    validation, values, target, family, model, base,
                    training_cutoff, populations[horizon_id]["maximum_label_maturity_timestamp"],
                    minimum_rank_diversity,
                ))
            models.extend(horizon_models)
            ensemble = np.mean(np.vstack([horizon_predictions[family] for family in families]), axis=0)
            available_horizons.append(horizon_id)
            validation_targets = np.asarray([
                row["target_values"][horizon_id] for row in validation
                if "target_values" in row
                if row["target_availability_state"][horizon_id] == "MATURE"
            ])
            diagnostics[horizon_id] = {
                "status": "READY", "training_rows": len(eligible), "validation_rows": len(validation),
                "feature_count": len(base["ordered_feature_ids"]),
                "models": [
                    {
                        "model_family": model["model_family"], "coefficient_vector": model["coefficient_vector"],
                        "intercept": model["intercept"],
                        "coefficient_norm": float(np.linalg.norm(model["coefficient_vector"])),
                        "sparsity": sparsity if model["model_family"] == "elastic_net" else None,
                        "convergence": convergence,
                    } for model in horizon_models
                ],
                "ensemble_prediction_mean": float(np.mean(ensemble)),
                "ensemble_prediction_standard_deviation": float(np.std(ensemble)),
                "target_mean": float(np.mean(validation_targets)) if len(validation_targets) else None,
                "target_dispersion": float(np.std(validation_targets)) if len(validation_targets) else None,
            }
        combined = _combined_scores(validation, predictions, available_horizons)
        cross = _cross_horizon_diagnostics(models, predictions, combined)
        if len(available_horizons) < 2:
            status = "INSUFFICIENT_HORIZONS"
            valid = False
        elif len(available_horizons) < len(HORIZON_IDS):
            status = "PARTIALLY_AVAILABLE"
            valid = True
        else:
            status = "READY"
            valid = True
        logical = {
            "contract_version": RESULT_CONTRACT, "status": status, "valid": valid,
            "blocking_reasons": [] if valid else ["FEWER_THAN_TWO_HORIZONS_AVAILABLE"],
            "warnings": ["OVERLAPPING_TARGET_OUTCOMES_REQUIRE_BLOCK_BOOTSTRAP"],
            "horizons_requested": list(HORIZON_IDS), "horizons_available": available_horizons,
            "model_families": list(families), "training_populations": populations,
            "target_maturity_evidence": {
                horizon: populations[horizon]["maximum_label_maturity_timestamp"] for horizon in HORIZON_IDS
            },
            "preprocessing": preprocessing, "models": models,
            "predictions": predictions, "prediction_checksum": canonical_hash(predictions),
            "combined_scores": combined, "combined_score_checksum": canonical_hash(combined),
            "diagnostics": {"per_horizon": diagnostics, "cross_horizon": cross},
            "missing_horizon_policy": _missing_state(available_horizons),
            "required_purge_horizon_sessions": max(
                target["horizon_sessions"] for target in base["target_contract"]["horizons"]
                if target["horizon_id"] in available_horizons
            ) if available_horizons else None,
            "required_embargo_convention": "registered_policy_at_least_longest_included_horizon",
            "block_bootstrap_compatible": True, "statistical_significance_performed": False,
            "configuration": config, "configuration_checksum": canonical_hash(config),
            "input_checksum": base["logical_input_checksum"],
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except _FittedMemberCallbackFailure as exc:
        assert exc.__cause__ is not None
        raise exc.__cause__
    except MultiHorizonError as exc:
        return _blocked(data, config, exc)
    except Exception as exc:  # pragma: no cover
        return _blocked(data, config, MultiHorizonError("NUMERICAL_FAILURE", type(exc).__name__))


def verify_multi_horizon_result(data: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    config = result.get("configuration", {})
    expected = fit_multi_horizon_linear_selector(
        data, training_cutoff=config.get("training_cutoff"),
        model_families=config.get("model_families", ("ridge", "elastic_net")),
        ridge_alpha=config.get("ridge_alpha", 1.0),
        elastic_alpha=config.get("elastic_alpha", 0.001),
        elastic_l1_ratio=config.get("elastic_l1_ratio", 0.25),
        elastic_tolerance=config.get("elastic_tolerance", 1e-4),
        elastic_maximum_iterations=config.get("elastic_maximum_iterations", 5000),
        minimum_training_rows=config.get("minimum_training_rows", 5),
        minimum_rank_diversity=config.get("minimum_rank_diversity", 2),
    )
    fields = (
        "status", "horizons_available", "training_populations", "preprocessing",
        "models", "predictions", "prediction_checksum", "combined_scores",
        "combined_score_checksum", "diagnostics", "configuration_checksum",
        "input_checksum", "logical_result_checksum",
    )
    reasons = [f"{field.upper()}_MISMATCH" for field in fields if result.get(field) != expected.get(field)]
    return {"contract_version": "multi_horizon_linear_verification_v1", "valid": not reasons, "blocking_reasons": reasons}


def _prediction_rows(validation, values, target, family, model, base, cutoff, maturity, minimum_diversity):
    groups = {}
    for row, value in zip(validation, values):
        groups.setdefault(row["decision_timestamp"], []).append((row, float(value)))
    output = []
    for decision, rows in sorted(groups.items()):
        ranked = sorted(rows, key=lambda item: (-item[1], item[0]["asset_id"], item[0]["row_id"]))
        if len(ranked) >= minimum_diversity and len({value for _, value in ranked}) < minimum_diversity:
            raise MultiHorizonError("INSUFFICIENT_HORIZONS", f"DEGENERATE_RANKING:{target['horizon_id']}:{family}")
        for rank, (row, value) in enumerate(ranked, 1):
            output.append({
                "contract_version": PREDICTION_CONTRACT, "row_id": row["row_id"],
                "asset_id": row["asset_id"], "decision_timestamp": decision,
                "horizon_id": target["horizon_id"], "horizon_sessions": target["horizon_sessions"],
                "model_family": family, "score": value,
                "score_semantics": "continuous_return_prediction",
                "within_date_rank": rank,
                "within_date_percentile_rank": (len(ranked) - rank) / max(len(ranked) - 1, 1),
                "model_checksum": model["model_checksum"], "training_cutoff": cutoff,
                "maximum_label_maturity_timestamp": maturity,
                "feature_schema_checksum": base["feature_schema_checksum"],
                "target_checksum": target["target_checksum"], "fold_identity": base["fold_identity"],
                "prediction_population_checksum": canonical_hash([item[0]["row_id"] for item in ranked]),
            })
    return sorted(output, key=lambda row: (
        row["horizon_sessions"], row["model_family"], row["decision_timestamp"], row["asset_id"], row["row_id"]
    ))


def _combined_scores(validation, predictions, available):
    by_key = {}
    for row in predictions:
        by_key.setdefault((row["row_id"], row["horizon_id"]), []).append(row)
    output = []
    for validation_row in validation:
        row_id = validation_row["row_id"]
        horizon_scores, horizon_percentiles = {}, {}
        for horizon in available:
            rows = by_key.get((row_id, horizon), [])
            if rows:
                horizon_scores[horizon] = float(np.mean([row["score"] for row in rows]))
                horizon_percentiles[horizon] = float(np.mean([row["within_date_percentile_rank"] for row in rows]))
        scores = {}
        for name, weights in COMBINATION_WEIGHTS.items():
            scores[name] = (
                sum(weights[horizon] * horizon_scores[horizon] for horizon in weights)
                if all(horizon in horizon_scores for horizon in weights) else None
            )
        if len(horizon_scores) >= 2:
            signs = np.sign(list(horizon_scores.values()))
            sign_agreement = max(np.mean(signs >= 0), np.mean(signs <= 0))
            rank_stability = 1 - float(np.std(list(horizon_percentiles.values())))
            persistence = 0.5 * float(sign_agreement) + 0.5 * max(0, rank_stability)
            sign_disagreement = 1 - float(sign_agreement)
            rank_range = max(horizon_percentiles.values()) - min(horizon_percentiles.values())
            short_long_conflict = (
                float(np.sign(horizon_scores.get("return_1s", 0)) != np.sign(horizon_scores.get("return_20s", 0)))
                if "return_1s" in horizon_scores and "return_20s" in horizon_scores else 0
            )
            disagreement = 0.4 * sign_disagreement + 0.4 * rank_range + 0.2 * short_long_conflict
        else:
            persistence = disagreement = None
        output.append({
            "contract_version": COMBINED_CONTRACT, "row_id": row_id,
            "asset_id": validation_row["asset_id"],
            "decision_timestamp": validation_row["decision_timestamp"],
            "horizons_contributed": sorted(horizon_scores, key=HORIZON_IDS.index),
            "horizon_scores": horizon_scores, "horizon_percentile_ranks": horizon_percentiles,
            "short_term_score": scores["short_term"], "medium_term_score": scores["medium_term"],
            "long_term_score": scores["long_term"], "persistence_score": persistence,
            "horizon_disagreement": disagreement,
            "combination_weights": COMBINATION_WEIGHTS,
        })
    return sorted(output, key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))


def _cross_horizon_diagnostics(models, predictions, combined):
    ridge_models = {model["horizon_id"]: np.asarray(model["coefficient_vector"]) for model in models if model["model_family"] == "ridge"}
    coefficient_similarity, sign_consistency = {}, {}
    for left_index, left in enumerate(HORIZON_IDS):
        for right in HORIZON_IDS[left_index + 1:]:
            if left in ridge_models and right in ridge_models:
                key = f"{left}__{right}"
                coefficient_similarity[key] = _correlation(ridge_models[left], ridge_models[right])
                sign_consistency[key] = float(np.mean(np.sign(ridge_models[left]) == np.sign(ridge_models[right])))
    ensemble = {}
    for row in predictions:
        ensemble.setdefault((row["row_id"], row["horizon_id"]), []).append(row["score"])
    prediction_correlation, rank_correlation, top_k_overlap, sign_agreement = {}, {}, {}, {}
    for left_index, left in enumerate(HORIZON_IDS):
        for right in HORIZON_IDS[left_index + 1:]:
            common = sorted({
                row_id for row_id, horizon in ensemble if horizon == left
            } & {
                row_id for row_id, horizon in ensemble if horizon == right
            })
            if not common:
                continue
            left_values = [np.mean(ensemble[(row_id, left)]) for row_id in common]
            right_values = [np.mean(ensemble[(row_id, right)]) for row_id in common]
            key = f"{left}__{right}"
            prediction_correlation[key] = _correlation(left_values, right_values)
            rank_correlation[key] = _spearman(left_values, right_values)
            k = min(3, len(common))
            left_top = {value for _, value in sorted(zip(left_values, common), reverse=True)[:k]}
            right_top = {value for _, value in sorted(zip(right_values, common), reverse=True)[:k]}
            top_k_overlap[key] = len(left_top & right_top) / k
            sign_agreement[key] = float(np.mean(np.sign(left_values) == np.sign(right_values)))
    persistence = [row["persistence_score"] for row in combined if row["persistence_score"] is not None]
    disagreement = [row["horizon_disagreement"] for row in combined if row["horizon_disagreement"] is not None]
    return {
        "ridge_coefficient_similarity": coefficient_similarity,
        "ridge_coefficient_sign_consistency": sign_consistency,
        "prediction_correlation": prediction_correlation, "rank_correlation": rank_correlation,
        "top_3_overlap": top_k_overlap, "score_sign_agreement": sign_agreement,
        "persistence_distribution": _summary(persistence),
        "disagreement_distribution": _summary(disagreement),
    }


def _missing_state(available):
    available = set(available)
    if available == set(HORIZON_IDS):
        return "ALL_HORIZONS_AVAILABLE"
    if available == {"return_1s"}:
        return "SHORT_ONLY"
    if available and available.issubset({"return_1s", "return_5s", "return_10s"}):
        return "LONG_HORIZON_MISSING" if len(available) >= 2 else "INSUFFICIENT_HORIZONS"
    if len(available) >= 2:
        return "SHORT_AND_MEDIUM"
    return "INSUFFICIENT_HORIZONS"


def _validated_input(data):
    base = multi_horizon_linear_input(
        data["rows"], target_contract=data["target_contract"],
        feature_schema_identity=data["feature_schema_identity"],
        dataset_identity=data["dataset_identity"], fold_identity=data["fold_identity"],
        source_population_checksum=data["source_population_checksum"],
    )
    if data.get("logical_input_checksum") and data["logical_input_checksum"] != base["logical_input_checksum"]:
        raise MultiHorizonError("INVALID_INPUT", "INPUT_CHECKSUM_MISMATCH")
    return base


def _correlation(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left, right):
    return _correlation(_ranks(left), _ranks(right))


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


def _summary(values):
    if not values:
        return None
    return {
        "mean": float(np.mean(values)), "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
    }


def _blocked(data, config, error):
    logical = {
        "contract_version": RESULT_CONTRACT,
        "status": error.status if error.status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [error.reason], "warnings": [],
        "horizons_requested": list(HORIZON_IDS), "horizons_available": [],
        "model_families": config.get("model_families", []),
        "training_populations": {}, "target_maturity_evidence": {},
        "preprocessing": {}, "models": [], "predictions": [], "prediction_checksum": None,
        "combined_scores": [], "combined_score_checksum": None, "diagnostics": {},
        "missing_horizon_policy": "INSUFFICIENT_HORIZONS",
        "configuration": config, "configuration_checksum": canonical_hash(config),
        "input_checksum": data.get("logical_input_checksum") if isinstance(data, Mapping) else None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise MultiHorizonError("TEMPORAL_VIOLATION", "TIMESTAMP_INVALID") from exc


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
