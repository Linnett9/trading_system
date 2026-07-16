from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "hierarchical_risk_parity_input_v1"
RESULT_CONTRACT = "hierarchical_risk_parity_result_v1"
COMPARISON_CONTRACT = "hierarchical_risk_parity_comparison_v1"
DISTANCE_CONVENTION = "sqrt_one_minus_correlation_over_two_v1"
LINKAGE_METHOD = "single"
QUASI_DIAGONAL_CONVENTION = "scipy_leaves_list_canonical_input_v1"
RECURSION_CONVENTION = "ordered_half_split_inverse_cluster_variance_v1"
SECTOR_AGGREGATION = "inverse_volatility_sector_return_v1"
PSD_TOLERANCE = 1e-8
CONSTRAINT_TOLERANCE = 1e-7
VOLATILITY_FLOOR = 1e-10
STATUSES = {
    "VALID", "INFEASIBLE", "INSUFFICIENT_DATA", "INVALID_INPUT",
    "UNSUPPORTED_CONFIGURATION", "CLUSTERING_FAILURE", "NUMERICAL_FAILURE",
}


class HRPInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def hrp_input(
    asset_ids: Sequence[str],
    *,
    return_matrix: Sequence[Sequence[float]] | None = None,
    observation_ids: Sequence[str] | None = None,
    covariance: Sequence[Sequence[float]] | None = None,
    covariance_estimator_identity: str = "sample_covariance_n_minus_1_v1",
    candidate_universe_identity: str = "synthetic_candidate_universe",
    selector_scores: Sequence[float] | None = None,
    requested_top_k: int | None = None,
    exposure_target: float = 1.0,
    maximum_weights: float | Sequence[float] = 1.0,
    minimum_weights: float | Sequence[float] = 0.0,
    sector_ids: Sequence[str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    liquidity_eligible: Sequence[bool] | None = None,
    previous_weights: Sequence[float] | None = None,
    cash_allowed: bool = True,
    distance_convention: str = DISTANCE_CONVENTION,
    linkage_method: str = LINKAGE_METHOD,
    quasi_diagonalisation: str = QUASI_DIAGONAL_CONVENTION,
    recursive_bisection: str = RECURSION_CONVENTION,
    annualisation_factor: float = 252.0,
    missingness_policy: str = "reject_any_missing",
    decision_timestamp: str = "synthetic",
    minimum_observations: int = 3,
) -> dict[str, Any]:
    assets = [str(value) for value in asset_ids]
    n = len(assets)
    if not assets or len(set(assets)) != n:
        raise HRPInputError("INVALID_INPUT", "ASSET_IDENTITIES_INVALID")
    if assets != sorted(assets):
        raise HRPInputError("INVALID_INPUT", "ASSETS_NOT_DETERMINISTICALLY_ORDERED")
    if distance_convention != DISTANCE_CONVENTION:
        raise HRPInputError("UNSUPPORTED_CONFIGURATION", "DISTANCE_CONVENTION_UNSUPPORTED")
    if linkage_method != LINKAGE_METHOD:
        raise HRPInputError("UNSUPPORTED_CONFIGURATION", "LINKAGE_METHOD_UNSUPPORTED")
    if quasi_diagonalisation != QUASI_DIAGONAL_CONVENTION or recursive_bisection != RECURSION_CONVENTION:
        raise HRPInputError("UNSUPPORTED_CONFIGURATION", "HRP_ORDER_OR_RECURSION_UNSUPPORTED")
    if missingness_policy != "reject_any_missing":
        raise HRPInputError("UNSUPPORTED_CONFIGURATION", "MISSINGNESS_POLICY_UNSUPPORTED")
    if not math.isfinite(annualisation_factor) or annualisation_factor <= 0:
        raise HRPInputError("INVALID_INPUT", "ANNUALISATION_FACTOR_INVALID")
    history = None
    observations: list[str] = []
    if return_matrix is not None:
        history = np.asarray(return_matrix, dtype=float)
        if history.ndim != 2 or history.shape[1] != n:
            raise HRPInputError("INVALID_INPUT", "RETURN_MATRIX_DIMENSION_MISMATCH")
        if history.shape[0] < minimum_observations:
            raise HRPInputError("INSUFFICIENT_DATA", "OBSERVATION_COUNT_INSUFFICIENT")
        if not np.isfinite(history).all():
            raise HRPInputError("INVALID_INPUT", "RETURN_MATRIX_NON_FINITE")
        observations = [str(value) for value in (observation_ids or [])]
        if len(observations) != history.shape[0] or len(set(observations)) != len(observations):
            raise HRPInputError("INVALID_INPUT", "OBSERVATION_IDENTITIES_INVALID")
        if observations != sorted(observations):
            raise HRPInputError("INVALID_INPUT", "OBSERVATIONS_NOT_DETERMINISTICALLY_ORDERED")
    supplied_covariance = None
    if covariance is not None:
        supplied_covariance = np.asarray(covariance, dtype=float)
        _validate_covariance(supplied_covariance, n)
    if history is None and supplied_covariance is None:
        raise HRPInputError("INVALID_INPUT", "RETURN_OR_COVARIANCE_INPUT_REQUIRED")
    if history is None and observation_ids:
        raise HRPInputError("INVALID_INPUT", "OBSERVATIONS_WITHOUT_RETURNS")
    scores = None
    if selector_scores is not None:
        scores = _finite_vector(selector_scores, n, "SELECTOR_SCORES")
    if requested_top_k is not None and (not isinstance(requested_top_k, int) or requested_top_k < 1):
        raise HRPInputError("INVALID_INPUT", "TOP_K_INVALID")
    if requested_top_k is not None and scores is None:
        raise HRPInputError("INVALID_INPUT", "TOP_K_REQUIRES_SELECTOR_SCORES")
    minimum = _bounds(minimum_weights, n, "MINIMUM_WEIGHTS")
    maximum = _bounds(maximum_weights, n, "MAXIMUM_WEIGHTS")
    if any(value < 0 for value in minimum + maximum) or any(low > high for low, high in zip(minimum, maximum)):
        raise HRPInputError("INVALID_INPUT", "STOCK_BOUNDS_INVALID")
    eligibility = list(liquidity_eligible if liquidity_eligible is not None else [True] * n)
    if len(eligibility) != n or any(not isinstance(value, (bool, np.bool_)) for value in eligibility):
        raise HRPInputError("INVALID_INPUT", "LIQUIDITY_MASK_INVALID")
    effective_minimum = [minimum[index] if eligibility[index] else 0.0 for index in range(n)]
    effective_maximum = [maximum[index] if eligibility[index] else 0.0 for index in range(n)]
    if not math.isfinite(exposure_target) or exposure_target < 0 or (cash_allowed and exposure_target > 1 + CONSTRAINT_TOLERANCE):
        raise HRPInputError("INVALID_INPUT", "EXPOSURE_TARGET_INVALID")
    if not cash_allowed and abs(exposure_target - 1) > CONSTRAINT_TOLERANCE:
        raise HRPInputError("INVALID_INPUT", "CASH_DISALLOWED_REQUIRES_FULL_EXPOSURE")
    if sum(effective_minimum) > exposure_target + CONSTRAINT_TOLERANCE or sum(effective_maximum) < exposure_target - CONSTRAINT_TOLERANCE:
        raise HRPInputError("INFEASIBLE", "STOCK_CAPS_CANNOT_MEET_EXPOSURE")
    sectors = [str(value) for value in (sector_ids or ["UNCLASSIFIED"] * n)]
    if len(sectors) != n or any(not value for value in sectors):
        raise HRPInputError("INVALID_INPUT", "SECTOR_MAPPING_INVALID")
    caps = {str(key): float(value) for key, value in sorted((sector_caps or {}).items())}
    if any(not math.isfinite(value) or value < 0 for value in caps.values()):
        raise HRPInputError("INVALID_INPUT", "SECTOR_CAP_INVALID")
    total_capacity = sum(min(caps.get(sector, exposure_target), sum(effective_maximum[index] for index, value in enumerate(sectors) if value == sector)) for sector in sorted(set(sectors)))
    if total_capacity < exposure_target - CONSTRAINT_TOLERANCE:
        raise HRPInputError("INFEASIBLE", "SECTOR_CAPS_CANNOT_MEET_EXPOSURE")
    previous = _finite_vector(previous_weights or [0] * n, n, "PREVIOUS_WEIGHTS")
    population_checksum = canonical_hash({"contract": "portfolio_risk_input_v1", "asset_ids": assets})
    observation_checksum = canonical_hash({"observation_ids": observations}) if observations else None
    history_checksum = canonical_hash({"observation_ids": observations, "returns": history.tolist()}) if history is not None else None
    config = {
        "covariance_estimator_identity": covariance_estimator_identity,
        "candidate_universe_identity": candidate_universe_identity, "requested_top_k": requested_top_k,
        "exposure_target": exposure_target, "maximum_weights": maximum, "minimum_weights": minimum,
        "sector_ids": sectors, "sector_caps": caps, "liquidity_eligible": eligibility,
        "cash_allowed": cash_allowed, "distance_convention": distance_convention,
        "linkage_method": linkage_method, "quasi_diagonalisation": quasi_diagonalisation,
        "recursive_bisection": recursive_bisection, "annualisation_factor": annualisation_factor,
        "missingness_policy": missingness_policy, "decision_timestamp": decision_timestamp,
    }
    return {
        "contract_version": INPUT_CONTRACT, "asset_ids": assets,
        "return_matrix": history.tolist() if history is not None else None,
        "observation_ids": observations, "covariance": supplied_covariance.tolist() if supplied_covariance is not None else None,
        "covariance_estimator_identity": covariance_estimator_identity,
        "candidate_universe_identity": candidate_universe_identity, "selector_scores": scores,
        "requested_top_k": requested_top_k, "exposure_target": exposure_target,
        "maximum_weights": maximum, "minimum_weights": minimum,
        "effective_maximum_weights": effective_maximum, "effective_minimum_weights": effective_minimum,
        "sector_ids": sectors, "sector_caps": caps, "liquidity_eligible": eligibility,
        "previous_weights": previous, "cash_allowed": cash_allowed,
        "distance_convention": distance_convention, "linkage_method": linkage_method,
        "quasi_diagonalisation": quasi_diagonalisation, "recursive_bisection": recursive_bisection,
        "annualisation_factor": annualisation_factor, "missingness_policy": missingness_policy,
        "decision_timestamp": decision_timestamp, "population_checksum": population_checksum,
        "observation_population_checksum": observation_checksum, "return_history_checksum": history_checksum,
        "configuration_checksum": canonical_hash(config),
    }


def correlation_distance(covariance: Sequence[Sequence[float]]) -> dict[str, Any]:
    matrix = np.asarray(covariance, dtype=float)
    _validate_covariance(matrix, matrix.shape[0])
    volatility = np.sqrt(np.maximum(np.diag(matrix), 0))
    if np.any(volatility <= VOLATILITY_FLOOR):
        raise HRPInputError("INVALID_INPUT", "CONSTANT_OR_NEAR_CONSTANT_SERIES")
    correlation = matrix / np.outer(volatility, volatility)
    if np.any(correlation < -1 - 1e-8) or np.any(correlation > 1 + 1e-8):
        raise HRPInputError("INVALID_INPUT", "CORRELATION_OUT_OF_BOUNDS")
    correlation = np.clip((correlation + correlation.T) / 2, -1, 1)
    np.fill_diagonal(correlation, 1)
    distance = np.sqrt(np.maximum((1 - correlation) / 2, 0))
    np.fill_diagonal(distance, 0)
    if not np.allclose(distance, distance.T, atol=1e-12) or not np.allclose(np.diag(distance), 0, atol=1e-12):
        raise HRPInputError("NUMERICAL_FAILURE", "DISTANCE_MATRIX_INVALID")
    return {
        "distance_contract": DISTANCE_CONVENTION, "correlation": correlation.tolist(),
        "distance": distance.tolist(), "correlation_checksum": canonical_hash(correlation.tolist()),
        "distance_checksum": canonical_hash(distance.tolist()),
    }


def run_hrp(contract: Mapping[str, Any], *, variant: str = "standard") -> dict[str, Any]:
    policy_ids = {
        "standard": "standard_candidate_universe_hrp_v1",
        "top_20": "hrp_top_20_v1", "top_40": "hrp_top_40_v1",
        "sector_first": "sector_first_hrp_v1",
    }
    try:
        data = _validated_contract(contract)
        if variant not in policy_ids:
            raise HRPInputError("UNSUPPORTED_CONFIGURATION", "HRP_VARIANT_UNSUPPORTED")
        selected = _selected_indexes(data, variant)
        if not selected:
            raise HRPInputError("INSUFFICIENT_DATA", "NO_ELIGIBLE_CANDIDATES")
        covariance = _input_covariance(data)
        if variant == "sector_first":
            raw_selected, evidence = _sector_first(data, selected)
            cluster = evidence["sector_cluster"]
            split_ledger = evidence["split_ledger"]
        else:
            raw_local, cluster = _hrp_core(covariance[np.ix_(selected, selected)], [data["asset_ids"][index] for index in selected])
            raw_selected = {index: raw_local[position] for position, index in enumerate(selected)}
            split_ledger = cluster["split_ledger"]
            evidence = None
        raw = np.zeros(len(data["asset_ids"]))
        for index, value in raw_selected.items(): raw[index] = value * data["exposure_target"]
        final = _post_process(data, raw)
        return _result(data, policy_ids[variant], variant, covariance, raw, final, selected, cluster, split_ledger, evidence)
    except HRPInputError as exc:
        return _blocked(contract, policy_ids.get(variant, "unknown_hrp"), variant, exc.status, exc.reason)
    except Exception as exc:  # pragma: no cover
        return _blocked(contract, policy_ids.get(variant, "unknown_hrp"), variant, "CLUSTERING_FAILURE", type(exc).__name__)


def verify_hrp_result(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        if result.get("population_checksum") != data["population_checksum"]: reasons.append("POPULATION_MISMATCH")
        if result.get("return_history_checksum") != data["return_history_checksum"]: reasons.append("RETURN_HISTORY_MISMATCH")
        variant = result.get("hrp_variant")
        recomputed = run_hrp(data, variant=variant)
        for key, reason in (
            ("covariance_checksum", "COVARIANCE_MISMATCH"),
            ("correlation_checksum", "CORRELATION_MISMATCH"),
            ("distance_checksum", "DISTANCE_MISMATCH"),
            ("linkage_matrix", "LINKAGE_MISMATCH"),
            ("leaf_order", "LEAF_ORDER_MISMATCH"),
            ("cluster_tree_checksum", "TREE_CHECKSUM_MISMATCH"),
            ("recursive_split_ledger", "SPLIT_LEDGER_MISMATCH"),
            ("raw_hrp_weights", "RAW_WEIGHT_MISMATCH"),
            ("final_weights", "FINAL_WEIGHT_MISMATCH"),
        ):
            if result.get(key) != recomputed.get(key): reasons.append(reason)
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical): reasons.append("RESULT_CHECKSUM_MISMATCH")
        return {"contract_version": "hierarchical_risk_parity_verification_v1", "valid": not reasons, "blocking_reasons": sorted(set(reasons))}
    except HRPInputError as exc:
        return {"contract_version": "hierarchical_risk_parity_verification_v1", "valid": False, "blocking_reasons": [exc.reason]}


def compare_hrp_policies(contract: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    data = _validated_contract(contract)
    covariance = _input_covariance(data)
    rows = {}
    for name, result in sorted(results.items()):
        if result.get("population_checksum") != data["population_checksum"]:
            raise HRPInputError("INVALID_INPUT", f"COMPARISON_POPULATION_MISMATCH:{name}")
        if result.get("return_history_checksum") not in {None, data["return_history_checksum"]}:
            raise HRPInputError("INVALID_INPUT", f"COMPARISON_HISTORY_MISMATCH:{name}")
        weights = np.asarray(result.get("final_weights", result.get("target_weights", ())), dtype=float)
        if weights.shape != (len(data["asset_ids"]),):
            raise HRPInputError("INVALID_INPUT", f"COMPARISON_WEIGHTS_INVALID:{name}")
        variance = float(weights @ covariance @ weights)
        sectors = _sector_exposures(data, weights)
        cluster_weights = result.get("raw_hrp_weights", weights.tolist())
        rows[name] = {
            "expected_variance": variance,
            "expected_volatility": math.sqrt(max(variance * data["annualisation_factor"], 0)),
            "concentration_hhi": float(weights @ weights),
            "effective_holdings": float(1 / (weights @ weights)) if weights @ weights > 0 else 0,
            "maximum_stock_weight": float(weights.max()), "sector_concentration_hhi": float(sum(value**2 for value in sectors.values())),
            "gross_turnover": float(np.abs(weights - np.asarray(data["previous_weights"])).sum()),
            "cluster_concentration_hhi": float(np.asarray(cluster_weights) @ np.asarray(cluster_weights)),
        }
    logical = {
        "contract_version": COMPARISON_CONTRACT, "status": "VALID", "valid": True,
        "population_checksum": data["population_checksum"], "return_history_checksum": data["return_history_checksum"],
        "policy_count": len(rows), "policies": rows, "historical_returns_computed": False,
        "sharpe_computed": False, "promotion_decision": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _hrp_core(covariance, asset_ids):
    if len(asset_ids) == 1:
        return np.asarray([1.0]), {
            "linkage_method": LINKAGE_METHOD, "linkage_matrix": [], "leaf_order": [0],
            "ordered_assets": asset_ids, "cluster_tree_checksum": canonical_hash({"single_asset": asset_ids[0]}),
            "split_ledger": [],
        }
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise HRPInputError("CLUSTERING_FAILURE", "SCIPY_CLUSTERING_UNAVAILABLE") from exc
    distance_payload = correlation_distance(covariance)
    distance = np.asarray(distance_payload["distance"])
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method=LINKAGE_METHOD, optimal_ordering=False)
    if tree.shape != (len(asset_ids) - 1, 4) or not np.isfinite(tree).all():
        raise HRPInputError("CLUSTERING_FAILURE", "LINKAGE_MATRIX_INVALID")
    order = leaves_list(tree).astype(int).tolist()
    if sorted(order) != list(range(len(asset_ids))):
        raise HRPInputError("CLUSTERING_FAILURE", "LEAF_ORDER_INVALID")
    weights, ledger = _recursive_bisection(covariance, order, asset_ids)
    tree_identity = {"method": LINKAGE_METHOD, "linkage": tree.tolist(), "leaf_order": order, "assets": asset_ids}
    return weights, {
        **distance_payload, "linkage_method": LINKAGE_METHOD, "linkage_matrix": tree.tolist(),
        "leaf_order": order, "ordered_assets": [asset_ids[index] for index in order],
        "cluster_tree_checksum": canonical_hash(tree_identity), "split_ledger": ledger,
    }


def _recursive_bisection(covariance, order, asset_ids):
    weights = np.ones(len(order))
    clusters = [list(order)]
    ledger = []
    round_number = 0
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1: continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_variance = _cluster_variance(covariance, left)
            right_variance = _cluster_variance(covariance, right)
            total = left_variance + right_variance
            if not math.isfinite(total) or total <= VOLATILITY_FLOOR:
                raise HRPInputError("NUMERICAL_FAILURE", "CLUSTER_RISK_INVALID")
            left_fraction = right_variance / total
            right_fraction = left_variance / total
            for index in left: weights[index] *= left_fraction
            for index in right: weights[index] *= right_fraction
            ledger.append({
                "round": round_number, "left_indexes": left, "right_indexes": right,
                "left_assets": [asset_ids[index] for index in left],
                "right_assets": [asset_ids[index] for index in right],
                "left_cluster_variance": left_variance, "right_cluster_variance": right_variance,
                "left_allocation_fraction": left_fraction, "right_allocation_fraction": right_fraction,
            })
            next_clusters.extend([left, right])
        clusters = next_clusters; round_number += 1
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise HRPInputError("NUMERICAL_FAILURE", "RAW_HRP_WEIGHT_INVALID")
    return weights / weights.sum(), ledger


def _cluster_variance(covariance, indexes):
    sub = covariance[np.ix_(indexes, indexes)]
    diagonal = np.diag(sub)
    if np.any(diagonal <= VOLATILITY_FLOOR):
        raise HRPInputError("INVALID_INPUT", "CLUSTER_ZERO_VARIANCE")
    inverse = 1 / diagonal; inverse /= inverse.sum()
    return float(inverse @ sub @ inverse)


def _sector_first(data, selected):
    history = np.asarray(data["return_matrix"]) if data["return_matrix"] is not None else None
    if history is None:
        raise HRPInputError("UNSUPPORTED_CONFIGURATION", "SECTOR_FIRST_REQUIRES_RETURN_HISTORY")
    sectors = sorted({data["sector_ids"][index] for index in selected})
    if any(not sector for sector in sectors):
        raise HRPInputError("INVALID_INPUT", "SECTOR_ID_MISSING")
    sector_series = []
    within = {}
    full_ledger = []
    for sector in sectors:
        indexes = [index for index in selected if data["sector_ids"][index] == sector]
        local_covariance = np.cov(history[:, indexes], rowvar=False, ddof=1)
        if len(indexes) == 1:
            local_covariance = np.asarray([[float(np.var(history[:, indexes[0]], ddof=1))]])
        local_weights, local_evidence = _hrp_core(local_covariance, [data["asset_ids"][index] for index in indexes])
        vol = np.std(history[:, indexes], axis=0, ddof=1)
        if np.any(vol <= VOLATILITY_FLOOR): raise HRPInputError("INVALID_INPUT", "SECTOR_CONSTANT_SERIES")
        aggregation = 1 / vol; aggregation /= aggregation.sum()
        sector_series.append(history[:, indexes] @ aggregation)
        within[sector] = {"indexes": indexes, "weights": local_weights.tolist(), "tree": local_evidence, "aggregation_weights": aggregation.tolist()}
        full_ledger.extend([{"level": "within_sector", "sector": sector, **row} for row in local_evidence["split_ledger"]])
    sector_matrix = np.column_stack(sector_series)
    sector_covariance = np.cov(sector_matrix, rowvar=False, ddof=1)
    if len(sectors) == 1: sector_covariance = np.asarray([[float(np.var(sector_matrix[:, 0], ddof=1))]])
    sector_weights, sector_tree = _hrp_core(sector_covariance, sectors)
    raw = {}
    for sector_position, sector in enumerate(sectors):
        evidence = within[sector]
        for local_position, index in enumerate(evidence["indexes"]):
            raw[index] = float(sector_weights[sector_position] * evidence["weights"][local_position])
    full_ledger.extend([{"level": "sector", **row} for row in sector_tree["split_ledger"]])
    return raw, {
        "sector_return_aggregation": SECTOR_AGGREGATION, "sector_ids": sectors,
        "sector_weights": sector_weights.tolist(), "sector_cluster": sector_tree,
        "within_sector": within, "split_ledger": full_ledger,
    }


def _selected_indexes(data, variant):
    eligible = [index for index, value in enumerate(data["liquidity_eligible"]) if value]
    if variant in {"top_20", "top_40"}:
        k = 20 if variant == "top_20" else 40
        if len(eligible) < k: raise HRPInputError("INSUFFICIENT_DATA", f"TOP_K_CANDIDATES_INSUFFICIENT:{k}")
        if data["selector_scores"] is None: raise HRPInputError("INVALID_INPUT", "TOP_K_REQUIRES_SELECTOR_SCORES")
        return sorted(eligible, key=lambda index: (-data["selector_scores"][index], data["asset_ids"][index]))[:k]
    return eligible


def _post_process(data, raw):
    weights = np.asarray(data["effective_minimum_weights"], dtype=float)
    maximum = np.asarray(data["effective_maximum_weights"], dtype=float)
    remaining = data["exposure_target"] - weights.sum()
    scores = np.asarray(raw, dtype=float)
    for _ in range(len(weights) * 5 + 5):
        if remaining <= CONSTRAINT_TOLERANCE: break
        sector_exposures = _sector_exposures(data, weights)
        room = np.maximum(maximum - weights, 0)
        active = (room > CONSTRAINT_TOLERANCE) & (scores > 0)
        if not np.any(active): raise HRPInputError("INFEASIBLE", "HRP_CAP_REDISTRIBUTION_EXHAUSTED")
        proposed = scores * active; proposed = proposed / proposed.sum() * remaining
        addition = np.minimum(proposed, room)
        for sector, cap in data["sector_caps"].items():
            indexes = np.asarray([index for index, value in enumerate(data["sector_ids"]) if value == sector], dtype=int)
            sector_room = max(cap - sector_exposures[sector], 0)
            total = float(addition[indexes].sum())
            if total > sector_room and total > 0: addition[indexes] *= sector_room / total
        if addition.sum() <= CONSTRAINT_TOLERANCE: raise HRPInputError("INFEASIBLE", "HRP_CAP_REDISTRIBUTION_STALLED")
        weights += addition; remaining -= float(addition.sum())
    if remaining > CONSTRAINT_TOLERANCE: raise HRPInputError("INFEASIBLE", "HRP_EXPOSURE_NOT_ALLOCATED")
    return weights


def _result(data, policy_id, variant, covariance, raw, final, selected, cluster, split_ledger, sector_evidence):
    distance_payload = correlation_distance(covariance[np.ix_(selected, selected)])
    change = final - np.asarray(data["previous_weights"])
    variance = float(final @ covariance @ final)
    sectors = _sector_exposures(data, final)
    selected_assets = [data["asset_ids"][index] for index in selected]
    excluded_assets = [asset for index, asset in enumerate(data["asset_ids"]) if index not in selected]
    logical = {
        "contract_version": RESULT_CONTRACT, "policy_id": policy_id, "policy_version": "1.0",
        "hrp_variant": variant, "status": "VALID", "valid": True, "blocking_reasons": [],
        "warnings": ["CONSTRAINED_POST_PROCESSING_APPLIED"] if not np.allclose(raw, final) else [],
        "asset_ids": data["asset_ids"], "selected_candidate_assets": selected_assets, "excluded_assets": excluded_assets,
        "raw_hrp_weights": raw.tolist(), "final_weights": final.tolist(),
        "post_processing_weight_changes": (final - raw).tolist(),
        "trade_weight_changes": change.tolist(), "gross_turnover": float(np.abs(change).sum()),
        "expected_variance": variance, "expected_volatility": math.sqrt(max(variance * data["annualisation_factor"], 0)),
        "covariance_identity": data["covariance_estimator_identity"], "covariance_checksum": canonical_hash(covariance.tolist()),
        "correlation_checksum": distance_payload["correlation_checksum"], "distance_checksum": distance_payload["distance_checksum"],
        "correlation_matrix": distance_payload["correlation"], "distance_matrix": distance_payload["distance"],
        "linkage_method": cluster["linkage_method"], "linkage_matrix": cluster["linkage_matrix"],
        "leaf_order": cluster["leaf_order"], "ordered_cluster_assets": cluster["ordered_assets"],
        "cluster_tree_checksum": cluster["cluster_tree_checksum"], "recursive_split_ledger": split_ledger,
        "sector_first_evidence": sector_evidence,
        "concentration": {"hhi": float(final @ final), "effective_holdings": float(1 / (final @ final)), "maximum_stock_weight": float(final.max())},
        "stock_cap_utilisation": [final[index] / cap if cap > 0 else None for index, cap in enumerate(data["effective_maximum_weights"])],
        "sector_exposures": sectors,
        "sector_cap_utilisation": {sector: value / data["sector_caps"][sector] for sector, value in sectors.items() if sector in data["sector_caps"] and data["sector_caps"][sector] > 0},
        "liquidity_exclusions": [asset for asset, eligible in zip(data["asset_ids"], data["liquidity_eligible"]) if not eligible],
        "population_checksum": data["population_checksum"], "observation_population_checksum": data["observation_population_checksum"],
        "return_history_checksum": data["return_history_checksum"], "configuration_checksum": data["configuration_checksum"],
        "adv_capacity_status": "UNVERIFIED",
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked(data, policy_id, variant, status, reason):
    logical = {
        "contract_version": RESULT_CONTRACT, "policy_id": policy_id, "policy_version": "1.0",
        "hrp_variant": variant, "status": status if status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [reason], "warnings": [], "asset_ids": list(data.get("asset_ids", ())) if isinstance(data, Mapping) else [],
        "selected_candidate_assets": [], "excluded_assets": [], "raw_hrp_weights": [], "final_weights": [],
        "post_processing_weight_changes": [], "trade_weight_changes": [], "gross_turnover": None,
        "expected_variance": None, "expected_volatility": None, "covariance_identity": None, "covariance_checksum": None,
        "correlation_checksum": None, "distance_checksum": None, "correlation_matrix": [], "distance_matrix": [],
        "linkage_method": LINKAGE_METHOD, "linkage_matrix": [], "leaf_order": [], "ordered_cluster_assets": [],
        "cluster_tree_checksum": None, "recursive_split_ledger": [], "sector_first_evidence": None,
        "concentration": {}, "stock_cap_utilisation": [], "sector_exposures": {}, "sector_cap_utilisation": {},
        "liquidity_exclusions": [], "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "observation_population_checksum": data.get("observation_population_checksum") if isinstance(data, Mapping) else None,
        "return_history_checksum": data.get("return_history_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": data.get("configuration_checksum") if isinstance(data, Mapping) else None,
        "adv_capacity_status": "UNVERIFIED",
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _validated_contract(contract):
    return hrp_input(
        contract["asset_ids"], return_matrix=contract.get("return_matrix"),
        observation_ids=contract.get("observation_ids") or None, covariance=contract.get("covariance"),
        covariance_estimator_identity=str(contract.get("covariance_estimator_identity", "sample_covariance_n_minus_1_v1")),
        candidate_universe_identity=str(contract.get("candidate_universe_identity", "synthetic_candidate_universe")),
        selector_scores=contract.get("selector_scores"), requested_top_k=contract.get("requested_top_k"),
        exposure_target=float(contract.get("exposure_target", 1)), maximum_weights=contract.get("maximum_weights", 1),
        minimum_weights=contract.get("minimum_weights", 0), sector_ids=contract.get("sector_ids"),
        sector_caps=contract.get("sector_caps"), liquidity_eligible=contract.get("liquidity_eligible"),
        previous_weights=contract.get("previous_weights"), cash_allowed=bool(contract.get("cash_allowed", True)),
        distance_convention=str(contract.get("distance_convention", DISTANCE_CONVENTION)),
        linkage_method=str(contract.get("linkage_method", LINKAGE_METHOD)),
        quasi_diagonalisation=str(contract.get("quasi_diagonalisation", QUASI_DIAGONAL_CONVENTION)),
        recursive_bisection=str(contract.get("recursive_bisection", RECURSION_CONVENTION)),
        annualisation_factor=float(contract.get("annualisation_factor", 252)),
        missingness_policy=str(contract.get("missingness_policy", "reject_any_missing")),
        decision_timestamp=str(contract.get("decision_timestamp", "synthetic")),
    )


def _input_covariance(data):
    covariance = np.asarray(data["covariance"]) if data["covariance"] is not None else np.cov(np.asarray(data["return_matrix"]), rowvar=False, ddof=1)
    _validate_covariance(covariance, len(data["asset_ids"]))
    if np.any(np.diag(covariance) <= VOLATILITY_FLOOR):
        raise HRPInputError("INVALID_INPUT", "CONSTANT_OR_NEAR_CONSTANT_SERIES")
    return covariance


def _validate_covariance(matrix, n):
    if matrix.shape != (n, n): raise HRPInputError("INVALID_INPUT", "COVARIANCE_DIMENSION_MISMATCH")
    if not np.isfinite(matrix).all(): raise HRPInputError("INVALID_INPUT", "COVARIANCE_NON_FINITE")
    if not np.allclose(matrix, matrix.T, atol=PSD_TOLERANCE, rtol=0): raise HRPInputError("INVALID_INPUT", "COVARIANCE_NOT_SYMMETRIC")
    if float(np.linalg.eigvalsh((matrix + matrix.T) / 2).min()) < -PSD_TOLERANCE: raise HRPInputError("INVALID_INPUT", "COVARIANCE_NOT_PSD")


def _sector_exposures(data, weights):
    return {sector: float(sum(weights[index] for index, value in enumerate(data["sector_ids"]) if value == sector)) for sector in sorted(set(data["sector_ids"]))}


def _finite_vector(values, n, owner):
    result = [float(value) for value in values]
    if len(result) != n: raise HRPInputError("INVALID_INPUT", f"{owner}_DIMENSION_MISMATCH")
    if not all(math.isfinite(value) for value in result): raise HRPInputError("INVALID_INPUT", f"{owner}_NON_FINITE")
    return result


def _bounds(values, n, owner):
    result = [float(values)] * n if isinstance(values, (int, float)) else [float(value) for value in values]
    if len(result) != n or not all(math.isfinite(value) for value in result): raise HRPInputError("INVALID_INPUT", f"{owner}_INVALID")
    return result


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
