from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "portfolio_risk_input_v1"
INVERSE_VOL_RESULT = "inverse_volatility_portfolio_result_v1"
SHRINKAGE_RESULT = "linear_shrinkage_covariance_result_v1"
MINIMUM_VARIANCE_RESULT = "minimum_variance_portfolio_result_v1"
COMPARISON_CONTRACT = "portfolio_risk_baseline_comparison_v1"
PSD_TOLERANCE = 1e-8
CONSTRAINT_TOLERANCE = 1e-7
SOLVER_TOLERANCE = 1e-10
MAX_ITERATIONS = 2000
STATUSES = {
    "VALID", "OPTIMAL", "INFEASIBLE", "INSUFFICIENT_DATA", "INVALID_INPUT",
    "UNSUPPORTED_CONFIGURATION", "NUMERICAL_FAILURE", "SOLVER_UNAVAILABLE",
}


class PortfolioRiskInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def portfolio_risk_input(
    asset_ids: Sequence[str],
    *,
    exposure_target: float = 1.0,
    return_history: Sequence[Sequence[float]] | None = None,
    observation_ids: Sequence[str] | None = None,
    supplied_volatilities: Sequence[float] | None = None,
    supplied_covariance: Sequence[Sequence[float]] | None = None,
    previous_weights: Sequence[float] | None = None,
    minimum_weights: float | Sequence[float] = 0.0,
    maximum_weights: float | Sequence[float] = 1.0,
    sector_ids: Sequence[str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    liquidity_eligible: Sequence[bool] | None = None,
    cash_allowed: bool = True,
    annualisation_factor: float = 252.0,
    missingness_policy: str = "reject_any_missing",
    covariance_estimator_identity: str = "ledoit_wolf_scaled_identity_v1",
    policy_identity: str = "portfolio_risk_baseline_v1",
    decision_timestamp: str = "synthetic",
    minimum_observations: int = 2,
) -> dict[str, Any]:
    assets = [str(value) for value in asset_ids]
    n = len(assets)
    if n == 0 or len(set(assets)) != n:
        raise PortfolioRiskInputError("INVALID_INPUT", "ASSET_IDENTITIES_INVALID")
    if assets != sorted(assets):
        raise PortfolioRiskInputError("INVALID_INPUT", "ASSETS_NOT_DETERMINISTICALLY_ORDERED")
    if missingness_policy != "reject_any_missing":
        raise PortfolioRiskInputError("UNSUPPORTED_CONFIGURATION", "MISSINGNESS_POLICY_UNSUPPORTED")
    if not math.isfinite(exposure_target) or exposure_target < 0 or (cash_allowed and exposure_target > 1 + CONSTRAINT_TOLERANCE):
        raise PortfolioRiskInputError("INVALID_INPUT", "EXPOSURE_TARGET_INVALID")
    if not cash_allowed and abs(exposure_target - 1) > CONSTRAINT_TOLERANCE:
        raise PortfolioRiskInputError("INVALID_INPUT", "CASH_DISALLOWED_REQUIRES_FULL_EXPOSURE")
    if not math.isfinite(annualisation_factor) or annualisation_factor <= 0:
        raise PortfolioRiskInputError("INVALID_INPUT", "ANNUALISATION_FACTOR_INVALID")
    minimum = _bounds(minimum_weights, n, "MINIMUM_WEIGHTS")
    maximum = _bounds(maximum_weights, n, "MAXIMUM_WEIGHTS")
    if any(value < 0 for value in minimum + maximum) or any(low > high for low, high in zip(minimum, maximum)):
        raise PortfolioRiskInputError("INVALID_INPUT", "STOCK_BOUNDS_INVALID")
    eligibility = list(liquidity_eligible if liquidity_eligible is not None else [True] * n)
    if len(eligibility) != n or any(not isinstance(value, (bool, np.bool_)) for value in eligibility):
        raise PortfolioRiskInputError("INVALID_INPUT", "LIQUIDITY_MASK_INVALID")
    effective_maximum = [cap if eligibility[index] else 0.0 for index, cap in enumerate(maximum)]
    effective_minimum = [floor if eligibility[index] else 0.0 for index, floor in enumerate(minimum)]
    if sum(effective_minimum) > exposure_target + CONSTRAINT_TOLERANCE or sum(effective_maximum) < exposure_target - CONSTRAINT_TOLERANCE:
        raise PortfolioRiskInputError("INFEASIBLE", "STOCK_CAPS_CANNOT_MEET_EXPOSURE")
    sectors = [str(value) for value in (sector_ids or ["UNCLASSIFIED"] * n)]
    if len(sectors) != n or any(not value for value in sectors):
        raise PortfolioRiskInputError("INVALID_INPUT", "SECTOR_MAPPING_INVALID")
    caps = {str(key): float(value) for key, value in sorted((sector_caps or {}).items())}
    if any(not math.isfinite(value) or value < 0 for value in caps.values()):
        raise PortfolioRiskInputError("INVALID_INPUT", "SECTOR_CAP_INVALID")
    total_capacity = sum(min(caps.get(sector, exposure_target), sum(effective_maximum[index] for index, value in enumerate(sectors) if value == sector)) for sector in sorted(set(sectors)))
    if total_capacity < exposure_target - CONSTRAINT_TOLERANCE:
        raise PortfolioRiskInputError("INFEASIBLE", "SECTOR_CAPS_CANNOT_MEET_EXPOSURE")
    previous = _finite_vector(previous_weights or [0.0] * n, n, "PREVIOUS_WEIGHTS")
    if any(value < -CONSTRAINT_TOLERANCE for value in previous):
        raise PortfolioRiskInputError("INVALID_INPUT", "PREVIOUS_WEIGHTS_NEGATIVE")
    history = None
    observations: list[str] = []
    if return_history is not None:
        history = np.asarray(return_history, dtype=float)
        if history.ndim != 2 or history.shape[1] != n:
            raise PortfolioRiskInputError("INVALID_INPUT", "RETURN_HISTORY_DIMENSION_MISMATCH")
        if history.shape[0] < minimum_observations:
            raise PortfolioRiskInputError("INSUFFICIENT_DATA", "RETURN_HISTORY_INSUFFICIENT")
        if not np.isfinite(history).all():
            raise PortfolioRiskInputError("INVALID_INPUT", "RETURN_HISTORY_NON_FINITE")
        observations = [str(value) for value in (observation_ids or [])]
        if len(observations) != history.shape[0] or len(set(observations)) != len(observations):
            raise PortfolioRiskInputError("INVALID_INPUT", "OBSERVATION_IDENTITIES_INVALID")
        if observations != sorted(observations):
            raise PortfolioRiskInputError("INVALID_INPUT", "OBSERVATIONS_NOT_DETERMINISTICALLY_ORDERED")
    elif observation_ids:
        raise PortfolioRiskInputError("INVALID_INPUT", "OBSERVATIONS_WITHOUT_RETURN_HISTORY")
    volatilities = None if supplied_volatilities is None else _finite_vector(supplied_volatilities, n, "SUPPLIED_VOLATILITIES")
    covariance = None
    if supplied_covariance is not None:
        covariance = np.asarray(supplied_covariance, dtype=float)
        _validate_covariance(covariance, n)
    if history is None and volatilities is None and covariance is None:
        raise PortfolioRiskInputError("INVALID_INPUT", "RISK_INPUT_MISSING")
    population_checksum = canonical_hash({"contract": INPUT_CONTRACT, "asset_ids": assets})
    observation_checksum = canonical_hash({"observation_ids": observations}) if observations else None
    config = {
        "exposure_target": exposure_target, "minimum_weights": minimum, "maximum_weights": maximum,
        "sector_ids": sectors, "sector_caps": caps, "liquidity_eligible": eligibility,
        "cash_allowed": cash_allowed, "annualisation_factor": annualisation_factor,
        "missingness_policy": missingness_policy, "covariance_estimator_identity": covariance_estimator_identity,
        "policy_identity": policy_identity, "decision_timestamp": decision_timestamp,
    }
    return {
        "contract_version": INPUT_CONTRACT, "asset_ids": assets, "exposure_target": exposure_target,
        "return_history": history.tolist() if history is not None else None, "observation_ids": observations,
        "supplied_volatilities": volatilities, "supplied_covariance": covariance.tolist() if covariance is not None else None,
        "previous_weights": previous, "minimum_weights": minimum, "maximum_weights": maximum,
        "effective_minimum_weights": effective_minimum, "effective_maximum_weights": effective_maximum,
        "sector_ids": sectors, "sector_caps": caps, "liquidity_eligible": eligibility,
        "cash_allowed": cash_allowed, "annualisation_factor": annualisation_factor,
        "missingness_policy": missingness_policy, "covariance_estimator_identity": covariance_estimator_identity,
        "policy_identity": policy_identity, "decision_timestamp": decision_timestamp,
        "population_checksum": population_checksum, "observation_population_checksum": observation_checksum,
        "configuration_checksum": canonical_hash(config),
    }


def estimate_linear_shrinkage(contract: Mapping[str, Any]) -> dict[str, Any]:
    estimator_id = "sklearn_ledoit_wolf_scaled_identity"
    config = {"assume_centered": False, "target": "mu_times_identity", "intensity": "estimated"}
    try:
        data = _validated_contract(contract)
        if data["return_history"] is None:
            raise PortfolioRiskInputError("INVALID_INPUT", "RETURN_HISTORY_REQUIRED_FOR_SHRINKAGE")
        try:
            import sklearn
            from sklearn.covariance import LedoitWolf, empirical_covariance
        except ImportError:
            return _shrinkage_blocked(data, "SOLVER_UNAVAILABLE", "SKLEARN_UNAVAILABLE", config)
        history = np.asarray(data["return_history"])
        fitted = LedoitWolf(store_precision=False, assume_centered=False).fit(history)
        sample = empirical_covariance(history, assume_centered=False)
        mu = float(np.trace(sample) / sample.shape[0])
        target = np.eye(sample.shape[0]) * mu
        intensity = float(fitted.shrinkage_)
        covariance = np.asarray(fitted.covariance_)
        _validate_covariance(covariance, len(data["asset_ids"]))
        logical = {
            "contract_version": SHRINKAGE_RESULT, "estimator_id": estimator_id, "estimator_version": "sklearn_1",
            "status": "VALID", "valid": True, "blocking_reasons": [], "warnings": [],
            "asset_ids": data["asset_ids"], "sample_covariance": sample.tolist(),
            "shrinkage_target": target.tolist(), "shrinkage_intensity": intensity,
            "shrinkage_intensity_source": "estimated_by_sklearn_ledoit_wolf",
            "covariance": covariance.tolist(), "covariance_checksum": canonical_hash(covariance.tolist()),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
            "estimator_library_version": sklearn.__version__, "population_checksum": data["population_checksum"],
            "observation_population_checksum": data["observation_population_checksum"],
            "configuration_checksum": canonical_hash(config),
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except PortfolioRiskInputError as exc:
        return _shrinkage_blocked(contract, exc.status, exc.reason, config)


def allocate_inverse_volatility(contract: Mapping[str, Any], *, minimum_volatility: float = 1e-6) -> dict[str, Any]:
    policy_id = "capped_inverse_volatility_v1"
    config = {"minimum_volatility": minimum_volatility, "redistribution": "deterministic_proportional_waterfill_v1"}
    try:
        data = _validated_contract(contract)
        if not math.isfinite(minimum_volatility) or minimum_volatility <= 0:
            raise PortfolioRiskInputError("INVALID_INPUT", "MINIMUM_VOLATILITY_INVALID")
        covariance = _risk_covariance(data)
        if data["supplied_volatilities"] is not None:
            volatilities = np.asarray(data["supplied_volatilities"])
        else:
            volatilities = np.sqrt(np.maximum(np.diag(covariance) * data["annualisation_factor"], 0))
        if not np.isfinite(volatilities).all() or np.any(volatilities < 0):
            raise PortfolioRiskInputError("INVALID_INPUT", "VOLATILITY_INVALID")
        floored = np.maximum(volatilities, minimum_volatility)
        scores = np.where(np.asarray(data["liquidity_eligible"]), 1 / floored, 0)
        weights = _capped_allocate(data, scores)
        warnings = ["VOLATILITY_FLOOR_APPLIED"] if np.any(volatilities < minimum_volatility) else []
        return _allocation_result(data, policy_id, "1.0", "VALID", weights, covariance, volatilities, config, warnings=warnings, solver_identity="deterministic_proportional_waterfill")
    except PortfolioRiskInputError as exc:
        return _allocation_blocked(contract, INVERSE_VOL_RESULT, policy_id, exc.status, exc.reason, config)


def allocate_minimum_variance(
    contract: Mapping[str, Any],
    shrinkage_result: Mapping[str, Any],
    *,
    solver_tolerance: float = SOLVER_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, Any]:
    policy_id = "linear_shrinkage_minimum_variance_v1"
    config = {"solver_tolerance": solver_tolerance, "max_iterations": max_iterations, "covariance_source": "verified_linear_shrinkage"}
    try:
        data = _validated_contract(contract)
        verification = verify_shrinkage_result(data, shrinkage_result)
        if not verification["valid"]:
            raise PortfolioRiskInputError("INVALID_INPUT", "SHRINKAGE_RESULT_VERIFICATION_FAILED")
        if solver_tolerance <= 0 or max_iterations < 1:
            raise PortfolioRiskInputError("UNSUPPORTED_CONFIGURATION", "SOLVER_CONFIGURATION_INVALID")
        try:
            import scipy
            from scipy.optimize import minimize
        except ImportError:
            return _allocation_blocked(data, MINIMUM_VARIANCE_RESULT, policy_id, "SOLVER_UNAVAILABLE", "SCIPY_UNAVAILABLE", config)
        covariance = np.asarray(shrinkage_result["covariance"])
        n = len(data["asset_ids"])
        constraints = [{"type": "eq", "fun": lambda weights: float(weights.sum() - data["exposure_target"])}]
        for sector, cap in data["sector_caps"].items():
            indexes = np.asarray([index for index, value in enumerate(data["sector_ids"]) if value == sector], dtype=int)
            constraints.append({"type": "ineq", "fun": lambda weights, indexes=indexes, cap=cap: float(cap - weights[indexes].sum())})
        initial = _capped_allocate(data, np.ones(n))
        solved = minimize(
            lambda weights: float(weights @ covariance @ weights), initial,
            method="SLSQP", bounds=list(zip(data["effective_minimum_weights"], data["effective_maximum_weights"])),
            constraints=constraints, options={"ftol": solver_tolerance, "maxiter": max_iterations, "disp": False},
        )
        if not solved.success:
            status = "INFEASIBLE" if int(solved.status) in {4, 8} else "NUMERICAL_FAILURE"
            return _allocation_blocked(data, MINIMUM_VARIANCE_RESULT, policy_id, status, f"SCIPY_SLSQP:{solved.status}:{solved.message}", config, solver_version=scipy.__version__)
        weights = np.asarray(solved.x)
        weights[np.abs(weights) < CONSTRAINT_TOLERANCE] = 0
        volatilities = np.sqrt(np.maximum(np.diag(covariance) * data["annualisation_factor"], 0))
        result = _allocation_result(data, policy_id, "1.0", "OPTIMAL", weights, covariance, volatilities, config, solver_identity="scipy.optimize.SLSQP", solver_version=scipy.__version__, iteration_count=int(solved.nit))
        if not verify_allocation(data, result)["valid"]:
            return _allocation_blocked(data, MINIMUM_VARIANCE_RESULT, policy_id, "NUMERICAL_FAILURE", "POST_SOLVE_VERIFICATION_FAILED", config, solver_version=scipy.__version__)
        return result
    except PortfolioRiskInputError as exc:
        return _allocation_blocked(contract, MINIMUM_VARIANCE_RESULT, policy_id, exc.status, exc.reason, config)


def verify_shrinkage_result(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        covariance = np.asarray(result.get("covariance", ()), dtype=float)
        if result.get("population_checksum") != data["population_checksum"] or result.get("asset_ids") != data["asset_ids"]:
            reasons.append("POPULATION_MISMATCH")
        try: _validate_covariance(covariance, len(data["asset_ids"]))
        except PortfolioRiskInputError as exc: reasons.append(exc.reason)
        if covariance.shape == (len(data["asset_ids"]), len(data["asset_ids"])) and result.get("covariance_checksum") != canonical_hash(covariance.tolist()):
            reasons.append("COVARIANCE_CHECKSUM_MISMATCH")
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical):
            reasons.append("RESULT_CHECKSUM_MISMATCH")
        return {"contract_version": "linear_shrinkage_verification_v1", "valid": not reasons, "blocking_reasons": sorted(set(reasons))}
    except PortfolioRiskInputError as exc:
        return {"contract_version": "linear_shrinkage_verification_v1", "valid": False, "blocking_reasons": [exc.reason]}


def verify_allocation(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        weights = np.asarray(result.get("target_weights", ()), dtype=float)
        n = len(data["asset_ids"])
        if result.get("population_checksum") != data["population_checksum"] or result.get("asset_ids") != data["asset_ids"]:
            reasons.append("POPULATION_MISMATCH")
        if weights.shape != (n,) or not np.isfinite(weights).all():
            return {"contract_version": "portfolio_risk_allocation_verification_v1", "valid": False, "blocking_reasons": ["TARGET_WEIGHTS_INVALID"]}
        residuals = _constraint_residuals(data, weights)
        if abs(residuals["exposure"]) > CONSTRAINT_TOLERANCE: reasons.append("EXPOSURE_FAILED")
        if residuals["minimum_weight"] < -CONSTRAINT_TOLERANCE: reasons.append("MINIMUM_WEIGHT_FAILED")
        if residuals["maximum_weight"] < -CONSTRAINT_TOLERANCE: reasons.append("MAXIMUM_WEIGHT_FAILED")
        if residuals["sector_cap"] is not None and residuals["sector_cap"] < -CONSTRAINT_TOLERANCE: reasons.append("SECTOR_CAP_FAILED")
        covariance = np.asarray(result.get("covariance", ()), dtype=float)
        try: _validate_covariance(covariance, n)
        except PortfolioRiskInputError as exc: reasons.append(exc.reason)
        if covariance.shape == (n, n):
            variance = float(weights @ covariance @ weights)
            if abs(variance - float(result.get("expected_variance", math.inf))) > 1e-7: reasons.append("VARIANCE_RECOMPUTATION_MISMATCH")
            if abs(math.sqrt(max(variance * data["annualisation_factor"], 0)) - float(result.get("expected_volatility", math.inf))) > 1e-7: reasons.append("VOLATILITY_RECOMPUTATION_MISMATCH")
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical): reasons.append("RESULT_CHECKSUM_MISMATCH")
        return {"contract_version": "portfolio_risk_allocation_verification_v1", "valid": not reasons, "blocking_reasons": sorted(set(reasons)), "constraint_residuals": residuals}
    except PortfolioRiskInputError as exc:
        return {"contract_version": "portfolio_risk_allocation_verification_v1", "valid": False, "blocking_reasons": [exc.reason]}


def compare_risk_policies(contract: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    data = _validated_contract(contract)
    rows = {}
    for name, result in sorted(results.items()):
        if result.get("population_checksum") != data["population_checksum"]:
            raise PortfolioRiskInputError("UNMATCHED_POPULATION", f"COMPARISON_POPULATION_MISMATCH:{name}")
        weights = np.asarray(result.get("target_weights", ()), dtype=float)
        if weights.shape != (len(data["asset_ids"]),):
            raise PortfolioRiskInputError("INVALID_INPUT", f"COMPARISON_WEIGHTS_INVALID:{name}")
        covariance = np.asarray(result.get("covariance") or _risk_covariance(data))
        variance = float(weights @ covariance @ weights)
        sectors = _sector_exposures(data, weights)
        rows[name] = {
            "expected_variance": variance, "expected_volatility": math.sqrt(max(variance * data["annualisation_factor"], 0)),
            "concentration_hhi": float(weights @ weights), "maximum_stock_weight": float(weights.max()),
            "sector_concentration_hhi": float(sum(value**2 for value in sectors.values())),
            "gross_turnover": float(np.abs(weights - np.asarray(data["previous_weights"])).sum()),
        }
    logical = {
        "contract_version": COMPARISON_CONTRACT, "status": "VALID", "valid": True,
        "population_checksum": data["population_checksum"], "policy_count": len(rows),
        "historical_returns_computed": False, "sharpe_computed": False, "superiority_claimed": False,
        "policies": rows,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _allocation_result(data, policy_id, version, status, weights, covariance, volatilities, config, *, warnings=(), solver_identity, solver_version=None, iteration_count=None):
    variance = float(weights @ covariance @ weights)
    sectors = _sector_exposures(data, weights)
    marginal = covariance @ weights
    contributions = weights * marginal
    logical = {
        "contract_version": MINIMUM_VARIANCE_RESULT if status == "OPTIMAL" else INVERSE_VOL_RESULT,
        "policy_id": policy_id, "policy_version": version, "estimator_id": data["covariance_estimator_identity"],
        "estimator_version": "1.0", "status": status, "valid": True, "blocking_reasons": [], "warnings": list(warnings),
        "asset_ids": data["asset_ids"], "target_weights": weights.tolist(), "exposure": float(weights.sum()),
        "expected_variance": variance, "expected_volatility": math.sqrt(max(variance * data["annualisation_factor"], 0)),
        "individual_volatility_estimates": volatilities.tolist(), "covariance": covariance.tolist(),
        "covariance_checksum": canonical_hash(covariance.tolist()),
        "risk_contributions": contributions.tolist(), "risk_contribution_sum": float(contributions.sum()),
        "stock_cap_utilisation": [weights[index] / cap if cap > 0 else None for index, cap in enumerate(data["effective_maximum_weights"])],
        "sector_exposures": sectors,
        "sector_cap_utilisation": {sector: value / data["sector_caps"][sector] for sector, value in sectors.items() if sector in data["sector_caps"] and data["sector_caps"][sector] > 0},
        "liquidity_exclusions": [asset for asset, eligible in zip(data["asset_ids"], data["liquidity_eligible"]) if not eligible],
        "concentration": {"hhi": float(weights @ weights), "maximum_stock_weight": float(weights.max())},
        "constraint_residuals": _constraint_residuals(data, weights),
        "solver_identity": solver_identity, "solver_version": solver_version, "solver_status": "success",
        "solver_tolerance": config.get("solver_tolerance"), "maximum_iterations": config.get("max_iterations"),
        "iteration_count": iteration_count, "population_checksum": data["population_checksum"],
        "configuration_checksum": canonical_hash(config), "adv_capacity_status": "UNVERIFIED",
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _allocation_blocked(data, contract_version, policy_id, status, reason, config, *, solver_version=None):
    logical = {
        "contract_version": contract_version, "policy_id": policy_id, "policy_version": "1.0",
        "estimator_id": data.get("covariance_estimator_identity") if isinstance(data, Mapping) else None,
        "estimator_version": "1.0", "status": status if status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [reason], "warnings": [],
        "asset_ids": list(data.get("asset_ids", ())) if isinstance(data, Mapping) else [], "target_weights": [],
        "exposure": None, "expected_variance": None, "expected_volatility": None,
        "individual_volatility_estimates": [], "covariance": [], "covariance_checksum": None,
        "risk_contributions": [], "risk_contribution_sum": None, "stock_cap_utilisation": [],
        "sector_exposures": {}, "sector_cap_utilisation": {}, "liquidity_exclusions": [],
        "concentration": {}, "constraint_residuals": {}, "solver_identity": "scipy.optimize.SLSQP",
        "solver_version": solver_version, "solver_status": "not_solved",
        "solver_tolerance": config.get("solver_tolerance"), "maximum_iterations": config.get("max_iterations"),
        "iteration_count": None, "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": canonical_hash(config), "adv_capacity_status": "UNVERIFIED",
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _shrinkage_blocked(data, status, reason, config):
    logical = {
        "contract_version": SHRINKAGE_RESULT, "estimator_id": "sklearn_ledoit_wolf_scaled_identity",
        "estimator_version": "sklearn_1", "status": status if status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": [reason], "warnings": [],
        "asset_ids": list(data.get("asset_ids", ())) if isinstance(data, Mapping) else [],
        "sample_covariance": [], "shrinkage_target": [], "shrinkage_intensity": None,
        "shrinkage_intensity_source": "estimated", "covariance": [], "covariance_checksum": None,
        "minimum_eigenvalue": None, "estimator_library_version": None,
        "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "observation_population_checksum": data.get("observation_population_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": canonical_hash(config),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _capped_allocate(data, scores):
    weights = np.asarray(data["effective_minimum_weights"], dtype=float)
    maximum = np.asarray(data["effective_maximum_weights"], dtype=float)
    remaining = float(data["exposure_target"] - weights.sum())
    scores = np.asarray(scores, dtype=float)
    for _ in range(len(weights) * 4 + 4):
        if remaining <= CONSTRAINT_TOLERANCE: break
        sector_exposures = _sector_exposures(data, weights)
        room = np.maximum(maximum - weights, 0)
        for index, sector in enumerate(data["sector_ids"]):
            if sector in data["sector_caps"]:
                room[index] = min(room[index], max(data["sector_caps"][sector] - sector_exposures[sector], 0))
        active = (room > CONSTRAINT_TOLERANCE) & (scores > 0)
        if not np.any(active):
            raise PortfolioRiskInputError("INFEASIBLE", "CAP_REDISTRIBUTION_EXHAUSTED")
        allocation = scores * active
        allocation = allocation / allocation.sum() * remaining
        addition = np.minimum(allocation, room)
        for sector, cap in data["sector_caps"].items():
            indexes = np.asarray([index for index, value in enumerate(data["sector_ids"]) if value == sector], dtype=int)
            sector_room = max(cap - sector_exposures[sector], 0)
            proposed = float(addition[indexes].sum())
            if proposed > sector_room and proposed > 0:
                addition[indexes] *= sector_room / proposed
        if addition.sum() <= CONSTRAINT_TOLERANCE:
            raise PortfolioRiskInputError("INFEASIBLE", "CAP_REDISTRIBUTION_STALLED")
        weights += addition; remaining -= float(addition.sum())
    if remaining > CONSTRAINT_TOLERANCE:
        raise PortfolioRiskInputError("INFEASIBLE", "EXPOSURE_NOT_ALLOCATED")
    return weights


def _risk_covariance(data):
    if data["supplied_covariance"] is not None:
        return np.asarray(data["supplied_covariance"])
    if data["return_history"] is not None:
        return np.cov(np.asarray(data["return_history"]), rowvar=False, ddof=1)
    vol = np.asarray(data["supplied_volatilities"])
    return np.diag((vol**2) / data["annualisation_factor"])


def _validated_contract(contract):
    return portfolio_risk_input(
        contract["asset_ids"], exposure_target=float(contract.get("exposure_target", 1)),
        return_history=contract.get("return_history"), observation_ids=contract.get("observation_ids") or None,
        supplied_volatilities=contract.get("supplied_volatilities"), supplied_covariance=contract.get("supplied_covariance"),
        previous_weights=contract.get("previous_weights"), minimum_weights=contract.get("minimum_weights", 0),
        maximum_weights=contract.get("maximum_weights", 1), sector_ids=contract.get("sector_ids"),
        sector_caps=contract.get("sector_caps"), liquidity_eligible=contract.get("liquidity_eligible"),
        cash_allowed=bool(contract.get("cash_allowed", True)), annualisation_factor=float(contract.get("annualisation_factor", 252)),
        missingness_policy=str(contract.get("missingness_policy", "reject_any_missing")),
        covariance_estimator_identity=str(contract.get("covariance_estimator_identity", "ledoit_wolf_scaled_identity_v1")),
        policy_identity=str(contract.get("policy_identity", "portfolio_risk_baseline_v1")),
        decision_timestamp=str(contract.get("decision_timestamp", "synthetic")),
    )


def _validate_covariance(covariance, n):
    if covariance.shape != (n, n): raise PortfolioRiskInputError("INVALID_INPUT", "COVARIANCE_DIMENSION_MISMATCH")
    if not np.isfinite(covariance).all(): raise PortfolioRiskInputError("INVALID_INPUT", "COVARIANCE_NON_FINITE")
    if not np.allclose(covariance, covariance.T, atol=PSD_TOLERANCE, rtol=0): raise PortfolioRiskInputError("INVALID_INPUT", "COVARIANCE_NOT_SYMMETRIC")
    if float(np.linalg.eigvalsh((covariance + covariance.T) / 2).min()) < -PSD_TOLERANCE: raise PortfolioRiskInputError("INVALID_INPUT", "COVARIANCE_NOT_PSD")


def _constraint_residuals(data, weights):
    sectors = _sector_exposures(data, weights)
    sector_values = [data["sector_caps"][sector] - value for sector, value in sectors.items() if sector in data["sector_caps"]]
    return {
        "exposure": float(weights.sum() - data["exposure_target"]),
        "minimum_weight": float(np.min(weights - np.asarray(data["effective_minimum_weights"]))),
        "maximum_weight": float(np.min(np.asarray(data["effective_maximum_weights"]) - weights)),
        "sector_cap": float(min(sector_values)) if sector_values else None,
        "liquidity": float(min((0 - weights[index] for index, eligible in enumerate(data["liquidity_eligible"]) if not eligible), default=0)),
    }


def _sector_exposures(data, weights):
    return {sector: float(sum(weights[index] for index, value in enumerate(data["sector_ids"]) if value == sector)) for sector in sorted(set(data["sector_ids"]))}


def _finite_vector(values, n, owner):
    result = [float(value) for value in values]
    if len(result) != n: raise PortfolioRiskInputError("INVALID_INPUT", f"{owner}_DIMENSION_MISMATCH")
    if not all(math.isfinite(value) for value in result): raise PortfolioRiskInputError("INVALID_INPUT", f"{owner}_NON_FINITE")
    return result


def _bounds(values, n, owner):
    result = [float(values)] * n if isinstance(values, (int, float)) else [float(value) for value in values]
    if len(result) != n or not all(math.isfinite(value) for value in result): raise PortfolioRiskInputError("INVALID_INPUT", f"{owner}_INVALID")
    return result


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
