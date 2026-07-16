from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "aim_portfolio_input_v1"
RESULT_CONTRACT = "aim_portfolio_result_v1"
POLICY_ID = "turnover_penalised_aim_portfolio_v1"
PSD_TOLERANCE = 1e-8
CONSTRAINT_TOLERANCE = 1e-7
SOLVER_TOLERANCE = 1e-10
MAX_ITERATIONS = 2000
STATUSES = {
    "OPTIMAL", "INFEASIBLE", "INVALID_INPUT", "UNSUPPORTED_CONFIGURATION",
    "NUMERICAL_FAILURE", "SOLVER_UNAVAILABLE",
}


class AimPortfolioInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def aim_portfolio_input(
    asset_ids: Sequence[str],
    expected_alpha: Sequence[float],
    covariance: Sequence[Sequence[float]],
    previous_weights: Sequence[float],
    *,
    exposure_target: float = 1.0,
    maximum_weights: float | Sequence[float] = 1.0,
    minimum_weights: float | Sequence[float] | None = None,
    long_only: bool = True,
    sector_ids: Sequence[str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    liquidity_eligible: Sequence[bool] | None = None,
    turnover_limit: float | None = None,
    l1_turnover_penalty: float = 0.0,
    l2_turnover_penalty: float = 0.0,
    risk_aversion: float = 0.0,
    cash_allowed: bool = True,
    ineligible_asset_policy: str = "liquidate",
    alpha_identity: str = "synthetic-alpha",
    covariance_identity: str = "synthetic-covariance",
    policy_identity: str = POLICY_ID,
    decision_timestamp: str = "synthetic",
) -> dict[str, Any]:
    assets = [str(value) for value in asset_ids]
    n = len(assets)
    if n == 0:
        raise AimPortfolioInputError("INVALID_INPUT", "ASSET_POPULATION_EMPTY")
    if len(set(assets)) != n:
        raise AimPortfolioInputError("INVALID_INPUT", "ASSET_IDENTITIES_NOT_UNIQUE")
    if assets != sorted(assets):
        raise AimPortfolioInputError("INVALID_INPUT", "ASSETS_NOT_DETERMINISTICALLY_ORDERED")
    alpha = _finite_vector(expected_alpha, n, "EXPECTED_ALPHA")
    previous = _finite_vector(previous_weights, n, "PREVIOUS_WEIGHTS")
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (n, n):
        raise AimPortfolioInputError("INVALID_INPUT", "COVARIANCE_DIMENSION_MISMATCH")
    if not np.isfinite(matrix).all():
        raise AimPortfolioInputError("INVALID_INPUT", "COVARIANCE_NON_FINITE")
    if not np.allclose(matrix, matrix.T, atol=PSD_TOLERANCE, rtol=0):
        raise AimPortfolioInputError("INVALID_INPUT", "COVARIANCE_NOT_SYMMETRIC")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    if float(eigenvalues.min()) < -PSD_TOLERANCE:
        raise AimPortfolioInputError("INVALID_INPUT", "COVARIANCE_NOT_POSITIVE_SEMIDEFINITE")
    if not long_only:
        raise AimPortfolioInputError("UNSUPPORTED_CONFIGURATION", "LONG_SHORT_NOT_SUPPORTED")
    maximum = _bounds(maximum_weights, n, "MAXIMUM_WEIGHTS")
    minimum = _bounds(0.0 if minimum_weights is None else minimum_weights, n, "MINIMUM_WEIGHTS")
    if any(value < 0 for value in minimum + maximum) or any(low > high for low, high in zip(minimum, maximum)):
        raise AimPortfolioInputError("INVALID_INPUT", "STOCK_BOUNDS_INVALID")
    if not math.isfinite(exposure_target) or exposure_target < 0 or (cash_allowed and exposure_target > 1 + CONSTRAINT_TOLERANCE):
        raise AimPortfolioInputError("INVALID_INPUT", "EXPOSURE_TARGET_INVALID")
    if not cash_allowed and abs(exposure_target - 1.0) > CONSTRAINT_TOLERANCE:
        raise AimPortfolioInputError("INVALID_INPUT", "CASH_DISALLOWED_REQUIRES_FULL_EXPOSURE")
    if any(value < -CONSTRAINT_TOLERANCE for value in previous):
        raise AimPortfolioInputError("INVALID_INPUT", "PREVIOUS_WEIGHTS_NEGATIVE")
    sectors = [str(value) for value in (sector_ids or ["UNCLASSIFIED"] * n)]
    if len(sectors) != n or any(not value for value in sectors):
        raise AimPortfolioInputError("INVALID_INPUT", "SECTOR_MAPPING_INVALID")
    caps = {str(key): float(value) for key, value in sorted((sector_caps or {}).items())}
    if any(not math.isfinite(value) or value < 0 for value in caps.values()):
        raise AimPortfolioInputError("INVALID_INPUT", "SECTOR_CAP_INVALID")
    eligibility = list(liquidity_eligible if liquidity_eligible is not None else [True] * n)
    if len(eligibility) != n or any(not isinstance(value, (bool, np.bool_)) for value in eligibility):
        raise AimPortfolioInputError("INVALID_INPUT", "LIQUIDITY_MASK_INVALID")
    if ineligible_asset_policy not in {"liquidate", "retain_only"}:
        raise AimPortfolioInputError("UNSUPPORTED_CONFIGURATION", "INELIGIBLE_ASSET_POLICY_UNSUPPORTED")
    penalties = (l1_turnover_penalty, l2_turnover_penalty, risk_aversion)
    if any(not math.isfinite(value) or value < 0 for value in penalties):
        raise AimPortfolioInputError("INVALID_INPUT", "OBJECTIVE_COEFFICIENT_INVALID")
    if turnover_limit is not None and (not math.isfinite(turnover_limit) or turnover_limit < 0):
        raise AimPortfolioInputError("INVALID_INPUT", "TURNOVER_LIMIT_INVALID")
    effective_maximum = list(maximum)
    effective_minimum = list(minimum)
    for index, eligible in enumerate(eligibility):
        if not eligible:
            effective_minimum[index] = 0.0
            effective_maximum[index] = 0.0 if ineligible_asset_policy == "liquidate" else min(effective_maximum[index], previous[index])
    if sum(effective_minimum) > exposure_target + CONSTRAINT_TOLERANCE or sum(effective_maximum) < exposure_target - CONSTRAINT_TOLERANCE:
        raise AimPortfolioInputError("INFEASIBLE", "STOCK_CAPS_CANNOT_MEET_EXPOSURE")
    for sector in sorted(set(sectors)):
        cap = caps.get(sector)
        if cap is not None:
            sector_minimum = sum(effective_minimum[index] for index, value in enumerate(sectors) if value == sector)
            if sector_minimum > cap + CONSTRAINT_TOLERANCE:
                raise AimPortfolioInputError("INFEASIBLE", f"SECTOR_MINIMUM_EXCEEDS_CAP:{sector}")
    total_sector_capacity = sum(min(caps.get(sector, exposure_target), sum(effective_maximum[index] for index, value in enumerate(sectors) if value == sector)) for sector in sorted(set(sectors)))
    if total_sector_capacity < exposure_target - CONSTRAINT_TOLERANCE:
        raise AimPortfolioInputError("INFEASIBLE", "SECTOR_CAPS_CANNOT_MEET_EXPOSURE")
    population_checksum = canonical_hash({"contract": INPUT_CONTRACT, "asset_ids": assets})
    config = {
        "exposure_target": exposure_target, "maximum_weights": maximum, "minimum_weights": minimum,
        "long_only": long_only, "sector_ids": sectors, "sector_caps": caps,
        "liquidity_eligible": eligibility, "turnover_limit": turnover_limit,
        "l1_turnover_penalty": l1_turnover_penalty, "l2_turnover_penalty": l2_turnover_penalty,
        "risk_aversion": risk_aversion, "cash_allowed": cash_allowed,
        "ineligible_asset_policy": ineligible_asset_policy, "alpha_identity": alpha_identity,
        "covariance_identity": covariance_identity, "policy_identity": policy_identity,
        "decision_timestamp": decision_timestamp,
    }
    return {
        "contract_version": INPUT_CONTRACT, "asset_ids": assets, "expected_alpha": alpha,
        "covariance": matrix.tolist(), "previous_weights": previous, **config,
        "effective_minimum_weights": effective_minimum, "effective_maximum_weights": effective_maximum,
        "asset_population_checksum": population_checksum, "configuration_checksum": canonical_hash(config),
        "covariance_near_psd_adjustment_required": bool(eigenvalues.min() < 0),
    }


def optimise_aim_portfolio(contract: Mapping[str, Any], *, solver_tolerance: float = SOLVER_TOLERANCE, max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    try:
        data = _validated_contract(contract)
        if solver_tolerance <= 0 or max_iterations < 1:
            raise AimPortfolioInputError("UNSUPPORTED_CONFIGURATION", "SOLVER_CONFIGURATION_INVALID")
        try:
            import scipy
            from scipy.optimize import minimize
        except ImportError:
            return _blocked(data, "SOLVER_UNAVAILABLE", "SCIPY_UNAVAILABLE", solver_tolerance, max_iterations)
        n = len(data["asset_ids"])
        alpha = np.asarray(data["expected_alpha"])
        covariance = _solver_covariance(data)
        previous = np.asarray(data["previous_weights"])
        k1 = float(data["l1_turnover_penalty"])
        k2 = float(data["l2_turnover_penalty"])
        risk_aversion = float(data["risk_aversion"])
        turnover_limit = data["turnover_limit"]

        def objective(vector):
            weights = vector[:n]; auxiliaries = vector[n:]
            change = weights - previous
            return float(-alpha @ weights + 0.5 * risk_aversion * weights @ covariance @ weights + k1 * auxiliaries.sum() + k2 * change @ change)

        constraints = [{"type": "eq", "fun": lambda vector: float(np.sum(vector[:n]) - data["exposure_target"])}]
        constraints.extend([
            {"type": "ineq", "fun": lambda vector, index=index: float(vector[n + index] - (vector[index] - previous[index]))}
            for index in range(n)
        ])
        constraints.extend([
            {"type": "ineq", "fun": lambda vector, index=index: float(vector[n + index] + (vector[index] - previous[index]))}
            for index in range(n)
        ])
        if turnover_limit is not None:
            constraints.append({"type": "ineq", "fun": lambda vector: float(turnover_limit - np.sum(vector[n:]))})
        sectors = data["sector_ids"]
        for sector, cap in data["sector_caps"].items():
            indexes = np.asarray([index for index, value in enumerate(sectors) if value == sector], dtype=int)
            constraints.append({"type": "ineq", "fun": lambda vector, indexes=indexes, cap=cap: float(cap - np.sum(vector[indexes]))})
        initial_weights = _initial_feasible_weights(data)
        initial = np.r_[initial_weights, np.abs(initial_weights - previous)]
        max_auxiliary = max(2.0, float(np.abs(previous).sum() + data["exposure_target"]))
        bounds = list(zip(data["effective_minimum_weights"], data["effective_maximum_weights"])) + [(0.0, max_auxiliary)] * n
        solved = minimize(
            objective, initial, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": solver_tolerance, "maxiter": max_iterations, "disp": False},
        )
        if not solved.success:
            status = "INFEASIBLE" if int(solved.status) in {4, 8} else "NUMERICAL_FAILURE"
            return _blocked(data, status, f"SCIPY_SLSQP:{solved.status}:{solved.message}", solver_tolerance, max_iterations, solver_version=scipy.__version__)
        weights = np.asarray(solved.x[:n])
        weights[np.abs(weights) < CONSTRAINT_TOLERANCE] = 0.0
        result = _optimal_result(data, weights, solver_tolerance, max_iterations, scipy.__version__, int(getattr(solved, "nit", 0)))
        verification = verify_aim_portfolio(data, result)
        if not verification["valid"]:
            return _blocked(data, "NUMERICAL_FAILURE", "POST_SOLVE_CONSTRAINT_VERIFICATION_FAILED", solver_tolerance, max_iterations, solver_version=scipy.__version__, extra=verification["blocking_reasons"])
        return result
    except AimPortfolioInputError as exc:
        return _blocked(contract, exc.status, exc.reason, solver_tolerance, max_iterations)
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        return _blocked(contract, "NUMERICAL_FAILURE", type(exc).__name__, solver_tolerance, max_iterations)


def verify_aim_portfolio(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        weights = np.asarray(result.get("target_weights", ()), dtype=float)
        n = len(data["asset_ids"])
        if list(result.get("asset_ids", ())) != data["asset_ids"] or result.get("population_checksum") != data["asset_population_checksum"]:
            reasons.append("ASSET_POPULATION_MISMATCH")
        if weights.shape != (n,) or not np.isfinite(weights).all():
            reasons.append("TARGET_WEIGHTS_INVALID")
            return {"contract_version": "aim_portfolio_verification_v1", "valid": False, "blocking_reasons": reasons, "constraint_residuals": {}}
        residuals = _constraint_residuals(data, weights)
        if abs(residuals["exposure"]) > CONSTRAINT_TOLERANCE: reasons.append("EXPOSURE_CONSTRAINT_FAILED")
        if residuals["minimum_weight"] < -CONSTRAINT_TOLERANCE: reasons.append("MINIMUM_WEIGHT_FAILED")
        if residuals["maximum_weight"] < -CONSTRAINT_TOLERANCE: reasons.append("MAXIMUM_WEIGHT_FAILED")
        if residuals["sector_cap"] is not None and residuals["sector_cap"] < -CONSTRAINT_TOLERANCE: reasons.append("SECTOR_CAP_FAILED")
        if residuals["turnover_limit"] is not None and residuals["turnover_limit"] < -CONSTRAINT_TOLERANCE: reasons.append("TURNOVER_LIMIT_FAILED")
        if residuals["liquidity"] is not None and residuals["liquidity"] < -CONSTRAINT_TOLERANCE: reasons.append("LIQUIDITY_CONSTRAINT_FAILED")
        components = objective_components(data, weights)
        reported = result.get("objective_components", {})
        if any(abs(float(reported.get(key, math.inf)) - value) > 1e-6 for key, value in components.items()):
            reasons.append("OBJECTIVE_RECOMPUTATION_MISMATCH")
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical):
            reasons.append("RESULT_CHECKSUM_MISMATCH")
        return {
            "contract_version": "aim_portfolio_verification_v1", "valid": not reasons,
            "blocking_reasons": sorted(set(reasons)), "constraint_residuals": residuals,
            "recomputed_objective_components": components,
        }
    except (AimPortfolioInputError, TypeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, AimPortfolioInputError) else type(exc).__name__
        return {"contract_version": "aim_portfolio_verification_v1", "valid": False, "blocking_reasons": [reason], "constraint_residuals": {}}


def objective_components(contract: Mapping[str, Any], weights: Sequence[float]) -> dict[str, float]:
    data = _validated_contract(contract)
    vector = np.asarray(weights, dtype=float)
    alpha = float(np.asarray(data["expected_alpha"]) @ vector)
    variance = float(vector @ np.asarray(data["covariance"]) @ vector)
    change = vector - np.asarray(data["previous_weights"])
    risk = 0.5 * float(data["risk_aversion"]) * variance
    l1 = float(data["l1_turnover_penalty"]) * float(np.abs(change).sum())
    l2 = float(data["l2_turnover_penalty"]) * float(change @ change)
    return {
        "expected_alpha_contribution": alpha, "covariance_risk_penalty": risk,
        "l1_turnover_penalty": l1, "l2_turnover_penalty": l2,
        "gross_objective_value": alpha - risk - l1 - l2,
    }


def comparison_controls(contract: Mapping[str, Any], *, top_k: int = 3) -> dict[str, Any]:
    data = _validated_contract(contract)
    n = len(data["asset_ids"])
    if top_k < 1:
        raise AimPortfolioInputError("INVALID_INPUT", "TOP_K_INVALID")
    eligible = [index for index, value in enumerate(data["liquidity_eligible"]) if value]
    ranked = sorted(eligible, key=lambda index: (-data["expected_alpha"][index], data["asset_ids"][index]))[:top_k]
    equal = np.zeros(n)
    if ranked:
        equal[ranked] = data["exposure_target"] / len(ranked)
    variances = np.diag(np.asarray(data["covariance"]))
    inverse = np.zeros(n)
    if ranked:
        raw = np.asarray([1 / math.sqrt(max(variances[index], PSD_TOLERANCE)) for index in ranked])
        inverse[ranked] = raw / raw.sum() * data["exposure_target"]
    unchanged = np.asarray(data["previous_weights"], dtype=float)
    if unchanged.sum() > 0:
        unchanged = unchanged * (data["exposure_target"] / unchanged.sum())
    controls = {}
    for name, weights in (("equal_weight_top_k", equal), ("inverse_volatility_top_k", inverse), ("unchanged_previous_holdings", unchanged)):
        controls[name] = _control_metrics(data, weights)
    return {
        "contract_version": "aim_portfolio_comparison_controls_v1",
        "historical_performance_computed": False, "superiority_claimed": False,
        "controls": controls,
    }


def _optimal_result(data, weights, tolerance, max_iterations, solver_version, iterations):
    previous = np.asarray(data["previous_weights"])
    change = weights - previous
    components = objective_components(data, weights)
    variance = float(weights @ np.asarray(data["covariance"]) @ weights)
    sectors = _sector_exposures(data, weights)
    residuals = _constraint_residuals(data, weights)
    warnings = []
    if data["covariance_near_psd_adjustment_required"]: warnings.append("NEAR_PSD_COVARIANCE_PROJECTED_FOR_SOLVER")
    if _economically_non_unique(data): warnings.append("NON_UNIQUE_OPTIMUM_POSSIBLE_STABLE_ASSET_ORDER_USED")
    logical = {
        "contract_version": RESULT_CONTRACT, "policy_id": data["policy_identity"], "policy_version": "1.0",
        "status": "OPTIMAL", "valid": True, "blocking_reasons": [], "warnings": warnings,
        "asset_ids": data["asset_ids"], "target_weights": weights.tolist(),
        "previous_weights": data["previous_weights"], "trade_weight_changes": change.tolist(),
        "gross_turnover": float(np.abs(change).sum()), "one_way_turnover": float(np.abs(change).sum() / 2),
        "turnover_convention": "gross=sum(abs(target-previous)); one_way=gross/2",
        "expected_alpha": components["expected_alpha_contribution"], "expected_variance": variance,
        "expected_volatility": math.sqrt(max(variance, 0)), "objective_components": components,
        "stock_cap_utilisation": [weights[index] / cap if cap > 0 else None for index, cap in enumerate(data["effective_maximum_weights"])],
        "sector_exposures": sectors,
        "sector_cap_utilisation": {sector: exposure / data["sector_caps"][sector] for sector, exposure in sectors.items() if sector in data["sector_caps"] and data["sector_caps"][sector] > 0},
        "liquidity_exclusions": [asset for asset, eligible in zip(data["asset_ids"], data["liquidity_eligible"]) if not eligible],
        "constraint_residuals": residuals, "solver_identity": "scipy.optimize.SLSQP",
        "solver_version": solver_version, "solver_status": "success", "solver_tolerance": tolerance,
        "maximum_iterations": max_iterations, "iteration_count": iterations,
        "population_checksum": data["asset_population_checksum"], "configuration_checksum": data["configuration_checksum"],
        "execution_model": False, "transaction_cost_model": False, "orders_generated": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked(data, status, reason, tolerance, max_iterations, *, solver_version=None, extra=()):
    logical = {
        "contract_version": RESULT_CONTRACT, "policy_id": data.get("policy_identity", POLICY_ID) if isinstance(data, Mapping) else POLICY_ID,
        "policy_version": "1.0", "status": status if status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": sorted(set([reason, *extra])), "warnings": [],
        "asset_ids": list(data.get("asset_ids", ())) if isinstance(data, Mapping) else [],
        "target_weights": [], "previous_weights": list(data.get("previous_weights", ())) if isinstance(data, Mapping) else [],
        "trade_weight_changes": [], "gross_turnover": None, "one_way_turnover": None,
        "turnover_convention": "gross=sum(abs(target-previous)); one_way=gross/2",
        "expected_alpha": None, "expected_variance": None, "expected_volatility": None,
        "objective_components": {}, "stock_cap_utilisation": [], "sector_exposures": {},
        "sector_cap_utilisation": {}, "liquidity_exclusions": [], "constraint_residuals": {},
        "solver_identity": "scipy.optimize.SLSQP", "solver_version": solver_version,
        "solver_status": "not_solved", "solver_tolerance": tolerance, "maximum_iterations": max_iterations,
        "iteration_count": None, "population_checksum": data.get("asset_population_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": data.get("configuration_checksum") if isinstance(data, Mapping) else None,
        "execution_model": False, "transaction_cost_model": False, "orders_generated": False,
    }
    logical["logical_result_checksum"] = canonical_hash(_identity_safe(logical))
    return {**logical, "creation_metadata": _creation_metadata()}


def _validated_contract(contract):
    return aim_portfolio_input(
        contract["asset_ids"], contract["expected_alpha"], contract["covariance"], contract["previous_weights"],
        exposure_target=float(contract.get("exposure_target", 1)), maximum_weights=contract.get("maximum_weights", 1),
        minimum_weights=contract.get("minimum_weights"), long_only=bool(contract.get("long_only", True)),
        sector_ids=contract.get("sector_ids"), sector_caps=contract.get("sector_caps"),
        liquidity_eligible=contract.get("liquidity_eligible"), turnover_limit=contract.get("turnover_limit"),
        l1_turnover_penalty=float(contract.get("l1_turnover_penalty", 0)),
        l2_turnover_penalty=float(contract.get("l2_turnover_penalty", 0)),
        risk_aversion=float(contract.get("risk_aversion", 0)), cash_allowed=bool(contract.get("cash_allowed", True)),
        ineligible_asset_policy=str(contract.get("ineligible_asset_policy", "liquidate")),
        alpha_identity=str(contract.get("alpha_identity", "synthetic-alpha")),
        covariance_identity=str(contract.get("covariance_identity", "synthetic-covariance")),
        policy_identity=str(contract.get("policy_identity", POLICY_ID)),
        decision_timestamp=str(contract.get("decision_timestamp", "synthetic")),
    )


def _initial_feasible_weights(data):
    minimum = np.asarray(data["effective_minimum_weights"])
    maximum = np.asarray(data["effective_maximum_weights"])
    weights = minimum.copy()
    remaining = data["exposure_target"] - weights.sum()
    for index in range(len(weights)):
        addition = min(remaining, maximum[index] - weights[index])
        weights[index] += addition; remaining -= addition
        if remaining <= CONSTRAINT_TOLERANCE: break
    if remaining > CONSTRAINT_TOLERANCE:
        raise AimPortfolioInputError("INFEASIBLE", "INITIAL_FEASIBLE_POINT_UNAVAILABLE")
    # SLSQP needs a sector-feasible start; deterministically shift excess.
    for sector, cap in data["sector_caps"].items():
        indexes = [index for index, value in enumerate(data["sector_ids"]) if value == sector]
        excess = weights[indexes].sum() - cap
        if excess <= CONSTRAINT_TOLERANCE: continue
        for index in reversed(indexes):
            reduction = min(excess, weights[index] - minimum[index]); weights[index] -= reduction; excess -= reduction
        for index in range(len(weights)):
            if data["sector_ids"][index] == sector: continue
            room = maximum[index] - weights[index]
            other_cap = data["sector_caps"].get(data["sector_ids"][index])
            if other_cap is not None:
                sector_current = sum(weights[j] for j, value in enumerate(data["sector_ids"]) if value == data["sector_ids"][index])
                room = min(room, other_cap - sector_current)
            addition = min(max(excess, 0), max(room, 0)); weights[index] += addition; excess -= addition
            if excess <= CONSTRAINT_TOLERANCE: break
        if excess > CONSTRAINT_TOLERANCE: raise AimPortfolioInputError("INFEASIBLE", "SECTOR_FEASIBLE_POINT_UNAVAILABLE")
    return weights


def _constraint_residuals(data, weights):
    minimum = np.asarray(data["effective_minimum_weights"]); maximum = np.asarray(data["effective_maximum_weights"])
    gross = float(np.abs(weights - np.asarray(data["previous_weights"])).sum())
    sector_values = [data["sector_caps"][sector] - exposure for sector, exposure in _sector_exposures(data, weights).items() if sector in data["sector_caps"]]
    liquidity_residuals = []
    for index, eligible in enumerate(data["liquidity_eligible"]):
        if not eligible:
            cap = 0 if data["ineligible_asset_policy"] == "liquidate" else data["previous_weights"][index]
            liquidity_residuals.append(cap - weights[index])
    return {
        "exposure": float(weights.sum() - data["exposure_target"]),
        "minimum_weight": float(np.min(weights - minimum)),
        "maximum_weight": float(np.min(maximum - weights)),
        "sector_cap": float(min(sector_values)) if sector_values else None,
        "turnover_limit": None if data["turnover_limit"] is None else float(data["turnover_limit"] - gross),
        "liquidity": float(min(liquidity_residuals)) if liquidity_residuals else None,
    }


def _sector_exposures(data, weights):
    return {sector: float(sum(weights[index] for index, value in enumerate(data["sector_ids"]) if value == sector)) for sector in sorted(set(data["sector_ids"]))}


def _solver_covariance(data):
    matrix = np.asarray(data["covariance"])
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return vectors @ np.diag(np.maximum(values, 0)) @ vectors.T


def _control_metrics(data, weights):
    components = objective_components(data, weights)
    variance = float(weights @ np.asarray(data["covariance"]) @ weights)
    return {
        "weights": weights.tolist(), "expected_alpha": components["expected_alpha_contribution"],
        "expected_variance": variance, "expected_volatility": math.sqrt(max(variance, 0)),
        "gross_turnover": float(np.abs(weights - np.asarray(data["previous_weights"])).sum()),
        "objective_components": components, "concentration_hhi": float(weights @ weights),
        "constraint_verification": _constraint_residuals(data, weights),
    }


def _economically_non_unique(data):
    return len(set(zip(data["expected_alpha"], np.diag(np.asarray(data["covariance"])).tolist()))) < len(data["asset_ids"])


def _finite_vector(values, n, owner):
    result = [float(value) for value in values]
    if len(result) != n: raise AimPortfolioInputError("INVALID_INPUT", f"{owner}_DIMENSION_MISMATCH")
    if not all(math.isfinite(value) for value in result): raise AimPortfolioInputError("INVALID_INPUT", f"{owner}_NON_FINITE")
    return result


def _bounds(values, n, owner):
    if isinstance(values, (int, float)):
        result = [float(values)] * n
    else:
        result = [float(value) for value in values]
    if len(result) != n or not all(math.isfinite(value) for value in result):
        raise AimPortfolioInputError("INVALID_INPUT", f"{owner}_INVALID")
    return result


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}


def _identity_safe(value):
    if isinstance(value, Mapping): return {str(key): _identity_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)): return [_identity_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value): return {"non_finite_float": repr(value)}
    return value
