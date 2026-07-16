from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "cvar_portfolio_input_v1"
RESULT_CONTRACT = "cvar_portfolio_result_v1"
PANEL_CONTRACT = "cvar_confidence_panel_v1"
COMPARISON_CONTRACT = "cvar_ex_ante_comparison_v1"
POLICY_VERSION = "1.0"
SUPPORTED_CONFIDENCE_LEVELS = (0.95, 0.975)
SOLVER_TOLERANCE = 1e-10
CONSTRAINT_TOLERANCE = 1e-7
MAX_ITERATIONS = 2000
STATUSES = {
    "OPTIMAL", "INFEASIBLE", "INSUFFICIENT_DATA", "INVALID_INPUT",
    "UNSUPPORTED_CONFIGURATION", "SOLVER_UNAVAILABLE", "NUMERICAL_FAILURE",
}


class CVaRInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def cvar_input(
    asset_ids: Sequence[str],
    scenario_ids: Sequence[str],
    scenario_returns: Sequence[Sequence[float]],
    expected_alpha: Sequence[float],
    previous_weights: Sequence[float],
    *,
    scenario_probabilities: Sequence[float] | None = None,
    exposure_target: float = 1.0,
    confidence_level: float = 0.95,
    cvar_risk_aversion: float = 1.0,
    l1_turnover_penalty: float = 0.0,
    l2_turnover_penalty: float = 0.0,
    turnover_limit: float | None = None,
    long_only: bool = True,
    minimum_weights: float | Sequence[float] = 0.0,
    maximum_weights: float | Sequence[float] = 1.0,
    sector_ids: Sequence[str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    liquidity_eligible: Sequence[bool] | None = None,
    cash_allowed: bool = True,
    ineligible_asset_policy: str = "liquidate",
    scenario_history_identity: str = "synthetic_scenario_history",
    scenario_generation_method: str = "synthetic_test_scenarios",
    scenario_generation_version: str = "1.0",
    scenario_horizon: str = "explicit_synthetic_horizon",
    scenarios_overlap: bool = False,
    policy_identity: str | None = None,
    decision_timestamp: str = "synthetic",
    minimum_scenarios: int = 10,
) -> dict[str, Any]:
    assets = [str(value) for value in asset_ids]
    scenarios = [str(value) for value in scenario_ids]
    n, count = len(assets), len(scenarios)
    if not assets or len(set(assets)) != n:
        raise CVaRInputError("INVALID_INPUT", "ASSET_IDENTITIES_INVALID")
    if assets != sorted(assets):
        raise CVaRInputError("INVALID_INPUT", "ASSETS_NOT_DETERMINISTICALLY_ORDERED")
    if not scenarios or len(set(scenarios)) != count:
        raise CVaRInputError("INVALID_INPUT", "SCENARIO_IDENTITIES_INVALID")
    if scenarios != sorted(scenarios):
        raise CVaRInputError("INVALID_INPUT", "SCENARIOS_NOT_DETERMINISTICALLY_ORDERED")
    matrix = np.asarray(scenario_returns, dtype=float)
    if matrix.shape != (count, n):
        raise CVaRInputError("INVALID_INPUT", "SCENARIO_RETURN_DIMENSION_MISMATCH")
    if count < minimum_scenarios:
        raise CVaRInputError("INSUFFICIENT_DATA", "SCENARIO_COUNT_INSUFFICIENT")
    if not np.isfinite(matrix).all():
        raise CVaRInputError("INVALID_INPUT", "SCENARIO_RETURN_NON_FINITE")
    alpha = _finite_vector(expected_alpha, n, "EXPECTED_ALPHA")
    previous = _finite_vector(previous_weights, n, "PREVIOUS_WEIGHTS")
    if any(value < -CONSTRAINT_TOLERANCE for value in previous):
        raise CVaRInputError("INVALID_INPUT", "PREVIOUS_WEIGHTS_NEGATIVE")
    probabilities = np.asarray(scenario_probabilities if scenario_probabilities is not None else [1 / count] * count, dtype=float)
    if probabilities.shape != (count,) or not np.isfinite(probabilities).all():
        raise CVaRInputError("INVALID_INPUT", "SCENARIO_PROBABILITY_DIMENSION_OR_FINITE_ERROR")
    if np.any(probabilities <= 0):
        raise CVaRInputError("INVALID_INPUT", "SCENARIO_PROBABILITY_NON_POSITIVE")
    if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-10):
        raise CVaRInputError("INVALID_INPUT", "SCENARIO_PROBABILITIES_NOT_NORMALISED")
    if not 0 < confidence_level < 1:
        raise CVaRInputError("INVALID_INPUT", "CONFIDENCE_LEVEL_INVALID")
    if confidence_level not in SUPPORTED_CONFIDENCE_LEVELS:
        raise CVaRInputError("UNSUPPORTED_CONFIGURATION", "CONFIDENCE_LEVEL_NOT_REGISTERED")
    coefficients = (cvar_risk_aversion, l1_turnover_penalty, l2_turnover_penalty)
    if any(not math.isfinite(value) or value < 0 for value in coefficients):
        raise CVaRInputError("INVALID_INPUT", "OBJECTIVE_COEFFICIENT_INVALID")
    if turnover_limit is not None and (not math.isfinite(turnover_limit) or turnover_limit < 0):
        raise CVaRInputError("INVALID_INPUT", "TURNOVER_LIMIT_INVALID")
    if not long_only:
        raise CVaRInputError("UNSUPPORTED_CONFIGURATION", "LONG_SHORT_NOT_SUPPORTED")
    if ineligible_asset_policy not in {"liquidate", "retain_only"}:
        raise CVaRInputError("UNSUPPORTED_CONFIGURATION", "INELIGIBLE_ASSET_POLICY_UNSUPPORTED")
    minimum = _bounds(minimum_weights, n, "MINIMUM_WEIGHTS")
    maximum = _bounds(maximum_weights, n, "MAXIMUM_WEIGHTS")
    if any(value < 0 for value in minimum + maximum) or any(low > high for low, high in zip(minimum, maximum)):
        raise CVaRInputError("INVALID_INPUT", "STOCK_BOUNDS_INVALID")
    eligibility = list(liquidity_eligible if liquidity_eligible is not None else [True] * n)
    if len(eligibility) != n or any(not isinstance(value, (bool, np.bool_)) for value in eligibility):
        raise CVaRInputError("INVALID_INPUT", "LIQUIDITY_MASK_INVALID")
    effective_minimum = list(minimum); effective_maximum = list(maximum)
    for index, eligible in enumerate(eligibility):
        if not eligible:
            effective_minimum[index] = 0.0
            effective_maximum[index] = 0.0 if ineligible_asset_policy == "liquidate" else min(maximum[index], previous[index])
    if not math.isfinite(exposure_target) or exposure_target < 0 or (cash_allowed and exposure_target > 1 + CONSTRAINT_TOLERANCE):
        raise CVaRInputError("INVALID_INPUT", "EXPOSURE_TARGET_INVALID")
    if not cash_allowed and abs(exposure_target - 1) > CONSTRAINT_TOLERANCE:
        raise CVaRInputError("INVALID_INPUT", "CASH_DISALLOWED_REQUIRES_FULL_EXPOSURE")
    if sum(effective_minimum) > exposure_target + CONSTRAINT_TOLERANCE or sum(effective_maximum) < exposure_target - CONSTRAINT_TOLERANCE:
        raise CVaRInputError("INFEASIBLE", "STOCK_CAPS_CANNOT_MEET_EXPOSURE")
    sectors = [str(value) for value in (sector_ids or ["UNCLASSIFIED"] * n)]
    if len(sectors) != n or any(not value for value in sectors):
        raise CVaRInputError("INVALID_INPUT", "SECTOR_MAPPING_INVALID")
    caps = {str(key): float(value) for key, value in sorted((sector_caps or {}).items())}
    if any(not math.isfinite(value) or value < 0 for value in caps.values()):
        raise CVaRInputError("INVALID_INPUT", "SECTOR_CAP_INVALID")
    capacity = sum(min(caps.get(sector, exposure_target), sum(effective_maximum[index] for index, value in enumerate(sectors) if value == sector)) for sector in sorted(set(sectors)))
    if capacity < exposure_target - CONSTRAINT_TOLERANCE:
        raise CVaRInputError("INFEASIBLE", "SECTOR_CAPS_CANNOT_MEET_EXPOSURE")
    policy = policy_identity or f"cvar_{str(confidence_level).replace('.', '_')}_portfolio_v1"
    asset_checksum = canonical_hash({"contract": "portfolio_risk_input_v1", "asset_ids": assets})
    scenario_checksum = canonical_hash({"scenario_ids": scenarios})
    return_checksum = canonical_hash({"scenario_ids": scenarios, "returns": matrix.tolist()})
    probability_checksum = canonical_hash({"scenario_ids": scenarios, "probabilities": probabilities.tolist()})
    config = {
        "exposure_target": exposure_target, "confidence_level": confidence_level,
        "cvar_risk_aversion": cvar_risk_aversion, "l1_turnover_penalty": l1_turnover_penalty,
        "l2_turnover_penalty": l2_turnover_penalty, "turnover_limit": turnover_limit,
        "long_only": long_only, "minimum_weights": minimum, "maximum_weights": maximum,
        "sector_ids": sectors, "sector_caps": caps, "liquidity_eligible": eligibility,
        "cash_allowed": cash_allowed, "ineligible_asset_policy": ineligible_asset_policy,
        "scenario_history_identity": scenario_history_identity,
        "scenario_generation_method": scenario_generation_method,
        "scenario_generation_version": scenario_generation_version,
        "scenario_horizon": scenario_horizon, "scenarios_overlap": scenarios_overlap,
        "policy_identity": policy, "decision_timestamp": decision_timestamp,
    }
    return {
        "contract_version": INPUT_CONTRACT, "asset_ids": assets, "scenario_ids": scenarios,
        "scenario_returns": matrix.tolist(), "scenario_probabilities": probabilities.tolist(),
        "probability_policy": "uniform" if scenario_probabilities is None else "supplied",
        "expected_alpha": alpha, "previous_weights": previous, **config,
        "effective_minimum_weights": effective_minimum, "effective_maximum_weights": effective_maximum,
        "asset_population_checksum": asset_checksum, "scenario_population_checksum": scenario_checksum,
        "scenario_return_checksum": return_checksum, "scenario_probability_checksum": probability_checksum,
        "configuration_checksum": canonical_hash(config),
        "scenario_count": count,
    }


def discrete_cvar(losses: Sequence[float], probabilities: Sequence[float], confidence_level: float) -> dict[str, Any]:
    values = np.asarray(losses, dtype=float); probs = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or probs.shape != values.shape or not np.isfinite(values).all() or not np.isfinite(probs).all():
        raise CVaRInputError("INVALID_INPUT", "DISCRETE_CVAR_INPUT_INVALID")
    if np.any(probs <= 0) or not math.isclose(float(probs.sum()), 1, abs_tol=1e-10):
        raise CVaRInputError("INVALID_INPUT", "DISCRETE_CVAR_PROBABILITY_INVALID")
    if not 0 < confidence_level < 1:
        raise CVaRInputError("INVALID_INPUT", "CONFIDENCE_LEVEL_INVALID")
    order = np.argsort(values, kind="mergesort")
    sorted_losses, sorted_probs = values[order], probs[order]
    cumulative = np.cumsum(sorted_probs)
    index = int(np.searchsorted(cumulative, confidence_level, side="left"))
    threshold = float(sorted_losses[index])
    excess = np.maximum(values - threshold, 0)
    cvar = threshold + float(probs @ excess) / (1 - confidence_level)
    return {"var_threshold": threshold, "cvar": cvar, "excess_losses": excess.tolist()}


def optimise_cvar_portfolio(contract: Mapping[str, Any], *, solver_tolerance: float = SOLVER_TOLERANCE, max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    try:
        data = _validated_contract(contract)
        if solver_tolerance <= 0 or max_iterations < 1:
            raise CVaRInputError("UNSUPPORTED_CONFIGURATION", "SOLVER_CONFIGURATION_INVALID")
        try:
            import scipy
            from scipy.optimize import minimize
        except ImportError:
            return _blocked(data, "SOLVER_UNAVAILABLE", "SCIPY_UNAVAILABLE", solver_tolerance, max_iterations)
        n, count = len(data["asset_ids"]), data["scenario_count"]
        returns = np.asarray(data["scenario_returns"]); probabilities = np.asarray(data["scenario_probabilities"])
        alpha = np.asarray(data["expected_alpha"]); previous = np.asarray(data["previous_weights"])
        q = data["confidence_level"]; risk = data["cvar_risk_aversion"]
        k1, k2 = data["l1_turnover_penalty"], data["l2_turnover_penalty"]

        def unpack(vector):
            return vector[:n], vector[n], vector[n + 1:n + 1 + count], vector[n + 1 + count:]

        def objective(vector):
            weights, zeta, excess, turnover_aux = unpack(vector)
            change = weights - previous
            cvar = zeta + probabilities @ excess / (1 - q)
            return float(-alpha @ weights + risk * cvar + k1 * turnover_aux.sum() + k2 * change @ change)

        constraints = [{"type": "eq", "fun": lambda vector: float(unpack(vector)[0].sum() - data["exposure_target"])}]
        for scenario in range(count):
            constraints.append({"type": "ineq", "fun": lambda vector, scenario=scenario: float(unpack(vector)[2][scenario] + unpack(vector)[1] + returns[scenario] @ unpack(vector)[0])})
        for index in range(n):
            constraints.append({"type": "ineq", "fun": lambda vector, index=index: float(unpack(vector)[3][index] - (unpack(vector)[0][index] - previous[index]))})
            constraints.append({"type": "ineq", "fun": lambda vector, index=index: float(unpack(vector)[3][index] + (unpack(vector)[0][index] - previous[index]))})
        if data["turnover_limit"] is not None:
            constraints.append({"type": "ineq", "fun": lambda vector: float(data["turnover_limit"] - unpack(vector)[3].sum())})
        for sector, cap in data["sector_caps"].items():
            indexes = np.asarray([index for index, value in enumerate(data["sector_ids"]) if value == sector], dtype=int)
            constraints.append({"type": "ineq", "fun": lambda vector, indexes=indexes, cap=cap: float(cap - unpack(vector)[0][indexes].sum())})
        initial_weights = _initial_weights(data)
        losses = -returns @ initial_weights
        initial_tail = discrete_cvar(losses, probabilities, q)
        initial = np.r_[initial_weights, initial_tail["var_threshold"], initial_tail["excess_losses"], np.abs(initial_weights - previous)]
        loss_bound = max(float(np.max(np.abs(returns))) * max(data["exposure_target"], 1), 1.0) * 10
        aux_bound = max(2.0, float(np.abs(previous).sum() + data["exposure_target"]))
        bounds = (
            list(zip(data["effective_minimum_weights"], data["effective_maximum_weights"]))
            + [(-loss_bound, loss_bound)] + [(0, loss_bound * 2)] * count + [(0, aux_bound)] * n
        )
        solved = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"ftol": solver_tolerance, "maxiter": max_iterations, "disp": False})
        if not solved.success:
            status = "INFEASIBLE" if int(solved.status) in {4, 8} else "NUMERICAL_FAILURE"
            return _blocked(data, status, f"SCIPY_SLSQP:{solved.status}:{solved.message}", solver_tolerance, max_iterations, solver_version=scipy.__version__)
        weights, zeta, excess, _ = unpack(solved.x)
        weights[np.abs(weights) < CONSTRAINT_TOLERANCE] = 0
        excess[np.abs(excess) < CONSTRAINT_TOLERANCE] = 0
        result = _optimal_result(data, weights, float(zeta), excess, solver_tolerance, max_iterations, scipy.__version__, int(solved.nit))
        verification = verify_cvar_result(data, result)
        if not verification["valid"]:
            return _blocked(data, "NUMERICAL_FAILURE", "POST_SOLVE_VERIFICATION_FAILED", solver_tolerance, max_iterations, solver_version=scipy.__version__, extra=verification["blocking_reasons"])
        return result
    except CVaRInputError as exc:
        return _blocked(contract, exc.status, exc.reason, solver_tolerance, max_iterations)
    except Exception as exc:  # pragma: no cover
        return _blocked(contract, "NUMERICAL_FAILURE", type(exc).__name__, solver_tolerance, max_iterations)


def verify_cvar_result(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        if result.get("asset_population_checksum") != data["asset_population_checksum"]: reasons.append("ASSET_POPULATION_MISMATCH")
        if result.get("scenario_population_checksum") != data["scenario_population_checksum"]: reasons.append("SCENARIO_POPULATION_MISMATCH")
        if result.get("scenario_return_checksum") != data["scenario_return_checksum"]: reasons.append("SCENARIO_RETURN_MISMATCH")
        if result.get("scenario_probability_checksum") != data["scenario_probability_checksum"]: reasons.append("SCENARIO_PROBABILITY_MISMATCH")
        weights = np.asarray(result.get("target_weights", ()), dtype=float)
        excess = np.asarray(result.get("excess_loss_variables", ()), dtype=float)
        if weights.shape != (len(data["asset_ids"]),) or excess.shape != (data["scenario_count"],):
            return {"contract_version": "cvar_portfolio_verification_v1", "valid": False, "blocking_reasons": ["RESULT_DIMENSION_MISMATCH"]}
        returns = np.asarray(data["scenario_returns"]); probabilities = np.asarray(data["scenario_probabilities"])
        losses = -returns @ weights; zeta = float(result.get("var_threshold"))
        reported_losses = np.asarray(result.get("scenario_loss_vector", ()), dtype=float)
        if reported_losses.shape != losses.shape or not np.allclose(reported_losses, losses, atol=1e-8, rtol=0):
            reasons.append("SCENARIO_LOSS_MISMATCH")
        required_excess = np.maximum(losses - zeta, 0)
        if np.any(excess < -CONSTRAINT_TOLERANCE) or np.any(excess + CONSTRAINT_TOLERANCE < required_excess): reasons.append("EXCESS_LOSS_CONSTRAINT_FAILED")
        recomputed_cvar = zeta + float(probabilities @ excess) / (1 - data["confidence_level"])
        if abs(recomputed_cvar - float(result.get("cvar", math.inf))) > 1e-6: reasons.append("CVAR_RECOMPUTATION_MISMATCH")
        expected_alpha = float(np.asarray(data["expected_alpha"]) @ weights)
        change = weights - np.asarray(data["previous_weights"]); gross = float(np.abs(change).sum())
        l1 = data["l1_turnover_penalty"] * gross; l2 = data["l2_turnover_penalty"] * float(change @ change)
        total = expected_alpha - data["cvar_risk_aversion"] * recomputed_cvar - l1 - l2
        checks = {
            "EXPECTED_ALPHA_MISMATCH": (expected_alpha, result.get("expected_alpha")),
            "CVAR_PENALTY_MISMATCH": (
                data["cvar_risk_aversion"] * recomputed_cvar,
                result.get("cvar_penalty"),
            ),
            "L1_PENALTY_MISMATCH": (l1, result.get("l1_turnover_penalty")),
            "L2_PENALTY_MISMATCH": (l2, result.get("l2_turnover_penalty")),
            "OBJECTIVE_MISMATCH": (total, result.get("total_objective")),
            "TURNOVER_MISMATCH": (gross, result.get("gross_turnover")),
        }
        for reason, (left, right) in checks.items():
            if abs(left - float(right)) > 1e-6: reasons.append(reason)
        residuals = _constraint_residuals(data, weights, gross)
        if abs(residuals["exposure"]) > CONSTRAINT_TOLERANCE: reasons.append("EXPOSURE_FAILED")
        if residuals["minimum_weight"] < -CONSTRAINT_TOLERANCE: reasons.append("MINIMUM_WEIGHT_FAILED")
        if residuals["maximum_weight"] < -CONSTRAINT_TOLERANCE: reasons.append("MAXIMUM_WEIGHT_FAILED")
        if residuals["sector_cap"] is not None and residuals["sector_cap"] < -CONSTRAINT_TOLERANCE: reasons.append("SECTOR_CAP_FAILED")
        if residuals["turnover_limit"] is not None and residuals["turnover_limit"] < -CONSTRAINT_TOLERANCE: reasons.append("TURNOVER_LIMIT_FAILED")
        active = [scenario for scenario, loss in zip(data["scenario_ids"], losses) if loss >= zeta - CONSTRAINT_TOLERANCE]
        if active != result.get("active_tail_scenario_ids"): reasons.append("ACTIVE_TAIL_MISMATCH")
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical): reasons.append("RESULT_CHECKSUM_MISMATCH")
        return {"contract_version": "cvar_portfolio_verification_v1", "valid": not reasons, "blocking_reasons": sorted(set(reasons)), "constraint_residuals": residuals}
    except (CVaRInputError, TypeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, CVaRInputError) else type(exc).__name__
        return {"contract_version": "cvar_portfolio_verification_v1", "valid": False, "blocking_reasons": [reason]}


def evaluate_confidence_panel(contract: Mapping[str, Any]) -> dict[str, Any]:
    base = _validated_contract(contract)
    results = {}
    for confidence in SUPPORTED_CONFIDENCE_LEVELS:
        candidate = dict(base); candidate["confidence_level"] = confidence
        candidate["policy_identity"] = f"cvar_{str(confidence).replace('.', '_')}_portfolio_v1"
        results[str(confidence)] = optimise_cvar_portfolio(candidate)
    if not all(result["valid"] for result in results.values()):
        status = "NUMERICAL_FAILURE"
    else:
        status = "OPTIMAL"
    left, right = results["0.95"], results["0.975"]
    logical = {
        "contract_version": PANEL_CONTRACT, "status": status, "valid": status == "OPTIMAL",
        "confidence_levels": list(SUPPORTED_CONFIDENCE_LEVELS), "results": results,
        "comparison": {
            "weight_differences_975_minus_95": (np.asarray(right.get("target_weights", [])) - np.asarray(left.get("target_weights", []))).tolist() if status == "OPTIMAL" else [],
            "cvar_difference": right.get("cvar") - left.get("cvar") if status == "OPTIMAL" else None,
            "expected_alpha_difference": right.get("expected_alpha") - left.get("expected_alpha") if status == "OPTIMAL" else None,
            "turnover_difference": right.get("gross_turnover") - left.get("gross_turnover") if status == "OPTIMAL" else None,
            "concentration_difference": right.get("concentration", {}).get("hhi", 0) - left.get("concentration", {}).get("hhi", 0) if status == "OPTIMAL" else None,
            "objective_difference": right.get("total_objective") - left.get("total_objective") if status == "OPTIMAL" else None,
            "monotonic_risk_claimed": False,
        },
        "asset_population_checksum": base["asset_population_checksum"],
        "scenario_population_checksum": base["scenario_population_checksum"],
        "scenario_return_checksum": base["scenario_return_checksum"],
        "scenario_probability_checksum": base["scenario_probability_checksum"],
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def compare_cvar_policies(contract: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    data = _validated_contract(contract); returns = np.asarray(data["scenario_returns"]); probabilities = np.asarray(data["scenario_probabilities"])
    rows = {}
    for name, result in sorted(results.items()):
        if result.get("asset_population_checksum", result.get("population_checksum")) != data["asset_population_checksum"]:
            raise CVaRInputError("INVALID_INPUT", f"COMPARISON_ASSET_POPULATION_MISMATCH:{name}")
        if result.get("scenario_return_checksum") != data["scenario_return_checksum"]:
            raise CVaRInputError("INVALID_INPUT", f"COMPARISON_SCENARIO_HISTORY_MISMATCH:{name}")
        weights = np.asarray(result.get("target_weights", result.get("final_weights", ())), dtype=float)
        if weights.shape != (len(data["asset_ids"]),): raise CVaRInputError("INVALID_INPUT", f"COMPARISON_WEIGHTS_INVALID:{name}")
        losses = -returns @ weights; tail = discrete_cvar(losses, probabilities, data["confidence_level"])
        sectors = _sector_exposures(data, weights)
        rows[name] = {
            "expected_alpha": float(np.asarray(data["expected_alpha"]) @ weights),
            "expected_scenario_loss": float(probabilities @ losses), "scenario_var": tail["var_threshold"],
            "scenario_cvar": tail["cvar"], "worst_scenario_loss": float(losses.max()),
            "gross_turnover": float(np.abs(weights - np.asarray(data["previous_weights"])).sum()),
            "concentration_hhi": float(weights @ weights), "maximum_stock_weight": float(weights.max()),
            "sector_concentration_hhi": float(sum(value**2 for value in sectors.values())),
        }
    logical = {
        "contract_version": COMPARISON_CONTRACT, "status": "OPTIMAL", "valid": True,
        "asset_population_checksum": data["asset_population_checksum"],
        "scenario_population_checksum": data["scenario_population_checksum"],
        "scenario_return_checksum": data["scenario_return_checksum"],
        "scenario_probability_checksum": data["scenario_probability_checksum"], "policy_count": len(rows),
        "policies": rows, "historical_returns_computed": False, "sharpe_computed": False, "promotion_decision": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _optimal_result(data, weights, zeta, excess, tolerance, max_iterations, solver_version, iterations):
    returns = np.asarray(data["scenario_returns"]); probabilities = np.asarray(data["scenario_probabilities"])
    losses = -returns @ weights; change = weights - np.asarray(data["previous_weights"])
    gross = float(np.abs(change).sum()); expected_alpha = float(np.asarray(data["expected_alpha"]) @ weights)
    expected_excess = float(probabilities @ excess); cvar = zeta + expected_excess / (1 - data["confidence_level"])
    cvar_penalty = data["cvar_risk_aversion"] * cvar
    l1 = data["l1_turnover_penalty"] * gross; l2 = data["l2_turnover_penalty"] * float(change @ change)
    total = expected_alpha - cvar_penalty - l1 - l2
    sectors = _sector_exposures(data, weights)
    active = [scenario for scenario, loss in zip(data["scenario_ids"], losses) if loss >= zeta - CONSTRAINT_TOLERANCE]
    warnings = []
    if len(set(np.round(losses, 12))) < len(losses): warnings.append("DUPLICATE_SCENARIO_LOSSES")
    if np.std(losses) <= CONSTRAINT_TOLERANCE: warnings.append("WEAKLY_IDENTIFIED_CONSTANT_SCENARIO_LOSS")
    logical = {
        "contract_version": RESULT_CONTRACT, "policy_id": data["policy_identity"], "policy_version": POLICY_VERSION,
        "status": "OPTIMAL", "valid": True, "blocking_reasons": [], "warnings": warnings,
        "confidence_level": data["confidence_level"], "asset_ids": data["asset_ids"], "scenario_ids": data["scenario_ids"],
        "target_weights": weights.tolist(), "previous_weights": data["previous_weights"], "trade_weight_changes": change.tolist(),
        "gross_turnover": gross, "one_way_turnover": gross / 2,
        "turnover_convention": "gross=sum(abs(target-previous)); one_way=gross/2",
        "expected_alpha": expected_alpha, "var_threshold": zeta, "expected_excess_loss": expected_excess,
        "cvar": cvar, "expected_scenario_loss": float(probabilities @ losses), "worst_scenario_loss": float(losses.max()),
        "cvar_penalty": cvar_penalty, "l1_turnover_penalty": l1, "l2_turnover_penalty": l2,
        "total_objective": total, "scenario_loss_vector": losses.tolist(), "excess_loss_variables": excess.tolist(),
        "active_tail_scenario_ids": active,
        "stock_cap_utilisation": [weights[index] / cap if cap > 0 else None for index, cap in enumerate(data["effective_maximum_weights"])],
        "sector_exposures": sectors,
        "sector_cap_utilisation": {sector: value / data["sector_caps"][sector] for sector, value in sectors.items() if sector in data["sector_caps"] and data["sector_caps"][sector] > 0},
        "liquidity_exclusions": [asset for asset, eligible in zip(data["asset_ids"], data["liquidity_eligible"]) if not eligible],
        "constraint_residuals": _constraint_residuals(data, weights, gross),
        "concentration": {"hhi": float(weights @ weights), "maximum_stock_weight": float(weights.max())},
        "solver_identity": "scipy.optimize.SLSQP", "solver_version": solver_version, "solver_status": "success",
        "solver_tolerance": tolerance, "constraint_tolerance": CONSTRAINT_TOLERANCE,
        "maximum_iterations": max_iterations, "iteration_count": iterations,
        "scenario_count": data["scenario_count"], "scenario_probabilities": data["scenario_probabilities"],
        "scenario_horizon": data["scenario_horizon"], "scenario_method_identity": data["scenario_generation_method"],
        "scenario_method_version": data["scenario_generation_version"], "scenarios_overlap": data["scenarios_overlap"],
        "probability_policy": data["probability_policy"],
        "asset_population_checksum": data["asset_population_checksum"],
        "scenario_population_checksum": data["scenario_population_checksum"],
        "scenario_return_checksum": data["scenario_return_checksum"],
        "scenario_probability_checksum": data["scenario_probability_checksum"],
        "configuration_checksum": data["configuration_checksum"],
        "adv_capacity_status": "UNVERIFIED", "execution_model": False, "orders_generated": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked(data, status, reason, tolerance, max_iterations, *, solver_version=None, extra=()):
    logical = {
        "contract_version": RESULT_CONTRACT, "policy_id": data.get("policy_identity") if isinstance(data, Mapping) else None,
        "policy_version": POLICY_VERSION, "status": status if status in STATUSES else "INVALID_INPUT",
        "valid": False, "blocking_reasons": sorted(set([reason, *extra])), "warnings": [],
        "confidence_level": data.get("confidence_level") if isinstance(data, Mapping) else None,
        "asset_ids": list(data.get("asset_ids", ())) if isinstance(data, Mapping) else [],
        "scenario_ids": list(data.get("scenario_ids", ())) if isinstance(data, Mapping) else [],
        "target_weights": [], "previous_weights": list(data.get("previous_weights", ())) if isinstance(data, Mapping) else [],
        "trade_weight_changes": [], "gross_turnover": None, "one_way_turnover": None,
        "turnover_convention": "gross=sum(abs(target-previous)); one_way=gross/2",
        "expected_alpha": None, "var_threshold": None, "expected_excess_loss": None, "cvar": None,
        "expected_scenario_loss": None, "worst_scenario_loss": None, "cvar_penalty": None,
        "l1_turnover_penalty": None, "l2_turnover_penalty": None, "total_objective": None,
        "scenario_loss_vector": [], "excess_loss_variables": [], "active_tail_scenario_ids": [],
        "stock_cap_utilisation": [], "sector_exposures": {}, "sector_cap_utilisation": {},
        "liquidity_exclusions": [], "constraint_residuals": {}, "concentration": {},
        "solver_identity": "scipy.optimize.SLSQP", "solver_version": solver_version, "solver_status": "not_solved",
        "solver_tolerance": tolerance, "constraint_tolerance": CONSTRAINT_TOLERANCE,
        "maximum_iterations": max_iterations, "iteration_count": None,
        "scenario_count": data.get("scenario_count") if isinstance(data, Mapping) else None,
        "scenario_probabilities": list(data.get("scenario_probabilities", ())) if isinstance(data, Mapping) else [],
        "scenario_horizon": data.get("scenario_horizon") if isinstance(data, Mapping) else None,
        "scenario_method_identity": data.get("scenario_generation_method") if isinstance(data, Mapping) else None,
        "scenario_method_version": data.get("scenario_generation_version") if isinstance(data, Mapping) else None,
        "scenarios_overlap": data.get("scenarios_overlap") if isinstance(data, Mapping) else None,
        "probability_policy": data.get("probability_policy") if isinstance(data, Mapping) else None,
        "asset_population_checksum": data.get("asset_population_checksum") if isinstance(data, Mapping) else None,
        "scenario_population_checksum": data.get("scenario_population_checksum") if isinstance(data, Mapping) else None,
        "scenario_return_checksum": data.get("scenario_return_checksum") if isinstance(data, Mapping) else None,
        "scenario_probability_checksum": data.get("scenario_probability_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": data.get("configuration_checksum") if isinstance(data, Mapping) else None,
        "adv_capacity_status": "UNVERIFIED", "execution_model": False, "orders_generated": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _validated_contract(contract):
    return cvar_input(
        contract["asset_ids"], contract["scenario_ids"], contract["scenario_returns"],
        contract["expected_alpha"], contract["previous_weights"],
        scenario_probabilities=contract.get("scenario_probabilities"),
        exposure_target=float(contract.get("exposure_target", 1)), confidence_level=float(contract.get("confidence_level", .95)),
        cvar_risk_aversion=float(contract.get("cvar_risk_aversion", 1)),
        l1_turnover_penalty=float(contract.get("l1_turnover_penalty", 0)),
        l2_turnover_penalty=float(contract.get("l2_turnover_penalty", 0)),
        turnover_limit=contract.get("turnover_limit"), long_only=bool(contract.get("long_only", True)),
        minimum_weights=contract.get("minimum_weights", 0), maximum_weights=contract.get("maximum_weights", 1),
        sector_ids=contract.get("sector_ids"), sector_caps=contract.get("sector_caps"),
        liquidity_eligible=contract.get("liquidity_eligible"), cash_allowed=bool(contract.get("cash_allowed", True)),
        ineligible_asset_policy=str(contract.get("ineligible_asset_policy", "liquidate")),
        scenario_history_identity=str(contract.get("scenario_history_identity", "synthetic_scenario_history")),
        scenario_generation_method=str(contract.get("scenario_generation_method", "synthetic_test_scenarios")),
        scenario_generation_version=str(contract.get("scenario_generation_version", "1.0")),
        scenario_horizon=str(contract.get("scenario_horizon", "explicit_synthetic_horizon")),
        scenarios_overlap=bool(contract.get("scenarios_overlap", False)),
        policy_identity=contract.get("policy_identity"), decision_timestamp=str(contract.get("decision_timestamp", "synthetic")),
    )


def _initial_weights(data):
    weights = np.asarray(data["effective_minimum_weights"], dtype=float)
    maximum = np.asarray(data["effective_maximum_weights"], dtype=float)
    remaining = data["exposure_target"] - weights.sum()
    for _ in range(len(weights) * 4 + 4):
        if remaining <= CONSTRAINT_TOLERANCE: break
        sectors = _sector_exposures(data, weights); room = np.maximum(maximum - weights, 0)
        active = room > CONSTRAINT_TOLERANCE
        if not np.any(active): raise CVaRInputError("INFEASIBLE", "INITIAL_WEIGHT_CAPACITY_EXHAUSTED")
        addition = active.astype(float); addition = addition / addition.sum() * remaining; addition = np.minimum(addition, room)
        for sector, cap in data["sector_caps"].items():
            indexes = np.asarray([index for index, value in enumerate(data["sector_ids"]) if value == sector])
            sector_room = max(cap - sectors[sector], 0); total = float(addition[indexes].sum())
            if total > sector_room and total > 0: addition[indexes] *= sector_room / total
        if addition.sum() <= CONSTRAINT_TOLERANCE: raise CVaRInputError("INFEASIBLE", "INITIAL_WEIGHT_ALLOCATION_STALLED")
        weights += addition; remaining -= float(addition.sum())
    if remaining > CONSTRAINT_TOLERANCE: raise CVaRInputError("INFEASIBLE", "INITIAL_EXPOSURE_NOT_ALLOCATED")
    return weights


def _constraint_residuals(data, weights, gross):
    sectors = _sector_exposures(data, weights)
    sector_values = [data["sector_caps"][sector] - value for sector, value in sectors.items() if sector in data["sector_caps"]]
    liquidity = [data["effective_maximum_weights"][index] - weights[index] for index, eligible in enumerate(data["liquidity_eligible"]) if not eligible]
    return {
        "exposure": float(weights.sum() - data["exposure_target"]),
        "minimum_weight": float(np.min(weights - np.asarray(data["effective_minimum_weights"]))),
        "maximum_weight": float(np.min(np.asarray(data["effective_maximum_weights"]) - weights)),
        "sector_cap": float(min(sector_values)) if sector_values else None,
        "liquidity": float(min(liquidity)) if liquidity else None,
        "turnover_limit": None if data["turnover_limit"] is None else float(data["turnover_limit"] - gross),
    }


def _sector_exposures(data, weights):
    return {sector: float(sum(weights[index] for index, value in enumerate(data["sector_ids"]) if value == sector)) for sector in sorted(set(data["sector_ids"]))}


def _finite_vector(values, n, owner):
    result = [float(value) for value in values]
    if len(result) != n: raise CVaRInputError("INVALID_INPUT", f"{owner}_DIMENSION_MISMATCH")
    if not all(math.isfinite(value) for value in result): raise CVaRInputError("INVALID_INPUT", f"{owner}_NON_FINITE")
    return result


def _bounds(values, n, owner):
    result = [float(values)] * n if isinstance(values, (int, float)) else [float(value) for value in values]
    if len(result) != n or not all(math.isfinite(value) for value in result): raise CVaRInputError("INVALID_INPUT", f"{owner}_INVALID")
    return result


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
