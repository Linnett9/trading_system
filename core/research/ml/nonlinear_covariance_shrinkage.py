from __future__ import annotations

import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_CONTRACT = "nonlinear_covariance_input_v1"
RESULT_CONTRACT = "nonlinear_covariance_result_v1"
ALLOCATION_CONTRACT = "nonlinear_minimum_variance_result_v1"
COMPARISON_CONTRACT = "linear_nonlinear_covariance_comparison_v1"
AUDIT_CONTRACT = "nonlinear_covariance_dependency_audit_v1"
READINESS_STATUS = "BLOCKED_NO_VERIFIED_IMPLEMENTATION"
PSD_TOLERANCE = 1e-8
STATUSES = {
    "VALID", "INSUFFICIENT_DATA", "INVALID_INPUT", "UNSUPPORTED_CONFIGURATION",
    "DEPENDENCY_UNAVAILABLE", "NUMERICAL_FAILURE", "METHOD_NOT_VERIFIED",
}


class NonlinearCovarianceInputError(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest().upper()


def dependency_audit(*, availability_overrides: Mapping[str, bool] | None = None) -> dict[str, Any]:
    overrides = dict(availability_overrides or {})
    candidates = [
        ("nonlinshrink", "nonlinshrink", "Analytical nonlinear shrinkage", "covariance", "package implementation; licence not locally evident"),
        ("riskfolio-lib", "riskfolio", "Riskfolio-Lib covariance estimators", "covariance", "package licence metadata if installed"),
        ("PyPortfolioOpt", "pypfopt", "PyPortfolioOpt risk models", "covariance", "MIT when installed; no verified nonlinear estimator selected"),
        ("statsmodels", "statsmodels", "Statsmodels covariance tools", "mixed", "BSD-style project; no selected nonlinear shrinkage owner"),
        ("scikit-learn", "sklearn", "Ledoit-Wolf/OAS linear shrinkage only", "covariance and precision", "BSD-3-Clause"),
        ("SciPy", "scipy", "Numerical linear algebra and optimization; no nonlinear shrinkage estimator", "neither estimator", "BSD-3-Clause"),
        ("QuEST", "quest", "QuEST nonlinear shrinkage candidate", "unknown", "not installed; licence not locally evident"),
    ]
    rows = []
    for distribution, module, method, output, licence in candidates:
        available = overrides.get(module, importlib.util.find_spec(module) is not None)
        version = _version(distribution) if available else None
        recognized_usable = bool(module == "nonlinshrink" and available)
        rows.append({
            "package": distribution, "module": module, "version": version,
            "available": available, "candidate_method": method,
            "returns": output, "licence_compatibility": licence,
            "deterministic_assessment": "requires fixed inputs; package implementation must be verified" if available else "not assessable",
            "high_dimensional_assessment": "method-specific; not assumed",
            "maintained_state": "not established by local environment audit",
            "independently_validated_here": False,
            "recognized_usable_backend": recognized_usable,
        })
    usable = [row for row in rows if row["recognized_usable_backend"]]
    logical = {
        "contract_version": AUDIT_CONTRACT, "candidates": rows,
        "selected_backend": usable[0]["module"] if len(usable) == 1 else None,
        "status": "SUPPORTED_DEPENDENCY_SELECTED" if len(usable) == 1 else READINESS_STATUS,
        "blocking_reasons": [] if len(usable) == 1 else ["NO_INSTALLED_INDEPENDENTLY_VERIFIED_NONLINEAR_SHRINKAGE_BACKEND"],
        "dependency_installation_performed": False,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def nonlinear_covariance_input(
    asset_ids: Sequence[str],
    observation_ids: Sequence[str],
    return_matrix: Sequence[Sequence[float]],
    *,
    annualisation_factor: float = 252.0,
    centring_convention: str = "demean_by_asset",
    sample_covariance_denominator: str = "n_minus_1",
    missingness_policy: str = "reject_any_missing",
    estimator_identity: str = "unselected_recognized_nonlinear_shrinkage",
    estimator_configuration: Mapping[str, Any] | None = None,
    decision_timestamp: str = "synthetic",
    allow_observations_fewer_than_assets: bool = True,
    minimum_observations: int = 2,
) -> dict[str, Any]:
    assets = [str(value) for value in asset_ids]
    observations = [str(value) for value in observation_ids]
    if not assets or len(set(assets)) != len(assets):
        raise NonlinearCovarianceInputError("INVALID_INPUT", "ASSET_IDENTITIES_INVALID")
    if assets != sorted(assets):
        raise NonlinearCovarianceInputError("INVALID_INPUT", "ASSETS_NOT_DETERMINISTICALLY_ORDERED")
    if not observations or len(set(observations)) != len(observations):
        raise NonlinearCovarianceInputError("INVALID_INPUT", "OBSERVATION_IDENTITIES_INVALID")
    if observations != sorted(observations):
        raise NonlinearCovarianceInputError("INVALID_INPUT", "OBSERVATIONS_NOT_DETERMINISTICALLY_ORDERED")
    matrix = np.asarray(return_matrix, dtype=float)
    if matrix.shape != (len(observations), len(assets)):
        raise NonlinearCovarianceInputError("INVALID_INPUT", "RETURN_MATRIX_DIMENSION_MISMATCH")
    if matrix.shape[0] < minimum_observations:
        raise NonlinearCovarianceInputError("INSUFFICIENT_DATA", "OBSERVATION_COUNT_INSUFFICIENT")
    if not np.isfinite(matrix).all():
        raise NonlinearCovarianceInputError("INVALID_INPUT", "RETURN_MATRIX_NON_FINITE")
    if matrix.shape[0] < matrix.shape[1] and not allow_observations_fewer_than_assets:
        raise NonlinearCovarianceInputError("UNSUPPORTED_CONFIGURATION", "HIGH_DIMENSIONAL_INPUT_DISABLED")
    if centring_convention != "demean_by_asset":
        raise NonlinearCovarianceInputError("UNSUPPORTED_CONFIGURATION", "CENTRING_CONVENTION_UNSUPPORTED")
    if sample_covariance_denominator != "n_minus_1":
        raise NonlinearCovarianceInputError("UNSUPPORTED_CONFIGURATION", "COVARIANCE_DENOMINATOR_UNSUPPORTED")
    if missingness_policy != "reject_any_missing":
        raise NonlinearCovarianceInputError("UNSUPPORTED_CONFIGURATION", "MISSINGNESS_POLICY_UNSUPPORTED")
    if not math.isfinite(annualisation_factor) or annualisation_factor <= 0:
        raise NonlinearCovarianceInputError("INVALID_INPUT", "ANNUALISATION_FACTOR_INVALID")
    # Compatible with Ticket 3A-A's asset-population identity.
    population_checksum = canonical_hash({"contract": "portfolio_risk_input_v1", "asset_ids": assets})
    history_checksum = canonical_hash({"observation_ids": observations, "returns": matrix.tolist()})
    config = {
        "annualisation_factor": annualisation_factor, "centring_convention": centring_convention,
        "sample_covariance_denominator": sample_covariance_denominator,
        "missingness_policy": missingness_policy, "estimator_identity": estimator_identity,
        "estimator_configuration": dict(estimator_configuration or {}),
        "decision_timestamp": decision_timestamp,
        "allow_observations_fewer_than_assets": allow_observations_fewer_than_assets,
    }
    return {
        "contract_version": INPUT_CONTRACT, "asset_ids": assets, "observation_ids": observations,
        "return_matrix": matrix.tolist(), **config, "population_checksum": population_checksum,
        "return_history_checksum": history_checksum, "configuration_checksum": canonical_hash(config),
        "observation_count": matrix.shape[0], "asset_count": matrix.shape[1],
        "high_dimensional": matrix.shape[0] < matrix.shape[1],
    }


def estimate_nonlinear_covariance(
    contract: Mapping[str, Any],
    *,
    backend: Any | None = None,
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        data = _validated_contract(contract)
        dependency = dict(audit or dependency_audit())
        if backend is None:
            status = "DEPENDENCY_UNAVAILABLE" if dependency.get("selected_backend") is None else "METHOD_NOT_VERIFIED"
            reason = "NO_VERIFIED_NONLINEAR_BACKEND_SELECTED" if status == "DEPENDENCY_UNAVAILABLE" else "SELECTED_BACKEND_NOT_BOUND_TO_VERIFIED_ADAPTER"
            return _blocked_covariance(data, status, reason, dependency)
        metadata_payload = getattr(backend, "method_metadata", None)
        required = {
            "estimator_id", "estimator_version", "published_method_family",
            "centring_convention", "sample_covariance_denominator",
            "eigenvalue_convention", "psd_guarantee", "dependency_identity",
            "dependency_version", "independently_verified",
        }
        if not isinstance(metadata_payload, Mapping) or not required.issubset(metadata_payload) or metadata_payload.get("independently_verified") is not True:
            return _blocked_covariance(data, "METHOD_NOT_VERIFIED", "BACKEND_METHOD_METADATA_NOT_VERIFIED", dependency)
        return _blocked_covariance(data, "METHOD_NOT_VERIFIED", "VERIFIED_BACKEND_ADAPTER_NOT_IMPLEMENTED_IN_TICKET", dependency)
    except NonlinearCovarianceInputError as exc:
        return _blocked_covariance(contract, exc.status, exc.reason, dict(audit or dependency_audit()))


def nonlinear_minimum_variance(
    contract: Mapping[str, Any],
    covariance_result: Mapping[str, Any],
    **_: Any,
) -> dict[str, Any]:
    try:
        data = _validated_contract(contract)
        if covariance_result.get("status") != "VALID" or covariance_result.get("valid") is not True:
            return _blocked_allocation(data, "METHOD_NOT_VERIFIED", "NONLINEAR_COVARIANCE_NOT_VALID")
        return _blocked_allocation(data, "METHOD_NOT_VERIFIED", "NONLINEAR_ALLOCATION_REQUIRES_VERIFIED_ESTIMATOR")
    except NonlinearCovarianceInputError as exc:
        return _blocked_allocation(contract, exc.status, exc.reason)


def compare_linear_and_nonlinear(
    contract: Mapping[str, Any],
    *,
    linear_result: Mapping[str, Any],
    nonlinear_result: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        data = _validated_contract(contract)
        reasons = []
        if linear_result.get("population_checksum") != data["population_checksum"]:
            reasons.append("LINEAR_POPULATION_MISMATCH")
        if linear_result.get("observation_population_checksum") != canonical_hash({"observation_ids": data["observation_ids"]}):
            reasons.append("LINEAR_OBSERVATION_POPULATION_MISMATCH")
        reasons.append("LINEAR_RETURN_HISTORY_IDENTITY_UNAVAILABLE")
        if nonlinear_result.get("population_checksum") != data["population_checksum"]:
            reasons.append("NONLINEAR_POPULATION_MISMATCH")
        if nonlinear_result.get("return_history_checksum") != data["return_history_checksum"]:
            reasons.append("NONLINEAR_RETURN_HISTORY_MISMATCH")
        if nonlinear_result.get("valid") is not True:
            reasons.append("NONLINEAR_ESTIMATOR_NOT_VALID")
        status = "VALID" if not reasons else ("INVALID_INPUT" if any("MISMATCH" in reason for reason in reasons) else "METHOD_NOT_VERIFIED")
        empirical = np.cov(np.asarray(data["return_matrix"]), rowvar=False, ddof=1)
        logical = {
            "contract_version": COMPARISON_CONTRACT, "status": status, "valid": not reasons,
            "blocking_reasons": sorted(set(reasons)), "warnings": ["NO_HISTORICAL_SUPERIORITY_CLAIM"],
            "population_checksum": data["population_checksum"], "return_history_checksum": data["return_history_checksum"],
            "empirical": _covariance_diagnostics(empirical),
            "linear": _safe_covariance_diagnostics(linear_result.get("covariance")),
            "nonlinear": _safe_covariance_diagnostics(nonlinear_result.get("nonlinear_covariance")),
            "minimum_variance_outputs_available": False,
            "historical_performance_computed": False, "superiority_claimed": False,
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}
    except NonlinearCovarianceInputError as exc:
        logical = {
            "contract_version": COMPARISON_CONTRACT, "status": exc.status, "valid": False,
            "blocking_reasons": [exc.reason], "warnings": [], "population_checksum": None,
            "return_history_checksum": None, "empirical": {}, "linear": {}, "nonlinear": {},
            "minimum_variance_outputs_available": False, "historical_performance_computed": False,
            "superiority_claimed": False,
        }
        logical["logical_result_checksum"] = canonical_hash(logical)
        return {**logical, "creation_metadata": _creation_metadata()}


def verify_nonlinear_result(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        if result.get("population_checksum") != data["population_checksum"]:
            reasons.append("POPULATION_MISMATCH")
        if result.get("return_history_checksum") != data["return_history_checksum"]:
            reasons.append("RETURN_HISTORY_MISMATCH")
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical):
            reasons.append("RESULT_CHECKSUM_MISMATCH")
        if result.get("valid") is True:
            covariance = np.asarray(result.get("nonlinear_covariance", ()), dtype=float)
            eigenvalues = np.asarray(result.get("shrunk_eigenvalues", ()), dtype=float)
            if covariance.shape != (data["asset_count"], data["asset_count"]):
                reasons.append("COVARIANCE_DIMENSION_MISMATCH")
            else:
                actual = np.linalg.eigvalsh(covariance)
                if not np.allclose(covariance, covariance.T, atol=PSD_TOLERANCE, rtol=0):
                    reasons.append("COVARIANCE_NOT_SYMMETRIC")
                if actual.min() < -PSD_TOLERANCE:
                    reasons.append("COVARIANCE_NOT_PSD")
                if eigenvalues.shape != actual.shape or not np.allclose(eigenvalues, actual):
                    reasons.append("EIGENVALUE_CONSISTENCY_FAILED")
                if result.get("covariance_checksum") != canonical_hash(covariance.tolist()):
                    reasons.append("COVARIANCE_CHECKSUM_MISMATCH")
                if abs(float(result.get("trace_after", math.inf)) - float(np.trace(covariance))) > 1e-8:
                    reasons.append("TRACE_CONSISTENCY_FAILED")
        return {"contract_version": "nonlinear_covariance_verification_v1", "valid": not reasons, "blocking_reasons": sorted(set(reasons))}
    except NonlinearCovarianceInputError as exc:
        return {"contract_version": "nonlinear_covariance_verification_v1", "valid": False, "blocking_reasons": [exc.reason]}


def verify_nonlinear_allocation(contract: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    reasons = []
    try:
        data = _validated_contract(contract)
        if result.get("population_checksum") != data["population_checksum"]:
            reasons.append("POPULATION_MISMATCH")
        if result.get("return_history_checksum") != data["return_history_checksum"]:
            reasons.append("RETURN_HISTORY_MISMATCH")
        logical = {key: value for key, value in result.items() if key not in {"creation_metadata", "logical_result_checksum"}}
        if result.get("logical_result_checksum") != canonical_hash(logical):
            reasons.append("RESULT_CHECKSUM_MISMATCH")
        if result.get("valid") is True:
            weights = np.asarray(result.get("target_weights", ()), dtype=float)
            if weights.shape != (data["asset_count"],) or not np.isfinite(weights).all():
                reasons.append("TARGET_WEIGHTS_INVALID")
        return {"contract_version": "nonlinear_minimum_variance_verification_v1", "valid": not reasons, "blocking_reasons": sorted(set(reasons))}
    except NonlinearCovarianceInputError as exc:
        return {"contract_version": "nonlinear_minimum_variance_verification_v1", "valid": False, "blocking_reasons": [exc.reason]}


def _blocked_covariance(data, status, reason, dependency):
    matrix = np.asarray(data.get("return_matrix", ()), dtype=float) if isinstance(data, Mapping) else np.asarray([])
    empirical = np.cov(matrix, rowvar=False, ddof=1) if matrix.ndim == 2 and matrix.shape[0] >= 2 else np.asarray([])
    empirical_values = np.linalg.eigvalsh(empirical) if empirical.ndim == 2 and empirical.size else np.asarray([])
    logical = {
        "contract_version": RESULT_CONTRACT, "status": status if status in STATUSES else "METHOD_NOT_VERIFIED",
        "valid": False, "blocking_reasons": [reason], "warnings": ["NO_NONLINEAR_COVARIANCE_PRODUCED"],
        "estimator_id": None, "estimator_version": None, "published_method_family": None,
        "dependency_identity": dependency.get("selected_backend"), "dependency_version": None,
        "empirical_covariance": empirical.tolist(), "empirical_eigenvalues": empirical_values.tolist(),
        "shrunk_eigenvalues": [], "nonlinear_covariance": [], "precision_matrix": None,
        "condition_number_before": _condition_number(empirical_values), "condition_number_after": None,
        "minimum_eigenvalue": None, "maximum_eigenvalue": None, "effective_rank": None,
        "trace_before": float(np.trace(empirical)) if empirical.size else None, "trace_after": None,
        "psd_validated": False, "symmetry_residual": None,
        "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "return_history_checksum": data.get("return_history_checksum") if isinstance(data, Mapping) else None,
        "configuration_checksum": data.get("configuration_checksum") if isinstance(data, Mapping) else None,
        "covariance_checksum": None, "dependency_audit_checksum": dependency.get("logical_result_checksum"),
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _blocked_allocation(data, status, reason):
    logical = {
        "contract_version": ALLOCATION_CONTRACT, "status": status if status in STATUSES else "METHOD_NOT_VERIFIED",
        "valid": False, "blocking_reasons": [reason], "warnings": [],
        "asset_ids": list(data.get("asset_ids", ())) if isinstance(data, Mapping) else [],
        "target_weights": [], "expected_variance": None, "expected_volatility": None,
        "concentration": {}, "sector_exposures": {}, "turnover": None,
        "solver_identity": "scipy.optimize.SLSQP", "solver_status": "not_run",
        "solver_tolerance": 1e-10, "constraint_tolerance": 1e-7, "maximum_iterations": 2000,
        "population_checksum": data.get("population_checksum") if isinstance(data, Mapping) else None,
        "return_history_checksum": data.get("return_history_checksum") if isinstance(data, Mapping) else None,
    }
    logical["logical_result_checksum"] = canonical_hash(logical)
    return {**logical, "creation_metadata": _creation_metadata()}


def _validated_contract(contract):
    return nonlinear_covariance_input(
        contract["asset_ids"], contract["observation_ids"], contract["return_matrix"],
        annualisation_factor=float(contract.get("annualisation_factor", 252)),
        centring_convention=str(contract.get("centring_convention", "demean_by_asset")),
        sample_covariance_denominator=str(contract.get("sample_covariance_denominator", "n_minus_1")),
        missingness_policy=str(contract.get("missingness_policy", "reject_any_missing")),
        estimator_identity=str(contract.get("estimator_identity", "unselected_recognized_nonlinear_shrinkage")),
        estimator_configuration=contract.get("estimator_configuration"),
        decision_timestamp=str(contract.get("decision_timestamp", "synthetic")),
        allow_observations_fewer_than_assets=bool(contract.get("allow_observations_fewer_than_assets", True)),
    )


def _covariance_diagnostics(covariance):
    values = np.linalg.eigvalsh(covariance)
    positive = values[values > PSD_TOLERANCE]
    probabilities = positive / positive.sum() if positive.size and positive.sum() > 0 else np.asarray([])
    effective_rank = float(math.exp(-np.sum(probabilities * np.log(probabilities)))) if probabilities.size else 0.0
    return {
        "condition_number": _condition_number(values), "minimum_eigenvalue": float(values.min()),
        "maximum_eigenvalue": float(values.max()), "effective_rank": effective_rank,
        "trace": float(np.trace(covariance)), "covariance_checksum": canonical_hash(covariance.tolist()),
    }


def _safe_covariance_diagnostics(value):
    try:
        covariance = np.asarray(value, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not covariance.size:
            return {"available": False}
        return {"available": True, **_covariance_diagnostics(covariance)}
    except (TypeError, ValueError):
        return {"available": False}


def _condition_number(values):
    if values.size == 0: return None
    positive = values[values > PSD_TOLERANCE]
    return float(positive.max() / positive.min()) if positive.size else None


def _version(distribution):
    try: return metadata.version(distribution)
    except metadata.PackageNotFoundError: return None


def _creation_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version()}
