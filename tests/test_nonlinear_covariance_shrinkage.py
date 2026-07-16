from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from core.research.ml.nonlinear_covariance_shrinkage import (
    NonlinearCovarianceInputError,
    canonical_json,
    compare_linear_and_nonlinear,
    dependency_audit,
    estimate_nonlinear_covariance,
    nonlinear_covariance_input,
    nonlinear_minimum_variance,
    verify_nonlinear_allocation,
    verify_nonlinear_result,
)
from core.research.ml.portfolio_risk_baselines import (
    estimate_linear_shrinkage,
    portfolio_risk_input,
)


def _returns():
    return [
        [-0.02, -0.04, -0.01],
        [-0.01, 0.00, 0.01],
        [0.00, 0.04, -0.01],
        [0.01, 0.00, 0.01],
        [0.02, -0.04, -0.01],
        [0.00, 0.04, 0.01],
    ]


def _contract(matrix=None, *, assets=None, observations=None, **kwargs):
    assets = assets or ["A", "B", "C"]
    matrix = matrix if matrix is not None else _returns()
    observations = observations or [f"2024-01-0{i}" for i in range(1, len(matrix) + 1)]
    return nonlinear_covariance_input(assets, observations, matrix, **kwargs)


def _logical(result):
    return {key: value for key, value in result.items() if key != "creation_metadata"}


def test_dependency_audit_selects_no_unverified_backend():
    audit = dependency_audit()
    assert audit["status"] == "BLOCKED_NO_VERIFIED_IMPLEMENTATION"
    assert audit["selected_backend"] is None
    assert not audit["dependency_installation_performed"]
    rows = {row["module"]: row for row in audit["candidates"]}
    assert rows["sklearn"]["available"]
    assert "linear shrinkage only" in rows["sklearn"]["candidate_method"]


def test_deterministic_blocked_estimation_empirical_diagnostics_and_checksum():
    contract = _contract()
    first = estimate_nonlinear_covariance(contract)
    second = estimate_nonlinear_covariance(contract)
    assert first["status"] == "DEPENDENCY_UNAVAILABLE"
    assert not first["valid"]
    assert first["nonlinear_covariance"] == []
    assert np.asarray(first["empirical_covariance"]).shape == (3, 3)
    assert first["empirical_eigenvalues"] == sorted(first["empirical_eigenvalues"])
    assert _logical(first) == _logical(second)
    assert first["logical_result_checksum"] == second["logical_result_checksum"]


def test_known_diagonal_high_correlation_near_singular_and_duplicates_fail_closed_safely():
    diagonal = _contract(matrix=[[-1, -2], [0, 0], [1, 2]], assets=["A", "B"])
    correlated = _contract(matrix=[[i, i * 1.001, -i] for i in range(1, 7)])
    duplicated = _contract(matrix=[[i, i, -i] for i in range(1, 7)])
    constant = _contract(matrix=[[1, 1, 1] for _ in range(6)])
    for contract in (diagonal, correlated, duplicated, constant):
        result = estimate_nonlinear_covariance(contract)
        assert result["status"] == "DEPENDENCY_UNAVAILABLE"
        assert result["nonlinear_covariance"] == []
        assert result["covariance_checksum"] is None


def test_high_dimensional_input_is_explicitly_supported_or_blocked_by_policy():
    matrix = [[float(row + column) for column in range(5)] for row in range(3)]
    supported = _contract(matrix=matrix, assets=["A", "B", "C", "D", "E"])
    assert supported["high_dimensional"] is True
    assert estimate_nonlinear_covariance(supported)["status"] == "DEPENDENCY_UNAVAILABLE"
    with pytest.raises(NonlinearCovarianceInputError, match="HIGH_DIMENSIONAL"):
        _contract(matrix=matrix, assets=["A", "B", "C", "D", "E"], allow_observations_fewer_than_assets=False)


def test_nonfinite_inadequate_and_ordering_inputs_rejected():
    with pytest.raises(NonlinearCovarianceInputError, match="NON_FINITE"):
        _contract(matrix=[[0, math.nan], [1, 1]], assets=["A", "B"])
    with pytest.raises(NonlinearCovarianceInputError, match="INSUFFICIENT"):
        _contract(matrix=[[0, 1]], assets=["A", "B"], minimum_observations=2)
    with pytest.raises(NonlinearCovarianceInputError, match="DETERMINISTICALLY_ORDERED"):
        _contract(assets=["B", "A", "C"])


class _UnverifiedBackend:
    method_metadata = {
        "estimator_id": "ad_hoc",
        "estimator_version": "1",
        "published_method_family": "none",
        "centring_convention": "demean_by_asset",
        "sample_covariance_denominator": "n_minus_1",
        "eigenvalue_convention": "unknown",
        "psd_guarantee": False,
        "dependency_identity": "fixture",
        "dependency_version": "1",
        "independently_verified": False,
    }


def test_method_not_verified_backend_fails_closed():
    audit = dependency_audit(availability_overrides={"nonlinshrink": True})
    result = estimate_nonlinear_covariance(_contract(), backend=_UnverifiedBackend(), audit=audit)
    assert result["status"] == "METHOD_NOT_VERIFIED"
    assert "BACKEND_METHOD_METADATA_NOT_VERIFIED" in result["blocking_reasons"]


def test_dependency_unavailable_controlled_fixture():
    audit = dependency_audit(availability_overrides={
        "nonlinshrink": False, "riskfolio": False, "pypfopt": False,
        "statsmodels": False, "quest": False,
    })
    result = estimate_nonlinear_covariance(_contract(), audit=audit)
    assert result["status"] == "DEPENDENCY_UNAVAILABLE"


def test_nonlinear_allocation_is_not_run_without_covariance():
    contract = _contract()
    covariance = estimate_nonlinear_covariance(contract)
    allocation = nonlinear_minimum_variance(contract, covariance)
    assert allocation["status"] == "METHOD_NOT_VERIFIED"
    assert allocation["target_weights"] == []
    assert allocation["solver_status"] == "not_run"
    assert verify_nonlinear_allocation(contract, allocation)["valid"]


def test_result_verifier_detects_changed_covariance_and_eigenvalues():
    contract = _contract()
    result = estimate_nonlinear_covariance(contract)
    assert verify_nonlinear_result(contract, result)["valid"]
    changed_covariance = copy.deepcopy(result)
    changed_covariance["nonlinear_covariance"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert "RESULT_CHECKSUM_MISMATCH" in verify_nonlinear_result(contract, changed_covariance)["blocking_reasons"]
    changed_eigenvalues = copy.deepcopy(result)
    changed_eigenvalues["shrunk_eigenvalues"] = [1, 1, 1]
    assert "RESULT_CHECKSUM_MISMATCH" in verify_nonlinear_result(contract, changed_eigenvalues)["blocking_reasons"]


def test_allocation_verifier_detects_changed_weights():
    contract = _contract()
    allocation = nonlinear_minimum_variance(contract, estimate_nonlinear_covariance(contract))
    changed = copy.deepcopy(allocation)
    changed["target_weights"] = [1, 0, 0]
    assert "RESULT_CHECKSUM_MISMATCH" in verify_nonlinear_allocation(contract, changed)["blocking_reasons"]


def test_linear_comparison_is_blocked_without_exact_history_and_nonlinear_result():
    nonlinear_contract = _contract()
    linear_contract = portfolio_risk_input(
        ["A", "B", "C"], return_history=_returns(),
        observation_ids=[f"2024-01-0{i}" for i in range(1, 7)],
    )
    linear = estimate_linear_shrinkage(linear_contract)
    nonlinear = estimate_nonlinear_covariance(nonlinear_contract)
    comparison = compare_linear_and_nonlinear(nonlinear_contract, linear_result=linear, nonlinear_result=nonlinear)
    assert not comparison["valid"]
    assert comparison["status"] == "METHOD_NOT_VERIFIED"
    assert "LINEAR_RETURN_HISTORY_IDENTITY_UNAVAILABLE" in comparison["blocking_reasons"]
    assert "NONLINEAR_ESTIMATOR_NOT_VALID" in comparison["blocking_reasons"]
    assert comparison["linear"]["available"]
    assert not comparison["nonlinear"]["available"]


def test_mismatched_population_comparison_is_rejected():
    contract = _contract()
    linear_contract = portfolio_risk_input(
        ["A", "B", "C"], return_history=_returns(),
        observation_ids=[f"2024-01-0{i}" for i in range(1, 7)],
    )
    linear = estimate_linear_shrinkage(linear_contract)
    linear["population_checksum"] = "other"
    comparison = compare_linear_and_nonlinear(contract, linear_result=linear, nonlinear_result=estimate_nonlinear_covariance(contract))
    assert comparison["status"] == "INVALID_INPUT"
    assert "LINEAR_POPULATION_MISMATCH" in comparison["blocking_reasons"]


def test_json_ordering_and_checksum_stability():
    result = estimate_nonlinear_covariance(_contract())
    encoded = canonical_json(_logical(result))
    assert encoded == canonical_json(json.loads(encoded))
    assert result["logical_result_checksum"] == estimate_nonlinear_covariance(_contract())["logical_result_checksum"]
