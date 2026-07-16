from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from core.research.ml.portfolio_risk_baselines import (
    PortfolioRiskInputError,
    allocate_inverse_volatility,
    allocate_minimum_variance,
    canonical_json,
    compare_risk_policies,
    estimate_linear_shrinkage,
    portfolio_risk_input,
    verify_allocation,
    verify_shrinkage_result,
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


def _contract(**overrides):
    values = {
        "asset_ids": ["A", "B", "C"],
        "return_history": _returns(),
        "observation_ids": [f"2024-01-0{i}" for i in range(1, 7)],
        "previous_weights": [1 / 3, 1 / 3, 1 / 3],
        "maximum_weights": 1.0,
        "sector_ids": ["S1", "S1", "S2"],
        "sector_caps": {"S1": 1.0, "S2": 1.0},
        "annualisation_factor": 1.0,
    }
    values.update(overrides)
    return portfolio_risk_input(**values)


def _logical(result):
    return {key: value for key, value in result.items() if key != "creation_metadata"}


def test_inverse_volatility_known_weights_and_exposure():
    contract = portfolio_risk_input(
        ["A", "B", "C"], supplied_volatilities=[1, 2, 4],
        previous_weights=[0, 0, 0], annualisation_factor=1,
    )
    result = allocate_inverse_volatility(contract)
    expected = np.asarray([1, 0.5, 0.25]); expected /= expected.sum()
    assert result["valid"]
    assert result["target_weights"] == pytest.approx(expected)
    assert result["exposure"] == pytest.approx(1)


def test_volatility_floor_zero_handling_and_deterministic_ties():
    contract = portfolio_risk_input(
        ["A", "B", "C"], supplied_volatilities=[0, 0, 1],
        previous_weights=[0, 0, 0], annualisation_factor=1,
    )
    first = allocate_inverse_volatility(contract, minimum_volatility=0.1)
    second = allocate_inverse_volatility(contract, minimum_volatility=0.1)
    assert first["target_weights"] == pytest.approx(second["target_weights"])
    assert first["target_weights"][0] == pytest.approx(first["target_weights"][1])
    assert "VOLATILITY_FLOOR_APPLIED" in first["warnings"]


def test_inverse_volatility_stock_cap_redistribution_sector_and_liquidity():
    contract = portfolio_risk_input(
        ["A", "B", "C"], supplied_volatilities=[0.1, 1, 1],
        previous_weights=[0, 0, 0], maximum_weights=[0.4, 1, 1],
        sector_ids=["S1", "S1", "S2"], sector_caps={"S1": 0.5, "S2": 1},
        annualisation_factor=1,
    )
    result = allocate_inverse_volatility(contract)
    assert result["target_weights"][0] <= 0.4 + 1e-7
    assert sum(result["target_weights"][:2]) <= 0.5 + 1e-7
    excluded = allocate_inverse_volatility(portfolio_risk_input(
        ["A", "B", "C"], supplied_volatilities=[0.1, 1, 1],
        previous_weights=[0, 0, 0], liquidity_eligible=[False, True, True],
        annualisation_factor=1,
    ))
    assert excluded["target_weights"][0] == 0
    assert "A" in excluded["liquidity_exclusions"]


def test_infeasible_caps_fail_closed():
    with pytest.raises(PortfolioRiskInputError, match="STOCK_CAPS"):
        _contract(maximum_weights=0.2)
    with pytest.raises(PortfolioRiskInputError, match="SECTOR_CAPS"):
        _contract(sector_caps={"S1": 0.2, "S2": 0.2})


def test_sample_covariance_target_intensity_symmetry_and_psd():
    contract = _contract()
    result = estimate_linear_shrinkage(contract)
    history = np.asarray(_returns())
    sample = np.cov(history, rowvar=False, ddof=0)
    target = np.eye(3) * np.trace(sample) / 3
    assert result["valid"]
    assert result["sample_covariance"] == pytest.approx(sample)
    assert result["shrinkage_target"] == pytest.approx(target)
    assert 0 <= result["shrinkage_intensity"] <= 1
    covariance = np.asarray(result["covariance"])
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-8


def test_inadequate_and_nonfinite_returns_rejected():
    with pytest.raises(PortfolioRiskInputError, match="INSUFFICIENT"):
        portfolio_risk_input(["A", "B"], return_history=[[0, 0]], observation_ids=["d"], minimum_observations=2)
    with pytest.raises(PortfolioRiskInputError, match="NON_FINITE"):
        portfolio_risk_input(["A", "B"], return_history=[[0, math.nan], [1, 1]], observation_ids=["a", "b"])


def test_minimum_variance_prefers_lower_variance_asset():
    contract = _contract()
    shrinkage = estimate_linear_shrinkage(contract)
    result = allocate_minimum_variance(contract, shrinkage)
    assert result["status"] == "OPTIMAL"
    # B has the largest variance in the fixture.
    assert result["target_weights"][1] < result["target_weights"][0]
    assert result["target_weights"][1] < result["target_weights"][2]
    assert result["risk_contribution_sum"] == pytest.approx(result["expected_variance"])


def test_minimum_variance_stock_and_sector_caps():
    stock_contract = _contract(maximum_weights=0.4)
    stock = allocate_minimum_variance(stock_contract, estimate_linear_shrinkage(stock_contract))
    assert max(stock["target_weights"]) <= 0.4 + 1e-7
    sector_contract = _contract(sector_caps={"S1": 0.5, "S2": 1})
    sector = allocate_minimum_variance(sector_contract, estimate_linear_shrinkage(sector_contract))
    assert sum(sector["target_weights"][:2]) <= 0.5 + 1e-7


def test_solver_configuration_failure_state():
    contract = _contract()
    result = allocate_minimum_variance(contract, estimate_linear_shrinkage(contract), max_iterations=0)
    assert result["status"] == "UNSUPPORTED_CONFIGURATION"
    assert not result["valid"]


def test_independent_verification_and_modified_weights():
    contract = _contract()
    inverse = allocate_inverse_volatility(contract)
    assert verify_allocation(contract, inverse)["valid"]
    modified = copy.deepcopy(inverse)
    modified["target_weights"][0] += 0.1
    verification = verify_allocation(contract, modified)
    assert not verification["valid"]
    assert "EXPOSURE_FAILED" in verification["blocking_reasons"]
    assert "RESULT_CHECKSUM_MISMATCH" in verification["blocking_reasons"]


def test_shrinkage_verifier_detects_changed_covariance():
    contract = _contract()
    shrinkage = estimate_linear_shrinkage(contract)
    assert verify_shrinkage_result(contract, shrinkage)["valid"]
    modified = copy.deepcopy(shrinkage)
    modified["covariance"][0][0] += 0.1
    verification = verify_shrinkage_result(contract, modified)
    assert not verification["valid"]
    assert "COVARIANCE_CHECKSUM_MISMATCH" in verification["blocking_reasons"]


def test_stable_json_and_checksums():
    contract = _contract()
    first = allocate_inverse_volatility(contract)
    second = allocate_inverse_volatility(contract)
    assert _logical(first) == _logical(second)
    encoded = canonical_json(_logical(first))
    assert encoded == canonical_json(json.loads(encoded))
    assert first["logical_result_checksum"] == second["logical_result_checksum"]


def test_exact_population_comparison_and_mismatch_rejection():
    contract = _contract()
    inverse = allocate_inverse_volatility(contract)
    minimum = allocate_minimum_variance(contract, estimate_linear_shrinkage(contract))
    comparison = compare_risk_policies(contract, {"inverse": inverse, "minimum": minimum})
    assert comparison["valid"] and comparison["policy_count"] == 2
    assert not comparison["historical_returns_computed"]
    changed = copy.deepcopy(inverse); changed["population_checksum"] = "other"
    with pytest.raises(PortfolioRiskInputError, match="MISMATCH"):
        compare_risk_policies(contract, {"changed": changed})
