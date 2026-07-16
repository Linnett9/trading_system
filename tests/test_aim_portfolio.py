from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.research.ml.aim_portfolio import (
    AimPortfolioInputError,
    aim_portfolio_input,
    canonical_json,
    comparison_controls,
    objective_components,
    optimise_aim_portfolio,
    verify_aim_portfolio,
)


def _contract(**overrides):
    values = {
        "asset_ids": ["A", "B", "C"],
        "expected_alpha": [0.03, 0.02, 0.01],
        "covariance": np.diag([0.04, 0.04, 0.04]).tolist(),
        "previous_weights": [1 / 3, 1 / 3, 1 / 3],
        "exposure_target": 1.0,
        "maximum_weights": 1.0,
        "sector_ids": ["S1", "S1", "S2"],
        "sector_caps": {"S1": 1.0, "S2": 1.0},
        "risk_aversion": 1.0,
    }
    values.update(overrides)
    return aim_portfolio_input(**values)


def _logical(result):
    return {key: value for key, value in result.items() if key != "creation_metadata"}


def test_deterministic_repeated_solution_and_json_identity():
    contract = _contract()
    first = optimise_aim_portfolio(contract)
    second = optimise_aim_portfolio(contract)
    assert first["status"] == "OPTIMAL"
    assert first["target_weights"] == pytest.approx(second["target_weights"], abs=1e-10)
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    encoded = canonical_json(_logical(first))
    assert encoded == canonical_json(json.loads(encoded))


def test_zero_alpha_equal_risk_is_equal_weight_and_tie_is_reported():
    result = optimise_aim_portfolio(_contract(expected_alpha=[0, 0, 0], previous_weights=[0, 0, 0]))
    assert result["target_weights"] == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert "NON_UNIQUE_OPTIMUM_POSSIBLE_STABLE_ASSET_ORDER_USED" in result["warnings"]


def test_dominant_alpha_and_risk_aversion_reduces_concentration():
    loose = optimise_aim_portfolio(_contract(expected_alpha=[1, 0, 0], risk_aversion=0, previous_weights=[0, 0, 0]))
    risk_aware = optimise_aim_portfolio(_contract(expected_alpha=[1, 0, 0], risk_aversion=100, previous_weights=[0, 0, 0]))
    assert loose["target_weights"][0] > 0.99
    assert risk_aware["target_weights"][0] < loose["target_weights"][0]


def test_l1_and_l2_penalties_reduce_and_smooth_turnover():
    base = _contract(expected_alpha=[0.2, 0, 0], previous_weights=[0, 0.5, 0.5], risk_aversion=1)
    unpenalised = optimise_aim_portfolio(base)
    l1 = optimise_aim_portfolio(_contract(expected_alpha=[0.2, 0, 0], previous_weights=[0, 0.5, 0.5], risk_aversion=1, l1_turnover_penalty=0.2))
    l2 = optimise_aim_portfolio(_contract(expected_alpha=[0.2, 0, 0], previous_weights=[0, 0.5, 0.5], risk_aversion=1, l2_turnover_penalty=1.0))
    assert l1["gross_turnover"] < unpenalised["gross_turnover"]
    assert l2["gross_turnover"] < unpenalised["gross_turnover"]
    assert l2["target_weights"][0] < unpenalised["target_weights"][0]
    zero = optimise_aim_portfolio(_contract(expected_alpha=[0.2, 0, 0], previous_weights=[0, 0.5, 0.5], risk_aversion=1, l1_turnover_penalty=0, l2_turnover_penalty=0))
    assert zero["target_weights"] == pytest.approx(unpenalised["target_weights"], abs=1e-8)


def test_exposure_stock_and_sector_caps_are_enforced():
    stock = optimise_aim_portfolio(_contract(expected_alpha=[1, 0, 0], maximum_weights=0.4, risk_aversion=0, previous_weights=[0, 0, 0]))
    assert sum(stock["target_weights"]) == pytest.approx(1)
    assert max(stock["target_weights"]) <= 0.4 + 1e-7
    sector = optimise_aim_portfolio(_contract(expected_alpha=[1, 0.9, 0], sector_caps={"S1": 0.5, "S2": 1.0}, risk_aversion=0, previous_weights=[0, 0, 0]))
    assert sum(sector["target_weights"][:2]) <= 0.5 + 1e-7
    assert sector["sector_exposures"]["S1"] == pytest.approx(0.5, abs=1e-6)


def test_liquidity_exclusion_and_retained_ineligible_policy():
    liquidate = optimise_aim_portfolio(_contract(liquidity_eligible=[False, True, True]))
    assert liquidate["target_weights"][0] == 0
    retained = optimise_aim_portfolio(_contract(
        liquidity_eligible=[False, True, True], ineligible_asset_policy="retain_only",
        previous_weights=[0.2, 0.4, 0.4], expected_alpha=[1, 0, 0],
    ))
    assert retained["target_weights"][0] <= 0.2 + 1e-7
    assert "A" in retained["liquidity_exclusions"]


def test_turnover_limit_uses_gross_weight_change():
    result = optimise_aim_portfolio(_contract(
        expected_alpha=[1, 0, 0], previous_weights=[0, 0.5, 0.5],
        risk_aversion=0, turnover_limit=0.4,
    ))
    assert result["status"] == "OPTIMAL"
    assert result["gross_turnover"] <= 0.4 + 1e-6
    assert result["one_way_turnover"] == pytest.approx(result["gross_turnover"] / 2)


def test_infeasible_stock_and_sector_caps_fail_closed():
    with pytest.raises(AimPortfolioInputError, match="STOCK_CAPS"):
        _contract(maximum_weights=0.2)
    with pytest.raises(AimPortfolioInputError, match="SECTOR_CAPS"):
        _contract(sector_caps={"S1": 0.2, "S2": 0.2})


def test_covariance_shape_symmetry_psd_and_near_psd_tolerance():
    with pytest.raises(AimPortfolioInputError, match="DIMENSION"):
        _contract(covariance=[[1, 0], [0, 1]])
    with pytest.raises(AimPortfolioInputError, match="SYMMETRIC"):
        _contract(covariance=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(AimPortfolioInputError, match="POSITIVE_SEMIDEFINITE"):
        _contract(covariance=[[1, 2, 0], [2, 1, 0], [0, 0, 1]])
    near = _contract(covariance=[[-1e-9, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = optimise_aim_portfolio(near)
    assert result["status"] == "OPTIMAL"
    assert "NEAR_PSD_COVARIANCE_PROJECTED_FOR_SOLVER" in result["warnings"]


def test_nonfinite_and_invalid_previous_weights_rejected():
    with pytest.raises(AimPortfolioInputError, match="NON_FINITE"):
        _contract(expected_alpha=[math.nan, 0, 0])
    with pytest.raises(AimPortfolioInputError, match="PREVIOUS_WEIGHTS_NEGATIVE"):
        _contract(previous_weights=[-0.1, 0.5, 0.6])


def test_independent_verifier_and_modified_weight_detection():
    contract = _contract()
    result = optimise_aim_portfolio(contract)
    verified = verify_aim_portfolio(contract, result)
    assert verified["valid"]
    modified = dict(result)
    modified["target_weights"] = [0.9, 0.1, 0]
    rejected = verify_aim_portfolio(contract, modified)
    assert not rejected["valid"]
    assert "OBJECTIVE_RECOMPUTATION_MISMATCH" in rejected["blocking_reasons"]
    assert "RESULT_CHECKSUM_MISMATCH" in rejected["blocking_reasons"]


def test_objective_components_are_exact_and_not_transaction_costs():
    contract = _contract(l1_turnover_penalty=0.1, l2_turnover_penalty=0.2)
    weights = [0.5, 0.25, 0.25]
    components = objective_components(contract, weights)
    expected_alpha = 0.5 * 0.03 + 0.25 * 0.02 + 0.25 * 0.01
    assert components["expected_alpha_contribution"] == pytest.approx(expected_alpha)
    assert components["gross_objective_value"] == pytest.approx(
        components["expected_alpha_contribution"]
        - components["covariance_risk_penalty"]
        - components["l1_turnover_penalty"]
        - components["l2_turnover_penalty"]
    )


def test_comparison_controls_are_ex_ante_only():
    controls = comparison_controls(_contract(), top_k=2)
    assert not controls["historical_performance_computed"]
    assert not controls["superiority_claimed"]
    assert set(controls["controls"]) == {
        "equal_weight_top_k", "inverse_volatility_top_k", "unchanged_previous_holdings",
    }
    assert all("expected_alpha" in row and "objective_components" in row for row in controls["controls"].values())
