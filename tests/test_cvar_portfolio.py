from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from core.research.ml.cvar_portfolio import (
    CVaRInputError,
    canonical_json,
    compare_cvar_policies,
    cvar_input,
    discrete_cvar,
    evaluate_confidence_panel,
    optimise_cvar_portfolio,
    verify_cvar_result,
)


def _returns() -> list[list[float]]:
    return [
        [0.002, -0.35 if index == 0 else 0.012, 0.004 if index % 2 == 0 else 0.002]
        for index in range(20)
    ]


def _contract(**overrides):
    values = {
        "asset_ids": ["A", "B", "C"],
        "scenario_ids": [f"S{index:02d}" for index in range(20)],
        "scenario_returns": _returns(),
        "expected_alpha": [0.001, 0.02, 0.004],
        "previous_weights": [1 / 3, 1 / 3, 1 / 3],
        "maximum_weights": 0.8,
        "cvar_risk_aversion": 0.2,
    }
    values.update(overrides)
    return cvar_input(**values)


def test_known_equal_probability_discrete_cvar():
    result = discrete_cvar(range(20), [0.05] * 20, 0.95)
    assert result["var_threshold"] == 18
    assert result["cvar"] == pytest.approx(19)


def test_known_weighted_discrete_cvar():
    result = discrete_cvar([0, 1, 10], [0.8, 0.15, 0.05], 0.95)
    assert result["var_threshold"] == 1
    assert result["cvar"] == pytest.approx(10)


@pytest.mark.parametrize("confidence", [0.95, 0.975])
def test_registered_confidence_levels_are_optimal_and_deterministic(confidence):
    contract = _contract(confidence_level=confidence)
    first = optimise_cvar_portfolio(contract)
    second = optimise_cvar_portfolio(contract)
    assert first["status"] == "OPTIMAL"
    assert first["target_weights"] == pytest.approx(second["target_weights"], abs=1e-10)
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    assert verify_cvar_result(contract, first)["valid"]


@pytest.mark.parametrize(
    ("confidence", "status"),
    [(0.0, "INVALID_INPUT"), (1.0, "INVALID_INPUT"), (0.9, "UNSUPPORTED_CONFIGURATION")],
)
def test_confidence_level_validation(confidence, status):
    result = optimise_cvar_portfolio({**_contract(), "confidence_level": confidence})
    assert result["status"] == status


@pytest.mark.parametrize(
    ("probabilities", "reason"),
    [([0.04] * 20, "SCENARIO_PROBABILITIES_NOT_NORMALISED"), ([-0.05] + [1.05 / 19] * 19, "SCENARIO_PROBABILITY_NON_POSITIVE")],
)
def test_probability_validation(probabilities, reason):
    with pytest.raises(CVaRInputError, match=reason):
        _contract(scenario_probabilities=probabilities)


def test_nonfinite_and_inadequate_scenarios_fail_closed():
    returns = _returns()
    returns[0][0] = np.nan
    with pytest.raises(CVaRInputError, match="SCENARIO_RETURN_NON_FINITE"):
        _contract(scenario_returns=returns)
    with pytest.raises(CVaRInputError, match="SCENARIO_COUNT_INSUFFICIENT"):
        cvar_input(["A"], ["S00"], [[0.0]], [0.0], [1.0])


def test_exposure_caps_sectors_liquidity_and_turnover_are_enforced():
    contract = _contract(
        previous_weights=[0.5, 0.0, 0.5],
        exposure_target=0.9,
        maximum_weights=[0.5, 0.5, 0.5],
        sector_ids=["X", "X", "Y"],
        sector_caps={"X": 0.5, "Y": 0.5},
        liquidity_eligible=[True, False, True],
        turnover_limit=0.2,
    )
    result = optimise_cvar_portfolio(contract)
    assert result["status"] == "OPTIMAL"
    assert sum(result["target_weights"]) == pytest.approx(0.9)
    assert result["target_weights"][1] == 0
    assert max(result["target_weights"]) <= 0.5 + 1e-7
    assert result["sector_exposures"]["X"] <= 0.5 + 1e-7
    assert result["gross_turnover"] <= 0.2 + 1e-7


def test_infeasible_caps_fail_closed():
    with pytest.raises(CVaRInputError, match="STOCK_CAPS_CANNOT_MEET_EXPOSURE"):
        _contract(maximum_weights=0.2)


def test_stronger_tail_penalty_reduces_tail_exposed_asset():
    weak = optimise_cvar_portfolio(_contract(cvar_risk_aversion=0.001))
    strong = optimise_cvar_portfolio(_contract(cvar_risk_aversion=1.0))
    assert weak["target_weights"][1] > strong["target_weights"][1]


def test_turnover_penalties_reduce_or_smooth_changes():
    base = optimise_cvar_portfolio(_contract(cvar_risk_aversion=0.05))
    l1 = optimise_cvar_portfolio(_contract(cvar_risk_aversion=0.05, l1_turnover_penalty=0.05))
    l2 = optimise_cvar_portfolio(_contract(cvar_risk_aversion=0.05, l2_turnover_penalty=0.5))
    assert l1["gross_turnover"] <= base["gross_turnover"] + 1e-7
    assert sum(np.square(l2["trade_weight_changes"])) <= sum(np.square(base["trade_weight_changes"])) + 1e-7
    assert base["l1_turnover_penalty"] == 0
    assert base["l2_turnover_penalty"] == 0


def test_equal_and_duplicate_scenarios_are_deterministic_and_warn():
    returns = [[0.01, 0.01, 0.01]] * 20
    contract = _contract(scenario_returns=returns, expected_alpha=[0, 0, 0])
    result = optimise_cvar_portfolio(contract)
    assert result["status"] == "OPTIMAL"
    assert "WEAKLY_IDENTIFIED_CONSTANT_SCENARIO_LOSS" in result["warnings"]
    assert result["target_weights"] == pytest.approx(optimise_cvar_portfolio(contract)["target_weights"])


def test_active_tail_scenarios_follow_reported_threshold():
    result = optimise_cvar_portfolio(_contract())
    expected = [
        scenario
        for scenario, loss in zip(result["scenario_ids"], result["scenario_loss_vector"])
        if loss >= result["var_threshold"] - 1e-7
    ]
    assert result["active_tail_scenario_ids"] == expected


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda c, r: c["scenario_returns"][0].__setitem__(0, c["scenario_returns"][0][0] + 0.01), "SCENARIO_RETURN_MISMATCH"),
        (lambda c, r: (c["scenario_probabilities"].__setitem__(0, 0.04), c["scenario_probabilities"].__setitem__(1, 0.06)), "SCENARIO_PROBABILITY_MISMATCH"),
        (lambda c, r: r["target_weights"].__setitem__(0, r["target_weights"][0] + 0.01), "EXPOSURE_FAILED"),
        (lambda c, r: r.__setitem__("var_threshold", r["var_threshold"] + 0.01), "CVAR_RECOMPUTATION_MISMATCH"),
        (lambda c, r: r["excess_loss_variables"].__setitem__(0, r["excess_loss_variables"][0] + 0.01), "CVAR_RECOMPUTATION_MISMATCH"),
        (lambda c, r: r.__setitem__("cvar", r["cvar"] + 0.01), "CVAR_RECOMPUTATION_MISMATCH"),
        (lambda c, r: r.__setitem__("cvar_penalty", r["cvar_penalty"] + 0.01), "RESULT_CHECKSUM_MISMATCH"),
        (lambda c, r: r.__setitem__("total_objective", r["total_objective"] + 0.01), "OBJECTIVE_MISMATCH"),
        (lambda c, r: r.__setitem__("scenario_return_checksum", "changed"), "SCENARIO_RETURN_MISMATCH"),
    ],
)
def test_verifier_detects_mutation(mutation, expected_reason):
    contract = _contract()
    result = optimise_cvar_portfolio(contract)
    changed_contract, changed_result = copy.deepcopy(contract), copy.deepcopy(result)
    mutation(changed_contract, changed_result)
    verification = verify_cvar_result(changed_contract, changed_result)
    assert not verification["valid"]
    assert expected_reason in verification["blocking_reasons"]


def test_json_and_logical_identity_are_stable_across_creation_time():
    contract = _contract()
    first = optimise_cvar_portfolio(contract)
    second = copy.deepcopy(first)
    second["creation_metadata"]["created_at"] = "different"
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    assert json.loads(canonical_json(second))["asset_ids"] == ["A", "B", "C"]


def test_confidence_panel_uses_matched_populations():
    panel = evaluate_confidence_panel(_contract())
    assert panel["status"] == "OPTIMAL"
    assert panel["confidence_levels"] == [0.95, 0.975]
    assert panel["comparison"]["monotonic_risk_claimed"] is False


def test_comparison_accepts_matched_and_rejects_mismatched_scenarios():
    contract = _contract()
    result = optimise_cvar_portfolio(contract)
    comparison = compare_cvar_policies(contract, {"cvar_95": result})
    assert comparison["valid"]
    changed = copy.deepcopy(result)
    changed["scenario_return_checksum"] = "different"
    with pytest.raises(CVaRInputError, match="COMPARISON_SCENARIO_HISTORY_MISMATCH"):
        compare_cvar_policies(contract, {"changed": changed})
