from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from core.research.ml.hierarchical_risk_parity import (
    HRPInputError,
    canonical_json,
    compare_hrp_policies,
    correlation_distance,
    hrp_input,
    run_hrp,
    verify_hrp_result,
)


def _factor_returns(asset_count=6, observation_count=20):
    rows = []
    for t in range(observation_count):
        market = math.sin(t / 3) * 0.02
        factor_a = math.cos(t / 2) * 0.01
        factor_b = math.sin(t / 2.5) * 0.012
        rows.append([
            market + (factor_a if index < asset_count // 2 else factor_b) + (index + 1) * (t - 8) * 1e-5
            for index in range(asset_count)
        ])
    return rows


def _contract(asset_count=6, **overrides):
    assets = [f"A{index:02d}" for index in range(asset_count)]
    rows = _factor_returns(asset_count)
    values = {
        "asset_ids": assets,
        "return_matrix": rows,
        "observation_ids": [f"2024-01-{index + 1:02d}" for index in range(len(rows))],
        "selector_scores": [float(asset_count - index) for index in range(asset_count)],
        "sector_ids": ["S1" if index < asset_count // 2 else "S2" for index in range(asset_count)],
        "sector_caps": {"S1": 1.0, "S2": 1.0},
        "previous_weights": [1 / asset_count] * asset_count,
        "annualisation_factor": 1.0,
    }
    values.update(overrides)
    return hrp_input(**values)


def _logical(result):
    return {key: value for key, value in result.items() if key != "creation_metadata"}


def test_known_correlation_distance_calculation():
    covariance = [[1, 0.5], [0.5, 1]]
    result = correlation_distance(covariance)
    assert result["distance"][0][1] == pytest.approx(0.5)
    assert result["distance"][1][0] == pytest.approx(0.5)
    assert result["distance"][0][0] == 0


def test_deterministic_linkage_leaf_order_and_asset_completeness():
    contract = _contract()
    first = run_hrp(contract)
    second = run_hrp(contract)
    assert first["valid"]
    assert first["linkage_matrix"] == second["linkage_matrix"]
    assert first["leaf_order"] == second["leaf_order"]
    assert first["logical_result_checksum"] == second["logical_result_checksum"]
    assert sorted(first["leaf_order"]) == list(range(6))
    assert sorted(first["ordered_cluster_assets"]) == contract["asset_ids"]


def test_recursive_weights_sum_and_lower_risk_cluster_receives_more():
    rows = []
    for t in range(20):
        rows.append([0.001 * (-1) ** t, 0.0012 * (-1) ** t, 0.03 * (-1) ** t, 0.035 * (-1) ** t])
    contract = hrp_input(
        ["A", "B", "C", "D"], return_matrix=rows,
        observation_ids=[f"d{index:02d}" for index in range(20)],
        sector_ids=["L", "L", "H", "H"], annualisation_factor=1,
    )
    result = run_hrp(contract)
    assert sum(result["raw_hrp_weights"]) == pytest.approx(1)
    assert sum(result["final_weights"]) == pytest.approx(1)
    assert sum(result["raw_hrp_weights"][:2]) > sum(result["raw_hrp_weights"][2:])
    assert result["recursive_split_ledger"]


def test_identical_near_perfect_uncorrelated_and_clustered_structures():
    duplicated = [[math.sin(t), math.sin(t), math.cos(t)] for t in range(10)]
    near = [[math.sin(t), math.sin(t) + 1e-8 * t, math.cos(t)] for t in range(10)]
    uncorrelated = [[math.sin(t), math.cos(t), (-1) ** t] for t in range(10)]
    for matrix in (duplicated, near, uncorrelated, _factor_returns(6, 20)):
        assets = [f"A{index}" for index in range(len(matrix[0]))]
        contract = hrp_input(assets, return_matrix=matrix, observation_ids=[f"d{index:02d}" for index in range(len(matrix))])
        first = run_hrp(contract)
        second = run_hrp(contract)
        assert first["valid"]
        assert first["leaf_order"] == second["leaf_order"]


def test_constant_nonfinite_and_inadequate_inputs_block():
    constant = hrp_input(["A", "B"], return_matrix=[[1, 2], [1, 3], [1, 4]], observation_ids=["a", "b", "c"])
    assert run_hrp(constant)["status"] == "INVALID_INPUT"
    with pytest.raises(HRPInputError, match="NON_FINITE"):
        hrp_input(["A", "B"], return_matrix=[[0, 1], [1, math.nan], [2, 3]], observation_ids=["a", "b", "c"])
    with pytest.raises(HRPInputError, match="INSUFFICIENT"):
        hrp_input(["A", "B"], return_matrix=[[0, 1], [1, 2]], observation_ids=["a", "b"], minimum_observations=3)


def test_top_20_top_40_and_tie_handling():
    contract20 = _contract(22, selector_scores=[1.0] * 22)
    top20 = run_hrp(contract20, variant="top_20")
    assert len(top20["selected_candidate_assets"]) == 20
    assert top20["selected_candidate_assets"] == contract20["asset_ids"][:20]
    contract40 = _contract(42)
    top40 = run_hrp(contract40, variant="top_40")
    assert len(top40["selected_candidate_assets"]) == 40
    insufficient = run_hrp(_contract(19), variant="top_20")
    assert insufficient["status"] == "INSUFFICIENT_DATA"


def test_stock_sector_cap_post_processing_and_liquidity_exclusion():
    stock = run_hrp(_contract(maximum_weights=0.25))
    assert max(stock["final_weights"]) <= 0.25 + 1e-7
    assert "CONSTRAINED_POST_PROCESSING_APPLIED" in stock["warnings"]
    sector = run_hrp(_contract(sector_caps={"S1": 0.4, "S2": 1.0}))
    assert sum(sector["final_weights"][:3]) <= 0.4 + 1e-7
    liquidity = run_hrp(_contract(liquidity_eligible=[False, True, True, True, True, True]))
    assert liquidity["final_weights"][0] == 0
    assert "A00" in liquidity["liquidity_exclusions"]


def test_infeasible_caps_fail_closed():
    with pytest.raises(HRPInputError, match="STOCK_CAPS"):
        _contract(maximum_weights=0.1)
    with pytest.raises(HRPInputError, match="SECTOR_CAPS"):
        _contract(sector_caps={"S1": 0.2, "S2": 0.2})


def test_sector_first_hierarchy_and_single_asset_sector():
    contract = _contract(
        sector_ids=["SOLO", "S1", "S1", "S2", "S2", "S2"],
        sector_caps={"SOLO": 1, "S1": 1, "S2": 1},
    )
    result = run_hrp(contract, variant="sector_first")
    assert result["valid"]
    evidence = result["sector_first_evidence"]
    assert evidence["sector_return_aggregation"] == "inverse_volatility_sector_return_v1"
    assert evidence["within_sector"]["SOLO"]["weights"] == [1.0]
    assert set(evidence["sector_ids"]) == {"SOLO", "S1", "S2"}


def test_missing_sector_rejected():
    with pytest.raises(HRPInputError, match="SECTOR_MAPPING"):
        _contract(sector_ids=["", "S1", "S1", "S2", "S2", "S2"])


def test_exact_return_history_checksum_changes_with_return():
    first = _contract()
    changed_rows = copy.deepcopy(first["return_matrix"])
    changed_rows[0][0] += 1e-6
    second = hrp_input(
        first["asset_ids"], return_matrix=changed_rows, observation_ids=first["observation_ids"],
        selector_scores=first["selector_scores"], sector_ids=first["sector_ids"],
    )
    assert first["population_checksum"] == second["population_checksum"]
    assert first["return_history_checksum"] != second["return_history_checksum"]


@pytest.mark.parametrize("field,reason", [
    ("linkage_matrix", "LINKAGE_MISMATCH"),
    ("leaf_order", "LEAF_ORDER_MISMATCH"),
    ("recursive_split_ledger", "SPLIT_LEDGER_MISMATCH"),
    ("raw_hrp_weights", "RAW_WEIGHT_MISMATCH"),
    ("final_weights", "FINAL_WEIGHT_MISMATCH"),
])
def test_verifier_detects_changed_result_components(field, reason):
    contract = _contract()
    result = run_hrp(contract)
    assert verify_hrp_result(contract, result)["valid"]
    changed = copy.deepcopy(result)
    if isinstance(changed[field], list) and changed[field]:
        if isinstance(changed[field][0], list):
            changed[field][0][0] = float(changed[field][0][0]) + 0.1
        elif isinstance(changed[field][0], dict):
            changed[field][0]["left_allocation_fraction"] += 0.1
        else:
            changed[field][0] = float(changed[field][0]) + 0.1
    verification = verify_hrp_result(contract, changed)
    assert reason in verification["blocking_reasons"]
    assert "RESULT_CHECKSUM_MISMATCH" in verification["blocking_reasons"]


def test_verifier_detects_changed_return_input():
    contract = _contract()
    result = run_hrp(contract)
    changed = copy.deepcopy(contract)
    changed["return_matrix"][0][0] += 0.001
    verification = verify_hrp_result(changed, result)
    assert "RETURN_HISTORY_MISMATCH" in verification["blocking_reasons"]


def test_json_ordering_checksum_and_matched_comparison():
    contract = _contract()
    standard = run_hrp(contract)
    sector = run_hrp(contract, variant="sector_first")
    comparison = compare_hrp_policies(contract, {"hrp": standard, "sector": sector})
    assert comparison["valid"] and comparison["policy_count"] == 2
    assert not comparison["historical_returns_computed"]
    encoded = canonical_json(_logical(standard))
    assert encoded == canonical_json(json.loads(encoded))
    changed = copy.deepcopy(sector); changed["return_history_checksum"] = "other"
    with pytest.raises(HRPInputError, match="HISTORY_MISMATCH"):
        compare_hrp_policies(contract, {"changed": changed})
