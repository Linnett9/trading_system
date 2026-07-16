from __future__ import annotations

import copy

import numpy as np
import pytest

from core.research.ml.stock_level.contextual_elastic_net_selector import (
    ContextualElasticNetError,
    build_contextual_design_matrix,
    canonical_json,
    compare_contextual_with_stock_only,
    context_sensitivity,
    contextual_elastic_net_input,
    contextual_interaction_contract,
    contextual_stability,
    fit_contextual_elastic_net,
    verify_contextual_elastic_net_result,
)


STOCK = ["drawdown_recovery", "liquidity", "momentum", "risk_adjusted_momentum", "stock_volatility", "value"]
CONTEXT = ["market_drawdown", "market_trend", "market_volatility"]


def _rows(*, context_signal=True, validation_date_start=5):
    rows = []
    contexts = {
        1: [-0.1, -1.0, 0.5],
        2: [-0.2, 1.0, 1.0],
        3: [-0.3, -1.0, 1.5],
        4: [-0.4, 1.0, 2.0],
        5: [-0.5, -1.0, 2.5],
        6: [-0.6, 1.0, 3.0],
    }
    for date_index in range(1, 7):
        split = "TRAINING" if date_index < validation_date_start else "VALIDATION"
        decision = f"2024-01-{date_index * 3:02d}T10:00:00Z"
        context = contexts[date_index]
        for asset_index in range(5):
            momentum = float(asset_index - 2)
            stock_values = [
                float(asset_index % 2), 1.0 + asset_index, momentum,
                momentum / (1 + asset_index), 0.5 + 0.2 * asset_index,
                float(2 - asset_index),
            ]
            target = 0.25 * momentum + 0.1 * stock_values[-1]
            if context_signal:
                target += 0.8 * momentum * context[2] + 0.3 * momentum * context[1]
            rows.append({
                "row_id": f"R{date_index}{asset_index}", "asset_id": f"A{asset_index}",
                "decision_timestamp": decision, "feature_availability_timestamp": decision,
                "stock_feature_ids": STOCK, "stock_feature_values": stock_values,
                "market_context_ids": CONTEXT, "market_context_values": context,
                "target_value": target,
                "target_maturity_timestamp": "2024-01-13T00:00:00Z" if split == "TRAINING" else "2024-02-01T00:00:00Z",
                "sample_weight": 1.0, "split": split,
            })
    return rows


def _input(rows=None):
    return contextual_elastic_net_input(
        rows or _rows(), target_horizon="ten_sessions",
        stock_feature_schema_identity="stock_schema_v1",
        market_context_schema_identity="context_schema_v1",
        interaction_contract_identity="contextual_interaction_contract_v1",
        training_fold_identity="train_v1", validation_fold_identity="valid_v1",
        dataset_identity="synthetic_dataset", source_population_checksum="population",
    )


def _interactions():
    return contextual_interaction_contract(STOCK, CONTEXT)


def test_deterministic_repeated_fit_and_estimator_identity():
    data, interactions = _input(), _interactions()
    first = fit_contextual_elastic_net(data, interactions)
    second = fit_contextual_elastic_net(data, interactions)
    assert first["status"] == "READY"
    assert first["model_checksum"] == second["model_checksum"]
    assert first["prediction_checksum"] == second["prediction_checksum"]
    assert first["estimator_identity"] == "sklearn.linear_model.ElasticNet"
    assert first["dependency_version"] == "1.6.1"


def test_bounded_interaction_list_and_schema_identity():
    contract = _interactions()
    assert [row["interaction_id"] for row in contract["interactions"]] == [
        "momentum_x_market_volatility", "momentum_x_market_trend",
        "drawdown_recovery_x_market_drawdown",
        "risk_adjusted_momentum_x_market_volatility",
        "liquidity_x_market_volatility", "stock_volatility_x_market_volatility",
    ]
    assert _input()["ordered_stock_feature_ids"] == STOCK
    assert _input()["ordered_market_context_ids"] == CONTEXT


def test_unknown_duplicate_and_all_pairs_interactions_rejected():
    with pytest.raises(ContextualElasticNetError, match="UNKNOWN_STOCK_FEATURE"):
        contextual_interaction_contract(STOCK, CONTEXT, interactions=[{
            "interaction_id": "bad", "stock_feature_id": "unknown", "market_context_id": "market_volatility",
        }])
    duplicate = [
        {"interaction_id": "one", "stock_feature_id": "momentum", "market_context_id": "market_volatility"},
        {"interaction_id": "two", "stock_feature_id": "momentum", "market_context_id": "market_volatility"},
    ]
    with pytest.raises(ContextualElasticNetError, match="DUPLICATE"):
        contextual_interaction_contract(STOCK, CONTEXT, interactions=duplicate)
    all_pairs = [
        {"interaction_id": f"{stock}_{context}", "stock_feature_id": stock, "market_context_id": context}
        for stock in STOCK for context in CONTEXT
    ]
    with pytest.raises(ContextualElasticNetError, match="FULL_ALL_PAIRS"):
        contextual_interaction_contract(STOCK, CONTEXT, interactions=all_pairs)


def test_inconsistent_context_within_date_rejected():
    rows = _rows()
    rows[1]["market_context_values"] = rows[1]["market_context_values"].copy()
    rows[1]["market_context_values"][2] += 1
    with pytest.raises(ContextualElasticNetError, match="MULTIPLE_CONTEXT_VECTORS"):
        _input(rows)


def test_stable_design_order_and_known_scaled_interaction_value():
    result = fit_contextual_elastic_net(_input(), _interactions())
    design = build_contextual_design_matrix(
        _input(), _interactions(), result["preprocessing"], split="VALIDATION"
    )
    assert [row["column_id"] for row in design["column_lineage"]] == result["model"]["ordered_design_column_ids"]
    interaction_index = [row["column_id"] for row in design["column_lineage"]].index(
        "interaction:momentum_x_market_volatility"
    )
    row = _input()["rows"][20]
    momentum_index, volatility_index = STOCK.index("momentum"), CONTEXT.index("market_volatility")
    expected = (
        (row["stock_feature_values"][momentum_index] - result["preprocessing"]["stock_location"][momentum_index])
        / result["preprocessing"]["stock_scale"][momentum_index]
        * (row["market_context_values"][volatility_index] - result["preprocessing"]["context_location"][volatility_index])
        / result["preprocessing"]["context_scale"][volatility_index]
    )
    assert design["matrix"][0][interaction_index] == pytest.approx(expected)


def test_training_only_preprocessing_and_constant_context_handling():
    result = fit_contextual_elastic_net(_input(), _interactions())
    training_context = np.asarray([
        row["market_context_values"] for row in _input()["rows"] if row["split"] == "TRAINING"
    ])
    assert result["preprocessing"]["context_location"] == pytest.approx(training_context.mean(axis=0))
    rows = _rows()
    for row in rows:
        row["market_context_values"][0] = -0.2
    result = fit_contextual_elastic_net(_input(rows), _interactions())
    assert "market_drawdown" in result["preprocessing"]["constant_context_feature_ids"]


def test_temporal_maturity_duplicate_and_nonfinite_validation():
    rows = _rows(validation_date_start=3)
    rows.sort(key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))
    assert fit_contextual_elastic_net(_input(rows), _interactions())["status"] == "TEMPORAL_VIOLATION"
    rows = _rows(); rows[0]["target_maturity_timestamp"] = "2024-02-01T00:00:00Z"
    assert fit_contextual_elastic_net(_input(rows), _interactions())["blocking_reasons"] == ["TRAINING_TARGET_NOT_MATURE_BY_VALIDATION"]
    rows = _rows(); rows[-1]["row_id"] = rows[-2]["row_id"]
    with pytest.raises(ContextualElasticNetError, match="ROW_IDENTITIES_NOT_UNIQUE"):
        _input(rows)
    rows = _rows(); rows[0]["stock_feature_values"][0] = np.nan
    with pytest.raises(ContextualElasticNetError, match="STOCK_FEATURE_NON_FINITE"):
        _input(rows)
    rows = _rows(); rows[0]["market_context_values"][0] = np.inf
    with pytest.raises(ContextualElasticNetError, match="CONTEXT_FEATURE_NON_FINITE"):
        _input(rows)


def test_inadequate_training_and_empty_validation():
    rows = _rows()
    short = [row for row in rows if row["split"] == "TRAINING"][:5] + [row for row in rows if row["split"] == "VALIDATION"]
    short.sort(key=lambda row: (row["decision_timestamp"], row["asset_id"], row["row_id"]))
    assert fit_contextual_elastic_net(_input(short), _interactions())["status"] == "INSUFFICIENT_DATA"
    training = [row for row in rows if row["split"] == "TRAINING"]
    assert fit_contextual_elastic_net(_input(training), _interactions())["blocking_reasons"] == ["VALIDATION_SAMPLE_EMPTY"]


def test_deterministic_rank_ties():
    rows = _rows()
    for row in rows:
        if row["split"] == "VALIDATION":
            row["stock_feature_values"] = rows[20]["stock_feature_values"].copy()
    result = fit_contextual_elastic_net(_input(rows), _interactions(), minimum_rank_diversity=1)
    for decision in sorted({row["decision_timestamp"] for row in result["predictions"]}):
        ranked = sorted(
            [row for row in result["predictions"] if row["decision_timestamp"] == decision],
            key=lambda row: row["within_date_rank"],
        )
        assert [row["asset_id"] for row in ranked] == sorted(row["asset_id"] for row in ranked)


def test_contextual_relationship_comparison_and_no_signal_fixture():
    comparison = compare_contextual_with_stock_only(_input(), _interactions())
    assert comparison["valid"]
    assert comparison["contextual"]["mse"] < comparison["stock_only"]["mse"]
    assert "interaction:momentum_x_market_volatility" in comparison["interaction_recovery"]
    no_signal = compare_contextual_with_stock_only(_input(_rows(context_signal=False)), _interactions())
    assert no_signal["contextual_improvement_assumed"] is False


def test_context_sensitivity_diagnostic():
    result = fit_contextual_elastic_net(_input(), _interactions())
    diagnostic = context_sensitivity(
        result, stock_feature_values=[1, 2, 1, 0.5, 0.8, -1],
        baseline_context_values=[-0.1, 1, 0.5],
        changed_context_values=[-0.5, -1, 3.0],
        baseline_context_identity="low_vol_positive_trend",
        changed_context_identity="high_vol_negative_trend",
    )
    assert diagnostic["valid"]
    assert diagnostic["total_score_change"] != 0
    assert diagnostic["affected_interaction_contributions"]


def test_sparsity_diagnostics_and_stability():
    first = fit_contextual_elastic_net(_input(), _interactions())
    rows = _rows()
    for row in rows:
        if row["split"] == "TRAINING":
            row["target_value"] += 0.001
    second = fit_contextual_elastic_net(_input(rows), _interactions())
    assert 0 <= first["coefficient_diagnostics"]["interaction_sparsity"] <= 1
    stability = contextual_stability([first, second])
    assert stability["valid"]
    changed = copy.deepcopy(second); changed["interaction_checksum"] = "different"
    assert contextual_stability([first, changed])["status"] == "INTERACTION_CONTRACT_MISMATCH"


def test_nonconvergence_state():
    result = fit_contextual_elastic_net(_input(), _interactions(), maximum_iterations=1)
    assert result["status"] == "NON_CONVERGENCE"


def test_verifier_success_and_mutations():
    data, interactions = _input(), _interactions()
    result = fit_contextual_elastic_net(data, interactions)
    assert verify_contextual_elastic_net_result(data, interactions, result)["valid"]
    changed_data = copy.deepcopy(data); changed_data["rows"][0]["market_context_values"][0] += 1
    assert not verify_contextual_elastic_net_result(changed_data, interactions, result)["valid"]
    changed_interactions = copy.deepcopy(interactions); changed_interactions["interactions"][0]["market_context_id"] = "market_trend"
    assert not verify_contextual_elastic_net_result(data, changed_interactions, result)["valid"]
    for mutator in (
        lambda value: value["model"]["coefficient_vector"].__setitem__(0, value["model"]["coefficient_vector"][0] + 1),
        lambda value: value["predictions"][0].__setitem__("continuous_score", value["predictions"][0]["continuous_score"] + 1),
        lambda value: value["predictions"][0].__setitem__("within_date_rank", value["predictions"][0]["within_date_rank"] + 1),
    ):
        changed = copy.deepcopy(result); mutator(changed)
        assert not verify_contextual_elastic_net_result(data, interactions, changed)["valid"]


def test_stable_canonical_json_and_checksum():
    result = fit_contextual_elastic_net(_input(), _interactions())
    changed = copy.deepcopy(result); changed["creation_metadata"]["created_at"] = "different"
    assert result["logical_result_checksum"] == changed["logical_result_checksum"]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
