from __future__ import annotations

import copy

import pytest

from core.research.ml.ranking_labels import grouped_ranking_dataset
from core.research.ml.selector_component_rows import (
    EVALUATION_OUTCOMES_CONTRACT,
    evaluation_outcome,
    join_prediction_outcomes,
    reject_evaluation_artifact,
    prediction_row,
    training_row,
    validate_model_row_roles,
)
from core.research.ml.stock_level.contextual_elastic_net_selector import (
    contextual_elastic_net_input,
)
from core.research.ml.stock_level.huber_selector import huber_selector_input
from core.research.ml.stock_level.lightgbm_lambdarank_selector import (
    validate_lambdarank_input,
)
from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
    validate_rank_xendcg_input,
)
from core.research.ml.stock_level.multi_horizon_linear_selector import (
    HORIZON_IDS,
    multi_horizon_linear_input,
    multi_horizon_target_contract,
)


def _tabular_rows():
    return [
        {
            "row_id": "train", "asset_id": "A",
            "decision_timestamp": "2024-01-01T00:00:00Z",
            "feature_availability_timestamp": "2024-01-01T00:00:00Z",
            "feature_ids": ["signal"], "feature_values": [1.0],
            "target_value": 0.2,
            "target_maturity_timestamp": "2024-01-02T00:00:00Z",
            "split": "TRAINING",
        },
        {
            "row_id": "predict", "asset_id": "B",
            "decision_timestamp": "2024-02-01T00:00:00Z",
            "feature_availability_timestamp": "2024-02-01T00:00:00Z",
            "feature_ids": ["signal"], "feature_values": [2.0],
            "split": "PREDICTION",
        },
    ]


def test_shared_roles_reject_prediction_outcomes_and_population_overlap():
    rows = _tabular_rows()
    validate_model_row_roles(
        rows, role_field="split",
        target_fields=("target_value", "target_maturity_timestamp"),
    )
    for field, value in (
        ("target_value", 1.0),
        ("actual_forward_return_10d", 1.0),
        ("relevance_label", 4),
    ):
        changed = copy.deepcopy(rows)
        changed[1][field] = value
        with pytest.raises(ValueError, match="prohibited outcome"):
            validate_model_row_roles(
                changed, role_field="split",
                target_fields=("target_value", "target_maturity_timestamp"),
            )
    overlap = copy.deepcopy(rows)
    overlap[1]["row_id"] = "train"
    with pytest.raises(ValueError, match="overlap"):
        validate_model_row_roles(
            overlap, role_field="split",
            target_fields=("target_value", "target_maturity_timestamp"),
        )


def test_versioned_rows_bind_ancestry_and_maturity():
    common = {
        "dataset_identity": "dataset", "campaign_identity": "campaign",
        "plan_job_identity": "job", "model_id": "huber",
        "symbol_identity": "A", "decision_date": "2024-01-01",
        "target_horizon": "return_10s", "fold_identity": "fold",
        "dataset_row_identity": "row", "feature_schema_identity": "schema",
        "feature_order_checksum": "features",
        "ordered_feature_values": [1.0],
    }
    training = training_row({
        **common, "training_boundary_identity": "2024-01-10T00:00:00Z",
        "purge_sessions": 10, "embargo_sessions": 10,
        "target_contract": "forward_return_10d",
        "target_availability_timestamp": "2024-01-09T00:00:00Z",
        "target_maturity_timestamp": "2024-01-09T00:00:00Z",
        "target_value": 0.2,
    })
    prediction = prediction_row({**common, "dataset_row_identity": "predict"})
    assert training["logical_row_checksum"]
    assert prediction["logical_row_checksum"]
    with pytest.raises(ValueError, match="not mature"):
        training_row({
            **training,
            "target_availability_timestamp": "2024-01-11T00:00:00Z",
        })


def test_huber_contextual_and_multi_horizon_accept_unlabeled_prediction_rows():
    huber = huber_selector_input(
        _tabular_rows(), target_horizon="return_10s",
        target_contract_identity="forward_return_10d",
        feature_schema_identity="schema", training_fold_identity="train-fold",
        validation_fold_identity="prediction-fold", dataset_identity="dataset",
        source_population_checksum="population",
    )
    assert huber["rows"][1]["split"] == "PREDICTION"
    assert "target_value" not in huber["rows"][1]

    contextual_rows = []
    for row in _tabular_rows():
        contextual_rows.append({
            key: value for key, value in row.items()
            if key not in {"feature_ids", "feature_values"}
        } | {
            "stock_feature_ids": ["signal"],
            "stock_feature_values": row["feature_values"],
            "market_context_ids": ["market"],
            "market_context_values": [0.5],
        })
    contextual = contextual_elastic_net_input(
        contextual_rows, target_horizon="return_10s",
        stock_feature_schema_identity="stock",
        market_context_schema_identity="context",
        interaction_contract_identity="interactions",
        training_fold_identity="train-fold",
        validation_fold_identity="prediction-fold",
        dataset_identity="dataset", source_population_checksum="population",
    )
    assert "target_value" not in contextual["rows"][1]

    targets = {horizon: 0.1 for horizon in HORIZON_IDS}
    maturities = {
        horizon: "2024-01-02T00:00:00Z" for horizon in HORIZON_IDS
    }
    states = {horizon: "MATURE" for horizon in HORIZON_IDS}
    multi_rows = copy.deepcopy(_tabular_rows())
    multi_rows[0].pop("target_value")
    multi_rows[0].pop("target_maturity_timestamp")
    multi_rows[0].update(
        target_values=targets, target_maturity_timestamps=maturities,
        target_availability_state=states,
    )
    multi = multi_horizon_linear_input(
        multi_rows, target_contract=multi_horizon_target_contract(),
        feature_schema_identity="schema", dataset_identity="dataset",
        fold_identity="fold", source_population_checksum="population",
    )
    assert set(multi["target_contract"]["horizons"][0]) >= {"horizon_id"}
    assert "target_values" not in multi["rows"][1]


def test_rankers_require_training_labels_but_not_prediction_labels():
    rows = []
    for role, day in (("TRAINING", "2024-01-01"), ("PREDICTION", "2024-02-01")):
        for index in range(2):
            row = {
                "row_id": f"{role}-{index}", "asset_id": f"A{index}",
                "decision_date": day, "feature_names": ["signal"],
                "feature_values": [float(index)],
                "feature_availability_timestamp": day,
                "split_role": role,
            }
            if role == "TRAINING":
                row.update(
                    label=index,
                    target_maturity_timestamp="2024-01-02T00:00:00Z",
                )
            rows.append(row)
    dataset = grouped_ranking_dataset(
        rows, label_type="quintile_integer",
        feature_schema_identity="schema",
        target_contract_identity="forward_return_10d",
        ranking_label_contract_identity="within_date_quintile_relevance_v1",
        split_identity="fold", allowed_cutoff="2024-01-10T00:00:00Z",
        minimum_group_size=2,
    )
    assert dataset["valid"]
    assert all(
        "label" not in row for row in dataset["rows"]
        if row["split_role"] == "PREDICTION"
    )
    assert validate_rank_xendcg_input(
        dataset, training_cutoff="2024-01-10T00:00:00Z"
    )["valid"]
    assert validate_lambdarank_input(
        dataset, training_cutoff="2024-01-10T00:00:00Z"
    )["valid"]


def test_evaluation_outcomes_are_non_model_inputs_and_join_by_identity():
    identity = {
        "dataset_row_id": "predict", "symbol_identity": "B",
        "prediction_date": "2024-02-01", "target_horizon": "return_10s",
        "fold_identity": "fold", "plan_job_identity": "job",
    }
    outcome = evaluation_outcome(
        prediction_join_identity=identity, target_value=0.4,
        target_availability_timestamp="2024-02-12T00:00:00Z",
        outcome_maturity_timestamp="2024-02-12T00:00:00Z",
        maturity_contract="maturity.v1",
        target_provenance={"target_contract": "forward_return_10d"},
    )
    assert outcome["contract_version"] == EVALUATION_OUTCOMES_CONTRACT
    with pytest.raises(ValueError, match="not model inputs"):
        reject_evaluation_artifact(outcome)
    joined = join_prediction_outcomes(
        [{"row_id": "predict"}], [outcome],
        join_identity=lambda row: identity,
    )
    assert joined[0]["evaluation_outcome"]["target_value"] == 0.4
    with pytest.raises(ValueError, match="missing evaluation"):
        join_prediction_outcomes(
            [{"row_id": "other"}], [outcome],
            join_identity=lambda row: {**identity, "dataset_row_id": row["row_id"]},
        )
