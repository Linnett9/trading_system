from __future__ import annotations

import numpy as np
import pytest

from core.research.ml.ranking_labels import canonical_hash, grouped_ranking_dataset
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.stock_level.lightgbm_production_selector import (
    fit_production_lightgbm_selector,
)
from core.research.ml.stock_level.lightgbm_rank_xendcg_selector import (
    fit_synthetic_rank_xendcg_selector,
)


@pytest.mark.parametrize(
    ("model_id", "objective"),
    [
        ("lightgbm_rank_xendcg", "rank_xendcg"),
        ("lightgbm_lambdarank", "lambdarank"),
    ],
)
def test_production_owner_uses_only_supplied_training_and_prediction_rows(
    model_id, objective
):
    dataset = _dataset()
    captured = {}

    class Model:
        def predict(self, matrix):
            captured["prediction_matrix"] = matrix.tolist()
            return np.arange(len(matrix), dtype=float)

    def fit(parameters, matrix, labels, groups):
        captured.update(
            parameters=parameters, training_matrix=matrix.tolist(),
            labels=labels.tolist(), groups=list(groups),
        )
        return Model()

    result = fit_production_lightgbm_selector(
        dataset, model_id=model_id,
        authoritative_context=_context(model_id, dataset),
        dependency_preflight=_dependency(objective),
        estimator_fitters={model_id: fit},
    )

    assert result["valid"]
    assert result["objective"] == objective
    assert result["production_owner"] is True
    assert result["synthetic_only"] is False
    assert result["promotion_evidence"] is False
    assert result["promoted"] is False
    assert captured["parameters"]["objective"] == objective
    assert captured["parameters"]["n_jobs"] == 1
    assert len(captured["training_matrix"]) == 5
    assert len(captured["prediction_matrix"]) == 5
    assert captured["groups"] == [5]
    assert {
        row["decision_date"]
        for row in result["prediction_contract"]["rows"]
    } == {"2024-03-15"}


def test_dependency_failure_precedes_estimator_fit():
    called = False

    def fit(*args):
        nonlocal called
        called = True

    dataset = _dataset()
    result = fit_production_lightgbm_selector(
        dataset, model_id="lightgbm_rank_xendcg",
        authoritative_context=_context("lightgbm_rank_xendcg", dataset),
        dependency_preflight={"status": "MISSING_DEPENDENCY"},
        estimator_fitters={"lightgbm_rank_xendcg": fit},
    )

    assert not result["valid"]
    assert called is False


def test_existing_synthetic_entry_point_remains_synthetic_only(monkeypatch):
    class Booster:
        def model_to_string(self):
            return "model"

        def dump_model(self):
            return {"tree_info": []}

        def feature_importance(self, kind):
            return np.asarray([1.0])

    class Model:
        booster_ = Booster()
        best_iteration_ = 0

        def predict(self, matrix):
            return np.arange(len(matrix), dtype=float)

    monkeypatch.setattr(
        "core.research.ml.stock_level.lightgbm_rank_xendcg_selector._fit",
        lambda *args: Model(),
    )
    result = fit_synthetic_rank_xendcg_selector(
        _dataset(), training_cutoff="2024-03-14T00:00:00Z"
    )
    assert result["diagnostics"]["synthetic_only"] is True
    assert result["diagnostics"]["promotion_evidence"] is False


def _dataset():
    rows = []
    for role, date in (
        ("TRAINING", "2024-01-02"), ("VALIDATION", "2024-03-15")
    ):
        for index in range(5):
            rows.append({
                "row_id": f"{date}-{index}", "asset_id": f"A{index}",
                "decision_date": date, "feature_names": ["signal"],
                "feature_values": [float(index)],
                "feature_availability_timestamp": date,
                "label": index,
                "target_maturity_timestamp": "2024-03-01T00:00:00Z",
                "split_role": role,
            })
    return grouped_ranking_dataset(
        rows, label_type="quintile_integer",
        feature_schema_identity="tree-schema",
        target_contract_identity="forward_return_10d",
        ranking_label_contract_identity="within_date_quintile_relevance_v1",
        split_identity="fold-1",
        allowed_cutoff="2024-03-14T00:00:00Z",
        minimum_group_size=5,
    )


def _context(model_id, dataset):
    entry = RegistryResolver(load_registry_bundle()).resolve(
        "selector_models", model_id, role="selector"
    ).entry
    return {
        "selector_dataset_identity": "dataset-id",
        "selector_dataset_checksum": "A" * 64,
        "operational_input_identity": "input-id",
        "operational_input_checksum": dataset["dataset_checksum"],
        "campaign_identity": "campaign-id",
        "production_plan_job_checksum": "C" * 64,
        "model_registry_identity": entry.entry_hash,
        "ranking_contract_identity": "daily_cross_sectional_ranking_problem_v1",
        "grouped_query_contract": "grouped_ranking_dataset_v1",
        "relevance_label_contract": "within_date_quintile_relevance_v1",
        "target_contract": "forward_return_10d",
        "horizon_contract": "return_10s",
        "fold_identity": dataset["split_identity"],
        "training_boundary_identity": "boundary-id",
        "outcome_maturity_cutoff": "2024-03-14T00:00:00Z",
        "purge_sessions": 20, "embargo_sessions": 5,
        "feature_schema": dataset["feature_schema_identity"],
        "ordered_feature_checksum": canonical_hash(dataset["feature_names"]),
        "model_configuration_checksum": entry.payload[
            "fitting_configuration_checksum"
        ],
        "seed": 1729, "source_commit": "fixture",
    }


def _dependency(objective):
    return {
        "status": "READY", "lightgbm_version": "4.6.0",
        "objective": objective,
    }
