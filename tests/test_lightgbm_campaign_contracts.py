from __future__ import annotations

import inspect
from importlib import import_module
import types

import pytest

from core.research.ml.ranking_labels import (
    RankingLabelError,
    grouped_ranking_dataset,
    mature_training_integer_relevance,
)
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.adapters import selector_model_adapter
from core.research.ml.selector_research_campaign import (
    build_selector_research_campaign,
)
from core.research.ml.selector_research_protocol import (
    REQUIRED_IDENTITIES,
    freeze_selector_research_protocol,
)
from core.research.ml.stock_level.wave4_selector_integration import (
    assess_lightgbm_ranking_dependency,
)
from core.research.ml.registries.types import RegistryValidationError


MODELS = ("lightgbm_rank_xendcg", "lightgbm_lambdarank")


def test_lightgbm_registry_objectives_and_publication_adapters_are_exact():
    resolver = RegistryResolver(load_registry_bundle())
    entries = {
        model: resolver.resolve(
            "selector_models", model, role="selector"
        ).entry.payload
        for model in MODELS
    }

    assert entries["lightgbm_rank_xendcg"]["objective"] == "rank_xendcg"
    assert entries["lightgbm_lambdarank"]["objective"] == "lambdarank"
    assert (
        entries["lightgbm_rank_xendcg"]["objective_identity"]
        != entries["lightgbm_lambdarank"]["objective_identity"]
    )
    assert (
        entries["lightgbm_rank_xendcg"]["fitting_configuration_checksum"]
        != entries["lightgbm_lambdarank"]["fitting_configuration_checksum"]
    )
    for model, entry in entries.items():
        adapter = selector_model_adapter(model, runner="ordinary")
        assert adapter.constructor_owner.endswith(
            "wave4_selector_integration:publish_wave4_component"
        )
        assert entry["implementation_status"] == (
            "IMPLEMENTED_AND_AUTHORITATIVE_RUNNABLE"
        )
        assert entry["ordinary_runner_support"] is True
        assert entry["implementation_owner"] == (
            "core.research.ml.stock_level.lightgbm_production_selector:"
            "fit_production_lightgbm_selector"
        )
        assert "fit_synthetic_" not in entry["implementation_owner"]
        module_name, callable_name = entry["synthetic_fixture_owner"].split(
            ":", 1
        )
        fixture_owner = getattr(import_module(module_name), callable_name)
        assert callable(fixture_owner)
        assert '"synthetic_only": True' in inspect.getsource(fixture_owner)
        assert entry["production_owner"] is True
        assert entry["synthetic_only"] is False
        assert entry["strict_oos_capable"] is True
        assert entry["campaign_execution_eligible"] is True
        assert entry["promotion_evidence"] is False
        assert entry["promoted"] is False
        assert entry["grouped_query_contract"] == "grouped_ranking_dataset_v1"
        assert entry["ranking_problem_contract"] == (
            "daily_cross_sectional_ranking_problem_v1"
        )
        assert entry["relevance_contract"] == (
            "within_date_quintile_relevance_v1"
        )
        assert entry["inner_n_jobs"] == 1
        assert entry["bounded_search_policy"]["maximum_configurations"] == 20


def test_nonfitting_dependency_preflight_statuses_are_explicit():
    compatible = types.SimpleNamespace(
        __version__="4.6.0", LGBMRanker=lambda **kwargs: None
    )
    ready = assess_lightgbm_ranking_dependency(
        objective="rank_xendcg", importer=lambda name: compatible
    )
    assert ready["status"] == "READY"
    assert ready["inner_n_jobs"] == 1
    assert ready["fitting_performed"] is False

    def missing(name):
        raise ModuleNotFoundError(name)

    assert assess_lightgbm_ranking_dependency(
        objective="rank_xendcg", importer=missing
    )["status"] == "MISSING_DEPENDENCY"
    wrong = types.SimpleNamespace(
        __version__="3.3.0", LGBMRanker=lambda **kwargs: None
    )
    assert assess_lightgbm_ranking_dependency(
        objective="rank_xendcg", importer=lambda name: wrong
    )["status"] == "UNSUPPORTED_VERSION"
    assert assess_lightgbm_ranking_dependency(
        objective="unsupported", importer=lambda name: compatible
    )["status"] == "UNSUPPORTED_OBJECTIVE"
    assert assess_lightgbm_ranking_dependency(
        objective="lambdarank", num_threads=2,
        importer=lambda name: compatible,
    )["status"] == "INVALID_CONFIGURATION"


def test_mature_integer_relevance_is_training_only_and_deterministic():
    rows = _target_rows()
    first = mature_training_integer_relevance(
        rows,
        target_contract_identity="forward_return_10d",
        maturity_cutoff="2024-01-20T00:00:00Z",
        bins=5,
    )
    second = mature_training_integer_relevance(
        list(reversed(rows)),
        target_contract_identity="forward_return_10d",
        maturity_cutoff="2024-01-20T00:00:00Z",
        bins=5,
    )

    assert first == second
    assert all(isinstance(value, int) for value in first["labels_by_row_id"].values())
    assert set(first["labels_by_row_id"].values()) == set(range(5))
    with pytest.raises(RankingLabelError, match="TRAINING_ROWS_ONLY"):
        mature_training_integer_relevance(
            [{**rows[0], "split_role": "VALIDATION"}, *rows[1:]],
            target_contract_identity="forward_return_10d",
            maturity_cutoff="2024-01-20T00:00:00Z",
        )
    with pytest.raises(RankingLabelError, match="IMMATURE"):
        mature_training_integer_relevance(
            [{**rows[0], "target_maturity_timestamp": "2024-02-01T00:00:00Z"},
             *rows[1:]],
            target_contract_identity="forward_return_10d",
            maturity_cutoff="2024-01-20T00:00:00Z",
        )
    with pytest.raises(RankingLabelError, match="NON_FINITE"):
        mature_training_integer_relevance(
            [{**rows[0], "realised_target": float("nan")}, *rows[1:]],
            target_contract_identity="forward_return_10d",
            maturity_cutoff="2024-01-20T00:00:00Z",
        )


def test_grouped_query_order_sizes_and_fold_boundaries_are_deterministic():
    labels = mature_training_integer_relevance(
        _target_rows(),
        target_contract_identity="forward_return_10d",
        maturity_cutoff="2024-01-20T00:00:00Z",
    )["labels_by_row_id"]
    rows = [
        {
            "row_id": row["row_id"],
            "asset_id": row["asset_id"],
            "decision_date": row["decision_date"],
            "feature_names": ["signal"],
            "feature_values": [row["realised_target"]],
            "feature_availability_timestamp": row["decision_date"],
            "label": labels[row["row_id"]],
            "target_maturity_timestamp": row["target_maturity_timestamp"],
            "split_role": "TRAINING",
        }
        for row in _target_rows()
    ]
    dataset = grouped_ranking_dataset(
        list(reversed(rows)),
        label_type="quintile_integer",
        feature_schema_identity="tree-schema",
        target_contract_identity="forward_return_10d",
        ranking_label_contract_identity="within_date_quintile_relevance_v1",
        split_identity="fold-1",
        allowed_cutoff="2024-01-20T00:00:00Z",
        minimum_group_size=5,
    )

    assert dataset["valid"]
    assert dataset["group_size_vector"] == [5, 5]
    assert sum(dataset["group_size_vector"]) == len(dataset["rows"])
    assert dataset["rows"] == sorted(
        dataset["rows"],
        key=lambda row: (row["decision_date"], row["asset_id"], row["row_id"]),
    )
    mixed = [dict(row) for row in rows]
    mixed[0]["split_role"] = "VALIDATION"
    blocked = grouped_ranking_dataset(
        mixed,
        label_type="quintile_integer",
        feature_schema_identity="tree-schema",
        target_contract_identity="forward_return_10d",
        ranking_label_contract_identity="within_date_quintile_relevance_v1",
        split_identity="fold-1",
        allowed_cutoff="2024-01-20T00:00:00Z",
        minimum_group_size=5,
    )
    assert blocked["status"] == "SPLIT_OVERLAP"


def test_phase_c_adds_ten_production_owned_components():
    campaign = build_selector_research_campaign(_protocol())
    phase_ab = [
        row for row in campaign["fitted_component_matrix"]
        if row["phase_id"] in {"phase_a", "phase_b"}
    ]
    phase_c = [
        row for row in campaign["fitted_component_matrix"]
        if row["phase_id"] == "phase_c"
    ]

    assert len(phase_ab) == 65
    assert len(phase_c) == 10
    assert campaign["expected_component_count"] == 75
    phase = next(row for row in campaign["phases"] if row["phase_id"] == "phase_c")
    assert phase["status"] == "CAMPAIGN_READY"


def test_component_campaign_has_lightgbm_production_dispatch():
    campaign = build_selector_research_campaign(_protocol())
    assert {
        row["model_id"] for row in campaign["fitted_component_matrix"]
    }.intersection(MODELS) == set(MODELS)


def _target_rows():
    rows = []
    for date_index, date in enumerate(("2024-01-02", "2024-01-03")):
        for asset_index in range(5):
            rows.append(
                {
                    "row_id": f"{date}-r{asset_index}",
                    "asset_id": f"A{asset_index}",
                    "decision_date": date,
                    "realised_target": float(asset_index + date_index / 10),
                    "target_maturity_timestamp": "2024-01-15T00:00:00Z",
                    "split_role": "TRAINING",
                }
            )
    return rows


def _protocol():
    return freeze_selector_research_protocol(
        campaign_identity="synthetic-selector-campaign",
        frozen_identities={
            name: {"identity": f"id-{name}", "checksum": f"sum-{name}"}
            for name in REQUIRED_IDENTITIES
        },
        source_commit="fixture-commit",
    )
