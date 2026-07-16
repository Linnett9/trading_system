from __future__ import annotations

import pytest

from core.research.ml.ranking import OrderedLogitRanker
from core.research.ml.registries.adapters import (
    selector_model_adapter,
    verify_registry_capabilities,
)
from core.research.ml.registries.types import RegistryValidationError
from core.research.ml.stock_level_benchmark_models import _build_tabular_model


def test_registry_capabilities_are_truthful_for_bounded_and_ordinary():
    adapter = selector_model_adapter("ordered_logit_ranker", runner="bounded")
    assert adapter.bounded_runner_support is True
    assert adapter.ordinary_runner_support is True
    assert adapter.ranking_problem_contract == "daily_cross_sectional_ranking_problem_v1"
    assert adapter.relevance_contract == "within_date_quintile_relevance_v1"
    assert selector_model_adapter("ordered_logit_ranker", runner="ordinary").canonical_model_id == "ordered_logit_ranker"
    assert verify_registry_capabilities()["bounded"] >= 1


def test_missing_bounded_constructor_is_reported(monkeypatch):
    import core.research.ml.stock_level.bounded_selector_runner as bounded

    monkeypatch.setattr(
        bounded,
        "SUPPORTED_MODELS",
        tuple(name for name in bounded.SUPPORTED_MODELS if name != "ordered_logit_ranker"),
    )
    with pytest.raises(
        RegistryValidationError,
        match="bounded runner has no constructor for ordered_logit_ranker",
    ):
        verify_registry_capabilities()


def test_ordered_logit_constructor_and_legacy_regressors_are_unchanged():
    assert isinstance(_build_tabular_model("ordered_logit_ranker", 42, 1), OrderedLogitRanker)
    ridge = _build_tabular_model("ridge", 42, 1)
    elastic = _build_tabular_model("elastic_net", 42, 1)
    assert ridge.get_params()["regressor__ridge__alpha"] == pytest.approx(1.0)
    assert elastic.get_params()["regressor__elasticnet__alpha"] == pytest.approx(0.001)


def test_invalid_nonfinite_and_missing_class_labels_fail_closed():
    model = OrderedLogitRanker()
    x = [[float(index)] for index in range(10)]
    groups = ["2026-01-01"] * 10
    row_ids = [str(index) for index in range(10)]
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, [*range(9), float("nan")], groups=groups, row_ids=row_ids)
    with pytest.raises(ValueError, match="too small"):
        model.fit(x[:4], list(range(4)), groups=groups[:4], row_ids=row_ids[:4])
