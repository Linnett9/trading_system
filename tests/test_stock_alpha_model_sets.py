import inspect
import json

import pytest
from unittest.mock import patch

from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.stock_alpha_model_sets import (
    BASELINE_MODELS,
    FULL_SEQUENCE_MODELS,
    TABULAR_MODELS,
    VALIDATED_STANDARD_SEQUENCE_MODELS,
    resolve_stock_alpha_model_set,
    resolve_stock_alpha_target_model_set,
)


def test_fast_model_set_contains_tabular_models_and_baselines():
    resolved = resolve_stock_alpha_model_set("fast")
    assert resolved.included_models == TABULAR_MODELS
    assert resolved.baseline_models == BASELINE_MODELS
    assert resolved.metadata()["deep_sequence_models_skipped_intentionally"] is True


def test_standard_and_full_model_sets():
    standard = resolve_stock_alpha_model_set("standard")
    full = resolve_stock_alpha_model_set("full")
    assert standard.included_models == TABULAR_MODELS + ("dlinear", "market_context_encoder")
    assert full.included_models == TABULAR_MODELS + FULL_SEQUENCE_MODELS
    assert "news_analysis_transformer" in full.included_models
    assert standard.metadata()["excluded_models"]


def test_validated_standard_model_set_contains_validated_executable_models_only():
    resolved = resolve_stock_alpha_model_set("validated_standard")
    assert resolved.included_models == TABULAR_MODELS + VALIDATED_STANDARD_SEQUENCE_MODELS
    assert len(resolved.included_models) == 12
    assert "news_analysis_transformer" not in resolved.included_models
    excluded = {row["name"]: row["exclusion_reason"] for row in resolved.excluded_models}
    assert excluded == {"news_analysis_transformer": "conditional_news_features_unavailable"}
    assert resolved.metadata()["effective_model_set"] == "validated_standard"


def test_sequence_switch_excludes_sequence_models_with_reason():
    resolved = resolve_stock_alpha_model_set("full", include_sequence_models=False)
    assert resolved.included_models == TABULAR_MODELS
    assert {row["exclusion_reason"] for row in resolved.excluded_models} == {"sequence_models_disabled"}


def test_invalid_model_set_has_clear_error():
    with pytest.raises(ValueError, match="fast, standard, validated_standard, or full"):
        resolve_stock_alpha_model_set("huge")


def test_ultrafast_target_model_set_contains_linear_models_only():
    resolved = resolve_stock_alpha_target_model_set("ultrafast")
    assert resolved.included_models == ("ridge", "elastic_net")
    assert resolved.baseline_models == BASELINE_MODELS
    assert {
        row["exclusion_reason"]
        for row in resolved.excluded_models
        if row["name"] in {"random_forest", "gradient_boosting"}
    } == {"tree_model_excluded_by_ultrafast_target_comparison"}


def test_stock_ranker_rejects_ultrafast_but_target_comparison_accepts_it():
    with pytest.raises(ValueError, match="stock_ranker_model_set must be fast, standard, validated_standard, or full"):
        StockLevelResearchConfig.from_mapping({"ml": {"stock_ranker_model_set": "ultrafast"}})
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_target_comparison_model_set": "ultrafast"}})
    assert settings.target_comparison_model_set == "ultrafast"


def test_profile_defaults_and_target_comparison_control():
    dev = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "dev"}})
    benchmark = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark"}})
    explicit = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark", "stock_ranker_model_set": "standard", "stock_target_comparison_model_set": "fast"}})
    assert (dev.ranker_model_set, dev.target_comparison_model_set) == ("fast", "fast")
    assert (benchmark.ranker_model_set, benchmark.target_comparison_model_set) == ("full", "full")
    assert (explicit.ranker_model_set, explicit.target_comparison_model_set) == ("standard", "fast")


def test_model_set_module_has_no_operational_imports():
    from core.research.ml.stock_level import stock_alpha_model_sets
    source = inspect.getsource(stock_alpha_model_sets)
    assert all(word not in source for word in ("broker", "paper_trading", "live_trading", "order_execution"))


def test_fast_factory_selection_does_not_construct_sequence_factories():
    from core.research.ml.stock_level.stock_level_model_ranking_benchmark import _factories_for_model_set
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "dev"}})
    with patch("core.research.ml.stock_level.stock_level_model_ranking_benchmark._sequence_model_factories", side_effect=AssertionError("must not be called")):
        tabular, sequence = _factories_for_model_set(settings, resolve_stock_alpha_model_set("fast"), sklearn_n_jobs=1, torch_num_threads=3)
    assert set(tabular) == set(TABULAR_MODELS)
    assert sequence == {}


def test_full_factory_selection_passes_torch_num_threads():
    from core.research.ml.stock_level.stock_level_model_ranking_benchmark import _factories_for_model_set
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "full"}})
    with patch("core.research.ml.stock_level.stock_level_model_ranking_benchmark._sequence_model_factories", return_value={}) as factory:
        _factories_for_model_set(settings, resolve_stock_alpha_model_set("full"), sklearn_n_jobs=1, torch_num_threads=3)
    assert factory.call_args.kwargs["torch_num_threads"] == 3


def test_resume_model_set_compatibility_rejects_mismatch_and_missing_metadata(tmp_path):
    from core.research.ml.stock_level.overnight_stock_alpha_runner import _stage_model_set_compatibility
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark", "stock_ranker_model_set": "fast"}})
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(resolve_stock_alpha_model_set("full").metadata()))
    valid, reason = _stage_model_set_compatibility("baseline_benchmark", {"json_path": path}, settings)
    assert valid is False and "current config requests stock_ranker_model_set=fast" in reason
    path.write_text("{}")
    valid, reason = _stage_model_set_compatibility("baseline_benchmark", {"json_path": path}, settings)
    assert valid is False and "without model-set metadata" in reason


def test_resume_model_set_compatibility_accepts_fast_and_checks_target_set(tmp_path):
    from core.research.ml.stock_level.overnight_stock_alpha_runner import _stage_model_set_compatibility
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark", "stock_ranker_model_set": "fast", "stock_target_comparison_model_set": "fast"}})
    path = tmp_path / "output.json"; path.write_text(json.dumps(resolve_stock_alpha_model_set("fast").metadata()))
    assert _stage_model_set_compatibility("enriched_benchmark", {"json_path": path}, settings) == (True, None)
    path.write_text(json.dumps(resolve_stock_alpha_model_set("full").metadata()))
    valid, reason = _stage_model_set_compatibility("target_comparison", {"json_path": path}, settings)
    assert valid is False and "stock_target_comparison_model_set=fast" in reason


def test_resume_model_set_compatibility_rejects_fast_when_ultrafast_requested(tmp_path):
    from core.research.ml.stock_level.overnight_stock_alpha_runner import _stage_model_set_compatibility
    settings = StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark", "stock_target_comparison_model_set": "ultrafast"}})
    path = tmp_path / "target.json"
    path.write_text(json.dumps(resolve_stock_alpha_target_model_set("fast").metadata()))
    valid, reason = _stage_model_set_compatibility("target_comparison", {"json_path": path}, settings)
    assert valid is False
    assert "stock_target_comparison_model_set=ultrafast" in reason
