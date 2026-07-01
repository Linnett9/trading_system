from __future__ import annotations

import pytest

from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.stock_alpha_stage_selection import StockAlphaStageSelector


def _settings() -> StockLevelResearchConfig:
    return StockLevelResearchConfig.from_mapping({"ml": {"stock_alpha_run_size": "benchmark"}})


def test_default_stage_selection_preserves_existing_enabled_stages():
    selection = StockAlphaStageSelector({"ml": {"stock_alpha_run_size": "benchmark"}}, _settings()).resolve()
    assert selection.enabled("stock_artifact") is True
    assert selection.enabled("target_comparison") is True
    assert selection.enabled("portfolio_policy_sweep") is True
    assert selection.skipped_by_user == []


def test_unknown_stage_name_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown stock-alpha stage"):
        StockAlphaStageSelector(
            {"ml": {"stock_alpha_stages": {"not_a_stage": False}}},
            _settings(),
        ).resolve()


def test_disabled_stage_is_reported_as_skipped_by_user():
    selection = StockAlphaStageSelector(
        {"ml": {"stock_alpha_stages": {"target_comparison": False}}},
        _settings(),
    ).resolve()
    assert selection.enabled("target_comparison") is False
    assert "target_comparison" in selection.payload()["skipped_by_user"]


def test_missing_disabled_dependency_blocks_downstream_stage():
    with pytest.raises(ValueError, match="requires alpha_features or an existing"):
        StockAlphaStageSelector(
            {"ml": {"stock_alpha_stages": {"alpha_features": False, "enriched_benchmark": True}}},
            _settings(),
            output_exists=lambda stage: False,
        ).resolve()


def test_existing_compatible_output_satisfies_disabled_dependency():
    selection = StockAlphaStageSelector(
        {"ml": {"stock_alpha_stages": {"enriched_benchmark": False, "target_comparison": True}}},
        _settings(),
        output_exists=lambda stage: stage == "enriched_benchmark",
    ).resolve()
    assert selection.enabled("enriched_benchmark") is False
    assert selection.enabled("target_comparison") is True


def test_explicit_existing_enriched_artifact_satisfies_alpha_features_dependency(tmp_path):
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    artifact.write_text("rebalance_date,symbol\n2024-01-01,SPY\n", encoding="utf-8")
    config = {
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "out"),
            "stock_alpha_run_size": "dev",
            "stock_level_prediction_artifacts_path": str(artifact),
            "stock_alpha_stages": {"alpha_features": False, "enriched_benchmark": True},
        }
    }
    settings = StockLevelResearchConfig.from_mapping(config)
    selection = StockAlphaStageSelector(config, settings, output_exists=lambda _stage: False).resolve()

    assert selection.enabled("alpha_features") is False
    assert selection.enabled("enriched_benchmark") is True


def test_missing_explicit_enriched_artifact_fails_clearly(tmp_path):
    config = {
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "out"),
            "stock_alpha_run_size": "dev",
            "stock_level_prediction_artifacts_path": str(tmp_path / "stock_level_prediction_artifacts_enriched.csv"),
            "stock_alpha_stages": {"alpha_features": False, "enriched_benchmark": True},
        }
    }
    settings = StockLevelResearchConfig.from_mapping(config)
    with pytest.raises(ValueError, match="ml.stock_level_prediction_artifacts_path pointing to an enriched artifact"):
        StockAlphaStageSelector(config, settings, output_exists=lambda _stage: False).resolve()


def test_alpha_features_disabled_requires_explicit_enriched_artifact(tmp_path):
    config = {
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "out"),
            "stock_alpha_run_size": "dev",
            "stock_alpha_stages": {"alpha_features": False, "enriched_benchmark": True},
        }
    }
    settings = StockLevelResearchConfig.from_mapping(config)
    with pytest.raises(ValueError, match="requires alpha_features or an existing"):
        StockAlphaStageSelector(config, settings, output_exists=lambda _stage: False).resolve()
