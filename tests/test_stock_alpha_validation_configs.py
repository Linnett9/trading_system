from __future__ import annotations

import json

import pytest

from config.config_loader import load_config
from core.research.framework.config import StockLevelResearchConfig
from core.research.ml.stock_level.run_manifest.paths import expected_stage_output_paths
from core.research.ml.stock_level.run_manifest.service import inspect_stock_alpha_run_status
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import build_stock_level_model_ranking_benchmark
from core.research.ml.stock_level.stock_alpha_model_sets import resolve_stock_alpha_model_set, resolve_stock_alpha_target_model_set
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir
from core.research.ml.stock_level.stock_alpha_stage_selection import StockAlphaStageSelector


def test_dev_ultrafast_config_resolves_to_dev_fast_and_ultrafast():
    config = load_config("config/config.stock_alpha_dev_ultrafast.yaml", overlay_project_config=True)
    settings = StockLevelResearchConfig.from_mapping(config)
    assert settings.run_size == "dev"
    assert settings.ranker_model_set == "fast"
    assert settings.target_comparison_model_set == "ultrafast"
    assert settings.min_train_dates == 12
    assert settings.test_window_dates == 4
    assert settings.embargo_dates == 1
    assert len(range(24)) > settings.min_train_dates + settings.embargo_dates
    selection = StockAlphaStageSelector(config, settings).resolve()
    assert selection.enabled("target_comparison") is True
    assert selection.enabled("portfolio_replay") is False
    assert selection.enabled("portfolio_policy_sweep") is False


def test_dev_diagnostic_ultrafast_config_uses_existing_dev_mechanics_with_isolated_output():
    tiny = load_config("config/config.stock_alpha_dev_ultrafast.yaml", overlay_project_config=True)
    diagnostic = load_config("config/config.stock_alpha_dev_diagnostic_ultrafast.yaml", overlay_project_config=True)
    tiny_settings = StockLevelResearchConfig.from_mapping(tiny)
    diagnostic_settings = StockLevelResearchConfig.from_mapping(diagnostic)

    assert diagnostic_settings.run_size == "dev"
    assert diagnostic_settings.ranker_model_set == "fast"
    assert diagnostic_settings.target_comparison_model_set == "ultrafast"
    assert diagnostic_settings.dev_max_dates == 180
    assert diagnostic_settings.dev_max_symbols == 200
    assert diagnostic_settings.dev_symbol_sample_method == "deterministic_hash"
    assert set(diagnostic_settings.dev_required_symbols) == {"SPY", "AAPL"}
    assert diagnostic_settings.dev_recent_dates_only is True
    assert diagnostic_settings.dev_max_dates > tiny_settings.dev_max_dates
    assert diagnostic_settings.dev_max_symbols > tiny_settings.dev_max_symbols
    assert (diagnostic_settings.min_train_dates, diagnostic_settings.test_window_dates, diagnostic_settings.embargo_dates) == (40, 10, 2)
    assert diagnostic_settings.dev_max_dates > diagnostic_settings.min_train_dates + diagnostic_settings.embargo_dates
    assert stock_alpha_output_dir(diagnostic) != stock_alpha_output_dir(tiny)
    assert stock_alpha_output_dir(diagnostic).as_posix().endswith("stock_alpha_diagnostic/dev")
    assert diagnostic["ml"]["stock_alpha_artifact_max_symbols"] == 200
    assert diagnostic["ml"]["stock_alpha_artifact_universe_paths"] == ["data/reference/universes/us_liquid_500.yaml"]


def test_benchmark_risk_controls_only_config_disables_upstream_and_enables_sweep():
    config = load_config("config/config.stock_alpha_benchmark_risk_controls_only.yaml", overlay_project_config=True)
    settings = StockLevelResearchConfig.from_mapping(config)
    selection = StockAlphaStageSelector(
        config,
        settings,
        output_exists=lambda stage: stage in {"alpha_features", "enriched_benchmark"},
    ).resolve()
    assert settings.run_size == "benchmark"
    assert settings.ranker_model_set == "fast"
    assert settings.target_comparison_model_set == "ultrafast"
    assert selection.enabled("stock_artifact") is False
    assert selection.enabled("enriched_benchmark") is False
    assert selection.enabled("portfolio_policy_sweep") is True


def test_benchmark_fast_config_keeps_non_dev_split_defaults_and_gates():
    config = load_config("config/config.stock_alpha_benchmark_fast.yaml", overlay_project_config=True)
    settings = StockLevelResearchConfig.from_mapping(config)
    assert settings.run_size == "benchmark"
    assert (settings.min_train_dates, settings.test_window_dates, settings.embargo_dates) == (52, 13, 2)
    assert config["ml"]["stock_alpha_experiment_report_require_all_outputs"] is False
    assert config["ml"]["stock_alpha_experiment_report_max_age_hours"] == 24


def test_dev_diagnostic_standard_validated_config_uses_validated_model_set_only(tmp_path):
    config = load_config("config/config.stock_alpha_dev_diagnostic_standard_validated.yaml", overlay_project_config=True)
    artifact = tmp_path / "stock_level_prediction_artifacts_enriched.csv"
    artifact.write_text("rebalance_date,symbol\n2024-01-01,SPY\n", encoding="utf-8")
    config["ml"]["stock_level_prediction_artifacts_path"] = str(artifact)
    settings = StockLevelResearchConfig.from_mapping(config)
    selection = StockAlphaStageSelector(
        config,
        settings,
        output_exists=lambda _stage: False,
    ).resolve()
    model_set = resolve_stock_alpha_model_set(settings.ranker_model_set)

    assert settings.run_size == "dev"
    assert settings.ranker_model_set == "validated_standard"
    assert settings.target_comparison_model_set == "ultrafast"
    assert settings.include_engineered_features is True
    assert settings.include_sequence_models is True
    assert settings.artifact_path == artifact
    assert stock_alpha_output_dir(config).as_posix().endswith("stock_alpha_diagnostic_standard/dev")
    assert len(model_set.included_models) == 12
    assert "news_analysis_transformer" not in model_set.included_models
    assert selection.enabled("enriched_benchmark") is True
    assert selection.enabled("baseline_benchmark") is False
    assert selection.enabled("portfolio_replay") is False


def test_news_feature_template_configs_load_with_research_guardrails():
    benchmark = load_config("config/config.stock_alpha_news_features_benchmark_fast_template.yaml", overlay_project_config=True)
    full = load_config("config/config.stock_alpha_news_features_full_template.yaml", overlay_project_config=True)

    for config, run_size in ((benchmark, "benchmark"), (full, "full")):
        ml = config["ml"]
        assert ml["stock_alpha_run_size"] == run_size
        assert ml["stock_alpha_news_contract_path"] == "data/news/stock_alpha_news_contract.csv"
        assert ml["stock_alpha_news_stock_rows_path"].endswith("stock_alpha_ensemble_average_rank_predictions.csv")
        assert ml["stock_alpha_news_features_path"].endswith("stock_alpha_news_features.csv")
        assert ml["stock_alpha_news_min_symbol_coverage"] == 0.60
        assert ml["stock_alpha_news_min_date_coverage"] == 0.60
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_news_transformer_diagnostic_configs_are_disabled_templates():
    dev = load_config("config/config.stock_alpha_dev_diagnostic_news_transformer.yaml", overlay_project_config=True)
    benchmark = load_config("config/config.stock_alpha_benchmark_diagnostic_news_transformer.yaml", overlay_project_config=True)

    assert dev["ml"]["stock_alpha_run_size"] == "dev"
    assert benchmark["ml"]["stock_alpha_run_size"] == "benchmark"
    for config in (dev, benchmark):
        ml = config["ml"]
        assert ml["stock_deep_diagnostic_model"] == "news_analysis_transformer"
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["stock_alpha_news_features_path"].endswith("stock_alpha_news_features.csv")
        assert ml["stock_alpha_stages"]["portfolio_replay"] is False
        assert ml["stock_alpha_stages"]["portfolio_policy_sweep"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_news_readiness_preflight_tiny_fixture_config_is_disabled_research_only():
    config = load_config(
        "config/config.stock_alpha_news_readiness_preflight_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]

    assert ml["stock_alpha_run_size"] == "dev"
    assert ml["stock_alpha_news_stock_rows_path"] == "tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"
    assert ml["stock_alpha_news_features_path"].endswith("stock_alpha_news_features.csv")
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False
    assert ml["promotion_thresholds_changed"] is False


def test_real_news_feature_and_preflight_templates_are_disabled_research_only():
    feature_config = load_config(
        "config/config.stock_alpha_news_features_real_template.yaml",
        overlay_project_config=True,
    )
    preflight_config = load_config(
        "config/config.stock_alpha_news_readiness_preflight_real_template.yaml",
        overlay_project_config=True,
    )

    for config in (feature_config, preflight_config):
        ml = config["ml"]
        assert ml["stock_alpha_run_size"] == "dev"
        assert ml["stock_alpha_news_features_path"].endswith("stock_alpha_news_features.csv")
        assert ml["stock_alpha_news_min_symbol_coverage"] == 0.60
        assert ml["stock_alpha_news_min_date_coverage"] == 0.60
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False

    assert feature_config["ml"]["stock_alpha_news_contract_path"] == "data/news/stock_alpha_news_contract.csv"
    assert feature_config["ml"]["stock_alpha_news_stock_rows_path"].endswith("stock_alpha_ensemble_average_rank_predictions.csv")
    assert preflight_config["ml"]["stock_alpha_news_stock_rows_path"].endswith("stock_alpha_ensemble_average_rank_predictions.csv")


def test_real_news_contract_ingest_template_loads_with_research_guardrails():
    config = load_config(
        "config/config.stock_alpha_news_contract_ingest_real_template.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]

    assert ml["stock_alpha_news_raw_path"] == "data/news/raw/stock_alpha_news_provider_export.csv"
    assert ml["stock_alpha_news_contract_path"] == "data/news/stock_alpha_news_contract.csv"
    assert ml["stock_alpha_news_contract_ingest_audit_dir"].endswith("news_contract_ingest")
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False
    assert ml["promotion_thresholds_changed"] is False


def test_tiny_raw_news_ingest_smoke_configs_load_with_research_guardrails():
    ingest = load_config(
        "config/config.stock_alpha_news_contract_ingest_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    features = load_config(
        "config/config.stock_alpha_news_features_tiny_ingest_fixture.yaml",
        overlay_project_config=True,
    )
    preflight = load_config(
        "config/config.stock_alpha_news_readiness_preflight_tiny_ingest_fixture.yaml",
        overlay_project_config=True,
    )

    assert ingest["ml"]["stock_alpha_news_raw_path"] == "tests/fixtures/stock_alpha_news/raw_provider_export_tiny.csv"
    assert ingest["ml"]["stock_alpha_news_contract_path"].endswith("stock_alpha_news_contract.csv")
    assert features["ml"]["stock_alpha_news_contract_path"] == ingest["ml"]["stock_alpha_news_contract_path"]
    assert features["ml"]["stock_alpha_news_stock_rows_path"] == "tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"
    assert preflight["ml"]["stock_alpha_news_features_path"] == features["ml"]["stock_alpha_news_features_path"]
    assert preflight["ml"]["stock_alpha_news_stock_rows_path"] == "tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"
    for config in (ingest, features, preflight):
        ml = config["ml"]
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False
    assert features["ml"]["stock_alpha_news_enable_transformer"] is False
    assert preflight["ml"]["stock_alpha_news_enable_transformer"] is False


def test_alias_raw_news_ingest_config_loads_with_provider_column_map():
    config = load_config(
        "config/config.stock_alpha_news_contract_ingest_alias_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    column_map = ml["stock_alpha_news_provider_column_map"]

    assert ml["stock_alpha_news_raw_path"] == "tests/fixtures/stock_alpha_news/raw_provider_export_alias_tiny.csv"
    assert ml["stock_alpha_news_contract_path"].endswith("stock_alpha_news_contract.csv")
    assert column_map["article_id"] == "id"
    assert column_map["symbol"] == "ticker"
    assert column_map["published_at_utc"] == "published_at"
    assert column_map["ingested_at"] == "collected_at"
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False
    assert ml["promotion_thresholds_changed"] is False


def test_news_provider_audit_configs_load_with_research_guardrails():
    alias = load_config(
        "config/config.stock_alpha_news_provider_audit_alias_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    real = load_config(
        "config/config.stock_alpha_news_provider_audit_real_template.yaml",
        overlay_project_config=True,
    )

    assert alias["ml"]["stock_alpha_news_raw_path"] == "tests/fixtures/stock_alpha_news/raw_provider_export_alias_tiny.csv"
    assert alias["ml"]["stock_alpha_news_provider_column_map"]["article_id"] == "id"
    assert alias["ml"]["stock_alpha_news_provider_audit_min_symbol_count"] == 2
    assert real["ml"]["stock_alpha_news_raw_path"] == "data/news/raw/stock_alpha_news_provider_export.csv"
    assert real["ml"]["stock_alpha_news_provider_audit_min_article_count"] == 100
    for config in (alias, real):
        ml = config["ml"]
        assert ml["stock_alpha_news_provider_audit_dir"].endswith("news_provider_audit")
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_news_coverage_audit_configs_load_with_research_guardrails():
    tiny = load_config(
        "config/config.stock_alpha_news_coverage_audit_tiny_ingest_fixture.yaml",
        overlay_project_config=True,
    )
    real = load_config(
        "config/config.stock_alpha_news_coverage_audit_real_template.yaml",
        overlay_project_config=True,
    )

    assert tiny["ml"]["stock_alpha_news_contract_path"].endswith("stock_alpha_news_contract.csv")
    assert tiny["ml"]["stock_alpha_news_stock_rows_path"] == "tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"
    assert tiny["ml"]["stock_alpha_news_coverage_min_symbol_coverage"] == 1.0
    assert real["ml"]["stock_alpha_news_contract_path"] == "data/news/stock_alpha_news_contract.csv"
    assert real["ml"]["stock_alpha_news_stock_rows_path"].endswith("stock_alpha_ensemble_average_rank_predictions.csv")
    assert real["ml"]["stock_alpha_news_coverage_min_symbol_coverage"] == 0.60
    for config in (tiny, real):
        ml = config["ml"]
        assert ml["stock_alpha_news_coverage_audit_dir"].endswith("news_coverage_audit")
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_news_pipeline_preflight_configs_load_with_research_guardrails():
    tiny = load_config(
        "config/config.stock_alpha_news_pipeline_preflight_tiny_fixture.yaml",
        overlay_project_config=True,
    )
    real = load_config(
        "config/config.stock_alpha_news_pipeline_preflight_real_template.yaml",
        overlay_project_config=True,
    )

    assert tiny["ml"]["stock_alpha_news_raw_path"] == "tests/fixtures/stock_alpha_news/raw_provider_export_alias_tiny.csv"
    assert tiny["ml"]["stock_alpha_news_provider_column_map"]["article_id"] == "id"
    assert tiny["ml"]["stock_alpha_news_stock_rows_path"] == "tests/fixtures/stock_alpha_news/stock_rows_tiny.csv"
    assert tiny["ml"]["stock_alpha_news_pipeline_preflight_output_dir"].endswith(
        "stock_alpha_news_pipeline_preflight_tiny_fixture/dev"
    )
    assert real["ml"]["stock_alpha_news_raw_path"] == "data/news/raw/stock_alpha_news_provider_export.csv"
    assert real["ml"]["stock_alpha_news_contract_path"] == "data/news/stock_alpha_news_contract.csv"
    assert real["ml"]["stock_alpha_news_stock_rows_path"].endswith("stock_alpha_ensemble_average_rank_predictions.csv")
    assert real["ml"]["stock_alpha_news_pipeline_preflight_output_dir"].endswith(
        "stock_alpha_news_pipeline_preflight_real/dev"
    )
    for config in (tiny, real):
        ml = config["ml"]
        assert ml["stock_alpha_run_size"] == "dev"
        assert ml["stock_alpha_news_provider_audit_dir"].endswith("news_provider_audit")
        assert ml["stock_alpha_news_contract_ingest_audit_dir"].endswith("news_contract_ingest")
        assert ml["stock_alpha_news_coverage_audit_dir"].endswith("news_coverage_audit")
        assert ml["stock_alpha_news_features_path"].endswith("stock_alpha_news_features.csv")
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_real_news_transformer_diagnostic_templates_are_dev_sized_and_gated():
    disabled = load_config(
        "config/config.stock_alpha_dev_diagnostic_news_transformer_real_disabled_template.yaml",
        overlay_project_config=True,
    )
    enabled = load_config(
        "config/config.stock_alpha_dev_diagnostic_news_transformer_real_enabled_template.yaml",
        overlay_project_config=True,
    )

    assert disabled["ml"]["stock_alpha_news_enable_transformer"] is False
    assert enabled["ml"]["stock_alpha_news_enable_transformer"] is True
    assert enabled["ml"]["stock_alpha_news_transformer_enablement_note"] == "preflight_passed_required_before_use"
    for config in (disabled, enabled):
        ml = config["ml"]
        assert ml["stock_deep_diagnostic_model"] == "news_analysis_transformer"
        assert ml["stock_alpha_run_size"] == "dev"
        assert ml["stock_alpha_dev_max_dates"] == 60
        assert ml["stock_alpha_dev_max_symbols"] == 25
        assert ml["stock_ranker_sequence_epochs"] == 1
        assert ml["stock_ranker_sequence_batch_size"] == 64
        assert ml["stock_alpha_news_contract_path"] == "data/news/stock_alpha_news_contract.csv"
        assert ml["stock_alpha_news_features_path"].endswith("stock_alpha_news_features.csv")
        assert ml["stock_alpha_stages"]["portfolio_replay"] is False
        assert ml["stock_alpha_stages"]["portfolio_policy_sweep"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_benchmark_risk_controls_only_fails_clearly_without_enriched_predictions():
    config = load_config("config/config.stock_alpha_benchmark_risk_controls_only.yaml", overlay_project_config=True)
    settings = StockLevelResearchConfig.from_mapping(config)
    with pytest.raises(ValueError, match="requires upstream stage 'enriched_benchmark'"):
        StockAlphaStageSelector(config, settings, output_exists=lambda _stage: False).resolve()


def test_run_status_resume_command_preserves_active_config_path_and_spacing(tmp_path):
    config = {
        "config_path": "config/config.stock_alpha_dev_ultrafast.yaml",
        "research": {"profile": "development"},
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "stock_alpha"),
            "stock_alpha_run_size": "dev",
            "stock_ranker_model_set": "fast",
            "stock_target_comparison_model_set": "ultrafast",
        },
    }
    payload = inspect_stock_alpha_run_status(config)
    assert "--mode ml-overnight-stock-alpha --config config/config.stock_alpha_dev_ultrafast.yaml" in payload["resume_command"]
    assert "alpha--config" not in payload["resume_command"]


def test_stale_target_comparison_message_identifies_model_set_mismatch(tmp_path):
    config = {
        "config_path": "config/config.stock_alpha_dev_ultrafast.yaml",
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "stock_alpha"),
            "stock_alpha_run_size": "benchmark",
            "stock_ranker_model_set": "fast",
            "stock_target_comparison_model_set": "ultrafast",
        },
    }
    settings = StockLevelResearchConfig.from_mapping(config)
    output_dir = settings.output_dir
    expected = expected_stage_output_paths(config, output_dir)
    target_json = expected["target_comparison"]["json_path"]
    target_json.parent.mkdir(parents=True)
    target_json.write_text(json.dumps(resolve_stock_alpha_target_model_set("fast").metadata()), encoding="utf-8")

    payload = inspect_stock_alpha_run_status(config)
    target = {stage["name"]: stage for stage in payload["stages"]}["target_comparison"]
    assert target["status"] == "stale"
    assert "existing output model set is fast" in target["stale_reason"]
    assert "active stock_target_comparison_model_set is ultrafast" in target["stale_reason"]
    guidance = {item["stage"]: item for item in payload["stale_stage_guidance"]}
    assert "Rerun target_comparison with the current config." in guidance["target_comparison"]["recommended_actions"]
    assert any("Disable target_comparison" in action for action in guidance["target_comparison"]["recommended_actions"])


def test_run_status_marks_old_diagnostic_stock_artifact_stale_when_universe_profile_changes(tmp_path):
    config = {
        "config_path": "config/config.stock_alpha_dev_diagnostic_ultrafast.yaml",
        "research": {"profile": "development"},
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "stock_alpha_diagnostic"),
            "stock_alpha_run_size": "dev",
            "stock_ranker_model_set": "fast",
            "stock_target_comparison_model_set": "ultrafast",
            "stock_alpha_dev_max_dates": 180,
            "stock_alpha_dev_max_symbols": 200,
            "stock_alpha_dev_recent_dates_only": True,
            "stock_alpha_dev_symbol_sample_method": "deterministic_hash",
            "stock_alpha_dev_required_symbols": ["SPY", "AAPL"],
            "stock_alpha_artifact_universe_paths": ["data/reference/universes/us_liquid_500.yaml"],
            "stock_alpha_artifact_max_symbols": 200,
            "stock_alpha_artifact_symbol_sample_method": "deterministic_hash",
        },
    }
    output_dir = stock_alpha_output_dir(config)
    output_dir.mkdir(parents=True)
    (output_dir / "stock_level_prediction_artifacts.csv").write_text("rebalance_date,symbol\n2024-01-01,SPY\n", encoding="utf-8")
    (output_dir / "stock_level_prediction_artifacts.md").write_text("# artifact\n", encoding="utf-8")
    (output_dir / "stock_level_prediction_artifacts.json").write_text(
        json.dumps({"mode": "stock_level_prediction_artifacts_research_only", "symbol_count": 32}),
        encoding="utf-8",
    )

    payload = inspect_stock_alpha_run_status(config)
    stock_artifact = {stage["name"]: stage for stage in payload["stages"]}["stock_artifact"]

    assert stock_artifact["status"] == "stale"
    assert "32 symbols" in stock_artifact["stale_reason"]
    assert "200 symbols" in stock_artifact["stale_reason"]
    assert "us_liquid_500" in stock_artifact["stale_reason"]


def test_run_status_keeps_matching_diagnostic_stock_artifact_reusable(tmp_path):
    config = {
        "config_path": "config/config.stock_alpha_dev_diagnostic_ultrafast.yaml",
        "research": {"profile": "development"},
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "stock_alpha_diagnostic"),
            "stock_alpha_run_size": "dev",
            "stock_ranker_model_set": "fast",
            "stock_target_comparison_model_set": "ultrafast",
            "stock_alpha_dev_max_dates": 180,
            "stock_alpha_dev_max_symbols": 200,
            "stock_alpha_dev_recent_dates_only": True,
            "stock_alpha_dev_symbol_sample_method": "deterministic_hash",
            "stock_alpha_dev_required_symbols": ["SPY", "AAPL"],
            "stock_alpha_artifact_universe_paths": ["data/reference/universes/us_liquid_500.yaml"],
            "stock_alpha_artifact_max_symbols": 200,
            "stock_alpha_artifact_symbol_sample_method": "deterministic_hash",
        },
    }
    output_dir = stock_alpha_output_dir(config)
    output_dir.mkdir(parents=True)
    (output_dir / "stock_level_prediction_artifacts.csv").write_text("rebalance_date,symbol\n2024-01-01,SPY\n", encoding="utf-8")
    (output_dir / "stock_level_prediction_artifacts.md").write_text("# artifact\n", encoding="utf-8")
    (output_dir / "stock_level_prediction_artifacts.json").write_text(
        json.dumps(
            {
                "mode": "stock_level_prediction_artifacts_research_only",
                "symbol_count": 180,
                "stock_alpha_artifact_profile": {
                    "stock_alpha_artifact_universe_paths": ["data/reference/universes/us_liquid_500.yaml"],
                    "stock_alpha_artifact_max_symbols": 200,
                    "stock_alpha_artifact_symbol_sample_method": "deterministic_hash",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = inspect_stock_alpha_run_status(config)
    stock_artifact = {stage["name"]: stage for stage in payload["stages"]}["stock_artifact"]

    assert stock_artifact["status"] == "completed"
    assert stock_artifact["stale_reason"] is None


def test_run_status_marks_ranker_benchmark_stale_when_split_settings_change(tmp_path):
    config = {
        "config_path": "config/config.stock_alpha_dev_diagnostic_ultrafast.yaml",
        "research": {"profile": "development"},
        "ml": {
            "stock_alpha_report_root": str(tmp_path / "stock_alpha_diagnostic"),
            "stock_alpha_run_size": "dev",
            "stock_ranker_model_set": "fast",
            "stock_target_comparison_model_set": "ultrafast",
            "stock_ranker_min_train_dates": 40,
            "stock_ranker_test_window_dates": 10,
            "stock_ranker_embargo_dates": 2,
            "stock_alpha_dev_max_dates": 180,
            "stock_alpha_dev_max_symbols": 200,
            "stock_alpha_dev_recent_dates_only": True,
            "stock_alpha_dev_symbol_sample_method": "deterministic_hash",
            "stock_alpha_dev_required_symbols": ["SPY", "AAPL"],
        },
    }
    output_dir = stock_alpha_output_dir(config)
    benchmark_dir = output_dir / "baseline"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "stock_level_model_oos_predictions.csv").write_text(
        "rebalance_date,symbol\n2024-01-01,SPY\n",
        encoding="utf-8",
    )
    (benchmark_dir / "stock_level_model_ranking_benchmark.json").write_text(
        json.dumps(
            {
                "effective_model_set": "fast",
                "included_models": ["ridge", "elastic_net", "random_forest", "gradient_boosting"],
                "source_path": str(output_dir / "stock_level_prediction_artifacts.csv"),
                "walk_forward": {
                    "min_train_dates": 52,
                    "test_window_dates": 13,
                    "embargo_rebalance_dates": 2,
                },
                "stock_alpha_run_profile": {
                    "stock_alpha_dev_max_dates": 180,
                    "stock_alpha_dev_max_symbols": 200,
                    "stock_alpha_dev_recent_dates_only": True,
                    "stock_alpha_dev_symbol_sample_method": "deterministic_hash",
                    "stock_alpha_dev_required_symbols": ["SPY", "AAPL"],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = inspect_stock_alpha_run_status(config)
    baseline = {stage["name"]: stage for stage in payload["stages"]}["baseline_benchmark"]

    assert baseline["status"] == "stale"
    assert "existing ranker split differs" in baseline["stale_reason"]
    assert "min_train_dates" in baseline["stale_reason"]


def test_low_rebalance_date_error_includes_split_details_and_config_path():
    rows = [
        {
            "rebalance_date": f"2024-01-{day:02d}",
            "symbol": "AAA",
            "predicted_momentum_20d": "0.1",
            "actual_forward_return_10d": "0.01",
        }
        for day in range(1, 25)
    ]
    with pytest.raises(ValueError) as exc:
        build_stock_level_model_ranking_benchmark(
            rows,
            min_train_dates=52,
            test_window_dates=13,
            embargo_dates=2,
            config_path="config/config.stock_alpha_dev_ultrafast.yaml",
            source_path="synthetic.csv",
            include_sequence_models=False,
        )
    message = str(exc.value)
    assert "available_rebalance_dates=24" in message
    assert "required_first_test_index=54" in message
    assert "min_train_dates=52" in message
    assert "test_window_dates=13" in message
    assert "embargo_dates=2" in message
    assert "active_config_path=config/config.stock_alpha_dev_ultrafast.yaml" in message
    assert "use benchmark/full data" in message
