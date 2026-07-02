from __future__ import annotations

from config.config_loader import load_config


REAL_DIAGNOSTIC_STOCK_ROWS_PATH = (
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
    "stock_alpha_ensemble_diagnostic/dev/ensemble/average_rank/"
    "stock_alpha_ensemble_average_rank_predictions.csv"
)


def test_real_news_templates_use_existing_diagnostic_stock_rows_path():
    config_keys = [
        ("config/config.stock_alpha_news_pipeline_preflight_real_template.yaml", "stock_alpha_news_stock_rows_path"),
        ("config/config.stock_alpha_news_collection_plan_real_template.yaml", "stock_alpha_stock_rows_path"),
        ("config/config.stock_alpha_dev_diagnostic_news_transformer_real_disabled_template.yaml", "stock_level_prediction_artifacts_path"),
        ("config/config.stock_alpha_news_feature_diagnostics_real_template.yaml", "stock_alpha_stock_rows_path"),
        ("config/config.stock_alpha_news_readiness_preflight_real_template.yaml", "stock_alpha_news_stock_rows_path"),
        ("config/config.stock_alpha_news_source_diagnostics_real_template.yaml", "stock_alpha_stock_rows_path"),
        ("config/config.stock_alpha_news_pipeline_inspect_real_template.yaml", "stock_alpha_news_stock_rows_path"),
        ("config/config.stock_alpha_news_coverage_audit_real_template.yaml", "stock_alpha_news_stock_rows_path"),
        ("config/config.stock_alpha_dev_diagnostic_news_transformer_real_enabled_template.yaml", "stock_level_prediction_artifacts_path"),
        ("config/config.stock_alpha_news_features_real_template.yaml", "stock_alpha_news_stock_rows_path"),
    ]

    for config_path, key in config_keys:
        ml = load_config(config_path, overlay_project_config=True)["ml"]
        assert ml[key] == REAL_DIAGNOSTIC_STOCK_ROWS_PATH


def test_real_historical_news_templates_label_provider_availability_policy():
    config_paths = [
        "config/config.stock_alpha_news_pipeline_preflight_real_template.yaml",
        "config/config.stock_alpha_news_coverage_audit_real_template.yaml",
        "config/config.stock_alpha_news_pipeline_inspect_real_template.yaml",
        "config/config.stock_alpha_news_features_real_template.yaml",
        "config/config.stock_alpha_news_readiness_preflight_real_template.yaml",
        "config/config.stock_alpha_dev_diagnostic_news_transformer_real_disabled_template.yaml",
        "config/config.stock_alpha_dev_diagnostic_news_transformer_real_enabled_template.yaml",
    ]

    for config_path in config_paths:
        ml = load_config(config_path, overlay_project_config=True)["ml"]
        assert ml["stock_alpha_news_pit_policy"] == "provider_available_at"
        assert ml["stock_alpha_news_availability_lag_hours"] == 24
        assert ml["stock_alpha_news_historical_provider_availability_enabled"] is True


def test_200_symbol_alpha_vantage_sec_edgar_collection_configs_are_bounded():
    dry = load_config(
        "config/config.stock_alpha_news_collect_alpha_vantage_sec_edgar_200symbol_dry_run.yaml",
        overlay_project_config=True,
    )
    write = load_config(
        "config/config.stock_alpha_news_collect_alpha_vantage_sec_edgar_200symbol_write_template.yaml",
        overlay_project_config=True,
    )

    for config in (dry, write):
        ml = config["ml"]
        collect = ml["stock_alpha_news_collect"]
        providers = collect["providers"]
        assert len(collect["symbols"]) == 200
        assert len(set(collect["symbols"])) == 200
        assert collect["symbols_per_batch"] == 25
        assert collect["provider_request_limit"] == 100
        assert collect["max_rows_per_provider"] == 1000
        assert providers["alpha_vantage"] == {
            "enabled": True,
            "api_key_env": "ALPHA_VANTAGE_API_KEY",
        }
        assert providers["sec_edgar"] == {"enabled": True}
        assert providers["gdelt"]["enabled"] is False
        assert providers["fmp"]["enabled"] is False
        assert providers["newsapi"]["enabled"] is False
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False

    assert dry["ml"]["stock_alpha_news_collect"]["dry_run"] is True
    assert dry["ml"]["stock_alpha_news_collect"]["rate_limit_sleep_seconds"] == 0
    assert write["ml"]["stock_alpha_news_collect"]["dry_run"] is False
    assert write["ml"]["stock_alpha_news_collect"]["allow_overwrite"] is False
    assert write["ml"]["stock_alpha_news_collect"]["merge_existing"] is True
    assert write["ml"]["stock_alpha_news_collect"]["backup_existing"] is True
    assert write["ml"]["stock_alpha_news_collect"]["rate_limit_sleep_seconds"] == 1
