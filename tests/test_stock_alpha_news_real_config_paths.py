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


def test_200_symbol_massive_collection_config_is_dry_run_only():
    config = load_config(
        "config/config.stock_alpha_news_collect_massive_200symbol_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    providers = collect["providers"]

    assert len(collect["symbols"]) == 200
    assert len(set(collect["symbols"])) == 200
    assert collect["dry_run"] is True
    assert collect["allow_overwrite"] is False
    assert collect["merge_existing"] is False
    assert collect["backup_existing"] is False
    assert collect["start_date"] == "2023-12-01"
    assert collect["end_date"] == "2026-04-20"
    assert collect["provider_request_limit"] == 100
    assert collect["max_rows_per_provider"] == 1000
    assert collect["max_pages_per_symbol"] == 1
    assert collect["symbols_per_batch"] == 25
    assert collect["rate_limit_sleep_seconds"] == 1
    assert providers["massive_stock_news"] == {
        "enabled": True,
        "api_key_env": "MASSIVE_API_KEY",
        "api_key_env_fallbacks": ["POLYGON_API_KEY"],
    }
    assert providers["alpha_vantage"]["enabled"] is False
    assert providers["sec_edgar"]["enabled"] is False
    assert providers["gdelt"]["enabled"] is False
    assert providers["fmp"]["enabled"] is False
    assert providers["newsapi"]["enabled"] is False
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False
    assert ml["promotion_thresholds_changed"] is False


def test_200_symbol_gdelt_collection_configs_are_dry_run_only():
    gdelt = load_config(
        "config/config.stock_alpha_news_collect_gdelt_200symbol_dry_run.yaml",
        overlay_project_config=True,
    )
    combined = load_config(
        "config/config.stock_alpha_news_collect_sec_edgar_gdelt_200symbol_dry_run.yaml",
        overlay_project_config=True,
    )

    for config in (gdelt, combined):
        ml = config["ml"]
        collect = ml["stock_alpha_news_collect"]
        providers = collect["providers"]
        assert len(collect["symbols"]) == 200
        assert len(set(collect["symbols"])) == 200
        assert collect["dry_run"] is True
        assert collect["allow_overwrite"] is False
        assert collect["merge_existing"] is False
        assert collect["backup_existing"] is False
        assert collect["start_date"] == "2023-12-01"
        assert collect["end_date"] == "2026-04-20"
        assert collect["provider_request_limit"] == 50
        assert collect["max_rows_per_symbol"] == 5
        assert collect["symbols_per_batch"] == 25
        assert collect["rate_limit_sleep_seconds"] == 1
        assert providers["gdelt"] == {"enabled": True}
        assert providers["alpha_vantage"]["enabled"] is False
        assert providers["fmp"]["enabled"] is False
        assert providers["newsapi"]["enabled"] is False
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False

    assert gdelt["ml"]["stock_alpha_news_collect"]["max_rows_per_provider"] == 500
    assert gdelt["ml"]["stock_alpha_news_collect"]["providers"]["sec_edgar"]["enabled"] is False
    assert combined["ml"]["stock_alpha_news_collect"]["max_rows_per_provider"] == 1000
    assert combined["ml"]["stock_alpha_news_collect"]["providers"]["sec_edgar"] == {"enabled": True}


def test_daily_confirmation_config_is_research_only_and_bounded():
    config = load_config(
        "config/config.stock_alpha_news_daily_confirmation_alpha_sec_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    confirmation = ml["stock_alpha_news_confirmation"]
    providers = confirmation["providers"]

    assert confirmation["enabled"] is True
    assert confirmation["dry_run"] is True
    assert confirmation["inspection_only"] is True
    assert confirmation["lookback_hours"] == 72
    assert confirmation["max_symbols"] == 20
    assert confirmation["max_articles_per_symbol"] == 5
    assert confirmation["max_provider_requests"] == 20
    assert confirmation["symbols"] == ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]
    assert providers["alpha_vantage"] == {
        "enabled": True,
        "api_key_env": "ALPHA_VANTAGE_API_KEY",
    }
    assert providers["sec_edgar"] == {"enabled": True}
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False
    assert ml["promotion_thresholds_changed"] is False


def test_company_press_release_rss_config_is_dry_run_only_with_verified_official_feeds():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    providers = collect["providers"]
    rss = providers["company_press_release_rss"]

    assert collect["enabled"] is True
    assert collect["dry_run"] is True
    assert collect["allow_overwrite"] is False
    assert collect["merge_existing"] is False
    assert collect["backup_existing"] is False
    assert collect["symbols"] == ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]
    assert collect["symbols_per_batch"] == 7
    assert collect["provider_request_limit"] == 140
    assert collect["max_rows_per_provider"] == 200
    assert collect["rate_limit_sleep_seconds"] == 0.5
    assert rss["enabled"] is True
    assert rss["max_rows_per_feed"] == 20
    assert "api_key_env" not in rss
    assert sorted(rss["feeds"]) == ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    assert rss["feeds"]["AAPL"] == [{
        "name": "Apple Newsroom",
        "url": "https://www.apple.com/newsroom/rss-feed.rss",
        "enabled": True,
        "event_type": "rss_news",
        "verified_source_url": "https://www.apple.com/newsroom/",
    }]
    assert rss["feeds"]["MSFT"] == [{
        "name": "Microsoft Source",
        "url": "https://news.microsoft.com/source/feed/",
        "enabled": True,
        "event_type": "rss_news",
        "verified_source_url": "https://news.microsoft.com/source/",
    }]
    assert rss["feeds"]["NVDA"] == [{
        "name": "NVIDIA Press Room",
        "url": "https://nvidianews.nvidia.com/releases.xml",
        "enabled": True,
        "event_type": "press_release",
        "verified_source_url": "https://www.nvidia.com/en-us/about-nvidia/rss/",
    }]
    assert rss["feeds"]["AMZN"] == [{
        "name": "Amazon News",
        "url": "https://www.aboutamazon.com/rss/feed.rss",
        "enabled": True,
        "event_type": "rss_news",
        "verified_source_url": "https://www.aboutamazon.com/news",
    }]
    assert rss["feeds"]["META"] == [{
        "name": "Meta Newsroom",
        "url": "https://about.fb.com/news/feed/",
        "enabled": True,
        "event_type": "rss_news",
        "verified_source_url": "https://about.fb.com/news/",
    }]
    assert rss["feeds"]["GOOGL"] == [{
        "name": "Google Blog Alphabet",
        "url": "https://blog.google/alphabet/rss/",
        "enabled": True,
        "event_type": "rss_news",
        "verified_source_url": "https://blog.google/alphabet/",
    }]
    assert rss["feeds"]["TSLA"][0]["enabled"] is False
    assert not str(rss["feeds"]["TSLA"][0].get("url", "")).strip()
    assert rss["feeds"]["TSLA"][0]["event_type"] == "press_release"
    assert rss["feeds"]["TSLA"][0]["verified_source_url"] == "https://ir.tesla.com/press"
    assert "no official RSS feed URL was verified" in rss["feeds"]["TSLA"][0]["note"]
    for name, provider in providers.items():
        if name != "company_press_release_rss":
            assert provider["enabled"] is False
    assert "stock_alpha_news_features_path" not in ml
    assert "stock_alpha_news_readiness_preflight_output_dir" not in ml
    assert "stock_alpha_news_source_diagnostics_report_dir" not in ml
    assert ml.get("model_type") != "news_analysis_transformer"
    assert ml.get("shadow_model_type") != "news_analysis_transformer"
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False
    assert ml["promotion_thresholds_changed"] is False
