from __future__ import annotations

from pathlib import Path

import yaml

from config.config_loader import load_config
from scripts.stock_alpha_news_universe_batches import build_universe_batches


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
    assert collect["max_symbols_per_run"] == 7
    assert collect["only_symbols"] == []
    assert collect["provider_request_limit"] == 140
    assert collect["max_rows_per_provider"] == 200
    assert collect["rate_limit_sleep_seconds"] == 0.5
    assert rss["enabled"] is True
    assert rss["max_rows_per_feed"] == 20
    assert rss["max_enabled_feeds_per_run"] == 6
    assert rss["skip_known_error_feeds"] is False
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


def test_company_press_release_rss_registry_and_25_symbol_config_are_dry_run_only():
    registry = yaml.safe_load(
        Path("config/news_source_registry.stock_alpha_rss.yaml").read_text(encoding="utf-8")
    )
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_25symbol_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    rss = collect["providers"]["company_press_release_rss"]

    expected_symbols = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "AVGO", "BRK.B", "JPM", "V", "MA", "XOM", "UNH", "COST",
        "HD", "PG", "JNJ", "ABBV", "NFLX", "CRM", "AMD", "ORCL",
        "BAC", "KO",
    ]
    known_error_symbols = ["AVGO", "JPM", "V", "XOM", "ABBV", "ORCL"]
    verified_symbols = [
        symbol for symbol in expected_symbols
        if registry[symbol]["status"] == "verified"
    ]
    disabled_symbols = [
        symbol for symbol in expected_symbols
        if registry[symbol]["status"] != "verified"
    ]

    assert collect["enabled"] is True
    assert collect["dry_run"] is True
    assert collect["allow_overwrite"] is False
    assert collect["merge_existing"] is False
    assert collect["backup_existing"] is False
    assert collect["source_registry_path"] == "config/news_source_registry.stock_alpha_rss.yaml"
    assert collect["symbols"] == expected_symbols
    assert len(collect["symbols"]) == 25
    assert collect["symbols_per_batch"] == 25
    assert collect["max_symbols_per_run"] == 25
    assert collect["only_symbols"] == []
    assert collect["provider_request_limit"] == 250
    assert collect["max_rows_per_provider"] == 250
    assert collect["max_rows_per_feed"] == 10
    assert rss["enabled"] is True
    assert rss["max_rows_per_feed"] == 10
    assert rss["max_enabled_feeds_per_run"] == 15
    assert rss["skip_known_error_feeds"] is True
    assert set(expected_symbols) <= {
        symbol for symbol in registry if not symbol.startswith("_")
    }
    assert len(verified_symbols) == 21
    assert disabled_symbols == ["TSLA", "BRK.B", "UNH", "BAC"]
    for symbol in verified_symbols:
        assert registry[symbol]["sources"]
        source = registry[symbol]["sources"][0]
        assert source["official"] is True
        assert source["enabled"] is True
        assert str(source["url"]).startswith("https://")
        feed = rss["feeds"][symbol][0]
        assert feed["enabled"] is True
        assert feed["official"] is True
        assert feed["url"] == source["url"]
        if symbol in known_error_symbols:
            assert source["known_error"] is True
            assert source["last_error_observed"] == "2026-07-02"
            assert feed["known_error"] is True
            assert feed["last_error_observed"] == "2026-07-02"
        else:
            assert source.get("known_error") is not True
            assert feed.get("known_error") is not True
    for symbol in disabled_symbols:
        assert registry[symbol]["sources"] == []
        assert rss["feeds"][symbol][0]["enabled"] is False
        assert not str(rss["feeds"][symbol][0].get("url", "")).strip()
    for name, provider in collect["providers"].items():
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


def test_company_press_release_rss_registry_covers_canonical_379_universe():
    from core.research.ml.stock_level.news_sources import load_validated_rss_registry

    symbols, feeds, report = load_validated_rss_registry(
        "data/reference/universes/us_liquid_500.yaml",
        "config/news_source_registry.stock_alpha_rss.yaml",
    )

    assert len(symbols) == len(set(symbols)) == 379
    assert report["registry_complete"] is True
    assert sum(report["classification_counts"].values()) == 379
    assert report["classification_counts"]["verified_rss_feed"] == 184
    assert report["classification_counts"]["disabled_pending_review"] == 187
    assert report["classification_counts"]["known_error_feed"] == 6
    assert report["classification_counts"]["no_verified_official_rss"] == 2
    assert report["known_error_feed_symbols"] == ["ABBV", "AVGO", "JPM", "ORCL", "V", "XOM"]
    assert report["sec_only_candidate_symbols"] == []
    assert {"CSCO", "IBM", "LRCX", "QCOM"} <= set(report["verified_rss_feed_symbols"])
    assert {"INTU", "KLAC", "NKE", "TMO", "VZ"} <= set(report["verified_rss_feed_symbols"])
    assert {"DHR", "FCX", "NEM", "SPGI", "TER"} <= set(report["verified_rss_feed_symbols"])
    assert {"INTC", "CAT", "BA", "GE", "PFE", "GLW", "SNPS", "SCHW"} <= set(report["verified_rss_feed_symbols"])
    assert {"HON", "GILD", "LMT", "SBUX", "AJG", "SRE", "AFL", "BRO", "THC", "GAP"} <= set(report["verified_rss_feed_symbols"])
    assert {"PGR", "MDT", "BLK", "SLB", "BMY", "LOW", "TGT", "CCL", "AZO"} <= set(report["verified_rss_feed_symbols"])
    assert {"UPS", "MCK", "SYK", "SHW", "ELV", "CVS", "FISV", "ADP", "DAL", "TT"} <= set(report["verified_rss_feed_symbols"])
    assert {"MO", "PH", "FDX", "CSX", "AMT", "JCI", "MMM", "CB", "SO"} <= set(report["verified_rss_feed_symbols"])
    assert {"COR", "WMB", "KR", "DVN", "DUK", "RF", "GD", "MNST"} <= set(report["verified_rss_feed_symbols"])
    assert {"FHN", "MAS", "CCEP", "BAX", "HAS", "EQR", "FLR", "AEO", "SWK", "BBWI", "VIAV", "IFF"} <= set(report["verified_rss_feed_symbols"])
    assert {"GEN", "WHR", "NYT", "SANM", "TXT", "EMN", "BRKR", "LUMN", "UNM", "HSIC"} <= set(report["verified_rss_feed_symbols"])
    assert {"AAP", "KSS", "ZION", "JHG", "AOS", "AMG", "AN", "AIT", "AGCO", "CALM", "ADC", "ARW", "AIZ", "BIO", "HRB"} <= set(report["verified_rss_feed_symbols"])
    assert {"CAKE", "MAT", "R", "PTEN", "KBH", "BPOP", "AFG", "AVT", "AXS", "ATR", "ACLS", "BC", "CBSH", "NOVT", "GNTX", "MTG"} <= set(report["verified_rss_feed_symbols"])
    assert {"CBRL", "BFH"} <= set(report["verified_rss_feed_symbols"])
    assert set(feeds) == set(report["verified_rss_feed_symbols"] + report["known_error_feed_symbols"])
    for symbol, symbol_feeds in feeds.items():
        for feed in symbol_feeds:
            assert feed["official"] is True
            assert feed["enabled"] is True
            assert str(feed["url"]).startswith("https://")
            assert str(feed["verified_source_url"]).startswith("https://")
            assert feed.get("known_error", False) is (symbol in report["known_error_feed_symbols"])


def test_company_press_release_rss_379_config_is_bounded_static_registry_dry_run():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    rss = collect["providers"]["company_press_release_rss"]

    assert collect["dry_run"] is True
    assert collect["output_written"] is False
    assert collect["universe_path"] == "data/reference/universes/us_liquid_500.yaml"
    assert collect["load_feeds_from_registry"] is True
    assert collect["max_symbols_per_run"] == 25
    assert collect["only_symbols"] == []
    assert collect["request_timeout_seconds"] == 20
    assert rss["max_enabled_feeds_per_run"] == 10
    assert rss["skip_known_error_feeds"] is True
    for name, provider in collect["providers"].items():
        assert provider["enabled"] is (name == "company_press_release_rss")
    assert "stock_alpha_news_features_path" not in ml
    assert "stock_alpha_news_readiness_preflight_output_dir" not in ml
    assert "stock_alpha_news_source_diagnostics_report_dir" not in ml
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["research_only"] is True
    assert ml["trading_impact"] == "none"
    assert ml["production_validated"] is False


def test_company_press_release_rss_batch_02_config_is_bounded_and_dry_run_only():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_02_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    expected = [
        "BRK-B", "V", "AMAT", "XLF", "COST", "QCOM", "LRCX", "ASML", "XLE",
        "NOW", "BAC", "XLV", "GS", "CSCO", "MA", "XLI", "JNJ", "CVX", "CAT",
        "BA", "IBM", "TXN", "GE", "PANW", "C",
    ]

    assert collect["dry_run"] is True
    assert collect["output_written"] is False
    assert collect["only_symbols"] == expected
    assert collect["max_symbols_per_run"] == 25
    assert collect["providers"]["company_press_release_rss"]["skip_known_error_feeds"] is True
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["trading_impact"] == "none"


def test_company_press_release_rss_batch_03_config_is_bounded_and_dry_run_only():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_03_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    expected = [
        "HD", "ADBE", "INTU", "KLAC", "PG", "BKNG", "WFC", "ABBV", "XLP",
        "PFE", "MRK", "XLY", "KO", "ACN", "APH", "PEP", "GLW", "ADI", "TMO",
        "VZ", "T", "NKE", "DIS", "SNPS", "MCD",
    ]

    assert collect["dry_run"] is True
    assert collect["output_written"] is False
    assert collect["only_symbols"] == expected
    assert collect["max_symbols_per_run"] == 25
    assert collect["providers"]["company_press_release_rss"]["skip_known_error_feeds"] is True
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["trading_impact"] == "none"


def test_company_press_release_rss_batch_04_config_is_bounded_and_dry_run_only():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_04_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    expected = [
        "MS", "XLU", "ABT", "BSX", "AXP", "ETN", "NEM", "SCHW", "RTX", "COF",
        "HON", "FCX", "F", "CMCSA", "GILD", "SPGI", "AMGN", "NEE", "TER", "UNP",
        "DHR", "SBUX", "COP", "LMT", "CIEN",
    ]

    assert collect["dry_run"] is True
    assert collect["output_written"] is False
    assert collect["only_symbols"] == expected
    assert collect["max_symbols_per_run"] == 25
    assert collect["providers"]["company_press_release_rss"]["skip_known_error_feeds"] is True
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["trading_impact"] == "none"


def test_company_press_release_rss_batch_05_config_is_bounded_and_dry_run_only():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_05_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    expected = [
        "TJX", "PGR", "MDT", "BLK", "SLB", "BMY", "LOW", "TGT", "CDNS", "MCHP",
        "CCL", "AZO", "UPS", "MCK", "XLB", "SYK", "SHW", "ELV", "CVS", "FISV",
        "ADP", "OXY", "DAL", "TT", "B",
    ]

    assert collect["dry_run"] is True
    assert collect["output_written"] is False
    assert collect["only_symbols"] == expected
    assert collect["max_symbols_per_run"] == 25
    assert collect["providers"]["company_press_release_rss"]["max_enabled_feeds_per_run"] == 19
    assert collect["providers"]["company_press_release_rss"]["skip_known_error_feeds"] is True
    assert ml["stock_alpha_news_enable_transformer"] is False
    assert ml["trading_impact"] == "none"


def test_company_press_release_rss_batch_write_raw_configs_are_bounded_and_safe():
    configs = [
        (
            "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_01_write_raw.yaml",
            [
                "SPY", "QQQ", "NVDA", "TSLA", "MU", "AAPL", "MSFT", "AMD", "AMZN",
                "META", "GOOGL", "AVGO", "INTC", "GLD", "ORCL", "NFLX", "MRVL",
                "UNH", "LLY", "TLT", "JPM", "WMT", "XOM", "XLK", "CRM",
            ],
            10,
        ),
        (
            "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_02_write_raw.yaml",
            [
                "BRK-B", "V", "AMAT", "XLF", "COST", "QCOM", "LRCX", "ASML", "XLE",
                "NOW", "BAC", "XLV", "GS", "CSCO", "MA", "XLI", "JNJ", "CVX", "CAT",
                "BA", "IBM", "TXN", "GE", "PANW", "C",
            ],
            10,
        ),
        (
            "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_03_write_raw.yaml",
            [
                "HD", "ADBE", "INTU", "KLAC", "PG", "BKNG", "WFC", "ABBV", "XLP",
                "PFE", "MRK", "XLY", "KO", "ACN", "APH", "PEP", "GLW", "ADI", "TMO",
                "VZ", "T", "NKE", "DIS", "SNPS", "MCD",
            ],
            11,
        ),
        (
            "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_04_write_raw.yaml",
            [
                "MS", "XLU", "ABT", "BSX", "AXP", "ETN", "NEM", "SCHW", "RTX", "COF",
                "HON", "FCX", "F", "CMCSA", "GILD", "SPGI", "AMGN", "NEE", "TER", "UNP",
                "DHR", "SBUX", "COP", "LMT", "CIEN",
            ],
            10,
        ),
        (
            "config/config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_05_write_raw.yaml",
            [
                "TJX", "PGR", "MDT", "BLK", "SLB", "BMY", "LOW", "TGT", "CDNS", "MCHP",
                "CCL", "AZO", "UPS", "MCK", "XLB", "SYK", "SHW", "ELV", "CVS", "FISV",
                "ADP", "OXY", "DAL", "TT", "B",
            ],
            19,
        ),
    ]

    for config_path, expected_symbols, expected_feed_cap in configs:
        config = load_config(config_path, overlay_project_config=True)
        ml = config["ml"]
        collect = ml["stock_alpha_news_collect"]
        rss = collect["providers"]["company_press_release_rss"]

        assert collect["dry_run"] is False
        assert collect["output_written"] is False
        assert collect["allow_overwrite"] is False
        assert collect["merge_existing"] is True
        assert collect["backup_existing"] is True
        assert ml["stock_alpha_news_collect_output_path"] == "data/news/raw/stock_alpha_news_provider_export.csv"
        assert collect["only_symbols"] == expected_symbols
        assert collect["max_symbols_per_run"] == 25
        assert collect["load_feeds_from_registry"] is True
        assert collect["source_registry_path"] == "config/news_source_registry.stock_alpha_rss.yaml"
        assert rss["enabled"] is True
        assert rss["max_enabled_feeds_per_run"] == expected_feed_cap
        assert rss["skip_known_error_feeds"] is True
        for name, provider in collect["providers"].items():
            assert provider["enabled"] is (name == "company_press_release_rss")
        assert "stock_alpha_news_features_path" not in ml
        assert "stock_alpha_news_readiness_preflight_output_dir" not in ml
        assert "stock_alpha_news_source_diagnostics_report_dir" not in ml
        assert ml["stock_alpha_news_enable_transformer"] is False
        assert ml["research_only"] is True
        assert ml["trading_impact"] == "none"
        assert ml["production_validated"] is False
        assert ml["promotion_thresholds_changed"] is False


def test_company_press_release_rss_batch_06_to_16_configs_are_bounded_and_safe():
    batches = build_universe_batches("data/reference/universes/us_liquid_500.yaml", 25)

    for batch_index in range(6, 17):
        expected_symbols = batches[batch_index - 1]
        for kind, expected_dry_run in (("dry_run", True), ("write_raw", False)):
            config_path = (
                "config/"
                f"config.stock_alpha_news_collect_company_press_release_rss_379symbol_batch_{batch_index:02d}_{kind}.yaml"
            )
            config = load_config(config_path, overlay_project_config=True)
            ml = config["ml"]
            collect = ml["stock_alpha_news_collect"]
            rss = collect["providers"]["company_press_release_rss"]

            assert collect["dry_run"] is expected_dry_run
            assert collect["output_written"] is False
            assert collect["allow_overwrite"] is False
            assert collect["merge_existing"] is (not expected_dry_run)
            assert collect["backup_existing"] is (not expected_dry_run)
            assert collect["only_symbols"] == expected_symbols
            assert collect["max_symbols_per_run"] == 25
            assert collect["load_feeds_from_registry"] is True
            assert collect["source_registry_path"] == "config/news_source_registry.stock_alpha_rss.yaml"
            assert ml["stock_alpha_news_collect_output_path"] == "data/news/raw/stock_alpha_news_provider_export.csv"
            assert rss["enabled"] is True
            assert rss["skip_known_error_feeds"] is True
            for name, provider in collect["providers"].items():
                assert provider["enabled"] is (name == "company_press_release_rss")
            assert "stock_alpha_news_features_path" not in ml
            assert "stock_alpha_news_readiness_preflight_output_dir" not in ml
            assert "stock_alpha_news_source_diagnostics_report_dir" not in ml
            assert ml["stock_alpha_news_enable_transformer"] is False
            assert ml["research_only"] is True
            assert ml["trading_impact"] == "none"
            assert ml["production_validated"] is False
            assert ml["promotion_thresholds_changed"] is False


def test_company_press_release_rss_25_symbol_errors_only_config_targets_known_error_feeds():
    config = load_config(
        "config/config.stock_alpha_news_collect_company_press_release_rss_25symbol_errors_only_dry_run.yaml",
        overlay_project_config=True,
    )
    ml = config["ml"]
    collect = ml["stock_alpha_news_collect"]
    rss = collect["providers"]["company_press_release_rss"]
    expected_symbols = ["ABBV", "AVGO", "JPM", "ORCL", "V", "XOM"]

    assert collect["enabled"] is True
    assert collect["dry_run"] is True
    assert collect["allow_overwrite"] is False
    assert collect["merge_existing"] is False
    assert collect["backup_existing"] is False
    assert collect["source_registry_path"] == "config/news_source_registry.stock_alpha_rss.yaml"
    assert collect["symbols"] == expected_symbols
    assert collect["only_symbols"] == expected_symbols
    assert collect["max_symbols_per_run"] == 6
    assert collect["symbols_per_batch"] == 6
    assert collect["provider_request_limit"] == 60
    assert collect["max_rows_per_provider"] == 60
    assert collect["max_rows_per_feed"] == 10
    assert collect["rate_limit_sleep_seconds"] == 0
    assert collect["request_timeout_seconds"] == 12
    assert rss["enabled"] is True
    assert rss["max_rows_per_feed"] == 10
    assert rss["max_enabled_feeds_per_run"] == 6
    assert rss["skip_known_error_feeds"] is False
    assert sorted(rss["feeds"]) == sorted(expected_symbols)
    for symbol in expected_symbols:
        feed = rss["feeds"][symbol][0]
        assert feed["enabled"] is True
        assert feed["official"] is True
        assert feed["known_error"] is True
        assert feed["last_error_observed"] == "2026-07-02"
        assert str(feed["url"]).startswith("https://")
    for name, provider in collect["providers"].items():
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
