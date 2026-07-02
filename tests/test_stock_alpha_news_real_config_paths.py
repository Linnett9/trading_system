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
