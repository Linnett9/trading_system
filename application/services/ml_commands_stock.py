from core.research.ml.stock_level_model_ranking_benchmark import write_stock_level_model_ranking_benchmark
from core.research.ml.stock_level_feature_attribution import write_stock_level_feature_attribution
from core.research.ml.stock_level_alpha_features import write_stock_level_alpha_features
from core.research.ml.stock_level.overnight_stock_alpha_runner import write_overnight_stock_alpha_experiment
from core.research.ml.stock_level.stock_level_target_comparison import write_stock_level_target_comparison
from core.research.ml.stock_level.stock_level_portfolio_replay import write_stock_level_portfolio_replay
from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import write_stock_level_portfolio_policy_sweep
from core.research.ml.stock_level.stock_alpha_experiment_report import write_stock_alpha_experiment_report
from core.research.ml.stock_level.stock_alpha_candidate_report import write_stock_alpha_candidate_report
from core.research.ml.stock_level.stock_alpha_deep_model_diagnostics import write_stock_alpha_deep_model_diagnostics
from core.research.ml.stock_level.stock_alpha_dev_smoke import write_stock_alpha_dev_smoke
from core.research.ml.stock_level.stock_alpha_ensemble import write_stock_alpha_ensemble
from core.research.ml.stock_level.stock_alpha_ensemble_portfolio_sweep import write_stock_alpha_ensemble_portfolio_sweep
from core.research.ml.stock_level.stock_alpha_experiment_preflight import write_stock_alpha_experiment_preflight
from core.research.ml.stock_level.stock_alpha_news_contract import write_stock_alpha_news_features_from_config
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import write_stock_alpha_news_contract_ingest
from core.research.ml.stock_level.stock_alpha_news_coverage_audit import write_stock_alpha_news_coverage_audit
from core.research.ml.stock_level.stock_alpha_news_provider_audit import write_stock_alpha_news_provider_audit
from core.research.ml.stock_level.stock_alpha_news_readiness_preflight import write_stock_alpha_news_readiness_preflight
from core.research.ml.stock_level.stock_alpha_parallelism_audit import write_stock_alpha_parallelism_audit
from core.research.ml.stock_level.run_manifest.service import write_stock_alpha_run_status


def run_ml_stock_level_alpha_benchmark(config):
    result = write_stock_level_model_ranking_benchmark(config)
    print("\nSTOCK-LEVEL ALPHA BENCHMARK SUITE")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Leaderboard CSV: {result.csv_path}")
    print(f"Leaderboard JSON: {result.json_path}")
    print(f"Leaderboard Markdown: {result.markdown_path}")
    print(f"OOS predictions: {result.predictions_path}")

def run_ml_stock_level_target_comparison(config):
    result = write_stock_level_target_comparison(config)
    print("\nSTOCK-LEVEL TARGET COMPARISON")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_level_portfolio_replay(config):
    result = write_stock_level_portfolio_replay(config)
    print("\nSTOCK-LEVEL PORTFOLIO REPLAY")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Summary CSV: {result.csv_path}")
    print(f"Summary JSON: {result.json_path}")
    print(f"Summary Markdown: {result.markdown_path}")
    print(f"Equity curves: {result.equity_curves_path}")
    print(f"Holdings: {result.holdings_path}")

def run_ml_stock_level_portfolio_policy_sweep(config):
    result = write_stock_level_portfolio_policy_sweep(config)
    print("\nSTOCK-LEVEL PORTFOLIO POLICY SWEEP")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Equity curves: {result.equity_curves_path}")
    print(f"Top holdings: {result.top_holdings_path}")

def run_ml_stock_alpha_experiment_report(config):
    result = write_stock_alpha_experiment_report(config)
    print("\nSTOCK-ALPHA EXPERIMENT REPORT")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Registry: {result.registry_path}")

def run_ml_stock_alpha_candidate_report(config):
    result = write_stock_alpha_candidate_report(config)
    ml = config.get("ml", {})
    print("\nSTOCK-ALPHA CANDIDATE REPORT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(f"Resolved run size: {ml.get('stock_alpha_run_size', 'benchmark')}")
    print(f"Output directory: {result.json_path.parent}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"CSV: {result.csv_path}")

def run_ml_stock_alpha_deep_diagnostics(config):
    result = write_stock_alpha_deep_model_diagnostics(config)
    print("\nSTOCK-ALPHA DEEP-MODEL DIAGNOSTICS")
    print("mode=research | run_size=dev | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"CSV: {result.csv_path}")

def run_ml_stock_alpha_ensemble(config):
    result = write_stock_alpha_ensemble(config)
    print("\nSTOCK-ALPHA AVERAGE-RANK ENSEMBLE")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Predictions CSV: {result.predictions_path}")
    print(f"Evaluation JSON: {result.json_path}")
    print(f"Leaderboard CSV: {result.leaderboard_csv_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_ensemble_portfolio_sweep(config):
    result = write_stock_alpha_ensemble_portfolio_sweep(config)
    print("\nSTOCK-ALPHA ENSEMBLE PORTFOLIO POLICY SWEEP")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Equity curves: {result.equity_curves_path}")
    print(f"Holdings: {result.holdings_path}")
    print(f"Trades: {result.trades_path}")

def run_ml_stock_alpha_experiment_preflight(config):
    result = write_stock_alpha_experiment_preflight(config)
    print("\nSTOCK-ALPHA EXPERIMENT PREFLIGHT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_features(config):
    result = write_stock_alpha_news_features_from_config(config)
    print("\nSTOCK-ALPHA NEWS FEATURE AGGREGATION")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Features CSV: {result.features_csv_path}")
    print(f"Audit JSON: {result.audit_json_path}")
    print(f"Audit Markdown: {result.audit_markdown_path}")

def run_ml_stock_alpha_news_contract_ingest(config):
    print("\nSTOCK-ALPHA NEWS CONTRACT INGEST")
    print("mode=research | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_contract_ingest(config)
    except (FileNotFoundError, ValueError) as exc:
        print("safe_to_generate_features=false")
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    print(f"Contract CSV: {result.contract_path}")
    print(f"Audit JSON: {result.audit_json_path}")
    print(f"Audit Markdown: {result.audit_markdown_path}")

def run_ml_stock_alpha_news_provider_audit(config):
    print("\nSTOCK-ALPHA NEWS PROVIDER AUDIT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    result = write_stock_alpha_news_provider_audit(config)
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"safe_for_pit_research={str(payload.get('safe_for_pit_research', False)).lower()}")
    for issue in payload.get("blocking_issues", []):
        print(f"blocking_issue={issue}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_coverage_audit(config):
    print("\nSTOCK-ALPHA NEWS COVERAGE AUDIT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    result = write_stock_alpha_news_coverage_audit(config)
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"safe_for_feature_generation={str(payload.get('safe_for_feature_generation', False)).lower()}")
    for issue in payload.get("blocking_issues", []):
        print(f"blocking_issue={issue}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_readiness_preflight(config):
    result = write_stock_alpha_news_readiness_preflight(config)
    print("\nSTOCK-ALPHA NEWS READINESS PREFLIGHT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_dev_smoke(config):
    result = write_stock_alpha_dev_smoke(config)
    print("\nSTOCK-ALPHA DEV SMOKE")
    print("mode=research | run_size=dev | production_validated=false")
    print(f"Report: {result}")

def run_ml_stock_alpha_parallelism_audit(config):
    result = write_stock_alpha_parallelism_audit(config)
    print("\nSTOCK-ALPHA PARALLELISM AUDIT")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_run_status(config):
    result = write_stock_alpha_run_status(config)
    print("\nSTOCK-ALPHA RUN STATUS")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_level_feature_attribution(config):
    result = write_stock_level_feature_attribution(config)
    print("\nSTOCK-LEVEL FEATURE ATTRIBUTION")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_level_alpha_features(config):
    result = write_stock_level_alpha_features(config)
    print("\nSTOCK-LEVEL ALPHA FEATURES")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Enriched artifact: {result.enriched_csv_path}")
    print(f"Audit CSV: {result.audit_csv_path}")
    print(f"Audit JSON: {result.audit_json_path}")
    print(f"Audit Markdown: {result.audit_markdown_path}")

def run_ml_overnight_stock_alpha(config):
    result = write_overnight_stock_alpha_experiment(config)
    print("\nOVERNIGHT STOCK-ALPHA EXPERIMENT")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Summary JSON: {result.json_path}")
    print(f"Summary Markdown: {result.markdown_path}")
