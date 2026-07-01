from pathlib import Path

from application.services.ml_commands_artifacts import _refresh_trading_research_leaderboard
from core.research.ml.return_mechanics_audit import write_return_mechanics_audit
from core.research.ml.champion_baseline_audit import write_champion_baseline_audit
from core.research.ml.replay.canonical_continuous_equity_replay import write_canonical_continuous_equity_replay
from core.research.ml.data_anomaly_quarantine import write_data_anomaly_quarantine
from core.research.ml.profit_concentration_audit import write_profit_concentration_audit
from core.research.ml.data_adjustment_validation import write_clean_data_replay, write_data_adjustment_audit, write_independent_period_validation
from core.research.ml.adjusted_data_comparison import write_adjusted_data_comparison, write_adjusted_price_replay
from core.research.ml.adjusted_replay_alignment_audit import write_adjusted_replay_alignment_audit
from core.research.ml.independent_period_expansion_audit import write_independent_period_expansion_audit
from core.research.ml.historical_coverage_audit import write_historical_coverage_audit
from core.research.ml.stock_level_prediction_artifacts import write_stock_level_prediction_artifacts
from core.research.ml.metrics.cross_sectional_ranking_diagnostics import write_cross_sectional_ranking_diagnostics
from core.research.ml.benchmark_relative_validation import write_benchmark_relative_validation
from core.research.ml.model_contract_audit import write_model_contract_audit
from infrastructure.data.stooq_parquet_data_feed import StooqParquetDataFeed


def run_ml_return_mechanics_audit(config):
    result = write_return_mechanics_audit(config)
    champion_result = write_champion_baseline_audit(config)
    canonical_result = write_canonical_continuous_equity_replay(config)
    anomaly_result = write_data_anomaly_quarantine(config)
    concentration_result = write_profit_concentration_audit(config)
    research_feed = StooqParquetDataFeed(
        str(
            config.get("ml", {}).get(
                "stooq_parquet_dir",
                "data/processed/stooq_parquet",
            )
        )
    )
    adjustment_result = write_data_adjustment_audit(config)
    independent_result = write_independent_period_validation(config)
    clean_replay_result = write_clean_data_replay(config, research_feed)
    adjusted_comparison_result = write_adjusted_data_comparison(config)
    adjusted_replay_result = write_adjusted_price_replay(config)
    adjusted_alignment_result = write_adjusted_replay_alignment_audit(config)
    expansion_result = write_independent_period_expansion_audit(config)
    historical_coverage_result = write_historical_coverage_audit(config)
    stock_artifact_result = write_stock_level_prediction_artifacts(config)
    ranking_result = write_cross_sectional_ranking_diagnostics(config)
    validation_result = write_benchmark_relative_validation(
        config,
        research_feed,
    )
    leaderboard_result = _refresh_trading_research_leaderboard(config)
    print("\nML RETURN MECHANICS AUDIT")
    print("mode=research | trading_impact=none")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Champion baseline CSV: {champion_result.csv_path}")
    print(f"Champion baseline JSON: {champion_result.json_path}")
    print(f"Champion baseline Markdown: {champion_result.markdown_path}")
    print(f"Canonical replay JSON: {canonical_result.json_path}")
    print(f"Anomaly quarantine JSON: {anomaly_result.json_path}")
    print(f"Profit concentration JSON: {concentration_result.json_path}")
    print(f"Data adjustment audit JSON: {adjustment_result.json_path}")
    print(f"Independent-period validation JSON: {independent_result.json_path}")
    print(f"Clean-data replay JSON: {clean_replay_result.json_path}")
    print(f"Adjusted data comparison JSON: {adjusted_comparison_result.json_path}")
    print(f"Adjusted price replay JSON: {adjusted_replay_result.json_path}")
    print(f"Adjusted replay alignment JSON: {adjusted_alignment_result.json_path}")
    print(f"Independent period expansion JSON: {expansion_result.json_path}")
    print(f"Historical coverage audit JSON: {historical_coverage_result.json_path}")
    print(f"Stock-level prediction artifacts JSON: {stock_artifact_result.json_path}")
    print(f"Cross-sectional ranking diagnostics JSON: {ranking_result.json_path}")
    print(f"Benchmark-relative validation JSON: {validation_result.json_path}")
    print(f"Promotion readiness: {validation_result.promotion_readiness_path}")
    print(f"Trading research leaderboard JSON: {leaderboard_result.json_path}")

def run_ml_model_contract_audit(config):
    output_dir = config.get("reports", {}).get(
        "ml_dir",
        config.get("ml", {}).get("output_dir", "reports/ml"),
    )
    markdown_path, json_path = write_model_contract_audit(output_dir)
    print("\nML MODEL CONTRACT AUDIT")
    print("mode=research | trading_impact=none")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
