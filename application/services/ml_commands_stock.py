from pathlib import Path
from typing import Any, Mapping

from core.research.ml.stock_level.stock_alpha_news_contract import write_stock_alpha_news_features_from_config
from core.research.ml.stock_level.stock_alpha_news_feature_diagnostics import write_stock_alpha_news_feature_diagnostics
from core.research.ml.stock_level.stock_alpha_news_free_source_collect import write_stock_alpha_news_free_source_collect
from core.research.ml.stock_level.stock_alpha_news_collection_plan import write_stock_alpha_news_collection_plan
from core.research.ml.stock_level.stock_alpha_news_historical_backfill import write_stock_alpha_news_historical_backfill
from core.research.ml.stock_level.stock_alpha_news_daily_confirmation import write_stock_alpha_news_daily_confirmation
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import write_stock_alpha_news_contract_ingest
from core.research.ml.stock_level.stock_alpha_news_coverage_audit import write_stock_alpha_news_coverage_audit
from core.research.ml.stock_level.stock_alpha_news_pipeline_preflight import write_stock_alpha_news_pipeline_preflight
from core.research.ml.stock_level.stock_alpha_news_pipeline_inspect import write_stock_alpha_news_pipeline_inspect
from core.research.ml.stock_level.stock_alpha_news_provider_audit import write_stock_alpha_news_provider_audit
from core.research.ml.stock_level.stock_alpha_news_provider_sample_check import write_stock_alpha_news_provider_sample_check
from core.research.ml.stock_level.stock_alpha_news_readiness_preflight import write_stock_alpha_news_readiness_preflight
from core.research.ml.stock_level.news_risk_overlay_research import (
    format_news_risk_overlay_summary,
    inspect_stock_alpha_news_risk_overlay_results,
    write_stock_alpha_news_risk_overlay_parallel_benchmark,
    write_stock_alpha_news_risk_overlay_research,
)
from core.research.ml.stock_level.stock_alpha_news_source_diagnostics import write_stock_alpha_news_source_diagnostics
from core.research.ml.stock_level.stock_alpha_news_source_setup_check import write_stock_alpha_news_source_setup_check
import core.research.ml.stock_level.news_sources.historical_canonical_corpus as historical_canonical_corpus


def _stock_alpha_news_historical_backfill_action(config: Mapping[str, Any]) -> str:
    settings = dict(
        dict(config.get("ml", {}) or {}).get("stock_alpha_news_historical_backfill", {}) or {}
    )
    return str(settings.get("action", "collect")).strip().lower() or "collect"


def _stock_alpha_news_historical_backfill_payload_path(
    config: Mapping[str, Any],
    result: Any,
) -> Path | None:
    action = _stock_alpha_news_historical_backfill_action(config)
    if action in {"collect", "backfill", "collect_until_done", "collect_until_drained"}:
        return result.summary_json_path
    if action == "assemble":
        return result.assembly_json_path
    raise ValueError(f"unsupported historical backfill action: {action}")


def run_ml_stock_level_alpha_benchmark(config):
    from core.research.ml.stock_level_model_ranking_benchmark import (
        write_stock_level_model_ranking_benchmark,
    )

    result = write_stock_level_model_ranking_benchmark(config)
    print("\nSTOCK-LEVEL ALPHA BENCHMARK SUITE")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Leaderboard CSV: {result.csv_path}")
    print(f"Leaderboard JSON: {result.json_path}")
    print(f"Leaderboard Markdown: {result.markdown_path}")
    print(f"OOS predictions: {result.predictions_path}")

def run_ml_stock_level_target_comparison(config):
    from core.research.ml.stock_level.stock_level_target_comparison import (
        write_stock_level_target_comparison,
    )

    result = write_stock_level_target_comparison(config)
    print("\nSTOCK-LEVEL TARGET COMPARISON")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_level_portfolio_replay(config):
    from core.research.ml.stock_level.stock_level_portfolio_replay import (
        write_stock_level_portfolio_replay,
    )

    result = write_stock_level_portfolio_replay(config)
    print("\nSTOCK-LEVEL PORTFOLIO REPLAY")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Summary CSV: {result.csv_path}")
    print(f"Summary JSON: {result.json_path}")
    print(f"Summary Markdown: {result.markdown_path}")
    print(f"Equity curves: {result.equity_curves_path}")
    print(f"Holdings: {result.holdings_path}")

def run_ml_stock_level_portfolio_policy_sweep(config):
    from core.research.ml.stock_level.stock_level_portfolio_policy_sweep import (
        write_stock_level_portfolio_policy_sweep,
    )

    result = write_stock_level_portfolio_policy_sweep(config)
    print("\nSTOCK-LEVEL PORTFOLIO POLICY SWEEP")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Equity curves: {result.equity_curves_path}")
    print(f"Top holdings: {result.top_holdings_path}")

def run_ml_stock_alpha_experiment_report(config):
    from core.research.ml.stock_level.stock_alpha_experiment_report import (
        write_stock_alpha_experiment_report,
    )

    result = write_stock_alpha_experiment_report(config)
    print("\nSTOCK-ALPHA EXPERIMENT REPORT")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"Registry: {result.registry_path}")

def run_ml_stock_alpha_candidate_report(config):
    from core.research.ml.stock_level.stock_alpha_candidate_report import (
        write_stock_alpha_candidate_report,
    )

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
    from core.research.ml.stock_level.stock_alpha_deep_model_diagnostics import (
        write_stock_alpha_deep_model_diagnostics,
    )

    result = write_stock_alpha_deep_model_diagnostics(config)
    print("\nSTOCK-ALPHA DEEP-MODEL DIAGNOSTICS")
    print("mode=research | run_size=dev | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")
    print(f"CSV: {result.csv_path}")

def run_ml_stock_alpha_ensemble(config):
    from core.research.ml.stock_level.stock_alpha_ensemble import (
        write_stock_alpha_ensemble,
    )

    result = write_stock_alpha_ensemble(config)
    print("\nSTOCK-ALPHA AVERAGE-RANK ENSEMBLE")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Predictions CSV: {result.predictions_path}")
    print(f"Evaluation JSON: {result.json_path}")
    print(f"Leaderboard CSV: {result.leaderboard_csv_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_ensemble_portfolio_sweep(config):
    from core.research.ml.stock_level.stock_alpha_ensemble_portfolio_sweep import (
        write_stock_alpha_ensemble_portfolio_sweep,
    )

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
    from core.research.ml.stock_level.stock_alpha_experiment_preflight import (
        write_stock_alpha_experiment_preflight,
    )

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

def run_ml_stock_alpha_news_feature_diagnostics(config):
    print("\nSTOCK-ALPHA NEWS FEATURE DIAGNOSTICS")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_feature_diagnostics(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"next_action={payload['next_action']}")
    for issue in payload["blocking_issues"]:
        print(f"blocking_issue={issue}")
    print("features_generated=false")
    print("files_ingested=false")
    print("readiness_invoked=false")
    print("diagnostics_invoked=false")
    print("model_training_invoked=false")
    print("news_transformer_enabled=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_collect_free_sources(config):
    print("\nSTOCK-ALPHA FREE NEWS SOURCE COLLECTION")
    print("mode=research | collection_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_free_source_collect(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"dry_run={str(payload['dry_run']).lower()}")
    print(f"next_action={payload['next_action']}")
    print(f"providers_skipped_missing_key={','.join(payload['providers_skipped_missing_key']) or 'none'}")
    print(f"providers_returned_zero_rows={','.join(payload['providers_returned_zero_rows']) or 'none'}")
    print(f"providers_rate_limited={','.join(payload['providers_rate_limited']) or 'none'}")
    print(f"providers_failed={payload['providers_failed'] or 'none'}")
    print(f"provider_row_counts={payload['provider_row_counts']}")
    print(f"symbol_count={payload['symbol_count']}")
    print(f"duplicate_headline_count={payload['duplicate_headline_count']}")
    print(f"duplicate_headline_rate={payload['duplicate_headline_rate']}")
    print(f"output_written={str(payload['output_written']).lower()}")
    print("features_generated=false")
    print("readiness_invoked=false")
    print("diagnostics_invoked=false")
    print("model_training_invoked=false")
    print("news_transformer_enabled=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_collection_plan(config):
    print("\nSTOCK-ALPHA NEWS COLLECTION PLAN")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    result = write_stock_alpha_news_collection_plan(config)
    import json
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"next_action={payload['next_action']}")
    print(f"current_raw_export_row_count={payload['current_raw_export_row_count']}")
    print(f"current_raw_export_symbol_count={payload['current_raw_export_symbol_count']}")
    print(f"article_threshold_gap={payload['article_threshold_gap']}")
    print(f"symbol_threshold_gap={payload['symbol_threshold_gap']}")
    print("collection_invoked=false")
    print("raw_export_written=false")
    print("model_training_invoked=false")
    print("news_transformer_enabled=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_historical_backfill(config):
    print("\nSTOCK-ALPHA HISTORICAL NEWS BACKFILL")
    print("mode=research | collection_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_historical_backfill(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json
    action = _stock_alpha_news_historical_backfill_action(config)
    payload_path = _stock_alpha_news_historical_backfill_payload_path(config, result)
    if payload_path is None or not payload_path.is_file():
        raise FileNotFoundError(
            f"historical backfill action '{action}' expected artifact was not written: {payload_path}"
        )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    print(f"action={payload.get('action')}")
    if "status_counts" in payload:
        print(f"status_counts={payload['status_counts']}")
        print(f"processed_this_run={payload['processed_this_run']}")
        for iteration in payload.get("iteration_summaries", []) or []:
            print(
                "iteration={iteration} status_counts={status_counts} "
                "processed_this_run={processed_this_run} skipped_complete={skipped_complete} "
                "completed_this_run={completed_this_run} partial_this_run={partial_this_run} "
                "failed_this_run={failed_this_run}".format(**iteration)
            )
    else:
        print(f"row_count={payload['row_count']}")
        print(f"incomplete_partition_count={payload['incomplete_partition_count']}")
    print("contract_ingest_invoked=false")
    print("features_generated=false")
    print("model_training_invoked=false")
    print("news_transformer_enabled=false")
    print(f"Manifest: {result.manifest_path}")
    print(f"JSON: {payload_path}")

def run_ml_stock_alpha_news_canonical_corpus(config):
    print("\nSTOCK-ALPHA NEWS CANONICAL CORPUS")
    print("mode=research | offline_derived_stage=true | trading_impact=none | production_validated=false")
    try:
        manifest = historical_canonical_corpus.materialize_historical_canonical_corpus_from_config(
            _historical_canonical_corpus_config(config)
        )
    except historical_canonical_corpus.HistoricalCanonicalCorpusError as exc:
        print("canonical_corpus_written=false")
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    print(f"source_row_count={manifest['source_row_count']}")
    print(f"canonical_row_count={manifest['canonical_row_count']}")
    print(f"row_count_reconciled={str(manifest['row_count_reconciled']).lower()}")
    print("contract_ingest_invoked=false")
    print("features_generated=false")
    print("model_training_invoked=false")
    print("model_inference_invoked=false")
    print("news_transformer_enabled=false")
    print(f"Canonical CSV: {manifest['output_files']['canonical_corpus_csv']}")
    print(f"Manifest JSON: {manifest['output_files']['manifest_json']}")
    print(f"Audit JSON: {manifest['output_files']['audit_json']}")
    print(f"Summary Markdown: {manifest['output_files']['summary_markdown']}")
    return manifest

def _historical_canonical_corpus_config(config):
    settings = dict(
        dict(config.get("ml", {}) or {}).get("stock_alpha_news_canonical_corpus", {}) or {}
    )
    if not settings:
        raise historical_canonical_corpus.HistoricalCanonicalCorpusError(
            "ml.stock_alpha_news_canonical_corpus config is required"
        )
    return historical_canonical_corpus.HistoricalCanonicalCorpusConfig(
        source_assembly_csv_path=settings.get("source_assembly_csv_path", ""),
        source_assembly_metadata_json_path=settings.get("source_assembly_metadata_json_path", ""),
        output_dir=settings.get("output_dir", ""),
        expected_source_checksum=settings.get("expected_source_checksum", ""),
        canonical_schema_version=settings.get(
            "canonical_schema_version",
            historical_canonical_corpus.CANONICAL_NEWS_SCHEMA_VERSION,
        ),
        transformation_version=settings.get(
            "transformation_version",
            historical_canonical_corpus.HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
        ),
        write_enabled=bool(settings.get("write_enabled", False)),
        ingested_at_utc=settings.get("ingested_at_utc"),
    )

def run_ml_stock_alpha_news_daily_confirmation(config):
    print("\nSTOCK-ALPHA DAILY NEWS CONFIRMATION")
    print("mode=research | confirmation_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_daily_confirmation(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"symbols_checked={payload['symbol_count']}")
    print(f"providers_attempted={','.join(payload['providers_attempted']) or 'none'}")
    print(f"providers_skipped_missing_key={','.join(payload['providers_skipped_missing_key']) or 'none'}")
    print(f"providers_rate_limited={','.join(payload['providers_rate_limited']) or 'none'}")
    print(f"providers_failed={payload['providers_failed'] or 'none'}")
    print(f"symbols_requiring_review={','.join(payload['symbols_requiring_review']) or 'none'}")
    print("orders_generated=false")
    print("broker_invoked=false")
    print("features_generated=false")
    print("readiness_invoked=false")
    print("diagnostics_invoked=false")
    print("model_training_invoked=false")
    print("news_transformer_enabled=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

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

def run_ml_stock_alpha_news_provider_sample_check(config):
    print("\nSTOCK-ALPHA NEWS PROVIDER SAMPLE CHECK")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_provider_sample_check(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    compatible = payload["compatible_with_contract_ingest"]
    print(f"compatible_with_contract_ingest={str(compatible).lower()}")
    print(f"next_action={payload['next_action']}")
    print("canonical_contract_written=false")
    print("features_generated=false")
    print("model_training_invoked=false")
    print("diagnostics_invoked=false")
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

def run_ml_stock_alpha_news_risk_overlay_research(config):
    mode = _news_risk_output_mode(config)
    if mode != "json":
        print("\nSTOCK-ALPHA NEWS RISK OVERLAY RESEARCH")
        print("mode=research | historical_only=true | trading_impact=none | transformer_trained=false | paper_orders=false")
    try:
        result = write_stock_alpha_news_risk_overlay_research(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    inspection = inspect_stock_alpha_news_risk_overlay_results(config)
    print(format_news_risk_overlay_summary(inspection.summary, inspection.artifact_status, mode=mode))

def run_ml_stock_alpha_news_risk_overlay_inspect(config):
    mode = _news_risk_output_mode(config)
    if mode != "json":
        print("\nSTOCK-ALPHA NEWS RISK OVERLAY INSPECT")
        print("mode=read_only | trading_impact=none | loads_bars=false | trains_model=false | paper_orders=false")
    inspection = inspect_stock_alpha_news_risk_overlay_results(config)
    print(format_news_risk_overlay_summary(inspection.summary, inspection.artifact_status, mode=mode))

def run_ml_stock_alpha_news_risk_overlay_parallel_benchmark(config):
    print("\nSTOCK-ALPHA NEWS RISK OVERLAY PARALLEL BENCHMARK")
    print("mode=research | read_only=true | trading_impact=none | paper_orders=false | live_orders=false")
    try:
        result = write_stock_alpha_news_risk_overlay_parallel_benchmark(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    print(f"Output directory: {result.output_dir}")
    print(f"Parallel benchmark JSON: {result.report_json_path}")

def _news_risk_output_mode(config):
    mode = str(
        (config.get("ml", {}) or {}).get(
            "stock_alpha_news_risk_overlay_output_mode",
            "summary",
        )
    )
    return mode if mode in {"summary", "verbose", "json", "artifact-list"} else "summary"

def run_ml_stock_alpha_news_readiness_preflight(config):
    result = write_stock_alpha_news_readiness_preflight(config)
    print("\nSTOCK-ALPHA NEWS READINESS PREFLIGHT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_source_diagnostics(config):
    print("\nSTOCK-ALPHA NEWS SOURCE DIAGNOSTICS")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_source_diagnostics(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"next_action={payload['next_action']}")
    for issue in payload["blocking_issues"]: print(f"blocking_issue={issue}")
    print("features_generated=false")
    print("files_ingested=false")
    print("readiness_invoked=false")
    print("model_training_invoked=false")
    print("diagnostics_invoked=false")
    print("news_transformer_enabled=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_source_setup_check(config):
    print("\nSTOCK-ALPHA NEWS SOURCE SETUP CHECK")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    result = write_stock_alpha_news_source_setup_check(config)
    import json
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"next_action={payload['next_action']}")
    print(f"providers_enabled={','.join(payload['providers_enabled']) or 'none'}")
    print(f"providers_missing_key={','.join(payload['enabled_providers_missing_key']) or 'none'}")
    print("collection_invoked=false")
    print("raw_export_written=false")
    print("model_training_invoked=false")
    print("news_transformer_enabled=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_pipeline_preflight(config):
    result = write_stock_alpha_news_pipeline_preflight(config)
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print("\nSTOCK-ALPHA NEWS PIPELINE PREFLIGHT")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(
        "pipeline_safe_for_news_transformer_training="
        f"{str(payload.get('pipeline_safe_for_news_transformer_training', False)).lower()}"
    )
    print(f"stopped_stage={payload.get('stopped_stage') or 'none'}")
    for issue in payload.get("blocking_issues", []):
        print(f"blocking_issue={issue}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_news_pipeline_inspect(config):
    print("\nSTOCK-ALPHA NEWS PIPELINE INSPECTION")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    try:
        result = write_stock_alpha_news_pipeline_inspect(config)
    except ValueError as exc:
        print(f"blocking_issue={exc}")
        raise SystemExit(1) from None
    import json

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    print(f"next_action={payload['next_action']}")
    enabled = payload["config_summary"]["stock_alpha_news_enable_transformer"]
    print(f"stock_alpha_news_enable_transformer={str(enabled).lower()}")
    print("files_ingested=false")
    print("features_generated=false")
    print("model_training_invoked=false")
    print("diagnostics_invoked=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_dev_smoke(config):
    from core.research.ml.stock_level.stock_alpha_dev_smoke import (
        write_stock_alpha_dev_smoke,
    )

    result = write_stock_alpha_dev_smoke(config)
    print("\nSTOCK-ALPHA DEV SMOKE")
    print("mode=research | run_size=dev | production_validated=false")
    print(f"Report: {result}")

def run_ml_stock_alpha_parallelism_audit(config):
    from core.research.ml.stock_level.stock_alpha_parallelism_audit import (
        write_stock_alpha_parallelism_audit,
    )

    result = write_stock_alpha_parallelism_audit(config)
    print("\nSTOCK-ALPHA PARALLELISM AUDIT")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_alpha_run_status(config):
    from core.research.ml.stock_level.run_manifest.service import (
        write_stock_alpha_run_status,
    )

    result = write_stock_alpha_run_status(config)
    print("\nSTOCK-ALPHA RUN STATUS")
    print("mode=research | inspection_only=true | trading_impact=none | production_validated=false")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_level_feature_attribution(config):
    from core.research.ml.stock_level_feature_attribution import (
        write_stock_level_feature_attribution,
    )

    result = write_stock_level_feature_attribution(config)
    print("\nSTOCK-LEVEL FEATURE ATTRIBUTION")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print(f"Markdown: {result.markdown_path}")

def run_ml_stock_level_alpha_features(config):
    from core.research.ml.stock_level_alpha_features import (
        write_stock_level_alpha_features,
    )

    result = write_stock_level_alpha_features(config)
    print("\nSTOCK-LEVEL ALPHA FEATURES")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Enriched artifact: {result.enriched_csv_path}")
    print(f"Audit CSV: {result.audit_csv_path}")
    print(f"Audit JSON: {result.audit_json_path}")
    print(f"Audit Markdown: {result.audit_markdown_path}")

def run_ml_overnight_stock_alpha(config):
    from core.research.ml.stock_level.overnight_stock_alpha_runner import (
        write_overnight_stock_alpha_experiment,
    )

    result = write_overnight_stock_alpha_experiment(config)
    print("\nOVERNIGHT STOCK-ALPHA EXPERIMENT")
    print("mode=research | trading_impact=none | production_validated=false")
    print(f"Summary JSON: {result.json_path}")
    print(f"Summary Markdown: {result.markdown_path}")
