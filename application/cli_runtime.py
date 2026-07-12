import logging

from application.cli_dispatch import dispatch
from application.cli_feed import build_feed
from application.cli_parser import parse_args
from application.services.research_profiles import apply_research_profile
from application.services.runtime_overrides import apply_runtime_overrides
from config.config_loader import load_config


FEEDLESS_MODES = {
        "dataset-audit",
        "import-stooq-bulk",
        "import-market-parquet",
        "ml-data-inventory",
        "ml-historical-bar-backfill-probe",
        "ml-historical-bar-backfill-collect",
        "ml-historical-bar-backfill-benchmark",
        "ml-historical-bar-feed-overlap",
        "ml-build-universes",
        "ml-model-contract-audit",
        "ml-run-inventory",
        "ml-clean-incomplete-runs",
        "ml-validate-artifacts",
        "ml-meta-ensemble",
        "ml-return-mechanics-audit",
        "ml-benchmark-return-audit",
        "ml-refresh-adjusted-prices",
        "ml-research-batch",
        "ml-online-intraday-benchmark",
        "ml-dual-momentum-stock-score-comparison",
        "ml-stock-level-alpha-benchmark",
        "ml-selector-portfolio-promotion",
        "ml-selector-target-tournament",
        "ml-selector-cost-aware-policy-evaluation",
        "ml-selector-confidence-ensemble",
        "ml-selector-feature-ablation",
        "ml-selector-universe-integrity-audit",
        "ml-stock-level-target-comparison",
        "ml-stock-level-portfolio-replay",
        "ml-stock-selector-rebalance-dataset",
        "ml-stock-level-portfolio-policy-sweep",
        "ml-stock-alpha-experiment-report",
        "ml-stock-alpha-candidate-report",
        "ml-stock-alpha-deep-diagnostics",
        "ml-stock-alpha-ensemble",
        "ml-stock-alpha-ensemble-portfolio-sweep",
        "ml-stock-alpha-experiment-preflight",
        "ml-stock-alpha-news-features",
        "ml-stock-alpha-news-feature-diagnostics",
        "ml-stock-alpha-news-contract-ingest",
        "ml-stock-alpha-news-collect-free-sources",
        "ml-stock-alpha-news-collection-plan",
        "ml-stock-alpha-news-historical-backfill",
        "ml-stock-alpha-news-canonical-corpus",
        "ml-stock-alpha-news-daily-confirmation",
        "ml-stock-alpha-news-coverage-audit",
        "ml-stock-alpha-news-risk-overlay-research",
        "ml-stock-alpha-news-risk-overlay-inspect",
        "ml-stock-alpha-news-risk-overlay-parallel-benchmark",
        "ml-stock-alpha-news-provider-audit",
        "ml-stock-alpha-news-provider-sample-check",
        "ml-stock-alpha-news-pipeline-preflight",
        "ml-stock-alpha-news-pipeline-inspect",
        "ml-stock-alpha-news-readiness-preflight",
        "ml-stock-alpha-news-source-diagnostics",
        "ml-stock-alpha-news-source-setup-check",
        "ml-stock-alpha-finbert-news-probe",
        "ml-stock-alpha-dev-smoke",
        "ml-stock-alpha-parallelism-audit",
        "ml-stock-alpha-run-status",
        "ml-stock-level-feature-attribution",
        "ml-stock-level-alpha-features",
        "ml-overnight-stock-alpha",
    }



def run_cli():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    loaded_config = apply_research_profile(
        load_config(args.config, overlay_project_config=True),
        getattr(args, "profile", None),
    )
    config = apply_runtime_overrides(
        loaded_config,
        args,
    )
    config["config_path"] = args.config
    feedless_modes = FEEDLESS_MODES
    feed = None if args.mode in feedless_modes else build_feed(config)

    dispatch(args, config, feed)
