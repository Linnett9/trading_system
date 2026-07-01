from application.cli_dispatch import dispatch
from application.cli_feed import build_feed
from application.cli_parser import parse_args
from application.services.research_profiles import apply_research_profile
from application.services.runtime_overrides import apply_runtime_overrides
from config.config_loader import load_config


FEEDLESS_MODES = {
        "import-stooq-bulk",
        "ml-data-inventory",
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
        "ml-stock-level-alpha-benchmark",
        "ml-stock-level-target-comparison",
        "ml-stock-level-portfolio-replay",
        "ml-stock-level-portfolio-policy-sweep",
        "ml-stock-alpha-experiment-report",
        "ml-stock-alpha-candidate-report",
        "ml-stock-alpha-deep-diagnostics",
        "ml-stock-alpha-ensemble",
        "ml-stock-alpha-ensemble-portfolio-sweep",
        "ml-stock-alpha-experiment-preflight",
        "ml-stock-alpha-news-features",
        "ml-stock-alpha-news-contract-ingest",
        "ml-stock-alpha-news-coverage-audit",
        "ml-stock-alpha-news-provider-audit",
        "ml-stock-alpha-news-pipeline-preflight",
        "ml-stock-alpha-news-readiness-preflight",
        "ml-stock-alpha-dev-smoke",
        "ml-stock-alpha-parallelism-audit",
        "ml-stock-alpha-run-status",
        "ml-stock-level-feature-attribution",
        "ml-stock-level-alpha-features",
        "ml-overnight-stock-alpha",
    }



def run_cli():
    args = parse_args()
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
