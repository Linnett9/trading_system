import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trading system command line interface"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "backtest",
            "optimize",
            "walk-forward",
            "compare-strategies",
            "data-audit",
            "dataset-audit",
            "relative-strength",
            "dual-momentum",
            "dual-momentum-walk-forward",
            "dual-momentum-risk-regimes",
            "dual-momentum-diagnosis",
            "ml-dual-momentum-stock-score-comparison",
            "paper-trade",
            "paper-fill",
            "paper-status",
            "paper-report",
            "paper-trading",
            "paper-dry-run",
            "paper-trial",
            "paper-weekly-summary",
            "paper-promotion-checklist",
            "paper-run",
            "paper-repair",
            "paper-reset",
            "multi-strategy",
            "multi-strategy-walk-forward",
            "ml-research",
            "ml-research-batch",
            "ml-online-intraday-benchmark",
            "ml-model-contract-audit",
            "ml-run-inventory",
            "ml-clean-incomplete-runs",
            "ml-validate-artifacts",
            "ml-smoke-test",
            "ml-data-inventory",
            "ml-historical-bar-backfill-probe",
            "ml-historical-bar-backfill-collect",
            "ml-historical-bar-backfill-benchmark",
            "ml-historical-bar-feed-overlap",
            "ml-build-universes",
            "ml-expanded-rebalance-dataset",
            "ml-meta-ensemble",
            "ml-return-mechanics-audit",
            "ml-benchmark-return-audit",
            "ml-refresh-adjusted-prices",
            "ml-stock-level-alpha-benchmark",
            "ml-stock-selector-bounded",
            "ml-stock-selector-final-fit",
            "ml-selector-exposure-comparison",
            "ml-selector-portfolio-promotion",
            "ml-selector-target-tournament",
            "ml-selector-cost-aware-policy-evaluation",
            "ml-selector-confidence-ensemble",
            "ml-selector-feature-ablation",
            "ml-selector-universe-integrity-audit",
            "ml-stock-fundamentals-preflight",
            "ml-stock-fundamentals-collect",
            "ml-stock-fundamentals-normalize",
            "ml-stock-fundamentals-audit",
            "ml-stock-fundamentals-snapshots",
            "ml-stock-fundamentals-enrich",
            "ml-stock-fundamentals-pipeline",
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
            "import-stooq-bulk",
            "import-market-parquet",
            "champion-robustness",
        ],
        default="backtest",
    )

    parser.add_argument(
        "--details",
        action="store_true",
        help="Show fold-level walk-forward details.",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Set CLI logging verbosity.",
    )
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="Print all strategy comparison rows instead of top-N.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the reduced research config for quick iteration.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override backtest symbols for this run.",
    )
    parser.add_argument(
        "--auto-discover-replay-symbols",
        action="store_true",
        help="For adjusted price refresh, include symbols used by replay artifacts.",
    )
    parser.add_argument(
        "--top",
        type=int,
        choices=[500, 1000],
        help="Import the top N raw Stooq symbols by available row count.",
    )
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="Import every raw Stooq symbol after asset/min-row filters.",
    )
    parser.add_argument(
        "--asset-class",
        choices=["stocks", "etfs", "all"],
        default="all",
        help="Filter raw Stooq imports by asset class.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        help="Filter raw Stooq imports to symbols with at least this many rows.",
    )
    parser.add_argument(
        "--exclude-warrants-units-rights",
        action="store_true",
        help="Exclude likely warrants, units, and rights from raw Stooq imports.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=["1Day", "daily", "5m", "1h", "hourly"],
        help="Override timeframes for local market Parquet import.",
    )
    parser.add_argument(
        "--years",
        type=int,
        help="Override backtest history length for this run.",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--feature-workers", type=int, default=None)
    parser.add_argument("--model-threads", type=int, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--sklearn-n-jobs", type=int, default=None)
    parser.add_argument("--oos-start-date", default=None)
    parser.add_argument("--oos-end-date", default=None)
    parser.add_argument("--max-oos-dates", type=int, default=None)
    parser.add_argument("--model-allowlist", nargs="+", default=None)
    parser.add_argument("--baseline-allowlist", nargs="+", default=None)
    parser.add_argument("--bounded-output-root", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-overwrite-incomplete-dates", action="store_true")
    parser.add_argument("--rf-estimators", type=int, default=None)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=None)
    parser.add_argument("--gb-estimators", type=int, default=None)
    parser.add_argument("--gb-max-depth", type=int, default=None)
    parser.add_argument("--gb-learning-rate", type=float, default=None)
    parser.add_argument("--training-row-cap", type=int, default=None)
    parser.add_argument("--news-risk-parallel", action="store_true")
    parser.add_argument("--news-risk-workers", type=int, default=None)
    parser.add_argument(
        "--news-risk-parallel-backend",
        choices=["thread", "process"],
        default=None,
    )
    parser.add_argument("--news-risk-parallel-min-items", type=int, default=None)
    parser.add_argument("--news-risk-parallel-chunk-size", type=int, default=None)
    parser.add_argument("--news-risk-parallel-benchmark-symbol-limit", type=int, default=None)
    parser.add_argument("--news-risk-output-dir", default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--artifact-list", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Only compare the named strategies for this run.",
    )
    parser.add_argument(
        "--grid-values",
        type=int,
        help="Limit each grid parameter to the first N configured values.",
    )
    parser.add_argument(
        "--universe",
        choices=["default", "etf", "stocks", "all", "stooq_test", "stooq_30"],
        default="default",
        help="Use a configured research universe preset.",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run the selected research mode's experiment grid.",
    )
    parser.add_argument(
        "--selector-mode",
        choices=["research", "paper", "production"],
        help=(
            "Override the dual-momentum walk-forward selector mode. "
            "Defaults to the configured mode."
        ),
    )
    parser.add_argument(
        "--decision-file",
        help="Paper decision JSON to fill. Defaults to the latest decision.",
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Confirm destructive paper state reset.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Config file to load. Can point at a frozen paper config.",
    )
    parser.add_argument(
        "--profile",
        choices=["development", "benchmark"],
        help="Apply a research profile for isolated ML cache/report outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview paper-trading orders without filling/submitting.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit/fill the approved paper-trading decision.",
    )
    parser.add_argument(
        "--confirm-fill",
        action="store_true",
        help="Confirm direct paper-fill command. Prefer paper-trading --submit.",
    )

    return parser.parse_args()
