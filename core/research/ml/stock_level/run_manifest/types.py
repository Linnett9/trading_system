from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
STAGE_ORDER = (
    "stock_artifact",
    "alpha_features",
    "baseline_benchmark",
    "enriched_benchmark",
    "target_comparison",
    "portfolio_replay",
    "portfolio_policy_sweep",
    "experiment_report",
    "optional_attribution",
    "overnight_summary",
)
STAGE_LABELS = {
    "stock_artifact": "stock artifact",
    "alpha_features": "alpha features",
    "baseline_benchmark": "baseline benchmark",
    "enriched_benchmark": "enriched benchmark",
    "target_comparison": "target comparison",
    "portfolio_replay": "portfolio replay",
    "portfolio_policy_sweep": "portfolio policy sweep",
    "experiment_report": "experiment report",
    "optional_attribution": "optional attribution",
    "overnight_summary": "overnight summary",
}
FINAL_STATUSES = {"completed", "skipped"}
LEGACY_OUTPUT_ROOT = Path("reports/ml/benchmark/ml")
DEFAULT_PYTHON = "/Users/brandonlinnett/.pyenv/versions/3.11.6/bin/python"

@dataclass(frozen=True)
class StockAlphaRunManifestPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class StockAlphaInterruptedSummaryPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class StockAlphaRunStatusPaths:
    json_path: Path
    markdown_path: Path
