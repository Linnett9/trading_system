from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.research.ml.stock_level.stock_level_alpha_features import write_stock_level_alpha_features
from core.research.ml.stock_level.feature_attribution.service import write_stock_level_feature_attribution
from core.research.ml.stock_level.stock_level_model_ranking_benchmark import write_stock_level_model_ranking_benchmark
from core.research.ml.stock_level.prediction_artifacts.service import write_stock_level_prediction_artifacts


RESEARCH_GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
SUMMARY_MODELS = (
    "original_baseline_artifact_benchmark",
    "enriched_feature_benchmark",
    "momentum_120d",
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)
METRIC_ALIASES = {
    "spearman_ic": "mean_spearman_ic",
    "top_minus_bottom_spread": "top_minus_bottom_spread",
    "spread_sharpe": "spread_sharpe",
    "risk_adjusted_spread": "risk_adjusted_spread",
    "top_decile_hit_rate": "top_decile_hit_rate",
}


@dataclass(frozen=True)
class OvernightStockAlphaPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class OvernightStockAlphaStages:
    stock_artifact: Callable[[dict[str, Any]], Any] = write_stock_level_prediction_artifacts
    alpha_features: Callable[[dict[str, Any]], Any] = write_stock_level_alpha_features
    benchmark: Callable[[dict[str, Any]], Any] = write_stock_level_model_ranking_benchmark
    attribution: Callable[[dict[str, Any]], Any] = write_stock_level_feature_attribution
    target_comparison: Callable[[dict[str, Any]], Any] | None = None
    portfolio_replay: Callable[[dict[str, Any]], Any] | None = None
    portfolio_policy_sweep: Callable[[dict[str, Any]], Any] | None = None
