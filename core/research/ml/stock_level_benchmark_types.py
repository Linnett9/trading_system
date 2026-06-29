from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.research.ml.stock_level.stock_level_alpha_features import (
    ENGINEERED_FEATURE_COLUMNS,
)


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
TARGET_COLUMN = "actual_forward_return_10d"
FEATURE_COLUMNS = (
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
CONTEXT_COLUMNS = (
    "breadth_above_sma_200",
    "spy_realized_volatility_21d",
    "spy_realized_volatility_63d",
    "spy_max_drawdown_63d",
    "spy_max_drawdown_126d",
)
AUXILIARY_TARGET_COLUMNS = (
    "actual_forward_return_5d",
    "actual_future_volatility",
    "actual_future_drawdown",
)
TARGET_OUTPUT_COLUMNS = (
    "actual_market_residual_return_10d",
    "actual_vol_adjusted_forward_return_10d",
    "actual_drawdown_adjusted_forward_return_10d",
    "actual_rank_normalized_forward_return_10d",
    "actual_top_decile_label_10d",
)
BASELINE_COLUMNS = {
    "momentum_120d": "predicted_momentum_120d",
    "risk_adjusted_momentum": "predicted_risk_adjusted_momentum",
}
TABULAR_MODEL_NAMES = (
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)
SEQUENCE_MODEL_NAMES = (
    "dlinear",
    "patchtst",
    "transformer",
    "itransformer",
    "momentum_transformer",
    "multitask_transformer",
    "market_context_encoder",
    "news_analysis_transformer",
    "temporal_fusion_transformer",
)
MODEL_NAMES = (*TABULAR_MODEL_NAMES, *SEQUENCE_MODEL_NAMES)
PREDICTION_PREFIX = "stock_level_predicted_forward_return_10d_"
ALL_FEATURE_COLUMNS = (*FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)


@dataclass(frozen=True)
class StockLevelModelRankingBenchmarkPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class ModelRunSpec:
    name: str
    kind: str
    factory: Callable[[], Any]
    feature_columns: tuple[str, ...]
