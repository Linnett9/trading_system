from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
PREDICTION_COLUMNS = (
    "predicted_probability",
    "predicted_forward_return_5d",
    "predicted_forward_return_10d",
    "predicted_future_volatility",
    "predicted_future_drawdown",
    "predicted_max_adverse_excursion",
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
BASELINE_PREDICTION_COLUMNS = (
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
ACTUAL_COLUMNS = (
    "actual_forward_return_5d",
    "actual_forward_return_10d",
    "actual_future_volatility",
    "actual_future_drawdown",
    "actual_max_adverse_excursion",
    "actual_market_residual_return_10d",
    "actual_vol_adjusted_forward_return_10d",
    "actual_drawdown_adjusted_forward_return_10d",
    "actual_rank_normalized_forward_return_10d",
    "actual_top_decile_label_10d",
)
TARGET_TYPES = {
    "actual_forward_return_10d": "raw",
    "actual_market_residual_return_10d": "residual",
    "actual_vol_adjusted_forward_return_10d": "volatility-adjusted",
    "actual_drawdown_adjusted_forward_return_10d": "drawdown-adjusted",
    "actual_rank_normalized_forward_return_10d": "rank-normalized",
    "actual_top_decile_label_10d": "classification",
}
CONTEXT_COLUMNS = (
    "breadth_above_sma_200",
    "spy_realized_volatility_21d",
    "spy_realized_volatility_63d",
    "spy_max_drawdown_63d",
    "spy_max_drawdown_126d",
)

@dataclass(frozen=True)
class StockLevelPredictionArtifactsPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
