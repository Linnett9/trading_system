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
ENGINEERED_FEATURE_COLUMNS = (
    "momentum_250d",
    "momentum_acceleration",
    "momentum_persistence",
    "momentum_consistency",
    "relative_momentum_vs_spy",
    "relative_momentum_vs_sector",
    "momentum_percentile",
    "distance_from_52_week_high",
    "drawdown_recovery_days",
    "rolling_max_drawdown_120d",
    "ulcer_index",
    "downside_deviation",
    "volatility_percentile",
    "volatility_trend",
    "volatility_regime",
    "ATR_percentile",
    "sector_relative_strength",
    "industry_relative_strength",
)
FEATURE_DEFINITIONS = {
    "momentum_250d": "Trailing 250-observation return using prices strictly before rebalance.",
    "momentum_acceleration": "OLS slope of 20d, 60d, and 120d momentum versus horizon.",
    "momentum_persistence": "Fraction of the latest 120 trailing 20d return windows that are positive.",
    "momentum_consistency": "R-squared of a linear trend fitted to 120 log closing prices.",
    "relative_momentum_vs_spy": "Stock 120d momentum minus SPY 120d momentum on the same date.",
    "relative_momentum_vs_sector": "Stock 120d momentum minus its sector cross-sectional mean.",
    "momentum_percentile": "Cross-sectional percentile of 120d momentum on each rebalance date.",
    "distance_from_52_week_high": "Latest close divided by the prior 252-observation high, minus one.",
    "drawdown_recovery_days": "Trading observations since the latest prior 252-observation high; zero at a high.",
    "rolling_max_drawdown_120d": "Worst peak-to-trough drawdown inside the prior 120 observations.",
    "ulcer_index": "Root mean square percentage drawdown over the prior 120 observations.",
    "downside_deviation": "Root mean square of negative daily returns over the prior 60 observations.",
    "volatility_percentile": "Percentile of current 20d volatility versus its prior 252 observations.",
    "volatility_trend": "Current 20d volatility divided by 60d volatility, minus one.",
    "volatility_regime": "Numeric volatility bucket: 0 low, 1 normal, 2 high.",
    "ATR_percentile": "Percentile of normalized ATR(14) versus its prior 252 observations.",
    "sector_relative_strength": "Within-sector percentile of 120d momentum on each rebalance date.",
    "industry_relative_strength": "Within-industry percentile of 120d momentum when industry metadata exists.",
}
@dataclass(frozen=True)
class StockLevelAlphaFeaturePaths:
    enriched_csv_path: Path
    audit_csv_path: Path
    audit_json_path: Path
    audit_markdown_path: Path
