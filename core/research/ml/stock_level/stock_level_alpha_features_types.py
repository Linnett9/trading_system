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
MARKET_CONTEXT_CONTRACT_VERSION = "stock_level_market_context_contract_v1"
BREADTH_CONTRACT_VERSION = "stock_level_breadth_contract_v1"
INDUSTRY_MAPPING_CONTRACT_VERSION = "stock_level_static_industry_mapping_contract_v1"
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
    "market_momentum_20d",
    "market_momentum_60d",
    "market_momentum_120d",
    "market_volatility_20d",
    "market_drawdown_60d",
    "market_distance_from_200d_average",
    "market_trend_state",
    "market_volatility_percentile",
    "breadth_positive_momentum_20d",
    "breadth_positive_momentum_60d",
    "breadth_above_long_term_trend",
    "breadth_cross_sectional_median_return",
    "breadth_return_dispersion",
    "breadth_advance_decline_ratio",
    "breadth_coverage",
    "relative_momentum_vs_industry",
    "industry_momentum_percentile",
)
ENRICHMENT_METADATA_COLUMNS = (
    "market_context_source_date",
    "market_context_availability_timestamp",
    "market_context_status",
    "market_context_contract_identity",
    "breadth_eligible_symbol_count",
    "breadth_observed_symbol_count",
    "breadth_contract_identity",
    "industry_id",
    "industry_mapping_identity",
    "industry_peer_count",
    "industry_mapping_available",
    "industry_relative_status",
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
    "market_momentum_20d": "SPY 20-observation trailing return using prices strictly before rebalance.",
    "market_momentum_60d": "SPY 60-observation trailing return using prices strictly before rebalance.",
    "market_momentum_120d": "SPY 120-observation trailing return using prices strictly before rebalance.",
    "market_volatility_20d": "SPY annualized 20-observation realized volatility using prices strictly before rebalance.",
    "market_drawdown_60d": "SPY max drawdown over the prior 60 observations.",
    "market_distance_from_200d_average": "SPY close divided by prior 200-observation average, minus one.",
    "market_trend_state": "1 when SPY is above its prior 200-observation average and 60d momentum is positive; otherwise 0.",
    "market_volatility_percentile": "SPY current 20d volatility percentile versus prior observations.",
    "breadth_positive_momentum_20d": "Decision-date fraction of eligible rows with positive 20d momentum.",
    "breadth_positive_momentum_60d": "Decision-date fraction of eligible rows with positive 60d momentum.",
    "breadth_above_long_term_trend": "Decision-date fraction of observed symbols above their prior 200-observation average.",
    "breadth_cross_sectional_median_return": "Decision-date median 20d momentum across observed symbols.",
    "breadth_return_dispersion": "Decision-date population standard deviation of 20d momentum across observed symbols.",
    "breadth_advance_decline_ratio": "Decision-date positive-to-negative 20d momentum ratio across observed symbols.",
    "breadth_coverage": "Observed symbol fraction for breadth calculations on the decision date.",
    "relative_momentum_vs_industry": "Stock 120d momentum minus its industry cross-sectional mean.",
    "industry_momentum_percentile": "Within-industry percentile of 120d momentum on each rebalance date.",
}
@dataclass(frozen=True)
class StockLevelAlphaFeaturePaths:
    enriched_parquet_path: Path
    audit_csv_path: Path
    audit_json_path: Path
    audit_markdown_path: Path
    enriched_sample_csv_path: Path | None = None

    @property
    def enriched_csv_path(self) -> Path:
        return self.enriched_parquet_path
