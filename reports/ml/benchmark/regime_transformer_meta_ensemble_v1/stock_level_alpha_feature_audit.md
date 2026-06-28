# Stock-Level Alpha Feature Audit

Research only. Trading impact: none. Production validated: false.

- Rows: 81485
- Engineered features: 18
- Source columns preserved: True
- Unique symbol/date rows: True
- Industry metadata available: False
- Promotion thresholds changed: false

| Feature | Populated | Missing | Availability | Definition |
|---|---:|---:|---:|---|
| momentum_250d | 81485 | 0 | 1.0000 | Trailing 250-observation return using prices strictly before rebalance. |
| momentum_acceleration | 81485 | 0 | 1.0000 | OLS slope of 20d, 60d, and 120d momentum versus horizon. |
| momentum_persistence | 81485 | 0 | 1.0000 | Fraction of the latest 120 trailing 20d return windows that are positive. |
| momentum_consistency | 81485 | 0 | 1.0000 | R-squared of a linear trend fitted to 120 log closing prices. |
| relative_momentum_vs_spy | 81485 | 0 | 1.0000 | Stock 120d momentum minus SPY 120d momentum on the same date. |
| relative_momentum_vs_sector | 6880 | 74605 | 0.0844 | Stock 120d momentum minus its sector cross-sectional mean. |
| momentum_percentile | 81485 | 0 | 1.0000 | Cross-sectional percentile of 120d momentum on each rebalance date. |
| distance_from_52_week_high | 81485 | 0 | 1.0000 | Latest close divided by the prior 252-observation high, minus one. |
| drawdown_recovery_days | 81485 | 0 | 1.0000 | Trading observations since the latest prior 252-observation high; zero at a high. |
| rolling_max_drawdown_120d | 81485 | 0 | 1.0000 | Worst peak-to-trough drawdown inside the prior 120 observations. |
| ulcer_index | 81485 | 0 | 1.0000 | Root mean square percentage drawdown over the prior 120 observations. |
| downside_deviation | 81485 | 0 | 1.0000 | Root mean square of negative daily returns over the prior 60 observations. |
| volatility_percentile | 81485 | 0 | 1.0000 | Percentile of current 20d volatility versus its prior 252 observations. |
| volatility_trend | 81485 | 0 | 1.0000 | Current 20d volatility divided by 60d volatility, minus one. |
| volatility_regime | 81485 | 0 | 1.0000 | Numeric volatility bucket: 0 low, 1 normal, 2 high. |
| ATR_percentile | 81485 | 0 | 1.0000 | Percentile of normalized ATR(14) versus its prior 252 observations. |
| sector_relative_strength | 6880 | 74605 | 0.0844 | Within-sector percentile of 120d momentum on each rebalance date. |
| industry_relative_strength | 0 | 81485 | 0.0000 | Within-industry percentile of 120d momentum when industry metadata exists. |
