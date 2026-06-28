# Stock-Level Prediction Artifacts

Research only. Trading impact: none. Production validated: false.

- Rows: 81485
- Symbols: 379
- Rebalance dates: 215
- Date range: ['2022-08-29', '2026-04-20']
- Average symbols per date: 379.00
- True stock-level rows: True
- Usable for stock-level ranking: True
- Suitable for true stock-level ranking diagnostics: True
- Suitability reason: stock-level rows include point-in-time baseline forecast signals

## Populated Predictions

- predicted_probability: 0
- predicted_forward_return_5d: 0
- predicted_forward_return_10d: 0
- predicted_future_volatility: 0
- predicted_future_drawdown: 0
- predicted_max_adverse_excursion: 0
- predicted_momentum_20d: 81485
- predicted_momentum_60d: 81485
- predicted_momentum_120d: 81485
- predicted_volatility_20d: 81485
- predicted_drawdown_60d: 81485
- predicted_liquidity_score: 81485
- predicted_risk_adjusted_momentum: 81485

## Missing Predictions

- predicted_probability: 81485
- predicted_forward_return_5d: 81485
- predicted_forward_return_10d: 81485
- predicted_future_volatility: 81485
- predicted_future_drawdown: 81485
- predicted_max_adverse_excursion: 81485
- predicted_momentum_20d: 0
- predicted_momentum_60d: 0
- predicted_momentum_120d: 0
- predicted_volatility_20d: 0
- predicted_drawdown_60d: 0
- predicted_liquidity_score: 0
- predicted_risk_adjusted_momentum: 0

## Missing Actual Targets

- actual_forward_return_5d: 0
- actual_forward_return_10d: 0
- actual_future_volatility: 0
- actual_future_drawdown: 0
- actual_max_adverse_excursion: 0

## Root Cause

Existing prediction_artifacts.csv rows are keyed by feature_id/variant_id and have blank symbol values; they predict strategy/variant outcomes rather than individual security outcomes.
