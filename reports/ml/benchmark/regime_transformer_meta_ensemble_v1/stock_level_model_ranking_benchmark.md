# Stock-Level Alpha Benchmark Suite

Research only. Trading impact: none. Production validated: false.

- Target: `actual_forward_return_10d`
- Eligible input rows: 81485
- OOS rows: 61019
- OOS dates: 161
- Completed models: 4
- Unavailable models: 0
- Split: chronological expanding window with 2 embargoed rebalance dates
- Promotion thresholds changed: false

## OOS Leaderboard

| Rank | Model / baseline | Kind | Dates | Pearson IC | Spearman IC | Top decile | Bottom decile | Spread | Sharpe | Top hit rate | Risk-adjusted spread |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | random_forest | ml_model | 161 | 0.037404 | 0.037697 | 0.016242 | 0.005740 | 0.010502 | 1.927162 | 0.553776 | 0.134566 |
| 2 | elastic_net | ml_model | 161 | 0.040636 | 0.031438 | 0.015367 | 0.004426 | 0.010941 | 1.841871 | 0.540373 | 0.054009 |
| 3 | ridge | ml_model | 161 | 0.040063 | 0.031274 | 0.014958 | 0.004729 | 0.010229 | 1.732194 | 0.538902 | 0.042575 |
| 4 | gradient_boosting | ml_model | 161 | 0.027063 | 0.028304 | 0.011666 | 0.005734 | 0.005932 | 1.067591 | 0.539065 | -0.004147 |
| 5 | momentum_120d | baseline | 161 | 0.039276 | 0.010287 | 0.014755 | 0.006393 | 0.008362 | 1.317096 | 0.558352 | 0.232537 |
| 6 | risk_adjusted_momentum | baseline | 161 | -0.006550 | -0.009412 | 0.005729 | 0.005612 | 0.000117 | 0.021042 | 0.532527 | 0.040560 |

## Best ML vs Momentum 120d

- Best ML model: random_forest
- Beats OOS-aligned momentum_120d: True
- Decision rule: higher mean Spearman IC and higher top-minus-bottom spread.
- Spearman IC delta: 0.027410
- Spread delta: 0.002140

## Full-Period Baseline Reference

| Baseline | Dates | Spearman IC | Spread | Top hit rate | Risk-adjusted spread |
|---|---:|---:|---:|---:|---:|
| momentum_120d | 215 | 0.001395 | 0.005119 | 0.550551 | 0.152994 |
| risk_adjusted_momentum | 215 | -0.019961 | -0.002607 | 0.523256 | -0.047286 |
