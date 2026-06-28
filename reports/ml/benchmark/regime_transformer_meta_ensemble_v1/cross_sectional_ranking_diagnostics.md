# Cross-Sectional Ranking Diagnostics

Research only. Trading impact: none. Production validated: false.

- Grouping: rebalance_date
- Target: actual_forward_return_10d
- Row entity type: symbol
- Stock-level available: True
- Any signal ranks future returns: True
- Future ranking model justified: True

## Signals

| Signal | Dates | Spearman | Top decile | Bottom decile | Spread | Top hit rate | Risk-adjusted spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| momentum_20d | 215 | -0.004450 | 0.011667 | 0.008956 | 0.002711 | 0.540514 | 0.070370 |
| momentum_60d | 215 | -0.019534 | 0.008527 | 0.009230 | -0.000703 | 0.530110 | -0.004959 |
| momentum_120d | 215 | 0.001395 | 0.012646 | 0.007527 | 0.005119 | 0.550551 | 0.152994 |
| risk_adjusted_momentum | 215 | -0.019961 | 0.004095 | 0.006702 | -0.002607 | 0.523256 | -0.047286 |
| liquidity_score | 215 | 0.001819 | 0.010756 | 0.013697 | -0.002941 | 0.570135 | 0.162737 |

## Verdict

- Best signal: momentum_120d
- Best top-minus-bottom spread: 0.005119
- Note: Diagnostics rank symbols within each rebalance date.
