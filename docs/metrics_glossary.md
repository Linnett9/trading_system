# Metrics Glossary

This glossary is based on current research modules, especially
`core/research/framework/ranking.py`,
`core/research/ml/stock_level/stock_level_portfolio_replay.py`,
`core/research/ml/stock_level/stock_level_portfolio_policy_sweep.py`, and
stock-alpha report writers.

## Stock-Alpha Ranking Metrics

| Metric | Definition | Better | Why it matters | Caveats | Where to find it |
|---|---|---|---|---|---|
| Pearson IC | Per-date Pearson correlation between signal and target, averaged as `mean_pearson_ic`. | Higher | Measures linear association between scores and realized target. | Sensitive to outliers and scale. | Benchmark leaderboard. |
| Spearman IC | Per-date Pearson correlation of signal ranks and target ranks. | Higher | Measures whether the model orders stocks correctly. | Ignores magnitude; ties can reduce information. | Benchmark leaderboard. |
| Mean Spearman IC | Average Spearman IC across eligible OOS dates. | Higher | Primary ranking quality metric in leaderboard sorting. | Can hide unstable date-by-date behaviour. | `mean_spearman_ic`. |
| Top decile return | Average target return of highest-scored 10% bucket per date. | Higher | Shows expected return of selected longs. | Bucket is at least one stock, so small universes are noisy. | `top_decile_return`. |
| Bottom decile return | Average target return of lowest-scored 10% bucket per date. | Lower for long-short spread | Helps evaluate separation between best and worst scores. | Less relevant for long-only if shorts are disabled. | `bottom_decile_return`. |
| Top-minus-bottom spread | Top decile return minus bottom decile return. | Higher | Captures cross-sectional separation. | Can look good while top decile is still negative. | `top_minus_bottom_spread`. |
| Spread Sharpe | Annualized Sharpe of per-date top-minus-bottom spreads. | Higher | Risk-adjusts spread consistency. | Annualization assumes repeated comparable periods. | `spread_sharpe`. |
| Top decile hit rate | Fraction of top-decile targets above zero. | Higher | Measures how often selected top bucket is positive. | Does not measure magnitude. | `top_decile_hit_rate`. |
| Risk-adjusted spread | Difference between top and bottom target/risk values, where risk uses future drawdown/volatility columns in evaluation. | Higher | Checks whether ranking survives risk adjustment. | Uses realized evaluation risk, not a tradable input. | `risk_adjusted_spread`. |
| OOS date count | Number of rebalance dates with out-of-sample predictions/evaluations. | Higher, all else equal | More dates make evidence less fragile. | More data can still be regime-specific. | `oos_date_count`. |
| Symbol count | Count of unique symbols in a run/report. | Context-dependent | Shows cross-sectional breadth. | Larger universes can include lower-quality data. | `symbol_count`, `input_symbol_count`, `oos_symbol_count`. |
| Eligible row/date/symbol counts | Rows/dates/symbols surviving required field filters. | Higher, if data quality holds | Shows how much data the model actually used. | High input count does not help if eligible count is low. | Target comparison and benchmark JSON. |

## Portfolio Metrics

| Metric | Definition | Better | Why it matters | Caveats | Where to find it |
|---|---|---|---|---|---|
| Gross return | Sum of period returns before costs/slippage/borrow drag. | Higher | Shows raw signal/policy return. | Not realistic by itself. | Replay/sweep summary. |
| Net return | Sum of period returns after configured drag. | Higher | Primary cost-aware policy metric. | Sum of period returns differs from compounded total return. | `net_return`. |
| Total return | Final equity minus one. | Higher | Compounded total portfolio outcome. | Sensitive to path and date range. | `total_return`. |
| Annualized return | Total return annualized using observed rebalance gaps. | Higher | Normalizes across different run lengths. | Assumes future cadence resembles test period. | `annualized_return`. |
| Mean period return | Average period net return. | Higher | Shows typical rebalance-period result. | Ignores compounding and volatility. | `mean_period_return`. |
| Volatility | Standard deviation of period net returns. | Lower for same return | Measures variability. | Low volatility with low return is not enough. | `volatility`. |
| Sharpe | Mean period return divided by volatility, annualized. | Higher | Risk-adjusted return metric. | Unstable with few periods or tiny volatility. | `sharpe`. |
| Max drawdown | Worst peak-to-trough equity loss, usually negative in these reports. | Less negative | Measures downside path risk. | Short histories understate rare drawdowns. | `max_drawdown`. |
| Calmar ratio | Annualized return divided by absolute max drawdown. | Higher | Return per drawdown. | Undefined when drawdown is zero or annualized return missing. | `calmar_ratio`. |
| Hit rate | Fraction of positive period returns. | Higher | Frequency of profitable periods. | Does not measure size of wins/losses. | `hit_rate`. |
| Average turnover | Average absolute weight change per period. | Lower for similar return | Proxy for trading intensity/cost sensitivity. | Needs execution assumptions to be realistic. | `average_turnover`. |
| Transaction cost drag | Return drag from turnover and configured costs; policy sweep combines cost, slippage, and borrow in this field. | Lower | Shows how much return is lost to frictions. | Uses simplified assumptions. | `transaction_cost_drag`. |
| Slippage drag | Drag from configured slippage in policy sweep. | Lower | Separates slippage from explicit cost where available. | Not a broker fill simulation. | `slippage_drag`. |
| Borrow cost | Cost applied to short exposure when configured in policy sweep. | Lower | Matters for long/short policies. | Only present/configured when short policies are allowed. | policy sweep `borrow_cost_bps` effects via drag. |
| Average number of positions | Mean holdings count per rebalance date. | Context-dependent | Shows diversification. | More positions can dilute signal. | `average_number_of_positions`. |
| Average/max position weight | Average and max absolute holding weights. | Lower concentration for same return | Shows concentration and cap usage. | Cap compliance does not guarantee liquidity. | `average_position_weight`, `max_position_weight`. |
| Best/worst period return | Maximum/minimum period net return. | Higher best, less negative worst | Highlights tails. | Single-period extremes can dominate perception. | `best_period_return`, `worst_period_return`. |
| Performance by year | Compounded net returns grouped by year. | Robust across years | Checks year dependence. | Calendar-year grouping can hide regimes. | `performance_by_year`. |
| Concentration metrics | Measures such as 95th percentile max position concentration. | Lower for same return | Detects dependence on a few names. | Depends on universe and policy constraints. | `position_concentration_percentile_95`. |
| Percentage of periods in cash | Fraction of periods where absolute weights are below full exposure. | Context-dependent | Shows how often constraints leave cash. | Cash can be defensive or a sign constraints bind too hard. | `percentage_of_periods_in_cash`. |

## Validation and Report Metrics

| Metric | Definition | Better | Why it matters | Where to find it |
|---|---|---|---|---|
| `skipped_target_count` | Number of target columns not completed. | Lower | Skipped targets weaken robustness. | Target comparison JSON. |
| `skip_reason_code` | Machine-readable target skip reason. | None | Explains missing target result. | Target comparison rows. |
| `target_non_null_count` | Number of rows with numeric target values. | Higher | Confirms target population. | Target comparison. |
| `validation_passed` | Experiment report has no validation errors. | True | Basic output correctness gate. | Experiment report. |
| `warning_count` | Count of validation warnings. | Lower | Warnings may indicate count/freshness concerns. | `len(validation.warnings)`. |
| `error_count` | Count of validation errors. | Zero | Errors should block trust in output bundle. | `len(validation.errors)`. |
| `promotion_thresholds_changed` | Whether promotion gates were changed. | False | Research outputs must not silently loosen gates. | Most report JSONs. |
| `production_validated` | Whether output is approved for production. | False for current research reports | Prevents accidental trading interpretation. | Most report JSONs. |
| `research_only` | Indicates artifact is research-only. | True | Confirms no trading impact. | Most report JSONs. |
| `trading_impact` | Declares execution impact. | `none` for research | Should remain no-impact for stock-alpha research. | Most report JSONs. |
| Requested workers | Configured worker count. | Matches intended run profile | Confirms config propagation. | Parallelism audit/stage reports. |
| Effective workers | Actual bounded worker count. | Expected value for run size | Confirms parallelism is active and capped. | Parallelism audit/stage reports. |
| Oversubscription warnings | Warnings about nested/concurrent workers. | None | Oversubscription can slow or destabilize runs. | Parallelism audit. |

## Interpretation Rules

- Ranking metrics answer whether scores order stocks correctly.
- Portfolio metrics answer whether OOS scores survive policy/cost assumptions.
- Validation metrics answer whether outputs are structurally trustworthy.
- None of these metrics authorizes paper/live trading.
- A green-looking metric with `validation_passed: false` is not usable.
- A positive return with `promotion_thresholds_changed: true` is a guardrail
  failure, not a candidate.

