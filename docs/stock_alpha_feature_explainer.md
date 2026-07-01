# Stock-Alpha Feature Explainer

This document explains the stock-alpha baseline features, engineered alpha
features, and future target labels used by the stock-level research pipeline.
It is a documentation guide, not benchmark evidence.

Primary modules and reports to inspect:

| Topic | Inspect |
| --- | --- |
| Baseline features and targets | `core/research/ml/stock_level/stock_level_prediction_artifacts.py` |
| Engineered features | `core/research/ml/stock_level/stock_level_alpha_features.py` |
| Feature availability | `stock_level_alpha_feature_audit.{csv,json,md}` |
| Base artifact coverage | `stock_level_prediction_artifacts.json` |
| Enriched artifact | `stock_level_prediction_artifacts_enriched.csv` |
| Target comparison | `target_comparison/stock_level_target_comparison.{csv,json,md}` |

All feature calculations described here should be interpreted as point-in-time
unless explicitly marked as a future target. Future target columns start with
`actual_` and must not be used as same-row input features.

## Reading Direction

| Direction | Meaning |
| --- | --- |
| Higher usually better | Larger values often imply a more attractive long candidate. |
| Lower usually better | Smaller values often imply lower risk or better position quality. |
| Context-dependent | The model must learn how to use the feature; high or low is not always good. |
| Target only | Future label used for training/evaluation, not a feature. |

## Baseline Features

These are written by `stock_level_prediction_artifacts.py` into
`stock_level_prediction_artifacts.csv`. They are built from price and dollar
volume history strictly before `rebalance_date`, plus any source prediction
columns found in `meta_auxiliary_predictions.csv`.

| Feature | Plain-English definition | Rough calculation and data needed | Point-in-time | Intended behavior | Why it might help | Why it might hurt | Missingness and inspection | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `predicted_momentum_20d` | Recent short-horizon price momentum. | Trailing return over the prior 20 observations from close prices. | Yes | Captures near-term strength and reversal-prone bursts. | Helps if recent winners continue. | Can chase short-term exhaustion or event spikes. | Missing when fewer than 21 prior closes or bad prices. Inspect `missing_prediction_counts` and the CSV column. | Higher usually better |
| `predicted_momentum_60d` | Medium-term price momentum. | Trailing return over the prior 60 observations from close prices. | Yes | Captures intermediate trend persistence. | Often less noisy than 20-day momentum. | Can lag turning points. | Missing when history is too short. Inspect `stock_level_prediction_artifacts.json`. | Higher usually better |
| `predicted_momentum_120d` | Longer stock-level momentum baseline. | Trailing return over the prior 120 observations from close prices. | Yes | Acts as the main simple ranking baseline. | Provides a strong, interpretable benchmark for ML models. | Can be crowded or slow to adapt after regime changes. | Missing when fewer than 121 prior closes. Inspect baseline coverage and OOS predictions. | Higher usually better |
| `predicted_volatility_20d` | Recent realized volatility. | Standard deviation of daily returns over the prior 20 observations. | Yes | Captures short-term risk and unstable price action. | Helps separate clean trends from noisy moves. | Penalizing volatility can miss breakout names. | Missing when fewer than 21 prior closes. Inspect artifact JSON and policy sweep sizing fields. | Lower usually better, but context-dependent |
| `predicted_drawdown_60d` | Worst recent drawdown from prior peaks. | Minimum peak-to-trough drawdown over the prior 60 observations. Values are usually zero or negative. | Yes | Captures recent damage and downside pressure. | Helps avoid damaged names or size risk. | Deep drawdown can precede rebounds. | Missing when fewer than 60 prior closes. Inspect `missing_prediction_counts`. | Higher usually better because less negative is better |
| `predicted_liquidity_score` | Recent dollar-volume liquidity proxy. | `log1p` of average dollar volume over the prior 63 observations. | Yes | Captures tradeability and data quality. | Helps avoid thin symbols with unstable returns. | Can favor mega-cap names and reduce alpha diversity. | Missing when dollar-volume history is absent. Inspect average dollar volume fields and artifact JSON. | Higher usually better |
| `predicted_risk_adjusted_momentum` | Momentum normalized by recent risk. | `predicted_momentum_60d` divided by max absolute 20-day volatility, 60-day drawdown, and a small floor. | Yes | Rewards trend per unit of recent risk. | Strong baseline when raw momentum is too volatile. | Can overreward low-volatility names with small absolute returns. | Missing when 60-day momentum is missing. Inspect base artifact and model/replay baselines. | Higher usually better |

## Engineered Alpha Features

These are written by `stock_level_alpha_features.py` into
`stock_level_prediction_artifacts_enriched.csv`. Availability is summarized in
`stock_level_alpha_feature_audit.json` under `features`.

| Feature | Plain-English definition | Rough calculation and data needed | Point-in-time | Intended behavior | Why it might help | Why it might hurt | Missingness and inspection | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `momentum_250d` | Approximate one-year trailing return. | Prior 250-observation close-to-close return. | Yes | Captures long-horizon trend. | Adds slower trend context beyond 120 days. | Can lag regime shifts and survivorship-like universe effects. | Needs more than 250 prior closes. Inspect feature audit availability. | Higher usually better |
| `momentum_acceleration` | Whether momentum improves or fades as horizon lengthens. | OLS slope across 20d, 60d, and 120d momentum values. | Yes | Captures trend acceleration or deceleration. | Helps distinguish fresh strength from aging strength. | Sensitive to noisy short-horizon momentum. | Missing if any input momentum horizon is missing. Inspect enriched CSV. | Context-dependent |
| `momentum_persistence` | Fraction of recent rolling 20-day windows that were positive. | Mean positive outcome over latest 120 trailing 20-day return windows. | Yes | Captures consistency of trend. | Helps favor steady momentum over one-off jumps. | Can underweight sharp recoveries. | Needs about 140 prior closes. Inspect feature audit. | Higher usually better |
| `momentum_consistency` | How linear the recent log-price trend is. | R-squared of a line fitted to the last 120 log closes. | Yes | Captures smoothness of trend. | Smooth trends can be easier to rank and size. | Can prefer slow movers and miss volatile winners. | Missing with fewer than 120 valid positive closes. Inspect audit definition and missing counts. | Higher usually better |
| `relative_momentum_vs_spy` | Stock momentum minus SPY momentum. | Stock 120-day momentum minus SPY 120-day momentum on the same date. | Yes | Captures market-relative strength. | Helps avoid names that only rose because the market rose. | SPY may be the wrong benchmark for some sectors. | Missing if stock or SPY history is too short. Inspect `stock_ranker_spy_symbol` and feature audit. | Higher usually better |
| `relative_momentum_vs_sector` | Stock momentum relative to its sector peer mean. | Stock `predicted_momentum_120d` minus same-date sector mean. | Yes | Captures within-sector strength. | Helps compare like with like. | Sector mapping gaps can shrink coverage. | Missing when `sector` is blank or peer values missing. Inspect sector reference and enriched CSV. | Higher usually better |
| `momentum_percentile` | Cross-sectional rank of 120-day momentum on each date. | Percentile rank of `predicted_momentum_120d` within a rebalance date. | Yes | Normalizes momentum across changing market regimes. | Helps models compare relative rank instead of raw returns. | Percentile loses magnitude information. | Missing when momentum is missing for a row/date. Inspect enriched CSV. | Higher usually better |
| `distance_from_52_week_high` | How far the latest close is below its prior 252-observation high. | Latest pre-rebalance close divided by prior 252-observation high minus one. | Yes | Captures proximity to highs. | Names near highs can be persistent winners. | Near-high names may be extended. | Needs at least 252 prior closes. Inspect feature audit. | Higher usually better because closer to zero is stronger |
| `drawdown_recovery_days` | Number of observations since the latest prior 252-observation high. | Count since most recent high in the prior 252 observations. | Yes | Captures how long a drawdown has persisted. | Helps detect stale drawdowns or fast recoveries. | Long recovery periods can also indicate value setups. | Needs at least 252 prior closes. Inspect feature audit. | Lower usually better, but context-dependent |
| `rolling_max_drawdown_120d` | Worst peak-to-trough loss over the prior 120 observations. | Minimum drawdown inside the trailing 120-observation window. | Yes | Captures recent downside path. | Helps identify unstable or damaged trends. | Penalizes volatile winners. | Needs at least 120 prior closes. Inspect feature audit. | Higher usually better because less negative is better |
| `ulcer_index` | Average depth of recent drawdowns. | Root mean square percentage drawdown over the prior 120 observations. | Yes | Captures persistent pain rather than one-day volatility. | Useful for risk-adjusted trend quality. | Can overpenalize rebounds from deep lows. | Needs at least 120 prior closes. Inspect feature audit. | Lower usually better |
| `downside_deviation` | Volatility of negative recent returns only. | Root mean square of negative daily returns over the prior 60 observations. | Yes | Captures downside-specific risk. | Separates upside volatility from downside volatility. | Can miss jump risk if the recent window is calm. | Needs more than 60 prior closes. Inspect feature audit. | Lower usually better |
| `volatility_percentile` | Current volatility relative to its own past year. | Percentile of current 20-day volatility versus prior 252 volatility observations. | Yes | Captures whether volatility is unusually high or low for the symbol. | Helps regime-sensitive models. | Percentile is symbol-relative and may hide absolute risk. | Needs roughly 272 closes. Inspect feature audit. | Context-dependent |
| `volatility_trend` | Short volatility versus medium volatility. | Current 20-day volatility divided by 60-day volatility minus one. | Yes | Captures whether risk is rising or falling. | Rising volatility can warn about unstable moves. | Rising volatility can also accompany breakouts. | Missing if 20d or 60d volatility is unavailable. Inspect enriched CSV. | Context-dependent |
| `volatility_regime` | Coarse volatility bucket. | Bucketed from `volatility_percentile`: 0 low, 1 normal, 2 high. | Yes | Lets models learn regime-specific behavior. | Simple, robust encoding of risk environment. | Buckets discard nuance at thresholds. | Missing when volatility percentile is missing. Inspect feature audit. | Context-dependent |
| `ATR_percentile` | Current normalized ATR versus prior history. | Percentile of ATR(14) divided by close versus prior 252 observations. | Yes | Captures range-based volatility including intraday high/low. | Adds information beyond close-to-close volatility. | Requires reliable high/low data. | Missing with short history or missing high/low. Inspect feature audit and parquet data. | Context-dependent |
| `sector_relative_strength` | Within-sector percentile of 120-day momentum. | Percentile rank within same `sector` and rebalance date. | Yes | Captures sector-relative leadership. | Reduces bias toward hot sectors when picking stocks. | Small sectors can make ranks noisy. | Missing when sector or momentum values are missing. Inspect sector metadata and audit. | Higher usually better |
| `industry_relative_strength` | Within-industry percentile of 120-day momentum. | Percentile rank within same `industry` when industry metadata exists. | Yes | Captures more granular peer leadership. | Useful if industry mapping is populated. | Currently depends on `industry` metadata availability; blanks reduce coverage. | Inspect `industry_metadata_available` in feature audit. | Higher usually better |

## Target and Label Columns

These are future labels written by `stock_level_prediction_artifacts.py`.
They measure outcomes after `rebalance_date`, so they are valid for training
and evaluation but must not become same-row input features.

| Target | What it measures | Why future label, not feature | Encourages model behavior | Useful when | Misleading when |
| --- | --- | --- | --- | --- | --- |
| `actual_forward_return_5d` | Forward 5-observation return from the rebalance close. | It uses future prices after the row date. | Shorter-horizon return ranking. | Testing fast alpha decay or quick rebalance behavior. | Costs, turnover, or event noise dominate. |
| `actual_forward_return_10d` | Forward 10-observation return; current default stock-ranker target. | It requires the future close 10 observations later. | Rank stocks by raw short-term forward return. | Comparing model IC/spread against momentum baselines. | Market-wide moves dominate stock-specific signal. |
| `actual_market_residual_return_10d` | Stock 10-day return minus market proxy 10-day return. | It subtracts future market return from future stock return. | Focus on market-relative stock selection. | Market direction is a confounder. | SPY is not a good benchmark for the symbol universe. |
| `actual_vol_adjusted_forward_return_10d` | 10-day forward return divided by pre-rebalance 20-day volatility. | The numerator is future return; only the volatility denominator is point-in-time. | Reward return per unit of recent risk. | Comparing risk-adjusted target behavior. | Very low volatility inflates small returns. |
| `actual_drawdown_adjusted_forward_return_10d` | Raw 10-day forward return penalized by future adverse drawdown. | It uses future path and future adverse movement. | Favor upside with less future pain. | Testing drawdown-aware label robustness. | It can over-penalize volatile names that recover by horizon end. |
| `actual_rank_normalized_forward_return_10d` | Cross-sectional percentile rank of 10-day forward return on the same rebalance date. | It uses future returns for all symbols on that date. | Learn relative ranking instead of raw magnitude. | Market regime changes make raw returns hard to compare. | Universe size is small or many ties/missing targets exist. |
| `actual_top_decile_label_10d` | Binary label for membership in the top decile by 10-day forward return. | It is derived from future cross-sectional outcomes. | Classify likely top-decile winners. | Top-bucket selection matters more than continuous return. | Sample size is small or top-decile threshold is unstable. |
| `actual_future_volatility` | Realized volatility over the future 10-observation window. | It uses future returns after the row date. | Predict or evaluate future risk. | Risk forecasting and downside-aware comparisons. | Used as a sizing input for the same row. |
| `actual_future_drawdown` | Worst drawdown from rebalance close over the future window. | It uses future prices after the row date. | Identify future downside path risk. | Evaluating adverse excursion and risk labels. | A temporary drawdown does not reflect final return objective. |
| `actual_max_adverse_excursion` | Minimum future return from the rebalance close in the future window. | It uses future intraperiod path. | Penalize names that go deeply against the position. | Stop-loss or path-risk research. | It can reject high-return names with early volatility. |

## Coverage Checklist

Use this quick checklist before interpreting features or targets:

| Question | Inspect |
| --- | --- |
| Did the base artifact write under the active canonical output directory? | `stock_level_prediction_artifacts.json` and `output_dir` metadata |
| Are baseline signals populated? | `missing_prediction_counts`, `populated_prediction_counts` |
| Are future targets populated? | `missing_actual_target_counts`, `target_audit` |
| Did alpha features read the canonical base artifact? | `stock_level_alpha_feature_audit.json` `source_path` / `source_artifact_path` |
| Which engineered features are usable? | `features[*].availability_rate` |
| Did dev/benchmark/full caps change row coverage? | `run_size`, `effective_row_count`, `effective_date_count`, `effective_symbol_count` |
| Are target comparisons skipped? | `skipped_targets`, `skip_reason_code` |

## Interpretation Guardrails

- High feature availability does not prove predictive value.
- A feature win in one benchmark does not prove causality.
- Future `actual_*` columns are labels, not input features.
- Portfolio replay and policy sweep remain simulations.
- Reports with `production_validated: false` do not authorize paper or live trading.
