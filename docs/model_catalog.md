# Model Catalog

This catalog was generated from `PROJECT_TREE.txt`,
`core/research/ml/models/`, `core/research/ml/models/registry.py`,
`core/research/ml/stock_level_benchmark_models.py`, and
`core/research/ml/stock_level_benchmark_types.py`.

Some model modules are used by older regime/classification research, while
stock-alpha ranking uses a regression adapter for tabular and sequence
rankers. Inspect the command path before assuming a model is active in a
specific run.

## Stock-Alpha Ranking Models and Baselines

| Name | Type | Implemented/registered in | Input | Output | Evaluated by | Use/risk |
|---|---|---|---|---|---|---|
| `momentum_120d` | baseline signal | `BASELINE_COLUMNS` in `stock_level_benchmark_types.py` | `predicted_momentum_120d` | ranking score | OOS-aligned ranking metrics and portfolio replay | Must remain in comparisons; simple trend baseline. |
| `risk_adjusted_momentum` | baseline signal | `BASELINE_COLUMNS` in `stock_level_benchmark_types.py` | `predicted_risk_adjusted_momentum` | ranking score | OOS-aligned ranking metrics | Useful baseline adjusting momentum by volatility/drawdown risk. |
| `ridge` | tabular linear regressor | `stock_level_benchmark_models.py` | stock-level feature columns | predicted 10d forward return | OOS leaderboard | Stable baseline; can underfit nonlinear effects. |
| `elastic_net` | tabular regularized linear regressor | `stock_level_benchmark_models.py` | stock-level feature columns | predicted 10d forward return | OOS leaderboard | Sparse-ish stable baseline; sensitive to feature scaling/alpha choices. |
| `random_forest` | tabular tree ensemble regressor | `stock_level_benchmark_models.py` | stock-level feature columns | predicted 10d forward return | OOS leaderboard | Captures nonlinear interactions; can overfit noisy cross-sections. |
| `gradient_boosting` | tabular boosted tree regressor | `stock_level_benchmark_models.py` | stock-level feature columns | predicted 10d forward return | OOS leaderboard | Strong nonlinear baseline; risk of fitting small/noisy patterns. |
| `dlinear` | sequence/deep regressor adapter | `stock_level_sequence_regressors.py`, `dlinear_model.py` | per-symbol feature sequences | predicted 10d forward return | OOS leaderboard | Cheap sequence baseline; if deep models cannot beat it, complexity may not help. |
| `patchtst` | sequence/deep regressor adapter | `stock_level_sequence_regressors.py`, `patchtst_model.py` | per-symbol feature sequences | predicted 10d forward return | OOS leaderboard | Patch-based transformer; more expensive and validation-hungry. |
| `transformer` | sequence/deep regressor adapter | `stock_level_sequence_regressors.py`, `transformer_model.py` | per-symbol feature sequences | predicted 10d forward return | OOS leaderboard | General sequence model; high overfit risk on small samples. |
| `itransformer` | sequence/deep regressor adapter | `stock_level_sequence_regressors.py`, `itransformer_model.py` | per-symbol feature sequences | predicted 10d forward return | OOS leaderboard | Inverted-transformer-style sequence model; expensive. |
| `momentum_transformer` | sequence/deep regressor adapter | `stock_level_sequence_regressors.py`, `momentum_transformer_model.py` | per-symbol feature sequences | predicted 10d forward return | OOS leaderboard | Momentum-specialized architecture; still must beat simple momentum. |
| `multitask_transformer` | sequence/deep regressor adapter | `stock_level_sequence_regressors.py`, `multitask_transformer_model.py` | sequences plus auxiliary targets for multitask path | predicted 10d forward return and auxiliary outputs | OOS leaderboard | May regularize via related targets; more moving parts. |
| `market_context_encoder` | context/sequence model | `stock_level_sequence_regressors.py`, `market_context_encoder_model.py` | feature columns plus market context columns | predicted 10d forward return | OOS leaderboard | Tests market context usefulness; can overfit regime noise. |
| `news_analysis_transformer` | news/context sequence model | `stock_level_sequence_regressors.py`, `news_analysis_transformer_model.py` | feature columns plus news/sentiment columns | predicted 10d forward return | OOS leaderboard when news columns exist | Code marks unavailable if no point-in-time news/sentiment columns exist. |
| `temporal_fusion_transformer` | sequence/context model | `stock_level_sequence_regressors.py`, `temporal_fusion_transformer_model.py` | feature columns plus market context columns | predicted 10d forward return | OOS leaderboard | Expensive context model; requires strong OOS validation. |

## Broader ML Research Models

| Model/family | Module | Main use in this repo | Notes |
|---|---|---|---|
| No-op model | `models/registry.py` | Safe wiring baseline for older ML research | Predicts neutral outputs. |
| Logistic regression | `models/registry.py` | Older regime/classification research | Deterministic scikit-learn baseline. |
| Random forest classifier | `models/registry.py` | Older classification research | Tree classifier with constrained depth. |
| Gradient boosting classifier | `models/registry.py` | Older classification research | Tree boosting baseline. |
| DLinear | `models/dlinear_model.py` | Sequence classification and stock-alpha regressor adapter | Linear sequence baseline. |
| iTransformer | `models/itransformer_model.py` | Sequence classification and stock-alpha regressor adapter | PyTorch dependency. |
| Market Context Encoder | `models/market_context_encoder_model.py` | Context-aware research and stock-alpha sequence adapter | Uses market context features in stock-alpha. |
| Momentum Transformer | `models/momentum_transformer_model.py` | Momentum-aware sequence research and stock-alpha adapter | Compares against simple momentum. |
| Multitask Transformer | `models/multitask_transformer_model.py` | Multitask sequence research and stock-alpha adapter | Handles classification plus regression targets in broader ML. |
| News Analysis Transformer | `models/news_analysis_transformer_model.py` | News/sentiment research and stock-alpha adapter when news columns exist | Synthetic news inputs are explicitly forbidden in stock-alpha ranking. |
| PatchTST | `models/patchtst_model.py` | Patch-based sequence research and stock-alpha adapter | More expensive than linear baselines. |
| Temporal Fusion Transformer | `models/temporal_fusion_transformer_model.py` | Context/sequence research and stock-alpha adapter | Lightweight/TFT-style module in this repo. |
| Transformer | `models/transformer_model.py` | General sequence research and stock-alpha adapter | Attention weights are not treated as reliable feature importances. |

## Model Application Flow in Stock-Alpha

1. `stock_level_model_ranking_benchmark.py` reads the artifact path from
   `StockLevelResearchConfig`.
2. Rows are filtered to valid `rebalance_date`, `symbol`, and target values.
3. Available feature columns are selected. Engineered columns are added only
   when `stock_ranker_include_engineered_features` is true and values exist.
4. Walk-forward partitions are chronological expanding-window splits.
5. Each model predicts only test rows.
6. Predictions are written as
   `stock_level_predicted_forward_return_10d_{model_name}`.
7. Baseline columns are evaluated alongside model predictions.
8. Leaderboards are ranked by descending mean Spearman IC, then descending
   top-minus-bottom spread.

## Why The Baselines Matter

Linear models are useful because they are stable, fast, and hard to beat if the
available features only contain simple monotonic relationships. Ridge and
ElasticNet are not glamorous, but they are good sanity checks.

Tree models may help when nonlinear feature interactions matter, for example
momentum behaving differently under high volatility or drawdown regimes. Their
risk is that they can fit quirks of dates/symbols that do not persist.

Sequence/deep models are more expensive and need stronger validation because
they have more capacity, more hyperparameters, and more ways to overfit. They
should not be trusted just because they are more complex.

Simple baselines like `momentum_120d` and `risk_adjusted_momentum` must remain
in comparisons because a stock-alpha model is not useful if it cannot beat
simple, transparent ranking rules on identical OOS dates.

## Common Failure and Overfitting Risks

- Too few rebalance dates for the split settings.
- Too few symbols per date for meaningful cross-sectional metrics.
- Missing target values or target labels populated only for narrow periods.
- Enriched features mostly blank due to short price histories.
- Model wins only one metric but loses to momentum after costs.
- Sequence model unavailable because dependencies or required columns are
  missing.
- `news_analysis_transformer` unavailable because no news/sentiment columns are
  present.
- Strong performance concentrated in one year, month, or symbol.
- Portfolio returns positive before costs but negative after cost/slippage drag.

