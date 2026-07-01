# Stock-Alpha Pipeline Deep Dive

This document was generated from `PROJECT_TREE.txt` and the current
stock-alpha modules under `core/research/ml/stock_level/`,
`core/research/ml/stock_level_benchmark_*.py`, and
`core/research/framework/`. It documents observed code paths and marks
uncertainty where behaviour depends on runtime config or generated artifacts.
No benchmark result is claimed here.

See also:
[architecture diagrams](architecture_diagrams.md),
[architecture diagram explainer](architecture_diagram_explainer.md),
[stock-alpha feature explainer](stock_alpha_feature_explainer.md), and
[model and gate explainer](model_and_gate_explainer.md).

## Purpose

In this codebase, stock alpha means cross-sectional, stock-level research:
given one row per `rebalance_date` and `symbol`, the pipeline asks whether
point-in-time signals can rank stocks by future returns, whether engineered
features help, and whether those out-of-sample rankings survive simple
portfolio simulations after costs.

The pipeline exists because older artifact-level prediction files are not
necessarily true stock-level rows. `stock_level_prediction_artifacts.py` states
that the stock-level artifact is meant to create one row per symbol per
rebalance date without replacing existing artifact-level prediction files.

The core questions are:

| Stage | Question answered |
|---|---|
| Stock-level artifact | Do we have one research row per symbol and rebalance date with baseline point-in-time signals and future labels? |
| Alpha features | Do additional point-in-time technical/cross-sectional features exist for those rows? |
| Model ranking benchmark | Can models rank stocks out of sample better than simple momentum baselines? |
| Target comparison | Is the result robust across target definitions, or only one label? |
| Portfolio replay | Do OOS predictions translate into portfolio returns under fixed policies and costs? |
| Policy sweep | Are policy choices robust, feasible, and better than baseline signals after costs? |
| Experiment report | Are outputs complete, fresh, canonical, and guardrail-compliant? |
| Optional attribution | Which features appear important for completed stock-level models? |
| Overnight summary | What did the whole sequential run produce and which comparisons won? |

## Flow

```text
data/reference/universes/
data/processed/stooq_parquet/
cache/ml/{development,benchmark}/expanded_rebalance_dataset.csv
reports/.../meta_auxiliary_predictions.csv
  -> stock_level_prediction_artifacts.csv
  -> stock_level_prediction_artifacts_enriched.csv
  -> baseline/stock_level_model_oos_predictions.csv
  -> enriched/stock_level_model_oos_predictions.csv
  -> target_comparison/
  -> portfolio_replay/
  -> portfolio_policy_sweep/
  -> stock_alpha_experiment_report.json
  -> optional stock_level_feature_attribution.json
  -> overnight_stock_alpha_summary.json
```

Canonical stock-alpha output directories are resolved by
`stock_alpha_paths.stock_alpha_output_dir()`:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/dev/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/full/
```

## Stage 1: Stock-Level Artifact

Main module:

- `core/research/ml/stock_level/stock_level_prediction_artifacts.py`

Entry command when run directly or through return-mechanics/overnight flow:

- `ml-return-mechanics-audit`
- `ml-overnight-stock-alpha`

Inputs:

- Expanded rebalance dataset:
  `ml.expanded_rebalance_dataset_path`, defaulting to
  `cache/ml/.../expanded_rebalance_dataset.csv`
- Meta auxiliary predictions:
  `output_dir / "meta_auxiliary_predictions.csv"`
- Universe YAML paths from `ml.expanded_rebalance_dataset.universe_paths`
- Stooq parquet data from `ml.stooq_parquet_dir`
- Sector reference from `ml.sector_reference_path` or inline
  `ml.sector_by_symbol`
- Market symbol from `ml.stock_ranker_market_symbol`, default `SPY`

Outputs:

- `stock_level_prediction_artifacts.csv`
- `stock_level_prediction_artifacts.json`
- `stock_level_prediction_artifacts.md`

Expected row shape:

- One row per `rebalance_date` and `symbol`
- Identity/context columns such as `sector`, source metadata, average dollar
  volume, and `true_stock_level_row`
- Point-in-time baseline signals such as `predicted_momentum_20d`,
  `predicted_momentum_60d`, `predicted_momentum_120d`,
  `predicted_volatility_20d`, `predicted_drawdown_60d`,
  `predicted_liquidity_score`, and `predicted_risk_adjusted_momentum`
- Future/evaluation labels such as `actual_forward_return_5d`,
  `actual_forward_return_10d`, `actual_future_volatility`,
  `actual_future_drawdown`, `actual_max_adverse_excursion`,
  `actual_market_residual_return_10d`,
  `actual_vol_adjusted_forward_return_10d`,
  `actual_drawdown_adjusted_forward_return_10d`,
  `actual_rank_normalized_forward_return_10d`, and
  `actual_top_decile_label_10d`

Important config keys:

- `stock_alpha_report_root`
- `stock_alpha_run_size`
- `stock_ranker_market_symbol`
- `stock_alpha_dev_required_symbols`
- `expanded_rebalance_dataset_path`
- `expanded_rebalance_dataset.universe_paths`
- `expanded_rebalance_dataset.max_symbols`
- `stooq_parquet_dir`

What can go wrong:

- Missing expanded rebalance dataset
- Missing `meta_auxiliary_predictions.csv`
- Universe paths do not exist or contain too few symbols
- Parquet data missing for required symbols
- Output written outside the canonical run-size directory
- Future labels missing because forward prices are unavailable near the end of
  history

Inspect these report fields:

- `row_count`
- `symbol_count`
- `rebalance_date_count`
- `date_range`
- `missing_prediction_counts`
- `populated_prediction_counts`
- `missing_actual_target_counts`
- `target_audit`
- `market_residual_label_generation`
- `leakage_safety_notes`
- `usable_for_stock_level_ranking`

What it proves:

- The pipeline can form stock-level rows and compute baseline signals/labels.

What it does not prove:

- That ML models add value.
- That labels are sufficient across all targets.
- That any portfolio should be traded.

## Stage 2: Alpha Feature Generation

Main module:

- `core/research/ml/stock_level/stock_level_alpha_features.py`

Inputs:

- `stock_level_base_prediction_artifacts_path`, defaulting to the canonical
  `stock_level_prediction_artifacts.csv`
- Parquet price histories from `StockLevelResearchConfig.parquet_dir`
- SPY/market symbol from `stock_ranker_spy_symbol`

Outputs:

- `stock_level_prediction_artifacts_enriched.csv`
- `stock_level_alpha_feature_audit.csv`
- `stock_level_alpha_feature_audit.json`
- `stock_level_alpha_feature_audit.md`

Engineered feature columns in code:

- `momentum_250d`
- `momentum_acceleration`
- `momentum_persistence`
- `momentum_consistency`
- `relative_momentum_vs_spy`
- `relative_momentum_vs_sector`
- `momentum_percentile`
- `distance_from_52_week_high`
- `drawdown_recovery_days`
- `rolling_max_drawdown_120d`
- `ulcer_index`
- `downside_deviation`
- `volatility_percentile`
- `volatility_trend`
- `volatility_regime`
- `ATR_percentile`
- `sector_relative_strength`
- `industry_relative_strength`

The module documents feature definitions in `FEATURE_DEFINITIONS`. The code uses
history strictly before the rebalance date for time-series features.

Important config keys:

- `stock_alpha_feature_n_jobs`
- `stooq_parquet_dir`
- `stock_ranker_spy_symbol`
- `stock_alpha_run_size`
- dev subset keys: `stock_alpha_dev_max_dates`,
  `stock_alpha_dev_max_symbols`, `stock_alpha_dev_recent_dates_only`

What can go wrong:

- Base artifact missing
- Parquet histories missing or too short for lookbacks
- Feature columns mostly blank because history is insufficient
- Sector/industry-relative features blank if metadata is missing
- Parallelism over-requested relative to symbol count

Inspect these report fields:

- `features`
- `parallelism.requested_workers`
- `parallelism.effective_workers`
- `source_artifact_path`
- `effective_row_count`
- `effective_date_count`
- `effective_symbol_count`

What it proves:

- Extra point-in-time features can be computed for available histories.

What it does not prove:

- That enriched features improve ranking or portfolio outcomes.

## Stage 3: Model Ranking Benchmark

Main modules:

- `core/research/ml/stock_level/stock_level_model_ranking_benchmark.py`
- `core/research/ml/stock_level_benchmark_data.py`
- `core/research/ml/stock_level_benchmark_execution.py`
- `core/research/ml/stock_level_benchmark_evaluation.py`
- `core/research/ml/stock_level_benchmark_models.py`
- `core/research/framework/ranking.py`

Inputs:

- `stock_level_prediction_artifacts_path`
- Baseline artifact for the baseline benchmark
- Enriched artifact for the enriched benchmark

Outputs:

- `stock_level_model_ranking_benchmark.csv`
- `stock_level_model_ranking_benchmark.json`
- `stock_level_model_ranking_benchmark.md`
- `stock_level_model_oos_predictions.csv`

Default feature columns:

- `predicted_momentum_20d`
- `predicted_momentum_60d`
- `predicted_momentum_120d`
- `predicted_volatility_20d`
- `predicted_drawdown_60d`
- `predicted_liquidity_score`
- `predicted_risk_adjusted_momentum`

When `stock_ranker_include_engineered_features` is true, available engineered
features from the alpha feature stage are added.

Model/baseline names in stock-level ranking:

- Tabular ML: `ridge`, `elastic_net`, `random_forest`, `gradient_boosting`
- Sequence/deep: `dlinear`, `patchtst`, `transformer`, `itransformer`,
  `momentum_transformer`, `multitask_transformer`,
  `market_context_encoder`, `news_analysis_transformer`,
  `temporal_fusion_transformer`
- Baselines: `momentum_120d`, `risk_adjusted_momentum`

Important config keys:

- `stock_ranker_target_column`
- `stock_ranker_min_train_dates`
- `stock_ranker_test_window_dates`
- `stock_ranker_embargo_dates`
- `stock_ranker_model_n_jobs`
- `sklearn_n_jobs`
- `stock_ranker_include_sequence_models`
- `stock_ranker_include_engineered_features`
- `stock_ranker_sequence_length`
- `stock_ranker_sequence_epochs`
- `stock_ranker_sequence_batch_size`
- `stock_ranker_sequence_device`

OOS prediction flow:

- Rows are filtered to those with `rebalance_date`, `symbol`, and a numeric
  target.
- Rows must be unique by `(rebalance_date, symbol)`.
- The split is chronological expanding-window.
- `min_train_dates` controls the first training window.
- `embargo_dates` leaves rebalance dates between train and test.
- `test_window_dates` controls each test window.
- Output predictions are only for test rows.
- Payload field `walk_forward.out_of_sample_only` should be true.

What can go wrong:

- Not enough rebalance dates for split settings
- Duplicate `(rebalance_date, symbol)` keys
- Missing target values
- Missing model dependencies
- Sequence models unavailable or erroring
- `news_analysis_transformer` unavailable because no news/sentiment columns
  exist; the code forbids synthetic news inputs
- Nested parallelism oversubscription

Inspect these report fields:

- `leaderboard`
- `best_ml_model`
- `best_ml_vs_momentum_120d`
- `ml_beats_momentum_120d`
- `input_row_count`
- `eligible_row_count`
- `excluded_incomplete_row_count`
- `oos_row_count`
- `oos_date_count`
- `oos_symbol_count`
- `walk_forward`
- `parallelism`
- `unavailable_models`
- `completed_models`

What it proves:

- A model generated chronological OOS predictions and can be compared to
  baseline signals on ranking metrics.

What it does not prove:

- That the strategy survives costs, turnover, concentration, or manual review.

## Baseline vs Enriched

The overnight runner creates two benchmark branches:

- Baseline benchmark: uses the base artifact and sets
  `stock_ranker_include_engineered_features=False`.
- Enriched benchmark: uses `stock_level_prediction_artifacts_enriched.csv` and
  sets `stock_ranker_include_engineered_features=True`.

Both are compared because an ML result is more useful if it improves after
adding leakage-safe engineered features. If the enriched branch performs worse,
the new features may be noisy, sparse, redundant, or overfit.

## Stage 4: Target Comparison

Main module:

- `core/research/ml/stock_level/stock_level_target_comparison.py`

Inputs:

- Usually the enriched artifact
- Target list from `stock_ranker_target_columns`

Default target comparison columns from config defaults:

- `actual_forward_return_10d`
- `actual_market_residual_return_10d`
- `actual_vol_adjusted_forward_return_10d`
- `actual_rank_normalized_forward_return_10d`

Outputs:

- `stock_level_target_comparison.csv`
- `stock_level_target_comparison.json`
- `stock_level_target_comparison.md`

What can go wrong:

- Target column missing
- Target column present but all null
- Too few eligible symbols, rows, or dates
- Target execution error for an otherwise present target

Inspect these report fields:

- `status`
- `skipped_target_count`
- `skipped_targets`
- `targets`
- `target_column_present`
- `target_non_null_count`
- `eligible_row_count`
- `eligible_date_count`
- `eligible_symbol_count`
- `beats_momentum_120d_identical_oos_dates`

What it proves:

- The signal can or cannot generalize across alternative target definitions.

What it does not prove:

- That the selected target is production-validated.

## Stage 5: Portfolio Replay

Main module:

- `core/research/ml/stock_level/stock_level_portfolio_replay.py`

Inputs:

- `stock_level_model_oos_predictions_path`
- `stock_level_model_ranking_benchmark_path`
- Configured `stock_portfolio_replay_signal_columns`

Outputs:

- `stock_level_portfolio_replay_summary.csv`
- `stock_level_portfolio_replay_summary.json`
- `stock_level_portfolio_replay_summary.md`
- `stock_level_portfolio_replay_equity_curves.csv`
- `stock_level_portfolio_replay_holdings.csv`

The module refuses benchmark metadata unless
`walk_forward.out_of_sample_only` is true. It uses OOS rows with `fold_id` and
the target `actual_forward_return_10d`.

Policies in code:

- `long_only_top_decile_equal_weight`
- `long_only_top_n_equal_weight`
- `long_only_score_weighted`
- optional short policies if shorting is allowed

Important config keys:

- `stock_portfolio_replay_enabled`
- `stock_portfolio_replay_top_n`
- `stock_portfolio_replay_cost_bps`
- `stock_portfolio_replay_slippage_bps`
- `stock_portfolio_replay_max_position_weight`
- `stock_portfolio_replay_min_position_weight`
- `stock_portfolio_replay_allow_short`
- `stock_portfolio_replay_signal_columns`

Inspect these report fields:

- `oos_only`
- `summary`
- `winners`
- `best_ml_vs_momentum_120d`
- `net_return`
- `sharpe`
- `max_drawdown`
- `average_turnover`
- `transaction_cost_drag`
- `date_count`

What it proves:

- A signal can be translated into a hypothetical OOS portfolio path under fixed
  rules and costs.

What it does not prove:

- Executability, production readiness, broker behaviour, or live/paper approval.

## Stage 6: Portfolio Policy Sweep

Main module:

- `core/research/ml/stock_level/stock_level_portfolio_policy_sweep.py`

Inputs:

- OOS predictions
- Benchmark JSON with `out_of_sample_only`
- Configured signals, baseline signals, policies, sizing methods, costs,
  turnover caps, volatility targets, and max config caps

Outputs:

- `stock_level_portfolio_policy_sweep.csv`
- `stock_level_portfolio_policy_sweep.json`
- `stock_level_portfolio_policy_sweep.md`
- `stock_level_portfolio_policy_sweep_equity_curves.csv`
- `stock_level_portfolio_policy_sweep_top_holdings.csv`

Policy/sizing dimensions in code include long-only and optional long/short
policies, equal/rank/score/softmax/inverse-volatility sizing, top-N choices,
position caps, cost/slippage assumptions, turnover caps, volatility targets,
and signals.

Important config keys:

- `stock_portfolio_policy_sweep_enabled`
- `stock_portfolio_policy_sweep_n_jobs`
- `stock_portfolio_policy_sweep_signals`
- `stock_portfolio_policy_sweep_baseline_signals`
- `stock_portfolio_policy_sweep_top_n_values`
- `stock_portfolio_policy_sweep_max_position_weights`
- `stock_portfolio_policy_sweep_cost_bps_values`
- `stock_portfolio_policy_sweep_slippage_bps_values`
- `stock_portfolio_policy_sweep_turnover_caps`
- `stock_portfolio_policy_sweep_volatility_targets`
- `stock_portfolio_policy_sweep_allow_short`
- `stock_portfolio_policy_sweep_max_configs_dev`
- `stock_portfolio_policy_sweep_max_configs_benchmark`
- `stock_portfolio_policy_sweep_max_configs_full`

Inspect these report fields:

- `policy_config_count`
- `baseline_coverage`
- `summary`
- `winners`
- `best_ml_vs_momentum_120d`
- `all_candidate_net_returns_negative`
- `best_return_is_negative`
- `parallelism`
- `infeasible_reason`

What it proves:

- Whether feasible policy settings can improve over baseline signals under the
  configured cost and sizing assumptions.

What it does not prove:

- Broker fill quality, live liquidity, tax impact, or approval to trade.

## Stage 7: Experiment Report

Main module:

- `core/research/ml/stock_level/stock_alpha_experiment_report.py`

Outputs:

- `stock_alpha_experiment_report.json`
- `stock_alpha_experiment_report.md`
- registry CSV at `stock_alpha_experiment_registry_path`

The report validates file existence, output roots, run size, guardrail fields,
OOS dates for key artifacts, stale mixed outputs, winners, and momentum
baseline comparison metadata.

Inspect these fields:

- `validation_passed`
- `validation.errors`
- `validation.warnings`
- `validation.unexpected_output_paths`
- `validation.legacy_output_paths_detected`
- `validation.legacy_output_paths_allowed`
- `registry_row`

What it proves:

- A bundle of generated outputs passed structural and guardrail checks.

What it does not prove:

- Economic quality or production approval.

## Stage 8: Optional Attribution

Main module:

- `core/research/ml/stock_level/stock_level_feature_attribution.py`

Inputs:

- OOS predictions
- benchmark paths
- model feature columns

Outputs:

- feature attribution CSV/JSON/Markdown

Attribution is optional and is disabled by default in overnight config. Treat it
as explanatory research only.

## Stage 9: Overnight Summary

Main module:

- `core/research/ml/stock_level/overnight_stock_alpha_runner.py`

Outputs:

- `overnight_stock_alpha_summary.json`
- `overnight_stock_alpha_summary.md`

The overnight runner orchestrates the stages sequentially, records stage
timings, validates returned stage paths against the canonical output root, and
summarizes baseline vs enriched comparisons, winners, portfolio results,
policy-sweep results, parallelism, and guardrails.

Inspect these fields:

- `artifacts`
- `artifact_status`
- `stage_timings`
- `comparisons`
- `winners`
- `parallelism`
- `portfolio_replay`
- `portfolio_policy_sweep`
- `output_root`
- `output_dir`
- `run_size`
- `legacy_output_paths_allowed`
- guardrails

## Dev, Benchmark, and Full

| Run size | What it is for | What it is not for |
|---|---|---|
| `dev` | Fast smoke/debug path with limited dates, symbols, workers, and policy configs. | Research conclusions or trading decisions. |
| `benchmark` | Standard research profile for comparing model/stage behaviour. | Automatic production validation. |
| `full` | Larger run size for deeper validation after dev and benchmark are clean. | Skipping correctness gates or manual review. |

Dev results are not research conclusions because the run is intentionally
subsampled and may disable expensive pieces such as attribution or sequence
models depending on the command.

## Why OOS Predictions Matter

Out-of-sample means each test row is predicted by a model trained only on prior
rebalance dates, with configured embargo dates between train and test. Portfolio
replay and policy sweep must use OOS predictions only; otherwise the simulated
portfolio would be using fitted/in-sample knowledge and the result would be a
leakage-prone backtest, not a research signal.
