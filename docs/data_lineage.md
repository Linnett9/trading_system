# Data Lineage

This document was generated from `PROJECT_TREE.txt` and the modules that build
expanded rebalance datasets and stock-alpha artifacts. It describes where data
comes from, how it moves, and which fields are point-in-time features versus
future labels.

## Repository Data Areas

| Path | Role | Notes |
|---|---|---|
| `data/reference/universes/` | Source/reference config | Universe YAMLs such as `current_32.yaml`, `us_liquid_100.yaml`, `us_liquid_250.yaml`, and `us_liquid_500.yaml`. |
| `data/reference/adjusted_prices/` | Reference/generated adjusted prices | Used by adjusted-price workflows; inspect producing command before editing. |
| `data/raw/stooq_bulk/data/daily/` | Raw data | Raw Stooq daily files. |
| `data/processed/stooq_parquet/` | Processed data | Parquet price histories consumed by ML/stock-alpha modules. |
| `cache/ml/development/` | Generated cache | Development-profile ML cache. |
| `cache/ml/benchmark/` | Generated cache | Benchmark-profile ML cache, including expanded rebalance datasets. |
| `reports/` | Generated reports | Research outputs, audits, stock-alpha outputs, logs referenced by reports. |
| `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/` | Canonical stock-alpha report root | Contains `dev/`, `benchmark/`, and `full/` run-size directories. |

Generated outputs are not usually source code and should not be treated as
inputs unless the workflow explicitly resumes from them.

## High-Level Lineage

```text
raw Stooq daily files
  -> processed Stooq parquet
  -> ML/research feature rows and expanded rebalance dataset
  -> stock-level prediction artifact
  -> enriched stock-level artifact
  -> chronological OOS predictions
  -> target comparison / portfolio replay / policy sweep
  -> experiment report / overnight summary
```

## Expanded Rebalance Dataset

Relevant modules:

- `core/research/ml/pipelines/rebalance_pipeline.py`
- `core/research/ml/data/rebalance_dataset.py`

The expanded rebalance dataset is written to
`ml.expanded_rebalance_dataset_path`, which profiles redirect to paths such as:

- `cache/ml/development/expanded_rebalance_dataset.csv`
- `cache/ml/benchmark/expanded_rebalance_dataset.csv`

The builder uses:

- `ml.expanded_rebalance_dataset.rebalance_frequencies`
- `ml.expanded_rebalance_dataset.top_n_values`
- `ml.expanded_rebalance_dataset.weightings`
- `ml.expanded_rebalance_dataset.universe_paths`
- `ml.expanded_rebalance_dataset.max_symbols`
- benchmark symbols such as `SPY`
- price candles by symbol
- sector metadata when available

Rows include rebalance information, selected symbols, exposure/weight
diagnostics, market context, future return/drawdown labels, and fields used by
older ML regime/classification research.

## Stock-Alpha Symbol Set

The stock-level artifact reads universe symbols through
`_universe_symbols()` in `stock_level_prediction_artifacts.py`.

It uses:

- `ml.expanded_rebalance_dataset.universe_paths`
- fallback `data/reference/universes/current_32.yaml`
- optional `ml.expanded_rebalance_dataset.max_symbols`

For price loading, it also includes required symbols from
`stock_alpha_dev_required_symbols`, defaulting to `SPY`/market symbol if not
otherwise configured. This ensures market data can exist even when SPY is not a
tradable candidate.

## Rebalance Dates

In `stock_level_prediction_artifacts.py`, dates are selected as:

- artifact dates from `meta_auxiliary_predictions.csv` when present, using
  `rebalance_date` or `date`
- otherwise expanded dataset dates using `rebalance_date` or `feature_date`

The exact date set therefore depends on generated upstream artifacts and the
profile cache/report paths.

## One Stock-Level Row

One row in `stock_level_prediction_artifacts.csv` represents:

```text
(rebalance_date, symbol)
```

The row joins:

- symbol identity and sector metadata
- point-in-time baseline signals from prior prices/volume
- existing artifact prediction fields when available
- market/context columns from the expanded rebalance dataset
- future labels computed from prices after the rebalance date

Rows are expected to be unique by `(rebalance_date, symbol)` before model
ranking.

## Point-in-Time Features

Point-in-time fields are allowed as model inputs because the code computes them
from data available at or before the rebalance date. Examples include:

- `predicted_momentum_20d`
- `predicted_momentum_60d`
- `predicted_momentum_120d`
- `predicted_volatility_20d`
- `predicted_drawdown_60d`
- `predicted_liquidity_score`
- `predicted_risk_adjusted_momentum`
- context fields such as `breadth_above_sma_200`,
  `spy_realized_volatility_21d`, `spy_realized_volatility_63d`,
  `spy_max_drawdown_63d`, and `spy_max_drawdown_126d`
- engineered alpha features such as `momentum_250d`,
  `relative_momentum_vs_spy`, `distance_from_52_week_high`,
  `rolling_max_drawdown_120d`, `ulcer_index`, and `ATR_percentile`

The feature-generation code uses history strictly before `rebalance_date` for
time-series features.

## Future Labels

Future/evaluation labels are not model inputs. They are allowed for training
targets and evaluation because supervised learning needs observed outcomes, but
they must never be used as prediction features for the same row.

Future label columns include:

- `actual_forward_return_5d`
- `actual_forward_return_10d`
- `actual_future_volatility`
- `actual_future_drawdown`
- `actual_max_adverse_excursion`
- `actual_market_residual_return_10d`
- `actual_vol_adjusted_forward_return_10d`
- `actual_drawdown_adjusted_forward_return_10d`
- `actual_rank_normalized_forward_return_10d`
- `actual_top_decile_label_10d`

The artifact audit includes leakage safety notes that actual targets use
post-rebalance prices and are evaluation fields only.

## SPY and Market Data

`stock_ranker_market_symbol` defaults to `SPY`. It is used to compute
`actual_market_residual_return_10d`:

```text
stock actual_forward_return_10d - SPY forward_return_10d
```

SPY history is also used in alpha features such as
`relative_momentum_vs_spy`. The artifact audit records
`market_residual_label_generation` including market symbol and load status.

## Dev Subsetting

`core/research/ml/stock_level/stock_alpha_run_profile.py` applies dev
subsetting. For `run_size != "dev"`, rows are unchanged except count metadata.

For `run_size == "dev"`:

- select recent dates if `stock_alpha_dev_recent_dates_only` is true
- cap dates by `stock_alpha_dev_max_dates`
- cap symbols by `stock_alpha_dev_max_symbols`
- sort by rebalance date and symbol

Dev smoke further caps dates/symbols/configs and disables attribution. Dev is a
pipeline check, not a research conclusion.

## Benchmark and Full

Benchmark and full do not use the dev row subset in
`apply_stock_alpha_run_profile()`. They differ through config/run-size caps,
especially policy sweep caps:

- `stock_portfolio_policy_sweep_max_configs_benchmark`
- `stock_portfolio_policy_sweep_max_configs_full`

Other runtime differences depend on profile/config values.

## Inspecting Counts After a Run

Useful JSON fields:

- Stock artifact:
  - `row_count`
  - `symbol_count`
  - `rebalance_date_count`
  - `date_range`
  - `target_audit`
- Alpha feature audit:
  - `effective_row_count`
  - `effective_date_count`
  - `effective_symbol_count`
  - `parallelism`
- Model benchmark:
  - `input_row_count`
  - `eligible_row_count`
  - `excluded_incomplete_row_count`
  - `input_date_count`
  - `input_symbol_count`
  - `oos_row_count`
  - `oos_date_count`
  - `oos_symbol_count`
- Target comparison:
  - `targets`
  - `target_non_null_count`
  - `eligible_row_count`
  - `eligible_date_count`
  - `eligible_symbol_count`
  - `skipped_targets`
- Experiment report:
  - `registry_row.row_count`
  - `registry_row.date_count`
  - `registry_row.symbol_count`
  - `registry_row.oos_date_count`

## Questions This Doc Can Answer After A Run

How much data was used?

- Inspect `row_count`, `eligible_row_count`, `oos_row_count`, and effective
  counts in stage JSON files.

How many symbols?

- Inspect `symbol_count`, `input_symbol_count`, `oos_symbol_count`, or
  `effective_symbol_count`.

How many rebalance dates?

- Inspect `rebalance_date_count`, `input_date_count`, `oos_date_count`, or
  `effective_date_count`.

What date range?

- Inspect stock artifact `date_range` and benchmark `walk_forward.folds`.

How many OOS dates?

- Inspect model benchmark `oos_date_count` and portfolio replay/sweep
  `date_count` values.

Which artifact was used?

- Inspect `source_artifact_path`, `artifact_status.path`, and experiment report
  `artifacts`.

Which target columns were populated?

- Inspect stock artifact `target_audit` and target comparison
  `target_non_null_count` by target.

