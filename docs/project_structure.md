# Project Structure

This guide was generated from `PROJECT_TREE.txt`. It documents the actual
folders and notable files visible in that tree, with small annotations from
module names and the CLI command surface. Do not treat it as proof that every
module behaves exactly as named; inspect the module before changing behaviour.

For a visual layer and workflow map, see
[architecture diagrams](architecture_diagrams.md).

## Top-Level Map

| Path | Classification | Purpose |
|---|---|---|
| `.github/` | CI/config | GitHub workflow definitions. |
| `application/` | application orchestration | CLI dispatch, service orchestration, and reporting adapters. |
| `cache/` | generated output | Cached datasets and intermediate ML data. |
| `config/` | config source | Defaults, config loading, environment handling, validation, and main config. |
| `configs/` | config source | Named baselines, champions, paper configs, experiments, and research configs. |
| `core/` | domain and research logic | Trading domain objects, engines, interfaces, risk, services, and research modules. |
| `data/` | data inputs/generated data | Reference universes, raw Stooq data, adjusted prices, and processed parquet. |
| `docs/` | documentation | Project notes and the generated guides in this docs set. |
| `infrastructure/` | external adapters | Alpaca, broker, alert, and data-feed adapters. |
| `strategies/` | strategy logic | Strategy implementations and filters. |
| `tests/` | tests | Pytest coverage for the repository. |
| `main.py` | CLI entrypoint | Starts the command-line application. |
| `PROJECT_TREE.txt` | documentation/source map | Source-of-truth tree used for these docs. |
| `README.md` | documentation | Top-level project introduction and command examples. |

## `.github/`

Purpose: CI and repository automation.

Notable files:

- `.github/workflows/alpaca-paper-trial.yml`

What belongs here: GitHub Actions workflows and repository automation config.

What should not belong here: Python domain logic, generated reports, secrets, or
runtime broker state.

## `application/`

Classification: application orchestration.

Purpose: turns CLI requests into coordinated service calls.

Notable files:

- `application/cli.py`: command parser and dispatcher
- `application/reporting/`: report formatting modules
- `application/services/`: workflow services and command wrappers

What belongs here: orchestration, CLI adapters, command-level flow, and
user-facing reporting wrappers.

What should not belong here: pure domain rules that should be reusable from
tests/backtests, broker-specific strategy logic, or generated output files.

### `application/reporting/`

Classification: reporting.

Modules include:

- `dual_momentum_reporter.py`
- `multi_strategy_reporter.py`
- `paper_reporter.py`
- `relative_strength_reporter.py`
- `walk_forward_reporter.py`

Purpose: format reports for application workflows.

What should not belong here: portfolio mutation, order placement, or strategy
selection side effects.

### `application/services/`

Classification: application services and command orchestration.

Notable groups:

- ML/research commands: `ml_commands.py`, `research_commands.py`,
  `champion_robustness_commands.py`, `research_profiles.py`,
  `runtime_overrides.py`
- strategy research commands: `dual_momentum_commands.py`,
  `multi_strategy_commands.py`, `relative_strength_commands.py`
- paper trading: `paper_commands.py`, `paper_dry_run.py`,
  `paper_monitoring_service.py`, `paper_service.py`,
  `paper_trading_approval.py`, `paper_trading_broker.py`,
  `paper_trading_reporting.py`, `paper_trading_runtime.py`,
  `paper_trading_service.py`, `paper_trading_types.py`
- data/import commands: `adjusted_price_commands.py`, `market_data_loader.py`,
  `stooq_bulk_commands.py`
- broker creation: `broker_factory.py`

What belongs here: command orchestration, profile application, runtime
overrides, service composition.

What should not belong here: low-level ML algorithms, reusable domain objects,
or broker-specific rules inside research workflows.

## `cache/`

Classification: generated output.

Visible paths:

- `cache/data/`
- `cache/ml/benchmark/`
- `cache/ml/development/`
- `cache/ml/multitask_transformer_32_symbol_smoke/`

Purpose: intermediate data, cached ML datasets, and run-specific scratch data.

What belongs here: regenerated cache files.

What should not belong here: source code, hand-authored configs, or canonical
documentation.

## `config/`

Classification: config source.

Notable files:

- `config.yaml`: main runtime config
- `config_defaults.py`
- `config_defaults_ml.py`
- `config_defaults_research.py`
- `config_defaults_runtime.py`
- `config_environment.py`
- `config_loader.py`
- `config_validation.py`

Purpose: load, combine, and validate config. ML defaults include stock-alpha
run-size, output-root, parallelism, report, and guardrail settings.

What should not belong here: generated report outputs or one-off experiment
results.

## `configs/`

Classification: config source.

Purpose: named experiment, champion, baseline, paper, and research
configuration files.

### `configs/baselines/`

Contains dual-momentum and ranked-top baseline YAMLs such as
`dual_momentum_inverse_vol.yaml`, `dual_momentum_scaled_fast_reentry.yaml`, and
ranked-top variants.

Use for: reusable baseline strategy/research definitions.

### `configs/champions/`

Contains frozen champion YAMLs:

- `ranked_top5_bimonthly_v1.yaml`
- `ranked_top5_monthly_exposure90_v1.yaml`

Use for: explicit champion definitions.

### `configs/experiments/`

Contains experiment config such as `hardened_bimonthly_top5.yaml`.

Use for: named experiment settings.

### `configs/paper/`

Contains paper config such as `alpaca_7day_trial.yaml`.

Use for: paper/shadow-trading workflows. Treat as execution-adjacent.

### `configs/research/`

Contains ML and research configs such as DLinear, PatchTST, transformer,
regime/meta-ensemble, stock-level alpha, high-turnover, and overlay comparison
configs.

`configs/research/profiles/` contains:

- `development.yaml`: development profile with `cache/ml/development` and
  `reports/ml/development`
- `benchmark.yaml`: benchmark profile with `cache/ml/benchmark`,
  `reports/ml/benchmark`, and 4-worker stock-alpha outer stages

## `core/`

Classification: domain logic, execution abstractions, and research logic.

Purpose: the reusable heart of the project.

### `core/engine/`

Classification: execution/backtest engines.

Files:

- `backtest_engine.py`
- `execution_engine.py`
- `trading_engine.py`

Use for: engine behaviour and simulation/execution flow.

Do not place here: ML report generation or application CLI code.

### `core/entities/`

Classification: domain entities.

Notable files include `signal.py`, `order.py`, `fill.py`, `portfolio.py`,
`position.py`, `trade.py`, `candle.py`, `risk_context.py`,
`strategy_context.py`, and result objects.

Use for: domain data objects shared by engines, services, strategies, and tests.

Do not place here: broker API calls or report-writing side effects.

### `core/execution/`

Classification: execution abstraction implementation.

Visible file:

- `simple_execution_model.py`

Use for: fill simulation and execution model assumptions.

### `core/indicators/`

Classification: domain utilities.

Files:

- `atr.py`
- `ema.py`
- `rsi.py`
- `sma.py`

Use for: indicator calculations.

### `core/interfaces/`

Classification: interfaces/protocols.

Files:

- `alert_service.py`
- `broker.py`
- `clock.py`
- `data_feed.py`
- `execution_model.py`
- `risk_manager.py`
- `strategy.py`

Use for: dependency inversion boundaries between domain/application code and
external systems.

### `core/paper/`

Classification: execution-adjacent paper trading.

Files:

- `paper_trading_engine.py`

Use for: local paper/shadow-trading engine behaviour.

Research-only modules should not depend on this unless explicitly intended and
tested.

### `core/risk/`

Classification: domain/risk logic.

Files include ATR, simple, volatility, and paper risk managers plus
`position_sizer.py`.

Use for: sizing and risk decisions.

### `core/services/`

Classification: domain services.

Files:

- `indicator_service.py`
- `market_data_service.py`
- `portfolio_engine.py`
- `trade_manager.py`

Use for: reusable services around indicators, market data, portfolio updates,
and trades.

## `core/research/`

Classification: research-only unless a module name explicitly indicates
execution simulation.

Top-level modules include backtest runner, champion robustness,
capital-utilization analysis, experiment reporting, market-regime analysis,
performance metrics, result cache, strategy comparison, strategy factory, trade
analysis, walk-forward, and portfolio utilities.

### `core/research/dual_momentum/`

Classification: research strategy package.

Modules include analytics, config snapshots, data, diagnostics, execution,
experiments, factory, models, portfolio, ranking, regimes, reporting, scoring,
and weighting.

Use for: dual-momentum research, diagnostics, walk-forward, and reporting.

### `core/research/framework/`

Classification: shared research infrastructure.

Modules include config, contracts, data, logging, parallel, ranking, registry,
reporting, and walk-forward.

Use for: shared research config objects, artifact writers, repositories,
logging, and common workflow utilities.

### `core/research/ml/`

Classification: ML research.

Major folders/modules:

- `allocation/`: allocation optimizers, variants, exposures, reporting, search,
  simulation, types, utilities
- `artifacts/`: artifact schema, validators, writers, experiment paths,
  feature cache, report annotation
- `audits/`: adjusted data, benchmark-relative validation, champion baseline,
  data adjustment, historical coverage, independent period, model contract,
  profit concentration, return mechanics
- `data/`: anomaly quarantine, inventory, datasets, history coverage,
  rebalance datasets, sector reference, sequence datasets, universe builder
- `features/`: features, labels, news sentiment
- `meta/`: meta auxiliary, comparison, dataset, ensemble, evaluation, horizon,
  IO, models, overlay, and types
- `metrics/`: calibration, diagnostics, evaluation, leaderboard,
  cross-sectional ranking diagnostics
- `models/`: DLinear, iTransformer, market context encoder, momentum
  transformer, multitask transformer, news analysis transformer, PatchTST,
  registry, temporal fusion transformer, transformer
- `overlays/`: drawdown review, overlay, rule overlay
- `pipelines/`: dataset, feature, label, model, and rebalance pipelines
- `replay/`: canonical continuous equity replay
- `stock_level/`: stock-alpha package and stage writers

Top-level compatibility modules also exist for many stock-level and audit
writers. Verify whether a top-level file is a wrapper before editing.

### `core/research/ml/stock_level/`

Classification: stock-level alpha research.

Files include:

- `overnight_stock_alpha_runner.py`
- `stock_alpha_dev_smoke.py`
- `stock_alpha_experiment_report.py`
- `stock_alpha_parallelism_audit.py`
- `stock_alpha_paths.py`
- `stock_alpha_run_profile.py`
- `stock_level_alpha_features.py`
- `stock_level_feature_attribution.py`
- `stock_level_model_ranking_benchmark.py`
- `stock_level_portfolio_policy_sweep.py`
- `stock_level_portfolio_replay.py`
- `stock_level_prediction_artifacts.py`
- `stock_level_sequence_regressors.py`
- `stock_level_target_comparison.py`
- `trading_research_leaderboard.py`

Use for: stock-level prediction artifacts, alpha features, benchmarks, target
comparison, portfolio replay, policy sweep, attribution, reports, and overnight
summary.

Do not place here: broker placement code, live-trading code, or paper order
submission.

### Other Research Packages

- `core/research/multi_strategy/`: aggregation, data, diagnostics,
  experiments, models, portfolio, sleeves
- `core/research/relative_strength/`: analytics, data, execution, experiments,
  models, portfolio, ranking

These appear to support research workflows for multi-strategy and
relative-strength systems. Inspect individual modules before assuming their
runtime side effects.

## `data/`

Classification: data inputs and generated/processed data.

Visible paths:

- `data/reference/universes/`: `current_32.yaml`, `us_liquid_100.yaml`,
  `us_liquid_250.yaml`, `us_liquid_500.yaml`
- `data/reference/adjusted_prices/`
- `data/raw/stooq_bulk/data/daily/`
- `data/processed/stooq_parquet/`

What belongs here: source/reference market data and processed datasets.

What should not belong here: Python application logic or generated reports.

## `infrastructure/`

Classification: external adapters.

Folders:

- `alerts/`: console alert adapter
- `alpaca/`: Alpaca data and stream feeds
- `broker/`: paper broker
- `brokers/`: Alpaca and fake broker adapters
- `data/`: CSV, Stooq, parquet, adjusted-price, Yahoo, and bulk import data
  adapters

Use for: code that talks to external services, files, feeds, or broker APIs.

Research code should depend on interfaces or generated artifacts rather than
embedding infrastructure logic directly.

## `strategies/`

Classification: strategy logic.

Strategies include Bollinger mean reversion, buy-and-hold, Donchian breakout,
EMA crossover, EMA/RSI filters and pullbacks, ensemble vote, opening range
breakout, RSI variants, and trend pullback. `strategies/filters/` contains
regime filters.

Strategies should emit signals and use `StrategyContext`; they should not
place broker orders directly.

## `tests/`

Classification: tests.

The test tree covers application startup, configs, backtests, engines, brokers,
data feeds, risk, indicators, paper workflows, research workflows, ML
pipelines, audits, stock-alpha stages, and guardrails.

Use focused tests during implementation. Long benchmark runs are not a
substitute for unit or workflow tests.

## Generated `reports/` and `logs/`

`PROJECT_TREE.txt` does not list full `reports/` or `logs/` contents, but config
and service modules write there. Treat these as generated outputs unless a
specific report is intentionally part of project documentation.

Common locations:

- `reports/ml/`
- `reports/ml/development/`
- `reports/ml/benchmark/`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/`
- `logs/`
