# Architecture Overview

This guide was generated from `PROJECT_TREE.txt` and a light read of the CLI,
service, profile, and stock-alpha orchestration modules. It describes the
repository as it exists in that tree. Where names alone do not prove runtime
behaviour, the uncertainty is called out.

For diagram-first navigation, see
[architecture diagrams](architecture_diagrams.md).

## High-Level Shape

The repository is a Python trading and research platform. It contains classic
backtesting, paper-trading support, research workflows, ML research, generated
reports, cache data, and tests.

The practical dependency direction is:

```text
main.py
  -> application/cli.py
    -> application/services/
      -> config/
      -> core/
      -> infrastructure/ only where an adapter/feed/broker is explicitly needed

tests/ validate behaviour across the layers.
cache/, reports/, and logs/ are generated runtime outputs.
```

Research code should remain research-only unless a module is explicitly built
as execution-adjacent. In particular, stock-alpha research modules should not
import broker, live, paper order execution, or order-placement logic.

## Layers

| Layer | Main locations | Role |
|---|---|---|
| CLI entrypoint | `main.py`, `application/cli.py` | Parses mode/profile flags, loads config, builds feeds for modes that need data, and dispatches to services. |
| Application services | `application/services/` | Orchestrates workflows. These modules call core research/domain code and print paths or summaries. |
| Reporting adapters | `application/reporting/` | Formats backtest, walk-forward, dual-momentum, multi-strategy, relative-strength, and paper reports. |
| Config/default/profile loading | `config/`, `configs/`, `application/services/research_profiles.py` | Loads defaults, runtime config, profiles, baselines, champions, experiments, research configs, and paper configs. |
| Core trading domain | `core/entities/`, `core/interfaces/`, `core/indicators/`, `core/services/`, `core/risk/` | Domain objects, protocols, indicators, portfolio/trade services, and risk managers. |
| Engines and execution abstractions | `core/engine/`, `core/execution/` | Backtest/trading engines and fill simulation/execution model code. |
| Paper trading | `core/paper/`, `application/services/paper_*`, `infrastructure/broker/`, `infrastructure/brokers/` | Local paper/shadow-trading workflow and broker adapters. Treat as execution-adjacent. |
| Infrastructure adapters | `infrastructure/` | External data feeds, Alpaca adapters, broker adapters, alert adapters, and importers. |
| Research framework | `core/research/framework/` | Shared config, contracts, data IO, logging, parallelism helpers, ranking, reporting, and walk-forward utilities for research modules. |
| Research workflows | `core/research/` | Backtests, experiments, dual momentum, relative strength, multi-strategy, metrics, result cache, and strategy comparison. |
| ML research | `core/research/ml/` | ML datasets, features, labels, models, audits, allocation, artifacts, meta-ensemble, stock-level alpha, validation, and runtime parallelism. |
| Strategies | `strategies/` | Strategy implementations and filters. |
| Generated outputs | `cache/`, `reports/`, `logs/`, `data/processed/` | Cached datasets, reports, logs, and processed Stooq parquet data. Usually not source code. |
| Tests | `tests/` | Pytest suite covering domain, application, infrastructure, research, ML, stock-alpha, and paper workflows. |

## CLI and Application Services

`application/cli.py` owns the command surface. The CLI mode names show the main
workflows, including:

- classic research modes such as `backtest`, `optimize`, `walk-forward`,
  `compare-strategies`, and `data-audit`
- dual-momentum modes such as `dual-momentum`,
  `dual-momentum-walk-forward`, `dual-momentum-risk-regimes`, and
  `dual-momentum-diagnosis`
- paper modes such as `paper-trade`, `paper-fill`, `paper-status`,
  `paper-report`, `paper-trading`, `paper-dry-run`, `paper-trial`,
  `paper-weekly-summary`, `paper-promotion-checklist`, `paper-run`,
  `paper-repair`, and `paper-reset`
- ML modes such as `ml-research`, `ml-research-batch`,
  `ml-model-contract-audit`, `ml-validate-artifacts`, `ml-meta-ensemble`,
  `ml-return-mechanics-audit`, and the stock-alpha modes

Service modules in `application/services/` should orchestrate, not hide domain
logic. For example, `ml_commands.py` calls ML research writers and prints the
resulting report paths. `paper_commands.py` and related paper modules are
execution-adjacent and should stay separate from research-only modules.

## Config and Profiles

Configuration is split across:

- `config/config.yaml` for the main runtime config
- `config/config_defaults.py`, `config/config_defaults_ml.py`,
  `config/config_defaults_research.py`, and `config/config_defaults_runtime.py`
  for default values
- `config/config_environment.py`, `config/config_loader.py`, and
  `config/config_validation.py` for environment, loading, and validation
- `configs/` for named baselines, champions, experiments, paper configs, and
  research configs
- `configs/research/profiles/` for isolated development and benchmark profiles

`application/services/research_profiles.py` applies profile-specific cache and
report roots. The benchmark profile uses `cache/ml/benchmark` and
`reports/ml/benchmark`; the development profile uses `cache/ml/development` and
`reports/ml/development`.

## Core Domain and Execution

`core/entities/` contains trading objects such as candles, signals, orders,
fills, positions, portfolios, trades, risk context, and result types. These
objects are part of the domain vocabulary.

`core/interfaces/` contains abstractions for brokers, clocks, data feeds,
execution models, risk managers, strategies, and alert services. Code that
needs external data or execution should depend on these abstractions where
possible.

`core/engine/` contains backtest, trading, and execution engines. `core/risk/`
contains risk managers and position sizing. `core/execution/` contains the
simple execution model.

Because this layer includes order and execution concepts, research-only modules
should import from it only when the workflow explicitly needs simulation and the
tests cover that boundary.

## Research and ML

`core/research/` is broad. It includes classic strategy research, dual momentum,
relative strength, multi-strategy portfolios, walk-forward utilities, and a
large ML research area.

`core/research/ml/` includes:

- allocation research
- artifact schema, validation, writers, and report annotation
- data inventory, datasets, rebalance datasets, sector references, and universe
  builders
- audits for adjusted data, returns, benchmark-relative validation, historical
  coverage, model contracts, profit concentration, and independent periods
- feature, label, model, and rebalance pipelines
- model implementations and registry
- overlay and replay utilities
- meta-ensemble research
- stock-level alpha research, including dev smoke, parallelism audit,
  overnight runner, feature attribution, model ranking, target comparison,
  portfolio replay, and policy sweep

Most ML outputs include guardrail metadata such as `research_only: true`,
`trading_impact: none`, `production_validated: false`, and
`promotion_thresholds_changed: false`. Those fields mean the output is evidence,
not a production approval.

## Generated Outputs and Caches

`PROJECT_TREE.txt` shows `cache/` and data directories. The repo also uses
generated `reports/` and `logs/` paths in config and command output. Treat these
as runtime artifacts unless a specific report is intentionally checked in.

Important generated areas include:

- `cache/ml/development/` and `cache/ml/benchmark/`
- `reports/ml/development/` and `reports/ml/benchmark/`
- `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/{dev,benchmark,full}/`
- `logs/` for run logs
- `data/processed/stooq_parquet/` for processed Stooq parquet data

## Tests

`tests/` contains focused pytest coverage across the system. The tree includes
tests for CLI startup, config loading, domain engines, broker/data adapters,
paper workflows, research workflows, ML pipelines, stock-alpha stages, and
guardrails.

For implementation work, prefer focused tests around the touched module. Long
benchmarks and overnight runs are operational checks, not default test runs.

## Dependency Guardrails

- Application services orchestrate.
- Config modules load and validate settings.
- Core domain logic should remain broker/data-provider agnostic.
- Infrastructure modules adapt external systems.
- Paper/live/order execution code should stay isolated from research-only
  modules.
- Research metrics and reports observe results; they should not control
  trading execution.
- Generated reports, logs, and cache files should not be treated as source.
