# Research Workflows

This guide was generated from `PROJECT_TREE.txt` plus the CLI mode and service
names. It documents likely workflow shape, entrypoints, modules, and outputs.
When runtime behaviour matters, inspect the command module and tests before
changing it.

Deep stock-alpha references:
[pipeline deep dive](stock_alpha_pipeline_deep_dive.md),
[data lineage](data_lineage.md), [model catalog](model_catalog.md),
[metrics glossary](metrics_glossary.md), and
[research gates and trade conditions](research_gates_and_trade_conditions.md).

## Workflow Index

| Workflow | Likely entry command | Main modules | Classification |
|---|---|---|---|
| Classic backtest/research | `backtest`, `optimize`, `walk-forward`, `compare-strategies`, `data-audit` | `application/services/research_commands.py`, `core/research/*`, `core/engine/*` | research/backtest |
| Dual momentum | `dual-momentum`, `dual-momentum-walk-forward`, `dual-momentum-risk-regimes`, `dual-momentum-diagnosis` | `application/services/dual_momentum_commands.py`, `core/research/dual_momentum/` | research |
| Relative strength | `relative-strength` | `application/services/relative_strength_commands.py`, `core/research/relative_strength/` | research |
| Multi-strategy | `multi-strategy`, `multi-strategy-walk-forward` | `application/services/multi_strategy_commands.py`, `core/research/multi_strategy/` | research |
| Champion robustness | `champion-robustness` | `application/services/champion_robustness_commands.py`, `core/research/champion_robustness.py` | research |
| ML research | `ml-research`, `ml-research-batch`, `ml-meta-ensemble` | `application/services/ml_commands.py`, `core/research/ml/` | research |
| ML audits | `ml-return-mechanics-audit`, `ml-model-contract-audit`, `ml-validate-artifacts` | `core/research/ml/audits/`, `core/research/ml/artifacts/` | research |
| Allocation research | part of ML research/audits | `core/research/ml/allocation/` | research |
| Stock-alpha research | stock-alpha CLI modes | `core/research/ml/stock_level/` | research-only |
| Paper trading | `paper-*` modes | `application/services/paper_*`, `core/paper/`, `infrastructure/broker*` | execution-adjacent |

## Classic Backtest and Strategy Research

Purpose: evaluate strategies, optimize parameters, run walk-forward validation,
compare strategy results, and audit data.

Likely entry commands:

```bash
python main.py --mode backtest
python main.py --mode optimize
python main.py --mode walk-forward
python main.py --mode compare-strategies
python main.py --mode data-audit
```

Main modules:

- `application/services/research_commands.py`
- `core/research/backtest_runner.py`
- `core/research/parameter_optimizer.py`
- `core/research/walk_forward.py`
- `core/research/strategy_comparison.py`
- `core/research/performance_metrics.py`
- `core/engine/backtest_engine.py`
- `strategies/`

Expected outputs: backtest reports, comparison CSV/JSON/Markdown files, and
walk-forward reports under configured report directories.

Run when: changing a strategy, engine assumption, risk sizing, indicators, or
research gate.

Do not run when: a focused unit test is enough or a long workflow would collide
with active edits.

## Dual Momentum Research

Purpose: evaluate and diagnose dual-momentum portfolios, risk-regime variants,
walk-forward behaviour, ranking, weighting, and champion configs.

Entry commands:

```bash
python main.py --mode dual-momentum --universe stocks
python main.py --mode dual-momentum-walk-forward --universe stocks
python main.py --mode dual-momentum-risk-regimes --universe stocks
python main.py --mode dual-momentum-diagnosis --universe stocks
```

Main modules:

- `application/services/dual_momentum_commands.py`
- `application/services/dual_momentum_config.py`
- `application/reporting/dual_momentum_reporter.py`
- `core/research/dual_momentum/`
- `configs/baselines/`
- `configs/champions/`

Expected outputs: dual-momentum reports, diagnostics, walk-forward summaries,
and champion/baseline comparison artifacts.

Run when: changing dual-momentum configs, ranking, weighting, regime handling,
or champion validation.

Do not run when: editing stock-alpha or ML artifacts that do not affect
dual-momentum logic.

## Champion Robustness and Baseline Audits

Purpose: stress-check champion behaviour and compare against baseline
expectations.

Entry command:

```bash
python main.py --mode champion-robustness
```

Related ML audit commands:

```bash
python main.py --mode ml-return-mechanics-audit --profile benchmark
```

Main modules:

- `application/services/champion_robustness_commands.py`
- `core/research/champion_robustness.py`
- `core/research/ml/audits/champion_baseline_audit.py`
- `core/research/ml/champion_baseline_audit.py`
- `configs/champions/`
- `configs/baselines/`

Expected outputs: robustness reports, champion baseline audit outputs, and
readiness/diagnostic summaries.

Run when: evaluating a candidate champion or checking benchmark-relative
behaviour.

Do not run when: you only need a stock-alpha dev smoke or syntax-level check.

## Adjusted Price and Data Validation Audits

Purpose: validate adjusted price data, return mechanics, replay alignment,
historical coverage, independent periods, and data anomalies.

Entry commands:

```bash
python main.py --mode ml-refresh-adjusted-prices
python main.py --mode ml-return-mechanics-audit --profile benchmark
```

Main modules:

- `application/services/adjusted_price_commands.py`
- `core/research/ml/audits/adjusted_data_*`
- `core/research/ml/audits/adjusted_price_replay.py`
- `core/research/ml/audits/adjusted_replay_alignment_audit.py`
- `core/research/ml/audits/data_adjustment_validation*.py`
- `core/research/ml/audits/return_mechanics*.py`
- `core/research/ml/audits/historical_coverage_audit.py`
- `core/research/ml/audits/independent_period_expansion_audit.py`
- `infrastructure/data/yahoo_adjusted_price_importer.py`
- `infrastructure/data/adjusted_price_csv_data_feed.py`

Expected outputs: JSON, CSV, and Markdown audit reports under `reports/ml...`;
updated adjusted price files under data reference paths when refresh commands
are run.

Run when: price data, adjusted replay, or return mechanics are suspect.

Do not run when: changing only documentation or config comments.

## ML Model Research

Purpose: train/evaluate ML models, build datasets, run research batches, create
meta-ensembles, inspect inventories, and validate artifacts.

Entry commands:

```bash
python main.py --mode ml-research --profile development
python main.py --mode ml-research-batch --profile benchmark
python main.py --mode ml-meta-ensemble --profile benchmark
python main.py --mode ml-data-inventory --profile benchmark
python main.py --mode ml-build-universes --profile benchmark
python main.py --mode ml-run-inventory --profile benchmark
python main.py --mode ml-clean-incomplete-runs --profile benchmark
```

Main modules:

- `application/services/ml_commands.py`
- `core/research/ml/experiment_runner.py`
- `core/research/ml/pipelines/`
- `core/research/ml/models/`
- `core/research/ml/meta/`
- `core/research/ml/data/`
- `configs/research/`

Expected outputs: model artifacts, leaderboards, inventory reports, datasets,
meta-ensemble reports, prediction artifacts, and run registries.

Run when: changing ML features, labels, model implementation, datasets, or
meta-ensemble logic.

Do not run when: a specific stock-alpha stage or artifact validator test is the
safer feedback loop.

## Artifact Validation and Reporting

Purpose: validate prediction artifact schemas, annotate reports, and confirm
expected outputs exist and remain within configured output roots.

Entry command:

```bash
python main.py --mode ml-validate-artifacts --profile benchmark
```

Main modules:

- `core/research/ml/artifact_validator.py`
- `core/research/ml/artifacts/artifact_schema.py`
- `core/research/ml/artifacts/artifact_validator.py`
- `core/research/ml/artifacts/artifact_writers.py`
- `core/research/ml/artifacts/report_annotation.py`
- `core/research/ml/stock_level/stock_alpha_experiment_report.py`

Expected outputs: artifact validation reports and experiment report validation
payloads.

Run when: adding/changing artifact columns, report paths, or output-root logic.

Do not run when: the output files are known stale and a stage needs to be
regenerated first.

## Allocation Research

Purpose: evaluate allocation policies, variants, exposure controls, searches,
and simulations.

Likely entry: ML research and meta-ensemble command paths that produce
allocation outputs.

Main modules:

- `core/research/ml/allocation/allocation_optimizer.py`
- `core/research/ml/allocation/allocation_v2.py`
- `core/research/ml/allocation/allocation_v2_variants.py`
- `core/research/ml/allocation/exposures.py`
- `core/research/ml/allocation/reporting.py`
- `core/research/ml/allocation/search.py`
- `core/research/ml/allocation/simulation.py`

Expected outputs: allocation policy comparison, optimizer results, selected
exposure paths, allocation reports, and leaderboard inputs.

Run when: changing allocation policy, exposure handling, or optimizer search.

Do not run when: validating stock-level artifact path plumbing only.

## Stock-Level Alpha Research

Purpose: build stock-level artifacts, add engineered alpha features, rank
models, compare targets, replay portfolios, sweep portfolio policies, generate
experiment reports, optionally attribute features, and write overnight
summaries.

Entry commands:

```bash
python main.py --mode ml-stock-alpha-dev-smoke --profile benchmark
python main.py --mode ml-stock-alpha-parallelism-audit --profile benchmark
python main.py --mode ml-stock-level-alpha-features --profile benchmark
python main.py --mode ml-stock-level-alpha-benchmark --profile benchmark
python main.py --mode ml-stock-level-target-comparison --profile benchmark
python main.py --mode ml-stock-level-portfolio-replay --profile benchmark
python main.py --mode ml-stock-level-portfolio-policy-sweep --profile benchmark
python main.py --mode ml-stock-alpha-experiment-report --profile benchmark
python main.py --mode ml-stock-level-feature-attribution --profile benchmark
python main.py --mode ml-overnight-stock-alpha --profile benchmark
```

Main modules:

- `application/services/ml_commands.py`
- `core/research/framework/config.py`
- `core/research/ml/runtime_parallelism.py`
- `core/research/ml/stock_level/stock_alpha_paths.py`
- `core/research/ml/stock_level/stock_alpha_run_profile.py`
- `core/research/ml/stock_level/stock_level_prediction_artifacts.py`
- `core/research/ml/stock_level/stock_level_alpha_features.py`
- `core/research/ml/stock_level/stock_level_model_ranking_benchmark.py`
- `core/research/ml/stock_level/stock_level_target_comparison.py`
- `core/research/ml/stock_level/stock_level_portfolio_replay.py`
- `core/research/ml/stock_level/stock_level_portfolio_policy_sweep.py`
- `core/research/ml/stock_level/stock_alpha_experiment_report.py`
- `core/research/ml/stock_level/stock_alpha_dev_smoke.py`
- `core/research/ml/stock_level/stock_alpha_parallelism_audit.py`
- `core/research/ml/stock_level/stock_level_feature_attribution.py`
- `core/research/ml/stock_level/overnight_stock_alpha_runner.py`

Expected outputs: canonical stock-alpha outputs under
`reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/{dev,benchmark,full}/`.

Run when: evaluating stock-level alpha model and portfolio behaviour.

Do not run when: broker/paper/live/order execution code is being changed, or
when a dev smoke has not passed before a longer benchmark/full run.

## Stock-Alpha Stage Notes

| Stage | Purpose | Expected outputs | Run-size notes |
|---|---|---|---|
| Dev smoke | Small safe end-to-end stock-alpha run | `stock_alpha_dev_smoke_report.{json,md}` | Forces `run_size=dev`, caps dates/symbols/configs, disables attribution. |
| Parallelism audit | Reports requested/effective workers and nested caps | `stock_alpha_parallelism_audit.{json,md}` | Run before benchmark/full. Benchmark expects 4 outer workers for feature/model/target/sweep with nested caps. |
| Stock-level artifact | Builds base stock-level prediction artifact | `stock_level_prediction_artifacts.{csv,json,md}` | Must land in canonical run-size output dir. |
| Alpha features | Enriches base artifact with engineered alpha features | enriched CSV plus alpha feature audit | Reads canonical base artifact. |
| Model ranking benchmark | Compares ranking models and writes OOS predictions | leaderboard CSV/JSON/MD and OOS predictions CSV | Benchmark/full may be longer than dev. |
| Target comparison | Compares target columns | target comparison CSV/JSON/MD | Missing targets may be skipped and reported. |
| Portfolio replay | Replays portfolio policies over OOS predictions | summary, equity curves, holdings | Research-only; does not place trades. |
| Policy sweep | Sweeps policy settings | sweep summary, equity curves, top holdings | Dev caps policy configs; benchmark/full expand. |
| Experiment report | Validates and summarizes stage outputs | experiment report JSON/MD and registry row | Uses output-root and guardrail checks. |
| Feature attribution | Optional feature attribution | attribution CSV/JSON/MD | Disabled by default for dev/overnight unless configured. |
| Overnight runner | Sequential full stock-alpha workflow | overnight summary JSON/MD | Orchestrates all stages. |

## Paper-Trading Related Commands

Paper modes are execution-adjacent, not research-only:

```bash
python main.py --mode paper-trade --universe stocks
python main.py --mode paper-status
python main.py --mode paper-report --universe stocks
python main.py --mode paper-trading --dry-run
python main.py --mode paper-trading --submit
```

Main modules:

- `application/services/paper_commands.py`
- `application/services/paper_trading_service.py`
- `application/services/paper_trading_runtime.py`
- `application/services/paper_trading_broker.py`
- `core/paper/paper_trading_engine.py`
- `infrastructure/broker/`
- `infrastructure/brokers/`

Do not mix paper order placement logic into research-only modules. Research
outputs can inform review, but `production_validated: false` means they are not
approved for live or paper trading.
