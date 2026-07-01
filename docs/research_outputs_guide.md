# Research Outputs Guide

This guide was generated from `PROJECT_TREE.txt`, CLI/service names, and
stock-alpha writer output names. Generated outputs are usually not source code.
They generally should not be committed unless a report is intentionally needed
for review or documentation.

Report paths can differ by profile and run size. For stock-alpha, canonical
outputs live under:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/dev/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/full/
```

See also: [data lineage](data_lineage.md), [metrics glossary](metrics_glossary.md),
[architecture diagram explainer](architecture_diagram_explainer.md),
[stock-alpha feature explainer](stock_alpha_feature_explainer.md),
[model and gate explainer](model_and_gate_explainer.md), and
[research gates and trade conditions](research_gates_and_trade_conditions.md).

The development profile generally uses `cache/ml/development` and
`reports/ml/development`. The benchmark profile generally uses
`cache/ml/benchmark` and `reports/ml/benchmark`.

## Output Types

| Output type | Writer/command | Contains | Inspect when | Warning signs |
|---|---|---|---|---|
| Stock-level prediction artifacts | `stock_level_prediction_artifacts.py`, `ml-return-mechanics-audit`, overnight stock-alpha | Base stock-level rows, target columns, audit metadata | Before alpha features or if base artifact is missing | Written outside canonical run dir, missing target columns, stale relative to source data |
| Enriched prediction artifacts | `stock_level_alpha_features.py` | Base artifact plus engineered alpha features | Before model ranking | Missing `rebalance_date` or `symbol`, unexpected row loss |
| Alpha feature audit | `stock_level_alpha_features.py` | Feature generation audit, metadata, source artifact path | After feature stage | Missing `features`, wrong source artifact, wrong output root |
| Model ranking benchmark | `stock_level_model_ranking_benchmark.py`, `ml-stock-level-alpha-benchmark` | Leaderboard, metrics, model comparison, guardrails | After benchmark stage | No OOS dates, empty leaderboard, guardrail mismatch |
| OOS predictions | `stock_level_model_ranking_benchmark.py` | Out-of-sample model predictions by date/symbol | Before portfolio replay/sweep | Missing baseline signal, missing predicted columns, wrong date/symbol shape |
| Target comparison | `stock_level_target_comparison.py` | Target availability and per-target benchmark summaries | When targets are skipped or changed | Missing target column, zero eligible dates/symbols, skipped targets not explained |
| Portfolio replay summary | `stock_level_portfolio_replay.py` | Replay summary, winners, net return, drawdown, turnover, cost drag | Before interpreting portfolio quality | Missing winners, zero date count, infeasible policies |
| Portfolio replay equity curves | `stock_level_portfolio_replay.py` | Equity by rebalance date and policy/signal | When returns look suspicious | Missing dates, equity discontinuity, empty curve |
| Portfolio replay holdings | `stock_level_portfolio_replay.py` | Holdings and weights by rebalance date | When concentration or sizing looks suspicious | Weights outside policy bounds, missing symbols |
| Portfolio policy sweep | `stock_level_portfolio_policy_sweep.py` | Policy grid summaries and winners | When selecting policy settings | Baseline unavailable, too many infeasible configs, missing winners |
| Policy sweep equity curves | `stock_level_portfolio_policy_sweep.py` | Equity curves for swept policies | When sweep winners need verification | Empty curves, date mismatch |
| Policy sweep top holdings | `stock_level_portfolio_policy_sweep.py` | Top holdings by policy/date | When checking concentration | Extreme concentration or missing holdings |
| Experiment report | `stock_alpha_experiment_report.py`, `ml-stock-alpha-experiment-report` | Validation summary, registry row, artifact paths, guardrails | Before trusting a stage bundle | Output-root errors, stale mixed outputs, guardrail errors |
| Dev smoke report | `stock_alpha_dev_smoke.py`, `ml-stock-alpha-dev-smoke` | Small end-to-end stock-alpha status, caps, target skips, validation | Before benchmark/full | Status not completed, validation errors, baseline missing |
| Parallelism audit | `stock_alpha_parallelism_audit.py`, `ml-stock-alpha-parallelism-audit` | Requested/effective workers, nested caps, warnings | Before benchmark/full | Effective workers not 4 for benchmark outer stages, nested caps above 1 |
| Overnight summary | `overnight_stock_alpha_runner.py`, `ml-overnight-stock-alpha` | Stage timings, artifacts, comparisons, winners, portfolio summaries, guardrails | After overnight run | Missing stages, legacy paths, `promotion_thresholds_changed: true` |
| Feature attribution | `stock_level_feature_attribution.py`, `ml-stock-level-feature-attribution` | Feature attribution CSV/JSON/Markdown | When explaining model drivers | Attribution enabled accidentally for quick smoke, stale benchmark input |
| Artifact validation reports | `ml-validate-artifacts`, `core/research/ml/artifacts/` | Schema/path validation results | When artifact columns or paths change | Schema errors, missing required files |
| Return mechanics audit | `ml-return-mechanics-audit`, `core/research/ml/audits/return_mechanics*.py` | Return calculation diagnostics | When returns look inconsistent | Data adjustment issues, benchmark mismatch |
| Adjusted data audits | `core/research/ml/audits/adjusted_data*.py` | Adjusted-price and clean replay diagnostics | When data quality is suspect | Alignment errors, stale adjusted data |
| Benchmark-relative validation | `benchmark_relative_validation.py` | Benchmark comparison and promotion readiness evidence | Before promoting research conclusions | Failed readiness, weak benchmark comparison |
| Historical coverage audit | `historical_coverage_audit.py` | History coverage by symbol/universe | Before large ML runs | Insufficient years or latest gap |
| Independent period audits | `independent_period_expansion_audit.py` and related modules | Independent-period validation | When expanding validation windows | Period leakage, insufficient samples |
| Model contract audit | `model_contract_audit.py`, `ml-model-contract-audit` | Checks model interface/contract expectations | When changing model APIs | Contract errors |
| Data inventory | `ml-data-inventory`, `data_inventory.py` | Available data inventory | Before universe expansion or data refresh | Missing symbols, stale data |
| Universe outputs | `ml-build-universes`, `universe_builder.py` | Universe YAMLs under `data/reference/universes/` | When universe definitions change | Too few symbols, insufficient liquidity/history |
| Allocation reports | `core/research/ml/allocation/` | Allocation comparisons, optimizer results, exposure outputs | When changing allocation logic | Invalid exposure path, unstable selected policy |
| Meta-ensemble reports | `ml-meta-ensemble`, `core/research/ml/meta/` | Meta dataset, ensemble comparison, auxiliary metrics | When changing meta model/research configs | Missing source predictions, degraded leaderboard |
| Trading research leaderboard | `trading_research_leaderboard.py` | Consolidated research leaderboard | After ML/audit refresh | Missing input reports, stale result rows |
| Logs | command wrappers/run scripts | Stage progress and tracebacks | During long runs | Exceptions, unexpected legacy paths, stalled stage |
| Cache files | `cache/ml/*` | Intermediate datasets and cached computations | When rerunning profiles | Cache from wrong profile, stale expanded dataset |

## Stock-Alpha Canonical Output Map

For a benchmark overnight run, expected paths are under:

```text
reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/benchmark/
```

Typical files and subdirectories:

```text
stock_level_prediction_artifacts.csv
stock_level_prediction_artifacts.json
stock_level_prediction_artifacts.md
stock_level_prediction_artifacts_enriched.csv
stock_level_alpha_feature_audit.csv
stock_level_alpha_feature_audit.json
stock_level_alpha_feature_audit.md
baseline/
  stock_level_model_ranking_benchmark.json
  stock_level_model_oos_predictions.csv
enriched/
  stock_level_model_ranking_benchmark.json
  stock_level_model_oos_predictions.csv
target_comparison/
  stock_level_target_comparison.json
portfolio_replay/
  stock_level_portfolio_replay_summary.json
  stock_level_portfolio_replay_equity_curves.csv
  stock_level_portfolio_replay_holdings.csv
portfolio_policy_sweep/
  stock_level_portfolio_policy_sweep.json
  stock_level_portfolio_policy_sweep_equity_curves.csv
  stock_level_portfolio_policy_sweep_top_holdings.csv
stock_alpha_experiment_report.json
stock_alpha_experiment_report.md
stock_alpha_parallelism_audit.json
stock_alpha_parallelism_audit.md
overnight_stock_alpha_summary.json
overnight_stock_alpha_summary.md
```

Some stages also write CSV and Markdown siblings for their JSON summaries.

## Generated vs Source

Usually source:

- `application/`
- `config/`
- `configs/`
- `core/`
- `infrastructure/`
- `strategies/`
- `tests/`
- `docs/`
- `README.md`
- `PROJECT_TREE.txt`

Usually generated:

- `cache/`
- `reports/`
- `logs/`
- `data/processed/`
- model binary files such as `.pt` or `.joblib` under reports
- run-specific CSV/JSON/Markdown outputs

Mixed or review carefully:

- `data/reference/` can contain curated reference data and generated adjusted
  price or universe files.
- `configs/` is source config, not generated output.

## Guardrail Fields

Research reports should preserve:

```text
research_only: true
trading_impact: none
production_validated: false
promotion_thresholds_changed: false
```

Warning signs:

- `promotion_thresholds_changed: true`
- `production_validated: true` in a research-only report
- missing `research_only`
- output paths outside the configured root
- legacy paths detected when legacy paths are not allowed
