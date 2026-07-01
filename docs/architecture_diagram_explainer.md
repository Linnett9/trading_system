# Architecture Diagram Explainer

This document explains the SVG diagrams in `docs/assets/architecture/`. It is
intended as a reading companion to [architecture_diagrams.md](architecture_diagrams.md).
The diagrams and explanations are documentation only. They do not claim
benchmark results, production validation, or permission to trade.

Classifications used below:

| Classification | Meaning |
| --- | --- |
| Source code | Hand-authored repository code or configuration. |
| Generated output | Cache, report, log, processed data, or run artifact. |
| Research-only | Produces research evidence, metrics, predictions, or simulations only. |
| Execution-adjacent | Touches paper/live state, broker abstractions, orders, fills, or broker adapters. |

## `system_architecture_overview.svg`

Purpose: one-page view of the repository architecture, dependency direction,
generated outputs, validation surfaces, and the blocked path from research
reports to paper/live execution.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| CLI entrypoints | `main.py` and `application/cli.py` receive user mode/profile flags and dispatch to application services. This keeps command parsing separate from domain logic. | CLI modes can route to the wrong service or grow hidden business logic. | `main.py`, `application/cli.py` | Source code |
| Application orchestration | `application/services/` coordinates workflows and `application/reporting/` formats user-facing summaries. Inputs are config objects, domain services, and adapters; outputs are stage calls and report paths. | Service code can become a dumping ground for model, risk, or broker rules. | `application/services/`, `application/reporting/` | Source code; some modules are execution-adjacent |
| Paper services | Paper/shadow-trading service modules sit near orchestration but are not research outputs. They receive approved paper workflows and update local paper state. | Research code can accidentally import or drive paper execution paths. | `application/services/paper_*.py`, `core/paper/` | Execution-adjacent |
| Configuration | `config/` loads defaults and validation, while `configs/` contains named YAML profiles. Development and benchmark profiles set cache/report roots and worker settings. | Runtime config can point reports at stale roots or silently alter guardrails. | `config/config_loader.py`, `config/config_defaults_ml.py`, `configs/research/profiles/` | Source code/config |
| Domain core | `core/entities/`, `core/interfaces/`, `core/engine/`, `core/risk/`, `core/services/`, and `strategies/` hold reusable domain behavior. Signals, fills, prices, strategy context, and risk decisions flow through here. | Broker-specific behavior can leak into strategies or portfolio state can be updated outside fills/prices. | `core/entities/`, `core/interfaces/`, `core/engine/`, `core/risk/`, `core/services/`, `strategies/` | Source code |
| Research-only workflows | `core/research/` and `core/research/ml/` produce experiments, metrics, model outputs, and simulated portfolio evidence. | Research outputs can be misread as execution approval, or stale artifacts can be mixed across roots. | `core/research/`, `core/research/ml/`, `reports/ml/` | Research-only source and generated output |
| Infrastructure adapters | `infrastructure/` adapts external data, broker APIs, persistence, and alerts. Data and broker responses flow in/out; strategy logic should not. | Adapter code can acquire business rules or be called from research-only modules. | `infrastructure/` | Source code; broker modules are execution-adjacent |
| Generated outputs | `cache/`, `reports/`, `logs/`, and `data/processed/` store intermediate data, reports, logs, and processed parquet. | Outputs can be stale, from the wrong profile, or reused as if they were source. | `cache/`, `reports/`, `logs/`, `data/processed/` | Generated output |
| Execution-adjacent surfaces | Paper state, orders, fills, and broker-facing code represent execution concerns. | Research evidence can be wired into orders without manual review and promotion gates. | `core/entities/order.py`, `core/entities/fill.py`, `core/execution/`, `core/paper/`, `infrastructure/broker/`, `infrastructure/brokers/` | Execution-adjacent |
| Tests | `tests/` validates behavior, guardrails, and architecture boundaries using focused fixtures and fakes. | Missing tests can allow path regressions, guardrail drift, or accidental execution imports. | `tests/` | Source code tests |
| Blocked research path | The red blocked arrow means research reports cannot directly authorize paper/live trading. | Treating `production_validated: false` output as a trade instruction is a critical process failure. | `docs/research_gates_and_trade_conditions.md`, report JSON guardrail fields | Research-only boundary |

## `stock_alpha_architecture_overview.svg`

Purpose: a single stock-alpha view from inputs through artifacts, model
evaluation, simulated portfolio research, validation reports, guardrails, and
the blocked broker-order path.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| Inputs | Universe YAMLs, Stooq parquet, expanded rebalance rows, meta auxiliary predictions, sector reference, SPY market symbol, and profile config feed the stock-alpha stages. | Missing symbols, stale parquet, absent `meta_auxiliary_predictions.csv`, or mismatched profile roots. | `data/reference/universes/`, `data/processed/stooq_parquet/`, `cache/ml/*/expanded_rebalance_dataset.csv`, `configs/research/profiles/` | Source config and generated data |
| Data and artifacts | The stock-level artifact and enriched artifact convert inputs into one row per `rebalance_date` and `symbol`, plus engineered features. | Base artifact can be missing, written outside the canonical root, or lose rows during enrichment. | `core/research/ml/stock_level/stock_level_prediction_artifacts.py`, `stock_level_alpha_features.py` | Research-only source and generated output |
| Feature audit | Records engineered feature definitions, availability, missing counts, and parallelism metadata. | Feature availability can collapse because history is too short or metadata is missing. | `stock_level_alpha_feature_audit.{csv,json,md}` | Generated output |
| Model and evaluation | Baseline and enriched ranking benchmarks train chronological walk-forward models and write OOS predictions. | Random splits, empty OOS windows, missing targets, or unavailable sequence/news models. | `stock_level_model_ranking_benchmark.py`, `stock_level_model_oos_predictions.csv` | Research-only |
| Target comparison | Re-runs benchmark summaries across target definitions and records skipped targets. | A target can be all-null, absent, too sparse, or misleading for the research question. | `stock_level_target_comparison.{csv,json,md}` | Research-only generated output |
| Portfolio research | Replay and policy sweep consume OOS predictions to simulate long/short or long-only policies after costs. | Infeasible policies, extreme turnover, missing baselines, or interpreting simulation as trading. | `stock_level_portfolio_replay.py`, `stock_level_portfolio_policy_sweep.py` | Research-only simulation |
| Validation and reporting | Experiment report, parallelism audit, and overnight summary consolidate paths, metrics, worker settings, and guardrails. | Mixed output roots, stale files, wrong run size, missing stages, or changed guardrails. | `stock_alpha_experiment_report.py`, `stock_alpha_parallelism_audit.py`, `overnight_stock_alpha_runner.py` | Research-only |
| Guardrails | Required metadata says the artifacts are research-only, have no trading impact, are not production validated, and did not change promotion thresholds. | Any missing or changed guardrail field should stop interpretation. | Report JSON fields `research_only`, `trading_impact`, `production_validated`, `promotion_thresholds_changed` | Research-only |
| Execution boundary | Research outputs cannot become broker orders. | A model winner or green triage can be treated as approval to trade. | `docs/research_gates_and_trade_conditions.md` | Boundary between research-only and execution-adjacent |
| Canonical output root | `stock_alpha/{dev,benchmark,full}/` keeps run sizes separated under `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/`. | Legacy output paths or copied files can hide propagation bugs. | `core/research/ml/stock_level/stock_alpha_paths.py`, experiment report validation | Generated output location |

## `repository_layers.svg`

Purpose: show practical dependency direction from entrypoint to application,
config, core, research, infrastructure, outputs, and tests.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| `main.py` | Thin top-level process entrypoint. It receives CLI invocation and delegates. | Hidden workflow logic can accumulate here. | `main.py` | Source code |
| `application/cli.py` | Parses modes such as stock-alpha, research, backtest, and paper commands. | Mode branches can route to the wrong command or couple unrelated workflows. | `application/cli.py` | Source code |
| `application/services/` | Orchestrates command-specific flows. | Research services can accidentally depend on paper/live modules. | `application/services/` | Source code; paper services are execution-adjacent |
| `application/reporting/` | Formats application-level reports. | Reporting can start controlling domain decisions. | `application/reporting/` | Source code |
| `config/` | Loads defaults and validates settings. | Defaults can relax guardrails or point to wrong roots. | `config/config_defaults_ml.py`, `config/config_validation.py` | Source code/config |
| `configs/` | Stores profile and experiment YAMLs. | Profile drift can make dev/benchmark output paths ambiguous. | `configs/research/profiles/` | Source config |
| `core/` | Domain entities, interfaces, engines, services, and research modules. | Domain code can become provider-specific. | `core/` | Source code |
| `core/research/` | Classic and ML research area. | Research code can mix generated output assumptions with source behavior. | `core/research/` | Research-only source |
| `research/framework/` | Shared config, data IO, registry, reporting, logging, ranking utilities. | Shared helpers can become too coupled to one experiment. | `core/research/framework/` | Research-only source |
| `core/research/ml/` | ML datasets, audits, models, artifacts, allocation, replay, and stock-alpha research. | Model outputs can be interpreted beyond their validation scope. | `core/research/ml/` | Research-only source |
| `ml/stock_level/` | Stock-alpha stages and reports. | Output-root propagation, missing targets, and stale artifacts. | `core/research/ml/stock_level/` | Research-only source |
| `infrastructure/` | Data and broker adapters. | Adapter rules can leak into strategy logic. | `infrastructure/` | Source code; broker adapters execution-adjacent |
| `strategies/` | Strategy implementations produce signals. | Strategies can emit orders or mutate portfolio state. | `strategies/` | Source code |
| `cache/`, `reports/`, `logs/` | Runtime/generated artifacts. | Stale or wrong-profile outputs. | `cache/`, `reports/`, `logs/` | Generated output |
| `tests/` | Focused checks and guardrail tests. | Missing coverage around changed modules. | `tests/` | Source code tests |

## `research_execution_boundary.svg`

Purpose: make the research-only/execution-adjacent boundary visible.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| `core/research/` | Research framework and experiments produce evidence and diagnostics. | Importing broker/paper/live code from research-only modules. | `core/research/` | Research-only source |
| `core/research/ml/` | ML-specific research stages and audits. | Report outputs can be mistaken for production signals. | `core/research/ml/` | Research-only source |
| `core/research/ml/stock_level/` | Stock-alpha feature, ranking, target, portfolio simulation, report, and overnight stages. | Canonical output roots can diverge between stages. | `core/research/ml/stock_level/` | Research-only source |
| Metrics and validation reports | JSON/CSV/Markdown outputs summarize correctness and quality. | Stale, incomplete, or mixed-root reports can look valid. | `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/` | Generated output |
| Portfolio replay | Simulates policy returns over OOS predictions. | Replay can be confused with paper trading. | `stock_level_portfolio_replay.py`, `stock_level_portfolio_replay_summary.json` | Research-only simulation |
| Portfolio policy sweep | Simulates policy grid alternatives. | Winning policy can be overfit to one run. | `stock_level_portfolio_policy_sweep.py`, `stock_level_portfolio_policy_sweep.json` | Research-only simulation |
| Boundary | The conceptual stop sign between evidence and execution. | A green candidate bypasses manual review or production validation. | `docs/research_gates_and_trade_conditions.md` | Boundary |
| Paper services | Local paper/shadow-trading orchestration. | Research commands call paper services directly. | `application/services/paper_*.py` | Execution-adjacent |
| `core/paper/` | Paper state and local ledger behavior. | Paper state mutates from research artifacts. | `core/paper/` | Execution-adjacent |
| `core/entities/order.py` | Order domain object. | Strategies or research emit orders directly. | `core/entities/order.py` | Execution-adjacent domain |
| `core/execution/` | Execution models and fill simulation. | Real broker behavior and simulation assumptions get blurred. | `core/execution/` | Execution-adjacent |
| `infrastructure/broker/`, `infrastructure/brokers/` | Broker adapters. | Broker adapters contain strategy logic or are invoked by research. | `infrastructure/broker/`, `infrastructure/brokers/` | Execution-adjacent |

## `stock_alpha_pipeline.svg`

Purpose: show the stock-alpha overnight stage order and the files each stage
depends on.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| Inputs | Universe, data, config, expanded dataset, and auxiliary predictions feed artifact generation. | Missing parquet, missing universe, missing meta auxiliary predictions. | `data/reference/universes/`, `data/processed/stooq_parquet/`, `cache/ml/*/expanded_rebalance_dataset.csv` | Source config and generated data |
| Stock artifact | Creates `stock_level_prediction_artifacts.{csv,json,md}` with baseline signals and future labels. | Missing labels, non-unique rows, wrong output directory. | `stock_level_prediction_artifacts.py`, artifact JSON audit | Research-only generated output |
| Alpha features | Reads canonical base artifact and writes enriched artifact plus feature audit. | Base artifact path points to legacy `reports/ml/benchmark/ml/`; history too short. | `stock_level_alpha_features.py`, `stock_level_alpha_feature_audit.json` | Research-only generated output |
| Baseline ranking | Runs model ranking with engineered features disabled. | Baseline unavailable, no OOS dates, insufficient training dates. | `baseline/stock_level_model_ranking_benchmark.json` | Research-only |
| Enriched ranking | Runs model ranking with engineered features enabled. | Engineered features mostly missing or overfit. | `enriched/stock_level_model_ranking_benchmark.json` | Research-only |
| OOS predictions | Stores prediction rows by date/symbol for completed models and baselines. | In-sample rows leak into portfolio stages or prediction columns missing. | `stock_level_model_oos_predictions.csv` | Generated output |
| Target comparison | Compares target definitions and records skipped targets. | Target all-null, too few eligible dates/symbols. | `target_comparison/stock_level_target_comparison.json` | Research-only |
| Portfolio replay | Runs fixed simulated portfolio policies over OOS predictions. | Turnover/cost drag makes apparent ranking edge unusable. | `portfolio_replay/stock_level_portfolio_replay_summary.json` | Research-only simulation |
| Policy sweep | Evaluates simulated policy grid. | Best policy is infeasible or too sensitive to constraints. | `portfolio_policy_sweep/stock_level_portfolio_policy_sweep.json` | Research-only simulation |
| Experiment report | Validates output root, freshness, guardrails, winners, and paths. | Detects stale mixed outputs, legacy paths, missing guardrails. | `stock_alpha_experiment_report.json` | Research-only generated output |
| Optional attribution | Explains feature contribution for supported completed model outputs. | Attribution can be stale or overinterpreted as causality. | `stock_level_feature_attribution.py`, attribution reports | Research-only |
| Overnight summary | Final stage rollup for `ml-overnight-stock-alpha`. | Missing stage paths, legacy output paths, wrong run size. | `overnight_stock_alpha_runner.py`, `overnight_stock_alpha_summary.json` | Research-only generated output |

## `data_lineage.svg`

Purpose: show how raw data, reference data, features, labels, artifacts, OOS
predictions, and reports relate.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| Raw Stooq daily files | External daily price/volume input. | Missing symbols or stale raw data. | `data/raw/stooq_bulk/data/daily/` | Generated/input data |
| Processed Stooq parquet | Local processed price store used by research stages. | Processing gaps, missing high/low/close, stale parquet. | `data/processed/stooq_parquet/` | Generated data |
| Universe YAML | Symbol list for artifact rows. | Too few symbols or wrong universe for run size. | `data/reference/universes/*.yaml` | Source/reference data |
| Sector reference | Sector mapping for relative features. | Missing sector creates blank relative features. | `core/research/ml/data/sector_reference.py`, `ml.sector_by_symbol` | Source/reference data |
| Market symbol | SPY or configured market proxy for residual/context features. | Market symbol unavailable or included incorrectly. | `stock_ranker_market_symbol`, `stock_ranker_spy_symbol` | Config/source data |
| Expanded rebalance dataset | Intermediate rebalance rows and context columns. | Cache from wrong profile or stale generation. | `cache/ml/{development,benchmark}/expanded_rebalance_dataset.csv` | Generated output |
| Point-in-time features | Trailing features available before `rebalance_date`. | Lookahead leakage or insufficient history. | `stock_level_prediction_artifacts.py`, `stock_level_alpha_features.py` | Research-only derived data |
| Future labels | Forward returns and future risk labels used only as targets. | Accidentally used as features or missing near end of history. | `ACTUAL_COLUMNS` in `stock_level_prediction_artifacts.py` | Future labels/generated output |
| Stock-level artifact | Base table with symbol/date rows. | Missing `rebalance_date`, `symbol`, features, or labels. | `stock_level_prediction_artifacts.csv` | Research-only generated output |
| Enriched artifact | Base table plus engineered alpha features. | Row loss or blank feature columns. | `stock_level_prediction_artifacts_enriched.csv` | Research-only generated output |
| OOS predictions | Chronological out-of-sample model predictions. | In-sample leakage or empty OOS windows. | `stock_level_model_oos_predictions.csv` | Research-only generated output |
| Reports | JSON/CSV/Markdown summaries and audits. | Guardrails missing, stale artifacts mixed. | `reports/ml/benchmark/regime_transformer_meta_ensemble_v1/stock_alpha/` | Generated output |

## `model_evaluation_loop.svg`

Purpose: show chronological stock-alpha model evaluation and how OOS
predictions feed portfolio simulation.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| Feature rows | Base or enriched predictor matrix. | Incomplete rows, missing features, leakage columns. | `FEATURE_COLUMNS`, `ENGINEERED_FEATURE_COLUMNS`, enriched CSV | Research-only data |
| Target column | The label being predicted, default `actual_forward_return_10d`. | Wrong target or sparse target can distort model selection. | `stock_ranker_target_column`, target comparison report | Future label/generated output |
| Chronological walk-forward split | Expanding-window split by rebalance date. | Random split or insufficient dates. | `walk_forward` in benchmark JSON | Research-only evaluation |
| Train on earlier dates | Fits models only on previous dates. | Training window too short or feature distribution unstable. | `stock_level_benchmark_execution.py` | Research-only |
| Embargo rebalance dates | Skips adjacent dates between train/test. | Overlap leakage if embargo is too small. | `stock_ranker_embargo_dates`, benchmark JSON | Research-only |
| Predict later OOS dates | Scores held-out later dates. | OOS windows empty or predictions missing for some symbols. | `stock_level_model_oos_predictions.csv` | Research-only generated output |
| Ranking metrics | IC, spread, hit rate, spread Sharpe, and related metrics evaluate ranking quality. | Positive one-metric result hides weak spread or unstable hit rate. | `core/research/framework/ranking.py`, benchmark JSON leaderboard | Research-only metrics |
| OOS predictions | Downstream input to replay/sweep. | Portfolio stage uses non-OOS data. | `stock_level_model_oos_predictions.csv` | Research-only generated output |
| Portfolio replay / policy sweep | Simulate ranking outputs under policies and costs. | Simulation is treated as execution approval. | replay and sweep reports | Research-only simulation |
| Baselines | `momentum_120d` and `risk_adjusted_momentum`. | ML comparison has no aligned baseline. | `BASELINE_COLUMNS` in `stock_level_benchmark_types.py` | Research-only |
| Tabular models | Ridge, ElasticNet, RandomForest, GradientBoosting. | Overfit or missing dependencies. | `stock_level_benchmark_models.py` | Research-only source |
| Sequence/context models | DLinear, PatchTST, Transformer variants, context/news/TFT adapters. | Higher runtime, sparse sequences, unavailable news columns. | `stock_level_sequence_regressors.py`, `stock_level_benchmark_models.py` | Research-only source |
| News model note | News model is unavailable unless point-in-time news/sentiment columns exist. | Synthetic or non-point-in-time news would create leakage. | `unavailable_models` in benchmark JSON | Research-only guardrail |

## `gates_and_decisions.svg`

Purpose: show how completed research runs move through correctness, quality,
triage, review, and deeper validation without authorizing trades.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| Completed research run | A configured stage bundle finished and wrote outputs. | Completion is mistaken for quality. | `overnight_stock_alpha_summary.json` | Research-only generated output |
| Correctness gates | Validate canonical root, required files, freshness, run size, guardrails, OOS dates, and feasible winners. | Legacy paths, stale mixed files, missing artifacts, changed guardrails. | `validate_stock_alpha_outputs()` in `stock_alpha_experiment_report.py` | Research-only validation |
| Quality gates | Interpret ranking, target, replay, and sweep metrics. | Good ranking metrics fail after costs or have poor drawdown. | benchmark, target, replay, and sweep JSON reports | Research-only validation |
| Candidate triage | Red/yellow/green research classification. | Green is overread as trade approval. | experiment report and research gate docs | Research-only |
| Red | Reject or fix pipeline. | Root cause ignored and rerun repeated blindly. | report errors and warnings | Research-only |
| Yellow | Promising but incomplete. | Missing evidence hidden by partial success. | skipped targets, unavailable models, warnings | Research-only |
| Green | Worth deeper validation. | Direct jump to paper/live. | `docs/research_gates_and_trade_conditions.md` | Research-only status |
| Manual review | Human inspection of evidence, paths, assumptions, and risks. | Review only checks winners, not data lineage or guardrails. | reports, logs, docs | Process boundary |
| Deeper validation | Larger benchmark/full validation. | Overfitting to benchmark profile. | benchmark/full outputs under canonical root | Research-only |
| Research report | Records evidence and guardrail metadata. | Missing `production_validated: false` or changed thresholds. | `stock_alpha_experiment_report.json` | Research-only generated output |
| Paper/live trading | Execution workflows outside research. | Research output drives orders. | paper/live modules and promotion docs | Execution-adjacent |
| Blocked | No direct authorization path. | A blocked path is bypassed manually. | guardrail fields and review checklist | Boundary |

## `output_report_map.svg`

Purpose: map the canonical stock-alpha run-size output root to the major files
that downstream stages and reviewers inspect.

| Box | What it means, what flows through it, and why it exists | What can go wrong | Inspect | Classification |
| --- | --- | --- | --- | --- |
| Canonical output root | Active output directory for `dev`, `benchmark`, or `full`. | Stages write outside the root or mix legacy files. | `stock_alpha_paths.py`, experiment report validation | Generated output location |
| Artifact files | Base stock-level CSV/JSON/Markdown. | Missing artifact or wrong root breaks alpha features. | `stock_level_prediction_artifacts.*` | Research-only generated output |
| Feature files | Enriched artifact and feature audit. | Feature audit points to wrong source artifact. | `stock_level_prediction_artifacts_enriched.csv`, `stock_level_alpha_feature_audit.*` | Research-only generated output |
| Ranking files | Model benchmark summaries and OOS predictions. | Empty leaderboard, no OOS dates, unavailable models. | `stock_level_model_ranking_benchmark.*`, `stock_level_model_oos_predictions.csv` | Research-only generated output |
| Target comparison files | Per-target availability and model summaries. | Targets skipped silently or compared on too few dates. | `target_comparison/stock_level_target_comparison.*` | Research-only generated output |
| Replay and sweep files | Simulated portfolio summaries, equity curves, holdings, and policy grids. | Winners infeasible, too much turnover, cost drag dominates. | `portfolio_replay/`, `portfolio_policy_sweep/` | Research-only simulation output |
| Experiment report | Final validation wrapper and registry row. | `validation_passed` false, output-root errors, guardrail errors. | `stock_alpha_experiment_report.{json,md}` | Research-only generated output |
| Optional attribution files | Feature attribution diagnostics where configured. | Attribution stale or unsupported for a model. | `stock_level_feature_attribution.*` | Research-only generated output |
| Overnight summary | Stage rollup with output metadata and guardrails. | Missing paths, failed stage, wrong run size. | `overnight_stock_alpha_summary.{json,md}` | Research-only generated output |
