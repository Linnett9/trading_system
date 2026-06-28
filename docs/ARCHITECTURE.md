# Research Architecture

## Purpose

The research subsystem is broker-agnostic, data-provider-agnostic domain and
analysis code. It may read local research artifacts and market-data adapters,
but it must never place orders, mutate paper/live state, or control production
portfolio execution.

This architecture is compatibility-first. Existing CLI commands, import paths,
report filenames, JSON fields, metrics, promotion gates, and research results
remain stable while common mechanics move behind small reusable abstractions.

## Dependency flow

```text
application/cli.py
    -> application/services/*_commands.py
        -> core/research pipelines and compatibility facades
            -> core/research/framework contracts and services
            -> core/research/ml models, features, evaluation, validation
                -> infrastructure/data adapters (only at pipeline boundaries)

core/research and core/research/framework
    -X-> broker, paper, live, order, or production portfolio modules
```

Dependencies point inward toward protocols and pure research services. External
data loading and filesystem output are adapters passed to or constructed at
pipeline boundaries. Strategy and production execution code do not depend on
research implementation details.

## Major modules

### `core/research/framework/contracts.py`

Defines structural protocols for extension points:

- `FeatureGenerator`
- `TargetGenerator`
- `WalkForwardSplitter`
- `PredictionWriter`
- `RankingEvaluator`
- `ReplayEngine`
- `ValidationGate`
- `BenchmarkRunner`
- `ReportWriter`

Protocols keep pipeline orchestration dependent on capabilities rather than
concrete implementations.

### `core/research/framework/config.py`

Provides typed, validated configuration access. `StockLevelResearchConfig`
centralizes the stock-level artifact paths, split settings, parallelism,
sequence settings, and attribution settings that were previously parsed
independently in several modules. Existing YAML keys remain supported.

### `core/research/framework/registry.py`

Provides ordered registries with duplicate protection and component metadata:

- `FeatureRegistry`
- `ModelRegistry`
- `BenchmarkRegistry`
- `ValidationRegistry`
- `ReportRegistry`

The stock-level model and alpha-feature registries are the first concrete
consumers. Registry ordering is deterministic and therefore safe for report and
leaderboard output.

### `core/research/framework/walk_forward.py`

`ExpandingWindowSplitter` is the canonical date-grouped expanding-window split
service. It owns train, embargo, and test date boundaries and returns immutable
`WalkForwardFold` values. Model fitting remains inside each model task; dates and
folds are not parallelized.

### `core/research/framework/ranking.py`

`CrossSectionalRankingEvaluator` owns common stock-selection metrics:

- Pearson IC
- Spearman rank IC
- top- and bottom-decile returns
- top-minus-bottom spread
- top-decile hit rate
- risk-adjusted spread
- annualized spread Sharpe

The alpha benchmark and feature-attribution pipeline use this evaluator, which
prevents metric drift between reports.

### `core/research/framework/parallel.py`

`ParallelTaskExecutor` runs independent tasks, preserves input/registry ordering,
and isolates task errors. The stock-level benchmark supplies one complete
walk-forward model task per worker. Per-model fold chronology remains unchanged.

### `core/research/framework/reporting.py`

`ResearchArtifactWriter` standardizes UTF-8 JSON, CSV, Markdown, and text output.
Pipelines continue to supply explicit field ordering and Markdown rendering, so
existing schemas and filenames are preserved.

### `core/research/framework/data.py`

Contains small filesystem repositories for CSV rows and JSON documents. These
separate artifact loading from feature, model, and evaluation logic.

### `core/research/framework/logging.py`

`ResearchStageLogger` emits structured start/completion events and elapsed time
for loading, feature generation, training/evaluation, and report generation.
It uses standard Python logging and does not force a logging backend.

## Stock-level Phase 2 pipelines

### Alpha feature generation

`core/research/ml/stock_level_alpha_features.py`:

1. loads the immutable base stock-level artifact;
2. loads point-in-time OHLC history;
3. generates registered momentum, drawdown, volatility, and relative factors;
4. writes a sibling enriched artifact;
5. writes availability/missingness audits.

The base CSV is never overwritten. Features use observations strictly before
each rebalance date.

### Alpha model benchmark

`core/research/ml/stock_level_model_ranking_benchmark.py`:

1. loads typed configuration and the selected artifact;
2. selects base plus available engineered features;
3. creates canonical expanding-window folds;
4. runs independent registered models through shared parallel execution;
5. evaluates all predictions through the common ranking evaluator;
6. writes the existing CSV/JSON/Markdown and OOS prediction artifacts.

### Feature attribution

`core/research/ml/stock_level_feature_attribution.py` reuses typed paths, the
canonical splitter, and the common ranking evaluator for coefficients, tree
importance, within-date permutation importance, and leave-one-feature-out
ablation.

## Backward compatibility

- Existing module paths remain valid.
- Existing CLI mode names remain valid.
- Compatibility helper functions delegate to framework services where needed.
- Existing report names and field ordering are unchanged.
- Existing configuration keys are accepted by typed config adapters.
- No promotion threshold or gate is owned by the framework layer.
- No production trading component imports the framework.

## Extension recipes

### Add a new alpha feature

1. Implement the point-in-time calculation in the alpha feature pipeline or a
   cohesive feature-family module.
2. Register its stable name and definition in `alpha_feature_registry()`.
3. Add the name to the enriched artifact schema.
4. Add future-data invariance, availability, and numerical unit tests.
5. Enable engineered features in research configuration.

The feature must return explicit missingness when its source data is unavailable;
it must not silently fabricate sector, industry, news, or fundamental inputs.

### Add a new model

1. Implement the regressor behind a zero-argument factory.
2. Register it in `stock_ranker_model_registry()` with family metadata.
3. Ensure the factory and model task are process-pickleable if model-level
   parallelism is enabled.
4. Add sequential/parallel equivalence and failure-isolation tests.

The model receives common folds and must emit OOS predictions only.

### Add a new target

1. Implement a `TargetGenerator`-compatible component.
2. Define label availability and maturity dates explicitly.
3. Add the target to a typed benchmark configuration rather than reading an
   ad-hoc key inside model code.
4. Verify embargo/purge requirements with chronological tests.

### Add a new benchmark

1. Implement a `BenchmarkRunner`-compatible orchestrator.
2. Compose repositories, a splitter, a registry, evaluators, and report writers.
3. Register the benchmark in a `BenchmarkRegistry` at the application boundary.
4. Keep model fitting, evaluation, and report rendering as separate stages.
5. Add a CLI command that delegates to the application service only.

### Add a new report

1. Keep report-specific schema/rendering in a cohesive renderer.
2. Use `ResearchArtifactWriter` for filesystem output.
3. Register reusable renderers in `ReportRegistry` where multiple pipelines need
   them.
4. Preserve explicit CSV field order and version JSON schemas deliberately.

### Add a new validation gate

1. Implement `ValidationGate` as a pure evaluator over a report payload.
2. Register it in `ValidationRegistry`.
3. Keep thresholds in typed configuration.
4. Never allow reporting or model code to mutate promotion thresholds.

## Intentionally deferred migrations

The repository contains more than one hundred research modules accumulated over
several generations. Moving all of them in one change would create large import
and output risk. The following remain behind their current stable facades and
should migrate incrementally when next modified:

- the large `meta_ensemble.py` orchestration;
- legacy classification `experiment_runner.py` report writes;
- canonical replay and adjusted-data audit renderers;
- dual-momentum and multi-strategy report persistence;
- older classification walk-forward implementations whose label/purge semantics
  differ from the stock-level expanding-window splitter;
- application-level ML batch process orchestration.

These are deliberate compatibility boundaries, not recommended patterns for new
code. New Phase 2 work should use `core/research/framework` abstractions.
