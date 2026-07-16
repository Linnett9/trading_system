# Ticket 5B-A — LightGBM Rank-XENDCG Selector Foundation

## Scope

This owner is a synthetic-only stock-ranking foundation. It consumes the frozen
`grouped_ranking_dataset_v1` contract, fits LightGBM 4.6.0 Rank-XENDCG, emits
continuous ranking scores and deterministic within-date ranks, and independently
recomputes the result for verification.

It does not load selector or market data and is not connected to component
publication, replay, evaluation, registries, portfolios or order generation.
Synthetic diagnostics are capability evidence only, never promotion evidence.

## Input and label boundary

`lightgbm_rank_xendcg_input_v1` binds the dataset, feature schema, target,
ranking-label, split, row population, label population, query sizes, training
cutoff and maximum training-label maturity identities.

Rows must already be ordered by decision date, canonical asset ID and row ID.
Each decision date is one contiguous query and has one split role. Training
queries precede validation queries. No row or group is dropped.

The primary registered contracts are:

- `within_date_quintile_relevance_v1`;
- `within_date_decile_relevance_v1`.

Another integer contract must be explicitly supplied as registered. Labels must
be present, integral and nonnegative. Continuous percentile, fractional,
negative and missing labels fail before LightGBM is called. Continuous labels
are never silently converted.

## Fixed model configuration

The versioned default is deliberately small and is not selected from a search:

```text
objective=rank_xendcg
metric=ndcg
eval_at=[1,3,5]
n_estimators=24
learning_rate=0.08
num_leaves=7
max_depth=3
min_child_samples=2
reg_alpha=0.05
reg_lambda=0.20
subsample=1.0
colsample_bytree=1.0
max_bin=31
random_state=1729
deterministic=True
force_col_wise=True
n_jobs=1 or 2
bagging_seed=1729
feature_fraction_seed=1729
data_random_seed=1729
verbosity=-1
```

Early stopping is disabled. A later ticket may expose it only as an explicit
synthetic capability with legal validation groups and identity-bound callbacks.
It must not become hidden tuning.

## Predictions and diagnostics

Raw LightGBM outputs are ranking scores, not probabilities or expected returns.
Within each decision date, higher scores rank first. Canonical asset ID and then
row ID break equal-score ordering deterministically. Percentile rank maps the
first item to one and the last to zero.

Diagnostics cover row/query counts, group-size distribution, label distribution,
score moments and range, tied scores, rank diversity, NDCG at 1/3/5, synthetic
Spearman rank correlation, top-three overlap and group-level NDCG dispersion.

Feature diagnostics report split and gain importance, normalised gain, gain
rank, unused features and concentration. Tree importance is not causal
attribution. Tree counts, depths, leaves, model-text checksum and byte size are
also recorded. Durations and paths are runtime metadata and do not affect
logical identity.

## Determinism and serialisation

Every call fits the identical input twice. Predictions must match at
`rtol=0`, `atol=1e-12`; ranks and feature importances must also match. Logical
booster and byte-level model-text determinism are reported separately.

When a caller supplies a temporary directory, the model is saved there,
checksummed, reloaded and required to reproduce validation predictions at the
same strict tolerance.

## Limitations and deferred challengers

Small synthetic fixtures prove plumbing and behavior, not historical usefulness.
No hyperparameter conclusion may be drawn from them. Future bounded search must
use experiment accounting and research-budget governance.

LambdaRank remains a later challenger because the master funnel establishes
Rank-XENDCG first. XGBoost remains deferred to avoid maintaining multiple ranking
libraries before the first bounded LightGBM path is complete.
