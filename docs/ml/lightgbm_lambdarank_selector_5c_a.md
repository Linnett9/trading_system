# Ticket 5C-A — LightGBM LambdaRank Selector Challenger

## Role and scope

This synthetic-only challenger applies LightGBM 4.6.0 LambdaRank to the same
frozen grouped-ranking contract used by Rank-XENDCG. It isolates the ranking
objective and explicit label-gain table while preserving row, group, split,
feature, target, maturity, seed and non-objective model conventions.

It is not connected to real selector data, publication, evaluation, replay,
portfolios, registries or order generation. Synthetic comparison results do not
select a production winner.

## Input and grouping

`lightgbm_lambdarank_input_v1` adapts the established Rank-XENDCG input
validation without changing it. Rows remain ordered by decision date, canonical
asset ID and row ID. Each date is one contiguous query with one split role.
Training and validation populations, feature order, checksums and training-label
maturity must remain intact.

Supported labels are frozen quintile and decile relevance. Continuous,
fractional, negative, missing and gain-table-exceeding labels fail before
LightGBM fitting.

## Fixed configuration

The configuration matches Ticket 5B-A except for the objective and explicit
label-gain table:

```text
objective=lambdarank
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

No parameter search or early stopping is used.

## Label-gain policy

`lightgbm_lambdarank_label_gain_v1` uses the explicit deterministic convention:

```text
gain(relevance) = 2^relevance - 1
```

Quintile levels `0..4` map to `[0,1,3,7,15]`. Decile levels `0..9` map to
`[0,1,3,7,15,31,63,127,255,511]`. The ordered relevance levels, gains,
maximum relevance and checksum enter input, configuration, model and prediction
identity. A reordered, changed or incompatible gain policy fails closed.

## Predictions and diagnostics

Higher raw LambdaRank scores rank first within each decision date. Canonical
asset ID and row ID deterministically break score ties. Scores are neither
probabilities nor expected returns.

Diagnostics include query and group-size summaries, relevance and gain
distributions, score moments, tie counts, rank diversity, NDCG at 1/3/5,
synthetic Rank IC, top-three overlap, group dispersion, split/gain importance,
unused features, tree depth and leaf distributions, model size, repeatability,
serialization and reload state. Tree importance is not causal attribution.

Fit and prediction durations are runtime metadata and do not alter logical
checksums.

## Matched Rank-XENDCG comparison

The comparison fits both objectives on identical training and validation rows,
features, labels, groups, splits, seeds, thread counts and non-objective
parameters. It compares NDCG, score-rank correlation, top-k overlap, score
dispersion, ties, group dispersion, feature importance, unused features, model
size, timings and repeatability.

Different score-sorted output orders are allowed; population identity is checked
using immutable row and dataset checksums and scores are aligned by row ID.

This comparison is synthetic capability evidence only. A real decision requires
strict-OOS multi-regime histories, ranking metrics, turnover and after-cost
portfolio outcomes.

## Deferred challengers

XGBoost remains deferred to avoid maintaining another ranking dependency before
the bounded LightGBM objective comparison is integrated and justified.
