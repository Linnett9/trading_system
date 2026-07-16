# Ticket 4C-A — Multi-Horizon Linear Selector Foundation

## Scope

This synthetic-only foundation coordinates separate linear selectors for:

- `return_1s`;
- `return_5s`;
- `return_10s`;
- `return_20s`.

It does not infer horizons from columns, read the authoritative selector dataset,
modify registries, publish components, or perform historical evaluation.

## Target and maturity contracts

`multi_horizon_target_contract_v1` records the horizon length, forward-return
definition, unit, benchmark adjustment, start/end rules, maturity rule,
overlapping-outcome state, target type, required observations and checksum.

The exact ordered horizon panel is 1, 5, 10 and 20 sessions. Multi-session
outcomes overlap adjacent daily decisions. Future inference must therefore use
Ticket 1D-A's dependency-aware block bootstrap.

Every input row carries separate target values, maturity timestamps and one of
`MATURE`, `IMMATURE` or `INVALID` for each horizon. A short target may be mature
while a longer target remains immature. Mature non-finite targets fail closed;
immature and invalid states remain visible.

## Horizon-specific populations and preprocessing

The default policy is
`independent_mature_population_per_horizon_v1`.

A training row enters horizon \(h\) only when:

- it is a training row;
- its horizon-\(h\) state is `MATURE`;
- its maturity timestamp is no later than the registered training cutoff;
- its decision timestamp precedes that cutoff.

Each horizon reports eligible, immature and invalid rows and independent
population checksums. Longer horizons may therefore have smaller populations.
Validation rows never enter fitting or preprocessing.

Each horizon independently estimates training means and population standard
deviations. Constant features are centred and assigned unit scale. There is no
full-panel scaling, validation-derived imputation or silent shared-population
assumption.

## Model panel

The fixed panel contains one Ridge and one Elastic Net per available horizon.

Ridge:

- `sklearn.linear_model.Ridge`;
- `alpha=1.0`;
- fitted intercept;
- deterministic `auto` solver convention;
- tolerance `1e-4`.

Elastic Net:

- `sklearn.linear_model.ElasticNet`;
- `alpha=0.001`;
- `l1_ratio=0.25`;
- fitted intercept;
- tolerance `1e-4`;
- maximum 5,000 iterations;
- cyclic coordinate selection.

No grid search is performed. Elastic Net convergence warnings or exhausted
iteration budgets fail closed.

The ordered-logit boundary does not reimplement ordered logit. Its adapter accepts
already-produced class probabilities, expected relevance, model/fold identity and
an exact ordinal target checksum. Expected relevance remains distinguishable from
continuous return predictions.

## Predictions and combined scores

Per-horizon scores rank descending within decision date. Canonical asset ID and
row ID break exact ties.

The horizon ensemble is the fixed arithmetic mean of Ridge and Elastic Net where
both are requested. Combined score weights are immutable:

\[
\text{short}=0.6s_{1}+0.4s_{5},
\]

\[
\text{medium}=0.5s_{5}+0.5s_{10},
\]

\[
\text{long}=0.3s_{10}+0.7s_{20}.
\]

A component is unavailable unless all horizons named by its formula exist. The
implementation does not silently renormalise missing weights.

## Persistence and disagreement

For a row with at least two horizon scores:

- sign agreement is the larger fraction of nonnegative or nonpositive scores;
- rank stability is \(1-\operatorname{std}\) of horizon percentile ranks;
- persistence is `0.5 × sign agreement + 0.5 × max(0, rank stability)`.

Disagreement is:

`0.4 × sign disagreement + 0.4 × percentile-rank range + 0.2 × short/long sign conflict`.

Persistence and disagreement are blocked when fewer than two horizons exist.
Agreement is not proof of correctness; correlated horizon errors can agree.

Missing-horizon states include `ALL_HORIZONS_AVAILABLE`, `SHORT_ONLY`,
`SHORT_AND_MEDIUM`, `LONG_HORIZON_MISSING` and
`INSUFFICIENT_HORIZONS`.

## Diagnostics and temporal implications

Per-horizon diagnostics report population counts, coefficients, intercepts,
norms, Elastic Net sparsity, convergence, prediction moments, target moments and
population identities.

Cross-horizon diagnostics report Ridge coefficient correlations and sign
consistency, prediction and rank correlations, top-three overlap, score-sign
agreement, persistence and disagreement distributions.

The longest available horizon governs the conservative purge horizon. Embargo
must follow a registered policy at least as strict as that longest included
horizon.

Combining forecasts can hide instability, mix different training populations and
amplify shared misspecification. Historical promotion requires matched strict-OOS
multi-regime evaluation after costs, statistical safeguards, complete search
accounting and protected final audit.

