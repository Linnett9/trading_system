# Ticket 4B-A — Contextual Elastic Net Selector Foundation

## Model and scope

The synthetic selector implements:

\[
s_{i,t}=x_{i,t}^{\top}\beta+x_{i,t}^{\top}A m_t,
\]

where \(x_{i,t}\) is the stock-feature vector and \(m_t\) is the common
market-context vector. The matrix \(A\) is represented only by a bounded,
registered interaction list. The implementation does not create the full
stock-feature × context Cartesian product.

This foundation does not read the authoritative selector dataset, modify the
model registry, publish components, or run historical evaluation.

## Estimator

The estimator is `sklearn.linear_model.ElasticNet` from scikit-learn 1.6.1. Its
objective convention is:

\[
\frac{1}{2n}\lVert y-Xw\rVert_2^2
+\alpha\,l_1\_ratio\,\lVert w\rVert_1
+\frac{\alpha(1-l_1\_ratio)}{2}\lVert w\rVert_2^2.
\]

The single bounded default configuration is:

- `alpha = 0.001`;
- `l1_ratio = 0.25`;
- intercept fitted;
- tolerance `1e-4`;
- maximum iterations `5000`;
- cyclic coordinate selection;
- warm start disabled;
- random seed `0` recorded for identity.

The installed estimator accepts sample weights. Cyclic coordinate descent is
deterministic for fixed ordered inputs. A convergence warning or exhausted
iteration budget produces `NON_CONVERGENCE`.

## Registered interaction list

`contextual_interaction_contract_v1` initially contains:

1. momentum × market volatility;
2. momentum × market trend;
3. drawdown recovery × market drawdown;
4. risk-adjusted momentum × market volatility;
5. liquidity × market volatility;
6. stock volatility × market volatility.

Each entry records canonical stock and context IDs, scaled-product
transformation, point-in-time availability rule, output unit, order and checksum.
Unknown inputs, duplicates, ambiguous identities and full all-pairs expansion are
rejected.

## Context and temporal integrity

Every row declares stock features and market context separately. There must be
exactly one context vector per decision date across all stocks. Context may vary
between dates. Features and context must be available by the decision timestamp.

Training decisions must precede validation, and all training targets must mature
by the validation boundary. Validation observations never enter preprocessing or
fitting.

## Preprocessing and design matrix

The versioned order is:

1. optionally winsorise stock and context bases using training-derived bounds;
2. standardise stock and context bases separately using training means and
   population standard deviations;
3. centre constant columns and assign unit scale;
4. construct registered interactions as products of scaled base variables;
5. concatenate stock main effects, optional context main effects, and interactions.

The default omits context main effects. Common context fields are identical across
stocks on a date and therefore do not directly differentiate stocks within that
date. Their interactions with stock features can change cross-sectional
sensitivity and therefore rankings.

No complete-design rescaling, imputation, automatic feature selection or
validation-derived transformation occurs. Column order and lineage are explicit.

## Predictions and diagnostics

Predictions are continuous selector scores, never probabilities. Within each date,
scores rank descending; canonical asset ID and row ID provide deterministic tie
breaking.

Diagnostics separate:

- stock main effects;
- optional context main effects;
- interaction effects.

For every coefficient they report magnitude, sign, nonzero state and lineage.
Aggregates include sparsity by column type, L1/L2 norms, maximum magnitude,
iterations, dual gap, prediction dispersion and rank diversity. Interaction
coefficients are not causal estimates.

The context-sensitivity helper holds one stock vector fixed and changes only the
context. It reports baseline and changed scores, total movement, affected
interaction contributions and unchanged stock-main contribution. This is a
behaviour diagnostic, not historical evidence.

## Stability and comparison

Stability requires identical design columns and interaction checksums. It reports
coefficient means, medians, standard deviations, sign consistency, nonzero and
interaction-selection frequencies, coefficient-rank correlations, unstable
columns, prediction-rank stability and convergence consistency.

The incremental synthetic comparison uses identical populations for conventional
stock-only Elastic Net and Contextual Elastic Net. It reports MSE, MAE, Spearman
correlation, prediction dispersion, sparsity and interaction recovery. One
fixture contains genuine context-dependent sensitivity; a separate no-context
fixture explicitly does not assume improvement.

Contextual interactions increase specification risk and collinearity. Synthetic
interaction recovery does not imply historical superiority. Promotion requires
matched strict-OOS multi-regime evaluation after costs, statistical safeguards,
complete search accounting and protected final audit.

