# Ticket 4A-A — Huber Regression Selector Foundation

## Scope and dependency

This is a deterministic, synthetic-only continuous-return selector foundation. It
does not read the authoritative selector dataset, publish a component, run replay,
or alter the active model registry.

The estimator is `sklearn.linear_model.HuberRegressor` from scikit-learn 1.6.1.
It supports positive sample weights and is deterministic for fixed inputs and
parameters because it has no stochastic sampling path.

## Huber loss and regularisation

The estimator jointly fits coefficients, intercept and robust scale. Standardised
residuals with magnitude below `epsilon` receive quadratic loss; larger residuals
receive linear loss. This is the recognised piecewise quadratic/linear Huber
convention.

The bounded default configuration is:

- `epsilon = 1.35`;
- `alpha = 0.0001`;
- intercept fitted;
- tolerance `1e-5`;
- maximum iterations `100`;
- warm start disabled.

`alpha` is the estimator's L2 coefficient regularisation. No broad parameter grid
is performed. A convergence warning or reaching the iteration bound produces
`NON_CONVERGENCE`; there is no heuristic success fallback.

## Input and temporal convention

`huber_selector_input_v1` records stable row and asset identities, decision and
feature-availability timestamps, ordered features, continuous target, target
maturity, sample weight, fold identities, dataset identity and population
checksums.

Rows are ordered by decision timestamp, canonical asset ID and row ID. Features
must use one exact canonical order. Training decisions must be strictly before the
validation boundary, and every training label must mature no later than that
boundary. Features must be available by their row's decision timestamp.
Validation rows never enter preprocessing or fitting.

The target is a continuous forward return or selector score target. Predictions
are continuous scores, not probabilities.

## Preprocessing

`huber_selector_preprocessing_v1` uses training-only standardisation:

\[
x'_{ij}=\frac{x_{ij}-\mu_j^{\rm train}}{s_j^{\rm train}}.
\]

Location is the training mean and scale is the population standard deviation
(`ddof=0`). Constant features are centred and assigned unit scale, and their
identities are reported. Validation observations use the immutable training
parameters.

Winsorisation is disabled by default. If explicitly configured, its quantile
limits and training-derived boundaries become part of the preprocessing and model
checksums. No imputation, feature selection, or full-panel scaling occurs.

## Prediction and ranking

`huber_selector_prediction_v1` records the continuous prediction, model and
population identities, fold, cutoff and label-availability evidence.

Within each decision date, predictions are ordered descending. Rank 1 is the
largest predicted return. Canonical asset ID and then stable row ID break exact
score ties. Percentile rank is `(group_size - rank) / (group_size - 1)`, with the
largest prediction at one and the smallest at zero for groups larger than one.

Degenerate rank diversity fails closed under the caller's explicit minimum
policy.

## Diagnostics and stability

Fit diagnostics report:

- coefficients and intercept;
- fitted robust scale;
- iteration count and convergence;
- training residual and validation prediction summaries;
- the estimator's outlier mask and count;
- coefficient norm and maximum absolute coefficient;
- feature-level coefficient table;
- prediction dispersion and rank diversity;
- training and validation counts.

These are model diagnostics, not validation metrics or promotion evidence.

The coefficient-stability helper requires identical feature order and schema
across fits. It reports coefficient mean, median, standard deviation, sign
consistency, maximum range, pairwise rank correlations, unstable features,
intercept dispersion, robust-scale dispersion and convergence consistency.

## Synthetic control and limitations

The control helper compares Huber with unregularised ordinary least squares on
identical clean and outlier-injected populations. It reports MSE, MAE, Spearman
correlation, coefficient norm, coefficient movement and validation-prediction
movement. The focused fixture demonstrates lower sensitivity by Huber to one
extreme training target.

Huber regression primarily protects against target residual outliers. It does not
solve feature outliers, nonlinear relationships, regime shifts, collinearity,
poor targets or weak cross-sectional rank structure. Outlier robustness does not
guarantee superior stock ranking. Historical promotion requires matched
strict-OOS multi-regime evaluation after transaction costs, statistical
safeguards, complete search accounting and protected final audit.

