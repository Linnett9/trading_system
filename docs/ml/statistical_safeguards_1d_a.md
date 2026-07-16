# Ticket 1D-A — Dependency-Aware Statistical Safeguards Foundation

Status: `IMPLEMENTED_SYNTHETIC_ONLY_INTEGRATION_DEFERRED`.

## Contracts

The implementation owner is `core/research/ml/statistical_safeguards.py`.

- Matched input contract: `matched_statistical_series_v1`
- Common result contract: `statistical_safeguard_result_v1`
- Explicit block policy: `explicit_circular_block_length_v1`

Logical result identity includes method/version, validity state, population checksum,
parameter checksum, deterministic seed, orientation, overlap horizon, and metrics.
Creation timestamps and Python runtime metadata do not affect logical identity.

## Methods

### Circular block bootstrap

Paired candidate and benchmark differences are resampled with circular blocks and
wraparound. Alignment and local serial ordering are preserved within each block.
The implementation reports the observed mean advantage, percentile confidence
interval, bootstrap standard error, and empirical one- and two-sided p-values.

Block length is explicit. It is never optimized from candidate outcomes. A block
shorter than the declared overlap horizon produces a warning.

### Deflated Sharpe Ratio

The implementation follows the Bailey–López de Prado finite-sample Sharpe variance
and expected-maximum-Sharpe formulation:

`Var(SR) = [1 - skew*SR + ((kurtosis - 1)/4)*SR^2] / (n - 1)`

The expected maximum under multiple trials uses the Euler-Mascheroni approximation
to the expected Gaussian maximum. The reported probability is the normal CDF of
the observed Sharpe minus that search-adjusted threshold, scaled by Sharpe variance.

The input Sharpe is assumed already annualized when that convention is selected.
Kurtosis is raw kurtosis, so Gaussian kurtosis is 3. Effective search count is
mandatory and is never inferred from successful runs or leaderboard length.

### Probability of Backtest Overfitting

The PBO owner implements bounded combinatorially symmetric cross-validation:

- half of the observations form each training partition;
- the in-sample winner is selected by mean candidate advantage;
- its corresponding out-of-sample rank is converted to a relative rank and logit;
- PBO is the fraction of selected winners with non-positive OOS rank logit.

When all combinations exceed the configured budget, a fixed-seed deterministic
selection of combination indexes is used. The full combination count and bounded
selection status remain visible.

### Superior Predictive Ability

The SPA owner is a studentized Hansen-style SPA foundation:

- candidate-minus-benchmark advantage is orientation normalized;
- candidate statistics are studentized;
- data-dependent consistent recentering is applied;
- circular blocks generate the composite-null maximum statistic;
- the p-value compares the bootstrapped maximum with the observed maximum.

Constant candidates are excluded with warnings. If all candidates are constant,
the result is `INSUFFICIENT_DATA`.

### Model Confidence Set

The MCS owner implements the recognized range-based elimination variant. At each
round it:

1. computes model mean losses;
2. bootstraps the centered range statistic with circular blocks;
3. stops when equal predictive ability cannot be rejected;
4. otherwise eliminates the worst mean-loss model.

Ties are deterministic by model ID. Identical candidates are retained.

### Seed dispersion

Repeated attempts of the same `(model_id, seed)` are collapsed to the highest
attempt number. They are not counted as independent seeds.

Per-model output includes count, mean, median, sample standard deviation, minimum,
maximum, range, interquartile range, best/worst seed, and coefficient of variation.
When multiple models share seeds, mean pairwise Spearman rank stability is reported.

## Status and failure semantics

All methods use:

- `VALID`
- `INSUFFICIENT_DATA`
- `INVALID_INPUT`
- `UNMATCHED_POPULATION`
- `UNSUPPORTED_CONFIGURATION`
- `NUMERICAL_FAILURE`

There is no silent row dropping. Non-finite values, duplicate observation identities,
unequal populations, invalid block sizes, missing search counts, and inadequate
candidate/observation counts fail closed.

## Assumptions and limitations

- Overlapping ten-session outcomes are serially dependent, so ordinary iid standard
  errors and iid bootstrap inference are not valid.
- Block length must be chosen before inspecting candidate outcomes to avoid adding
  another unrecorded search dimension.
- Effective search count must include material failed and rejected trials because
  those trials contributed to the selection process.
- DSR and PBO quantify selection risk; they do not repair temporal leakage or
  replace strict-OOS artifact lineage.
- Statistical significance is not automatic portfolio-promotion evidence. Costs,
  capacity, lineage, stability, and governance gates still apply.
- A final holdout must not be repeatedly reused for selection. Repeated access
  converts it into development data and must increase recorded search exposure.
- No optimal block-length estimator is implemented in this ticket.
- No live selector evaluation or historical inference was performed.

## Verification

Focused command:

```powershell
pytest -q tests/test_statistical_safeguards.py
```

Result: `15 passed in 0.75s`.
