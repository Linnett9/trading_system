# Ticket 1E-A — Probabilistic Prediction Metrics Foundation

Status: `IMPLEMENTED_SYNTHETIC_ONLY_INTEGRATION_DEFERRED`.

Implementation owner: `core/research/ml/probabilistic_metrics.py`.

## Contracts

- Input: `probabilistic_prediction_input_v1`
- Result: `probabilistic_metric_result_v1`
- Probability tolerance: `1e-8`
- Log-loss clipping: `[1e-15, 1]`
- Calibration policy: `equal_width_top_confidence_10_bins_v1`
- Interval score: `winkler_interval_score_v1`

Population identity is the SHA-256 hash of the ordered observation identities.
Logical result identity includes the metric version, population, configuration,
metric values, weighting, aggregation and target unit. Creation timestamps and
runtime metadata do not affect logical identity.

## Classification and ordinal metrics

- Multiclass negative log likelihood:
  `-log(clip(p_true, epsilon, 1))`
- Multiclass Brier score: sum of squared probability errors across classes.
- Ranked probability score: squared differences between cumulative predicted and
  observed class probabilities, averaged across the `K-1` ordered boundaries.
- Expected relevance: probability-weighted class index.
- Expected calibration error: deterministic equal-width bins of top-class
  confidence, weighted by bin sample weight.
- Classwise calibration: the same binning policy applied one-vs-rest.
- Sharpness: mean sum of squared class probabilities.
- Predictive entropy: mean categorical entropy.
- Top-class accuracy is diagnostic only and is explicitly not the primary metric.

Ordered classes benefit from RPS because probability assigned to an adjacent class
is penalized less than probability assigned to a distant class. Ordinary Brier
score does not encode that ordering.

## Quantile metrics

Pinball loss at quantile `q` is:

`max(q * (y - forecast), (q - 1) * (y - forecast))`

Exact equality has zero loss and counts as realised below-or-equal for calibration.
Outputs include per-quantile pinball loss, empirical below-or-equal rate,
calibration error, exceedance rate, aggregate pinball loss, and separate lower- and
upper-tail level lists.

Quantile crossings are reported by count and total positive crossing magnitude.
They may be warnings for diagnostics or hard contract failures when non-crossing
is required.

## CRPS

The implemented CRPS is the empirical predictive-sample formulation:

`CRPS(F, y) = E|X-y| - 0.5 E|X-X'|`

Expectations are empirical averages over the supplied predictive samples, including
all ordered sample pairs in the second term. At least two finite predictive samples
per observation are required. No unsupported closed-form distribution CRPS is
claimed.

## Prediction intervals

Diagnostics include empirical coverage, conditional coverage by supplied
deterministic bucket, mean and median width, lower/upper misses, uncovered count,
and calibration-versus-sharpness.

The interval score for nominal coverage `1-alpha` is:

`width + (2/alpha)(lower-y) I[y<lower] + (2/alpha)(y-upper) I[y>upper]`

Coverage without width is misleading because arbitrarily wide intervals can cover
every outcome while providing little useful information.

## Distributional scoring

The only implemented parametric family is Gaussian location-scale:

`NLL = 0.5 log(2*pi*scale^2) + (y-mean)^2 / (2*scale^2)`

Scale must be finite and strictly positive. Other families return
`UNSUPPORTED_PREDICTION_TYPE`; there is no automatic Gaussian fallback.

## Matched comparison

Metric comparison requires identical population checksums. It reports candidate and
benchmark metric values, their difference, direction-adjusted improvement, and
matched observation count. It performs no significance test.

Later Ticket 1D-A integration must consume ordered date-level metric differences
and apply the pre-specified dependency-aware block policy.

## Statuses

- `VALID`
- `INSUFFICIENT_DATA`
- `INVALID_INPUT`
- `UNMATCHED_POPULATION`
- `UNSUPPORTED_PREDICTION_TYPE`
- `NUMERICAL_FAILURE`

No rows are silently dropped. Non-finite inputs, invalid probability rows, immature
targets, crossed required intervals/quantiles, invalid scales, and population
mismatches fail closed.

## Assumptions and limitations

- Good likelihood is insufficient for promotion: temporal OOS lineage, calibration,
  portfolio value, costs, capacity and stability remain separate gates.
- Calibration and sharpness trade off. Concentrated predictions are useful only
  when their confidence is empirically justified.
- Quantile crossing is structurally incoherent for a monotone predictive
  distribution and must remain visible.
- Metrics are valid only after target maturity; prediction availability must not
  postdate maturity.
- Matched populations are mandatory because candidate-specific missing rows can
  manufacture metric improvements.
- Statistical significance is deliberately deferred to Ticket 1D-A.
- No real selector or historical evaluation was performed.

## Verification

```powershell
pytest -q tests/test_probabilistic_metrics.py
```

Result: `15 passed in 0.67s`.
