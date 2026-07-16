# Ticket 1F-A — Volatility Forecast and Targeting Metrics

## Scope

This module evaluates supplied synthetic point forecasts and supplied realised
values. It does not estimate a volatility model, manufacture realised volatility,
control exposure, evaluate a portfolio, or generate orders.

The common input and result contracts are `volatility_metric_input_v1` and
`volatility_metric_result_v1`.

## Representations and annualisation

Every input declares one of:

- `variance`;
- `volatility`;
- `annualised_variance`;
- `annualised_volatility`.

Metrics do not silently convert representations. Conversion is performed only by
the versioned conversion helper, with an explicit annualisation factor \(A\):

\[
\sigma_{\rm annual}=\sigma_{\rm daily}\sqrt A,\qquad
v_{\rm annual}=v_{\rm daily}A.
\]

Negative and non-finite values and non-positive annualisation factors are
rejected. Conversion results preserve ordered observation identities and record
source values, converted values, units, configuration, population, and checksums.
MSE and MAE values must not be compared across incompatible units.

## Forecast losses and diagnostics

Signed forecast error is:

\[
e_t=\widehat x_t-x_t.
\]

Positive error is over-prediction and negative error is under-prediction. The
foundation reports weighted MSE, RMSE, MAE and signed bias, unweighted median
absolute error, counts, frequencies, dispersion, means, a forecast-to-realised
ratio, calibration slope through the origin, and an optional weighted
intercept-and-slope regression.

The regression returns `INSUFFICIENT_DATA` within its diagnostic payload when
there are fewer than three observations or the design is singular. These are
point-forecast calibration and bias diagnostics, not probabilistic calibration.

Optional normalised MSE uses the explicitly registered
`weighted_mean_realised_squared_v1` denominator. No other denominator is inferred.

## QLIKE convention

QLIKE is calculated only when both representations are the same explicit
variance representation:

\[
\operatorname{QLIKE}_t
=\frac{v_t}{\widehat v_t}
-\log\left(\frac{v_t}{\widehat v_t}\right)-1.
\]

Lower is better and a perfect forecast has zero loss. This ratio form penalises
under- and over-forecasting asymmetrically.

Forecast variance must be strictly positive. A zero forecast is rejected unless
the caller supplies a strictly positive `qlike_forecast_floor`. Applying that
floor produces `QLIKE_FORECAST_FLOOR_APPLIED`, and the floor is part of the
configuration checksum. There is no undocumented epsilon or clipping.

Because the requested ratio formulation contains \(\log(v/\widehat v)\), realised
variance equal to zero is explicitly rejected as
`QLIKE_REALISED_VARIANCE_ZERO_UNDEFINED`. This avoids returning infinity or
silently imposing a realised-value floor.

The result retains observation-level QLIKE losses when requested and reports the
weighted mean, median, and 5%, 50%, and 95% sample quantiles.

## Volatility-target convention

Target error is:

\[
e_t=\sigma^{\rm realised}_t-\sigma^{\rm target}_t.
\]

Positive error is an overshoot. The target and realised portfolio series must use
the input's explicit volatility representation and unit. Targets must be positive
because percentage target errors are part of the contract.

The module reports signed, absolute, squared, percentage and aggregate errors,
overshoot and undershoot frequencies, maximum overshoot and undershoot, and
weighted proportions inside caller-specified percentage bands. Exposure values
may be recorded for lineage but never alter a metric.

## Horizons, timing, comparison, and verification

Horizon identity is part of every result. Forecast availability must be no later
than its decision cutoff, and realised maturity must be strictly after forecast
availability. Overlapping outcomes produce an explicit warning. Later inference
must use Ticket 1D-A's dependency-aware block-bootstrap contracts; this module
computes no p-values.

Matched comparisons require identical population, horizon, forecast and realised
representations, unit, annualisation factor, and timing checksum. Differences are
candidate minus benchmark; improvement negates the difference because lower is
better.

The read-only verifier reruns the registered metric configuration and checks
population, input values, timing, representation, unit, annualisation, aggregates,
configuration identity, and logical result identity. Creation timestamps do not
affect logical identity.

## Limitations

QLIKE is useful because it is scale-sensitive and emphasizes variance forecast
misspecification, but it remains dependent on the quality of the realised
variance proxy and horizon alignment. Good forecast metrics do not automatically
produce better exposure decisions: constraints, turnover, costs, tail events and
forecast timing intervene. Strict-OOS portfolio utility after costs remains the
ultimate decision metric.

