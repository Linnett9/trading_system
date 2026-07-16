# Ticket 3A-A — Inverse-Volatility and Linear-Shrinkage Portfolio Baselines

Status: `IMPLEMENTED_SYNTHETIC_ONLY_INTEGRATION_DEFERRED`.

Implementation owner: `core/research/ml/portfolio_risk_baselines.py`.

## Contracts

- Input: `portfolio_risk_input_v1`
- Inverse volatility: `inverse_volatility_portfolio_result_v1`
- Linear shrinkage: `linear_shrinkage_covariance_result_v1`
- Minimum variance: `minimum_variance_portfolio_result_v1`
- Comparison: `portfolio_risk_baseline_comparison_v1`

Assets and observations must be unique and canonically ordered. No row or asset is
silently removed. Population and observation-population checksums are stable
SHA-256 identities. Creation metadata does not affect logical result identity.

## Inverse-volatility convention

The raw score is:

`score_i = 1 / max(annualised_volatility_i, minimum_volatility)`

Scores are scaled to the requested risky exposure. Allocation begins at configured
stock floors and then applies deterministic proportional water filling subject to:

- long-only weights;
- stock maximum weights;
- optional stock minimum weights;
- sector caps;
- liquidity eligibility;
- exact exposure.

Assets failing liquidity eligibility receive no new allocation. Zero or near-zero
volatility uses an explicit positive floor and emits `VOLATILITY_FLOOR_APPLIED`.
Stock- and sector-cap residual allocation is redistributed deterministically in
canonical asset order. Infeasible capacity fails closed.

Supplied volatility is interpreted as already annualised. Volatility derived from a
covariance diagonal is multiplied by the configured annualisation factor before
taking the square root.

## Linear covariance shrinkage

The estimator is the genuine scikit-learn Ledoit–Wolf estimator:

- estimator ID: `sklearn_ledoit_wolf_scaled_identity`;
- observed scikit-learn version: `1.6.1`;
- `assume_centered=False`;
- target: `mu * I`;
- `mu = trace(sample_covariance) / asset_count`;
- shrinkage intensity: estimated by `sklearn.covariance.LedoitWolf`.

The reported empirical covariance uses the matching maximum-likelihood convention
used by scikit-learn. The result includes:

- empirical covariance;
- scaled-identity target;
- fitted shrinkage intensity;
- final covariance;
- covariance checksum;
- minimum eigenvalue.

This is not a fixed user-supplied shrinkage coefficient and is not presented as
nonlinear shrinkage.

## Minimum-variance allocation

The optimizer solves:

`minimize w' Sigma_shrink w`

subject to:

- long-only weights;
- exact exposure;
- stock floors and caps;
- sector caps;
- liquidity eligibility;
- optional cash residual outside the requested risky exposure.

Solver:

- `scipy.optimize.SLSQP`;
- SciPy `1.17.1`;
- tolerance `1e-10`;
- constraint verification tolerance `1e-7`;
- maximum iterations `2000`.

There is no heuristic fallback reported as optimal. A solver success is accepted
only after independent allocation verification.

The result reports variance, annualised volatility, marginal/component risk
contributions, concentration HHI, maximum stock weight, sector exposure and cap
utilisation.

## Independent verification

Three read-only verification paths check:

1. Shrinkage covariance population, symmetry, PSD state, covariance checksum and
   logical result checksum.
2. Allocation population, exposure, bounds, sectors, liquidity, variance,
   volatility and logical result checksum.
3. Matched policy comparison population identity.

Changing a target weight or covariance entry invalidates verification.

## Ex-ante comparison

The comparison contract accepts any compatible result interface, including future
Ticket 2D-A results, when it supplies identical population identity and target
weights. It reports:

- expected variance and volatility;
- concentration HHI;
- maximum stock weight;
- sector concentration;
- turnover from supplied previous weights.

It does not calculate realised return, Sharpe ratio, replay performance or policy
superiority.

## Point-in-time assumptions

Future historical use requires returns available by the decision cutoff,
point-in-time eligibility, sectors and liquidity, reconciled holdings, a registered
exposure target, and immutable input identities. No future covariance observation
may enter estimation.

Current static historical-universe limitations remain applicable. ADV capacity is
`UNVERIFIED`.

## Statuses

- `VALID`
- `OPTIMAL`
- `INFEASIBLE`
- `INSUFFICIENT_DATA`
- `INVALID_INPUT`
- `UNSUPPORTED_CONFIGURATION`
- `NUMERICAL_FAILURE`
- `SOLVER_UNAVAILABLE`

## Verification

```powershell
pytest -q tests/test_portfolio_risk_baselines.py
```

Result: `13 passed in 3.14s`.

No real data, fitting, prediction, replay, evaluation, portfolio mutation or order
generation occurred.
