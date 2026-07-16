# Ticket 3B-A — Nonlinear Covariance Shrinkage Foundation

Status: `BLOCKED_NO_VERIFIED_IMPLEMENTATION`.

This is an intentional fail-closed outcome. No nonlinear covariance estimator was
approximated or mislabeled.

Implementation owner: `core/research/ml/nonlinear_covariance_shrinkage.py`.

## Dependency and implementation audit

| Candidate | Observed state | Version | Assessment |
|---|---|---|---|
| `nonlinshrink` | Not installed | — | Recognized analytical nonlinear-shrinkage candidate, but unavailable |
| Riskfolio-Lib | Not installed | — | No local estimator available or validated |
| PyPortfolioOpt | Not installed | — | No local nonlinear estimator available |
| QuEST-related module | Not installed | — | No local QuEST implementation |
| statsmodels | Not installed | — | No local implementation |
| scikit-learn | Installed | `1.6.1` | Ledoit–Wolf and OAS are linear shrinkage; not nonlinear |
| SciPy | Installed | `1.17.1` | Numerical tools only; no nonlinear covariance estimator |

No dependency was installed.

The repository search found no existing analytical nonlinear shrinkage, QuEST, or
verified rotationally invariant covariance owner.

Licence notes visible from known projects:

- scikit-learn and SciPy use BSD-style licences.
- Candidate-package licence compatibility was not asserted for absent packages
  without locally available metadata.

Maintained/abandoned state and independent mathematical validation were not inferred
from package names. Those checks remain necessary before dependency adoption.

## Why no method was selected

The ticket requires a recognized published formulation with exact centring,
covariance denominator, eigenvalue convention, high-dimensional behavior, PSD
guarantee and independently testable output.

The installed environment does not provide such an implementation. scikit-learn's
Ledoit–Wolf estimator is correctly retained as the Ticket 3A-A linear baseline; it
cannot be renamed nonlinear.

No ad hoc eigenvalue clipping, smoothing, kernel transform, or condition-number
regularization was implemented.

## Contracts delivered

- Input: `nonlinear_covariance_input_v1`
- Dependency audit: `nonlinear_covariance_dependency_audit_v1`
- Covariance result: `nonlinear_covariance_result_v1`
- Minimum-variance result: `nonlinear_minimum_variance_result_v1`
- Comparison: `linear_nonlinear_covariance_comparison_v1`
- Covariance verification: `nonlinear_covariance_verification_v1`
- Allocation verification: `nonlinear_minimum_variance_verification_v1`

The input contract records:

- ordered asset and observation identities;
- finite return matrix;
- annualisation;
- `demean_by_asset` centring;
- sample covariance denominator `n-1`;
- missingness rejection;
- high-dimensional policy;
- population and full return-history checksums.

High-dimensional input (`observations < assets`) can be represented explicitly, but
no capability is claimed until a selected estimator proves support.

## Blocked estimator behavior

The default estimator result is:

- status `DEPENDENCY_UNAVAILABLE`;
- no estimator ID/version claim;
- no shrunk eigenvalues;
- no nonlinear covariance;
- no precision matrix;
- no covariance checksum;
- no minimum-variance weights.

Empirical covariance and empirical eigenvalues are included only as input
diagnostics. They are not presented as nonlinear output.

A supplied backend must provide complete recognized-method metadata and an
independent-verification flag. An unverified backend returns
`METHOD_NOT_VERIFIED`. The current ticket deliberately contains no adapter that
turns arbitrary backend output into a valid result.

## Linear comparison limitation

Ticket 3A-A records:

- exact asset-population checksum;
- exact observation-identity checksum;
- covariance checksum.

It does not currently record a checksum of the complete return matrix in its
linear-shrinkage result. Therefore exact return-history equivalence across the
linear and future nonlinear estimators cannot yet be proven from result artifacts
alone.

The comparison contract consequently reports:

`LINEAR_RETURN_HISTORY_IDENTITY_UNAVAILABLE`

and remains blocked even before considering the absent nonlinear result.
Ticket 3A-A was not modified.

## Mathematical and research limitations

Nonlinear eigenvalue shrinkage may improve covariance estimation when sample
eigenvalues are noisy, particularly in high-dimensional settings. Improvement is
not guaranteed for every covariance, sample size, portfolio constraint, or utility
function.

Covariance quality must ultimately be judged through:

- out-of-sample portfolio variance;
- allocation stability;
- turnover and costs;
- concentration;
- utility under matched constraints;
- dependency-aware statistical evidence.

This ticket is not historical validation and produces no portfolio target.

## Statuses

- `VALID`
- `INSUFFICIENT_DATA`
- `INVALID_INPUT`
- `UNSUPPORTED_CONFIGURATION`
- `DEPENDENCY_UNAVAILABLE`
- `NUMERICAL_FAILURE`
- `METHOD_NOT_VERIFIED`

## Verification

```powershell
pytest -q tests/test_nonlinear_covariance_shrinkage.py
```

Result: `13 passed in 3.55s`.

No real data, fitting, prediction, replay, evaluation, portfolio mutation or order
generation occurred.
