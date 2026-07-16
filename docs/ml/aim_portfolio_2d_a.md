# Ticket 2D-A — Turnover-Penalised Aim Portfolio Foundation

Status: `IMPLEMENTED_SYNTHETIC_ONLY_INTEGRATION_DEFERRED`.

Implementation owner: `core/research/ml/aim_portfolio.py`.

## Contracts

- Input: `aim_portfolio_input_v1`
- Result: `aim_portfolio_result_v1`
- Policy: `turnover_penalised_aim_portfolio_v1`
- Independent verification: `aim_portfolio_verification_v1`
- Comparison controls: `aim_portfolio_comparison_controls_v1`

Asset-population identity is the SHA-256 hash of the canonically ordered asset IDs.
Creation timestamps do not affect the logical result checksum.

## Objective

The implemented maximization objective is:

`alpha' w - (lambda/2) w' Sigma w - kappa1 ||w-w_previous||_1 - kappa2 ||w-w_previous||_2^2`

The solver minimizes its exact negative. The L1 norm is represented with auxiliary
variables `t_i` and linear constraints:

- `t_i >= w_i - w_previous_i`
- `t_i >= -(w_i - w_previous_i)`

The result reports separately:

- expected-alpha contribution;
- covariance-risk penalty;
- L1 turnover penalty;
- L2 turnover penalty;
- gross objective value.

These are ex-ante quantities. Expected alpha is not realized return. Covariance
risk is not ex-post volatility. Turnover penalties are not transaction costs.
Target weights are not executable orders.

## Constraints

- Long-only weights.
- Exact sum of risky weights equals the requested exposure target.
- Optional cash is the residual outside that exposure; no cash asset is optimized.
- Per-asset maximum weights.
- Optional per-asset minimum weights.
- Per-sector maximum exposure.
- Liquidity eligibility.
- Gross turnover limit.
- Deterministic newly ineligible asset policy:
  - `liquidate`: target cap becomes zero;
  - `retain_only`: target cannot exceed the previous holding.

Turnover convention:

- gross turnover: `sum(abs(target - previous))`;
- one-way turnover: gross turnover divided by two.

Constraints are never silently relaxed. Infeasible stock or sector capacity fails
closed.

## Solver

- Solver: `scipy.optimize.SLSQP`
- Installed SciPy observed during implementation: `1.17.1`
- Solver tolerance: `1e-10`
- Constraint verification tolerance: `1e-7`
- Maximum iterations: `2000`
- PSD tolerance: `1e-8`

Materially non-PSD covariance matrices are rejected. A covariance matrix with only
a small negative eigenvalue inside tolerance is projected onto the PSD cone for the
solver and reported with a warning.

There is no unconstrained or heuristic success fallback. A nominal solver success
is accepted only if the independent verifier confirms the original constraints,
objective components, population identity, and result checksum.

## Determinism

- Assets must arrive in stable sorted canonical order.
- Initial feasible weights are constructed in canonical asset order.
- Equivalent worst/best choices use asset ID tie-breaking.
- Repeated solutions use identical solver configuration.
- Economically duplicated alpha/variance pairs emit
  `NON_UNIQUE_OPTIMUM_POSSIBLE_STABLE_ASSET_ORDER_USED`.

## Result states

- `OPTIMAL`
- `INFEASIBLE`
- `INVALID_INPUT`
- `UNSUPPORTED_CONFIGURATION`
- `NUMERICAL_FAILURE`
- `SOLVER_UNAVAILABLE`

Inputs fail closed for non-finite values, dimension mismatch, asymmetric or
materially non-PSD covariance, negative penalties, invalid previous weights,
inconsistent stock bounds, infeasible sector caps, invalid liquidity masks, and
unsupported long-short configuration.

## Comparison controls

Synthetic ex-ante controls are available for:

- equal-weight top-k;
- inverse-volatility top-k;
- unchanged previous holdings.

They report expected alpha, covariance variance/volatility, turnover, penalties,
objective value, concentration, and constraint residuals. They do not compute
historical performance or claim superiority.

## Point-in-time and lineage requirements

Future integration must require:

- strict-OOS selector alpha;
- point-in-time covariance data;
- point-in-time sector classifications;
- point-in-time liquidity eligibility;
- reconciled previous holdings;
- upstream exposure-controller output;
- registered policy and cost identities.

Synthetic or current classifications must never be represented as historical
point-in-time evidence. ADV capacity remains `UNVERIFIED` until suitable
point-in-time ADV inputs exist.

## Verification

```powershell
pytest -q tests/test_aim_portfolio.py
```

Result: `13 passed in 1.60s`.

No real data, replay, portfolio mutation, order generation or execution occurred.
