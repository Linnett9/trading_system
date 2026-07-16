# Ticket 3D-A — CVaR Portfolio Foundation

## Scope

This foundation is deterministic and synthetic-only. It creates target portfolio
weights; it does not create executable orders, estimate an exposure model, consume
historical market data, or make a promotion decision. ADV capacity is `UNVERIFIED`.

The registered research panel contains exactly 95% and 97.5% CVaR.

## Formulation

For supplied scenario return row \(r_s\), portfolio loss is
\(L_s(w)=-r_s^\top w\). The optimizer uses the Rockafellar–Uryasev representation:

\[
\operatorname{CVaR}_q =
\zeta + \frac{1}{1-q}\sum_s p_s u_s,\qquad
u_s \ge L_s(w)-\zeta,\quad u_s\ge0.
\]

It maximizes

\[
\hat\alpha^\top w
-\lambda_{\rm CVaR}\operatorname{CVaR}_q
-\kappa_1\lVert w-w_{\rm previous}\rVert_1
-\kappa_2\lVert w-w_{\rm previous}\rVert_2^2.
\]

The implementation minimizes the algebraic negative of this expression. The
reported VaR threshold is the fitted RU threshold \(\zeta\). For standalone
weighted discrete CVaR, VaR is the first ordered loss whose cumulative probability
is at least \(q\); probability mass strictly above that threshold supplies the
excess term. An active tail scenario has loss greater than or equal to
`zeta - 1e-7`.

Scenario probabilities must be strictly positive and sum to one. Omitting them
selects equal probabilities. Confidence level is a tail-risk confidence level,
not a prediction quantile. Ex-ante scenario CVaR is not realised historical
expected shortfall.

## Solver and constraints

The deterministic solver is `scipy.optimize.SLSQP`, using:

- solver tolerance `1e-10`;
- independent constraint tolerance `1e-7`;
- maximum iterations `2000`;
- a canonical, cap-aware feasible initial allocation;
- explicit weight, threshold, excess-loss, and absolute-turnover variables.

Supported constraints are exact requested exposure, long-only stock floors and
caps, sector caps, point-in-time liquidity eligibility, optional cash residual,
and a gross-turnover limit. Gross turnover is
`sum(abs(target - previous))`; one-way turnover is half that value. Newly
ineligible holdings are either liquidated or bounded by their previous weight
under the explicitly selected policy. Constraints are never relaxed.

Outcomes distinguish `OPTIMAL`, `INFEASIBLE`, `INSUFFICIENT_DATA`,
`INVALID_INPUT`, `UNSUPPORTED_CONFIGURATION`, `SOLVER_UNAVAILABLE`, and
`NUMERICAL_FAILURE`. Solver success is accepted only after the independent
verifier recomputes population identities, losses, RU constraints, risk,
penalties, objective, portfolio constraints, tail membership, and logical
identity.

## Determinism and degeneracy

Assets and scenarios must have unique, canonically sorted identities. Stable loss
sorting defines weighted discrete ties. Canonical ordering and deterministic
initialization make repeated equivalent problems reproducible. Duplicate losses
and constant-loss weak identification are reported as warnings. Equivalent
continuous optima remain subject to numerical solver precision; no heuristic
fallback is accepted.

## Scenario and point-in-time requirements

Scenarios are supplied, never manufactured by this module. Their history,
generation method and version, horizon, overlap flag, ordered population,
probabilities, and complete return matrix are recorded. Future historical use
must prove that selector alpha, scenarios, eligibility, sector membership,
liquidity, previous holdings, exposure target, and registered policy/risk
identities were available at the decision cutoff. No future return observation
may enter scenario construction.

Individual daily returns must not be treated as multi-session scenarios without
an explicit horizon contract. Current historical-universe limitations remain in
force.

## Limitations

Sample CVaR is highly sensitive to scenario quality, tail coverage, dependence,
horizon, and probability choices. A small or misspecified scenario set can give
precise-looking but fragile weights. CVaR optimization does not guarantee a low
realised drawdown. Historical policy value must ultimately be assessed on
strict-OOS decisions after transaction costs, with statistical safeguards and
point-in-time capacity evidence.

