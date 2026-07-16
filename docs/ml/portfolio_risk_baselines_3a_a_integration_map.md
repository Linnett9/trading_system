# Ticket 3A-A deferred integration map

`DEFER UNTIL ACTIVE PUBLICATION AND SELECTOR COMPONENT RUNS FINISH`

| Integration point | Required future input | Mandatory gate | Future output | Why deferred |
|---|---|---|---|---|
| Five-date selector evaluation | Five strict-OOS selector partitions and point-in-time return histories | Exact asset/date population and cutoff-safe observations | Equal-weight, inverse-volatility and shrunk-minimum-variance ex-ante comparison | Components do not yet exist |
| 46-date multi-regime evaluation | Atomic panel plus matched covariance inputs | Same policy population on all dates/regimes | Regime-level risk, concentration and turnover summaries | Real panel has not run |
| Ticket 2D-A aim portfolio | Compatible population, exposure and covariance identities | Same asset ordering and constraint convention | Matched ex-ante risk-baseline table | Live integration into the aim optimizer is excluded |
| Fixed-bps transaction costs | Verified target-to-previous weight changes | Preserve optimizer risk result separately from cost accounting | 5/10/25/50 bps turnover-cost scenarios | This ticket reports no realized cost/performance |
| Exposure controller | Registered point-in-time exposure target | Controller timestamp and identity precede allocation | Scaled risky weights and cash residual | Upstream integration is deferred |
| Future nonlinear shrinkage | Same return-history population | Separate estimator/version and verification contract | Nonlinear-shrinkage covariance baseline | Explicit later Wave 3 ticket |
| Future HRP | Verified covariance and distance/linkage contract | Deterministic clustering identity | HRP target weights and risk comparison | Explicit later Wave 3 ticket |
| Future CVaR | Scenario/sample contract and tail probability | Point-in-time scenarios and optimization verification | CVaR-efficient target weights | Explicit later Wave 3 ticket |
| Ticket 1D-A safeguards | Ordered date-level policy-risk differences | Matched populations and pre-specified blocks | Dependency-aware comparison inference | Requires completed historical evaluation |
| Promotion reports | Strict-OOS inputs, verified allocation, costs, capacity and inference | Final holdout protected from repeated selection | Policy eligibility and blocking reasons | Promotion is outside this foundation |

Future integration must retain input population, observation population, covariance,
configuration and policy checksums. A failed verification must block downstream use.
Static sector/liquidity data must not be described as historical point-in-time
evidence, and ADV capacity must remain `UNVERIFIED` until validated inputs exist.

