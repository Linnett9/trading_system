# Ticket 2D-A deferred integration map

`DEFER UNTIL ACTIVE PUBLICATION AND SELECTOR COMPONENT RUNS FINISH`

| Integration point | Required future input | Mandatory gate | Future output | Why deferred |
|---|---|---|---|---|
| Five-date selector evaluation | Five strict-OOS selector alpha partitions, point-in-time covariance, sectors, liquidity and reconciled holdings | Exact asset/date population and upstream exposure identity | Per-date aim-portfolio target-weight diagnostic | Components and authoritative parents are still being published |
| 46-date multi-regime evaluation | Atomic multi-regime selector aggregate plus matched risk inputs | Same panel and asset population for every policy | Regime-aware ex-ante objective and constraint summaries | Real panel has not run |
| Fixed-bps accounting | Target-to-previous weight changes and registered 5/10/25/50 bps cost contract | Keep optimization penalty distinct from realized cost accounting | Post-optimization cost scenario table | Transaction costs are deliberately outside the optimizer objective in this ticket |
| Point-in-time ADV capacity | Historical ADV as of each decision timestamp | Exact asset mapping, source timestamp and capacity policy | Capacity utilization and verified eligibility constraints | Current ADV scenarios remain unverified |
| Inverse-volatility/covariance baselines | Identical covariance and asset population | Same exposure, caps and eligibility constraints | Matched ex-ante baseline comparison | Only synthetic control helpers exist |
| Exposure controller | Registered point-in-time exposure target | Controller artifact identity and decision timestamp precede optimization | Exact risky exposure target and cash residual | Upstream live integration is excluded |
| Parent-order generation | Independently verified target weights plus reconciled live/paper holdings | Separate execution, lot, cash, compliance and broker gates | Proposed parent-order artifact | Optimizer target weights are not orders |
| Ticket 1D-A inference | Ordered date-level policy metric differences | Matched population checksum and pre-specified block policy | Dependency-aware comparison confidence | Requires completed historical evaluation |
| Promotion reports | Strict-OOS alpha, verified aim results, costs, capacity and statistical evidence | Final holdout not reused for selection | Policy eligibility and blocking reasons | Promotion is outside the foundation ticket |

Future consumers must preserve policy, alpha, covariance, population and configuration
identities. A solver result must pass the independent verifier before any downstream
consumer can use it. No blocked or infeasible result may be converted into fallback
weights while retaining the aim-portfolio method identity.

