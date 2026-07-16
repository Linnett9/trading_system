# Ticket 3C-A deferred integration map

`DEFER UNTIL ACTIVE PUBLICATION AND SELECTOR COMPONENT RUNS FINISH`

| Integration point | Required future input | Mandatory gate | Future output | Why deferred |
|---|---|---|---|---|
| Five-date selector evaluation | Five strict-OOS candidate universes and cutoff-safe return histories | Exact candidate/date/history identity | Standard, top-20, top-40 and sector-first HRP diagnostics | Components do not yet exist |
| 46-date multi-regime evaluation | Atomic panel with identical policy definitions | Same clustering conventions and point-in-time inputs | Regime-level HRP risk, turnover and stability | Historical evaluation has not run |
| Ticket 3A-A baselines | Equal population and full return-history checksum | Same exposure, caps, sectors and liquidity | Equal/inverse-vol/linear-min-var/HRP comparison | Live integration is excluded |
| Ticket 2D-A aim portfolio | Compatible target-weight result and covariance identity | Same assets, histories and exposure controller | Aim-versus-HRP ex-ante comparison | Aim owner must remain untouched |
| Future CVaR | Point-in-time scenario contract | Same candidate universe and tail horizon | CVaR-versus-HRP comparison | Later Wave 3 ticket |
| Fixed transaction costs | Verified HRP target-to-prior weight changes | Separate raw/constrained weights and registered costs | 5/10/25/50 bps scenarios | No replay or realized performance here |
| Exposure controller | Registered point-in-time exposure target | Controller identity precedes allocation | Scaled risky weights and cash residual | Upstream integration is deferred |
| Ticket 1D-A safeguards | Ordered date-level policy metric differences | Matched histories and pre-specified blocks | Dependency-aware HRP comparisons | Requires real completed histories |
| Promotion reports | Strict-OOS candidates, verified HRP, costs, capacity and inference | Protected final holdout | Eligibility and blocking evidence | Promotion is outside this foundation |

Future integration must preserve exact candidate-universe, asset-population,
observation-population, return-history, covariance, tree and configuration
identities. Static sector or liquidity mappings must not be represented as
historical point-in-time evidence. ADV capacity remains `UNVERIFIED`.

