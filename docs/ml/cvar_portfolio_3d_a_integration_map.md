# Ticket 3D-A Deferred Integration Map

All entries have the same directive:

`DEFER UNTIL ACTIVE PUBLICATION AND SELECTOR COMPONENT RUNS FINISH`

| Future integration | Required gate |
|---|---|
| Five-date selector evaluation | Exact strict-OOS alpha and point-in-time scenario lineage |
| 46-date multi-regime evaluation | Same registered policy across matched regimes and populations |
| Ticket 2D-A aim portfolio | Matched asset, cutoff, exposure, holdings, and cost assumptions |
| Ticket 3A-A inverse-volatility and linear-shrinkage baselines | Exact asset population and scenario-return checksum |
| Ticket 3C-A HRP and sector-first HRP | Exact asset population and scenario-return checksum |
| Fixed transaction-cost scenarios | Registered cost identity and reconciled previous holdings |
| Point-in-time liquidity and ADV | Eligibility as of cutoff; ADV remains `UNVERIFIED` until proven |
| Exposure-controller outputs | Immutable registered exposure target passed into the input contract |
| Ticket 1D-A statistical safeguards | Dependency-aware inference on strict-OOS policy differences |
| Promotion reports | Verified lineage, costs, safeguards, capacity, and authoritative evaluation |

The comparison helper is ex-ante only. It blocks scenario comparison unless every
policy carries the exact scenario-return checksum. It does not calculate realised
returns, Sharpe ratios, or promotion outcomes.

