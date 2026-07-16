# Ticket 4B-A Deferred Integration Map

All integrations are marked:

`DEFER UNTIL ACTIVE SELECTOR PUBLICATION AND COMPONENT-READINESS WORK FINISHES`

| Future integration | Required gate |
|---|---|
| Selector model registry | Review proposed contextual model and interaction identities after active readers finish |
| Authoritative selector dataset | Frozen stock and point-in-time context schemas |
| Bounded strict-OOS runner | Training-only preprocessing and one-context-vector-per-date validation |
| Five-date component publication | Verified lineage, scores, ranks and interaction checksum |
| 46-date multi-regime evaluation | Matched Huber, Ridge, Elastic Net and ordered-logit controls |
| Portfolio-policy panel | Frozen continuous scores passed to registered policies |
| Wave 3 risk baselines | Matched populations, costs and exposure assumptions |
| Ticket 1D-A safeguards | Dependency-aware comparison across regimes and dates |
| Experiment search accounting | Complete contextual trials and bounded interaction identity |
| Protected final-audit governance | Closed development and unchanged interaction contract |
| Promotion reports | Strict-OOS after-cost evidence; no causal interpretation of interactions |

Proposed future registry characteristics:

- canonical model ID: `contextual_elastic_net`;
- estimator: `sklearn.linear_model.ElasticNet`;
- task: continuous stock-level forward-return selector;
- stock and context schemas recorded separately;
- interaction contract fixed to the six-entry Ticket 4B-A list;
- initial configuration fixed to `alpha=0.001`, `l1_ratio=0.25`;
- predictions are continuous scores, not probabilities.

