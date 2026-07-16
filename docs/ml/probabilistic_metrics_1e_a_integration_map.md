# Ticket 1E-A deferred integration map

`DEFER UNTIL ACTIVE PUBLICATION AND SELECTOR COMPONENT RUNS FINISH`

| Integration point | Required future input | Population and maturity gate | Output to add | Why deferred |
|---|---|---|---|---|
| Ordered-logit component validation | Verified strict-OOS class probabilities, relevance labels and decision timestamps | Exact component row checksum; probabilities sum to one; labels mature before evaluation | NLL, Brier, RPS, expected relevance, calibration and sharpness section | Authoritative ordered-logit components do not yet exist |
| Selector evaluation aggregates | Completed date-level probabilistic component metrics | Every model shares the same resolved dates and row populations | Per-date and aggregate probabilistic metric partitions | Live evaluator is excluded from this ticket |
| Future quantile LightGBM | Strict-OOS registered quantile forecasts | Registered sorted quantile levels; no silent row loss; crossing policy declared | Pinball, calibration, tail and interval diagnostics | Model/registry contract is not yet implemented |
| Future NGBoost/probabilistic models | Explicit supported family parameters or predictive samples | Family/parameter validation and exact target population | Family NLL and/or empirical CRPS | Only Gaussian NLL and empirical-sample CRPS are currently supported |
| Volatility forecasting | Mature realised-volatility targets and probabilistic forecasts | Target unit/horizon identity and exact maturity evidence | Distributional score, quantile calibration and intervals | No authoritative probabilistic volatility artifact |
| Exposure-tail prediction | Mature tail-event labels, quantiles or samples | Exact exposure target contract and matched population | Tail pinball, exceedance calibration, CRPS and interval diagnostics | Exposure-tail model outputs do not yet exist |
| Ticket 1D-A safeguards | Ordered date-level candidate-minus-benchmark metric differences | Same population checksum and pre-specified block length | Block-bootstrap confidence, SPA/MCS comparison, and search-adjusted evidence | Significance belongs to Ticket 1D-A and requires real completed histories |
| Promotion reports | Verified probabilistic metrics plus strict-OOS lineage and statistical evidence | No repeated final-holdout selection; complete costs/capacity/stability gates | Probabilistic quality gate with blocking reasons | Promotion must wait for active publication and components |

Integration must preserve the input population checksum, configuration checksum,
target unit, metric version and logical result checksum. Blocked metric results must
never be converted into zero, null-imputed, or partially matched comparison values.

