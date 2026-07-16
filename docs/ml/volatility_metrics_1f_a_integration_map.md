# Ticket 1F-A Deferred Integration Map

Every integration below is marked:

`DEFER UNTIL ACTIVE PUBLICATION, SELECTOR COMPONENT, AND FIVE-MINUTE ARCHIVE WORK FINISHES`

| Future integration | Required evidence |
|---|---|
| Deterministic exposure evaluation | Matched point-in-time forecasts, targets, exposure decisions, horizons, and costs |
| EWMA volatility targeting | Registered EWMA identity and strict forecast-availability cutoff |
| HAR and HARQ models | Immutable feature, horizon, maturity, and model identities |
| Realised-GARCH or HEAVY model | Verified dependency and point-in-time realised-measure construction |
| Small LightGBM volatility model | Frozen features, bounded fitting, and strict-OOS predictions |
| Five-minute realised-volatility features | Completed archive lineage and no future intraday observations |
| Selector and portfolio reports | Exact matched decision populations and units |
| Ticket 1D-A statistical safeguards | Block bootstrap for overlapping outcomes; no inference in this owner |
| Promotion reports | Strict-OOS forecast evidence plus after-cost portfolio utility |

Forecast performance must remain separate from exposure-policy performance.
Neither a lower QLIKE nor a lower target error alone authorizes promotion.

