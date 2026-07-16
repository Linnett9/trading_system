# Ticket 4A-A Deferred Integration Map

All integrations are marked:

`DEFER UNTIL ACTIVE SELECTOR PUBLICATION AND COMPONENT-READINESS WORK FINISHES`

| Future integration | Required gate |
|---|---|
| Selector model registry | Proposed `huber_regressor` entry reviewed after active registry readers finish |
| Authoritative selector dataset | Frozen point-in-time feature and target contracts |
| Bounded strict-OOS component runner | Training-only preprocessing and label-maturity enforcement |
| Five-date component publication | Verified prediction artifact and lineage contracts |
| 46-date multi-regime evaluation | Matched Ridge, Elastic Net and Huber populations |
| Top-k, cohorts, hysteresis and aim portfolio | Frozen continuous scores passed through registered policies |
| Wave 3 risk baselines | Matched selector populations and portfolio assumptions |
| Ticket 1D-A safeguards | Dependency-aware inference across dates and regimes |
| Experiment search accounting | Complete Huber trials, failures, retries and seeds |
| Protected final-audit protocol | Closed development and authorized one-time audit |
| Promotion reports | Strict-OOS after-cost evidence; synthetic diagnostics are insufficient |

Proposed future registry characteristics:

- canonical model ID: `huber_regressor`;
- estimator: `sklearn.linear_model.HuberRegressor`;
- task: continuous stock-level forward-return selector;
- preprocessing: training-only standardisation;
- prediction: continuous score, never probability;
- initial bounded panel: the single registered default configuration from Ticket
  4A-A.

