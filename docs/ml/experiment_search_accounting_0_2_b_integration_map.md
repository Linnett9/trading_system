# Ticket 0.2B Deferred Integration Map

All integrations are marked:

`DEFER UNTIL ACTIVE SELECTOR PUBLICATION AND COMPONENT RUNS FINISH`

| Future integration | Required gate |
|---|---|
| Existing experiment ledger | Read-only consumption of committed `experiment_ledger_event_v1` events |
| Bounded selector runs | Registered hypothesis, campaign, trial, attempt and budget metadata |
| Selector component publication | Complete material trial population and campaign checksum |
| Multi-regime evaluation | Matched campaign/date-panel identities |
| Ticket 1D-A Deflated Sharpe Ratio | Verified DSR effective-search count |
| Ticket 1D-A PBO | Complete candidate population and search-campaign identity |
| Promotion reports | Valid promotion-accounting linkage and full failed/rejected history |
| Final-holdout governance | Explicit holdout-use state and immutable final validation panel |
| Research-protocol budgets | Registered configuration, seed and extension authorization |

The derived views must never become a competing append-only writer. The existing
ledger remains authoritative and every snapshot must be reproducible from its
event population and governance registrations.

