# Ticket 0.2B — Experiment Search Accounting and Materialised Views

## Authority and scope

The append-only `experiment_ledger_event_v1` ledger remains the authoritative
writer and event history. This foundation is a deterministic, read-only adapter.
Its JSON and Parquet files are derived snapshots that can be deleted and rebuilt;
they are not ledger records and never append lifecycle events.

This ticket used synthetic events only and did not read the live ledger.

## Identities

`experiment_hypothesis_v1` identifies the registered statement, research family,
primary metric, benchmark, continuation and rejection rules, and registration
time.

`experiment_search_campaign_v1` binds a hypothesis to model, dataset, feature,
target, portfolio, cost, risk, date-window and search-budget identities.

`experiment_logical_trial_v1` identifies a hyperparameter configuration, random
seed, dataset, fold, training and validation dates, model and source commit. A
logical trial can contain multiple process attempts. Events are neither attempts
nor trials: events describe lifecycle changes; an experiment run ID groups events
into an attempt; equivalent attempts collapse into one logical trial.

Legacy events are preserved but receive `LEGACY_UNASSIGNED` when hypothesis and
campaign governance is absent. No identity is invented.

## Counting policy

The policy is `material_trial_counting_policy_v1`.

One distinct hyperparameter × seed × dataset × fold identity is one logical trial.
An explicit consistent `trial_id` may provide that identity. The following rules
apply:

- identical retries and resumed executions remain visible as attempts but count
  once;
- cache reuse and `SKIPPED_COMPLETE` do not increase the count;
- different seeds count separately;
- different complete hyperparameter identities count separately;
- completed and materially evaluated failed trials count;
- materially evaluated rejected trials count;
- rejection before material evaluation is reported separately and excluded;
- completed trials later invalidated remain visible and count;
- duplicate events are invalid and never add to a count;
- incomplete lifecycles remain visible, produce warnings, and do not count.

The initial DSR effective-search count equals the material effective-search count.
Any future policy that differs must report explicit difference reasons. These
counts are inputs to later Deflated Sharpe Ratio and PBO work; this owner performs
neither calculation.

## Campaign budgets

The base material budget is planned configurations multiplied by planned seeds.
An explicit authorised extension increases the effective budget. Material trials
consume the budget whether completed, failed, rejected after evaluation, or later
invalidated.

Accounting fails closed for missing budgets, cross-campaign trial identity,
campaign/hypothesis inconsistency, or budget excess without an authorised
extension. Continuation authorization requires a recorded reason. Campaign
summaries disclose attempts, outcomes, remaining budget, utilisation, closure,
continuation and stop state.

## Reconstruction and lifecycle policy

Events are sorted by event timestamp and event ID. Event IDs must be unique.
Supported paths include:

- `PLANNED → STARTED → COMPLETED|FAILED|REJECTED|CANCELLED|SKIPPED_COMPLETE`;
- `PLANNED → REJECTED|CANCELLED`;
- `COMPLETED → INVALIDATED`.

Malformed transitions are not repaired. Raw event IDs remain attached to process
attempts, and failed/rejected attempt history remains present when a later retry
completes.

## Materialised views

The materialiser writes only to a caller-supplied directory:

- canonical JSON snapshot;
- logical-trial Parquet table;
- process-attempt Parquet table;
- campaign-summary Parquet table;
- hypothesis-summary Parquet table;
- effective-search-count Parquet table;
- JSON validation report.

Rows and columns have deterministic order. Parquet files have no index, use
explicit selected columns, and carry
`experiment_search_materialisation_v1` schema metadata. Nested values are encoded
as canonical JSON strings. Verification checks JSON equality, Parquet schema
metadata, row counts and exact selected-column contents.

## Promotion linkage

`experiment_promotion_accounting_v1` binds a decision to its hypothesis,
campaign, selected and benchmark trials, complete material-search count, counting
policy, final validation panel, holdout-use state, reason and report checksum.
Promotion accounting blocks if campaign accounting is incomplete, the selected
trial is outside the campaign, failed/rejected material trials are missing, or the
reported search count differs.

