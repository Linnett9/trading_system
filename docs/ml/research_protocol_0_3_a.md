# Ticket 0.3A — Protected Final Audit and Research-Budget Governance

## Scope

`research_protocol_v1` is a synthetic-only governance contract. It verifies
identities and evidence produced by existing dataset, fold, panel, experiment,
search-accounting, statistical and promotion owners. It does not generate folds,
run evaluation, access a live holdout, or write an audit ledger.

## Development, validation, and final audit

Dates have one explicit domain:

- `TRAINING`;
- `INNER_VALIDATION`;
- `DEVELOPMENT_EVALUATION`;
- `FINAL_AUDIT`;
- `EXCLUDED_EMBARGO`;
- `EXCLUDED_PURGE`;
- `UNASSIGNED`.

Training and inner validation support model construction. The deterministic
development panel supports bounded comparison and continuation decisions. The
final audit is a protected, disjoint period used only after development closes.
It is not another tuning panel.

Final-audit dates cannot appear in training, validation, development evaluation,
or panel selection. Dates are unique and ordered; no row is silently dropped.

## Purging and embargo

Fold verification requires chronological training strictly before validation.
When a training target matures on or after the validation boundary, its training
date must appear in the explicit purge identity. Embargo dates must be excluded
from training. Purging prevents label-outcome overlap; embargo reduces boundary
dependence. This owner verifies supplied fold evidence and does not replace the
existing fold implementation.

## Frozen development history

`frozen_development_history_v1` records dataset, row population, decision dates,
target contract, latest source availability, freeze time, commit, correction
policy, and amendment lineage.

After freeze, additions or corrections require a new protocol version. An
amendment must change the row checksum and link to the prior frozen-history
checksum. Final-audit observations cannot be reclassified as development evidence
under the old version.

## Search-budget hierarchy

`research_budget_policy_v1` sets limits for model families, hyperparameter
configurations, seeds, total material trials, extensions, final-audit access, and
the shared hypothesis-level budget.

Ticket 0.2B remains the owner of logical-trial reconstruction and effective search
count. This protocol consumes that verified count. A new campaign ID does not
reset the shared hypothesis budget. Extensions require a permitted reason,
approval record, and an available extension slot. Missing or exceeded budgets
block readiness.

## Final-audit access

`final_audit_access_event_v1` is a validation contract, not a live writer. Each
event binds requester, protocol, hypothesis, campaign, trial, dataset, purpose,
result, outcome, commit and post-access change state.

Ordinary authorized access consumes the protocol allowance. An operational rerun
does not consume another access only when it references the prior event, corrects
infrastructure failure, and preserves model, policy and population identities.
Changing any of those identities contaminates the protocol.

Contamination includes audit-driven changes to:

- hyperparameters;
- feature selection;
- model-family selection;
- portfolio-policy selection;
- cost assumptions;
- promotion thresholds.

Unauthorized access, population mismatch, access-budget excess, or relabelling
audit evidence as development evidence fails closed.

## State machine

The normal path is:

`DRAFT → DEVELOPMENT_FROZEN → DEVELOPMENT_ACTIVE → DEVELOPMENT_CLOSED → FINAL_AUDIT_AUTHORIZED → FINAL_AUDIT_COMPLETE → PROMOTION_DECIDED`

Any active state may transition to `INVALIDATED` where contracted. Final audit
cannot begin before development closes. Development cannot reopen after audit.
Promotion cannot precede required audit completion, and an invalidated protocol
cannot authorize promotion.

## Promotion governance

`promotion_governance_v1` requires complete Ticket 0.2B accounting, a selected
trial, effective search count, development result, final-audit result, cost and
portfolio identities, statistical safeguards, DSR/PBO evidence where required,
decision reason and approval.

Supported decisions are `PROMOTE`, `REJECT`, `CONTINUE_DEVELOPMENT`,
`INVALIDATE`, and `BLOCKED`. A requested promote or reject decision becomes
`BLOCKED` when governance evidence is incomplete or contaminated.

Logical checksums exclude creation timestamps. Independent verification
recomputes protocol, date, frozen-history, fold, panel, budget, access-event,
promotion and readiness identities.

