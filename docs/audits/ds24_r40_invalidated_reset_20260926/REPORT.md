# DS24 R40 invalidated reset closeout

Date: 2026-09-26
Classification: `DS24_CLEAN_V2_BLOCKED_DATA_OR_CONFIG_AUTHORITY`

## Outcome

The contaminated V1 tournament is retired and cannot autostart. No obsolete DS24
process was live at the reset boundary, so no process was killed. The exact startup
entry was disabled by recoverable rename; the matching scheduled task was absent.
DS26 remained active and untouched. No PAPER or LIVE order was placed.

The clean Dell tournament was not launched. Its only remaining gate failure is that
the six FULL_CLEAN worker commands are unresolved. The legacy R40 workers directly
consume V1/deleted-stage state, so binding them to the new supervisor would silently
reintroduce the retired authority.

## Migrated static authority

`config/ds24_clean_v2` now owns the ordered 101 predictors, predictor groups, feature,
target, eligibility and calendar contracts, exact recoverable model configurations,
three tournament lanes, five-session refit policy, common metrics/costs, host ownership,
and Results Ledger prior evidence. The static bundle SHA-256 is
`72807406e0642f0ab20123beebfcbd65c92c8f9351cf95cf7545a3f671209385`.

The Results Ledger was read only as prior evidence from `Authoritative Experiments`,
`Selected Candidates`, and `Final Model Decisions`. Its captured XLSX export SHA-256
is `2497b99c865dfb9c5392b999a13febee875eea665ade56381ef51b5cdfd02032`.

## Clean physical authorities

- Feature authority: `CANONICAL_5M_FEATURE_AUTHORITY_FULL_V2_PIT_REPAIRED`
- Feature logical SHA-256: `ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d`
- Feature population: 5,654 partitions / 114,497,377 rows
- Target logical SHA-256: `41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1`
- Target delta: 514 symbols / 2,251,012 rows / 1,699,821 trainable rows
- Explicit empty target-window symbols: `JHG`
- Maximum historical outcome consumed: `2026-09-23T20:00:00Z`
- Frozen prospective validation start: `2026-09-24`

The V2 sidecar replaces the three confirmed cross-session leaks and eight additional
future-dependent or future-timestamp-derived fields discovered by the stronger gate.
All 101 predictors pass nine future-bar perturbation cases spanning seven symbols,
2017/2024, normal/early-close sessions, open/mid/near-close decisions, and extended
hours. Both feature and target composites carry immutable partition inventories and
hashes for cross-host verification.

## Validation

- Focused plus adjacent tests: 62 passed
- Scoped compilation: passed
- Architecture conformance: 1,748 modules, 11,282 edges, zero cycles
- Deleted-stage reference audit: 229 classified references, zero active clean-runtime dependencies
- Obsolete DS24 processes: zero
- PAPER orders: zero
- LIVE orders: zero

## Cleanup and retention

The obsolete generated DS24 component tree was already absent when this controlled
reset began; its 210 pre-existing tracked deletions were preserved rather than folded
into the scoped implementation commit. Canonical raw data, V1 immutable feature data,
source, tests, configs, forensic audits, `data/`, and `mac_aux_runs/` were retained.
The preserved forensic audit remains anchored at commit `d29fb4488`.

Generated feature/target authorities and the causality certificate remain under
`data/` and `research_runs/`; they are intentionally not committed.
