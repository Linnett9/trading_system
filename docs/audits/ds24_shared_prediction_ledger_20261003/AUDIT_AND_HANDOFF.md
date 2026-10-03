# DS24 Shared Prediction Ledger Audit and Handoff

## Safety and baseline

Development used branch `ds24-shared-prediction-ledger-20261003` in isolated
worktree `.worktrees/ds24-shared-prediction-ledger`, based on Git commit
`78e1bd97930124feb90e30f265cf60cc835b2425`. No live process was stopped,
restarted, signalled, reconciled, or imported for execution.

The pre-change live capture at `2026-10-03T12:39:58.4498071Z` found:

- CLEAN V2 generation `1791030552692796600`;
- supervisor PID `21936` and random-forest worker PID `20860`;
- active CLEAN source hash
  `f1c1d53928392083c98be39cf04d11727ce31b09c0692382e9a2825a4f09aaf5`;
- random forest `RUNNING`, metrics cursor `28,691`, 87 completed refits;
- Momentum `DEFERRED_RESOURCE_CAPACITY`, metrics cursor `129,636`;
- no failure registered for the current generation. Historical failure
  pointers exist and were not modified.

## Phase 1: exact Dell warm-up audit

The historical Dell implementation is Git object
`09730bcfd322218b81b20c72b2d30064f2b7fdf2`, formerly at
`scripts/local/ds24_p8_r14_e3g_c2_r7_r7_full_burn_in_worker.py`. Its shared
predecessor/object `0651b3a6b872b8f4688ba139e35d6033f01ce1b0` records the same boundary.

The exact rule was:

- `T0 = 2018-01-02T14:35:00Z`;
- physical training burn-in from `2016-01-04T14:35:00Z` through
  `2017-12-29T21:00:00Z`;
- the boundary was a decision timestamp, not a count of sessions, dates, or
  refits;
- the worker selected `scored = [ts for ts in spine if ts >= T0]`;
- therefore pre-T0 predictions were not generated or persisted;
- both returns and Rank IC implicitly began at T0 because the evaluator
  received no pre-T0 score rows;
- the boundary was shared by Ridge and PCA-Ridge in that worker. It was a
  worker-local constant rather than a reusable tournament/evaluator contract.

The new authority names this `EVALUATION_BURN_IN`, preserves the exact T0, and
makes one deliberate output-only extension: pre-boundary predictions are now
ranked and persistable with `evaluation_eligible=false`. Both headline returns
and Rank IC remain excluded before T0, preserving the historical metric
population. This is unrelated to the protected true holdout and does not read
it.

The later CLEAN V2 tournament contract starts FULL_CLEAN scoring after each
family's rolling training lookback and did not carry T0 as an explicit shared
evaluation authority. The running random-forest V3 summary confirms that
different provisional behavior: its first resolved timestamp is
`2016-02-02T14:35:00Z`, with 25,048 valid Rank-IC timestamps and 315 daily
return rows at the audit point. Those historical V3 artifacts remain readable
and are not rewritten. The new V2 research-output contract is the explicit
authority for future evaluation/research consumers at a controlled code/run
boundary.

## Shared ledger contract

Schema `DS24_RANKED_PREDICTION_LEDGER_V1` contains only:

`decision_timestamp`, `asset_id`, `family`, `raw_prediction`,
`cross_section_rank`, `rank_percentile`, `eligible_universe_size`, `refit_id`,
`refit_timestamp`, `evaluation_eligible`, `source_hash`,
`feature_authority_hash`, `target_authority_hash`, `static_bundle_hash`, and
`run_id`.

Rank one is the highest raw score. Ties use ascending stable asset identity,
matching the established Dell metrics writer's deterministic `score_rank`.
`rank_percentile` is derived from that total rank with the highest score at 1.

Target values, labels, realized/realised returns, forward returns, and feature
matrices are forbidden. Outcomes may be joined later through the read-only API
only when the caller supplies a certified authority ID and explicit holding
period.

## Storage and policy

Research-output policy `DS24_RESEARCH_OUTPUT_POLICY_V2` authorizes this compact
artifact while retaining the V3 metrics-only policy as historical read-only
authority. Parts are Zstandard-compressed Parquet, partitioned by family and
decision month. Part names are content-addressed and deterministic. Publication
is temporary-write then atomic replace; the compact manifest is published
atomically after the immutable part. A crash after part publication is
recoverable because retry validates and adopts the deterministic orphan.

Duplicate `(family, decision_timestamp, asset_id)` keys are rejected. An exact
same-content retry is an idempotent no-op. Candidate overlap checks read only
the key columns from the bounded family/month partition.

The manifest is protected by an exclusive-create `.manifest.lock` carrying
hostname, PID, token and creation time. Writers serialize all manifest
read/check/publish work. A dead same-host owner can be recovered without
signalling it (Windows uses a non-signalling process handle); a live or remote
owner is never stolen and acquisition fails closed on timeout.

## Caller integration and current CLEAN V2 compatibility

`DS24PredictionLedgerPublisher` is the application boundary. It accepts the
established CLEAN V2 scoring columns (`family`, `decision_timestamp`,
`asset_id`, `prediction`, `eligible`), filters exactly the production eligible
universe, derives `evaluation_eligible` from `EVALUATION_BURN_IN`, and publishes
through the infrastructure adapter. The caller cannot supply that flag.

The committed base worker publishes before its existing metrics commit. This
ordering is deliberate: if the metrics commit fails, the ledger retry is an
idempotent no-op; publishing after a metrics commit could make a timestamp look
complete while its reusable predictions were absent.

The dirty live worker was inspected read-only. It retains the same prediction
columns and authority keys but now scores timestamp batches and calls
`commit_prediction_batches`. The shared publisher therefore supports both a
single cross-section and `publish_batches`; current Mac/Dell callers need only
insert the batch publication immediately before the existing batch metrics
commit. No adaptive-capacity or reservation implementation was copied.

## Bounded real-authority validation

Validation opened only the live random-forest `pending_scores_v3.parquet` and
`resume_state.json`, selected the complete real eligible cross-section at
`2017-10-24T18:05:00Z`, and wrote to an ephemeral namespace inside the isolated
worktree. It did not open a target-value dataset or the true holdout.

- all 487 production-eligible rows were persisted;
- raw scores and production `score_rank` matched exactly;
- exact percentiles and deterministic order matched;
- 483 tied-score rows exercised the declared asset-ID tie break;
- provenance hashes matched the live state;
- the pre-T0 rows remained persisted with `evaluation_eligible=false`;
- the exact boundary was false at 14:30 and true at 14:35 UTC;
- retry added zero parts and produced zero duplicate keys;
- top-five and bottom-five queries each returned five rows;
- no target/return value column was present.

Machine-readable evidence is in `BOUNDED_REAL_VALIDATION.json`.

## Historical recovery classification

The requested classification is campaign-wide, so the small recoverable tail
does not upgrade an incomplete full-history recovery:

| Campaign | Classification | Evidence and estimated work |
| --- | --- | --- |
| Dell current tournament | **D — `MODEL_RETRAIN_REQUIRED`** | `pending_scores_v3.parquet` retains exact raw predictions only for the unresolved tail. Resolved per-T metrics/top selections are not a full eligible-universe ledger, while checkpoint retention keeps only recent models. Reconstructing all completed history therefore requires refitting the retired packages and rescoring their certified input windows. At the audit point this covered 87 completed RF refits / 28,757 scored timestamps, with work growing as the live run progresses. |
| Mac LightGBM qualifier | **D — `MODEL_RETRAIN_REQUIRED`** | The accessible Dell Mac-export package contains certified input authorities and contracts but no transferred LightGBM resume state, pending-score file, full score manifest, prediction batch, or historical model-checkpoint series. Metrics-only retention cannot recover full raw cross-sections, and one latest checkpoint could not cover 51 historical refits. Rebuild scope is the 51 refit packages / 16,494 scored timestamps through `2017-12-29T20:00:00Z`, after ledger persistence is installed. |

The legacy Rank-XENDCG OOF partitions identified by the prior forensic audit
belong to a different producer cadence and do not recover this five-minute Mac
qualifier. No reconstruction, rescoring, or retraining was run.

## Validation buckets and dependency review

- A — new feature tests: 20/20 passed (ranking, ties, percentiles, full
  universe, policy boundary, publisher, immutable Parquet, locking,
  idempotence, duplicate rejection, crash recovery and read-only queries).
- B — runnable adjacent DS24 tests: 42/42 passed across cost-aware policy,
  cost-aware reconstruction and forward-metrics capability.
- C — seven modules cannot collect from base commit `78e1bd9` because source
  present only in the dirty live checkout is absent:
  `test_ds24_low_usage_lightgbm_ranking_readiness.py`,
  `test_ds24_clean_v2.py`, `test_ds24_v3_sequence_policy_worker.py`,
  `test_ds24_v3_lightgbm_ranking_policy_worker.py`,
  `test_ds24_r42_forward_metrics_and_ready_queue.py`,
  `test_ds24_p8_r14_e3g_c2_r7_r44a_disk_capacity.py`, and
  `test_ds24_p8_r14_e3g_c2_r7_r40_activation.py`.

The broader runnable remainder previously produced 201 passes / 54 failures;
the failures require absent generated authorities or pre-existing R49/R50
fixtures and do not import the new modules. They are environment gaps, not
feature regressions. The architecture checker itself is absent from the base.
An AST import audit found zero layer violations and zero cycles: core owns pure
ranking/burn-in/query rules; infrastructure depends downward on core for
filesystem persistence; application depends on both to orchestrate; the script
is only the composition root.

## Cross-host handoff

Mac should cherry-pick the two shared commits containing:

- `core/research/ml/ds24/prediction_ledger_contract.py`;
- `core/research/ml/ds24/prediction_ledger_queries.py`;
- `core/research/ml/ds24/research_output.py`;
- `application/services/ds24_prediction_ledger_service.py`;
- `infrastructure/data/ds24_prediction_ledger.py`;
- `config/ds24_clean_v2/research_output_policy_v2.json`;
- `core/research/ml/ds24/evaluation_burn_in.py`;
- `config/ds24_clean_v2/evaluation_burn_in_v1.json`;
- their focused tests and this handoff.

The Dell adapter commit changes only
`scripts/local/ds24_clean_v2_family_worker.py`; Mac must not cherry-pick that
commit because its worker contains Mac-specific batching/resource work. Mac
should instead instantiate the shared publisher and call `publish_batches`
immediately before `commit_prediction_batches`, preserving its existing
resource-policy files. Mac must not merge Dell adaptive-capacity, supervisor,
reservation, admission, resume-state, or resource-policy modules for this
feature.
