# DS24 Dell ranked-prediction ledger activation

## Result

The shared `DS24_RANKED_PREDICTION_LEDGER_V1` and
`DS24_EVALUATION_BURN_IN_2016_2017_V1` authorities were activated on the
current Dell CLEAN V2 source without replacing the newer Dell scheduler or
adaptive-capacity implementation. The existing tournament run ID and every
family resume directory were preserved.

The live source moved from
`f1c1d53928392083c98be39cf04d11727ce31b09c0692382e9a2825a4f09aaf5`
to `fc0d3e68b3971fa9773797cde1b5819d12c3f603888bc953a71fb22ffae5a68d`.
The integration commit is
`2c8bd2ef8117c747f8d677ab4097f0ff79f8fde8`.

Random Forest resumed with its existing model checkpoint under the
`RESUME_COMPATIBLE_OPERATIONAL_ONLY_CHANGE` classification. It retained 87
completed refits and cursor 28,889, then published the next 66 timestamps and
advanced to cursor 28,955 through `2017-10-27T19:00:00Z`.

## Publication and burn-in semantics

The shared Dell worker publishes every canonical prediction batch to the
ledger immediately before the existing incremental evaluator commit boundary.
The ledger is therefore the first durable output. Before
`2018-01-02T14:35:00Z`, rows are retained with
`evaluation_eligible=false` and bypass headline Rank IC and return evaluation.
At and after the boundary, only publisher-authorised eligible batches reach
the evaluator. A post-ledger/pre-evaluator crash remains retryable because
only persisted burn-in timestamps are accepted as ledger-only completion.

Historical coverage is explicit per family. Families with prior scores carry
`ledger_history_complete=false` and
`historical_reconstruction=MODEL_RETRAIN_REQUIRED`; no historical completeness
is inferred or fabricated.

## Dell family coverage

Every registered Dell family uses
`scripts/local/ds24_clean_v2_family_worker.py`, produces the canonical batch
schema through `_score_model`, and reaches the single
`DS24PredictionLedgerPublisher.publish_batches` boundary.

| Family | Lane | Shared scoring path | Shared ledger boundary | Enabled |
|---|---|---|---|---|
| random_forest | FULL_CLEAN | canonical worker / `_score_model` | publisher before evaluator | yes |
| transformer | FULL_CLEAN | canonical worker / `_score_model` | publisher before evaluator | yes |
| huber | FULL_CLEAN | canonical worker / `_score_model` | publisher before evaluator | yes |
| ridge_C5 | FULL_CLEAN | canonical worker / `_score_model` | publisher before evaluator | yes |
| gradient_boosting_C0 | FULL_CLEAN | canonical worker / `_score_model` | publisher before evaluator | yes |
| elastic_net_C5 | SHORT_REQUALIFICATION | canonical worker / `_score_model` | publisher before evaluator | yes |
| elastic_net_C6 | SHORT_REQUALIFICATION | canonical worker / `_score_model` | publisher before evaluator | yes |
| gradient_boosting_C0_W20 | SHORT_REQUALIFICATION | canonical worker / `_score_model` | publisher before evaluator | yes |
| gradient_boosting_C0_W40 | SHORT_REQUALIFICATION | canonical worker / `_score_model` | publisher before evaluator | yes |
| gradient_boosting_C0_W80 | SHORT_REQUALIFICATION | canonical worker / `_score_model` | publisher before evaluator | yes |
| momentum | CONTROL | canonical worker / `_score_model` | publisher before evaluator | yes |
| equal_weight_no_model | CONTROL | canonical worker / `_score_model` | publisher before evaluator | yes |

A newly registered family needs no family-specific ledger code when it is
routed through this worker and returns the canonical prediction batch schema.
The regression suite covers a future mock family, Random Forest, and Huber,
including idempotent retry and forbidden target/outcome columns.

## Live validation

The first RF ledger part contains 32,155 rows across 66 complete timestamps,
with 476 to 504 assets per cross-section. Contract validation recomputed exact
rank ordering and percentiles from the stored raw predictions. Duplicate keys
are zero. File and logical content hashes, all provenance hashes, the exact
schema, finite raw scores, and absence of forbidden target/return columns were
verified. The Parquet column chunks report ZSTD compression. All rows are
correctly `evaluation_eligible=false` because the session is before T0.

The 147,430-byte part measures 4.585 compressed bytes per row. Annualising
the full-session file (including part overhead) gives 37,152,360 bytes per
family-year, 371,523,600 bytes for the ten predictive Dell families, or
445,828,320 bytes for all twelve score-producing Dell families including the
two controls. This is a one-session empirical projection and will vary with
universe size and compression cardinality.

## Verification

- Focused ledger and burn-in suite: 24 passed.
- Broad current-Dell suite: 275 passed, 1 deselected, 3 third-party
  deprecation warnings.
- The one deselected test requires the production AAPL dataset, which is
  intentionally absent from the isolated source-only worktree.
- Scoped `compileall`: passed.
- `git diff --check`: passed apart from line-ending notices.
- The repository snapshot does not contain
  `scripts/check_architecture_conformance.py`; manual import inspection found
  no upward dependency from the new core modules into application,
  infrastructure, or scripts.
- The live deployment was hash-compared file by file and the resulting live
  clean-source hash exactly matched the isolated integration hash.

The pre-upgrade recovery evidence is in
`PRE_LEDGER_DELL_ACTIVATION_STATE.json`; exact first-part and runtime evidence
is in `LIVE_VALIDATION.json`.

## Operations

Monitor:

```powershell
python scripts/local/ds24_clean_v2_monitor.py --host dell
```

Governed stop:

```powershell
python scripts/local/ds24_clean_v2_supervisor.py --host dell --stop
```

No Mac source, state, process, or output was changed.
