# OPS-4A — Memory-Safe Streaming Daily-Spine Verification

## Root cause

Selector parent run `20260716T091011Z` failed in stage 4 while
`read_stock_level_artifact` loaded a complete Parquet table and called
`table.to_pylist()`. Converting every feature and target value into Python
objects exhausted available memory.

The legacy reader remains unchanged for existing bounded callers. Stage 4
(`--verify-only`) now uses `ParquetFile.iter_batches`.

## Streaming path

The verifier performs:

1. a symbol-only bounded pass for canonical registry resolution;
2. a projected bounded pass over base and enriched artifacts;
3. incremental row validation and augmentation;
4. exact duplicate, population and alignment checks in temporary SQLite;
5. incremental SHA-256 calculation over canonically sorted row IDs;
6. automatic temporary-database cleanup.

No whole PyArrow table is constructed and no complete artifact is passed to
`to_pylist()`. Conversion occurs only on each bounded record batch. PyArrow
threading is disabled for this operational path.

The default batch size is 65,536 rows and can be changed using
`--stream-batch-size`. `--stream-temp-root` may select the parent directory for
the automatically cleaned SQLite resource.

## Projection

Only fields needed for existing stage-4 semantics are requested when present:

- source row ID, symbol and rebalance/session date;
- decision and feature-cutoff timestamps;
- target horizon and maturity timestamps;
- realised stock, benchmark and residual targets;
- source-dataset and calendar identities.

Feature matrices, predictions and unrelated diagnostic columns are not read.
The original full schema is still inspected from Parquet metadata so unknown
columns and schema evidence remain visible.

## Exact validation and checksums

Temporary SQLite retains compact identity and comparison fields, not feature
rows. Indexed queries preserve exact duplicate stock/date/horizon detection,
base/enriched set equality, duplicate row IDs, target/benchmark/timestamp
comparison and bounded mismatch samples.

Dataset IDs use the same canonical sorted row-ID JSON representation as the
legacy implementation. Focused fixtures prove streaming and legacy IDs and
row-population checksums are identical. Physical base/enriched row order may
differ, matching the existing order-independent identity contract.

Reports retain `daily_stock_spine_verification.v1`, `READY`/`BLOCKED`, and CLI
exit codes 0/2. `streaming_diagnostics` adds projected columns, row groups,
batches, rows, maximum batch rows, configured batch size, temporary bytes,
single-worker policy, and explicit `whole_table_to_pylist_used: false`.

## Expected memory behavior

Python row dictionaries are bounded to one record batch. Exact retained state
is stored in SQLite on disk, so process memory scales primarily with the batch
size and the small symbol/date summary sets rather than total artifact width and
row count.

## Production resume

The existing failed run state requires the explicit `-Resume` switch:

```powershell
powershell `
  -NoProfile `
  -ExecutionPolicy Bypass `
  -File scripts/selector_parent_publication_runbook.ps1 `
  -Resume `
  -FromStage 4 `
  -ThroughStage 10 `
  -RunId 20260716T091011Z
```

The command shown without `-Resume` is not valid because the run ID already has
state. The default transcript remains the same run-owned `transcript.txt` and is
opened with append semantics; a different transcript may be supplied explicitly
if operational policy prefers it.

Stage 4 is resumable from `failed`, increments its attempt count, and overwrites
its run-owned report files safely. Verify-only stage 4 creates no canonical
spine or feature outputs, so no partial publication cleanup is required.

## Rollback

Rollback consists of removing the verify-only streaming branch and batch
iterator. The compatible legacy reader and stage-5 publication behavior were
not changed.
