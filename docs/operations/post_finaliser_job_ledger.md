# Post-Finaliser Operational Job Ledger

## Purpose

`post_finaliser_job_ledger.v1` records the critical path from the active Alpaca
five-minute archive finaliser through selector publication, strict-OOS component
production, matched evaluation, ranking-tree comparison and the Wave 6 decision.

The ledger is operational scheduling evidence. It does not claim that synthetic
model foundations are historically validated or promoted.

## Current dependency graph

```text
JOB-001 five-minute finalisation
  -> JOB-002 full archive validation
    -> JOB-003 selector stage-4 resource preflight
      -> JOB-004 resume selector stages 4-10
        -> JOB-005 produce 15 strict-OOS components
          -> JOB-006 validate and freeze panel
            -> JOB-007 base matched evaluation
              -> JOB-008 add bounded challengers
                -> JOB-009 multi-regime comparison
                  -> JOB-010 statistical/promotion gate
                    -> JOB-011 Wave 6 decision
```

No downstream job becomes ready while its immediate dependency is incomplete.

## Five-minute archive gate

The authoritative completion target is:

- 5,654 planned partitions;
- 5,654 completed partitions;
- zero pending or failed partitions;
- zero invalid rows or conflicting duplicates;
- no temporary artifacts.

Archive validation must then report `valid=true`, exactly 5,654 partitions,
zero invalid rows and no temporary files. The earlier 660-partition validation
is explicitly classified as partial and cannot satisfy JOB-002.

JOB-001 is `EXCLUSIVE_MACHINE`. JOB-002 is conservatively classified
`HIGH_MEMORY` based on the existing validator's complete archive traversal.

## Selector same-run resume

Run `20260716T091011Z` must resume rather than restart. JOB-003 checks:

- JOB-001 and JOB-002 completion;
- no active finaliser process;
- OPS-4A readiness;
- compatible run-state schema and exact run ID;
- stages 1-3 complete;
- stage 4 failed or pending;
- free physical memory and disk above ledger policy thresholds.

The default thresholds are 8 GiB free physical memory and 20 GiB free disk.
They are conservative operational policy, not universal or mathematical
requirements, and may be changed in a reviewed ledger revision.

The rendered command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/selector_parent_publication_runbook.ps1 `
  -Resume `
  -FromStage 4 `
  -ThroughStage 10 `
  -RunId 20260716T091011Z `
  -TranscriptPath reports/ml/readiness/selector_evaluation_1c_e/runs/20260716T091011Z/transcript_resume_stage4_10.txt
```

The checker only renders this command. It never executes it.

## Component and evaluation path

JOB-005 requires exactly 15 components:

```text
5 dates × {ridge, elastic_net, ordered_logit_ranker}
```

Each component must have complete finite predictions, deterministic ranks,
strict label cutoff and complete model, feature, target, dataset and registry
identity.

JOB-007 proves the matched evaluation pipeline using momentum and the three
base models across daily top-k, ten-session cohorts, rank hysteresis and
turnover-penalised aim policies. Cost scenarios are 5/10/25/50 bps. Capacity is
1.0/2.5/5.0 percent ADV or explicitly `UNVERIFIED`.

Only after that base pipeline works may JOB-008 add Huber, Contextual Elastic
Net, eligible multi-horizon linear selectors, Rank-XENDCG and LambdaRank with
hypothesis, campaign, configuration, seed and effective-search accounting.

## Implementation history

The ledger distinguishes:

- `IMPLEMENTED_SYNTHETICALLY`;
- `INTEGRATED`;
- `HISTORICALLY_VALIDATED`;
- `PROMOTED`.

Ranking-label contracts, selector challengers, safeguards, risk baselines and
most Wave 4/5 owners are recorded as synthetic foundations. LightGBM dependency
availability and the existing registry foundation may be integrated, but no
ranking GBDT is described as historically validated or promoted.

## Readiness checker

Typical usage:

```powershell
python scripts/check_post_finaliser_job_readiness.py `
  --ledger config/operations/post_finaliser_job_ledger_v1.json `
  --selector-run-id 20260716T091011Z `
  --summary
```

Other views:

```powershell
python scripts/check_post_finaliser_job_readiness.py --next-job
python scripts/check_post_finaliser_job_readiness.py --job JOB-003
python scripts/check_post_finaliser_job_readiness.py --json
```

The checker opens only configured JSON state, progress, validation and readiness
files. It uses Windows process metadata, physical-memory metadata and filesystem
free-space metadata. It imports no PyArrow package, reads no Parquet file,
scans no archive tree and executes no workflow.

Missing and malformed files, failed partitions, partial validation, active
finaliser state, incompatible selector identity, invalid OPS-4A evidence, low
resources, blocked stage 10 and incomplete component rosters produce explicit
blockers rather than inferred readiness.

## Next actions

- JOB-001 active or incomplete: allow the existing finaliser to complete or
  resume it under its existing operational procedure.
- JOB-002 ready: schedule full archive validation during an appropriate
  resource window.
- JOB-003 ready: review the reported resource evidence and exact resume command.
- JOB-004 ready: execute only after human operational approval.
- JOB-005 ready: produce the fixed 15-component roster.
- JOB-006 onward: follow the ledger dependency chain without skipping gates.
