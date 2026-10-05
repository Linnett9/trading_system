# DS24 V11X Mac Validation Handoff

## Immutable release authority

- Branch: `codex/ds24-v11x-throughput`
- V11X source commit: `695eda3d6fed75d981de21813ddcee73d740ead9`
- Required base: `55e3e78a8` (`Document DS24 Dell ledger activation`)
- Cross-host-normalized CLEAN V2 source hash at the V11X source commit:
  `e2867279e57349298034737bd37b38e593d89ee59d0fe0ec5fe665ddbb3fc681`
- Dell runtime source hash used for the bounded benchmark evidence:
  `6313ffa04a359a88a9172ac1812ca003b7a4f1601fd0a0d5a19e1470663a139c`

The source commit is intentionally separate from this handoff commit.  Mac
validation must check out the exact source commit above, or a descendant whose
additional changes have been independently reviewed.

## Shared files

- `core/research/ml/ds24/cross_host_throughput.py`
  - host-neutral 23-stage package profile contract;
  - explicit `MEASURED`, `UNINSTRUMENTED`, and `NOT_APPLICABLE` states;
  - typed workload and resource counters;
  - authority and host-family ownership isolation;
  - p50/p90/p99 family summaries and fail-closed distributed critical path.
- `scripts/local/ds24_v11x_profile_report.py`
  - read-only seven-lane report generation;
  - missing Mac or Dell lanes remain explicitly uninstrumented.
- `tests/test_ds24_v11x_cross_host_throughput.py`
  - shared contract, ownership, identity, duplicate, reporter, and RF override
    regression coverage.

## Dell-only files

- `scripts/local/ds24_clean_v2_family_benchmark.py`
- `docs/audits/ds24_v11x_cross_host/DELL_BOUNDED_BENCHMARK_SCORECARD.json`

The Mac must not execute the Random Forest benchmark or use its measured
threading result as Mac performance evidence.  The benchmark tool is included
for reproducibility of the Dell result and as a reference for isolated,
read-only package benchmarking.  It does not authorise Random Forest on Mac.

## Mac ownership

Only these CLEAN V2 families are Mac-owned under V11X:

- `lightgbm_rank_xendcg`
- `momentum_transformer`
- `market_context_encoder`
- `temporal_fusion_transformer`

Do not launch Random Forest on Mac and do not import Dell runtime checkpoints.

## Required Mac validation

Run from a clean checkout with the Mac authority installed:

```bash
git fetch origin codex/ds24-v11x-throughput
git switch --detach 695eda3d6fed75d981de21813ddcee73d740ead9
python3 -m pytest tests/test_ds24_v11x_cross_host_throughput.py -q
python3 -m compileall -q \
  core/research/ml/ds24/cross_host_throughput.py \
  scripts/local/ds24_v11x_profile_report.py
python3 /Users/brandonlinnett/Desktop/trading_system/scripts/check_architecture_conformance.py \
  core/research/ml/ds24 scripts/local
```

The required base predates the checked-in architecture guard.  The command
above intentionally invokes the Mac operational repository's trusted current
guard while the working directory remains the V11X checkout.  It must report
zero cycles for the scoped DS24/local-script surface.

Before any Mac runtime integration, also run the established Mac gates:

```bash
python3 scripts/local/ds24_clean_v2_mac_preflight.py \
  --expected-source-commit 695eda3d6fed75d981de21813ddcee73d740ead9 \
  --expected-source-hash e2867279e57349298034737bd37b38e593d89ee59d0fe0ec5fe665ddbb3fc681 \
  --expected-bundle-hash 4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc \
  --expected-feature-hash ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d \
  --expected-target-authority-hash 41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1
```

If the Mac's accepted base has a different normalized source identity, stop and
reconcile the base/source authorities; do not bypass the source-bound gate.

## Parity expectations

Every emitted package profile must:

- contain all 23 stages exactly once;
- bind the Mac host, owned family, run, package, source, and scientific authority;
- mark unavailable timings `UNINSTRUMENTED` rather than infer them;
- reject duplicate package evidence and mixed source/scientific identities;
- use directly observed CPU/RSS/swap/I/O counters where available.

Instrumentation must not change features, targets, PIT/maturity, eligibility,
training windows, refit cadence, sequence/query membership, predictions,
evaluation, publication, or checkpoint semantics.

## Mac benchmarks still required

1. End-to-end package p50/p90/p99 for all four Mac families.
2. RankXENDCG query selection, grouping/sort, matrix, dataset, fit, inference,
   and publication attribution with complete-query parity.
3. Sequence endpoint/index-map and tensor materialisation benchmarks for all
   three sequence families.
4. Bounded batch-size and thread-cap trials with RSS, swap, CPU, thermal, and
   output parity evidence.
5. Mac-specific cache hit/miss/rebuild/eviction evidence.
6. Bounded next-package prefetch contention test.
7. Shared vectorised evaluation/publication parity on Mac authority.

Dell validation is not Mac validation.  No shared optimisation is cross-host
approved until these checks pass locally on the Mac.
