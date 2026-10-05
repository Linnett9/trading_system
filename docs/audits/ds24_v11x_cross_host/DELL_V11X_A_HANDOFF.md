# DS24 V11X-A Dell Handoff

Classification: `V11X_A_SHARED_RELEASE_READY_MAC_VALIDATION_REQUIRED`

## Dell runtime preserved

- Runtime source hash:
  `6313ffa04a359a88a9172ac1812ca003b7a4f1601fd0a0d5a19e1470663a139c`
- Supervisor PID/generation: `18928` / `1791214672260700200`
- Random Forest snapshot: 124 refits / 41,027 scores /
  `2018-07-24T19:00:00Z`
- Huber snapshot: 54 refits / 17,771 scores / `2017-02-27T20:00:00Z`
- Elastic Net C5 snapshot: 18 refits / 6,006 scores /
  `2017-05-12T19:00:00Z`

The supervisor and exactly those three workers were left running.  No stop,
restart, cursor mutation, production ledger write, or fourth worker occurred.

## Critical-path evidence

The active V1 telemetry does not record complete, disjoint end-to-end package
wall time for enough refits, so production p50/p90/p99 remain
`UNINSTRUMENTED`.  It is not valid to derive them from process liveness.

Direct active-source observations identified:

- RF combined read/panel/join: 143.480 seconds (one sample, then resumed score);
- Huber combined read/panel/join: 941.205 seconds mean (two samples);
- Elastic C5 population slicing: 1,674.250 seconds (one sample).

These combined V1 stages cannot distinguish Parquet read, join, conversion,
sorting, or slice costs.  The released V11X contract provides that separation
for a later safe runtime integration.

## Bounded real-authority benchmarks

All trials used the same first RF package, eight assets, 18 requested score
timestamps, immutable production data, temporary evaluator namespaces, and no
production writes.

| RF threads | Fit seconds | Package seconds | Package reduction | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21.880 | 23.281 | baseline | 347,234,304 |
| 2 | 11.436 | 12.887 | 44.65% | 347,693,056 |
| 3 | 8.066 | 9.546 | 59.00% | 348,995,584 |

For both parallel candidates, keys, panel, ranks, rank IC, sleeve maturity, and
transaction-cost fingerprints were exact.  Maximum raw prediction difference
was `8.673617379884035e-19`, consistent with parallel floating-point reduction
order.  Huber and Elastic bounded baselines were also recorded in
`DELL_BOUNDED_BENCHMARK_SCORECARD.json`.

`n_jobs=3` clears the bounded performance threshold but is not deployed.  The
eight-asset package does not establish full-universe memory/commit behavior or
the effect on concurrent production Huber/Elastic capacity.  Running an
unmanaged full-size benchmark beside the healthy tournament would violate the
resource-safety requirement.

## Release

- Branch: `codex/ds24-v11x-throughput`
- Required base: `55e3e78a8`
- Source commit: `695eda3d6fed75d981de21813ddcee73d740ead9`
- Normalized source hash:
  `e2867279e57349298034737bd37b38e593d89ee59d0fe0ec5fe665ddbb3fc681`

Shared source is the host-neutral profile contract and report generator.
The bounded family benchmark and Dell scorecard are Dell-only.

## Commands

Monitor:

```text
python scripts/local/ds24_clean_v2_monitor.py --host dell --compact
```

Supported safe stop, not executed:

```text
python scripts/local/ds24_clean_v2_supervisor.py --host dell --stop
```

## Residual work

The next Dell step is a governor-admitted full-package RF `n_jobs=1/2/3`
benchmark or a checkpoint-safe one-package canary that measures concurrent
Huber/Elastic availability.  Only then may `n_jobs=3` be admitted for production.
