# DS24 RF immutable-ledger forensic decision

## Decision

**Classification:** `RF_IMMUTABLE_LEDGER_GENUINE_SCIENTIFIC_CONFLICT`

The failed publication must remain failed closed. The existing ledger part was
not changed, no recovery source was installed, and no DS24 process was
restarted.

The existing rows and a read-only reconstruction have identical keys, ranks,
eligibility, model vintage, refit timestamp, and feature/target/static/run
authorities. They do not have byte-identical `raw_prediction` values. The
preserved failure recorded 480 conflicting rows at
`2019-04-08T13:40:00Z`. Repeated reconstruction from the hash-verified saved
model produced the same 482-key population but a scheduling-dependent number
of non-identical floats (476 when scored alone, 470 in one reconstruction of
the 78-timestamp production batch, and zero in a later isolated verification
of that same batch). The maximum observed absolute delta was
`3.469446951953614e-18`; ranks were unchanged. The ability to alternate between
byte-exact and last-bit-different output without an authority or artifact
identity change confirms nondeterministic parallel floating-point aggregation;
it cannot retroactively make the preserved 480-conflict publication exact.

This is consistent with non-associative floating-point aggregation during
parallel Random Forest prediction (`n_jobs=3`). It is not eligible for the
repository's existing exact-replay rule: `raw_prediction` is an immutable
scientific ledger field, and no pre-existing contract permits a prediction
tolerance or canonical rounding. Accepting these rows would broaden the
scientific identity boundary prohibited by the ticket.

## Frozen evidence

The pre-change evidence is under `frozen_evidence/`. Every file in that
directory has its Windows read-only attribute set. It includes the supervisor
terminal state, admission, RF failure/exit/resume/model checkpoint,
checkpoint-compatibility evidence, ledger manifest, the exact involved part,
resource state, and all governing authorities.

Authoritative identities:

- active R4 normalized source: `46088c818989f4af3637fb36cfa738daf47f999c2e3164834c49aa40ec22bc08`
- model artifact: `a2f3a515579d3ababc8b9db9dc3c8be5ad93a7f9f1a4af60916fbc1f91045512`
- model vintage/refit ID: `c7ac67d60f66833a514d2b2846e00a05035feb110c392fc0b92a120fbf3158cf`
- feature authority: `ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d`
- target authority: `41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1`
- static authority: `4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc`
- target contract: `8e2d5458044c17cf60bc1d8e46b71357599f3837ddb53d64d9b00d322d17f419`
- refit/training cutoff: `2019-04-08T13:35:00Z`

## Row comparison

| Measure | Result |
|---|---:|
| Existing rows | 482 |
| Incoming reconstructed rows | 482 |
| Intersecting keys | 482 |
| Existing-only keys | 0 |
| Incoming-only keys | 0 |
| Exact payload matches in preserved failure | 2 (inferred from 482 equal keys and 480 reported conflicts) |
| Conflicting payloads in preserved failure | 480 |
| Full-batch reconstruction conflict range | 0 to 470 |
| Max observed absolute prediction delta | `3.469446951953614e-18` |
| Max observed relative prediction delta | `4.556656170715778e-16` |
| Rank disagreements | 0 |
| Eligibility disagreements | 0 |
| Model/refit-ID disagreements | 0 |
| Preprocessing-identity disagreements | 0 |
| Training-cutoff disagreements | 0 |
| Target/authority disagreements | 0 |

The failed incoming payload was not durably stored, so its exact maximum delta
cannot be recovered. The exception preserves its conflict count and first
three keys. The reconstruction report preserves bounded field-level examples
and hashes of the model, cache, and ledger part.

Existing part:

`research_runs/ds24_clean_v2/DS24_CLEAN_V2_TOURNAMENT_R1_20260926/prediction_ledger_v1/family=random_forest/period=2019-04/part-20190408T133500Z-20190408T190000Z-ffcee8b95df79c60.parquet`

- file SHA-256 before/after: `d5d2e304f56b0c1eb8b77b913069134219e4dcec6144f8e65d9919f7c898ebf5` / unchanged
- logical SHA-256: `ffcee8b95df79c609aad14be26920646cb605edbabdef5d6928163e36eb5bec0`
- part rows: 32,008
- duplicate keys in part: 0

## Ledger behavior and validation

The current ledger boundary already provides strict idempotency:

- exact scientific rows are no-ops and preserve existing part hashes;
- exact partial overlap is retained and only new rows are appended;
- any different scientific row, including a different prediction, fails
  closed;
- operational `source_hash` changes alone do not manufacture a scientific
  conflict.

No tolerance, rounding, last-write-wins behavior, deletion, manifest edit, or
ledger rewrite was added. The existing historical operational-provenance
repair remains intact.

Validation completed:

- prediction-ledger contract, recovery, and checkpoint compatibility tests:
  **34 passed**;
- involved part duplicate-key scan: **0 duplicates**;
- reconstruction used the hash-verified saved model and current-source shared
  panel cache without publishing;
- no production source files were changed by this forensic ticket.

## Stopped-state handoff

- supervisor classification: `DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED`
- generation: `1791383785450381300`
- live supervisor/workers: none (the persisted PID is historical)
- admitted family scope remains exactly RF + Huber + Elastic C5
- RF: cursor 52,571; 160 completed refits; `FAILED_CLOSED`; metrics/evaluator
  last completed timestamp `2019-04-05T19:00:00Z`
- Huber: cursor 50,591; 154 completed refits; persisted state `RUNNING`, but no
  live process; metrics last completed timestamp `2019-02-22T20:00:00Z`
- Elastic C5: cursor 6,600; 20 completed refits;
  `DEFERRED_RESOURCE_CAPACITY`; metrics last completed timestamp
  `2017-05-04T19:00:00Z`
- RF v3 evaluator duplicate count: 0
- free C: 7.28 GiB at the frozen failure snapshot; subsequent stopped-state
  checks reported 19.94-22.00 GiB free while unrelated workloads continued.
  The system-managed `C:\pagefile.sys`
  remained allocated at
  28 GiB after the stopped workers' earlier commit-pressure peak (current
  pagefile use was only about 0.6 GiB). It was not resized because doing so is
  a host-level change that commonly requires a reboot and is outside this
  forensic recovery. No additional repository artifact was safely eligible
  for deletion under the ticket's protections.
- available RAM: 18.76 GiB at the final recheck; no PowerShell background Job
  and no DS24 process

No recovery candidate commit/source exists because Classification A was not
proven. The active source remains the qualified R4 source above. The qualified
rollback source remains
`6313ffa04a359a88129ef6c4db7360be4ce35f30d6408c3b39a64c17877f7513`.

Monitor command:

```powershell
python scripts/local/ds24_clean_v2_monitor.py --host dell
```

Safe-stop command (if a future supervisor is started):

```powershell
python scripts/local/ds24_clean_v2_supervisor.py --host dell --stop
```
