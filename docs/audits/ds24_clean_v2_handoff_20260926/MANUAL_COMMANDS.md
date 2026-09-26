# DS24 CLEAN V2 Manual Terminal Commands

Current classification:

`DS24_CLEAN_V2_READER_REPAIRED_MANUAL_ADMISSION_AND_RESUME_REQUIRED`

The V2 feature sidecar and target extension are complete and unchanged. Do not run either builder with `--build`; do not edit or republish completed partitions. The existing full materialized certificate remains valid. A bounded reader preflight, a source-bound replacement admission, and retry/resume remain manual terminal actions.

Frozen identities:

- source implementation commit: `5093782de4a2542f1993e45b6c11dc104a786d2f`
- clean source hash: `00b56411c667a196cca6a22c682a89c9282b12baeb5a23557edc9b453c175f3a`
- static authority bundle: `4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc`
- model-registry file / logical hash: `e57926407bb34f253c49d1c5a8541f1f233b4d1770e27bdab1f406b8e2045742` / `0421b56cc534a5cd86f15e8a208929cf5ff2205a68069bcca71fa49e626f49b5`
- tournament-contract file / logical hash: `15cf8aec7b1ef81e6682c0c82886255f0e80ebe1acdb2abf6d618ae4b1fcd5bb` / `c8307036d89fed26d6081838414d6a5e6a29d7101bfa8ef4080d7b75a8f6ec53`
- feature authority: `ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d`
- target authority: `41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1`
- refit policy: `REFIT_EVERY_5_TRADING_SESSIONS_V1`

## First-package sidecar incident resolution

The first `random_forest` package refits at `2016-02-02T14:35:00Z` from 20 training sessions and scores the five sessions from 2016-02-02 through 2016-02-08. Its bounded AAPL/2016 feature read requests 4,226 base rows.

The retired reader converted exchange `session_date` labels into a half-open UTC-midnight sidecar window. It therefore selected 4,225 sidecar rows and failed `4225 != 4226` after an inner join. The omitted identity was:

- `asset_id=asset_9eeaca47907f9f47`
- `decision_timestamp=2016-02-09T00:00:00Z`
- `session_date=2016-02-08`
- `session_type=after_hours`

That key exists in the immutable sidecar. Its eleven repair values and eleven provenance timestamps are correctly null because extended-hours rows are not admitted to the repaired regular-session formulas. Across the requested package, all 4,226 exact keys match; 2,548 matched rows legitimately have at least one null repair value. Identity columns have no nulls or duplicates and both sides use the same hashed asset ID plus `datetime64[us, UTC]` timestamp precision.

The repaired reader bounds sidecar IO from the selected base timestamps, then performs a one-to-one left join with an explicit presence marker. It validates identity coverage independently from repair missingness, never restores a V1 repair value, and still fails closed for null, duplicate, mismatched, or genuinely absent keys. The materialized sidecar and target data require no repair.

The required one-asset `assemble_sessions` check then exposed a second, downstream reader compatibility failure: all-null legacy breadth-context columns load as object dtype, and current pandas rejects assigning those objects into newly created numeric panel columns. Context predictors are now explicitly coerced to numeric, preserving missing values as `NaN` without filling or changing model-specific preprocessing. The bounded AAPL assembly completes with 1,949 target-eligible regular-session rows.

The preserved `random_forest` and `transformer` failures are now reconciled to `FAILED_CLOSED`, retaining the original exception, timestamp, log path, completed-refit list, and metrics cursor. Retry attempts receive a new host/run/attempt generation; stale failures cannot override a newer running or completed attempt. The monitor reads only `supervisor_status_<host>.json` and filters current ownership, so the older generic status cannot reintroduce `iTransformer` or another retired family.

## Reconciled XENDCG authority

The prior `24 / 0.08 / 7 / 2` entry was traced to the generic daily stock-level `fixed_rank_xendcg_configuration`; it was not the recovered DS24 Mac producer and is rejected for this tournament.

The sole carry-forward `lightgbm_rank_xendcg` candidate now ports frozen producer function `core.research.ml.ds24.mac_aux_queue_r44f2._fit_lightgbm`, source SHA-256 `5c5e8ff1486d870f8c2d3b1ceb2c406f5544e38a34e5a768a816abc06076b17d`. Its model parameters are `n_estimators=25`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=10`, `random_state=1729`, and four LightGBM threads.

The preserved producer semantics are:

- stable ordering by decision timestamp and asset;
- a row-level `tail(24000)` cap, which may leave the oldest query group partial;
- `+/-inf -> NaN -> 0.0` preprocessing with no fitted imputer;
- one query group per decision timestamp;
- within-timestamp `rank(method="first", pct=True)` labels transformed by `floor(rank * 4.999)` into relevance values 0 through 4.

The CLEAN V2 qualifier scores this model on every registered five-minute timestamp in the frozen qualifier years. That wrapper creates new clean evidence; it does not reuse the producer's old once-per-session OOF results. No second XENDCG candidate exists.

## Concrete tournament budget

The registered SPY decision spine contains 2,695 sessions and 177,072 timestamps from `2016-01-04T14:35:00Z` through `2026-09-23T19:00:00Z`. The qualifier dates remain exactly 2017, 2019, 2020, 2022, and 2024: five complete calendar-year surfaces totalling 1,259 sessions, 82,669 timestamps, and 255 refit packages. This is not a smoke test.

`none` means no configured sample cap. All fitted candidates use `REFIT_EVERY_5_TRADING_SESSIONS_V1`; controls use the same five-session scoring packages but perform no fit.

| Candidate | Host | Lane | Lookback sessions | Scoring sessions | Scoring timestamps | Expected fits | Sample cap | Refit schedule |
|---|---|---|---:|---:|---:|---:|---|---|
| random_forest | dell | FULL_CLEAN | 20 | 2,675 | 175,752 | 535 | none | every 5 sessions |
| transformer | dell | FULL_CLEAN | 20 | 2,675 | 175,752 | 535 | 24,000 training examples | every 5 sessions |
| huber | dell | FULL_CLEAN | 20 | 2,675 | 175,752 | 535 | none | every 5 sessions |
| ridge_C5 | dell | FULL_CLEAN | 20 | 2,675 | 175,752 | 535 | none | every 5 sessions |
| gradient_boosting_C0 | dell | FULL_CLEAN | 20 | 2,675 | 175,752 | 535 | none | every 5 sessions |
| elastic_net_C5 | dell | SHORT_REQUALIFICATION | 20 | 1,259 | 82,669 | 255 | none | every 5 sessions; restart by qualifier year |
| elastic_net_C6 | dell | SHORT_REQUALIFICATION | 20 | 1,259 | 82,669 | 255 | none | every 5 sessions; restart by qualifier year |
| gradient_boosting_C0_W20 | dell | SHORT_REQUALIFICATION | 20 | 1,259 | 82,669 | 255 | none | every 5 sessions; restart by qualifier year |
| gradient_boosting_C0_W40 | dell | SHORT_REQUALIFICATION | 40 | 1,259 | 82,669 | 255 | none | every 5 sessions; restart by qualifier year |
| gradient_boosting_C0_W80 | dell | SHORT_REQUALIFICATION | 80 | 1,259 | 82,669 | 255 | none | every 5 sessions; restart by qualifier year |
| lightgbm_rank_xendcg | mac | SHORT_REQUALIFICATION | 20 | 1,259 | 82,669 | 255 | 24,000 training rows | every 5 sessions; restart by qualifier year |
| momentum_transformer | mac | UNSCORED_DISCOVERY | 20 | 1,259 | 82,669 | 255 | 24,000 training examples | every 5 sessions; restart by qualifier year |
| market_context_encoder | mac | UNSCORED_DISCOVERY | 20 | 1,259 | 82,669 | 255 | 24,000 training examples | every 5 sessions; restart by qualifier year |
| temporal_fusion_transformer | mac | UNSCORED_DISCOVERY | 20 | 1,259 | 82,669 | 255 | 24,000 training examples | every 5 sessions; restart by qualifier year |
| momentum | dell | CONTROLS | 20 | 2,675 | 175,752 | 0 | not applicable | five-session score packages; no fit |
| equal_weight_no_model | dell | CONTROLS | 20 | 2,675 | 175,752 | 0 | not applicable | five-session score packages; no fit |

`iTransformer` remains retired source-only: no registry family, lane, host ownership, command, or automatic continuation.

## Gradient Boosting C0 / C0_W20 reuse decision

Reuse is not allowed. The two entries share the data authority, lookback, estimator parameters, preprocessing, and seed, but the full schedule continues its five-session phase across calendar-year boundaries while the qualifier schedule restarts inside each frozen qualifier year. Only 51 of 255 qualifier refit boundaries align; 1,008 of 1,259 qualifier sessions therefore use a different fit boundary and different training rows. Family identity and policy hash also make the evaluation keys different. Both executions are required.

## 1. Optional one-shot completion check

These commands are status-only. Do not add `--build`.

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_build_sidecar.py --status
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 feature-sidecar status failed.' }
python .\scripts\local\ds24_clean_v2_build_target_delta.py --status
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 target-delta status failed.' }
```

The reports must remain `5,654 / 5,654` and `514 / 514`, with logical hashes equal to the frozen feature and target authorities above.

## 2. Run the bounded production-reader preflight

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_reader_preflight.py --family random_forest --asset AAPL --year 2016
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 bounded production-reader preflight failed. Do not resume.' }
```

Require `DS24_CLEAN_V2_BOUNDED_READER_PREFLIGHT_PASS`, `production_reader_rows: 4226`, `production_assembly_rows: 1949`, `production_assembly_assets: ["AAPL"]`, exact key coverage, `model_fit_performed: false`, and `data_written: false`.

## 3. Preserve the existing full certificate

Do not repeat full certification for this source-only reader/control-plane repair. The existing certificate remains internally valid and records `passed: true`, `certification_scope: FULL_MATERIALIZED_AUTHORITY`, `101 / 101 FEATURES CAUSAL UNDER FUTURE-BAR PERTURBATION`, feature authority `ac2be1f9...`, target authority `41dd1fd3...`, and the unchanged static bundle `4952431d...`. It does not claim the reader source hash, and no certified data, formula, predictor, target, or static contract changed. Do not edit the certificate.

## 4. Publish a replacement Dell manual admission

The pre-fix admission is stale because it does not carry the current clean source hash. Do not proceed unless step 2 passed and this command writes a new admission.

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --preflight --write-admission
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell preflight failed. Do not launch.' }
```

Require classification `DS24_CLEAN_V2_READY_FOR_MANUAL_TOURNAMENT_LAUNCH`, `ready: true`, and no blocking reasons.

## 5. Resume Dell

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --resume
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell supervisor resume failed.' }
```

One-shot status:

```powershell
python .\scripts\local\ds24_clean_v2_monitor.py --host dell
```

Safe stop and resume:

```powershell
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --stop
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell safe-stop request failed.' }
python .\scripts\local\ds24_clean_v2_monitor.py --host dell
# Resume only after the monitor reports DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE.
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --resume
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell supervisor resume failed.' }
```

## 6. Mac preflight and launch

Run after this commit, the completed V2 data authorities, and the passing certificate are present at the Mac repository path.

```powershell
$MacHost = Read-Host 'Mac SSH host (for example user@hostname)'
$MacRepo = Read-Host 'Absolute Mac trading_system path'
$ExpectedSourceCommit = '5093782de4a2542f1993e45b6c11dc104a786d2f'
$ExpectedSourceHash = '00b56411c667a196cca6a22c682a89c9282b12baeb5a23557edc9b453c175f3a'
$ExpectedBundleHash = '4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc'
$ExpectedFeatureHash = 'ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d'
$ExpectedTargetHash = '41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1'
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_mac_preflight.py --expected-source-commit '$ExpectedSourceCommit' --expected-source-hash '$ExpectedSourceHash' --expected-bundle-hash '$ExpectedBundleHash' --expected-feature-hash '$ExpectedFeatureHash' --expected-target-authority-hash '$ExpectedTargetHash'"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac preflight failed. Do not launch the Mac queue.' }
```

Proceed only when the Mac output has `ready: true`, no blocking reasons, and a `manual_admission_path`.

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_supervisor.py --host mac --launch"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac supervisor launch failed.' }
```

One-shot status:

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_monitor.py --host mac"
```

Safe stop and resume:

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_supervisor.py --host mac --stop"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac safe-stop request failed.' }
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_monitor.py --host mac"
# Resume only after the monitor reports DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE.
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_supervisor.py --host mac --resume"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac supervisor resume failed.' }
```

Neither supervisor can submit PAPER or LIVE orders. A failed or stale admission, changed source/config/data hash, incomplete manifest, active obsolete DS24 process, failed certificate, or insufficient disk blocks worker launch.
