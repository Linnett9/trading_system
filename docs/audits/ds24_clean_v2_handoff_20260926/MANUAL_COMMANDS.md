# DS24 CLEAN V2 Manual Terminal Commands

Classification before these commands:

`DS24_CLEAN_V2_IMPLEMENTATION_READY_MANUAL_DATA_BUILD_REQUIRED`

Run A through J in the same Windows PowerShell session on the Dell. Commands A and C are resumable and safe to repeat against the already-complete manifests. Command E deliberately performs the full materialized-authority scan and may run for a long time. Do not run F or G unless E exits successfully.

Frozen identities:

- source implementation commit: `dfa9aec45f0dc3f97d02c1ad7dbdfc7563e7779c`
- clean source hash: `fa3b8fe8f6769d07b578d292cc583f28331a65813f8f1f8d1d1586aeae841619`
- static authority bundle: `6fa25f60c3685cbd278bfd505fe9623b7079eefc6fad538216d82e64b4f2e87d`
- feature authority: `ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d`
- target authority: `41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1`
- refit policy: `REFIT_EVERY_5_TRADING_SESSIONS_V1`

## A. Resume/build V2 feature sidecar

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_build_sidecar.py --build --workers 2
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 feature-sidecar build failed.' }
```

## B. Monitor sidecar progress

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_build_sidecar.py --status
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 feature-sidecar status failed.' }
```

The terminal state must report `5,654 / 5,654`, `manifest_complete: true`, and logical SHA-256 `ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d`.

## C. Resume/build target delta

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_build_target_delta.py --build --workers 2
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 target-delta build failed.' }
```

## D. Monitor target progress

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_build_target_delta.py --status
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 target-delta status failed.' }
```

The terminal state must report `514 / 514`, `manifest_complete: true`, and logical SHA-256 `41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1`.

## E. Run final causality/data certification

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_certify.py --full-materialized-authority
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 full causality/data certification failed. Do not train.' }
```

This must end with `passed: true`, `certification_scope: FULL_MATERIALIZED_AUTHORITY`, and `101 / 101 FEATURES CAUSAL UNDER FUTURE-BAR PERTURBATION`.

## F. Inspect clean tournament preflight and publish manual admission

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --preflight --write-admission
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell preflight failed. Do not launch.' }
```

Proceed only when the output classification is `DS24_CLEAN_V2_READY_FOR_MANUAL_TOURNAMENT_LAUNCH`, `ready` is `true`, and `blocking_reasons` is empty.

## G. Launch Dell clean supervisor

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --launch
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell supervisor launch failed.' }
```

## H. Monitor Dell clean tournament

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_monitor.py --host dell
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell monitor failed.' }
```

## I. Stop Dell clean tournament safely

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --stop
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell safe-stop request failed.' }
python .\scripts\local\ds24_clean_v2_monitor.py --host dell
```

Wait until H reports `DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE` before using J.

## J. Resume Dell clean tournament

```powershell
Set-Location -LiteralPath 'C:\Users\Brandon\trading_system'
python .\scripts\local\ds24_clean_v2_supervisor.py --host dell --resume
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Dell supervisor resume failed.' }
```

## K. Mac preflight

Run this from Windows PowerShell after the repository, completed V2 data authorities, and passing certificate are present at the Mac repository path.

```powershell
$MacHost = Read-Host 'Mac SSH host (for example user@hostname)'
$MacRepo = Read-Host 'Absolute Mac trading_system path'
$ExpectedSourceCommit = 'dfa9aec45f0dc3f97d02c1ad7dbdfc7563e7779c'
$ExpectedSourceHash = 'fa3b8fe8f6769d07b578d292cc583f28331a65813f8f1f8d1d1586aeae841619'
$ExpectedBundleHash = '6fa25f60c3685cbd278bfd505fe9623b7079eefc6fad538216d82e64b4f2e87d'
$ExpectedFeatureHash = 'ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d'
$ExpectedTargetHash = '41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1'
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_mac_preflight.py --expected-source-commit '$ExpectedSourceCommit' --expected-source-hash '$ExpectedSourceHash' --expected-bundle-hash '$ExpectedBundleHash' --expected-feature-hash '$ExpectedFeatureHash' --expected-target-authority-hash '$ExpectedTargetHash'"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac preflight failed. Do not launch the Mac queue.' }
```

Proceed only when the Mac output has `ready: true`, no blocking reasons, and a `manual_admission_path`.

## L. Launch Mac clean tournament

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_supervisor.py --host mac --launch"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac supervisor launch failed.' }
```

## M. Monitor Mac clean tournament

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_monitor.py --host mac"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac monitor failed.' }
```

## N. Mac safe stop/resume

Safe stop:

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_supervisor.py --host mac --stop"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac safe-stop request failed.' }
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_monitor.py --host mac"
```

Wait until M reports `DS24_CLEAN_V2_TOURNAMENT_STOPPED_RESUMABLE`, then resume:

```powershell
ssh $MacHost "cd -- '$MacRepo' && python3 scripts/local/ds24_clean_v2_supervisor.py --host mac --resume"
if ($LASTEXITCODE -ne 0) { throw 'DS24 V2 Mac supervisor resume failed.' }
```

Neither supervisor can submit PAPER or LIVE orders. A failed or stale admission, changed source/config/data hash, incomplete manifest, active obsolete DS24 process, failed certificate, or insufficient disk blocks worker launch.
