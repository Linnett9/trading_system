# Ticket 56A Immutable Candidate Lineage Preservation

Ticket 56A-P preserves the historical candidate replay evidence for:

- Candidate: `single_model:raw_return_10d__ridge__seed_1729`
- Historical commit: `cf5f69dfb88b100245b7b7b973e3ec6bce25feb0`
- Classification: `ARTIFACT_REPLAY_REPRODUCED_TRAINING_NOT_REPRODUCED`
- Artifact replay classification: `REPRODUCED`
- Training reproduction classification: `NOT_REPRODUCED`
- Preserved and replayed net cumulative return: `0.41308711504230433`
- Absolute replay difference: `0.0`

No model training was rerun. The candidate was not promoted.

## Stable Evidence

The complete generated Ticket 56A replay bundle was copied from:

`C:/tmp/ticket_56a_immutable_candidate_lineage_20260729`

to the existing ignored audit convention:

`reports/audits/ticket_56a_immutable_candidate_lineage_20260729`

The tracked compact manifest is:

`docs/audits/ticket_56a_preserved_evidence_manifest.json`

Because `reports/` is ignored, the tracked manifest is the durable Git reference for the preserved local evidence bundle.

## Canonical Replay Commands

Original command captured by the Ticket 56A environment report:

```powershell
cd C:/tmp/trading_system_ticket_56a_historical_replay_20260729; python C:/tmp/ticket_56a_immutable_candidate_lineage_20260729/ticket_56a_replay.py
```

Equivalent command using the preserved replay script:

```powershell
cd C:/tmp/trading_system_ticket_56a_historical_replay_20260729; python C:/Users/Brandon/trading_system/reports/audits/ticket_56a_immutable_candidate_lineage_20260729/ticket_56a_replay.py
```

If the historical worktree has been removed, recreate it first:

```powershell
git worktree add --detach C:/tmp/trading_system_ticket_56a_historical_replay_20260729 cf5f69dfb88b100245b7b7b973e3ec6bce25feb0
```

## Bundle Hash

Deterministic bundle manifest SHA-256:

`6d29b42d6c4c8bfb7f07511f2c5748d2ad006ba58812ab9ec183b912d8e30255`

Method: SHA-256 over UTF-8 text lines sorted by file name, each line formatted as `name<TAB>length<TAB>sha256<LF>`.

## Bundle Files

| File | Length | SHA-256 |
| --- | ---: | --- |
| `environment_report.json` | 1960 | `64c1440675bf8d1f33b04f87cf4a115ccd90c909a02f4552085f836f2640f245` |
| `environment_report.md` | 244 | `1c49be934d92370def015e8f7df38670df987c69c4dad819d1a5ce79565870c4` |
| `hash_table.csv` | 7270 | `2b450cd3292e4a2fa8dc9b15c3fb03cd20a9370a08e968f50cb2ae1af262a156` |
| `hash_table.json` | 12368 | `ceb1801bf2076fab3d23ac4ebc8727d2452c54af7f8d449d0878375fdfd74c34` |
| `immutable_candidate_manifest.json` | 20867 | `8fcd14ab7a80f0f17d797f69f2589e1f250c88b794ed78a2480fb41bc64dc61b` |
| `missing_lineage_report.json` | 2599 | `3cac7c3eacb38ff15d9f9217722a668a7a1f609e080a7cbf392dc30dd438d79f` |
| `missing_lineage_report.md` | 1002 | `7a15fafbd9f6b6db18d8301ea611757cf9cd63ba95070266f7a511c6adf4ca7a` |
| `replay_comparison_report.json` | 9609 | `9b4898b48d8117416222bdf3bf171447b18754eceb1fbaa480e2c5e823a53ab4` |
| `replay_comparison_report.md` | 843 | `a07207c3bb7db2352e4d348c96e54200a709866cd9804ed06699c0fb661c8df6` |
| `ticket_56a_replay.py` | 33633 | `c988b53018fcba6c1cf7c123d7e4064a7096ebea51e88c5d4e1e21fb2281b003` |

## Evidence Hashes

- Source artifact: `0d99df8363e4ef763bf9ee5418b73904d5e582dcd94e1e9c1feb43496361bba7`
- OOS predictions: `f85d8ccfd59f9c6bc6db815b6de31d28143bb56dcdaa08255ee2c33388dde6cf`
- Promotion evidence: `c0d2fd781cc23b6a9c52bf7d3400af97e7706d7a12851854c037a3c23fa77299`
- Target contracts: `7a9e665cacd0e9ba823444ef30ed975cb5d21e5f23cb2bc933275139fb1ee40f`
- Tournament report: `1c333efb48692ed3d37000e2bc2dd01e7014c7fe39afed1d8c4145b9e0533154`
- Universe: `1df2ee40a746c01d82e731a90addb23694291d01732c225ad50e7f613b053fb2`
- Replay input: `7fba24a50b60d49c899984b9432e139c7436ecc48efe6c58b3ac976bd591cbc3`
- Replay output: `e8bf68b0897d2812fe8d12316c3bb63f3bdda60371c3d1d3d3d504eb89bb46b2`
- Replay bounded rows: `27ea601f87bba1b52cadb9b091ce06714c4a473c651187be4fe2cbe5a1928848`

## Target Version Status

The historical replay evidence uses `stock_level_target_provenance_v1`. The current selector target registry expects `stock_level_target_provenance_v2`.

Ticket 56A determined that v1 behavior is reconstructable from the historical commit for artifact replay, but migration is not safe to infer silently. Regenerated v2 evidence would represent a new experiment.

## Environment Differences

- Replay Python: `3.11.9`
- Replay platform: `Windows-10-10.0.26200-SP0`
- Key packages: pandas `3.0.3`, pyarrow `24.0.0`, numpy `2.0.2`, scikit-learn `1.6.1`, scipy `1.17.1`, PyYAML `6.0.2`
- Original Python version, lockfile, and installed package set were not preserved in candidate lineage.
- Historical `cf5f69df` code does not implement multi-seed seed-suffixed candidate generation.

## Validation

- All six required evidence files exist.
- All JSON files in the copied bundle parse.
- Source and destination file hashes match.
- Hashes in this acceptance reference match `docs/audits/ticket_56a_preserved_evidence_manifest.json`.
- Historical commit object exists.
- Replay result remains exact.
- No model training was run.
- No promotion was performed.
- No unrelated historical-worktree files were copied.
