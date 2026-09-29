# DS24 CLEAN V2 Mac reconstruction handoff

This package publishes source and metadata only. It contains no Parquet dataset,
model checkpoint, model artifact, research metric, admission record, log, or
runtime state.

The pinned publication commit is recorded in `publication_pointer.json` inside
the accompanying metadata ZIP. Fetch the branch, detach at that commit, and
verify `builder_source_information.json` before reconstructing data.

The four files under `inventories/` use one normalized schema. A Mac base
partition is reusable only when its repository-relative path, byte size, and
SHA-256 all match the corresponding entry. Rebuild every missing or mismatched
V1 partition locally before producing CLEAN V2 outputs.

Run the existing authoritative builders in this order after the V1 bases and
canonical raw bars are complete:

```bash
python scripts/local/ds24_clean_v2_build_sidecar.py --build --workers 2
python scripts/local/ds24_clean_v2_build_target_delta.py --build --workers 2
python scripts/local/ds24_clean_v2_certify.py --full-materialized-authority
```

The exact target implementation is
`core/research/ml/five_minute_target_dataset.py`. Do not replace or reimplement
it. Its complete local import closure and per-file hashes are recorded in
`builder_source_information.json`.

The feature repair remains exactly these eleven fields:

- `overnight_gap`
- `previous_session_return`
- `two_session_return`
- `opening_range_position`
- `session_return_30m`
- `opening_return_30m`
- `minutes_since_open`
- `minutes_until_close`
- `session_progress`
- `opening_period_flag`
- `early_close_session_flag`

The frozen authorities are:

- Feature: `ac2be1f9aeea31a9767ed69c9fd84bad82c59deaaa8ce3e945d4cb756b01029d`
- Target: `41dd1fd36d7081dc2aeaf39bd74a4228d17fbf56f3e1a7b2ea987499bf923eb1`
- Static bundle: `4952431d7a6ec781a861d196a5d27b63128d80bad693e56bcafd8d7d420981dc`
- Refit: `REFIT_EVERY_5_TRADING_SESSIONS_V1`
