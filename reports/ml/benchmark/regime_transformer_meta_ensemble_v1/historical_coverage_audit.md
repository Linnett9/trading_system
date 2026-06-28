# Historical Coverage Audit

Research only. Trading impact: none. Production validated: false.

Current bottleneck: meta_or_canonical_artifacts
Minimum independent periods: 36
Full model rerun required: True

|layer|earliest|latest|rebalance dates|independent periods|
|---|---:|---:|---:|---:|
|raw_stooq_parquet|1962-01-02|2026-06-24|0|None|
|adjusted_yahoo_reference|1990-01-02|2026-06-25|9187|None|
|source_prediction_artifacts|2021-01-04|2026-04-20|311|None|
|meta_auxiliary_predictions|2022-08-29|2026-04-20|215|None|
|canonical_replay|2022-08-29|2026-04-20|215|21|

## Blockers

- meta_or_canonical_artifacts_start_after_source_predictions
- overlapping_label_windows_reduce_independent_count
- prediction_artifacts_start_after_price_history
- too_few_canonical_independent_periods
- too_few_valid_adjusted_independent_periods

## Overnight Command

`python3.10 main.py --mode ml-research-batch --config configs/research/regime_transformer_meta_ensemble_v1.yaml --profile benchmark`
