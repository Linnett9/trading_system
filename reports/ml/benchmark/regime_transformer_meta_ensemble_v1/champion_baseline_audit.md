# Champion Baseline Audit

Research only. Trading impact: none. Production validated: false.

## Semantics

- Current champion baseline exact replay: False
- Why equal to always full: both use constant allocation exposure of 1.0 in allocation_v2

## Baselines

|baseline|semantic type|available|target exposure|total return|continuous return|drawdown|turnover|costs|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|champion_full_exposure_diagnostic|diagnostic_full_allocation_exposure|True|1.000000|428270.943003||0.902879|0.000000|0.000000|
|always_full_exposure|diagnostic_full_allocation_exposure|True|1.000000|428270.943003||0.902879|0.000000|0.000000|
|exact_champion_replay|exact_champion_replay|True|0.900000|319350.932417|2.557085|0.778372|||

## Exact Replay

- Available: True
- Reason: ok
- Period-grid return: 319350.932417
- Continuous equity return: 2.557085

## Red Flags

- current_allocation_baseline_compounds_overlapping_forward_periods
- current_champion_baseline_is_diagnostic_not_exact_replay
- old_champion_baseline_name_is_misleading
- stooq_adjustment_status_unknown

Research only. Trading impact: none. Production validated: false.
