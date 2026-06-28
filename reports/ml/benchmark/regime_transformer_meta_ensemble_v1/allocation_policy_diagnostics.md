# Allocation Policy Diagnostics v2

|policy|mean|median|min|max|std|changes|max_change|constant|dominated|
|---|---|---|---|---|---|---|---|---|---|
|risk_adjusted_allocation_balanced|0.761221|0.756250|0.250000|1.000000|0.183440|178|0.300000|false|false|
|return_only_allocation|0.556421|0.550000|0.000000|1.000000|0.348623|188|0.858333|false|false|
|champion_baseline|1.000000|1.000000|1.000000|1.000000|0.000000|0|0.000000|true|false|
|always_full_exposure|1.000000|1.000000|1.000000|1.000000|0.000000|0|0.000000|true|false|
|risk_adjusted_allocation_aggressive|0.795581|0.800000|0.333333|1.000000|0.138390|191|0.366667|false|false|
|return_only_allocation_aggressive|0.735291|0.750000|0.300000|1.000000|0.135991|191|0.333333|false|false|
|binary_exposure_overlay|0.755271|0.700000|0.700000|1.000000|0.093828|93|0.300000|false|false|
|risk_adjusted_allocation_conservative|0.616163|0.650000|0.204167|0.800000|0.172258|169|0.200000|false|false|
|return_only_allocation_balanced|0.619535|0.625000|0.100000|1.000000|0.178050|193|0.300000|false|false|
|best_grid_search_diagnostic_policy|0.652920|0.662500|0.233333|0.800000|0.125499|176|0.200000|false|false|
|return_only_allocation_conservative|0.430455|0.425000|0.025000|0.800000|0.159619|193|0.200000|false|false|
|meta_ensemble_allocation|0.212287|0.016667|0.000000|1.000000|0.315520|124|1.000000|false|false|
|always_half_exposure|0.500000|0.500000|0.500000|0.500000|0.000000|0|0.000000|true|false|
|risk_adjusted_allocation|0.160013|0.000000|0.000000|1.000000|0.257816|135|1.000000|false|false|
|always_zero_exposure|0.000000|0.000000|0.000000|0.000000|0.000000|0|0.000000|true|false|

## Prediction To Exposure

- risk_adjusted_allocation_balanced: return_corr=0.665988, volatility_corr=0.077015, drawdown_corr=0.585767
- return_only_allocation: return_corr=0.876987, volatility_corr=0.410835, drawdown_corr=0.301091
- champion_baseline: return_corr=, volatility_corr=, drawdown_corr=
- always_full_exposure: return_corr=, volatility_corr=, drawdown_corr=
- risk_adjusted_allocation_aggressive: return_corr=0.772122, volatility_corr=0.251582, drawdown_corr=0.396217
- return_only_allocation_aggressive: return_corr=0.808468, volatility_corr=0.469866, drawdown_corr=0.126789
- binary_exposure_overlay: return_corr=0.340685, volatility_corr=-0.020120, drawdown_corr=0.329573
- risk_adjusted_allocation_conservative: return_corr=0.491676, volatility_corr=-0.148715, drawdown_corr=0.739071
- return_only_allocation_balanced: return_corr=0.814029, volatility_corr=0.463655, drawdown_corr=0.155183
- best_grid_search_diagnostic_policy: return_corr=0.604833, volatility_corr=-0.021799, drawdown_corr=0.649645
- return_only_allocation_conservative: return_corr=0.810594, volatility_corr=0.450790, drawdown_corr=0.181152
- meta_ensemble_allocation: return_corr=0.404981, volatility_corr=0.008071, drawdown_corr=0.358744
- always_half_exposure: return_corr=, volatility_corr=, drawdown_corr=
- risk_adjusted_allocation: return_corr=0.671303, volatility_corr=0.064942, drawdown_corr=0.502574
- always_zero_exposure: return_corr=, volatility_corr=, drawdown_corr=

Research only. Trading impact: none. Production validated: false.
