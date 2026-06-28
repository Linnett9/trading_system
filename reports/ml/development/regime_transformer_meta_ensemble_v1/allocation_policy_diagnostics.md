# Allocation Policy Diagnostics v2

|policy|mean|median|min|max|std|changes|max_change|constant|dominated|
|---|---|---|---|---|---|---|---|---|---|
|champion_baseline|1.000000|1.000000|1.000000|1.000000|0.000000|0|0.000000|true|false|
|always_full_exposure|1.000000|1.000000|1.000000|1.000000|0.000000|0|0.000000|true|false|
|binary_exposure_overlay|0.826705|0.800000|0.700000|1.000000|0.117792|22|0.250000|false|false|
|risk_adjusted_allocation_aggressive|0.664836|0.675000|0.333333|0.900000|0.122184|42|0.366667|false|false|
|return_only_allocation_aggressive|0.627399|0.602778|0.333333|0.900000|0.128642|43|0.366667|false|false|
|risk_adjusted_allocation_balanced|0.558807|0.550000|0.175000|0.812500|0.144540|37|0.300000|false|false|
|meta_ensemble_allocation|0.422348|0.333333|0.000000|1.000000|0.392640|22|0.833333|false|false|
|best_grid_search_diagnostic_policy|0.505650|0.500000|0.266667|0.675000|0.103510|38|0.191667|false|false|
|return_only_allocation_balanced|0.490483|0.468750|0.137500|0.850000|0.159386|42|0.300000|false|false|
|always_half_exposure|0.500000|0.500000|0.500000|0.500000|0.000000|0|0.000000|true|false|
|risk_adjusted_allocation_conservative|0.390609|0.352083|0.145833|0.612500|0.125448|36|0.200000|false|false|
|return_only_allocation|0.355429|0.330556|0.066667|0.675000|0.159490|43|0.283333|false|false|
|return_only_allocation_conservative|0.326957|0.315625|0.062500|0.650000|0.135035|43|0.200000|false|false|
|risk_adjusted_allocation|0.132008|0.100000|0.000000|0.466667|0.117852|41|0.241667|false|false|
|always_zero_exposure|0.000000|0.000000|0.000000|0.000000|0.000000|0|0.000000|true|false|

## Prediction To Exposure

- champion_baseline: return_corr=, volatility_corr=, drawdown_corr=
- always_full_exposure: return_corr=, volatility_corr=, drawdown_corr=
- binary_exposure_overlay: return_corr=0.532088, volatility_corr=-0.062178, drawdown_corr=0.168844
- risk_adjusted_allocation_aggressive: return_corr=0.463507, volatility_corr=-0.156490, drawdown_corr=0.267431
- return_only_allocation_aggressive: return_corr=0.503286, volatility_corr=-0.024762, drawdown_corr=0.131896
- risk_adjusted_allocation_balanced: return_corr=0.384751, volatility_corr=-0.307574, drawdown_corr=0.411461
- meta_ensemble_allocation: return_corr=0.532088, volatility_corr=-0.062178, drawdown_corr=0.168844
- best_grid_search_diagnostic_policy: return_corr=0.317284, volatility_corr=-0.375739, drawdown_corr=0.470199
- return_only_allocation_balanced: return_corr=0.505591, volatility_corr=-0.015246, drawdown_corr=0.125035
- always_half_exposure: return_corr=, volatility_corr=, drawdown_corr=
- risk_adjusted_allocation_conservative: return_corr=0.267564, volatility_corr=-0.464295, drawdown_corr=0.546867
- return_only_allocation: return_corr=0.908970, volatility_corr=0.345332, drawdown_corr=-0.289299
- return_only_allocation_conservative: return_corr=0.501118, volatility_corr=-0.001959, drawdown_corr=0.112588
- risk_adjusted_allocation: return_corr=0.650843, volatility_corr=-0.058143, drawdown_corr=0.079043
- always_zero_exposure: return_corr=, volatility_corr=, drawdown_corr=

Research only. Trading impact: none. Production validated: false.
