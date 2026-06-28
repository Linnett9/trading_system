# Allocation Policy Leaderboard v2

|rank|policy_name|policy_kind|total_return|max_drawdown|sharpe|sortino|calmar|return_per_unit_drawdown|turnover|estimated_transaction_costs|mean_exposure|
|---|---|---|---|---|---|---|---|---|---|---|---|
|1|risk_adjusted_allocation_balanced|allocation_policy|650879.773534|0.704720|4.164739|21.589748|53.554320|923600.546501|21.906250|0.010953|0.761221|
|2|return_only_allocation|allocation_policy|636088.825192|0.329503|4.269300|37.237173|113.802495|1930448.217357|38.388889|0.019194|0.556421|
|3|champion_baseline|diagnostic_baseline|428270.943003|0.902879|3.797880|12.249672|37.164215|474339.196640|0.000000|0.000000|1.000000|
|4|always_full_exposure|diagnostic_baseline|428270.943003|0.902879|3.797880|12.249672|37.164215|474339.196640|0.000000|0.000000|1.000000|
|5|risk_adjusted_allocation_aggressive|allocation_policy|404642.837013|0.769715|4.022874|17.755087|42.903167|525704.937484|17.225000|0.008612|0.795581|
|6|return_only_allocation_aggressive|allocation_policy|140863.860444|0.756101|3.994009|16.794843|32.406317|186302.900414|15.350000|0.007675|0.735291|
|7|binary_exposure_overlay|allocation_policy|82867.821132|0.805287|3.786299|14.652911|26.154137|102904.696818|9.700000|0.004850|0.755271|
|8|risk_adjusted_allocation_conservative|allocation_policy|81449.396978|0.554481|4.312729|25.007649|37.797087|146892.938320|18.581250|0.009291|0.616163|
|9|return_only_allocation_balanced|allocation_policy|73217.058376|0.648887|4.062185|20.225000|31.327250|112834.900360|20.287500|0.010144|0.619535|
|10|best_grid_search_diagnostic_policy|allocation_policy|65493.871035|0.662291|4.154110|19.652376|29.727349|98889.924586|14.998611|0.007499|0.652920|
|11|return_only_allocation_conservative|allocation_policy|6059.215758|0.472616|4.097540|23.815772|20.730389|12820.592029|17.920833|0.008960|0.430455|
|12|meta_ensemble_allocation|allocation_policy|2015.824163|0.128871|3.172360|60.520900|54.274814|15642.134079|34.033333|0.017017|0.212287|
|13|always_half_exposure|diagnostic_baseline|1112.394501|0.676334|3.797697|12.248323|8.570864|1644.740237|0.500000|0.000250|0.500000|
|14|risk_adjusted_allocation|allocation_policy|293.039857|0.029093|3.363332|83.582604|128.012367|10072.507485|26.266667|0.013133|0.160013|
|15|always_zero_exposure|diagnostic_baseline|-0.000500|0.000500|-0.523892|-0.522672|-0.273236|-1.000000|1.000000|0.000500|0.000000|

## Outcome Winners

- Total return: risk_adjusted_allocation_balanced
- Max drawdown: always_zero_exposure
- Sharpe: risk_adjusted_allocation_conservative
- Sortino: risk_adjusted_allocation
- Calmar: risk_adjusted_allocation
- Dominated: none
- Too defensive: meta_ensemble_allocation, risk_adjusted_allocation, always_zero_exposure
- Too choppy: risk_adjusted_allocation_balanced, return_only_allocation, risk_adjusted_allocation_aggressive, return_only_allocation_aggressive, return_only_allocation_balanced, best_grid_search_diagnostic_policy, return_only_allocation_conservative

Research only. Trading impact: none. Production validated: false.
