# Allocation Policy Leaderboard v2

|rank|policy_name|policy_kind|total_return|max_drawdown|sharpe|sortino|calmar|return_per_unit_drawdown|turnover|estimated_transaction_costs|mean_exposure|
|---|---|---|---|---|---|---|---|---|---|---|---|
|1|champion_baseline|diagnostic_baseline|1.120447|0.091513|7.291425|17.352484|18.201983|12.243642|0.000000|0.000000|1.000000|
|2|always_full_exposure|diagnostic_baseline|1.120447|0.091513|7.291425|17.352484|18.201983|12.243642|0.000000|0.000000|1.000000|
|3|binary_exposure_overlay|allocation_policy|0.953036|0.067735|7.372436|21.070561|20.588137|14.070095|2.300000|0.001150|0.826705|
|4|risk_adjusted_allocation_aggressive|allocation_policy|0.719182|0.052890|7.287425|19.846766|19.427742|13.597633|4.641667|0.002321|0.664836|
|5|return_only_allocation_aggressive|allocation_policy|0.663533|0.051267|7.091591|18.954724|18.381326|12.942761|4.825000|0.002413|0.627399|
|6|risk_adjusted_allocation_balanced|allocation_policy|0.634610|0.036490|7.531544|25.933122|24.620785|17.391291|4.737500|0.002369|0.558807|
|7|meta_ensemble_allocation|allocation_policy|0.606832|0.010569|5.873451|53.612414|81.032524|57.415674|7.666667|0.003833|0.422348|
|8|best_grid_search_diagnostic_policy|allocation_policy|0.532317|0.039909|7.293486|21.499847|18.666151|13.338391|3.511111|0.001756|0.505650|
|9|return_only_allocation_balanced|allocation_policy|0.527805|0.031891|7.144806|24.560029|23.148588|16.550066|5.868750|0.002934|0.490483|
|10|always_half_exposure|diagnostic_baseline|0.460726|0.046544|7.286828|17.341145|13.736743|9.898782|0.500000|0.000250|0.500000|
|11|risk_adjusted_allocation_conservative|allocation_policy|0.418876|0.029459|6.873384|24.272802|19.632531|14.218816|4.397222|0.002199|0.390609|
|12|return_only_allocation|allocation_policy|0.393006|0.015928|7.357905|39.863394|33.960714|24.673961|5.583333|0.002792|0.355429|
|13|return_only_allocation_conservative|allocation_policy|0.340491|0.018977|6.939195|28.271395|24.533832|17.942024|5.097917|0.002549|0.326957|
|14|risk_adjusted_allocation|allocation_policy|0.157378|0.008365|5.041915|25.470149|25.109677|18.813729|4.300000|0.002150|0.132008|
|15|always_zero_exposure|diagnostic_baseline|-0.000500|0.000500|-1.155336|-1.142131|-1.304365|-1.000000|1.000000|0.000500|0.000000|

## Outcome Winners

- Total return: champion_baseline
- Max drawdown: always_zero_exposure
- Sharpe: risk_adjusted_allocation_balanced
- Sortino: meta_ensemble_allocation
- Calmar: meta_ensemble_allocation
- Dominated: none
- Too defensive: risk_adjusted_allocation, always_zero_exposure
- Too choppy: risk_adjusted_allocation_aggressive, return_only_allocation_aggressive, risk_adjusted_allocation_balanced, best_grid_search_diagnostic_policy, return_only_allocation_balanced, risk_adjusted_allocation_conservative, return_only_allocation, return_only_allocation_conservative, risk_adjusted_allocation

Research only. Trading impact: none. Production validated: false.
