# Trading Research Leaderboard

Trading outcomes determine rank. Classification metrics are diagnostics only.

|rank|candidate|type|total return|max drawdown|Sharpe|Sortino|Calmar|turnover|costs|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|1|itransformer|base_model|1.1789|0.0772||||1.5000||
|2|champion_baseline|allocation_baseline|1.1204|0.0915|7.2914|17.3525|18.2020|0.0000|0.0000|
|3|always_full_exposure|allocation_baseline|1.1204|0.0915|7.2914|17.3525|18.2020|0.0000|0.0000|
|4|transformer|base_model|1.1036|0.0635||||0.9000||
|5|multitask_transformer|base_model|1.1036|0.0635||||0.9000||
|6|news_analysis_transformer|base_model|1.1036|0.0635||||0.9000||
|7|random_forest|base_model|1.0882|0.0818||||1.5000||
|8|temporal_fusion_transformer|base_model|1.0882|0.0818||||1.5000||
|9|dlinear|base_model|1.0703|0.0818||||1.5000||
|10|logistic_regression|base_model|1.0306|0.0763||||0.9000||
|11|gradient_boosting|base_model|1.0306|0.0763||||0.9000||
|12|momentum_transformer|base_model|1.0306|0.0763||||0.9000||
|13|patchtst|base_model|1.0270|0.1082||||1.5000||
|14|market_context_encoder|base_model|0.9611|0.0635||||0.3000||
|15|binary_exposure_overlay|allocation_policy|0.9530|0.0677|7.3724|21.0706|20.5881|2.3000|0.0012|
|16|risk_adjusted_allocation_aggressive|allocation_policy|0.7192|0.0529|7.2874|19.8468|19.4277|4.6417|0.0023|
|17|return_only_allocation_aggressive|allocation_policy|0.6635|0.0513|7.0916|18.9547|18.3813|4.8250|0.0024|
|18|risk_adjusted_allocation_balanced|allocation_policy|0.6346|0.0365|7.5315|25.9331|24.6208|4.7375|0.0024|
|19|meta_ensemble_allocation|allocation_policy|0.6068|0.0106|5.8735|53.6124|81.0325|7.6667|0.0038|
|20|best_grid_search_diagnostic_policy|allocation_policy|0.5323|0.0399|7.2935|21.4998|18.6662|3.5111|0.0018|
|21|return_only_allocation_balanced|allocation_policy|0.5278|0.0319|7.1448|24.5600|23.1486|5.8688|0.0029|
|22|always_half_exposure|allocation_baseline|0.4607|0.0465|7.2868|17.3411|13.7367|0.5000|0.0003|
|23|selected_random_optimizer_diagnostic_policy|allocation_optimizer|0.4466|0.0226|7.7523|32.6654|27.3421|3.3840|0.0017|
|24|risk_adjusted_allocation_conservative|allocation_policy|0.4189|0.0295|6.8734|24.2728|19.6325|4.3972|0.0022|
|25|return_only_allocation|allocation_policy|0.3930|0.0159|7.3579|39.8634|33.9607|5.5833|0.0028|
|26|return_only_allocation_conservative|allocation_policy|0.3405|0.0190|6.9392|28.2714|24.5338|5.0979|0.0025|
|27|risk_adjusted_allocation|allocation_policy|0.1574|0.0084|5.0419|25.4701|25.1097|4.3000|0.0022|
|28|meta_ensemble_logistic|meta_ensemble|0.0169|0.0647||||1.5000||
|29|always_zero_exposure|allocation_baseline|-0.0005|0.0005|-1.1553|-1.1421|-1.3044|1.0000|0.0005|

## Classification Diagnostics

|model|role|balanced accuracy|walk-forward balanced accuracy|Brier|ECE|
|---|---|---:|---:|---:|---:|
|champion_baseline||||||
|logistic_regression||0.9420|0.9227|0.0398|0.0415|
|random_forest||0.9859|0.9640|0.0799|0.2490|
|gradient_boosting||0.9859|0.9484|0.0177|0.0397|
|dlinear||0.6408|0.5887|0.3152|0.2821|
|patchtst||0.6980|0.5393|0.2836|0.2590|
|transformer||0.8644|0.6454|0.0929|0.0639|
|itransformer||0.5901|0.5030|0.3691|0.3580|
|momentum_transformer||0.8087|0.7364|0.1767|0.1728|
|multitask_transformer||0.8055|0.6325|0.1281|0.0868|
|market_context_encoder||0.4647|0.4763|0.2604|0.1456|
|news_analysis_transformer||0.8434|0.6497|0.1142|0.0854|
|temporal_fusion_transformer||0.8722|0.6816|0.0847|0.0633|
|meta_ensemble_logistic|configured_meta_model|0.9641|0.7389|0.0351|0.0354|
|selected_classifier|selected_classifier|0.9886|0.8150|0.0109|0.0072|
|selected_calibrated|selected_calibrated|0.9886|0.8150|0.0109|0.0072|
|selected_overlay|selected_overlay|0.9886|0.8150|0.0109|0.0072|

## Meta Auxiliary Forecast Diagnostics

|target|available|MAE|RMSE|Pearson|Spearman|directional accuracy|
|---|---|---:|---:|---:|---:|---:|
|actual_forward_return_5d|True|0.0095|0.0128|0.2273|0.2431|0.3961|
|actual_forward_return_10d|True|0.0161|0.0196|0.3541|0.2952|0.3647|
|actual_future_volatility|True|0.0049|0.0060|0.4441|0.4230||
|actual_future_drawdown|True|0.0262|0.0328|0.4717|0.4645||
|actual_max_adverse_excursion|True|0.0187|0.0235|0.4217|0.3647||
|actual_max_favourable_excursion|True|0.0330|0.0451|0.3296|0.4323||

Research only. Trading impact: none. Production validated: false.
