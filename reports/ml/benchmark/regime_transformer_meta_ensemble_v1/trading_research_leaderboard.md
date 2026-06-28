# Trading Research Leaderboard

Canonical continuous returns determine rank when available. Old period-grid returns are diagnostic only. Classification metrics are diagnostics only.

|rank|candidate|type|canonical return|diagnostic period-grid return|anomaly-adjusted return|max drawdown|Sharpe|turnover|costs|promotion status|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|1|equal_weight_selected_universe|tradable_benchmark|5.2048|5.2048||0.3818|1.2664|9.9000||blocked|
|2|always_full_champion_universe|tradable_benchmark|3.5442|3.5442||0.3690|1.1198|10.5007||blocked|
|3|exact_champion_replay|canonical_replay|2.6883|319350.9324|0.7808|0.1523|1.2895|0.0000|0.0000|blocked|
|4|selected_bayesian_optimizer_diagnostic_policy|allocation_optimizer|1.6762|23572.2635|0.9361|0.1369|1.3142|0.2358|0.0001|blocked|
|5|qqq_buy_and_hold|tradable_benchmark|1.1519|1.1519||0.1754|1.0788|1.0000||blocked|
|6|spy_buy_and_hold|tradable_benchmark|0.8153|0.8153||0.1365|1.1720|1.0000||blocked|
|7|risk_adjusted_allocation_balanced|allocation_policy||650879.7735||0.7047|4.1647|21.9063|0.0110||
|8|return_only_allocation|allocation_policy||636088.8252||0.3295|4.2693|38.3889|0.0192||
|9|champion_baseline|allocation_baseline||428270.9430||0.9029|3.7979|0.0000|0.0000||
|10|always_full_exposure|allocation_baseline||428270.9430||0.9029|3.7979|0.0000|0.0000||
|11|risk_adjusted_allocation_aggressive|allocation_policy||404642.8370||0.7697|4.0229|17.2250|0.0086||
|12|return_only_allocation_aggressive|allocation_policy||140863.8604||0.7561|3.9940|15.3500|0.0077||
|13|binary_exposure_overlay|allocation_policy||82867.8211||0.8053|3.7863|9.7000|0.0048||
|14|risk_adjusted_allocation_conservative|allocation_policy||81449.3970||0.5545|4.3127|18.5813|0.0093||
|15|return_only_allocation_balanced|allocation_policy||73217.0584||0.6489|4.0622|20.2875|0.0101||
|16|best_grid_search_diagnostic_policy|allocation_policy||65493.8710||0.6623|4.1541|14.9986|0.0075||
|17|return_only_allocation_conservative|allocation_policy||6059.2158||0.4726|4.0975|17.9208|0.0090||
|18|meta_ensemble_allocation|allocation_policy||2015.8242||0.1289|3.1724|34.0333|0.0170||
|19|always_half_exposure|allocation_baseline||1112.3945||0.6763|3.7977|0.5000|0.0003||
|20|risk_adjusted_allocation|allocation_policy||293.0399||0.0291|3.3633|26.2667|0.0131||
|21|multitask_transformer|base_model||0.7366||0.1643||1.5000|||
|22|logistic_regression|base_model||0.7282||0.1643||1.5000|||
|23|temporal_fusion_transformer|base_model||0.7019||0.1662||1.5000|||
|24|transformer|base_model||0.6912||0.1646||1.5000|||
|25|itransformer|base_model||0.6594||0.1662||1.5000|||
|26|dlinear|base_model||0.6576||0.1921||2.4000|||
|27|gradient_boosting|base_model||0.6049||0.1662||0.9000|||
|28|random_forest|base_model||0.5997||0.1643||0.3000|||
|29|news_analysis_transformer|base_model||0.5960||0.1643||0.9000|||
|30|patchtst|base_model||0.5920||0.1643||0.3000|||
|31|market_context_encoder|base_model||0.5920||0.1643||0.3000|||
|32|momentum_transformer|base_model||0.5655||0.1659||0.9000|||
|33|meta_ensemble_logistic|meta_ensemble||0.0890||0.8021||4.5000|||
|34|always_zero_exposure|allocation_baseline||-0.0005||0.0005|-0.5239|1.0000|0.0005||

## Classification Diagnostics

|model|role|balanced accuracy|walk-forward balanced accuracy|Brier|ECE|
|---|---|---:|---:|---:|---:|
|champion_baseline||||||
|logistic_regression||0.7377|0.7776|0.2475|0.3107|
|random_forest||0.8341|0.8082|0.1267|0.2673|
|gradient_boosting||0.8523|0.8389|0.0677|0.1283|
|dlinear||0.4725|0.5278|0.2755|0.2858|
|patchtst||0.5519|0.5458|0.0994|0.1070|
|transformer||0.6270|0.6524|0.1067|0.1024|
|itransformer||0.8151|0.7083|0.0873|0.0932|
|momentum_transformer||0.5766|0.6056|0.1525|0.1607|
|multitask_transformer||0.6274|0.6611|0.1528|0.1616|
|market_context_encoder||0.4961|0.5460|0.0727|0.0705|
|news_analysis_transformer||0.6460|0.6597|0.0565|0.0528|
|temporal_fusion_transformer||0.6990|0.6405|0.0661|0.0643|
|meta_ensemble_logistic|configured_meta_model|0.8121|0.6803|0.1176|0.1695|
|selected_classifier|selected_classifier|0.8427|0.6989|0.0736|0.0946|
|selected_calibrated|selected_calibrated|0.7927|0.6793|0.0614|0.0897|
|selected_overlay|selected_overlay|0.8427|0.6989|0.0736|0.0946|

## Meta Auxiliary Forecast Diagnostics

|target|available|MAE|RMSE|Pearson|Spearman|directional accuracy|
|---|---|---:|---:|---:|---:|---:|
|actual_forward_return_5d|True|0.0434|0.0596|0.2357|0.2369|0.5704|
|actual_forward_return_10d|True|0.0633|0.0834|0.2847|0.3086|0.5901|
|actual_future_volatility|True|0.0059|0.0075|0.8018|0.7437||
|actual_future_drawdown|True|0.0431|0.0522|0.6406|0.6172||
|actual_max_adverse_excursion|True|0.0468|0.0582|0.5106|0.4906||
|actual_max_favourable_excursion|True|0.0810|0.1135|0.8293|0.8022||

Research only. Trading impact: none. Production validated: false.
