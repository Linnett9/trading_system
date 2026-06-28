# Benchmark-Relative and Tradability Validation

Research only. Trading impact: none. Production validated: false.

|candidate|canonical return|anomaly-adjusted|drawdown|Sharpe|turnover|top 5 dates|excess SPY|excess QQQ|excess equal-weight|status|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|spy_buy_and_hold|0.815268|0.439837|0.136547|1.172006|1.000000|0.589372|0.000000|-0.336621|-4.389487|blocked|
|qqq_buy_and_hold|1.151888|0.534814|0.175404|1.078786|1.000000|0.615917|0.336621|0.000000|-4.052867|blocked|
|equal_weight_selected_universe|5.204755|0.636729|0.381793|1.266448|9.900000|0.753435|4.389487|4.052867|0.000000|blocked|
|always_full_champion_universe|3.544168|0.363346|0.369018|1.119770|10.500739|0.777915|2.728900|2.392280|-1.660587|blocked|
|exact_champion_replay|2.688314|0.780818|0.152310|1.289483|10.583323|0.669671|1.873046|1.536425|-2.516442|blocked|
|selected_bayesian_optimizer_diagnostic_policy|1.676211|0.936140|0.136895|1.314250|7.608471|0.753429|0.860943|0.524323|-3.528544|blocked|
