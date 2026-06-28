# Regime Transformer Meta Ensemble Leaderboard

|model|selection_role|selected_model|holdout_balanced_accuracy|walk_forward_balanced_accuracy|calibration_method|brier_score|expected_calibration_error|overlay_start_date|overlay_end_date|overlay_sample_count|overlay_baseline_return|overlay_adjusted_return|overlay_return_delta|overlay_max_drawdown_improvement|turnover|reduced_exposure_days|promotion_gate_score|promotion_candidate|finite_sanity_check|selection_reason|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|champion_baseline|||||||||||||0.0000|0.0000|0.0000|0||False|True||
|logistic_regression|||0.7377|0.7776|isotonic|0.2475|0.3107||||0.9469|0.7282|-0.2187|0.0669|1.5000|85||False|||
|random_forest|||0.8341|0.8082|platt|0.1267|0.2673||||0.9604|0.5997|-0.3607|0.0669|0.3000|100||False|||
|gradient_boosting|||0.8523|0.8389|isotonic|0.0677|0.1283||||0.9641|0.6049|-0.3592|0.0672|0.9000|95||False|||
|dlinear|||0.4725|0.5278|platt|0.2755|0.2858||||0.8943|0.6576|-0.2367|0.0413|2.4000|83||False|||
|patchtst|||0.5519|0.5458|isotonic|0.0994|0.1070||||0.9469|0.5920|-0.3549|0.0669|0.3000|100||False|||
|transformer|||0.6270|0.6524|isotonic|0.1067|0.1024||||0.9604|0.6912|-0.2692|0.0667|1.5000|90||False|||
|itransformer|||0.8151|0.7083|raw|0.0873|0.0932||||0.9641|0.6594|-0.3047|0.0672|1.5000|90||False|||
|momentum_transformer|||0.5766|0.6056|temperature_scaling|0.1525|0.1607||||0.8943|0.5655|-0.3288|0.0675|0.9000|95||False|||
|multitask_transformer|||0.6274|0.6611|isotonic|0.1528|0.1616||||0.9604|0.7366|-0.2238|0.0669|1.5000|85||False|||
|market_context_encoder|||0.4961|0.5460|platt|0.0727|0.0705||||0.9469|0.5920|-0.3549|0.0669|0.3000|100||False|||
|news_analysis_transformer|||0.6460|0.6597|isotonic|0.0565|0.0528||||0.9469|0.5960|-0.3509|0.0669|0.9000|95||False|||
|temporal_fusion_transformer|||0.6990|0.6405|temperature_scaling|0.0661|0.0643||||0.9641|0.7019|-0.2621|0.0672|1.5000|85||False|||
|meta_ensemble_logistic|configured_meta_model|meta_ensemble_logistic|0.8121|0.6803|platt|0.1176|0.1695|2024-07-15|2026-04-20|966|0.0999|0.0890|-0.0109|0.1008|4.5000|95||False|True|configured ml.meta_model_type baseline row|
|selected_classifier|selected_classifier|meta_ensemble_gradient_boosting|0.8427|0.6989|isotonic|0.0736|0.0946|2024-07-15|2026-04-20|966|0.0999|0.0770|-0.0229|0.1042|3.3000|99|2.0905|False|True|highest holdout balanced accuracy; not automatically selected as trading overlay|
|selected_calibrated|selected_calibrated|meta_ensemble_random_forest|0.7927|0.6793|isotonic|0.0614|0.0897|2024-07-15|2026-04-20|966|0.0999|0.0877|-0.0122|0.0991|5.1000|94|2.0530|False|True|lowest Brier score with ECE as tie-breaker|
|selected_overlay|selected_overlay|meta_ensemble_gradient_boosting|0.8427|0.6989|isotonic|0.0736|0.0946|2024-07-15|2026-04-20|966|0.0999|0.0770|-0.0229|0.1042|3.3000|99|2.0905|False|True|highest promotion-gate utility balancing walk-forward accuracy, Brier/ECE, overlay return delta, drawdown impact, turnover, and reduced-exposure days|

Research only. Trading impact: none. Production validated: false.