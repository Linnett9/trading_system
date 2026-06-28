# Regime Transformer Meta Ensemble Leaderboard

|model|selection_role|selected_model|holdout_balanced_accuracy|walk_forward_balanced_accuracy|calibration_method|brier_score|expected_calibration_error|overlay_start_date|overlay_end_date|overlay_sample_count|overlay_baseline_return|overlay_adjusted_return|overlay_return_delta|overlay_max_drawdown_improvement|turnover|reduced_exposure_days|promotion_gate_score|promotion_candidate|finite_sanity_check|selection_reason|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|champion_baseline|||||||||||||0.0000|0.0000|0.0000|0||False|True||
|dlinear|||0.7173|0.6999|raw|0.2218|0.1543||||0.3161|0.2389|-0.0772|0.0155|2.1000|55||False|||
|patchtst|||0.7063|0.6788|raw|0.2597|0.2510||||0.3168|0.2022|-0.1146|0.0134|2.7000|51||False|||
|transformer|||0.8033|0.7911|temperature_scaling|0.1785|0.1711||||0.3258|0.2313|-0.0945|0.0155|2.7000|42||False|||
|meta_ensemble_logistic|configured_meta_model|meta_ensemble_logistic|0.7382|0.6607|platt|0.1687|0.1406|2024-07-15|2026-04-20|966|0.0673|0.0675|0.0002|0.0192|6.0000|55||False|True|configured ml.meta_model_type baseline row|
|selected_classifier|selected_classifier|meta_ensemble_lightgbm|0.7875|0.6976|raw|0.1492|0.1246|2024-07-15|2026-04-20|966|0.0673|0.0651|-0.0022|0.0307|6.0000|60|1.8469|False|True|highest holdout balanced accuracy; not automatically selected as trading overlay|
|selected_calibrated|selected_calibrated|meta_ensemble_lightgbm|0.7875|0.6976|raw|0.1492|0.1246|2024-07-15|2026-04-20|966|0.0673|0.0651|-0.0022|0.0307|6.0000|60|1.8469|False|True|lowest Brier score with ECE as tie-breaker|
|selected_overlay|selected_overlay|meta_ensemble_lightgbm|0.7875|0.6976|raw|0.1492|0.1246|2024-07-15|2026-04-20|966|0.0673|0.0651|-0.0022|0.0307|6.0000|60|1.8469|False|True|highest promotion-gate utility balancing walk-forward accuracy, Brier/ECE, overlay return delta, drawdown impact, turnover, and reduced-exposure days|

Research only. Trading impact: none. Production validated: false.