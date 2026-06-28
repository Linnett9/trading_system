# Regime Transformer Meta Ensemble Leaderboard

|model|selection_role|selected_model|holdout_balanced_accuracy|walk_forward_balanced_accuracy|calibration_method|brier_score|expected_calibration_error|overlay_start_date|overlay_end_date|overlay_sample_count|overlay_baseline_return|overlay_adjusted_return|overlay_return_delta|overlay_max_drawdown_improvement|turnover|reduced_exposure_days|promotion_gate_score|promotion_candidate|finite_sanity_check|selection_reason|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|champion_baseline|||||||||||||0.0000|0.0000|0.0000|0||False|True||
|logistic_regression|||0.9420|0.9227|raw|0.0398|0.0415||||1.3298|1.0306|-0.2992|0.0319|0.9000|32||False|||
|random_forest|||0.9859|0.9640|isotonic|0.0799|0.2490||||1.5697|1.0882|-0.4815|-0.0000|1.5000|33||False|||
|gradient_boosting|||0.9859|0.9484|temperature_scaling|0.0177|0.0397||||1.3298|1.0306|-0.2992|0.0319|0.9000|32||False|||
|dlinear|||0.6408|0.5887|raw|0.3152|0.2821||||1.5697|1.0703|-0.4994|0.0000|1.5000|28||False|||
|patchtst|||0.6980|0.5393|platt|0.2836|0.2590||||1.3298|1.0270|-0.3028|0.0000|1.5000|18||False|||
|transformer|||0.8644|0.6454|platt|0.0929|0.0639||||1.5697|1.1036|-0.4661|0.0184|0.9000|37||False|||
|itransformer|||0.5901|0.5030|temperature_scaling|0.3691|0.3580||||1.5697|1.1789|-0.3908|0.0047|1.5000|22||False|||
|momentum_transformer|||0.8087|0.7364|temperature_scaling|0.1767|0.1728||||1.3298|1.0306|-0.2992|0.0319|0.9000|32||False|||
|multitask_transformer|||0.8055|0.6325|raw|0.1281|0.0868||||1.5697|1.1036|-0.4661|0.0184|0.9000|37||False|||
|market_context_encoder|||0.4647|0.4763|platt|0.2604|0.1456||||1.5697|0.9611|-0.6086|0.0184|0.3000|41||False|||
|news_analysis_transformer|||0.8434|0.6497|raw|0.1142|0.0854||||1.5697|1.1036|-0.4661|0.0184|0.9000|37||False|||
|temporal_fusion_transformer|||0.8722|0.6816|raw|0.0847|0.0633||||1.5697|1.0882|-0.4815|-0.0000|1.5000|33||False|||
|meta_ensemble_logistic|configured_meta_model|meta_ensemble_logistic|0.9641|0.7389|raw|0.0351|0.0354|2025-07-21|2026-04-20|414|0.0174|0.0169|-0.0005|0.0268|1.5000|33||False|True|configured ml.meta_model_type baseline row|
|selected_classifier|selected_classifier|meta_ensemble_gradient_boosting|0.9886|0.8150|raw|0.0109|0.0072|2025-07-21|2026-04-20|414|0.0174|0.0168|-0.0006|0.0268|1.5000|33|2.6284|False|True|highest holdout balanced accuracy; not automatically selected as trading overlay|
|selected_calibrated|selected_calibrated|meta_ensemble_gradient_boosting|0.9886|0.8150|raw|0.0109|0.0072|2025-07-21|2026-04-20|414|0.0174|0.0168|-0.0006|0.0268|1.5000|33|2.6284|False|True|lowest Brier score with ECE as tie-breaker|
|selected_overlay|selected_overlay|meta_ensemble_gradient_boosting|0.9886|0.8150|raw|0.0109|0.0072|2025-07-21|2026-04-20|414|0.0174|0.0168|-0.0006|0.0268|1.5000|33|2.6284|False|True|highest promotion-gate utility balancing walk-forward accuracy, Brier/ECE, overlay return delta, drawdown impact, turnover, and reduced-exposure days|

Research only. Trading impact: none. Production validated: false.