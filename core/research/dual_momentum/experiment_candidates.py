from copy import deepcopy
from itertools import product

def dual_momentum_candidate_configs(dual_config):
    grid = dual_config.get("experiment_grid", {})

    top_values = grid.get("top_n", [dual_config.get("top_n", 3)])
    rebalance_values = grid.get(
        "rebalance_frequency",
        [dual_config.get("rebalance_frequency", "monthly")],
    )
    momentum_values = grid.get(
        "momentum_periods",
        [dual_config.get("momentum_periods", [126, 252])],
    )
    regime_confirmation_mode_values = grid.get(
        "regime_confirmation_mode",
        [dual_config.get("regime_confirmation_mode", "primary")],
    )
    asset_filter_values = grid.get(
        "use_asset_trend_filter",
        [dual_config.get("use_asset_trend_filter", True)],
    )
    volatility_values = grid.get(
        "target_volatility",
        [dual_config.get("target_volatility")],
    )
    drawdown_values = grid.get(
        "max_drawdown_guard",
        [dual_config.get("max_drawdown_guard")],
    )
    breadth_values = grid.get(
        "min_breadth_percent",
        [dual_config.get("min_breadth_percent", 0)],
    )
    breadth_scaled_exposure_values = grid.get(
        "breadth_scaled_exposure_enabled",
        [dual_config.get("breadth_scaled_exposure_enabled", False)],
    )
    drawdown_recovery_scaling_values = grid.get(
        "drawdown_recovery_scaling_enabled",
        [dual_config.get("drawdown_recovery_scaling_enabled", False)],
    )
    volatility_shock_values = grid.get(
        "volatility_shock_filter_enabled",
        [dual_config.get("volatility_shock_filter_enabled", False)],
    )
    selection_mode_values = grid.get(
        "selection_mode",
        [dual_config.get("selection_mode", "ranked")],
    )
    min_selection_score_values = grid.get(
        "min_selection_score",
        [dual_config.get("min_selection_score", 0)],
    )
    max_selected_assets_values = grid.get(
        "max_selected_assets",
        [dual_config.get("max_selected_assets")],
    )
    weighting_values = grid.get(
        "weighting",
        [dual_config.get("weighting", "equal")],
    )
    max_position_weight_values = grid.get(
        "max_position_weight",
        [dual_config.get("max_position_weight")],
    )
    strict_kill_switch_values = grid.get(
        "strict_drawdown_kill_switch",
        [dual_config.get("strict_drawdown_kill_switch", False)],
    )
    mixed_exposure_values = grid.get(
        "mixed_risk_exposure",
        [dual_config.get("mixed_risk_exposure", 0.50)],
    )
    risk_off_exposure_values = grid.get(
        "risk_off_risk_exposure",
        [dual_config.get("risk_off_risk_exposure", 0)],
    )
    fallback_allocation_values = grid.get(
        "fallback_allocation",
        [dual_config.get("fallback_allocation", 0)],
    )
    decay_exit_values = grid.get(
        "decay_exit_enabled",
        [dual_config.get("decay_exit_enabled", False)],
    )
    rank_drop_values = grid.get(
        "rank_drop_exit_top_n",
        [dual_config.get("rank_drop_exit_top_n")],
    )
    rank_deterioration_exit_values = grid.get(
        "rank_deterioration_exit_enabled",
        [dual_config.get("rank_deterioration_exit_enabled", False)],
    )
    rank_deterioration_exit_rank_values = grid.get(
        "rank_deterioration_exit_rank",
        [dual_config.get("rank_deterioration_exit_rank")],
    )
    chop_filter_values = grid.get(
        "chop_filter_enabled",
        [dual_config.get("chop_filter_enabled", False)],
    )
    chop_exposure_values = grid.get(
        "chop_risk_exposure",
        [dual_config.get("chop_risk_exposure", 0.50)],
    )
    quality_filter_values = grid.get(
        "quality_filter_enabled",
        [dual_config.get("quality_filter_enabled", False)],
    )
    quality_improving_values = grid.get(
        "quality_require_momentum_improving",
        [dual_config.get("quality_require_momentum_improving", False)],
    )
    cooldown_values = grid.get(
        "cooldown_enabled",
        [dual_config.get("cooldown_enabled", False)],
    )
    short_term_penalty_values = grid.get(
        "short_term_weakness_penalty_enabled",
        [dual_config.get("short_term_weakness_penalty_enabled", False)],
    )
    short_term_penalty_floor_values = grid.get(
        "short_term_weakness_penalty_floor",
        [dual_config.get("short_term_weakness_penalty_floor", -0.02)],
    )
    short_term_penalty_weight_values = grid.get(
        "short_term_weakness_penalty_weight",
        [dual_config.get("short_term_weakness_penalty_weight", 1.0)],
    )
    rank_hysteresis_values = grid.get(
        "rank_hysteresis_enabled",
        [dual_config.get("rank_hysteresis_enabled", False)],
    )
    rank_hysteresis_margin_values = grid.get(
        "rank_hysteresis_margin",
        [dual_config.get("rank_hysteresis_margin", 2)],
    )
    rank_hysteresis_max_rank_values = grid.get(
        "rank_hysteresis_max_rank",
        [dual_config.get("rank_hysteresis_max_rank")],
    )
    max_rebalance_replacements_values = grid.get(
        "max_rebalance_replacements",
        [dual_config.get("max_rebalance_replacements")],
    )
    replacement_score_gap_values = grid.get(
        "replacement_score_gap",
        [dual_config.get("replacement_score_gap", 0)],
    )
    rebalance_min_trade_weight_values = grid.get(
        "rebalance_min_trade_weight",
        [dual_config.get("rebalance_min_trade_weight", 0)],
    )
    leadership_values = grid.get(
        "leadership_filter_enabled",
        [dual_config.get("leadership_filter_enabled", False)],
    )
    relative_strength_filter_values = grid.get(
        "relative_strength_filter_enabled",
        [dual_config.get("relative_strength_filter_enabled", False)],
    )
    relative_strength_filter_symbol_values = grid.get(
        "relative_strength_filter_symbol",
        [dual_config.get("relative_strength_filter_symbol", "SPY")],
    )
    relative_strength_filter_period_values = grid.get(
        "relative_strength_filter_period",
        [dual_config.get("relative_strength_filter_period", 63)],
    )
    relative_strength_filter_min_excess_values = grid.get(
        "relative_strength_filter_min_excess",
        [dual_config.get("relative_strength_filter_min_excess", 0)],
    )
    benchmark_sleeve_values = grid.get(
        "benchmark_sleeve_allocation",
        [dual_config.get("benchmark_sleeve_allocation", 0)],
    )
    ranking_score_values = grid.get(
        "ranking_score_mode",
        [dual_config.get("ranking_score_mode", "average_momentum")],
    )
    relative_strength_weight_values = grid.get(
        "relative_strength_weight",
        [dual_config.get("relative_strength_weight", 0.25)],
    )
    transaction_cost_values = grid.get(
        "transaction_cost_bps",
        [dual_config.get("transaction_cost_bps", 2.0)],
    )
    commission_values = grid.get(
        "commission_bps",
        [dual_config.get("commission_bps", 0.0)],
    )
    slippage_values = grid.get(
        "slippage_bps",
        [dual_config.get("slippage_bps", 0.0)],
    )
    spread_cost_values = grid.get(
        "spread_cost_bps",
        [dual_config.get("spread_cost_bps", 0.0)],
    )

    for (
        top_n,
        rebalance,
        momentum_periods,
        regime_confirmation_mode,
        use_asset_filter,
        target_volatility,
        max_drawdown_guard,
        min_breadth_percent,
        breadth_scaled_exposure_enabled,
        drawdown_recovery_scaling_enabled,
        volatility_shock_filter_enabled,
        selection_mode,
        min_selection_score,
        max_selected_assets,
        weighting,
        max_position_weight,
        strict_drawdown_kill_switch,
        mixed_risk_exposure,
        risk_off_risk_exposure,
        fallback_allocation,
        decay_exit_enabled,
        rank_drop_exit_top_n,
        rank_deterioration_exit_enabled,
        rank_deterioration_exit_rank,
        chop_filter_enabled,
        chop_risk_exposure,
        quality_filter_enabled,
        quality_require_momentum_improving,
        cooldown_enabled,
        short_term_weakness_penalty_enabled,
        short_term_weakness_penalty_floor,
        short_term_weakness_penalty_weight,
        rank_hysteresis_enabled,
        rank_hysteresis_margin,
        rank_hysteresis_max_rank,
        max_rebalance_replacements,
        replacement_score_gap,
        rebalance_min_trade_weight,
        leadership_filter_enabled,
        relative_strength_filter_enabled,
        relative_strength_filter_symbol,
        relative_strength_filter_period,
        relative_strength_filter_min_excess,
        benchmark_sleeve_allocation,
        ranking_score_mode,
        relative_strength_weight,
        transaction_cost_bps,
        commission_bps,
        slippage_bps,
        spread_cost_bps,
    ) in product(
        top_values,
        rebalance_values,
        momentum_values,
        regime_confirmation_mode_values,
        asset_filter_values,
        volatility_values,
        drawdown_values,
        breadth_values,
        breadth_scaled_exposure_values,
        drawdown_recovery_scaling_values,
        volatility_shock_values,
        selection_mode_values,
        min_selection_score_values,
        max_selected_assets_values,
        weighting_values,
        max_position_weight_values,
        strict_kill_switch_values,
        mixed_exposure_values,
        risk_off_exposure_values,
        fallback_allocation_values,
        decay_exit_values,
        rank_drop_values,
        rank_deterioration_exit_values,
        rank_deterioration_exit_rank_values,
        chop_filter_values,
        chop_exposure_values,
        quality_filter_values,
        quality_improving_values,
        cooldown_values,
        short_term_penalty_values,
        short_term_penalty_floor_values,
        short_term_penalty_weight_values,
        rank_hysteresis_values,
        rank_hysteresis_margin_values,
        rank_hysteresis_max_rank_values,
        max_rebalance_replacements_values,
        replacement_score_gap_values,
        rebalance_min_trade_weight_values,
        leadership_values,
        relative_strength_filter_values,
        relative_strength_filter_symbol_values,
        relative_strength_filter_period_values,
        relative_strength_filter_min_excess_values,
        benchmark_sleeve_values,
        ranking_score_values,
        relative_strength_weight_values,
        transaction_cost_values,
        commission_values,
        slippage_values,
        spread_cost_values,
    ):
        candidate = deepcopy(dual_config)
        candidate.update({
            "experiment_name": "grid",
            "top_n": top_n,
            "momentum_periods": momentum_periods,
            "rebalance_frequency": rebalance,
            "regime_confirmation_mode": regime_confirmation_mode,
            "use_asset_trend_filter": use_asset_filter,
            "target_volatility": target_volatility,
            "max_drawdown_guard": max_drawdown_guard,
            "min_breadth_percent": min_breadth_percent,
            "breadth_scaled_exposure_enabled": (
                breadth_scaled_exposure_enabled
            ),
            "drawdown_recovery_scaling_enabled": (
                drawdown_recovery_scaling_enabled
            ),
            "volatility_shock_filter_enabled": (
                volatility_shock_filter_enabled
            ),
            "selection_mode": selection_mode,
            "min_selection_score": min_selection_score,
            "max_selected_assets": max_selected_assets,
            "weighting": weighting,
            "max_position_weight": max_position_weight,
            "strict_drawdown_kill_switch": strict_drawdown_kill_switch,
            "mixed_risk_exposure": mixed_risk_exposure,
            "risk_off_risk_exposure": risk_off_risk_exposure,
            "fallback_allocation": fallback_allocation,
            "decay_exit_enabled": decay_exit_enabled,
            "rank_drop_exit_top_n": rank_drop_exit_top_n,
            "rank_deterioration_exit_enabled": (
                rank_deterioration_exit_enabled
            ),
            "rank_deterioration_exit_rank": rank_deterioration_exit_rank,
            "chop_filter_enabled": chop_filter_enabled,
            "chop_risk_exposure": chop_risk_exposure,
            "quality_filter_enabled": quality_filter_enabled,
            "quality_require_momentum_improving": (
                quality_require_momentum_improving
            ),
            "cooldown_enabled": cooldown_enabled,
            "short_term_weakness_penalty_enabled": (
                short_term_weakness_penalty_enabled
            ),
            "short_term_weakness_penalty_floor": (
                short_term_weakness_penalty_floor
            ),
            "short_term_weakness_penalty_weight": (
                short_term_weakness_penalty_weight
            ),
            "rank_hysteresis_enabled": rank_hysteresis_enabled,
            "rank_hysteresis_margin": rank_hysteresis_margin,
            "rank_hysteresis_max_rank": rank_hysteresis_max_rank,
            "max_rebalance_replacements": max_rebalance_replacements,
            "replacement_score_gap": replacement_score_gap,
            "rebalance_min_trade_weight": rebalance_min_trade_weight,
            "leadership_filter_enabled": leadership_filter_enabled,
            "relative_strength_filter_enabled": (
                relative_strength_filter_enabled
            ),
            "relative_strength_filter_symbol": (
                relative_strength_filter_symbol
            ),
            "relative_strength_filter_period": (
                relative_strength_filter_period
            ),
            "relative_strength_filter_min_excess": (
                relative_strength_filter_min_excess
            ),
            "benchmark_sleeve_allocation": benchmark_sleeve_allocation,
            "ranking_score_mode": ranking_score_mode,
            "relative_strength_weight": relative_strength_weight,
            "transaction_cost_bps": transaction_cost_bps,
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "spread_cost_bps": spread_cost_bps,
        })

        yield candidate

    for item in dual_config.get("experiment_variants", []):
        candidate = deepcopy(dual_config)
        candidate.update(item.get("overrides", {}))
        candidate["experiment_name"] = item["name"]
        yield candidate
