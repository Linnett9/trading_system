from core.research.dual_momentum_portfolio import (
    DualMomentumPortfolioBacktester,
)


def build_dual_momentum_tester(config, dual_config):
    return DualMomentumPortfolioBacktester(
        starting_equity=config["backtest"]["starting_equity"],
        experiment_name=dual_config.get("experiment_name"),
        champion_id=dual_config.get("champion_id"),
        champion_source_config_name=dual_config.get(
            "champion_source_config_name",
        ),
        champion_config_path=dual_config.get("champion_config_path"),
        top_n=dual_config.get("top_n", 3),
        momentum_periods=dual_config.get("momentum_periods", [126, 252]),
        regime_symbol=dual_config.get("regime_symbol", "SPY"),
        regime_confirmation_symbols=dual_config.get(
            "regime_confirmation_symbols",
        ),
        regime_confirmation_mode=dual_config.get(
            "regime_confirmation_mode",
            "primary",
        ),
        regime_sma_period=dual_config.get("regime_sma_period", 200),
        rebalance_frequency=dual_config.get(
            "rebalance_frequency",
            "monthly",
        ),
        target_exposure=dual_config.get("target_exposure", 1.0),
        benchmark_symbol=dual_config.get("benchmark_symbol", "SPY"),
        transaction_cost_bps=dual_config.get("transaction_cost_bps", 2.0),
        commission_bps=dual_config.get("commission_bps", 0.0),
        slippage_bps=dual_config.get("slippage_bps", 0.0),
        spread_cost_bps=dual_config.get("spread_cost_bps", 0.0),
        use_asset_trend_filter=dual_config.get(
            "use_asset_trend_filter",
            True,
        ),
        asset_sma_period=dual_config.get("asset_sma_period", 200),
        target_volatility=dual_config.get("target_volatility"),
        volatility_lookback=dual_config.get("volatility_lookback", 63),
        max_drawdown_guard=dual_config.get("max_drawdown_guard"),
        drawdown_guard_cooldown=dual_config.get(
            "drawdown_guard_cooldown",
            1,
        ),
        min_breadth_percent=dual_config.get("min_breadth_percent", 0),
        breadth_scaled_exposure_enabled=dual_config.get(
            "breadth_scaled_exposure_enabled",
            False,
        ),
        breadth_exposure_tiers=dual_config.get("breadth_exposure_tiers"),
        breadth_exposure_floor=dual_config.get(
            "breadth_exposure_floor",
            0,
        ),
        drawdown_recovery_scaling_enabled=dual_config.get(
            "drawdown_recovery_scaling_enabled",
            False,
        ),
        drawdown_recovery_exposure_caps=dual_config.get(
            "drawdown_recovery_exposure_caps",
        ),
        volatility_shock_filter_enabled=dual_config.get(
            "volatility_shock_filter_enabled",
            False,
        ),
        volatility_shock_symbol=dual_config.get("volatility_shock_symbol"),
        volatility_shock_short_lookback=dual_config.get(
            "volatility_shock_short_lookback",
            21,
        ),
        volatility_shock_long_lookback=dual_config.get(
            "volatility_shock_long_lookback",
            126,
        ),
        volatility_shock_ratio_threshold=dual_config.get(
            "volatility_shock_ratio_threshold",
            2.0,
        ),
        volatility_shock_exposure_multiplier=dual_config.get(
            "volatility_shock_exposure_multiplier",
            0.50,
        ),
        selection_mode=dual_config.get("selection_mode", "ranked"),
        min_selection_score=dual_config.get("min_selection_score", 0),
        max_selected_assets=dual_config.get("max_selected_assets"),
        weighting=dual_config.get("weighting", "equal"),
        max_position_weight=dual_config.get("max_position_weight"),
        weight_volatility_lookback=dual_config.get(
            "weight_volatility_lookback",
            dual_config.get("volatility_lookback", 63),
        ),
        strict_drawdown_kill_switch=dual_config.get(
            "strict_drawdown_kill_switch",
            False,
        ),
        risk_off_symbols=dual_config.get("risk_off_symbols", []),
        risk_off_top_n=dual_config.get("risk_off_top_n", 1),
        risk_off_momentum_periods=dual_config.get(
            "risk_off_momentum_periods",
        ),
        risk_regime_mode=dual_config.get("risk_regime_mode", "binary"),
        mixed_risk_exposure=dual_config.get("mixed_risk_exposure", 0.50),
        risk_off_risk_exposure=dual_config.get("risk_off_risk_exposure", 0),
        fast_reentry_enabled=dual_config.get("fast_reentry_enabled", False),
        fast_reentry_symbols=dual_config.get("fast_reentry_symbols", []),
        fast_reentry_sma_period=dual_config.get(
            "fast_reentry_sma_period",
            100,
        ),
        fast_reentry_momentum_period=dual_config.get(
            "fast_reentry_momentum_period",
            63,
        ),
        fast_reentry_breadth_percent=dual_config.get(
            "fast_reentry_breadth_percent",
            0.60,
        ),
        fallback_symbols=dual_config.get("fallback_symbols", []),
        fallback_allocation=dual_config.get("fallback_allocation", 0),
        fallback_min_risk_assets=dual_config.get(
            "fallback_min_risk_assets",
            3,
        ),
        fallback_momentum_periods=dual_config.get(
            "fallback_momentum_periods",
        ),
        decay_exit_enabled=dual_config.get("decay_exit_enabled", False),
        decay_momentum_period=dual_config.get("decay_momentum_period", 63),
        rank_drop_exit_top_n=dual_config.get("rank_drop_exit_top_n"),
        rank_deterioration_exit_enabled=dual_config.get(
            "rank_deterioration_exit_enabled",
            False,
        ),
        rank_deterioration_exit_rank=dual_config.get(
            "rank_deterioration_exit_rank",
        ),
        chop_filter_enabled=dual_config.get("chop_filter_enabled", False),
        chop_lookback=dual_config.get("chop_lookback", 63),
        min_chop_momentum=dual_config.get("min_chop_momentum", 0.02),
        chop_risk_exposure=dual_config.get("chop_risk_exposure", 0.50),
        quality_filter_enabled=dual_config.get(
            "quality_filter_enabled",
            False,
        ),
        quality_momentum_period=dual_config.get(
            "quality_momentum_period",
            21,
        ),
        quality_sma_period=dual_config.get("quality_sma_period", 50),
        quality_require_momentum_improving=dual_config.get(
            "quality_require_momentum_improving",
            False,
        ),
        avoid_short_term_weakness=dual_config.get(
            "avoid_short_term_weakness",
            False,
        ),
        short_term_momentum_period=dual_config.get(
            "short_term_momentum_period",
            21,
        ),
        short_term_momentum_floor=dual_config.get(
            "short_term_momentum_floor",
            -0.02,
        ),
        short_term_weakness_penalty_enabled=dual_config.get(
            "short_term_weakness_penalty_enabled",
            False,
        ),
        short_term_weakness_penalty_period=dual_config.get(
            "short_term_weakness_penalty_period",
            21,
        ),
        short_term_weakness_penalty_floor=dual_config.get(
            "short_term_weakness_penalty_floor",
            -0.02,
        ),
        short_term_weakness_penalty_weight=dual_config.get(
            "short_term_weakness_penalty_weight",
            1.0,
        ),
        cooldown_enabled=dual_config.get("cooldown_enabled", False),
        cooldown_periods=dual_config.get("cooldown_periods", 2),
        cooldown_loss_threshold=dual_config.get(
            "cooldown_loss_threshold",
            -0.03,
        ),
        rank_hysteresis_enabled=dual_config.get(
            "rank_hysteresis_enabled",
            False,
        ),
        rank_hysteresis_margin=dual_config.get("rank_hysteresis_margin", 2),
        rank_hysteresis_max_rank=dual_config.get(
            "rank_hysteresis_max_rank",
        ),
        max_rebalance_replacements=dual_config.get(
            "max_rebalance_replacements",
        ),
        replacement_score_gap=dual_config.get("replacement_score_gap", 0),
        rebalance_min_trade_weight=dual_config.get(
            "rebalance_min_trade_weight",
            0,
        ),
        rebalance_drift_band=dual_config.get("rebalance_drift_band", 0),
        eligible_for_paper_selector=dual_config.get(
            "eligible_for_paper_selector",
            True,
        ),
        eligible_for_production_selector=dual_config.get(
            "eligible_for_production_selector",
            False,
        ),
        leadership_filter_enabled=dual_config.get(
            "leadership_filter_enabled",
            False,
        ),
        leadership_symbol=dual_config.get("leadership_symbol", "SPY"),
        leadership_momentum_periods=dual_config.get(
            "leadership_momentum_periods",
        ),
        relative_strength_filter_enabled=dual_config.get(
            "relative_strength_filter_enabled",
            False,
        ),
        relative_strength_filter_symbol=dual_config.get(
            "relative_strength_filter_symbol",
            "SPY",
        ),
        relative_strength_filter_period=dual_config.get(
            "relative_strength_filter_period",
            63,
        ),
        relative_strength_filter_min_excess=dual_config.get(
            "relative_strength_filter_min_excess",
            0,
        ),
        benchmark_sleeve_symbols=dual_config.get(
            "benchmark_sleeve_symbols",
            [],
        ),
        benchmark_sleeve_allocation=dual_config.get(
            "benchmark_sleeve_allocation",
            0,
        ),
        benchmark_sleeve_momentum_periods=dual_config.get(
            "benchmark_sleeve_momentum_periods",
        ),
        benchmark_sleeve_top_n=dual_config.get("benchmark_sleeve_top_n", 1),
        benchmark_participation_filter_enabled=dual_config.get(
            "benchmark_participation_filter_enabled",
            False,
        ),
        benchmark_participation_period=dual_config.get(
            "benchmark_participation_period",
            63,
        ),
        benchmark_participation_min_return=dual_config.get(
            "benchmark_participation_min_return",
            0.03,
        ),
        benchmark_participation_max_selected_excess=dual_config.get(
            "benchmark_participation_max_selected_excess",
            0,
        ),
        sector_map=dual_config.get("sector_map"),
        max_sector_weight=dual_config.get("max_sector_weight"),
        ranking_score_mode=dual_config.get(
            "ranking_score_mode",
            "average_momentum",
        ),
        enhanced_momentum_periods=dual_config.get(
            "enhanced_momentum_periods",
        ),
        enhanced_momentum_weights=dual_config.get(
            "enhanced_momentum_weights",
        ),
        relative_strength_symbol=dual_config.get(
            "relative_strength_symbol",
            "SPY",
        ),
        relative_strength_periods=dual_config.get(
            "relative_strength_periods",
        ),
        relative_strength_weight=dual_config.get(
            "relative_strength_weight",
            0.25,
        ),
        volatility_penalty_weight=dual_config.get(
            "volatility_penalty_weight",
            0.05,
        ),
        ranking_volatility_lookback=dual_config.get(
            "ranking_volatility_lookback",
            63,
        ),
    )
