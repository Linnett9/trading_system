class DualMomentumConfigSnapshotMixin:

    def _config_snapshot(self):
        return {
            "experiment_name": self.experiment_name,
            "champion_id": self.champion_id,
            "champion_source_config_name": self.champion_source_config_name,
            "champion_config_path": self.champion_config_path,
            "top_n": self.top_n,
            "momentum_periods": self.momentum_periods,
            "regime_symbol": self.regime_symbol,
            "regime_confirmation_symbols": self.regime_confirmation_symbols,
            "regime_confirmation_mode": self.regime_confirmation_mode,
            "regime_sma_period": self.regime_sma_period,
            "rebalance_frequency": self.rebalance_frequency,
            "target_exposure": self.target_exposure,
            "benchmark_symbol": self.benchmark_symbol,
            "transaction_cost_bps": self.transaction_cost_bps,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "spread_cost_bps": self.spread_cost_bps,
            "effective_transaction_cost_bps": (
                self.effective_transaction_cost_bps
            ),
            "use_asset_trend_filter": self.use_asset_trend_filter,
            "asset_sma_period": self.asset_sma_period,
            "target_volatility": self.target_volatility,
            "volatility_lookback": self.volatility_lookback,
            "max_drawdown_guard": self.max_drawdown_guard,
            "drawdown_guard_cooldown": self.drawdown_guard_cooldown,
            "min_breadth_percent": self.min_breadth_percent,
            "breadth_scaled_exposure_enabled": (
                self.breadth_scaled_exposure_enabled
            ),
            "breadth_exposure_tiers": self.breadth_exposure_tiers,
            "breadth_exposure_floor": self.breadth_exposure_floor,
            "drawdown_recovery_scaling_enabled": (
                self.drawdown_recovery_scaling_enabled
            ),
            "drawdown_recovery_exposure_caps": (
                self.drawdown_recovery_exposure_caps
            ),
            "volatility_shock_filter_enabled": (
                self.volatility_shock_filter_enabled
            ),
            "volatility_shock_symbol": self.volatility_shock_symbol,
            "volatility_shock_short_lookback": (
                self.volatility_shock_short_lookback
            ),
            "volatility_shock_long_lookback": (
                self.volatility_shock_long_lookback
            ),
            "volatility_shock_ratio_threshold": (
                self.volatility_shock_ratio_threshold
            ),
            "volatility_shock_exposure_multiplier": (
                self.volatility_shock_exposure_multiplier
            ),
            "selection_mode": self.selection_mode,
            "min_selection_score": self.min_selection_score,
            "max_selected_assets": self.max_selected_assets,
            "weighting": self.weighting,
            "max_position_weight": self.max_position_weight,
            "weight_volatility_lookback": self.weight_volatility_lookback,
            "strict_drawdown_kill_switch": self.strict_drawdown_kill_switch,
            "risk_off_symbols": self.risk_off_symbols,
            "risk_off_top_n": self.risk_off_top_n,
            "risk_off_momentum_periods": self.risk_off_momentum_periods,
            "risk_regime_mode": self.risk_regime_mode,
            "mixed_risk_exposure": self.mixed_risk_exposure,
            "risk_off_risk_exposure": self.risk_off_risk_exposure,
            "fast_reentry_enabled": self.fast_reentry_enabled,
            "fast_reentry_symbols": self.fast_reentry_symbols,
            "fast_reentry_sma_period": self.fast_reentry_sma_period,
            "fast_reentry_momentum_period": self.fast_reentry_momentum_period,
            "fast_reentry_breadth_percent": self.fast_reentry_breadth_percent,
            "fallback_symbols": self.fallback_symbols,
            "fallback_allocation": self.fallback_allocation,
            "fallback_min_risk_assets": self.fallback_min_risk_assets,
            "fallback_momentum_periods": self.fallback_momentum_periods,
            "decay_exit_enabled": self.decay_exit_enabled,
            "decay_momentum_period": self.decay_momentum_period,
            "rank_drop_exit_top_n": self.rank_drop_exit_top_n,
            "rank_deterioration_exit_enabled": (
                self.rank_deterioration_exit_enabled
            ),
            "rank_deterioration_exit_rank": (
                self.rank_deterioration_exit_rank
            ),
            "chop_filter_enabled": self.chop_filter_enabled,
            "chop_lookback": self.chop_lookback,
            "min_chop_momentum": self.min_chop_momentum,
            "chop_risk_exposure": self.chop_risk_exposure,
            "quality_filter_enabled": self.quality_filter_enabled,
            "quality_momentum_period": self.quality_momentum_period,
            "quality_sma_period": self.quality_sma_period,
            "quality_require_momentum_improving": (
                self.quality_require_momentum_improving
            ),
            "avoid_short_term_weakness": self.avoid_short_term_weakness,
            "short_term_momentum_period": self.short_term_momentum_period,
            "short_term_momentum_floor": self.short_term_momentum_floor,
            "short_term_weakness_penalty_enabled": (
                self.short_term_weakness_penalty_enabled
            ),
            "short_term_weakness_penalty_period": (
                self.short_term_weakness_penalty_period
            ),
            "short_term_weakness_penalty_floor": (
                self.short_term_weakness_penalty_floor
            ),
            "short_term_weakness_penalty_weight": (
                self.short_term_weakness_penalty_weight
            ),
            "cooldown_enabled": self.cooldown_enabled,
            "cooldown_periods": self.cooldown_periods,
            "cooldown_loss_threshold": self.cooldown_loss_threshold,
            "rank_hysteresis_enabled": self.rank_hysteresis_enabled,
            "rank_hysteresis_margin": self.rank_hysteresis_margin,
            "rank_hysteresis_max_rank": self.rank_hysteresis_max_rank,
            "max_rebalance_replacements": self.max_rebalance_replacements,
            "replacement_score_gap": self.replacement_score_gap,
            "rebalance_min_trade_weight": self.rebalance_min_trade_weight,
            "rebalance_drift_band": self.rebalance_drift_band,
            "eligible_for_paper_selector": self.eligible_for_paper_selector,
            "eligible_for_production_selector": (
                self.eligible_for_production_selector
            ),
            "leadership_filter_enabled": self.leadership_filter_enabled,
            "leadership_symbol": self.leadership_symbol,
            "leadership_momentum_periods": self.leadership_momentum_periods,
            "relative_strength_filter_enabled": (
                self.relative_strength_filter_enabled
            ),
            "relative_strength_filter_symbol": (
                self.relative_strength_filter_symbol
            ),
            "relative_strength_filter_period": (
                self.relative_strength_filter_period
            ),
            "relative_strength_filter_min_excess": (
                self.relative_strength_filter_min_excess
            ),
            "benchmark_sleeve_symbols": self.benchmark_sleeve_symbols,
            "benchmark_sleeve_allocation": self.benchmark_sleeve_allocation,
            "benchmark_sleeve_momentum_periods": (
                self.benchmark_sleeve_momentum_periods
            ),
            "benchmark_sleeve_top_n": self.benchmark_sleeve_top_n,
            "benchmark_participation_filter_enabled": (
                self.benchmark_participation_filter_enabled
            ),
            "benchmark_participation_period": (
                self.benchmark_participation_period
            ),
            "benchmark_participation_min_return": (
                self.benchmark_participation_min_return
            ),
            "benchmark_participation_max_selected_excess": (
                self.benchmark_participation_max_selected_excess
            ),
            "max_sector_weight": self.max_sector_weight,
            "sector_map": self.sector_map,
            "ranking_score_mode": self.ranking_score_mode,
            "enhanced_momentum_periods": self.enhanced_momentum_periods,
            "enhanced_momentum_weights": self.enhanced_momentum_weights,
            "relative_strength_symbol": self.relative_strength_symbol,
            "relative_strength_periods": self.relative_strength_periods,
            "relative_strength_weight": self.relative_strength_weight,
            "volatility_penalty_weight": self.volatility_penalty_weight,
            "ranking_volatility_lookback": self.ranking_volatility_lookback,
        }
