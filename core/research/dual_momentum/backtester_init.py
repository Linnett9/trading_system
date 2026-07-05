from __future__ import annotations


class DualMomentumBacktesterInitMixin:
    def __init__(
        self,
        starting_equity: float = 500,
        experiment_name: str | None = None,
        champion_id: str | None = None,
        champion_source_config_name: str | None = None,
        champion_config_path: str | None = None,
        top_n: int = 3,
        momentum_periods: list[int] | None = None,
        regime_symbol: str = "SPY",
        regime_confirmation_symbols: list[str] | None = None,
        regime_confirmation_mode: str = "primary",
        regime_sma_period: int = 200,
        rebalance_frequency: str = "monthly",
        target_exposure: float = 1.0,
        benchmark_symbol: str = "SPY",
        transaction_cost_bps: float = 2.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        spread_cost_bps: float = 0.0,
        use_asset_trend_filter: bool = True,
        asset_sma_period: int = 200,
        target_volatility: float | None = None,
        volatility_lookback: int = 63,
        max_drawdown_guard: float | None = None,
        drawdown_guard_cooldown: int = 1,
        min_breadth_percent: float = 0,
        breadth_scaled_exposure_enabled: bool = False,
        breadth_exposure_tiers: list[list[float]] | None = None,
        breadth_exposure_floor: float = 0,
        drawdown_recovery_scaling_enabled: bool = False,
        drawdown_recovery_exposure_caps: list[list[float]] | None = None,
        volatility_shock_filter_enabled: bool = False,
        volatility_shock_symbol: str | None = None,
        volatility_shock_short_lookback: int = 21,
        volatility_shock_long_lookback: int = 126,
        volatility_shock_ratio_threshold: float = 2.0,
        volatility_shock_exposure_multiplier: float = 0.50,
        selection_mode: str = "ranked",
        min_selection_score: float = 0,
        max_selected_assets: int | None = None,
        weighting: str = "equal",
        max_position_weight: float | None = None,
        weight_volatility_lookback: int = 63,
        strict_drawdown_kill_switch: bool = False,
        risk_off_symbols: list[str] | None = None,
        risk_off_top_n: int = 1,
        risk_off_momentum_periods: list[int] | None = None,
        risk_regime_mode: str = "binary",
        mixed_risk_exposure: float = 0.50,
        risk_off_risk_exposure: float = 0,
        fast_reentry_enabled: bool = False,
        fast_reentry_symbols: list[str] | None = None,
        fast_reentry_sma_period: int = 100,
        fast_reentry_momentum_period: int = 63,
        fast_reentry_breadth_percent: float = 0.60,
        fallback_symbols: list[str] | None = None,
        fallback_allocation: float = 0,
        fallback_min_risk_assets: int = 3,
        fallback_momentum_periods: list[int] | None = None,
        decay_exit_enabled: bool = False,
        decay_momentum_period: int = 63,
        rank_drop_exit_top_n: int | None = None,
        rank_deterioration_exit_enabled: bool = False,
        rank_deterioration_exit_rank: int | None = None,
        chop_filter_enabled: bool = False,
        chop_lookback: int = 63,
        min_chop_momentum: float = 0.02,
        chop_risk_exposure: float = 0.50,
        quality_filter_enabled: bool = False,
        quality_momentum_period: int = 21,
        quality_sma_period: int = 50,
        quality_require_momentum_improving: bool = False,
        avoid_short_term_weakness: bool = False,
        short_term_momentum_period: int = 21,
        short_term_momentum_floor: float = -0.02,
        short_term_weakness_penalty_enabled: bool = False,
        short_term_weakness_penalty_period: int = 21,
        short_term_weakness_penalty_floor: float = -0.02,
        short_term_weakness_penalty_weight: float = 1.0,
        cooldown_enabled: bool = False,
        cooldown_periods: int = 2,
        cooldown_loss_threshold: float = -0.03,
        rank_hysteresis_enabled: bool = False,
        rank_hysteresis_margin: int = 2,
        rank_hysteresis_max_rank: int | None = None,
        max_rebalance_replacements: int | None = None,
        replacement_score_gap: float = 0,
        rebalance_min_trade_weight: float = 0,
        rebalance_drift_band: float = 0,
        eligible_for_paper_selector: bool = True,
        eligible_for_production_selector: bool = False,
        leadership_filter_enabled: bool = False,
        leadership_symbol: str = "SPY",
        leadership_momentum_periods: list[int] | None = None,
        relative_strength_filter_enabled: bool = False,
        relative_strength_filter_symbol: str = "SPY",
        relative_strength_filter_period: int = 63,
        relative_strength_filter_min_excess: float = 0,
        benchmark_sleeve_symbols: list[str] | None = None,
        benchmark_sleeve_allocation: float = 0,
        benchmark_sleeve_momentum_periods: list[int] | None = None,
        benchmark_sleeve_top_n: int = 1,
        benchmark_participation_filter_enabled: bool = False,
        benchmark_participation_period: int = 63,
        benchmark_participation_min_return: float = 0.03,
        benchmark_participation_max_selected_excess: float = 0,
        sector_map: dict[str, str] | None = None,
        max_sector_weight: float | None = None,
        ranking_score_mode: str = "average_momentum",
        enhanced_momentum_periods: list[int] | None = None,
        enhanced_momentum_weights: list[float] | None = None,
        relative_strength_symbol: str = "SPY",
        relative_strength_periods: list[int] | None = None,
        relative_strength_weight: float = 0.25,
        volatility_penalty_weight: float = 0.05,
        ranking_volatility_lookback: int = 63,
    ):
        self.starting_equity = starting_equity
        self.experiment_name = experiment_name
        self.champion_id = champion_id
        self.champion_source_config_name = champion_source_config_name
        self.champion_config_path = champion_config_path
        self.top_n = top_n
        self.momentum_periods = momentum_periods or [126, 252]
        self.regime_symbol = regime_symbol
        self.regime_confirmation_symbols = (
            regime_confirmation_symbols or [regime_symbol]
        )
        self.regime_confirmation_mode = regime_confirmation_mode
        self.regime_sma_period = regime_sma_period
        self.rebalance_frequency = rebalance_frequency
        self.target_exposure = target_exposure
        self.benchmark_symbol = benchmark_symbol
        self.transaction_cost_bps = transaction_cost_bps
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.spread_cost_bps = spread_cost_bps
        self.effective_transaction_cost_bps = (
            transaction_cost_bps
            + commission_bps
            + slippage_bps
            + spread_cost_bps
        )
        self.use_asset_trend_filter = use_asset_trend_filter
        self.asset_sma_period = asset_sma_period
        self.target_volatility = target_volatility
        self.volatility_lookback = volatility_lookback
        self.max_drawdown_guard = max_drawdown_guard
        self.drawdown_guard_cooldown = drawdown_guard_cooldown
        self.min_breadth_percent = min_breadth_percent
        self.breadth_scaled_exposure_enabled = (
            breadth_scaled_exposure_enabled
        )
        self.breadth_exposure_tiers = breadth_exposure_tiers or [
            [0.70, 1.00],
            [0.50, 0.75],
            [0.30, 0.50],
        ]
        self.breadth_exposure_floor = breadth_exposure_floor
        self.drawdown_recovery_scaling_enabled = (
            drawdown_recovery_scaling_enabled
        )
        self.drawdown_recovery_exposure_caps = (
            drawdown_recovery_exposure_caps
            or [
                [0.15, 0.50],
                [0.10, 0.75],
            ]
        )
        self.volatility_shock_filter_enabled = (
            volatility_shock_filter_enabled
        )
        self.volatility_shock_symbol = (
            volatility_shock_symbol or self.regime_symbol
        )
        self.volatility_shock_short_lookback = (
            volatility_shock_short_lookback
        )
        self.volatility_shock_long_lookback = volatility_shock_long_lookback
        self.volatility_shock_ratio_threshold = (
            volatility_shock_ratio_threshold
        )
        self.volatility_shock_exposure_multiplier = (
            volatility_shock_exposure_multiplier
        )
        self.selection_mode = selection_mode
        self.min_selection_score = min_selection_score
        self.max_selected_assets = max_selected_assets
        self.weighting = weighting
        self.max_position_weight = max_position_weight
        self.weight_volatility_lookback = weight_volatility_lookback
        self.strict_drawdown_kill_switch = strict_drawdown_kill_switch
        self.risk_off_symbols = risk_off_symbols or []
        self.risk_off_top_n = risk_off_top_n
        self.risk_off_momentum_periods = (
            risk_off_momentum_periods or self.momentum_periods
        )
        self.risk_regime_mode = risk_regime_mode
        self.mixed_risk_exposure = mixed_risk_exposure
        self.risk_off_risk_exposure = risk_off_risk_exposure
        self.fast_reentry_enabled = fast_reentry_enabled
        self.fast_reentry_symbols = fast_reentry_symbols or []
        self.fast_reentry_sma_period = fast_reentry_sma_period
        self.fast_reentry_momentum_period = fast_reentry_momentum_period
        self.fast_reentry_breadth_percent = fast_reentry_breadth_percent
        self.fallback_symbols = fallback_symbols or []
        self.fallback_allocation = fallback_allocation
        self.fallback_min_risk_assets = fallback_min_risk_assets
        self.fallback_momentum_periods = (
            fallback_momentum_periods or self.momentum_periods
        )
        self.decay_exit_enabled = decay_exit_enabled
        self.decay_momentum_period = decay_momentum_period
        self.rank_drop_exit_top_n = rank_drop_exit_top_n
        self.rank_deterioration_exit_enabled = (
            rank_deterioration_exit_enabled
        )
        self.rank_deterioration_exit_rank = rank_deterioration_exit_rank
        self.chop_filter_enabled = chop_filter_enabled
        self.chop_lookback = chop_lookback
        self.min_chop_momentum = min_chop_momentum
        self.chop_risk_exposure = chop_risk_exposure
        self.quality_filter_enabled = quality_filter_enabled
        self.quality_momentum_period = quality_momentum_period
        self.quality_sma_period = quality_sma_period
        self.quality_require_momentum_improving = (
            quality_require_momentum_improving
        )
        self.avoid_short_term_weakness = avoid_short_term_weakness
        self.short_term_momentum_period = short_term_momentum_period
        self.short_term_momentum_floor = short_term_momentum_floor
        self.short_term_weakness_penalty_enabled = (
            short_term_weakness_penalty_enabled
        )
        self.short_term_weakness_penalty_period = (
            short_term_weakness_penalty_period
        )
        self.short_term_weakness_penalty_floor = (
            short_term_weakness_penalty_floor
        )
        self.short_term_weakness_penalty_weight = (
            short_term_weakness_penalty_weight
        )
        self.cooldown_enabled = cooldown_enabled
        self.cooldown_periods = cooldown_periods
        self.cooldown_loss_threshold = cooldown_loss_threshold
        self.rank_hysteresis_enabled = rank_hysteresis_enabled
        self.rank_hysteresis_margin = rank_hysteresis_margin
        self.rank_hysteresis_max_rank = rank_hysteresis_max_rank
        self.max_rebalance_replacements = max_rebalance_replacements
        self.replacement_score_gap = replacement_score_gap
        self.rebalance_min_trade_weight = rebalance_min_trade_weight
        self.rebalance_drift_band = rebalance_drift_band
        self.eligible_for_paper_selector = eligible_for_paper_selector
        self.eligible_for_production_selector = eligible_for_production_selector
        self.leadership_filter_enabled = leadership_filter_enabled
        self.leadership_symbol = leadership_symbol
        self.leadership_momentum_periods = (
            leadership_momentum_periods or [21, 63]
        )
        self.relative_strength_filter_enabled = (
            relative_strength_filter_enabled
        )
        self.relative_strength_filter_symbol = relative_strength_filter_symbol
        self.relative_strength_filter_period = relative_strength_filter_period
        self.relative_strength_filter_min_excess = (
            relative_strength_filter_min_excess
        )
        self.benchmark_sleeve_symbols = benchmark_sleeve_symbols or []
        self.benchmark_sleeve_allocation = benchmark_sleeve_allocation
        self.benchmark_sleeve_momentum_periods = (
            benchmark_sleeve_momentum_periods or [63]
        )
        self.benchmark_sleeve_top_n = benchmark_sleeve_top_n
        self.benchmark_participation_filter_enabled = (
            benchmark_participation_filter_enabled
        )
        self.benchmark_participation_period = benchmark_participation_period
        self.benchmark_participation_min_return = (
            benchmark_participation_min_return
        )
        self.benchmark_participation_max_selected_excess = (
            benchmark_participation_max_selected_excess
        )
        self.sector_map = sector_map or {}
        self.max_sector_weight = max_sector_weight
        self.ranking_score_mode = ranking_score_mode
        self.enhanced_momentum_periods = (
            enhanced_momentum_periods or [21, 63, 126]
        )
        self.enhanced_momentum_weights = (
            enhanced_momentum_weights or [0.20, 0.35, 0.45]
        )
        self.relative_strength_symbol = relative_strength_symbol
        self.relative_strength_periods = relative_strength_periods or [21, 63]
        self.relative_strength_weight = relative_strength_weight
        self.volatility_penalty_weight = volatility_penalty_weight
        self.ranking_volatility_lookback = ranking_volatility_lookback
