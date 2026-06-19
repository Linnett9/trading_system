from core.entities.backtest_result import BacktestResult
from core.entities.signal_diagnostics import SignalDiagnostics
from core.research.dual_momentum.analytics import DualMomentumAnalyticsMixin
from core.research.dual_momentum.models import (
    DualMomentumSelection,
    DualMomentumResult,
)
from core.research.performance_metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from core.research.walk_forward import normalize_datetime
from core.services.portfolio_engine import EquityPoint
from core.research.dual_momentum.weighting import DualMomentumWeightingMixin
from core.research.dual_momentum.execution import DualMomentumExecutionMixin
from core.research.dual_momentum.ranking import DualMomentumRankingMixin
from core.research.dual_momentum.regimes import DualMomentumRegimeMixin
from core.research.dual_momentum.data import DualMomentumDataMixin
from core.research.dual_momentum.config_snapshot import (
    DualMomentumConfigSnapshotMixin,
)


class DualMomentumPortfolioBacktester(
    DualMomentumAnalyticsMixin,
    DualMomentumWeightingMixin,
    DualMomentumExecutionMixin,
    DualMomentumRankingMixin,
    DualMomentumRegimeMixin,
    DualMomentumDataMixin,
    DualMomentumConfigSnapshotMixin,
):
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

    def run(
        self,
        candles_by_symbol: dict[str, list],
        start_at=None,
        end_at=None,
    ) -> DualMomentumResult:
        prices_by_symbol = self._prices_by_symbol(candles_by_symbol)
        timestamps = self._common_timestamps(
            prices_by_symbol,
            start_at=start_at,
            end_at=end_at,
        )
        cash = self.starting_equity
        positions: dict[str, float] = {}
        entry_values: dict[str, float] = {}
        equity_curve = []
        returns = []
        selections = []
        trade_pnls = []
        exposure_values = []
        position_values = []
        turnover_value = 0
        estimated_cost = 0
        last_rebalance_key = None
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        peak_equity = self.starting_equity
        guard_rebalances_remaining = 0
        kill_switch_active = False
        cooldowns: dict[str, int] = {}

        for timestamp in timestamps:
            prices = self._prices_at(prices_by_symbol, timestamp)
            equity = self._equity(cash, positions, prices)
            peak_equity = max(peak_equity, equity)
            current_drawdown = (
                (peak_equity - equity) / peak_equity
                if peak_equity
                else 0
            )

            if self._should_rebalance(timestamp, last_rebalance_key):
                last_rebalance_key = self._rebalance_key(timestamp)
                self._tick_cooldowns(cooldowns)
                risk_on = self._risk_on(
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                )
                breadth_passes = self._breadth_passes(
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                )
                guard_active = self._drawdown_guard_active(
                    current_drawdown,
                    guard_rebalances_remaining,
                )
                kill_switch_triggered = False
                if (
                    self.max_drawdown_guard is not None
                    and current_drawdown >= self.max_drawdown_guard
                    and not kill_switch_active
                ):
                    if self.strict_drawdown_kill_switch:
                        kill_switch_active = True
                        kill_switch_triggered = True
                    else:
                        guard_rebalances_remaining = (
                            self.drawdown_guard_cooldown
                        )
                    guard_active = True

                if guard_rebalances_remaining > 0:
                    guard_rebalances_remaining -= 1

                if kill_switch_active:
                    guard_active = True
                    if (
                        not kill_switch_triggered
                        and risk_on
                        and breadth_passes
                    ):
                        kill_switch_active = False
                        guard_active = False
                        peak_equity = equity

                risk_assets_allowed = (
                    risk_on and breadth_passes and not guard_active
                )
                chop_filter_active = (
                    risk_assets_allowed
                    and self._chop_filter_active(
                        timestamp,
                        prices_by_symbol,
                    )
                )
                fast_reentry = (
                    not risk_assets_allowed
                    and not guard_active
                    and self.fast_reentry_enabled
                    and self._fast_reentry_signal(
                        timestamp=timestamp,
                        prices_by_symbol=prices_by_symbol,
                    )
                )
                partial_risk = (
                    not risk_assets_allowed
                    and not fast_reentry
                    and not guard_active
                    and self.risk_regime_mode == "scaled"
                    and self.risk_off_risk_exposure > 0
                )

                if risk_assets_allowed:
                    ranked = self._rank_symbols(
                        timestamp,
                        prices_by_symbol,
                        blocked_symbols=set(cooldowns),
                    )
                    selected = self._select_symbols(ranked)
                    selected = self._apply_rank_hysteresis(
                        selected,
                        ranked,
                        positions,
                    )
                    if chop_filter_active:
                        regime_exposure = self.chop_risk_exposure
                        regime_label = "chop-filter"
                    else:
                        regime_exposure = 1.0
                        regime_label = "risk-on"
                elif fast_reentry:
                    ranked = self._rank_symbols(
                        timestamp,
                        prices_by_symbol,
                        blocked_symbols=set(cooldowns),
                    )
                    selected = self._select_symbols(ranked)
                    selected = self._apply_rank_hysteresis(
                        selected,
                        ranked,
                        positions,
                    )
                    regime_exposure = self.mixed_risk_exposure
                    regime_label = "fast-reentry"
                elif partial_risk:
                    ranked = self._rank_symbols(
                        timestamp,
                        prices_by_symbol,
                        blocked_symbols=set(cooldowns),
                    )
                    selected = self._select_symbols(ranked)
                    selected = self._apply_rank_hysteresis(
                        selected,
                        ranked,
                        positions,
                    )
                    regime_exposure = self.risk_off_risk_exposure
                    regime_label = "partial-risk"
                elif self.risk_off_symbols:
                    ranked = self._rank_symbols(
                        timestamp,
                        prices_by_symbol,
                        allowed_symbols=set(self.risk_off_symbols),
                        momentum_periods=self.risk_off_momentum_periods,
                        apply_quality_filter=False,
                        apply_relative_strength_filter=False,
                    )
                    selected = [
                        symbol for symbol, _ in ranked[:self.risk_off_top_n]
                    ]
                    regime_exposure = 1.0
                    regime_label = "defensive"
                else:
                    ranked = []
                    selected = []
                    regime_exposure = 0
                    regime_label = "cash"

                if risk_assets_allowed or fast_reentry or partial_risk:
                    regime_exposure = self._scale_regime_exposure(
                        regime_exposure,
                        timestamp,
                        prices_by_symbol,
                        current_drawdown,
                    )

                target_weights = self._target_weights(
                    selected=selected,
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                )
                fallback_symbols = self._fallback_symbols(
                    selected=selected,
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                    risk_asset_mode=(
                        risk_assets_allowed or fast_reentry or partial_risk
                    ),
                )
                target_weights = self._apply_benchmark_sleeve_weights(
                    target_weights,
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                    risk_asset_mode=(
                        risk_assets_allowed or fast_reentry or partial_risk
                    ),
                )
                target_weights = self._apply_fallback_weights(
                    target_weights,
                    fallback_symbols,
                )
                selected = list(target_weights)
                exposure_target = (
                    self._target_exposure_for_rebalance(returns)
                    * regime_exposure
                )
                selections.append(
                    DualMomentumSelection(
                        timestamp=timestamp,
                        symbols=selected,
                        scores=dict(ranked),
                        risk_on=risk_assets_allowed,
                        regime_label=regime_label,
                        regime_exposure=regime_exposure,
                        exposure_target=exposure_target,
                        fallback_symbols=fallback_symbols,
                        breadth_passes=breadth_passes,
                        fast_reentry=fast_reentry,
                        drawdown_guard_active=guard_active,
                        target_weights=target_weights,
                        chop_filter_active=chop_filter_active,
                        cooldown_symbols=sorted(cooldowns),
                    )
                )
                (
                    cash,
                    pnls,
                    sold,
                    bought,
                    traded_value,
                    cost,
                    cooldown_symbols,
                ) = self._rebalance(
                    positions=positions,
                    entry_values=entry_values,
                    selected=selected,
                    target_weights=target_weights,
                    prices=prices,
                    cash=cash,
                    equity=equity,
                    target_exposure=exposure_target,
                )
                trade_pnls.extend(pnls)
                sell_signals += sold
                buy_signals += bought
                turnover_value += traded_value
                estimated_cost += cost
                self._apply_cooldowns(cooldowns, cooldown_symbols)
            else:
                if (
                    self.decay_exit_enabled
                    or self.rank_deterioration_exit_enabled
                ):
                    (
                        cash,
                        pnls,
                        sold,
                        traded_value,
                        cost,
                        cooldown_symbols,
                    ) = self._apply_decay_exits(
                        positions=positions,
                        entry_values=entry_values,
                        prices=prices,
                        timestamp=timestamp,
                        prices_by_symbol=prices_by_symbol,
                        cash=cash,
                    )
                    trade_pnls.extend(pnls)
                    sell_signals += sold
                    turnover_value += traded_value
                    estimated_cost += cost
                    self._apply_cooldowns(cooldowns, cooldown_symbols)
                hold_signals += 1

            equity = self._equity(cash, positions, prices)
            equity_curve.append(EquityPoint(timestamp=timestamp, equity=equity))

            if len(equity_curve) > 1:
                previous = equity_curve[-2].equity
                returns.append(
                    (equity - previous) / previous
                    if previous
                    else 0
                )

            exposure = self._position_value(positions, prices)
            exposure_values.append(exposure / equity if equity else 0)
            position_values.append(
                exposure / len(positions)
                if positions
                else 0
            )

        final_prices = (
            self._prices_at(prices_by_symbol, timestamps[-1])
            if timestamps
            else {}
        )
        final_equity = (
            self._equity(cash, positions, final_prices)
            if timestamps
            else self.starting_equity
        )
        result = BacktestResult(
            starting_equity=self.starting_equity,
            final_equity=final_equity,
            total_return=total_return(self.starting_equity, final_equity),
            max_drawdown=max_drawdown([point.equity for point in equity_curve]),
            sharpe=sharpe_ratio(returns),
            closed_trades=len(trade_pnls),
            open_trades=len(positions),
            equity_curve=equity_curve,
            profit_factor=self._profit_factor(trade_pnls),
            trade_analysis=self._trade_analysis(trade_pnls, exposure_values),
            capital_utilization=self._capital_utilization(
                exposure_values,
                position_values,
            ),
            signal_diagnostics=SignalDiagnostics(
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
            ),
        )
        benchmark_return = self._benchmark_return(prices_by_symbol, timestamps)
        equal_weight_return = self._equal_weight_benchmark(
            prices_by_symbol,
            timestamps,
        )
        elapsed_days = self._elapsed_days(equity_curve)
        elapsed_years = elapsed_days / 365.25 if elapsed_days > 0 else 0
        turnover_percent = (
            turnover_value / self.starting_equity
            if self.starting_equity
            else 0
        )
        annualized_turnover = (
            turnover_percent / elapsed_years
            if elapsed_years
            else 0
        )
        turnover_per_rebalance = (
            turnover_percent / len(selections)
            if selections
            else 0
        )
        cagr_value = cagr(
            self.starting_equity,
            final_equity,
            elapsed_days,
        )
        calmar_value = (
            cagr_value / result.max_drawdown
            if result.max_drawdown
            else 0
        )

        return DualMomentumResult(
            result=result,
            selections=selections,
            benchmark_return=benchmark_return,
            excess_return=result.total_return - benchmark_return,
            equal_weight_return=equal_weight_return,
            excess_vs_equal_weight=result.total_return - equal_weight_return,
            turnover_percent=turnover_percent,
            annualized_turnover_percent=annualized_turnover,
            turnover_per_rebalance_percent=turnover_per_rebalance,
            rebalance_count=len(selections),
            estimated_cost=estimated_cost,
            cost_drag_percent=(
                estimated_cost / self.starting_equity
                if self.starting_equity
                else 0
            ),
            cagr=cagr_value,
            calmar=calmar_value,
            annual_returns=self._period_returns(equity_curve, "annual"),
            monthly_returns=self._period_returns(equity_curve, "monthly"),
            rolling_12_month_returns=self._rolling_12_month_returns(
                equity_curve,
            ),
            drawdown_statistics=self._drawdown_statistics(equity_curve),
            config=self._config_snapshot(),
        )
