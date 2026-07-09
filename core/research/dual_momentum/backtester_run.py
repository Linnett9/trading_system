from __future__ import annotations

from core.entities.backtest_result import BacktestResult
from core.entities.signal_diagnostics import SignalDiagnostics
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
from core.services.portfolio_engine import EquityPoint


class DualMomentumBacktesterRunMixin:
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
                candidate_count = 0
                selected_count_before_hysteresis = 0

                if risk_assets_allowed:
                    ranked = self._rank_symbols(
                        timestamp,
                        prices_by_symbol,
                        blocked_symbols=set(cooldowns),
                    )
                    candidate_count = len(ranked)
                    selected = self._select_symbols(ranked)
                    selected_count_before_hysteresis = len(selected)
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
                    candidate_count = len(ranked)
                    selected = self._select_symbols(ranked)
                    selected_count_before_hysteresis = len(selected)
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
                    candidate_count = len(ranked)
                    selected = self._select_symbols(ranked)
                    selected_count_before_hysteresis = len(selected)
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
                    candidate_count = len(ranked)
                    selected = [
                        symbol for symbol, _ in ranked[:self.risk_off_top_n]
                    ]
                    selected_count_before_hysteresis = len(selected)
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
                        candidate_count=candidate_count,
                        selected_count_before_hysteresis=(
                            selected_count_before_hysteresis
                        ),
                        final_holding_count=len(selected),
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
