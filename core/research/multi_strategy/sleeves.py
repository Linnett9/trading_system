from core.entities.backtest_result import BacktestResult
from core.research.dual_momentum_portfolio import (
    DualMomentumPortfolioBacktester,
)
from core.research.performance_metrics import (
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from core.research.relative_strength_portfolio import (
    RelativeStrengthPortfolioBacktester,
)
from core.research.walk_forward import normalize_datetime
from core.services.portfolio_engine import EquityPoint


class MultiStrategySleeveMixin:

    def _normalized_sleeves(self):
        enabled = [
            sleeve
            for sleeve in self.sleeves
            if sleeve.get("enabled", True) and sleeve.get("weight", 0) > 0
        ]
        total_weight = sum(sleeve["weight"] for sleeve in enabled)

        if total_weight <= 0:
            return []

        return [
            {
                **sleeve,
                "weight": sleeve["weight"] / total_weight,
            }
            for sleeve in enabled
        ]

    def _build_sleeve_tester(self, sleeve_config, weight):
        starting_equity = self.starting_equity * weight
        parameters = sleeve_config.get("parameters", {})

        if sleeve_config["name"] == "relative_strength":
            return RelativeStrengthPortfolioBacktester(
                starting_equity=starting_equity,
                top_n=parameters.get("top_n", 2),
                momentum_periods=parameters.get(
                    "momentum_periods",
                    [63, 126],
                ),
                sma_period=parameters.get("sma_period", 200),
                rebalance_frequency=parameters.get(
                    "rebalance_frequency",
                    "monthly",
                ),
                target_exposure=parameters.get("target_exposure", 1.0),
                benchmark_symbol=parameters.get(
                    "benchmark_symbol",
                    self.benchmark_symbol,
                ),
                transaction_cost_bps=parameters.get(
                    "transaction_cost_bps",
                    2.0,
                ),
            )

        return DualMomentumPortfolioBacktester(
            starting_equity=starting_equity,
            top_n=parameters.get("top_n", 5),
            momentum_periods=parameters.get("momentum_periods", [63, 126]),
            regime_symbol=parameters.get("regime_symbol", "SPY"),
            regime_confirmation_symbols=parameters.get(
                "regime_confirmation_symbols",
            ),
            regime_confirmation_mode=parameters.get(
                "regime_confirmation_mode",
                "primary",
            ),
            regime_sma_period=parameters.get("regime_sma_period", 200),
            rebalance_frequency=parameters.get(
                "rebalance_frequency",
                "monthly",
            ),
            target_exposure=parameters.get("target_exposure", 1.0),
            benchmark_symbol=parameters.get(
                "benchmark_symbol",
                self.benchmark_symbol,
            ),
            transaction_cost_bps=parameters.get("transaction_cost_bps", 2.0),
            use_asset_trend_filter=parameters.get(
                "use_asset_trend_filter",
                True,
            ),
            asset_sma_period=parameters.get("asset_sma_period", 200),
            target_volatility=parameters.get("target_volatility"),
            volatility_lookback=parameters.get("volatility_lookback", 63),
            max_drawdown_guard=parameters.get("max_drawdown_guard"),
            drawdown_guard_cooldown=parameters.get(
                "drawdown_guard_cooldown",
                1,
            ),
            min_breadth_percent=parameters.get("min_breadth_percent", 0),
            breadth_scaled_exposure_enabled=parameters.get(
                "breadth_scaled_exposure_enabled",
                False,
            ),
            breadth_exposure_tiers=parameters.get("breadth_exposure_tiers"),
            breadth_exposure_floor=parameters.get(
                "breadth_exposure_floor",
                0,
            ),
            drawdown_recovery_scaling_enabled=parameters.get(
                "drawdown_recovery_scaling_enabled",
                False,
            ),
            drawdown_recovery_exposure_caps=parameters.get(
                "drawdown_recovery_exposure_caps",
            ),
            volatility_shock_filter_enabled=parameters.get(
                "volatility_shock_filter_enabled",
                False,
            ),
            volatility_shock_symbol=parameters.get("volatility_shock_symbol"),
            volatility_shock_short_lookback=parameters.get(
                "volatility_shock_short_lookback",
                21,
            ),
            volatility_shock_long_lookback=parameters.get(
                "volatility_shock_long_lookback",
                126,
            ),
            volatility_shock_ratio_threshold=parameters.get(
                "volatility_shock_ratio_threshold",
                2.0,
            ),
            volatility_shock_exposure_multiplier=parameters.get(
                "volatility_shock_exposure_multiplier",
                0.50,
            ),
            selection_mode=parameters.get("selection_mode", "ranked"),
            weighting=parameters.get("weighting", "equal"),
            max_position_weight=parameters.get("max_position_weight"),
            weight_volatility_lookback=parameters.get(
                "weight_volatility_lookback",
                parameters.get("volatility_lookback", 63),
            ),
            strict_drawdown_kill_switch=parameters.get(
                "strict_drawdown_kill_switch",
                False,
            ),
            risk_off_symbols=parameters.get("risk_off_symbols", []),
            risk_off_top_n=parameters.get("risk_off_top_n", 1),
            risk_off_momentum_periods=parameters.get(
                "risk_off_momentum_periods",
            ),
            risk_regime_mode=parameters.get("risk_regime_mode", "binary"),
            mixed_risk_exposure=parameters.get("mixed_risk_exposure", 0.50),
            risk_off_risk_exposure=parameters.get(
                "risk_off_risk_exposure",
                0,
            ),
            fast_reentry_enabled=parameters.get(
                "fast_reentry_enabled",
                False,
            ),
            fast_reentry_sma_period=parameters.get(
                "fast_reentry_sma_period",
                100,
            ),
            fast_reentry_momentum_period=parameters.get(
                "fast_reentry_momentum_period",
                63,
            ),
            fast_reentry_breadth_percent=parameters.get(
                "fast_reentry_breadth_percent",
                0.60,
            ),
            rank_deterioration_exit_enabled=parameters.get(
                "rank_deterioration_exit_enabled",
                False,
            ),
            rank_deterioration_exit_rank=parameters.get(
                "rank_deterioration_exit_rank",
            ),
            avoid_short_term_weakness=parameters.get(
                "avoid_short_term_weakness",
                False,
            ),
            short_term_momentum_period=parameters.get(
                "short_term_momentum_period",
                21,
            ),
            short_term_momentum_floor=parameters.get(
                "short_term_momentum_floor",
                -0.02,
            ),
            short_term_weakness_penalty_enabled=parameters.get(
                "short_term_weakness_penalty_enabled",
                False,
            ),
            short_term_weakness_penalty_period=parameters.get(
                "short_term_weakness_penalty_period",
                21,
            ),
            short_term_weakness_penalty_floor=parameters.get(
                "short_term_weakness_penalty_floor",
                -0.02,
            ),
            short_term_weakness_penalty_weight=parameters.get(
                "short_term_weakness_penalty_weight",
                1.0,
            ),
            rank_hysteresis_enabled=parameters.get(
                "rank_hysteresis_enabled",
                False,
            ),
            rank_hysteresis_margin=parameters.get(
                "rank_hysteresis_margin",
                2,
            ),
            rank_hysteresis_max_rank=parameters.get(
                "rank_hysteresis_max_rank",
            ),
            max_rebalance_replacements=parameters.get(
                "max_rebalance_replacements",
            ),
            replacement_score_gap=parameters.get("replacement_score_gap", 0),
            rebalance_min_trade_weight=parameters.get(
                "rebalance_min_trade_weight",
                0,
            ),
            relative_strength_filter_enabled=parameters.get(
                "relative_strength_filter_enabled",
                False,
            ),
            relative_strength_filter_symbol=parameters.get(
                "relative_strength_filter_symbol",
                "SPY",
            ),
            relative_strength_filter_period=parameters.get(
                "relative_strength_filter_period",
                63,
            ),
            relative_strength_filter_min_excess=parameters.get(
                "relative_strength_filter_min_excess",
                0,
            ),
            benchmark_sleeve_symbols=parameters.get(
                "benchmark_sleeve_symbols",
                [],
            ),
            benchmark_sleeve_allocation=parameters.get(
                "benchmark_sleeve_allocation",
                0,
            ),
            benchmark_sleeve_momentum_periods=parameters.get(
                "benchmark_sleeve_momentum_periods",
            ),
            benchmark_sleeve_top_n=parameters.get(
                "benchmark_sleeve_top_n",
                1,
            ),
            benchmark_participation_filter_enabled=parameters.get(
                "benchmark_participation_filter_enabled",
                False,
            ),
            benchmark_participation_period=parameters.get(
                "benchmark_participation_period",
                63,
            ),
            benchmark_participation_min_return=parameters.get(
                "benchmark_participation_min_return",
                0.03,
            ),
            benchmark_participation_max_selected_excess=parameters.get(
                "benchmark_participation_max_selected_excess",
                0,
            ),
            sector_map=parameters.get("sector_map"),
            max_sector_weight=parameters.get("max_sector_weight"),
        )

    def _run_sleeve(
        self,
        tester,
        sleeve_config,
        candles_by_symbol,
        start_at=None,
        end_at=None,
    ):
        if sleeve_config["name"] == "relative_strength":
            result = tester.run(candles_by_symbol).result
        else:
            result = tester.run(
                candles_by_symbol,
                start_at=start_at,
                end_at=end_at,
            ).result

        return self._trim_result(result, start_at=start_at, end_at=end_at)

    def _sleeve_candles(self, sleeve_config, candles_by_symbol):
        symbols = sleeve_config.get("parameters", {}).get("symbols")
        if not symbols:
            return candles_by_symbol

        return {
            symbol: candles_by_symbol[symbol]
            for symbol in symbols
            if symbol in candles_by_symbol
        }

    def _trim_result(self, result, start_at=None, end_at=None):
        if start_at is None and end_at is None:
            return result

        normalized_start = normalize_datetime(start_at) if start_at else None
        normalized_end = normalize_datetime(end_at) if end_at else None
        trimmed_curve = [
            point
            for point in result.equity_curve
            if (
                normalized_start is None
                or normalize_datetime(point.timestamp) >= normalized_start
            )
            and (
                normalized_end is None
                or normalize_datetime(point.timestamp) <= normalized_end
            )
        ]

        if not trimmed_curve:
            return BacktestResult(
                starting_equity=result.starting_equity,
                final_equity=result.starting_equity,
                total_return=0,
                max_drawdown=0,
                sharpe=0,
                closed_trades=0,
                open_trades=0,
                equity_curve=[],
            )

        first_equity = trimmed_curve[0].equity
        scale = result.starting_equity / first_equity if first_equity else 1
        scaled_curve = [
            EquityPoint(
                timestamp=point.timestamp,
                equity=point.equity * scale,
            )
            for point in trimmed_curve
        ]
        returns = []
        for index in range(1, len(scaled_curve)):
            previous = scaled_curve[index - 1].equity
            current = scaled_curve[index].equity
            returns.append((current / previous) - 1 if previous else 0)

        final_equity = scaled_curve[-1].equity
        return BacktestResult(
            starting_equity=result.starting_equity,
            final_equity=final_equity,
            total_return=total_return(result.starting_equity, final_equity),
            max_drawdown=max_drawdown(
                [point.equity for point in scaled_curve],
            ),
            sharpe=sharpe_ratio(returns),
            closed_trades=result.closed_trades,
            open_trades=result.open_trades,
            equity_curve=scaled_curve,
            profit_factor=result.profit_factor,
            trade_analysis=result.trade_analysis,
            capital_utilization=result.capital_utilization,
            signal_diagnostics=result.signal_diagnostics,
        )
