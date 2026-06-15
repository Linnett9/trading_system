from dataclasses import dataclass
from pathlib import Path
import json
import math

from core.entities.backtest_result import BacktestResult
from core.entities.capital_utilization import CapitalUtilization
from core.entities.signal_diagnostics import SignalDiagnostics
from core.entities.trade_analysis import TradeAnalysis
from core.research.performance_metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from core.research.walk_forward import normalize_datetime
from core.services.portfolio_engine import EquityPoint


@dataclass(frozen=True)
class DualMomentumSelection:
    timestamp: object
    symbols: list[str]
    scores: dict[str, float]
    risk_on: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbols": self.symbols,
            "scores": self.scores,
            "risk_on": self.risk_on,
        }


@dataclass(frozen=True)
class DualMomentumResult:
    result: BacktestResult
    selections: list[DualMomentumSelection]
    benchmark_return: float
    excess_return: float
    equal_weight_return: float
    excess_vs_equal_weight: float
    turnover_percent: float
    annualized_turnover_percent: float
    turnover_per_rebalance_percent: float
    rebalance_count: int
    estimated_cost: float
    cost_drag_percent: float
    cagr: float
    calmar: float
    annual_returns: dict[int, float]
    monthly_returns: dict[str, float]
    rolling_12_month_returns: dict[str, float]
    drawdown_statistics: dict
    config: dict

    def save_json(
        self,
        report_dir: str = "reports/summary",
        filename: str = "dual_momentum_portfolio.json",
    ) -> Path:
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        payload = {
            "result": self.result.to_dict(),
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "equal_weight_return": self.equal_weight_return,
            "excess_vs_equal_weight": self.excess_vs_equal_weight,
            "turnover_percent": self.turnover_percent,
            "annualized_turnover_percent": (
                self.annualized_turnover_percent
            ),
            "turnover_per_rebalance_percent": (
                self.turnover_per_rebalance_percent
            ),
            "rebalance_count": self.rebalance_count,
            "estimated_cost": self.estimated_cost,
            "cost_drag_percent": self.cost_drag_percent,
            "cagr": self.cagr,
            "calmar": self.calmar,
            "annual_returns": self.annual_returns,
            "monthly_returns": self.monthly_returns,
            "rolling_12_month_returns": self.rolling_12_month_returns,
            "drawdown_statistics": self.drawdown_statistics,
            "config": self.config,
            "selections": [
                selection.to_dict()
                for selection in self.selections
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class DualMomentumPortfolioBacktester:

    def __init__(
        self,
        starting_equity: float = 500,
        top_n: int = 3,
        momentum_periods: list[int] | None = None,
        regime_symbol: str = "SPY",
        regime_sma_period: int = 200,
        rebalance_frequency: str = "monthly",
        target_exposure: float = 1.0,
        benchmark_symbol: str = "SPY",
        transaction_cost_bps: float = 2.0,
        use_asset_trend_filter: bool = True,
        asset_sma_period: int = 200,
        target_volatility: float | None = None,
        volatility_lookback: int = 63,
        max_drawdown_guard: float | None = None,
        drawdown_guard_cooldown: int = 1,
        min_breadth_percent: float = 0,
        selection_mode: str = "ranked",
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
        fast_reentry_sma_period: int = 100,
        fast_reentry_momentum_period: int = 63,
        fast_reentry_breadth_percent: float = 0.60,
    ):
        self.starting_equity = starting_equity
        self.top_n = top_n
        self.momentum_periods = momentum_periods or [126, 252]
        self.regime_symbol = regime_symbol
        self.regime_sma_period = regime_sma_period
        self.rebalance_frequency = rebalance_frequency
        self.target_exposure = target_exposure
        self.benchmark_symbol = benchmark_symbol
        self.transaction_cost_bps = transaction_cost_bps
        self.use_asset_trend_filter = use_asset_trend_filter
        self.asset_sma_period = asset_sma_period
        self.target_volatility = target_volatility
        self.volatility_lookback = volatility_lookback
        self.max_drawdown_guard = max_drawdown_guard
        self.drawdown_guard_cooldown = drawdown_guard_cooldown
        self.min_breadth_percent = min_breadth_percent
        self.selection_mode = selection_mode
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
        self.fast_reentry_sma_period = fast_reentry_sma_period
        self.fast_reentry_momentum_period = fast_reentry_momentum_period
        self.fast_reentry_breadth_percent = fast_reentry_breadth_percent

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
                    ranked = self._rank_symbols(timestamp, prices_by_symbol)
                    selected = self._select_symbols(ranked)
                    regime_exposure = 1.0
                elif fast_reentry:
                    ranked = self._rank_symbols(timestamp, prices_by_symbol)
                    selected = self._select_symbols(ranked)
                    regime_exposure = self.mixed_risk_exposure
                elif partial_risk:
                    ranked = self._rank_symbols(timestamp, prices_by_symbol)
                    selected = self._select_symbols(ranked)
                    regime_exposure = self.risk_off_risk_exposure
                elif self.risk_off_symbols:
                    ranked = self._rank_symbols(
                        timestamp,
                        prices_by_symbol,
                        allowed_symbols=set(self.risk_off_symbols),
                        momentum_periods=self.risk_off_momentum_periods,
                    )
                    selected = [
                        symbol for symbol, _ in ranked[:self.risk_off_top_n]
                    ]
                    regime_exposure = 1.0
                else:
                    ranked = []
                    selected = []
                    regime_exposure = 0

                target_weights = self._target_weights(
                    selected=selected,
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                )
                selections.append(
                    DualMomentumSelection(
                        timestamp=timestamp,
                        symbols=selected,
                        scores=dict(ranked),
                        risk_on=risk_assets_allowed,
                    )
                )
                exposure_target = (
                    self._target_exposure_for_rebalance(returns)
                    * regime_exposure
                )
                (
                    cash,
                    pnls,
                    sold,
                    bought,
                    traded_value,
                    cost,
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
            else:
                hold_signals += 1

            equity = self._equity(cash, positions, prices)
            equity_curve.append(EquityPoint(timestamp=timestamp, equity=equity))

            if len(equity_curve) > 1:
                previous = equity_curve[-2].equity
                returns.append((equity - previous) / previous if previous else 0)

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
            config={
                "top_n": self.top_n,
                "momentum_periods": self.momentum_periods,
                "regime_symbol": self.regime_symbol,
                "regime_sma_period": self.regime_sma_period,
                "rebalance_frequency": self.rebalance_frequency,
                "target_exposure": self.target_exposure,
                "benchmark_symbol": self.benchmark_symbol,
                "transaction_cost_bps": self.transaction_cost_bps,
                "use_asset_trend_filter": self.use_asset_trend_filter,
                "asset_sma_period": self.asset_sma_period,
                "target_volatility": self.target_volatility,
                "volatility_lookback": self.volatility_lookback,
                "max_drawdown_guard": self.max_drawdown_guard,
                "drawdown_guard_cooldown": self.drawdown_guard_cooldown,
                "min_breadth_percent": self.min_breadth_percent,
                "selection_mode": self.selection_mode,
                "weighting": self.weighting,
                "max_position_weight": self.max_position_weight,
                "weight_volatility_lookback": (
                    self.weight_volatility_lookback
                ),
                "strict_drawdown_kill_switch": (
                    self.strict_drawdown_kill_switch
                ),
                "risk_off_symbols": self.risk_off_symbols,
                "risk_off_top_n": self.risk_off_top_n,
                "risk_off_momentum_periods": self.risk_off_momentum_periods,
                "risk_regime_mode": self.risk_regime_mode,
                "mixed_risk_exposure": self.mixed_risk_exposure,
                "risk_off_risk_exposure": self.risk_off_risk_exposure,
                "fast_reentry_enabled": self.fast_reentry_enabled,
                "fast_reentry_sma_period": self.fast_reentry_sma_period,
                "fast_reentry_momentum_period": (
                    self.fast_reentry_momentum_period
                ),
                "fast_reentry_breadth_percent": (
                    self.fast_reentry_breadth_percent
                ),
            },
        )

    def _rebalance(
        self,
        positions,
        entry_values,
        selected,
        target_weights,
        prices,
        cash,
        equity,
        target_exposure,
    ):
        pnls = []
        sold = 0
        bought = 0
        traded_value = 0
        total_cost = 0
        selected_symbols = set(selected)

        for symbol in list(positions):
            if symbol in selected_symbols:
                continue

            value = positions[symbol] * prices[symbol]
            cost = self._transaction_cost(value)
            cash += value - cost
            pnls.append(value - entry_values.get(symbol, value) - cost)
            traded_value += value
            total_cost += cost
            sold += 1
            del positions[symbol]
            entry_values.pop(symbol, None)

        if not selected:
            return cash, pnls, sold, bought, traded_value, total_cost

        for symbol in selected:
            if symbol not in prices or prices[symbol] <= 0:
                continue

            target_value = equity * target_exposure * target_weights.get(
                symbol,
                0,
            )
            if target_value <= 0:
                continue

            current_value = positions.get(symbol, 0) * prices[symbol]
            difference = target_value - current_value

            if difference > 0:
                value = min(difference, cash)
                cost = self._transaction_cost(value)
                investable_value = max(0, value - cost)
                positions[symbol] = (
                    positions.get(symbol, 0)
                    + investable_value / prices[symbol]
                )
                entry_values[symbol] = entry_values.get(symbol, 0) + value
                cash -= value
                traded_value += investable_value
                total_cost += cost
                bought += 1
                continue

            if difference < 0 and symbol in positions:
                sell_value = min(abs(difference), current_value)
                quantity_to_sell = sell_value / prices[symbol]
                cost = self._transaction_cost(sell_value)
                original_entry = entry_values.get(symbol, current_value)
                entry_reduction = (
                    original_entry
                    * (sell_value / current_value)
                    if current_value
                    else 0
                )
                cash += sell_value - cost
                positions[symbol] -= quantity_to_sell
                entry_values[symbol] = original_entry - entry_reduction
                traded_value += sell_value
                total_cost += cost

                if positions[symbol] <= 1e-12:
                    pnls.append(sell_value - entry_reduction - cost)
                    del positions[symbol]
                    entry_values.pop(symbol, None)
                    sold += 1

        return cash, pnls, sold, bought, traded_value, total_cost

    def _select_symbols(self, ranked):
        if self.selection_mode == "all_positive":
            return [symbol for symbol, _ in ranked]

        return [symbol for symbol, _ in ranked[:self.top_n]]

    def _target_weights(self, selected, timestamp, prices_by_symbol):
        if not selected:
            return {}

        if self.weighting == "inverse_volatility":
            weights = self._inverse_volatility_weights(
                selected,
                timestamp,
                prices_by_symbol,
            )
        else:
            weights = {
                symbol: 1 / len(selected)
                for symbol in selected
            }

        return self._cap_weights(weights)

    def _inverse_volatility_weights(
        self,
        selected,
        timestamp,
        prices_by_symbol,
    ):
        inverse_volatilities = {}

        for symbol in selected:
            prices = prices_by_symbol.get(symbol)
            if not prices:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)
            if index is None:
                continue

            volatility = self._realized_volatility(
                prices,
                timestamps,
                index,
                self.weight_volatility_lookback,
            )
            inverse_volatilities[symbol] = (
                1 / volatility
                if volatility > 0
                else 1
            )

        total = sum(inverse_volatilities.values())
        if total <= 0:
            return {
                symbol: 1 / len(selected)
                for symbol in selected
            }

        return {
            symbol: inverse_volatilities.get(symbol, 0) / total
            for symbol in selected
        }

    def _cap_weights(self, weights):
        if self.max_position_weight is None:
            return weights

        remaining_symbols = set(weights)
        remaining_weight = 1.0
        capped = {}

        while remaining_symbols and remaining_weight > 0:
            raw_total = sum(weights[symbol] for symbol in remaining_symbols)
            if raw_total <= 0:
                break

            capped_this_round = False
            for symbol in list(remaining_symbols):
                proposed_weight = (
                    remaining_weight
                    * weights[symbol]
                    / raw_total
                )
                if proposed_weight > self.max_position_weight:
                    capped[symbol] = self.max_position_weight
                    remaining_weight -= self.max_position_weight
                    remaining_symbols.remove(symbol)
                    capped_this_round = True

            if not capped_this_round:
                for symbol in remaining_symbols:
                    capped[symbol] = (
                        remaining_weight
                        * weights[symbol]
                        / raw_total
                    )
                break

        return capped

    def _rank_symbols(
        self,
        timestamp,
        prices_by_symbol,
        allowed_symbols=None,
        momentum_periods=None,
    ):
        ranked = []
        periods = momentum_periods or self.momentum_periods

        for symbol, prices in prices_by_symbol.items():
            if symbol == self.regime_symbol:
                continue

            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)
            if index is None:
                continue

            if (
                self.use_asset_trend_filter
                and not self._above_sma(prices, timestamps, index)
            ):
                continue

            score = self._momentum_score(prices, timestamps, index, periods)
            if score is not None and score > 0:
                ranked.append((symbol, score))

        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _risk_on(self, timestamp, prices_by_symbol):
        prices = prices_by_symbol.get(self.regime_symbol)
        if not prices:
            return False

        timestamps = sorted(prices)
        index = self._timestamp_index(timestamps, timestamp)
        if index is None or index < self.regime_sma_period:
            return False

        close = prices[timestamps[index]]
        sma = sum(
            prices[timestamps[position]]
            for position in range(
                index - self.regime_sma_period + 1,
                index + 1,
            )
        ) / self.regime_sma_period

        return close > sma

    def _fast_reentry_signal(self, timestamp, prices_by_symbol):
        return (
            self._regime_above_sma(
                timestamp,
                prices_by_symbol,
                self.fast_reentry_sma_period,
            )
            or self._regime_momentum_positive(
                timestamp,
                prices_by_symbol,
                self.fast_reentry_momentum_period,
            )
            or self._breadth_passes_threshold(
                timestamp,
                prices_by_symbol,
                self.fast_reentry_breadth_percent,
            )
        )

    def _regime_above_sma(self, timestamp, prices_by_symbol, period):
        prices = prices_by_symbol.get(self.regime_symbol)
        if not prices:
            return False

        timestamps = sorted(prices)
        index = self._timestamp_index(timestamps, timestamp)
        if index is None or index < period:
            return False

        close = prices[timestamps[index]]
        sma = sum(
            prices[timestamps[position]]
            for position in range(index - period + 1, index + 1)
        ) / period
        return close > sma

    def _regime_momentum_positive(self, timestamp, prices_by_symbol, period):
        prices = prices_by_symbol.get(self.regime_symbol)
        if not prices:
            return False

        timestamps = sorted(prices)
        index = self._timestamp_index(timestamps, timestamp)
        if index is None or index < period:
            return False

        previous = prices[timestamps[index - period]]
        current = prices[timestamps[index]]
        return previous > 0 and current > previous

    def _above_sma(self, prices, timestamps, index):
        if index < self.asset_sma_period:
            return False

        close = prices[timestamps[index]]
        sma = sum(
            prices[timestamps[position]]
            for position in range(
                index - self.asset_sma_period + 1,
                index + 1,
            )
        ) / self.asset_sma_period

        return close > sma

    def _breadth_passes(self, timestamp, prices_by_symbol):
        if self.min_breadth_percent <= 0:
            return True

        return self._breadth_passes_threshold(
            timestamp,
            prices_by_symbol,
            self.min_breadth_percent,
        )

    def _breadth_passes_threshold(
        self,
        timestamp,
        prices_by_symbol,
        threshold,
    ):
        if threshold <= 0:
            return True

        checked = 0
        passing = 0

        for symbol, prices in prices_by_symbol.items():
            if symbol == self.regime_symbol:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)
            if index is None or index < self.asset_sma_period:
                continue

            checked += 1
            if self._above_sma(prices, timestamps, index):
                passing += 1

        if checked == 0:
            return False

        return (passing / checked) >= threshold

    def _target_exposure_for_rebalance(self, returns):
        if self.target_volatility is None:
            return self.target_exposure

        if len(returns) < self.volatility_lookback:
            return self.target_exposure

        recent = returns[-self.volatility_lookback:]
        mean = sum(recent) / len(recent)
        variance = sum((value - mean) ** 2 for value in recent) / len(recent)
        annualized_volatility = math.sqrt(variance) * math.sqrt(252)

        if annualized_volatility <= 0:
            return self.target_exposure

        volatility_scalar = self.target_volatility / annualized_volatility
        return min(self.target_exposure, self.target_exposure * volatility_scalar)

    def _realized_volatility(self, prices, timestamps, index, lookback):
        if index < 1:
            return 0

        start = max(1, index - lookback + 1)
        returns = []

        for position in range(start, index + 1):
            previous = prices[timestamps[position - 1]]
            current = prices[timestamps[position]]
            if previous:
                returns.append((current / previous) - 1)

        if not returns:
            return 0

        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        return math.sqrt(variance)

    def _drawdown_guard_active(
        self,
        current_drawdown,
        guard_rebalances_remaining,
    ):
        if self.max_drawdown_guard is None:
            return False

        return (
            current_drawdown >= self.max_drawdown_guard
            or guard_rebalances_remaining > 0
        )

    def _momentum_score(self, prices, timestamps, index, periods=None):
        scores = []
        periods = periods or self.momentum_periods

        for period in periods:
            if index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]
            if previous <= 0:
                return None

            scores.append((current / previous) - 1)

        return sum(scores) / len(scores) if scores else None

    def _prices_by_symbol(self, candles_by_symbol):
        return {
            symbol: {
                candle.timestamp: candle.close
                for candle in candles
            }
            for symbol, candles in candles_by_symbol.items()
            if candles
        }

    def _common_timestamps(self, prices_by_symbol, start_at=None, end_at=None):
        if not prices_by_symbol:
            return []

        common = set.intersection(
            *[
                set(prices.keys())
                for prices in prices_by_symbol.values()
            ]
        )
        max_lookback = max(
            [
                self.regime_sma_period,
                self.asset_sma_period if self.use_asset_trend_filter else 0,
            ]
            + self.momentum_periods
        )
        timestamps = sorted(common)[max_lookback:]

        if start_at is not None:
            normalized_start = normalize_datetime(start_at)
            timestamps = [
                timestamp for timestamp in timestamps
                if normalize_datetime(timestamp) >= normalized_start
            ]

        if end_at is not None:
            normalized_end = normalize_datetime(end_at)
            timestamps = [
                timestamp for timestamp in timestamps
                if normalize_datetime(timestamp) <= normalized_end
            ]

        return timestamps

    def _prices_at(self, prices_by_symbol, timestamp):
        return {
            symbol: prices[timestamp]
            for symbol, prices in prices_by_symbol.items()
            if timestamp in prices
        }

    def _timestamp_index(self, timestamps, timestamp):
        try:
            return timestamps.index(timestamp)
        except ValueError:
            return None

    def _should_rebalance(self, timestamp, last_rebalance_key):
        return self._rebalance_key(timestamp) != last_rebalance_key

    def _rebalance_key(self, timestamp):
        if self.rebalance_frequency == "weekly":
            calendar = timestamp.isocalendar()
            return calendar.year, calendar.week

        return timestamp.year, timestamp.month

    def _equity(self, cash, positions, prices):
        return cash + self._position_value(positions, prices)

    def _position_value(self, positions, prices):
        return sum(
            quantity * prices[symbol]
            for symbol, quantity in positions.items()
            if symbol in prices
        )

    def _transaction_cost(self, trade_value):
        return trade_value * (self.transaction_cost_bps / 10_000)

    def _profit_factor(self, pnls):
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl <= 0))
        return gross_profit / gross_loss if gross_loss else 0

    def _trade_analysis(self, pnls, exposure_values):
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        return TradeAnalysis(
            total_trades=len(pnls),
            win_rate=len(wins) / len(pnls) if pnls else 0,
            average_win=gross_profit / len(wins) if wins else 0,
            average_loss=sum(losses) / len(losses) if losses else 0,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            expectancy=sum(pnls) / len(pnls) if pnls else 0,
            profit_factor=(
                gross_profit / gross_loss
                if gross_loss
                else 0
            ),
            time_in_market_percent=(
                sum(1 for exposure in exposure_values if exposure > 0)
                / len(exposure_values)
                if exposure_values
                else 0
            ),
        )

    def _capital_utilization(self, exposure_values, position_values):
        average_exposure = (
            sum(exposure_values) / len(exposure_values)
            if exposure_values
            else 0
        )
        return CapitalUtilization(
            average_position_value=(
                sum(position_values) / len(position_values)
                if position_values
                else 0
            ),
            average_exposure_percent=average_exposure,
            max_exposure_percent=max(exposure_values) if exposure_values else 0,
            average_cash_percent=1 - average_exposure,
            average_leverage=average_exposure,
        )

    def _benchmark_return(self, prices_by_symbol, timestamps):
        if not timestamps:
            return 0

        prices = prices_by_symbol.get(self.benchmark_symbol)
        if not prices:
            return self._equal_weight_benchmark(prices_by_symbol, timestamps)

        start = prices[timestamps[0]]
        end = prices[timestamps[-1]]
        return (end / start) - 1 if start else 0

    def _equal_weight_benchmark(self, prices_by_symbol, timestamps):
        returns = []

        for symbol, prices in prices_by_symbol.items():
            if symbol == self.regime_symbol:
                continue

            start = prices.get(timestamps[0]) if timestamps else None
            end = prices.get(timestamps[-1]) if timestamps else None
            if start:
                returns.append((end / start) - 1)

        return sum(returns) / len(returns) if returns else 0

    def _period_returns(self, equity_curve, period):
        if not equity_curve:
            return {}

        grouped = {}
        for point in equity_curve:
            if period == "annual":
                key = point.timestamp.year
            else:
                key = point.timestamp.strftime("%Y-%m")
            grouped.setdefault(key, []).append(point.equity)

        return {
            key: (values[-1] / values[0]) - 1 if values[0] else 0
            for key, values in grouped.items()
        }

    def _rolling_12_month_returns(self, equity_curve):
        if len(equity_curve) < 252:
            return {}

        returns = {}
        for index in range(252, len(equity_curve)):
            start = equity_curve[index - 252]
            end = equity_curve[index]
            if start.equity:
                returns[end.timestamp.strftime("%Y-%m-%d")] = (
                    end.equity / start.equity
                ) - 1

        return returns

    def _elapsed_days(self, equity_curve):
        if len(equity_curve) < 2:
            return 0

        return (equity_curve[-1].timestamp - equity_curve[0].timestamp).days

    def _drawdown_statistics(self, equity_curve):
        peak = None
        max_dd = 0
        current_drawdown = 0
        drawdowns = []
        longest_days = 0
        current_start = None

        for point in equity_curve:
            if peak is None or point.equity >= peak:
                peak = point.equity
                current_start = None
                current_drawdown = 0
                continue

            current_drawdown = (peak - point.equity) / peak if peak else 0
            drawdowns.append(current_drawdown)
            max_dd = max(max_dd, current_drawdown)

            if current_start is None:
                current_start = point.timestamp

            longest_days = max(
                longest_days,
                (point.timestamp - current_start).days,
            )

        return {
            "max_drawdown": max_dd,
            "average_drawdown": (
                sum(drawdowns) / len(drawdowns)
                if drawdowns
                else 0
            ),
            "current_drawdown": current_drawdown,
            "longest_drawdown_days": longest_days,
        }
