from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import json

from core.entities.backtest_result import BacktestResult
from core.entities.capital_utilization import CapitalUtilization
from core.entities.signal_diagnostics import SignalDiagnostics
from core.entities.trade_analysis import TradeAnalysis
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


@dataclass(frozen=True)
class MultiStrategySleeveResult:
    name: str
    weight: float
    result: BacktestResult

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class MultiStrategyPortfolioResult:
    result: BacktestResult
    sleeves: list[MultiStrategySleeveResult]
    benchmark_return: float
    excess_return: float
    equal_weight_return: float
    excess_vs_equal_weight: float
    diagnostics: dict
    config: dict

    def save_json(
        self,
        report_dir: str = "reports/summary",
        filename: str = "multi_strategy_portfolio.json",
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
            "diagnostics": self.diagnostics,
            "config": self.config,
            "sleeves": [sleeve.to_dict() for sleeve in self.sleeves],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class MultiStrategyPortfolioBacktester:

    def __init__(
        self,
        starting_equity: float = 500,
        sleeves: list[dict] | None = None,
        benchmark_symbol: str = "SPY",
        warmup_days: int = 500,
    ):
        self.starting_equity = starting_equity
        self.sleeves = sleeves or []
        self.benchmark_symbol = benchmark_symbol
        self.warmup_days = warmup_days

    def run(
        self,
        candles_by_symbol: dict[str, list],
        start_at=None,
        end_at=None,
    ) -> MultiStrategyPortfolioResult:
        normalized_sleeves = self._normalized_sleeves()
        sleeve_results = []

        for sleeve_config in normalized_sleeves:
            weight = sleeve_config["weight"]
            tester = self._build_sleeve_tester(sleeve_config, weight)
            sleeve_candles = self._slice_candles(
                candles_by_symbol,
                start_at=start_at,
                end_at=end_at,
            )
            result = self._run_sleeve(
                tester,
                sleeve_config,
                self._sleeve_candles(sleeve_config, sleeve_candles),
                start_at=start_at,
                end_at=end_at,
            )
            sleeve_results.append(
                MultiStrategySleeveResult(
                    name=sleeve_config["name"],
                    weight=weight,
                    result=result,
                )
            )

        combined_result = self._combine_results(sleeve_results)
        prices_by_symbol = self._prices_by_symbol(candles_by_symbol)
        timestamps = [point.timestamp for point in combined_result.equity_curve]
        benchmark_return = self._benchmark_return(prices_by_symbol, timestamps)
        equal_weight_return = self._equal_weight_return(
            prices_by_symbol,
            timestamps,
        )
        diagnostics = self._diagnostics(
            combined_result=combined_result,
            sleeve_results=sleeve_results,
            prices_by_symbol=prices_by_symbol,
            timestamps=timestamps,
        )

        return MultiStrategyPortfolioResult(
            result=combined_result,
            sleeves=sleeve_results,
            benchmark_return=benchmark_return,
            excess_return=combined_result.total_return - benchmark_return,
            equal_weight_return=equal_weight_return,
            excess_vs_equal_weight=(
                combined_result.total_return - equal_weight_return
            ),
            diagnostics=diagnostics,
            config={
                "benchmark_symbol": self.benchmark_symbol,
                "warmup_days": self.warmup_days,
                "sleeves": normalized_sleeves,
            },
        )

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

    def _combine_results(self, sleeve_results):
        if not sleeve_results:
            return BacktestResult(
                starting_equity=self.starting_equity,
                final_equity=self.starting_equity,
                total_return=0,
                max_drawdown=0,
                sharpe=0,
                closed_trades=0,
                open_trades=0,
                equity_curve=[],
            )

        common_timestamps = set(
            point.timestamp
            for point in sleeve_results[0].result.equity_curve
        )
        for sleeve in sleeve_results[1:]:
            common_timestamps &= set(
                point.timestamp for point in sleeve.result.equity_curve
            )

        timestamps = sorted(common_timestamps)
        equity_by_sleeve = {
            sleeve.name: {
                point.timestamp: point.equity
                for point in sleeve.result.equity_curve
            }
            for sleeve in sleeve_results
        }
        equity_curve = [
            EquityPoint(
                timestamp=timestamp,
                equity=sum(
                    equity_by_sleeve[sleeve.name][timestamp]
                    for sleeve in sleeve_results
                ),
            )
            for timestamp in timestamps
        ]
        returns = []
        for index in range(1, len(equity_curve)):
            previous = equity_curve[index - 1].equity
            current = equity_curve[index].equity
            returns.append((current / previous) - 1 if previous else 0)

        final_equity = (
            equity_curve[-1].equity
            if equity_curve
            else self.starting_equity
        )
        exposure = self._weighted_metric(
            sleeve_results,
            "capital_utilization",
            "average_exposure_percent",
        )
        time_in_market = self._weighted_metric(
            sleeve_results,
            "trade_analysis",
            "time_in_market_percent",
        )

        return BacktestResult(
            starting_equity=self.starting_equity,
            final_equity=final_equity,
            total_return=total_return(self.starting_equity, final_equity),
            max_drawdown=max_drawdown([point.equity for point in equity_curve]),
            sharpe=sharpe_ratio(returns),
            closed_trades=sum(
                sleeve.result.closed_trades for sleeve in sleeve_results
            ),
            open_trades=sum(
                sleeve.result.open_trades for sleeve in sleeve_results
            ),
            equity_curve=equity_curve,
            profit_factor=self._average_profit_factor(sleeve_results),
            trade_analysis=TradeAnalysis(
                total_trades=sum(
                    sleeve.result.closed_trades for sleeve in sleeve_results
                ),
                profit_factor=self._average_profit_factor(sleeve_results),
                time_in_market_percent=time_in_market,
            ),
            capital_utilization=CapitalUtilization(
                average_exposure_percent=exposure,
                max_exposure_percent=max(
                    (
                        sleeve.result.capital_utilization.max_exposure_percent
                        * sleeve.weight
                        for sleeve in sleeve_results
                    ),
                    default=0,
                ),
                average_cash_percent=1 - exposure,
                average_leverage=exposure,
            ),
            signal_diagnostics=SignalDiagnostics(
                buy_signals=sum(
                    sleeve.result.signal_diagnostics.buy_signals
                    for sleeve in sleeve_results
                ),
                sell_signals=sum(
                    sleeve.result.signal_diagnostics.sell_signals
                    for sleeve in sleeve_results
                ),
                hold_signals=sum(
                    sleeve.result.signal_diagnostics.hold_signals
                    for sleeve in sleeve_results
                ),
            ),
        )

    def _weighted_metric(self, sleeve_results, parent_name, field_name):
        return sum(
            getattr(getattr(sleeve.result, parent_name), field_name)
            * sleeve.weight
            for sleeve in sleeve_results
        )

    def _average_profit_factor(self, sleeve_results):
        values = [
            sleeve.result.profit_factor
            for sleeve in sleeve_results
            if sleeve.result.profit_factor > 0
        ]
        return sum(values) / len(values) if values else 0

    def _slice_candles(self, candles_by_symbol, start_at=None, end_at=None):
        if start_at is None and end_at is None:
            return candles_by_symbol

        warmup_start = (
            normalize_datetime(start_at) - timedelta(days=self.warmup_days)
            if start_at is not None
            else None
        )
        normalized_end = normalize_datetime(end_at) if end_at is not None else None

        return {
            symbol: [
                candle
                for candle in candles
                if (
                    warmup_start is None
                    or normalize_datetime(candle.timestamp) >= warmup_start
                )
                and (
                    normalized_end is None
                    or normalize_datetime(candle.timestamp) <= normalized_end
                )
            ]
            for symbol, candles in candles_by_symbol.items()
        }

    def _prices_by_symbol(self, candles_by_symbol):
        return {
            symbol: {
                candle.timestamp: candle.close
                for candle in candles
            }
            for symbol, candles in candles_by_symbol.items()
            if candles
        }

    def _benchmark_return(self, prices_by_symbol, timestamps):
        if not timestamps:
            return 0

        prices = prices_by_symbol.get(self.benchmark_symbol)
        if not prices:
            return self._equal_weight_return(prices_by_symbol, timestamps)

        start = prices.get(timestamps[0])
        end = prices.get(timestamps[-1])
        return (end / start) - 1 if start else 0

    def _equal_weight_return(self, prices_by_symbol, timestamps):
        if not timestamps:
            return 0

        returns = []
        for prices in prices_by_symbol.values():
            start = prices.get(timestamps[0])
            end = prices.get(timestamps[-1])
            if start:
                returns.append((end / start) - 1)

        return sum(returns) / len(returns) if returns else 0

    def _diagnostics(
        self,
        combined_result,
        sleeve_results,
        prices_by_symbol,
        timestamps,
    ):
        return {
            "annual": self._period_diagnostics(
                combined_result,
                sleeve_results,
                prices_by_symbol,
                timestamps,
                period="annual",
            ),
            "monthly": self._period_diagnostics(
                combined_result,
                sleeve_results,
                prices_by_symbol,
                timestamps,
                period="monthly",
            ),
        }

    def _period_diagnostics(
        self,
        combined_result,
        sleeve_results,
        prices_by_symbol,
        timestamps,
        period,
    ):
        grouped_timestamps = {}
        for timestamp in timestamps:
            key = (
                str(timestamp.year)
                if period == "annual"
                else timestamp.strftime("%Y-%m")
            )
            grouped_timestamps.setdefault(key, []).append(timestamp)

        equity_by_timestamp = {
            point.timestamp: point.equity
            for point in combined_result.equity_curve
        }
        sleeve_equity = {
            sleeve.name: {
                point.timestamp: point.equity
                for point in sleeve.result.equity_curve
            }
            for sleeve in sleeve_results
        }
        diagnostics = {}

        for key, grouped in grouped_timestamps.items():
            start = grouped[0]
            end = grouped[-1]
            bot_return = self._series_return(equity_by_timestamp, start, end)
            benchmark_return = self._benchmark_return(
                prices_by_symbol,
                [start, end],
            )
            equal_weight_return = self._equal_weight_return(
                prices_by_symbol,
                [start, end],
            )
            sleeve_returns = {
                sleeve.name: self._series_return(
                    sleeve_equity[sleeve.name],
                    start,
                    end,
                )
                for sleeve in sleeve_results
                if start in sleeve_equity[sleeve.name]
                and end in sleeve_equity[sleeve.name]
            }
            diagnostics[key] = {
                "bot_return": bot_return,
                "benchmark_return": benchmark_return,
                "equal_weight_return": equal_weight_return,
                "excess_vs_benchmark": bot_return - benchmark_return,
                "excess_vs_equal_weight": bot_return - equal_weight_return,
                "regime_label": self._regime_label(
                    bot_return,
                    benchmark_return,
                    equal_weight_return,
                ),
                "sleeve_returns": sleeve_returns,
            }

        return diagnostics

    def _series_return(self, values_by_timestamp, start, end):
        start_value = values_by_timestamp.get(start)
        end_value = values_by_timestamp.get(end)
        return (end_value / start_value) - 1 if start_value else 0

    def _regime_label(
        self,
        bot_return,
        benchmark_return,
        equal_weight_return,
    ):
        if benchmark_return > 0.10 and bot_return < benchmark_return:
            return "missed_benchmark_rally"
        if benchmark_return < 0 and bot_return > benchmark_return:
            return "defensive_success"
        if bot_return < 0 and benchmark_return > 0:
            return "wrong_risk_exposure"
        if abs(benchmark_return) < 0.05 and bot_return < 0:
            return "whipsaw_or_bad_selection"
        if bot_return > benchmark_return and bot_return > equal_weight_return:
            return "outperformed"
        return "mixed"
