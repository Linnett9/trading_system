from datetime import datetime, timezone
import math

from core.entities.benchmark_result import BenchmarkResult
from core.entities.walk_forward_result import (
    WalkForwardFoldResult,
    WalkForwardResult,
)
from core.research.backtest_runner import run_backtest
from core.research.parameter_optimizer import (
    ParameterOptimizer,
    parameter_overrides,
)


def parse_date(value: str) -> datetime:
    return normalize_datetime(datetime.fromisoformat(value))


def normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value

    return value.astimezone(timezone.utc).replace(tzinfo=None)


def candles_between(candles, start: datetime, end: datetime):
    normalized_start = normalize_datetime(start)
    normalized_end = normalize_datetime(end)

    return [
        candle for candle in candles
        if (
            normalized_start
            <= normalize_datetime(candle.timestamp)
            <= normalized_end
        )
    ]


def buy_and_hold_return(candles) -> float:
    if len(candles) < 2:
        return 0

    first_close = candles[0].close
    last_close = candles[-1].close

    if first_close == 0:
        return 0

    return (last_close / first_close) - 1


def benchmark_metrics(candles) -> BenchmarkResult:
    if len(candles) < 2:
        return BenchmarkResult()

    closes = [candle.close for candle in candles]
    returns = []

    for previous, current in zip(closes, closes[1:]):
        if previous != 0:
            returns.append((current - previous) / previous)

    average_return = sum(returns) / len(returns) if returns else 0
    sharpe = 0

    if returns:
        variance = sum(
            (value - average_return) ** 2
            for value in returns
        ) / len(returns)
        std = math.sqrt(variance)
        sharpe = (
            (average_return / std) * math.sqrt(252)
            if std != 0
            else 0
        )

    peak = closes[0]
    max_drawdown = 0

    for close in closes:
        peak = max(peak, close)
        if peak != 0:
            max_drawdown = max(max_drawdown, (peak - close) / peak)

    return BenchmarkResult(
        total_return=buy_and_hold_return(candles),
        max_drawdown=max_drawdown,
        sharpe=sharpe,
    )


def excess_return_per_unit_risk(excess_return, max_drawdown):
    if max_drawdown <= 0:
        return 0

    return excess_return / max_drawdown


def evaluate_fold(
    test_result,
    benchmark,
    excess_return,
    research_config,
):
    min_closed_trades = research_config.get("min_closed_trades", 0)
    min_profit_factor = research_config.get("min_profit_factor", 0)
    max_drawdown = research_config.get("max_drawdown", 1)
    min_time_in_market = research_config.get("min_time_in_market", 0)
    max_time_in_market = research_config.get("max_time_in_market", 1)
    require_positive_excess = research_config.get(
        "require_positive_excess",
        True,
    )
    require_sharpe_edge = research_config.get(
        "require_sharpe_edge",
        False,
    )
    min_sharpe = research_config.get("min_sharpe", 0)

    failure_reasons = []

    if test_result.closed_trades < min_closed_trades:
        failure_reasons.append(
            f"closed_trades below minimum ({test_result.closed_trades} < "
            f"{min_closed_trades})"
        )

    if test_result.profit_factor < min_profit_factor:
        failure_reasons.append(
            f"profit_factor below minimum "
            f"({test_result.profit_factor:.2f} < {min_profit_factor:.2f})"
        )

    if test_result.max_drawdown > max_drawdown:
        failure_reasons.append(
            f"max_drawdown above maximum "
            f"({test_result.max_drawdown:.2%} > {max_drawdown:.2%})"
        )

    time_in_market = test_result.trade_analysis.time_in_market_percent
    if time_in_market < min_time_in_market:
        failure_reasons.append(
            f"time_in_market below minimum "
            f"({time_in_market:.2%} < {min_time_in_market:.2%})"
        )

    if time_in_market > max_time_in_market:
        failure_reasons.append(
            f"time_in_market above maximum "
            f"({time_in_market:.2%} > {max_time_in_market:.2%})"
        )

    if test_result.sharpe < min_sharpe:
        failure_reasons.append(
            f"test_sharpe below minimum "
            f"({test_result.sharpe:.2f} < {min_sharpe:.2f})"
        )

    if require_positive_excess and excess_return <= 0:
        failure_reasons.append("excess_return is not positive")

    if require_sharpe_edge and (
        test_result.sharpe <= benchmark.sharpe
        and test_result.sharpe < 1
    ):
        failure_reasons.append(
            "test_sharpe does not beat benchmark_sharpe or 1.00"
        )

    return not failure_reasons, "; ".join(failure_reasons)


class WalkForwardTester:

    def __init__(
        self,
        config: dict,
        metric_name: str = "sharpe",
    ):
        self.config = config
        self.metric_name = metric_name

    def run(
        self,
        candles,
        symbol: str,
        folds: list[dict],
        grid: dict,
    ) -> WalkForwardResult:
        research_config = self.config["research"]
        optimizer_min_closed_trades = research_config.get(
            "optimizer_min_closed_trades",
            research_config.get("min_closed_trades", 0),
        )
        optimizer = ParameterOptimizer(
            config=self.config,
            metric_name=self.metric_name,
            min_closed_trades=optimizer_min_closed_trades,
        )
        fold_results = []

        for fold in folds:
            train_start = parse_date(fold["train_start"])
            train_end = parse_date(fold["train_end"])
            test_start = parse_date(fold["test_start"])
            test_end = parse_date(fold["test_end"])

            train_candles = candles_between(candles, train_start, train_end)
            test_candles = candles_between(candles, test_start, test_end)

            training_results = optimizer.run(
                candles=train_candles,
                symbol=symbol,
                grid=grid,
            )

            if not training_results:
                continue

            best_training_result = training_results[0]
            test_result = run_backtest(
                candles=test_candles,
                symbol=symbol,
                config=self.config,
                overrides=parameter_overrides(
                    best_training_result.parameters
                ),
            )
            benchmark = benchmark_metrics(test_candles)
            benchmark_return = benchmark.total_return
            excess_return = test_result.total_return - benchmark_return
            risk_adjusted_excess = excess_return_per_unit_risk(
                excess_return,
                test_result.max_drawdown,
            )
            passed, failure_reason = evaluate_fold(
                test_result,
                benchmark,
                excess_return,
                research_config,
            )

            fold_results.append(
                WalkForwardFoldResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_training_result=best_training_result,
                    test_result=test_result,
                    benchmark_return=benchmark_return,
                    benchmark=benchmark,
                    excess_return=excess_return,
                    excess_return_per_unit_risk=risk_adjusted_excess,
                    passed=passed,
                    failure_reason=failure_reason,
                )
            )

        return WalkForwardResult(
            symbol=symbol,
            timeframe=self.config["backtest"]["timeframe"],
            folds=fold_results,
        )
