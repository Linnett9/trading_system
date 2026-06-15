from itertools import product

from core.entities.backtest_result import BacktestResult
from core.entities.optimization_result import OptimizationResult
from core.research.performance_metrics import (
    calmar_ratio,
    composite_score,
)
from core.research.backtest_runner import run_backtest
from core.research.result_cache import ResultCache


def expand_grid(grid: dict) -> list[dict]:
    if "__parameter_sets__" in grid:
        return grid["__parameter_sets__"]

    keys = list(grid.keys())
    values = [grid[key] for key in keys]

    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def parameter_overrides(parameters: dict) -> dict:
    overrides = {
        "strategy": {},
        "risk": {},
        "position_sizing": {},
    }

    for key, value in parameters.items():
        if (
            key.startswith("ema_")
            or key.startswith("rsi_")
            or key.startswith("donchian_")
            or key.startswith("pullback_")
            or key.startswith("trend_")
            or key.startswith("bollinger_")
            or key.startswith("ensemble_")
            or key == "use_regime_filter"
            or key == "use_volatility_filter"
            or key == "use_volume_filter"
            or key == "use_breakout_vote"
            or key == "require_sideways_regime"
            or key == "min_relative_volume"
            or key == "min_bandwidth"
            or key == "max_bandwidth"
            or key == "min_adx"
            or key == "name"
        ):
            overrides["strategy"][key] = value
        elif key in {
            "sizing_mode",
            "target_exposure",
            "dollar_amount",
            "position_max_exposure",
            "fallback_atr",
            "fallback_volatility",
        }:
            if key == "sizing_mode":
                overrides["position_sizing"]["mode"] = value
            elif key == "position_max_exposure":
                overrides["position_sizing"]["max_exposure"] = value
            else:
                overrides["position_sizing"][key] = value
        else:
            overrides["risk"][key] = value

    if (
        "target_exposure" in overrides["position_sizing"]
        and "max_exposure" not in overrides["position_sizing"]
    ):
        overrides["position_sizing"]["max_exposure"] = (
            overrides["position_sizing"]["target_exposure"]
        )

    return {
        section: values
        for section, values in overrides.items()
        if values
    }


def valid_parameters(parameters: dict) -> bool:
    fast = parameters.get("ema_fast_period")
    slow = parameters.get("ema_slow_period")

    if fast is not None and slow is not None and fast >= slow:
        return False

    stop = parameters.get("atr_stop_multiplier")
    take_profit = parameters.get("atr_take_profit_multiplier")

    if (
        stop is not None
        and take_profit is not None
        and take_profit <= stop
    ):
        return False

    target_exposure = parameters.get("target_exposure")
    max_exposure = parameters.get("position_max_exposure")

    if target_exposure is not None and target_exposure <= 0:
        return False

    if max_exposure is not None and max_exposure <= 0:
        return False

    if (
        target_exposure is not None
        and max_exposure is not None
        and target_exposure > max_exposure
    ):
        return False

    return True


def benchmark_return(candles) -> float:
    if len(candles) < 2:
        return 0

    first_close = candles[0].close
    last_close = candles[-1].close

    if first_close == 0:
        return 0

    return (last_close / first_close) - 1


def get_metric(
    result,
    metric_name: str,
    benchmark_return_value: float = 0,
    target_trades: int = 20,
) -> float:
    if metric_name == "sharpe":
        return result.sharpe

    if metric_name == "total_return":
        return result.total_return

    if metric_name == "final_equity":
        return result.final_equity

    if metric_name == "max_drawdown":
        return -result.max_drawdown

    if metric_name == "calmar":
        return calmar_ratio(result.total_return, result.max_drawdown)

    if metric_name == "excess_return":
        return result.total_return - benchmark_return_value

    if metric_name == "excess_calmar":
        return calmar_ratio(
            result.total_return - benchmark_return_value,
            result.max_drawdown,
        )

    if metric_name == "profit_factor":
        return result.profit_factor

    if metric_name == "composite":
        return composite_score(
            excess_return=result.total_return - benchmark_return_value,
            sharpe=result.sharpe,
            max_drawdown_value=result.max_drawdown,
            profit_factor_value=result.profit_factor,
            closed_trades=result.closed_trades,
            target_trades=target_trades,
        )

    raise ValueError(f"Unknown optimization metric: {metric_name}")


class ParameterOptimizer:

    def __init__(
        self,
        config: dict,
        metric_name: str = "sharpe",
        min_closed_trades: int = 0,
    ):
        self.config = config
        self.metric_name = metric_name
        self.min_closed_trades = min_closed_trades
        research_config = config.get("research", {})
        cache_config = config.get("cache", {})
        self.cache = ResultCache(
            cache_dir=cache_config.get("results_dir", "cache/results"),
            enabled=cache_config.get(
                "enabled",
                research_config.get("cache_results", False),
            ),
        )

    def run(self, candles, symbol: str, grid: dict) -> list[OptimizationResult]:
        research_config = self.config.get("research", {})

        if research_config.get("two_stage_enabled"):
            return self._run_two_stage(candles, symbol, grid)

        return self._run_grid(candles, symbol, grid)

    def _run_grid(
        self,
        candles,
        symbol: str,
        grid: dict,
    ) -> list[OptimizationResult]:
        results = []
        research_config = self.config.get("research", {})
        benchmark_return_value = benchmark_return(candles)
        target_trades = research_config.get("min_closed_trades", 20)

        for parameters in expand_grid(grid):
            if not valid_parameters(parameters):
                continue

            result = self._run_backtest_cached(
                candles,
                symbol,
                parameters,
            )
            if result.closed_trades < self.min_closed_trades:
                continue

            metric_value = get_metric(
                result,
                self.metric_name,
                benchmark_return_value=benchmark_return_value,
                target_trades=target_trades,
            )

            results.append(
                OptimizationResult(
                    parameters=parameters,
                    metric_name=self.metric_name,
                    metric_value=metric_value,
                    result=result,
                )
            )

        return sorted(
            results,
            key=lambda item: item.metric_value,
            reverse=True,
        )

    def _run_two_stage(
        self,
        candles,
        symbol: str,
        grid: dict,
    ) -> list[OptimizationResult]:
        research_config = self.config.get("research", {})
        top_n = research_config.get("two_stage_top_n", 5)
        stage_one_limit = research_config.get(
            "stage_one_max_combinations",
            100,
        )
        parameters = [
            item for item in expand_grid(grid)
            if valid_parameters(item)
        ]
        stage_one_grid = self._grid_from_parameters(
            parameters[:stage_one_limit]
        )
        stage_one_results = self._run_grid(candles, symbol, stage_one_grid)

        if not stage_one_results:
            return []

        refined_parameters = []
        for result in stage_one_results[:top_n]:
            refined_parameters.extend(
                self._refine_parameters(result.parameters, grid)
            )

        refined_grid = self._grid_from_parameters(refined_parameters)
        stage_two_results = self._run_grid(candles, symbol, refined_grid)

        return stage_two_results or stage_one_results

    def _run_backtest_cached(self, candles, symbol, parameters):
        cache_key = {
            "symbol": symbol,
            "strategy": self.config.get("strategy", {}),
            "risk": self.config.get("risk", {}),
            "position_sizing": self.config.get("position_sizing", {}),
            "research": {
                "early_stop_max_drawdown": (
                    self.config.get("research", {})
                    .get("early_stop_max_drawdown")
                ),
                "early_stop_equity_floor_pct": (
                    self.config.get("research", {})
                    .get("early_stop_equity_floor_pct")
                ),
            },
            "parameters": parameters,
            "candles": self._candle_signature(candles),
        }
        cached = self.cache.get(cache_key)

        if cached is not None:
            return BacktestResult.from_dict(cached)

        result = run_backtest(
            candles=candles,
            symbol=symbol,
            config=self.config,
            overrides=parameter_overrides(parameters),
        )
        self.cache.set(cache_key, self._cache_payload(result))

        return result

    def _cache_payload(self, result: BacktestResult) -> dict:
        payload = result.to_dict()
        payload["equity_curve"] = []
        return payload

    def _candle_signature(self, candles) -> dict:
        if not candles:
            return {"count": 0}

        return {
            "count": len(candles),
            "start": candles[0].timestamp,
            "end": candles[-1].timestamp,
            "first_close": candles[0].close,
            "last_close": candles[-1].close,
        }

    def _grid_from_parameters(self, parameters: list[dict]) -> dict:
        return {
            "__parameter_sets__": parameters,
        }

    def _refine_parameters(self, parameters: dict, grid: dict) -> list[dict]:
        refined = [parameters]

        for key, values in grid.items():
            if key not in parameters or not isinstance(values, list):
                continue

            index = values.index(parameters[key])
            nearby_values = values[max(0, index - 1): index + 2]

            for value in nearby_values:
                candidate = dict(parameters)
                candidate[key] = value
                if valid_parameters(candidate):
                    refined.append(candidate)

        unique = []
        seen = set()
        for candidate in refined:
            marker = tuple(sorted(candidate.items()))
            if marker not in seen:
                seen.add(marker)
                unique.append(candidate)

        return unique
