from core.research.multi_strategy.aggregation import (
    MultiStrategyAggregationMixin,
)
from core.research.multi_strategy.data import MultiStrategyDataMixin
from core.research.multi_strategy.diagnostics import (
    MultiStrategyDiagnosticsMixin,
)
from core.research.multi_strategy.models import (
    MultiStrategyPortfolioResult,
    MultiStrategySleeveResult,
)
from core.research.multi_strategy.sleeves import MultiStrategySleeveMixin


class MultiStrategyPortfolioBacktester(
    MultiStrategyAggregationMixin,
    MultiStrategyDataMixin,
    MultiStrategyDiagnosticsMixin,
    MultiStrategySleeveMixin,
):

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
