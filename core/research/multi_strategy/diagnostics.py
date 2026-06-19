class MultiStrategyDiagnosticsMixin:

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
