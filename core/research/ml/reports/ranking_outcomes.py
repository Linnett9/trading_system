from __future__ import annotations

from typing import Any, Mapping

from core.research.ml.labels import MLLabelBuildResult


def outcomes_by_feature_date(
    config: Mapping[str, Any],
    label_result: MLLabelBuildResult,
    candles_by_symbol: dict[str, list[Any]],
    champion_equity_curve: list[Any],
) -> dict[str, dict[str, float | None]]:
    benchmark_symbol = str(config.get("ml", {}).get("benchmark_symbols", ["SPY"])[0])
    benchmark_closes = {
        candle.timestamp.date().isoformat(): candle.close
        for candle in candles_by_symbol.get(benchmark_symbol, [])
        if candle.close > 0
    }
    equity_by_date = {
        point.timestamp.date().isoformat(): point.equity
        for point in champion_equity_curve
        if point.equity > 0
    }
    equity_dates = sorted(equity_by_date)
    index_by_date = {value: index for index, value in enumerate(equity_dates)}
    outcomes = {}
    for row in label_result.rows:
        feature_date = str(row["feature_date"])
        label_end_date = str(row["label_end_date"])
        strategy_return = period_return(equity_by_date, feature_date, label_end_date)
        benchmark_return = period_return(benchmark_closes, feature_date, label_end_date)
        outcomes[feature_date] = {
            "strategy_return": strategy_return,
            "excess_spy_return": (
                strategy_return - benchmark_return
                if strategy_return is not None and benchmark_return is not None
                else None
            ),
            "drawdown_event": future_drawdown_event(
                equity_dates,
                equity_by_date,
                index_by_date,
                feature_date,
                label_end_date,
            ),
        }
    return outcomes


def period_return(
    values_by_date: dict[str, float],
    start_date: str,
    end_date: str,
) -> float | None:
    start = values_by_date.get(start_date)
    end = values_by_date.get(end_date)
    return (end / start) - 1.0 if start and end else None


def future_drawdown_event(
    dates: list[str],
    values_by_date: dict[str, float],
    index_by_date: dict[str, int],
    start_date: str,
    end_date: str,
) -> float | None:
    start_index = index_by_date.get(start_date)
    end_index = index_by_date.get(end_date)
    if start_index is None or end_index is None:
        return None
    peak = values_by_date[dates[start_index]]
    maximum_drawdown = 0.0
    for date in dates[start_index:end_index + 1]:
        value = values_by_date[date]
        peak = max(peak, value)
        maximum_drawdown = min(maximum_drawdown, (value / peak) - 1.0)
    return float(maximum_drawdown <= -0.10)
