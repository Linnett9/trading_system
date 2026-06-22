from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from core.entities.candle import Candle


@dataclass(frozen=True)
class MLLabelBuildResult:
    rows: list[dict[str, int | float | str]]
    dropped_rows_insufficient_horizon: int
    label_name: str


class RiskRegimeLabelBuilder:
    """Label each feature date from SPY's strictly future return."""

    def __init__(self, horizon_days: int):
        if horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero")
        self.horizon_days = horizon_days

    def build(
        self,
        feature_rows: list[dict[str, float | str]],
        benchmark_candles: list[Candle],
    ) -> MLLabelBuildResult:
        closes_by_date = {
            candle.timestamp.date(): candle.close
            for candle in sorted(benchmark_candles, key=lambda item: item.timestamp)
            if candle.close > 0
        }
        trading_dates = sorted(closes_by_date)
        index_by_date = {trading_date: index for index, trading_date in enumerate(trading_dates)}
        rows: list[dict[str, int | str]] = []
        dropped_rows = 0

        for feature in feature_rows:
            feature_date = date.fromisoformat(str(feature["feature_date"]))
            index = index_by_date.get(feature_date)
            if index is None or index + self.horizon_days >= len(trading_dates):
                dropped_rows += 1
                continue

            label_start = trading_dates[index + 1]
            label_end = trading_dates[index + self.horizon_days]
            future_return = (closes_by_date[label_end] / closes_by_date[feature_date]) - 1.0
            if not feature_date < label_start <= label_end:
                raise ValueError("Label window must be strictly after the feature date")

            rows.append({
                "feature_date": feature_date.isoformat(),
                "label_start_date": label_start.isoformat(),
                "label_end_date": label_end.isoformat(),
                "risk_regime": int(future_return > 0),
                "future_return": future_return,
            })

        return MLLabelBuildResult(
            rows=rows,
            dropped_rows_insufficient_horizon=dropped_rows,
            label_name="risk_regime",
        )


class DrawdownRiskLabelBuilder:
    """Label a future peak-to-trough drawdown that exceeds a risk threshold."""

    def __init__(self, horizon_days: int, threshold: float):
        if horizon_days <= 0:
            raise ValueError("horizon_days must be greater than zero")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        self.horizon_days = horizon_days
        self.threshold = threshold

    def build(
        self,
        feature_rows: list[dict[str, float | str]],
        benchmark_candles: list[Candle],
    ) -> MLLabelBuildResult:
        closes_by_date = {
            candle.timestamp.date(): candle.close
            for candle in sorted(benchmark_candles, key=lambda item: item.timestamp)
            if candle.close > 0
        }
        trading_dates = sorted(closes_by_date)
        index_by_date = {
            trading_date: index
            for index, trading_date in enumerate(trading_dates)
        }
        rows: list[dict[str, int | float | str]] = []
        dropped_rows = 0

        for feature in feature_rows:
            feature_date = date.fromisoformat(str(feature["feature_date"]))
            index = index_by_date.get(feature_date)
            if index is None or index + self.horizon_days >= len(trading_dates):
                dropped_rows += 1
                continue

            label_start = trading_dates[index + 1]
            label_end = trading_dates[index + self.horizon_days]
            window_prices = [
                closes_by_date[trading_date]
                for trading_date in trading_dates[index:index + self.horizon_days + 1]
            ]
            maximum_drawdown = _maximum_drawdown(window_prices)
            if not feature_date < label_start <= label_end:
                raise ValueError("Label window must be strictly after the feature date")

            rows.append({
                "feature_date": feature_date.isoformat(),
                "label_start_date": label_start.isoformat(),
                "label_end_date": label_end.isoformat(),
                "drawdown_risk": int(maximum_drawdown <= -self.threshold),
                "future_max_drawdown": maximum_drawdown,
            })

        return MLLabelBuildResult(
            rows=rows,
            dropped_rows_insufficient_horizon=dropped_rows,
            label_name="drawdown_risk",
        )


class ChampionSuccessLabelBuilder:
    """Label whether the frozen champion beats SPY over the future horizon."""

    def __init__(self, horizon_days: int):
        self.horizon_days = horizon_days

    def build(self, feature_rows, benchmark_candles, equity_curve) -> MLLabelBuildResult:
        closes = {candle.timestamp.date(): candle.close for candle in benchmark_candles}
        equity = {point.timestamp.date(): point.equity for point in equity_curve}
        dates = sorted(set(closes) & set(equity))
        index_by_date = {value: index for index, value in enumerate(dates)}
        rows = []
        dropped = 0
        for feature in feature_rows:
            feature_date = date.fromisoformat(str(feature["feature_date"]))
            index = index_by_date.get(feature_date)
            if index is None or index + self.horizon_days >= len(dates):
                dropped += 1
                continue
            label_start = dates[index + 1]
            label_end = dates[index + self.horizon_days]
            champion_return = (equity[label_end] / equity[feature_date]) - 1.0
            benchmark_return = (closes[label_end] / closes[feature_date]) - 1.0
            rows.append({
                "feature_date": feature_date.isoformat(),
                "label_start_date": label_start.isoformat(),
                "label_end_date": label_end.isoformat(),
                "champion_success": int(champion_return > benchmark_return),
                "champion_excess_return": champion_return - benchmark_return,
            })
        return MLLabelBuildResult(rows, dropped, "champion_success")


def write_label_rows(
    path: Path,
    rows: list[dict[str, int | float | str]],
    label_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_date",
        "label_start_date",
        "label_end_date",
        label_name,
        {
            "risk_regime": "future_return",
            "drawdown_risk": "future_max_drawdown",
            "champion_success": "champion_excess_return",
        }[label_name],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maximum_drawdown(prices: list[float]) -> float:
    peak = prices[0]
    maximum_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        maximum_drawdown = min(maximum_drawdown, (price / peak) - 1.0)
    return maximum_drawdown
