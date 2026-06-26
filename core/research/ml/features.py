from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from math import sqrt
from pathlib import Path
from statistics import mean, median, pstdev

from core.entities.candle import Candle


FEATURE_LOOKBACK_DAYS = 252


@dataclass(frozen=True)
class MLFeatureBuildResult:
    rows: list[dict[str, float | str]]
    dropped_rows: int
    date_range: tuple[str, str] | None


class HistoricalFeatureBuilder:
    """Build market features using prices available at each feature date only."""

    def __init__(
        self,
        benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ"),
        lookback_days: int = FEATURE_LOOKBACK_DAYS,
    ):
        if lookback_days < FEATURE_LOOKBACK_DAYS:
            raise ValueError(
                f"lookback_days must be at least {FEATURE_LOOKBACK_DAYS}"
            )
        if len(benchmark_symbols) < 2:
            raise ValueError("benchmark_symbols must contain at least SPY and QQQ")
        self.benchmark_symbols = tuple(str(symbol).upper() for symbol in benchmark_symbols)
        self.spy_symbol = self.benchmark_symbols[0]
        self.qqq_symbol = self.benchmark_symbols[1]
        self.lookback_days = lookback_days

    def build(
        self,
        candles_by_symbol: dict[str, list[Candle]],
    ) -> MLFeatureBuildResult:
        required_symbols = set(self.benchmark_symbols)
        missing_symbols = sorted(required_symbols - set(candles_by_symbol))
        if missing_symbols:
            raise ValueError(
                "ML features require benchmark data for: "
                f"{', '.join(missing_symbols)}"
            )

        close_by_symbol = {
            symbol: self._close_by_date(candles)
            for symbol, candles in candles_by_symbol.items()
        }
        common_dates = sorted(
            set.intersection(*(set(closes) for closes in close_by_symbol.values()))
        )
        rows: list[dict[str, float | str]] = []
        dropped_rows = 0

        for index, feature_date in enumerate(common_dates):
            if index < self.lookback_days:
                dropped_rows += 1
                continue

            window_dates = common_dates[index - self.lookback_days : index + 1]
            histories = {
                symbol: [close_by_symbol[symbol][day] for day in window_dates]
                for symbol in close_by_symbol
            }
            rows.append(self._feature_row(feature_date, histories))

        date_range = None
        if rows:
            date_range = (str(rows[0]["feature_date"]), str(rows[-1]["feature_date"]))

        return MLFeatureBuildResult(
            rows=rows,
            dropped_rows=dropped_rows,
            date_range=date_range,
        )

    def _feature_row(
        self,
        feature_date: date,
        histories: dict[str, list[float]],
    ) -> dict[str, float | str]:
        spy = histories[self.spy_symbol]
        qqq = histories[self.qqq_symbol]
        universe = [
            prices for symbol, prices in histories.items()
            if symbol not in {self.spy_symbol, self.qqq_symbol}
        ]
        if not universe:
            universe = [spy]

        row = {
            "feature_date": feature_date.isoformat(),
            "spy_return_1m": self._return(spy, 21),
            "spy_return_3m": self._return(spy, 63),
            "spy_return_6m": self._return(spy, 126),
            "spy_return_12m": self._return(spy, 252),
            "spy_return_12m_ex_latest_month": self._return_excluding_recent(spy),
            "spy_distance_sma_200": self._distance_from_sma(spy, 200),
            "spy_distance_sma_100": self._distance_from_sma(spy, 100),
            "spy_realized_volatility_21d": self._realized_volatility(spy, 21),
            "spy_realized_volatility_63d": self._realized_volatility(spy, 63),
            "spy_max_drawdown_63d": self._max_drawdown(spy, 63),
            "spy_max_drawdown_126d": self._max_drawdown(spy, 126),
            "spy_above_sma_200": float(spy[-1] > mean(spy[-200:])),
            "qqq_return_1m": self._return(qqq, 21),
            "qqq_return_3m": self._return(qqq, 63),
            "qqq_return_6m": self._return(qqq, 126),
            "breadth_above_sma_200": self._fraction_above_sma(universe, 200),
            "breadth_above_sma_100": self._fraction_above_sma(universe, 100),
            "breadth_positive_6m_momentum": self._fraction_positive_return(universe, 126),
            "universe_return_1m_mean": mean(self._return(prices, 21) for prices in universe),
            "universe_return_1m_median": median(self._return(prices, 21) for prices in universe),
            "universe_return_1m_dispersion": self._dispersion(
                [self._return(prices, 21) for prices in universe]
            ),
        }
        for symbol in self.benchmark_symbols:
            safe_symbol = self._safe_feature_symbol(symbol)
            if safe_symbol in {"spy", "qqq"}:
                continue
            prices = histories[symbol]
            row.update({
                f"{safe_symbol}_return_1m": self._return(prices, 21),
                f"{safe_symbol}_return_3m": self._return(prices, 63),
                f"{safe_symbol}_return_6m": self._return(prices, 126),
                f"{safe_symbol}_distance_sma_200": self._distance_from_sma(prices, 200),
                f"{safe_symbol}_realized_volatility_21d": (
                    self._realized_volatility(prices, 21)
                ),
                f"{safe_symbol}_max_drawdown_63d": self._max_drawdown(prices, 63),
                f"{safe_symbol}_above_sma_200": float(prices[-1] > mean(prices[-200:])),
            })
        return row

    def _close_by_date(self, candles: list[Candle]) -> dict[date, float]:
        closes: dict[date, float] = {}
        for candle in sorted(candles, key=lambda item: item.timestamp):
            if candle.close <= 0:
                continue
            closes[candle.timestamp.date()] = candle.close
        return closes

    def _return(self, prices: list[float], days: int) -> float:
        return (prices[-1] / prices[-(days + 1)]) - 1.0

    def _return_excluding_recent(self, prices: list[float]) -> float:
        return (prices[-22] / prices[-253]) - 1.0

    def _distance_from_sma(self, prices: list[float], days: int) -> float:
        average = mean(prices[-days:])
        return (prices[-1] / average) - 1.0

    def _realized_volatility(self, prices: list[float], days: int) -> float:
        returns = [
            (current / previous) - 1.0
            for previous, current in zip(prices[-(days + 1):], prices[-days:])
        ]
        return pstdev(returns) * sqrt(252)

    def _max_drawdown(self, prices: list[float], days: int) -> float:
        peak = prices[-days]
        max_drawdown = 0.0
        for price in prices[-days:]:
            peak = max(peak, price)
            max_drawdown = min(max_drawdown, (price / peak) - 1.0)
        return max_drawdown

    def _fraction_above_sma(self, histories: list[list[float]], days: int) -> float:
        return mean(float(prices[-1] > mean(prices[-days:])) for prices in histories)

    def _fraction_positive_return(
        self,
        histories: list[list[float]],
        days: int,
    ) -> float:
        return mean(float(self._return(prices, days) > 0) for prices in histories)

    def _dispersion(self, values: list[float]) -> float:
        return pstdev(values) if len(values) > 1 else 0.0

    def _safe_feature_symbol(self, symbol: str) -> str:
        return "".join(
            character.lower() if character.isalnum() else "_"
            for character in symbol
        ).strip("_")


def write_feature_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["feature_date"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_champion_state_features(
    rows: list[dict[str, float | str]],
    selections: list[object],
) -> list[dict[str, float | str]]:
    """Add only the latest champion decision available on each feature date."""
    ordered_selections = sorted(selections, key=lambda item: item.timestamp)
    enriched_rows: list[dict[str, float | str]] = []
    selection_index = 0
    active_selection = None
    previous_symbols: set[str] = set()
    previous_weights: dict[str, float] = {}
    recent_turnovers: list[float] = []

    for row in rows:
        feature_date = date.fromisoformat(str(row["feature_date"]))
        while (
            selection_index < len(ordered_selections)
            and ordered_selections[selection_index].timestamp.date() <= feature_date
        ):
            active_selection = ordered_selections[selection_index]
            current_symbols = set(active_selection.symbols)
            replacements = len(current_symbols - previous_symbols)
            current_weights = active_selection.target_weights or {}
            turnover = sum(
                abs(float(current_weights.get(symbol, 0)) - previous_weights.get(symbol, 0))
                for symbol in set(current_weights) | set(previous_weights)
            )
            previous_symbols = current_symbols
            previous_weights = {
                symbol: float(weight)
                for symbol, weight in current_weights.items()
            }
            recent_turnovers.append(turnover)
            recent_turnovers = recent_turnovers[-3:]
            selection_index += 1

        enriched = dict(row)
        if active_selection is None:
            enriched.update(_empty_champion_state())
        else:
            weights = active_selection.target_weights or {}
            scores = [
                float(active_selection.scores[symbol])
                for symbol in active_selection.symbols
                if symbol in active_selection.scores
            ]
            enriched.update({
                "champion_exposure": float(active_selection.exposure_target),
                "champion_holding_count": float(len(active_selection.symbols)),
                "champion_average_rank_score": mean(scores) if scores else 0.0,
                "champion_largest_position_weight": (
                    max((abs(float(weight)) for weight in weights.values()), default=0.0)
                    * float(active_selection.exposure_target)
                ),
                "champion_cash_weight": max(0.0, 1.0 - float(active_selection.exposure_target)),
                "champion_risk_on": float(active_selection.risk_on),
                "champion_breadth_passes": float(active_selection.breadth_passes),
                "champion_drawdown_guard_active": float(active_selection.drawdown_guard_active),
                "champion_chop_filter_active": float(active_selection.chop_filter_active),
                "champion_last_rebalance_turnover": recent_turnovers[-1] if recent_turnovers else 0.0,
                "champion_average_turnover_3_rebalances": (
                    mean(recent_turnovers) if recent_turnovers else 0.0
                ),
                "champion_replacements": float(replacements),
            })
        enriched_rows.append(enriched)
    return enriched_rows


def _empty_champion_state() -> dict[str, float]:
    return {
        "champion_exposure": 0.0,
        "champion_holding_count": 0.0,
        "champion_average_rank_score": 0.0,
        "champion_largest_position_weight": 0.0,
        "champion_cash_weight": 1.0,
        "champion_risk_on": 0.0,
        "champion_breadth_passes": 0.0,
        "champion_drawdown_guard_active": 0.0,
        "champion_chop_filter_active": 0.0,
        "champion_last_rebalance_turnover": 0.0,
        "champion_average_turnover_3_rebalances": 0.0,
        "champion_replacements": 0.0,
    }
