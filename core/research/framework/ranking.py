from __future__ import annotations

import math
from statistics import mean
from typing import Any, Sequence


class CrossSectionalRankingEvaluator:
    def __init__(
        self,
        *,
        target_column: str,
        date_column: str = "rebalance_date",
        drawdown_column: str = "actual_future_drawdown",
        volatility_column: str = "actual_future_volatility",
        annualization_factor: float = 52.0,
    ):
        self.target_column = target_column
        self.date_column = date_column
        self.drawdown_column = drawdown_column
        self.volatility_column = volatility_column
        self.annualization_factor = float(annualization_factor)

    def evaluate(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        name: str,
        signal_column: str,
        kind: str,
    ) -> dict[str, Any]:
        by_date: dict[str, list[dict[str, float]]] = {}
        for row in rows:
            signal = finite_number(row.get(signal_column))
            target = finite_number(row.get(self.target_column))
            if signal is None or target is None:
                continue
            risk = max(
                abs(finite_number(row.get(self.drawdown_column)) or 0.0),
                abs(finite_number(row.get(self.volatility_column)) or 0.0),
                1e-6,
            )
            by_date.setdefault(str(row[self.date_column]), []).append(
                {"score": signal, "target": target, "risk_target": target / risk}
            )
        date_metrics = [
            self._date_metrics(group)
            for _, group in sorted(by_date.items())
            if len(group) >= 2
        ]
        return {
            "rank": None,
            "name": name,
            "kind": kind,
            "signal_column": signal_column,
            "mean_pearson_ic": average([row["pearson"] for row in date_metrics]),
            "mean_spearman_ic": average([row["spearman"] for row in date_metrics]),
            "top_decile_return": average(
                [row["top_decile_return"] for row in date_metrics]
            ),
            "bottom_decile_return": average(
                [row["bottom_decile_return"] for row in date_metrics]
            ),
            "top_minus_bottom_spread": average(
                [row["top_minus_bottom_spread"] for row in date_metrics]
            ),
            "top_decile_hit_rate": average(
                [row["top_decile_hit_rate"] for row in date_metrics]
            ),
            "risk_adjusted_spread": average(
                [row["risk_adjusted_spread"] for row in date_metrics]
            ),
            "spread_sharpe": annualized_sharpe(
                [row["top_minus_bottom_spread"] for row in date_metrics],
                self.annualization_factor,
            ),
            "date_count": len(date_metrics),
            "row_count": sum(row["row_count"] for row in date_metrics),
        }

    @staticmethod
    def _date_metrics(rows: list[dict[str, float]]) -> dict[str, Any]:
        ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
        bucket_size = max(1, math.ceil(len(ordered) * 0.10))
        top = ordered[:bucket_size]
        bottom = ordered[-bucket_size:]
        top_return = mean(row["target"] for row in top)
        bottom_return = mean(row["target"] for row in bottom)
        return {
            "row_count": len(ordered),
            "pearson": pearson(
                [row["score"] for row in ordered],
                [row["target"] for row in ordered],
            ),
            "spearman": spearman(
                [row["score"] for row in ordered],
                [row["target"] for row in ordered],
            ),
            "top_decile_return": top_return,
            "bottom_decile_return": bottom_return,
            "top_minus_bottom_spread": top_return - bottom_return,
            "top_decile_hit_rate": mean(
                1.0 if row["target"] > 0.0 else 0.0 for row in top
            ),
            "risk_adjusted_spread": (
                mean(row["risk_target"] for row in top)
                - mean(row["risk_target"] for row in bottom)
            ),
        }


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(ranks(left), ranks(right))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            output[original_index] = rank
        index = end
    return output


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_variance = sum((x - left_mean) ** 2 for x in left)
    right_variance = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_variance * right_variance)
    return numerator / denominator if denominator > 0.0 else None


def average(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def annualized_sharpe(
    values: list[float | None],
    annualization_factor: float,
) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if len(finite) < 2:
        return None
    value_mean = mean(finite)
    variance = sum((value - value_mean) ** 2 for value in finite) / (len(finite) - 1)
    return (
        value_mean / math.sqrt(variance) * math.sqrt(annualization_factor)
        if variance > 0.0
        else None
    )


def finite_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
