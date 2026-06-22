from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev


def build_champion_rebalance_rows(
    feature_rows: list[dict[str, float | str]],
    selections: list[object],
    equity_curve: list[object],
    benchmark_candles: list[object],
    horizon_days: int,
    candles_by_symbol: dict[str, list[object]] | None = None,
    sector_by_symbol: dict[str, str] | None = None,
) -> list[dict[str, float | str]]:
    features_by_date = {str(row["feature_date"]): row for row in feature_rows}
    equity_by_date = {point.timestamp.date().isoformat(): point.equity for point in equity_curve}
    benchmark_by_date = {candle.timestamp.date().isoformat(): candle.close for candle in benchmark_candles}
    dates = sorted(set(equity_by_date) & set(benchmark_by_date))
    index_by_date = {value: index for index, value in enumerate(dates)}
    rows = []
    previous_symbols: set[str] = set()
    previous_rebalance_index: int | None = None
    previous_breadth: float | None = None
    recent_champion_returns: list[float] = []
    recent_champion_excess_returns: list[float] = []
    closes_by_symbol = _close_by_symbol(candles_by_symbol or {})
    sector_by_symbol = sector_by_symbol or {}
    for selection in sorted(selections, key=lambda item: item.timestamp):
        rebalance_date = selection.timestamp.date().isoformat()
        index = index_by_date.get(rebalance_date)
        if index is None or index + horizon_days >= len(dates):
            continue
        feature = features_by_date.get(rebalance_date)
        if feature is None:
            continue
        end_date = dates[index + horizon_days]
        symbols = set(selection.symbols)
        scores = [selection.scores[symbol] for symbol in symbols if symbol in selection.scores]
        equity_path = [equity_by_date[date] for date in dates[index:index + horizon_days + 1]]
        peak = equity_path[0]
        future_drawdown = 0.0
        for value in equity_path:
            peak = max(peak, value)
            future_drawdown = min(future_drawdown, (value / peak) - 1.0)
        champion_return = (equity_by_date[end_date] / equity_by_date[rebalance_date]) - 1.0
        benchmark_return = (benchmark_by_date[end_date] / benchmark_by_date[rebalance_date]) - 1.0
        weights = selection.target_weights or {}
        normalized_weights = _normalized_weights(weights)
        recent_champion_return = 0.0
        recent_champion_excess_return = 0.0
        if previous_rebalance_index is not None:
            previous_date = dates[previous_rebalance_index]
            recent_champion_return = (
                (equity_by_date[rebalance_date] / equity_by_date[previous_date]) - 1.0
            )
            recent_benchmark_return = (
                (benchmark_by_date[rebalance_date] / benchmark_by_date[previous_date]) - 1.0
            )
            recent_champion_excess_return = (
                recent_champion_return - recent_benchmark_return
            )
            recent_champion_returns.append(recent_champion_return)
            recent_champion_excess_returns.append(recent_champion_excess_return)
            recent_champion_returns = recent_champion_returns[-2:]
            recent_champion_excess_returns = recent_champion_excess_returns[-2:]
        current_breadth = float(feature["breadth_above_sma_200"])
        rows.append({
            "rebalance_date": rebalance_date,
            "outcome_end_date": end_date,
            "selected_symbols": ",".join(sorted(symbols)),
            "regime_label": str(getattr(selection, "regime_label", "")),
            "selection_count": len(symbols),
            "exposure_target": float(selection.exposure_target),
            "cash_weight": max(0.0, 1.0 - float(selection.exposure_target)),
            "average_rank_score": mean(scores) if scores else 0.0,
            "selected_score_dispersion": pstdev(scores) if len(scores) > 1 else 0.0,
            "largest_weight": max(normalized_weights.values(), default=0.0),
            "selection_weight_herfindahl": sum(
                weight * weight for weight in normalized_weights.values()
            ),
            "selection_overlap_with_prior": (
                len(symbols & previous_symbols) / len(previous_symbols)
                if previous_symbols else 0.0
            ),
            "selection_average_pairwise_correlation_63d": (
                _average_pairwise_correlation(symbols, closes_by_symbol, rebalance_date)
            ),
            "selection_sector_concentration": _sector_concentration(
                normalized_weights,
                sector_by_symbol,
            ),
            "selection_sector_coverage": _sector_coverage(symbols, sector_by_symbol),
            "replacements": len(symbols - previous_symbols),
            "risk_on": float(selection.risk_on),
            "breadth_passes": float(selection.breadth_passes),
            "drawdown_guard_active": float(selection.drawdown_guard_active),
            "chop_filter_active": float(selection.chop_filter_active),
            "spy_distance_sma_200": float(feature["spy_distance_sma_200"]),
            "spy_realized_volatility_21d": float(feature["spy_realized_volatility_21d"]),
            "spy_realized_volatility_63d": float(feature["spy_realized_volatility_63d"]),
            "spy_volatility_ratio_21d_63d": _ratio(
                float(feature["spy_realized_volatility_21d"]),
                float(feature["spy_realized_volatility_63d"]),
            ),
            "spy_max_drawdown_63d": float(feature["spy_max_drawdown_63d"]),
            "spy_max_drawdown_126d": float(feature["spy_max_drawdown_126d"]),
            "breadth_above_sma_200": current_breadth,
            "breadth_change_since_last_rebalance": (
                current_breadth - previous_breadth
                if previous_breadth is not None else 0.0
            ),
            "recent_champion_return": recent_champion_return,
            "recent_champion_excess_return": recent_champion_excess_return,
            "recent_champion_return_2_rebalances": (
                mean(recent_champion_returns) if recent_champion_returns else 0.0
            ),
            "recent_champion_excess_return_2_rebalances": (
                mean(recent_champion_excess_returns)
                if recent_champion_excess_returns else 0.0
            ),
            "champion_return_next_period": champion_return,
            "benchmark_return_next_period": benchmark_return,
            "champion_excess_return": champion_return - benchmark_return,
            "future_max_drawdown": future_drawdown,
            "good_period": int(champion_return > 0),
            "bad_period": int(champion_return < -0.03),
            "underperforms_spy": int(champion_return < benchmark_return),
            "drawdown_event": int(future_drawdown <= -0.10),
        })
        previous_symbols = symbols
        previous_rebalance_index = index
        previous_breadth = current_breadth
    return rows


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 1.0


def _close_by_symbol(candles_by_symbol: dict[str, list[object]]) -> dict[str, dict[str, float]]:
    return {
        symbol: {
            candle.timestamp.date().isoformat(): float(candle.close)
            for candle in candles
            if candle.close > 0
        }
        for symbol, candles in candles_by_symbol.items()
    }


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    absolute_weights = {symbol: abs(float(weight)) for symbol, weight in weights.items()}
    total = sum(absolute_weights.values())
    return (
        {symbol: weight / total for symbol, weight in absolute_weights.items()}
        if total else {}
    )


def _average_pairwise_correlation(
    symbols: set[str],
    closes_by_symbol: dict[str, dict[str, float]],
    rebalance_date: str,
    lookback_days: int = 63,
) -> float:
    correlations = []
    ordered_symbols = sorted(symbols)
    for left_index, left_symbol in enumerate(ordered_symbols):
        for right_symbol in ordered_symbols[left_index + 1:]:
            left = closes_by_symbol.get(left_symbol, {})
            right = closes_by_symbol.get(right_symbol, {})
            dates = sorted(
                date for date in set(left) & set(right) if date <= rebalance_date
            )[-(lookback_days + 1):]
            if len(dates) < lookback_days + 1:
                continue
            left_returns = [
                (left[current] / left[previous]) - 1.0
                for previous, current in zip(dates, dates[1:])
            ]
            right_returns = [
                (right[current] / right[previous]) - 1.0
                for previous, current in zip(dates, dates[1:])
            ]
            correlation = _correlation(left_returns, right_returns)
            if correlation is not None:
                correlations.append(correlation)
    return mean(correlations) if correlations else 0.0


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    if not left_variance or not right_variance:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    return covariance / (left_variance * right_variance) ** 0.5


def _sector_concentration(
    weights: dict[str, float],
    sector_by_symbol: dict[str, str],
) -> float:
    sector_weights: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = sector_by_symbol.get(symbol)
        if sector:
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    return max(sector_weights.values(), default=0.0)


def _sector_coverage(symbols: set[str], sector_by_symbol: dict[str, str]) -> float:
    return sum(symbol in sector_by_symbol for symbol in symbols) / len(symbols) if symbols else 0.0


def write_rebalance_dataset(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["rebalance_date"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
