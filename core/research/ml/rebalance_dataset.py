from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

import yaml

from core.research.dual_momentum_factory import build_dual_momentum_tester


def build_champion_rebalance_rows(
    feature_rows: list[dict[str, float | str]],
    selections: list[object],
    equity_curve: list[object],
    benchmark_candles: list[object],
    horizon_days: int,
    candles_by_symbol: dict[str, list[object]] | None = None,
    sector_by_symbol: dict[str, str] | None = None,
    pairwise_correlation_cache: dict[tuple[tuple[str, ...], str], float] | None = None,
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
    pairwise_correlation_cache = pairwise_correlation_cache or {}
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
        forward_return_5d = _forward_return(equity_by_date, dates, index, 5)
        forward_return_10d = _forward_return(equity_by_date, dates, index, 10)
        future_path_metrics = _future_path_metrics(equity_path)
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
                _average_pairwise_correlation(
                    symbols,
                    closes_by_symbol,
                    rebalance_date,
                    cache=pairwise_correlation_cache,
                )
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
            "forward_return_5d": forward_return_5d,
            "forward_return_10d": forward_return_10d,
            "future_volatility": future_path_metrics["future_volatility"],
            "future_drawdown": future_drawdown,
            "future_max_drawdown": future_drawdown,
            "max_adverse_excursion": future_path_metrics["max_adverse_excursion"],
            "max_favourable_excursion": future_path_metrics[
                "max_favourable_excursion"
            ],
            "good_period": int(champion_return > 0),
            "bad_period": int(champion_return < -0.03),
            "underperforms_spy": int(champion_return < benchmark_return),
            "drawdown_event": int(future_drawdown <= -0.10),
        })
        previous_symbols = symbols
        previous_rebalance_index = index
        previous_breadth = current_breadth
    return rows


def build_expanded_rebalance_rows(
    config: dict,
    feature_rows: list[dict[str, float | str]],
    candles_by_symbol: dict[str, list[object]],
    benchmark_symbol: str,
    horizon_days: int,
    sector_by_symbol: dict[str, str] | None = None,
) -> tuple[list[dict[str, float | str]], dict]:
    ml_config = config.get("ml", {})
    expanded_config = ml_config.get("expanded_rebalance_dataset", {})
    frequencies = expanded_config.get(
        "rebalance_frequencies",
        ["monthly", "biweekly", "weekly"],
    )
    top_n_values = expanded_config.get("top_n_values", [3, 5, 7])
    weightings = expanded_config.get("weightings", ["equal", "inverse_volatility"])
    universe_paths = expanded_config.get(
        "universe_paths",
        [
            "data/reference/universes/current_32.yaml",
            "data/reference/universes/us_liquid_100.yaml",
        ],
    )
    base_dual_config = dict(config.get("research", {}).get("dual_momentum", {}))
    benchmark_candles = candles_by_symbol.get(benchmark_symbol, [])
    rows: list[dict[str, float | str]] = []
    variant_payloads = []
    pairwise_correlation_cache: dict[tuple[tuple[str, ...], str], float] = {}
    seen_universe_symbol_sets: dict[tuple[str, ...], str] = {}

    for universe_path in universe_paths:
        universe_name, symbols = _universe_symbols(universe_path, base_dual_config)
        max_symbols = expanded_config.get("max_symbols")
        if max_symbols:
            symbols = symbols[: int(max_symbols)]
        available_symbols = [
            symbol for symbol in symbols
            if symbol in candles_by_symbol
        ]
        symbol_set_key = tuple(sorted(set(available_symbols)))
        duplicate_of = seen_universe_symbol_sets.get(symbol_set_key)
        if duplicate_of is not None:
            for frequency in frequencies:
                for top_n in top_n_values:
                    for weighting in weightings:
                        variant_payloads.append({
                            "variant_id": (
                                f"{universe_name}_{frequency}_top{int(top_n)}_{weighting}"
                            ),
                            "universe": universe_name,
                            "duplicate_of": duplicate_of,
                            "row_count": 0,
                            "skipped": True,
                            "reason": "duplicate_universe_symbol_set",
                        })
            continue
        seen_universe_symbol_sets[symbol_set_key] = universe_name
        for frequency in frequencies:
            for top_n in top_n_values:
                for weighting in weightings:
                    variant_id = (
                        f"{universe_name}_{frequency}_top{int(top_n)}_{weighting}"
                    )
                    if len(available_symbols) < max(2, int(top_n)):
                        variant_payloads.append({
                            "variant_id": variant_id,
                            "universe": universe_name,
                            "row_count": 0,
                            "skipped": True,
                            "reason": "insufficient_available_symbols",
                        })
                        continue
                    variant_config = {
                        **base_dual_config,
                        "symbols": available_symbols,
                        "rebalance_frequency": frequency,
                        "top_n": int(top_n),
                        "max_selected_assets": int(top_n),
                        "weighting": str(weighting),
                        "experiment_name": variant_id,
                    }
                    result = build_dual_momentum_tester(config, variant_config).run(
                        {symbol: candles_by_symbol[symbol] for symbol in available_symbols}
                    )
                    variant_rows = build_champion_rebalance_rows(
                        feature_rows,
                        result.selections,
                        result.result.equity_curve,
                        benchmark_candles,
                        horizon_days,
                        candles_by_symbol=candles_by_symbol,
                        sector_by_symbol=sector_by_symbol,
                        pairwise_correlation_cache=pairwise_correlation_cache,
                    )
                    for row_index, row in enumerate(variant_rows):
                        feature_date = str(row["rebalance_date"])
                        enriched = dict(row)
                        enriched.update({
                            "feature_id": f"{variant_id}_{feature_date}_{row_index}",
                            "feature_date": feature_date,
                            "label_start_date": _next_label_start_date(
                                feature_date,
                                row["outcome_end_date"],
                            ),
                            "label_end_date": str(row["outcome_end_date"]),
                            "variant_id": variant_id,
                            "variant_rebalance_frequency": str(frequency),
                            "variant_top_n": int(top_n),
                            "variant_weighting": str(weighting),
                            "variant_universe": universe_name,
                            "variant_universe_symbol_count": len(available_symbols),
                            "volatility_adjusted_excess_return": _ratio(
                                float(row["champion_excess_return"]),
                                max(float(row.get("spy_realized_volatility_21d", 0.0)), 1e-9),
                            ),
                        })
                        enriched["should_reduce_exposure"] = should_reduce_exposure_label(
                            enriched,
                            drawdown_threshold=float(
                                expanded_config.get("reduce_drawdown_threshold", 0.08)
                            ),
                            excess_return_threshold=float(
                                expanded_config.get("reduce_excess_return_threshold", -0.01)
                            ),
                            volatility_adjusted_threshold=float(
                                expanded_config.get(
                                    "reduce_volatility_adjusted_threshold",
                                    -0.10,
                                )
                            ),
                        )
                        rows.append(enriched)
                    variant_payloads.append({
                        "variant_id": variant_id,
                        "universe": universe_name,
                        "rebalance_frequency": str(frequency),
                        "top_n": int(top_n),
                        "weighting": str(weighting),
                        "available_symbols": len(available_symbols),
                        "row_count": len(variant_rows),
                        "skipped": False,
                    })
    audit = {
        "row_count": len(rows),
        "variant_count": len(variant_payloads),
        "variants": variant_payloads,
        "backtest_years": config.get("backtest", {}).get("years"),
        "ml_research_years": ml_config.get("research_years"),
        "effective_research_years": config.get("backtest", {}).get("years"),
        "should_reduce_exposure_rate": _row_rate(rows, "should_reduce_exposure"),
        "drawdown_event_rate": _row_rate(rows, "drawdown_event"),
        "underperforms_spy_rate": _row_rate(rows, "underperforms_spy"),
        "universe_paths": [str(path) for path in universe_paths],
        "research_only": True,
        "trading_impact": "none",
    }
    return rows, audit


def write_expanded_rebalance_audit(path: Path, audit: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def should_reduce_exposure_label(
    row: dict[str, float | str],
    drawdown_threshold: float = 0.08,
    excess_return_threshold: float = -0.01,
    volatility_adjusted_threshold: float = -0.10,
) -> int:
    return int(
        float(row["future_max_drawdown"]) <= -abs(drawdown_threshold)
        or float(row["champion_excess_return"]) <= excess_return_threshold
        or float(row["volatility_adjusted_excess_return"]) <= volatility_adjusted_threshold
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 1.0


def _forward_return(
    values_by_date: dict[str, float],
    dates: list[str],
    start_index: int,
    horizon_days: int,
) -> float | None:
    end_index = start_index + horizon_days
    if end_index >= len(dates):
        return None
    start = values_by_date[dates[start_index]]
    end = values_by_date[dates[end_index]]
    return (end / start) - 1.0 if start else None


def _future_path_metrics(values: list[float]) -> dict[str, float]:
    if len(values) < 2 or values[0] <= 0:
        return {
            "future_volatility": 0.0,
            "max_adverse_excursion": 0.0,
            "max_favourable_excursion": 0.0,
        }
    returns = [
        (current / previous) - 1.0
        for previous, current in zip(values, values[1:])
        if previous > 0
    ]
    cumulative_returns = [(value / values[0]) - 1.0 for value in values]
    return {
        "future_volatility": pstdev(returns) if len(returns) > 1 else 0.0,
        "max_adverse_excursion": min(cumulative_returns),
        "max_favourable_excursion": max(cumulative_returns),
    }


def _universe_symbols(
    universe_path: str,
    base_dual_config: dict,
) -> tuple[str, list[str]]:
    path = Path(str(universe_path))
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        symbols = [str(symbol).upper() for symbol in payload.get("symbols", [])]
        return str(payload.get("name", path.stem)), symbols
    symbols = [str(symbol).upper() for symbol in base_dual_config.get("symbols", [])]
    return path.stem, symbols


def _next_label_start_date(feature_date: str, label_end_date: str) -> str:
    start = date.fromisoformat(feature_date) + timedelta(days=1)
    end = date.fromisoformat(str(label_end_date))
    return min(start, end).isoformat()


def _row_rate(rows: list[dict[str, float | str]], key: str) -> float | None:
    return sum(int(row[key]) for row in rows) / len(rows) if rows else None


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
    cache: dict[tuple[tuple[str, ...], str], float] | None = None,
) -> float:
    cache_key = (tuple(sorted(symbols)), rebalance_date)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
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
    value = mean(correlations) if correlations else 0.0
    if cache is not None:
        cache[cache_key] = value
    return value


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
