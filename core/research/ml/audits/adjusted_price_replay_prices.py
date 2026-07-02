from __future__ import annotations

from typing import Any

from core.research.ml.audits.adjusted_data_loading import _number
from core.research.ml.audits.adjusted_price_replay_config import _adjusted_replay_config


def _weighted_period_return(
    row: dict[str, Any],
    closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]] | None = None,
    config: dict[str, Any] | None = None,
) -> float | None:
    raw_closes = raw_closes or {}
    config = config or _adjusted_replay_config({})
    symbols = [str(symbol).upper() for symbol in row.get("selected_symbols", [])]
    if not symbols:
        return None
    raw_weights = {
        str(symbol).upper(): _number(weight)
        for symbol, weight in (row.get("target_weights", {}) or {}).items()
    }
    returns = {}
    for symbol in symbols:
        start = _replay_close(
            closes,
            raw_closes,
            symbol,
            str(row.get("rebalance_date", "")),
            config,
        )
        end = _replay_close(
            closes,
            raw_closes,
            symbol,
            str(row.get("outcome_end_date", "")),
            config,
        )
        if start is None or end is None or start <= 0:
            return None
        returns[symbol] = (end / start) - 1.0
    if len(returns) != len(symbols):
        return None
    weights = {
        symbol: raw_weights.get(symbol)
        for symbol in returns
        if raw_weights.get(symbol) is not None
    }
    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0.0:
        weights = {symbol: 1.0 / len(returns) for symbol in returns}
    else:
        weights = {
            symbol: float(weight) / total_weight
            for symbol, weight in weights.items()
        }
    return sum(returns[symbol] * weights[symbol] for symbol in weights)

def _raw_fallback_available(
    raw_closes: dict[str, dict[str, float]],
    symbol: str,
    start_date: str,
    end_date: str,
) -> bool:
    start = raw_closes.get(symbol, {}).get(start_date)
    end = raw_closes.get(symbol, {}).get(end_date)
    return start is not None and end is not None and start > 0

def _replay_close(
    adjusted_closes: dict[str, dict[str, float]],
    raw_closes: dict[str, dict[str, float]],
    symbol: str,
    day: str,
    config: dict[str, Any],
) -> float | None:
    adjusted = adjusted_closes.get(symbol, {}).get(day)
    if adjusted is not None and adjusted > 0:
        return adjusted
    if config["missing_symbol_policy"] != "fallback_raw":
        return None
    raw = raw_closes.get(symbol, {}).get(day)
    if raw is not None and raw > 0:
        return raw
    return None
