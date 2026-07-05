from __future__ import annotations

from typing import Any


def _canonical_schedule(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    exact = canonical.get("candidates", {}).get("exact_champion_replay", {})
    return [
        dict(row)
        for row in exact.get("rows", []) or []
        if row.get("included_in_canonical") and not row.get("exclusion_reason")
    ]

def _market_baseline(
    name: str,
    schedule: list[dict[str, Any]],
    closes: dict[str, dict[str, float]],
    symbol: str,
) -> dict[str, Any]:
    rows = []
    for period in schedule:
        value = _price_return(closes, symbol, period)
        if value is None:
            continue
        rows.append(_baseline_row(period, value, {symbol: 1.0}))
    return {"candidate_name": name, "rows": rows}

def _selected_universe_baseline(
    name: str,
    schedule: list[dict[str, Any]],
    closes: dict[str, dict[str, float]],
    *,
    equal_weight: bool,
) -> dict[str, Any]:
    rows = []
    for period in schedule:
        symbols = [str(symbol) for symbol in period.get("selected_symbols", [])]
        source_weights = dict(period.get("target_weights", {}) or {})
        returns = {
            symbol: value
            for symbol in symbols
            if (value := _price_return(closes, symbol, period)) is not None
        }
        if not returns:
            continue
        if equal_weight:
            weights = {symbol: 1.0 / len(returns) for symbol in returns}
        else:
            weights = {
                symbol: float(source_weights.get(symbol, 0.0) or 0.0)
                for symbol in returns
            }
            total_weight = sum(weights.values())
            weights = (
                {symbol: weight / total_weight for symbol, weight in weights.items()}
                if total_weight > 0.0
                else {symbol: 1.0 / len(returns) for symbol in returns}
            )
        period_return = sum(returns[symbol] * weights[symbol] for symbol in returns)
        rows.append(_baseline_row(period, period_return, weights))
    return {"candidate_name": name, "rows": rows}

def _baseline_row(
    period: dict[str, Any],
    period_return: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "rebalance_date": str(period.get("rebalance_date", "")),
        "outcome_end_date": str(period.get("outcome_end_date", "")),
        "period_return": period_return,
        "net_return": period_return,
        "exposure": 1.0,
        "selected_symbols": sorted(weights),
        "target_weights": weights,
    }

def _canonical_candidate(
    canonical: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    candidate = canonical.get("candidates", {}).get(name, {})
    rows = [
        dict(row)
        for row in candidate.get("rows", []) or []
        if row.get("included_in_canonical") and not row.get("exclusion_reason")
    ]
    return {"candidate_name": name, "rows": rows}

def _price_return(
    closes: dict[str, dict[str, float]],
    symbol: str,
    period: dict[str, Any],
) -> float | None:
    values = closes.get(symbol.upper(), {})
    start = values.get(str(period.get("rebalance_date", "")))
    end = values.get(str(period.get("outcome_end_date", "")))
    if start is None or end is None or start <= 0.0:
        return None
    return (end / start) - 1.0
