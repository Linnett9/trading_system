from __future__ import annotations

from typing import Any

from core.research.ml.audits.benchmark_relative_validation_types import COST_STRESS_BPS, RESEARCH_METADATA
from core.research.ml.audits.benchmark_relative_validation_math import _compound, _equity_curve, _max_drawdown, _number, _sharpe, _sortino


def _score_candidate(
    candidate: dict[str, Any],
    flagged_dates: set[str],
) -> dict[str, Any]:
    rows = sorted(candidate.get("rows", []), key=lambda row: row["rebalance_date"])
    if not rows:
        return {
            "candidate_name": candidate["candidate_name"],
            "available": False,
            "skip_reason": "no aligned canonical rows",
            **RESEARCH_METADATA,
        }
    returns = [float(row.get("net_return") or 0.0) for row in rows]
    anomaly_returns = [
        float(row.get("net_return") or 0.0)
        for row in rows
        if str(row.get("rebalance_date")) not in flagged_dates
    ]
    turnovers = _turnover_by_row(rows)
    total_return = _compound(returns)
    anomaly_return = _compound(anomaly_returns)
    positive_returns = sorted((value for value in returns if value > 0.0), reverse=True)
    positive_total = sum(positive_returns)
    symbol_contributions = _symbol_contributions(rows)
    positive_symbol_total = sum(
        value for value in symbol_contributions.values() if value > 0.0
    )
    cost_returns = {
        f"cost_stressed_return_{bps}bps": _compound([
            period_return - (turnover * bps / 10_000.0)
            for period_return, turnover in zip(returns, turnovers)
        ])
        for bps in COST_STRESS_BPS
    }
    ratio = max(
        0.0,
        (total_return - anomaly_return) / max(abs(total_return), 1e-12),
    )
    curve = _equity_curve(returns)
    return {
        "candidate_name": candidate["candidate_name"],
        "available": True,
        "canonical_non_overlap_return": total_return,
        "anomaly_adjusted_return": anomaly_return,
        "anomaly_dependency_ratio": ratio,
        "max_drawdown": _max_drawdown(curve),
        "sharpe": _sharpe(returns, rows),
        "sortino": _sortino(returns, rows),
        "turnover": sum(turnovers),
        **cost_returns,
        "top_1_date_profit_share": (
            positive_returns[0] / positive_total
            if positive_returns and positive_total else None
        ),
        "top_5_date_profit_share": (
            sum(positive_returns[:5]) / positive_total
            if positive_returns and positive_total else None
        ),
        "top_1_symbol_profit_share": (
            max(symbol_contributions.values(), default=0.0) / positive_symbol_total
            if positive_symbol_total else None
        ),
        "canonical_period_count": len(rows),
        "flagged_period_count": sum(
            str(row.get("rebalance_date")) in flagged_dates for row in rows
        ),
        **RESEARCH_METADATA,
    }

def _merge_existing_concentration(
    row: dict[str, Any],
    concentration: dict[str, Any],
) -> dict[str, Any]:
    if not row.get("available") or not concentration:
        return row
    anomaly_return = next(
        (
            _number(scenario.get("summary", {}).get("total_return"))
            for scenario in concentration.get("scenarios", []) or []
            if scenario.get("scenario_name") == "remove_anomaly_dates"
        ),
        None,
    )
    metrics = concentration.get("profit_concentration", {})
    total_return = float(row["canonical_non_overlap_return"])
    if anomaly_return is not None:
        row["anomaly_adjusted_return"] = anomaly_return
        row["anomaly_dependency_ratio"] = max(
            0.0,
            (total_return - anomaly_return) / max(abs(total_return), 1e-12),
        )
    mappings = {
        "top_1_date_profit_share": "top_1_date_positive_return_share",
        "top_5_date_profit_share": "top_5_date_positive_return_share",
        "top_1_symbol_profit_share": "top_1_symbol_contribution_share",
    }
    for output_name, source_name in mappings.items():
        value = _number(metrics.get(source_name))
        if value is not None:
            row[output_name] = value
    return row

def _turnover_by_row(rows: list[dict[str, Any]]) -> list[float]:
    previous: dict[str, float] = {}
    turnovers = []
    for row in rows:
        exposure = float(row.get("exposure", 1.0) or 0.0)
        weights = {
            str(symbol): float(weight) * exposure
            for symbol, weight in (row.get("target_weights", {}) or {}).items()
        }
        if not weights:
            symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
            weights = {
                symbol: exposure / len(symbols) for symbol in symbols
            } if symbols else {}
        assets = set(previous) | set(weights)
        asset_change = sum(abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in assets)
        cash_change = abs((1.0 - sum(weights.values())) - (1.0 - sum(previous.values())))
        turnovers.append(0.5 * (asset_change + cash_change))
        previous = weights
    return turnovers

def _symbol_contributions(rows: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        symbols = [str(symbol) for symbol in row.get("selected_symbols", [])]
        if not symbols:
            continue
        weights = dict(row.get("target_weights", {}) or {})
        total_weight = sum(float(weights.get(symbol, 0.0) or 0.0) for symbol in symbols)
        for symbol in symbols:
            weight = (
                float(weights.get(symbol, 0.0) or 0.0) / total_weight
                if total_weight > 0.0 else 1.0 / len(symbols)
            )
            output[symbol] = output.get(symbol, 0.0) + float(
                row.get("net_return") or 0.0
            ) * weight
    return output
