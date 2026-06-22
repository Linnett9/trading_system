from __future__ import annotations

from collections import Counter
from datetime import datetime


def build_champion_robustness_report(
    result,
    candles_by_symbol: dict,
    sector_by_symbol: dict[str, str],
    transaction_cost_scenarios: dict[str, object],
    universe_stress_scenarios: dict[str, object],
    benchmark_symbol: str = "SPY",
) -> dict:
    baseline = _summary(result, candles_by_symbol, benchmark_symbol)
    concentration = _concentration(result.selections, candles_by_symbol, sector_by_symbol)
    rolling = _rolling_windows(result, candles_by_symbol.get(benchmark_symbol, []))
    cost_summaries = {
        name: _summary(item, candles_by_symbol, benchmark_symbol)
        for name, item in transaction_cost_scenarios.items()
    }
    stress_summaries = {
        name: _summary(item, candles_by_symbol, benchmark_symbol)
        for name, item in universe_stress_scenarios.items()
    }
    points = result.result.equity_curve
    data_start = min(
        (candles[0].timestamp for candles in candles_by_symbol.values() if candles),
        default=None,
    )
    full_history_guard_passed = (
        len(points) > 1
        and (points[-1].timestamp - points[0].timestamp).days >= (9 * 365 - 10)
    )
    failures = []
    if baseline["excess_return_vs_spy"] <= 0:
        failures.append("does_not_beat_spy")
    if concentration["sector_mapping_coverage"] < 1:
        failures.append("missing_sector_mapping")
    if any(item["total_return"] <= 0 for item in cost_summaries.values()):
        failures.append("not_cost_resilient")
    if not full_history_guard_passed:
        failures.append("full_history_guard_failed")
    return {
        "mode": "deterministic_champion_robustness_research_only",
        "baseline": baseline,
        "concentration": concentration,
        "transaction_cost_sensitivity": cost_summaries,
        "universe_stress_tests": stress_summaries,
        "rolling_three_year": rolling,
        "effective_backtest_start_date": points[0].timestamp.date().isoformat() if points else None,
        "effective_backtest_end_date": points[-1].timestamp.date().isoformat() if points else None,
        "warmup_days_removed": (points[0].timestamp - data_start).days if points and data_start else None,
        "rebalance_count": len(result.selections),
        "equity_curve_point_count": len(points),
        "paper_candidate_checks": {
            "full_history_guard_passed": full_history_guard_passed,
            "period_exclusion_audit_passed": False,
            "cost_sensitivity_passed": not any(item["total_return"] <= 0 for item in cost_summaries.values()),
            "sector_concentration_warning": concentration["max_sector_weight"] is not None and concentration["max_sector_weight"] > 0.60,
            "correlation_warning": concentration["max_pairwise_correlation"] is not None and concentration["max_pairwise_correlation"] > 0.85,
            "turnover_warning": baseline["annualized_turnover"] > 7.0,
        },
        "robustness_pass": not failures,
        "main_failure_reason": failures[0] if failures else None,
        "production_readiness": "research_only",
        "research_only": True,
        "trading_impact": "none",
    }


def period_exclusion_summary(result, candles_by_symbol: dict, benchmark_symbol: str, start: str, end: str) -> dict:
    start_at = datetime.fromisoformat(start)
    end_at = datetime.fromisoformat(end)
    points = [point for point in result.result.equity_curve if not start_at <= point.timestamp <= end_at]
    summary = _curve_summary(points, candles_by_symbol.get(benchmark_symbol, []), benchmark_symbol)
    source_points = result.result.equity_curve
    overlapping_points = [
        point for point in source_points if start_at <= point.timestamp <= end_at
    ]
    return {
        "method": "post_hoc_equity_curve_exclusion",
        "counterfactual_strategy_rerun": False,
        "excluded_period": f"{start} to {end}",
        "backtest_start_date": source_points[0].timestamp.date().isoformat() if source_points else None,
        "backtest_end_date": source_points[-1].timestamp.date().isoformat() if source_points else None,
        "period_overlapped_backtest": bool(overlapping_points),
        "excluded_point_count": len(overlapping_points),
        "baseline_point_count": len(source_points),
        "remaining_point_count": len(points),
        "warning": (
            "Excluded period does not overlap the effective backtest curve."
            if not overlapping_points else None
        ),
        "metrics": summary,
    }


def _summary(result, candles_by_symbol: dict, benchmark_symbol: str) -> dict:
    summary = _curve_summary(result.result.equity_curve, candles_by_symbol.get(benchmark_symbol, []), benchmark_symbol)
    selections = result.selections
    returns = _rebalance_returns(result.result.equity_curve, selections)
    summary.update({
        "cagr": result.cagr,
        "sharpe_ratio": result.result.sharpe,
        "calmar_ratio": result.calmar,
        "turnover": result.turnover_percent,
        "annualized_turnover": result.annualized_turnover_percent,
        "average_holdings": _mean([len(item.symbols) for item in selections]),
        "average_cash_weight": _mean([1 - item.exposure_target for item in selections]),
        "average_exposure": _mean([item.exposure_target for item in selections]),
        "rebalance_win_rate": _mean([float(value > 0) for value in returns]),
        "average_rebalance_return": _mean(returns),
        "best_rebalance_return": max(returns) if returns else None,
        "worst_rebalance_return": min(returns) if returns else None,
    })
    return summary


def _curve_summary(points, benchmark_candles, benchmark_symbol: str) -> dict:
    if len(points) < 2:
        return {"total_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0, "benchmark_return_vs_spy": None, "excess_return_vs_spy": None, "underperformance_frequency_vs_spy": None}
    returns = [(right.equity / left.equity) - 1 for left, right in zip(points, points[1:]) if left.equity]
    benchmark = {item.timestamp.date(): item.close for item in benchmark_candles}
    aligned = [(point, benchmark.get(point.timestamp.date())) for point in points]
    aligned = [(point, price) for point, price in aligned if price]
    benchmark_return = (aligned[-1][1] / aligned[0][1]) - 1 if len(aligned) > 1 else None
    benchmark_daily = [(right[1] / left[1]) - 1 for left, right in zip(aligned, aligned[1:])]
    underperformance = _mean([float(left < right) for left, right in zip(returns, benchmark_daily)]) if benchmark_daily else None
    return {
        "total_return": (points[-1].equity / points[0].equity) - 1,
        "annualized_volatility": _volatility(returns),
        "max_drawdown": _max_drawdown([point.equity for point in points]),
        "benchmark_symbol": benchmark_symbol,
        "benchmark_return_vs_spy": benchmark_return,
        "excess_return_vs_spy": ((points[-1].equity / points[0].equity) - 1) - benchmark_return if benchmark_return is not None else None,
        "underperformance_frequency_vs_spy": underperformance,
    }


def _concentration(selections, candles_by_symbol, sector_by_symbol):
    correlations=[]; weights=[]; sectors=[]; overlaps=[]; dispersions=[]; previous=set()
    for selection in selections:
        symbols=set(selection.symbols); normalized=_normalized(selection.target_weights or {})
        weights.extend(normalized.values()); sectors.append(_sector_weights(normalized, sector_by_symbol));
        overlaps.append(len(symbols & previous) / len(previous) if previous else 0.0); previous=symbols
        scores=[selection.scores[s] for s in symbols if s in selection.scores]; dispersions.append(_std(scores))
        correlations.extend(_pairwise_correlations(symbols, candles_by_symbol, selection.timestamp))
    mapped=sum(symbol in sector_by_symbol for selection in selections for symbol in selection.symbols)
    selected=sum(len(selection.symbols) for selection in selections)
    return {
        "average_pairwise_correlation": _mean(correlations), "max_pairwise_correlation": max(correlations) if correlations else None,
        "average_herfindahl_weight_concentration": _mean([sum(value * value for value in _normalized(item.target_weights or {}).values()) for item in selections]),
        "max_single_name_weight": max(weights) if weights else None,
        "average_sector_concentration": _mean([sum(value * value for value in item.values()) for item in sectors]),
        "max_sector_weight": max((max(item.values(), default=0.0) for item in sectors), default=None),
        "average_sectors_held": _mean([len(item) for item in sectors]), "average_overlap_with_prior": _mean(overlaps),
        "average_score_dispersion": _mean(dispersions), "sector_mapping_coverage": mapped / selected if selected else 1.0,
    }


def _rolling_windows(result, benchmark_candles):
    points=result.result.equity_curve; window=756; output=[]
    for end in range(window, len(points), 63):
        segment=points[end-window:end+1]; summary=_curve_summary(segment, benchmark_candles, "SPY")
        output.append({"end_date": segment[-1].timestamp.date().isoformat(), **summary})
    return output


def _rebalance_returns(points, selections):
    equity={point.timestamp.date(): point.equity for point in points}; dates=sorted(equity); values=[]
    for current, following in zip(selections, selections[1:]):
        start=current.timestamp.date(); end=following.timestamp.date()
        if start in equity and end in equity: values.append((equity[end]/equity[start])-1)
    return values


def _pairwise_correlations(symbols, candles_by_symbol, timestamp):
    returns=[]
    for symbol in symbols:
        closes=[c.close for c in candles_by_symbol.get(symbol, []) if c.timestamp <= timestamp][-64:]
        returns.append([(right/left)-1 for left,right in zip(closes, closes[1:]) if left])
    values=[]
    for index,left in enumerate(returns):
        for right in returns[index+1:]:
            if len(left)==len(right) and len(left)>1: values.append(_correlation(left,right))
    return values


def _normalized(weights):
    total=sum(weights.values()); return {key:value/total for key,value in weights.items()} if total else {}
def _sector_weights(weights, mapping):
    values=Counter()
    for symbol,weight in weights.items():
        if symbol in mapping: values[mapping[symbol]]+=weight
    return dict(values)
def _mean(values):
    return sum(values)/len(values) if values else None
def _std(values):
    average=_mean(values); return (sum((value-average)**2 for value in values)/len(values))**0.5 if values and average is not None else 0.0
def _volatility(returns):
    return _std(returns)*(252**0.5) if returns else 0.0
def _max_drawdown(values):
    peak=values[0]; drawdown=0.0
    for value in values: peak=max(peak,value); drawdown=min(drawdown,(value/peak)-1)
    return drawdown
def _correlation(left,right):
    left_mean=_mean(left); right_mean=_mean(right); denominator=(sum((x-left_mean)**2 for x in left)*sum((x-right_mean)**2 for x in right))**0.5
    return sum((x-left_mean)*(y-right_mean) for x,y in zip(left,right))/denominator if denominator else 0.0
