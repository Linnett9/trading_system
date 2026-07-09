from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from core.research.framework.ranking import finite_number


GUARDRAILS = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
    "promotion_thresholds_changed": False,
}


@dataclass(frozen=True)
class BarCadenceReplayResult:
    summary: dict[str, Any]
    periods: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    payload: dict[str, Any]


def build_bar_cadence_portfolio_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal_column: str,
    timeframe: str,
    top_n: int = 25,
    max_position_weight: float = 0.05,
    min_position_weight: float = 0.0,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    decision_frequency_bars: int = 1,
    scoring_cadence: str | None = None,
    decision_cadence: str | None = None,
    retraining_cadence: str = "external",
    execution_timing: str = "next_eligible_bar_open",
    max_workers: int = 1,
    starting_equity: float = 1.0,
    annualization_factor: float | None = None,
    min_cagr_periods: int | None = None,
) -> BarCadenceReplayResult:
    """Replay scored bars with bar-close decisions and next-bar execution.

    The input rows are expected to be point-in-time feature/scoring rows for one
    timeframe. The score at timestamp T may only affect orders executed at the
    next eligible bar timestamp.
    """
    if execution_timing != "next_eligible_bar_open":
        raise ValueError("execution_timing must be next_eligible_bar_open")
    if top_n < 1:
        raise ValueError("top_n must be at least one")
    if decision_frequency_bars < 1:
        raise ValueError("decision_frequency_bars must be at least one")
    if not 0.0 < max_position_weight <= 1.0:
        raise ValueError("max_position_weight must be in (0, 1]")
    if min_position_weight < 0.0:
        raise ValueError("min_position_weight cannot be negative")
    if starting_equity <= 0.0:
        raise ValueError("starting_equity must be positive")

    grouped = _group_rows(rows, signal_column=signal_column, timeframe=timeframe)
    timestamps = sorted(grouped)
    decision_timestamps = timestamps[::decision_frequency_bars]
    weights_by_decision = _target_weights_by_decision(
        grouped,
        decision_timestamps,
        top_n=top_n,
        cap=max_position_weight,
        floor=min_position_weight,
    )

    periods: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    gross_equity = float(starting_equity)
    net_equity = float(starting_equity)
    for index, decision_timestamp in enumerate(decision_timestamps):
        execution_timestamp = _next_timestamp(timestamps, decision_timestamp)
        if execution_timestamp is None:
            break
        next_decision_timestamp = (
            decision_timestamps[index + 1]
            if index + 1 < len(decision_timestamps)
            else None
        )
        exit_timestamp = (
            _next_timestamp(timestamps, next_decision_timestamp)
            if next_decision_timestamp is not None
            else None
        )
        if exit_timestamp is None:
            break
        weights = weights_by_decision.get(decision_timestamp, {})
        period_decisions = _decision_rows(
            decision_timestamp,
            execution_timestamp,
            previous_weights,
            weights,
        )
        turnover = sum(
            abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in set(weights) | set(previous_weights)
        )
        gross = _gross_open_to_open_return(
            grouped,
            weights,
            execution_timestamp,
            exit_timestamp,
        )
        cost_fraction = turnover * (float(cost_bps) + float(slippage_bps)) / 10_000.0
        transaction_cost_amount = net_equity * cost_fraction
        net = gross - cost_fraction
        gross_equity *= 1.0 + gross
        net_equity = net_equity * (1.0 + gross) - transaction_cost_amount
        periods.append(
            {
                "decision_timestamp": _iso(decision_timestamp),
                "execution_timestamp": _iso(execution_timestamp),
                "exit_timestamp": _iso(exit_timestamp),
                "execution_timing": execution_timing,
                "signal_column": signal_column,
                "gross_return": gross,
                "transaction_cost_fraction": cost_fraction,
                "transaction_cost_amount": transaction_cost_amount,
                "net_return": net,
                "turnover": turnover,
                "gross_equity": gross_equity,
                "net_equity": net_equity,
                "equity": net_equity,
                "position_count": len(weights),
                "gross_exposure": sum(abs(weight) for weight in weights.values()),
            }
        )
        decisions.extend(period_decisions)
        previous_weights = weights

    scores = [
        {
            "score_timestamp": _iso(timestamp),
            "symbol": row["symbol"],
            "signal_column": signal_column,
            "score": row["score"],
            "timeframe": timeframe,
        }
        for timestamp in timestamps
        for row in sorted(grouped[timestamp], key=lambda item: item["symbol"])
        if row["score"] is not None
    ]
    factor = annualization_factor if annualization_factor is not None else _annualization_factor(timeframe)
    summary = _summary(
        periods,
        decisions,
        starting_equity=starting_equity,
        annualization_factor=factor,
        min_cagr_periods=min_cagr_periods if min_cagr_periods is not None else _min_cagr_periods(timeframe),
    )
    requested_workers = max(1, int(max_workers))
    effective_workers = min(requested_workers, len(decision_timestamps) or 1)
    payload = {
        "mode": "bar_cadence_portfolio_replay_research_only",
        "timeframe": timeframe,
        "signal_column": signal_column,
        "scoring_cadence": scoring_cadence or f"every_completed_{timeframe}_bar",
        "decision_cadence": decision_cadence or f"every_{decision_frequency_bars}_{timeframe}_bar",
        "retraining_cadence": retraining_cadence,
        "execution_timing": execution_timing,
        "score_timestamp_semantics": "completed_bar_close",
        "decision_timestamp_semantics": "after_completed_bar_score",
        "annualization_convention": _annualization_convention(timeframe, factor),
        "score_count": len(scores),
        "decision_event_count": len(decisions),
        "period_count": len(periods),
        "parallelism": {
            "requested_workers": requested_workers,
            "effective_workers": 1,
            "backend": "sequential_stateful_replay",
            "stateful_execution_workers": 1,
            "independent_unit_workers": effective_workers,
            "nested_sklearn_n_jobs": 1,
            "nested_torch_num_threads": 1,
            "work_unit": "chronological portfolio state propagation",
            "full_dataset_copy_per_worker": False,
            "parallelism_audit": (
                "BUY/SELL/HOLD decisions, turnover, holdings, and equity are "
                "path-dependent, so execution/accounting is chronological. "
                "Parallel workers are reserved for independent model/policy/cost units."
            ),
        },
        "summary": summary,
        **GUARDRAILS,
    }
    return BarCadenceReplayResult(summary, periods, decisions, scores, payload)


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal_column: str,
    timeframe: str,
) -> dict[datetime, list[dict[str, Any]]]:
    grouped: dict[datetime, list[dict[str, Any]]] = {}
    for raw in rows:
        if str(raw.get("timeframe", timeframe)) != timeframe:
            continue
        timestamp = _timestamp(raw.get("timestamp"))
        symbol = str(raw.get("symbol", "")).strip().upper()
        if timestamp is None or not symbol:
            continue
        open_price = finite_number(raw.get("open"))
        close_price = finite_number(raw.get("close"))
        if open_price is None or close_price is None or open_price <= 0.0:
            continue
        grouped.setdefault(timestamp, []).append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "open": open_price,
                "close": close_price,
                "score": finite_number(raw.get(signal_column)),
            }
        )
    return grouped


def _target_weights_by_decision(
    grouped: Mapping[datetime, list[dict[str, Any]]],
    decision_timestamps: Sequence[datetime],
    *,
    top_n: int,
    cap: float,
    floor: float,
) -> dict[datetime, dict[str, float]]:
    return {
        timestamp: _weights(grouped.get(timestamp, []), top_n, cap, floor)
        for timestamp in decision_timestamps
    }


def _weights(rows: Sequence[Mapping[str, Any]], top_n: int, cap: float, floor: float) -> dict[str, float]:
    ordered = sorted(
        (row for row in rows if row["score"] is not None),
        key=lambda row: (-float(row["score"]), str(row["symbol"])),
    )[:top_n]
    if not ordered:
        return {}
    raw_weight = min(float(cap), 1.0 / len(ordered))
    if raw_weight < floor:
        return {}
    return {str(row["symbol"]): raw_weight for row in ordered}


def _decision_rows(
    decision_timestamp: datetime,
    execution_timestamp: datetime,
    previous: Mapping[str, float],
    current: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for symbol in sorted(set(previous) | set(current)):
        before = float(previous.get(symbol, 0.0))
        after = float(current.get(symbol, 0.0))
        if after > before:
            action = "BUY"
        elif after < before:
            action = "SELL"
        else:
            action = "HOLD"
        rows.append(
            {
                "decision_timestamp": _iso(decision_timestamp),
                "execution_timestamp": _iso(execution_timestamp),
                "symbol": symbol,
                "action": action,
                "previous_weight": before,
                "target_weight": after,
                "weight_delta": after - before,
            }
        )
    return rows


def _gross_open_to_open_return(
    grouped: Mapping[datetime, list[dict[str, Any]]],
    weights: Mapping[str, float],
    execution_timestamp: datetime,
    exit_timestamp: datetime,
) -> float:
    entry = {row["symbol"]: row["open"] for row in grouped.get(execution_timestamp, [])}
    exit_ = {row["symbol"]: row["open"] for row in grouped.get(exit_timestamp, [])}
    gross = 0.0
    for symbol, weight in weights.items():
        entry_price = entry.get(symbol)
        exit_price = exit_.get(symbol)
        if entry_price is None or exit_price is None or entry_price <= 0.0:
            continue
        gross += float(weight) * (float(exit_price) / float(entry_price) - 1.0)
    return gross


def _summary(
    periods: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    starting_equity: float,
    annualization_factor: float | None,
    min_cagr_periods: int,
) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in periods]
    gross_returns = [float(row["gross_return"]) for row in periods]
    net_equity_curve = [starting_equity, *[float(row["net_equity"]) for row in periods]]
    gross_equity_curve = [starting_equity, *[float(row["gross_equity"]) for row in periods]]
    ending_net_equity = net_equity_curve[-1]
    ending_gross_equity = gross_equity_curve[-1]
    peak = starting_equity
    max_drawdown = 0.0
    for value in net_equity_curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    action_counts = {
        action: sum(1 for row in decisions if row["action"] == action)
        for action in ("BUY", "SELL", "HOLD")
    }
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    annualized_volatility = volatility * (annualization_factor ** 0.5) if annualization_factor else None
    net_total_return = ending_net_equity / starting_equity - 1.0
    gross_total_return = ending_gross_equity / starting_equity - 1.0
    total_transaction_cost_amount = sum(float(row["transaction_cost_amount"]) for row in periods)
    cagr = _cagr(ending_net_equity / starting_equity, annualization_factor, len(returns), min_cagr_periods)
    sharpe = (
        mean(returns) / volatility * (annualization_factor ** 0.5)
        if returns and volatility > 0.0 and annualization_factor
        else None
    )
    return {
        "starting_equity": starting_equity,
        "ending_gross_equity": ending_gross_equity,
        "ending_net_equity": ending_net_equity,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
        "total_return": net_total_return,
        "total_return_semantics": "net_total_return",
        "cagr": cagr,
        "annualization_factor": annualization_factor,
        "annualization_min_cagr_periods": min_cagr_periods,
        "net_return": sum(returns),
        "gross_return": sum(gross_returns),
        "total_transaction_cost_amount": total_transaction_cost_amount,
        "transaction_cost_fraction_of_starting_equity": total_transaction_cost_amount / starting_equity,
        "return_drag_attributable_to_costs": gross_total_return - net_total_return,
        "transaction_cost_drag": gross_total_return - net_total_return,
        "cost_drag": gross_total_return - net_total_return,
        "sharpe": sharpe,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "mean_period_return": mean(returns) if returns else None,
        "volatility": volatility,
        "average_turnover": mean([float(row["turnover"]) for row in periods]) if periods else None,
        "trade_count": sum(1 for row in decisions if row["action"] in {"BUY", "SELL"}),
        "average_holdings": mean([int(row["position_count"]) for row in periods]) if periods else 0.0,
        "average_exposure": mean([float(row["gross_exposure"]) for row in periods]) if periods else 0.0,
        "period_count": len(periods),
        "action_counts": action_counts,
    }


def _annualization_factor(timeframe: str) -> float:
    if timeframe == "1Day":
        return 252.0
    if timeframe == "1h":
        return 6.5 * 252.0
    if timeframe == "5m":
        return 78.0 * 252.0
    return 252.0


def _min_cagr_periods(timeframe: str) -> int:
    if timeframe == "1Day":
        return 20
    if timeframe == "1h":
        return 65
    if timeframe == "5m":
        return 390
    return 20


def _annualization_convention(timeframe: str, annualization_factor: float | None) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "annualization_factor": annualization_factor,
        "basis": (
            "252 trading days"
            if timeframe == "1Day"
            else "6.5 regular-session hours * 252 trading days"
            if timeframe == "1h"
            else "78 regular-session 5-minute bars * 252 trading days"
            if timeframe == "5m"
            else "configured annualization factor"
        ),
        "short_smoke_metrics_are_diagnostic": True,
    }


def _cagr(equity_multiple: float, annualization: float | None, periods: int, min_periods: int) -> float | None:
    if not annualization or periods < min_periods or equity_multiple <= 0.0:
        return None
    exponent = annualization / periods
    if exponent > 100.0:
        return None
    try:
        return equity_multiple ** exponent - 1.0
    except OverflowError:
        return None


def _next_timestamp(timestamps: Sequence[datetime], timestamp: datetime | None) -> datetime | None:
    if timestamp is None:
        return None
    for candidate in timestamps:
        if candidate > timestamp:
            return candidate
    return None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(value: datetime) -> str:
    return value.isoformat()
