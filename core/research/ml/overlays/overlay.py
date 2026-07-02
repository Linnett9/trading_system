from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowOverlayResult:
    base_total_return: float
    overlay_total_return: float
    base_max_drawdown: float
    overlay_max_drawdown: float
    reduced_exposure_days: int
    evaluated_days: int
    overlay_turnover: float
    estimated_cost: float


def simulate_shadow_overlay(
    equity_by_date: dict[str, float],
    probabilities_by_date: dict[str, float],
    threshold: float,
    reduced_exposure: float,
    rebalance_dates: set[str] | None = None,
    transaction_cost_bps: float = 0.0,
    reduce_when: str = "below_threshold",
) -> ShadowOverlayResult | None:
    dates = sorted(date for date in probabilities_by_date if date in equity_by_date)
    if len(dates) < 2:
        return None

    base_start = equity_by_date[dates[0]]
    base_equity = base_start
    overlay_equity = base_start
    base_peak = base_start
    overlay_peak = overlay_equity
    base_max_drawdown = 0.0
    overlay_max_drawdown = 0.0
    reduced_days = 0
    overlay_turnover = 0.0
    estimated_cost = 0.0
    active_multiplier = 1.0

    for previous_date, current_date in zip(dates, dates[1:]):
        previous_equity = equity_by_date[previous_date]
        current_equity = equity_by_date[current_date]
        base_return = (current_equity / previous_equity) - 1.0
        if rebalance_dates is None or previous_date in rebalance_dates:
            probability = probabilities_by_date[previous_date]
            next_multiplier = (
                reduced_exposure
                if should_reduce_exposure(probability, threshold, reduce_when)
                else 1.0
            )
            turnover = abs(next_multiplier - active_multiplier)
            overlay_turnover += turnover
            cost = turnover * transaction_cost_bps / 10_000
            overlay_equity *= 1.0 - cost
            estimated_cost += cost
            active_multiplier = next_multiplier
        if active_multiplier < 1.0:
            reduced_days += 1
        base_equity *= 1.0 + base_return
        overlay_equity *= 1.0 + (base_return * active_multiplier)
        base_peak = max(base_peak, base_equity)
        overlay_peak = max(overlay_peak, overlay_equity)
        base_max_drawdown = min(base_max_drawdown, (base_equity / base_peak) - 1.0)
        overlay_max_drawdown = min(
            overlay_max_drawdown,
            (overlay_equity / overlay_peak) - 1.0,
        )

    return ShadowOverlayResult(
        base_total_return=(base_equity / base_start) - 1.0,
        overlay_total_return=(overlay_equity / base_start) - 1.0,
        base_max_drawdown=base_max_drawdown,
        overlay_max_drawdown=overlay_max_drawdown,
        reduced_exposure_days=reduced_days,
        evaluated_days=len(dates) - 1,
        overlay_turnover=overlay_turnover,
        estimated_cost=estimated_cost,
    )


def should_reduce_exposure(
    probability: float,
    threshold: float,
    reduce_when: str,
) -> bool:
    if reduce_when == "above_or_equal_threshold":
        return float(probability) >= float(threshold)
    if reduce_when == "below_threshold":
        return float(probability) < float(threshold)
    raise ValueError(f"Unsupported overlay reduce_when rule: {reduce_when}")


def overlay_decision_rule(label_type: str) -> tuple[str, str]:
    if label_type == "should_reduce_exposure":
        return (
            "above_or_equal_threshold",
            "reduce_exposure_when_should_reduce_exposure_probability_gte_threshold",
        )
    if label_type == "drawdown_risk":
        return (
            "above_or_equal_threshold",
            "reduce_exposure_when_drawdown_risk_probability_gte_threshold",
        )
    return (
        "below_threshold",
        "reduce_exposure_when_champion_success_probability_lt_threshold",
    )
