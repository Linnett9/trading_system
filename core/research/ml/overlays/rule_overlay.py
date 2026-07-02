from __future__ import annotations

def run_rule_exposure_study(rows: list[dict[str, str]], transaction_cost_bps: float) -> list[dict]:
    rules = [
        ("baseline", lambda row: 1.0),
        ("weak_trend_breadth_cap50", lambda row: 0.50 if float(row["spy_distance_sma_200"]) < 0 and float(row["breadth_above_sma_200"]) < 0.50 else 1.0),
        ("high_volatility_cap70", lambda row: 0.70 if float(row["spy_realized_volatility_21d"]) >= 0.22 else 1.0),
        ("combined_stress_cap50", lambda row: 0.50 if float(row["spy_distance_sma_200"]) < 0 and float(row["breadth_above_sma_200"]) < 0.50 and float(row["spy_realized_volatility_21d"]) >= 0.22 else 1.0),
    ]
    return [_simulate_rule(name, rule, rows, transaction_cost_bps) for name, rule in rules]


def run_volatility_managed_walk_forward(
    rows: list[dict[str, str]],
    transaction_cost_bps: float,
    fold_count: int = 3,
) -> list[dict]:
    if len(rows) < fold_count * 3:
        return []
    test_size = len(rows) // (fold_count + 2)
    scenarios = []
    for target_volatility in (0.14, 0.16, 0.18):
        for exposure_floor in (0.50, 0.70):
            folds = []
            for fold_number in range(1, fold_count + 1):
                start = test_size * (fold_number + 1)
                test_rows = rows[start:start + test_size]
                if not test_rows:
                    continue
                def rule(row, target=target_volatility, floor=exposure_floor):
                    base_exposure = float(row["exposure_target"])
                    if base_exposure < 0.80:
                        return base_exposure
                    volatility = float(row["spy_realized_volatility_21d"])
                    multiplier = (
                        min(1.0, max(floor, target / volatility))
                        if volatility else 1.0
                    )
                    return base_exposure * multiplier
                result = _simulate_rule(
                    "volatility_managed",
                    rule,
                    test_rows,
                    transaction_cost_bps,
                )
                baseline = _simulate_rule(
                    "baseline",
                    lambda row: 1.0,
                    test_rows,
                    transaction_cost_bps,
                )
                folds.append({
                    "fold": fold_number,
                    "baseline": baseline,
                    "overlay": result,
                })
            scenarios.append({
                "target_volatility": target_volatility,
                "exposure_floor": exposure_floor,
                "folds": folds,
            })
    return scenarios


def run_drawdown_risk_diagnostics(rows: list[dict[str, str]]) -> dict:
    """Measure whether pre-rebalance conditions precede champion drawdowns."""
    baseline_event_rate = _event_rate(rows)
    conditions = [
        (
            "recent_spy_drawdown",
            "spy_max_drawdown_63d <= -0.10",
            lambda row: float(row["spy_max_drawdown_63d"]) <= -0.10,
        ),
        (
            "market_stress_transition",
            "SPY below SMA 200, breadth deteriorating, and volatility rising",
            lambda row: (
                float(row["spy_distance_sma_200"]) < 0.0
                and float(row["breadth_change_since_last_rebalance"]) <= -0.10
                and float(row["spy_volatility_ratio_21d_63d"]) >= 1.15
            ),
        ),
        (
            "market_stress_with_champion_weakness",
            "market stress transition plus negative recent champion excess return",
            lambda row: (
                float(row["spy_distance_sma_200"]) < 0.0
                and float(row["breadth_change_since_last_rebalance"]) <= -0.10
                and float(row["spy_volatility_ratio_21d_63d"]) >= 1.15
                and float(row["recent_champion_excess_return_2_rebalances"]) < 0.0
            ),
        ),
    ]
    return {
        "outcome": "drawdown_event: future champion drawdown <= -10%",
        "baseline_drawdown_event_rate": baseline_event_rate,
        "conditions": [
            _condition_diagnostic(name, description, predicate, rows, baseline_event_rate)
            for name, description, predicate in conditions
        ],
        "research_only": True,
    }


def _condition_diagnostic(name, description, predicate, rows, baseline_event_rate):
    matched_rows = [row for row in rows if predicate(row)]
    event_rate = _event_rate(matched_rows)
    return {
        "condition": name,
        "description": description,
        "matched_rebalances": len(matched_rows),
        "drawdown_event_count": sum(int(row["drawdown_event"]) for row in matched_rows),
        "drawdown_event_rate": event_rate,
        "drawdown_event_lift": (
            event_rate / baseline_event_rate
            if event_rate is not None and baseline_event_rate else None
        ),
        "mean_champion_return_next_period": _mean_return(matched_rows),
    }


def _event_rate(rows: list[dict[str, str]]) -> float | None:
    return sum(int(row["drawdown_event"]) for row in rows) / len(rows) if rows else None


def _mean_return(rows: list[dict[str, str]]) -> float | None:
    return (
        sum(float(row["champion_return_next_period"]) for row in rows) / len(rows)
        if rows else None
    )


def _simulate_rule(name, rule, rows, transaction_cost_bps):
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    previous_effective_exposure = 0.0
    turnover = 0.0
    triggered = 0
    for row in rows:
        base_exposure = float(row["exposure_target"])
        cap = rule(row)
        effective_exposure = min(base_exposure, cap)
        if effective_exposure < base_exposure:
            triggered += 1
        turnover += abs(effective_exposure - previous_effective_exposure)
        cost = abs(effective_exposure - previous_effective_exposure) * transaction_cost_bps / 10_000
        multiplier = effective_exposure / base_exposure if base_exposure else 0.0
        equity *= (1.0 - cost) * (1.0 + float(row["champion_return_next_period"]) * multiplier)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
        previous_effective_exposure = effective_exposure
    return {
        "rule": name,
        "total_return": equity - 1.0,
        "max_drawdown": max_drawdown,
        "triggered_rebalances": triggered,
        "rebalance_count": len(rows),
        "overlay_turnover": turnover,
        "transaction_cost_bps": transaction_cost_bps,
    }
