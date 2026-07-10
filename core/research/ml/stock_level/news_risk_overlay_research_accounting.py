from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping


def portfolio_comparison(
    rows: Iterable[Mapping[str, Any]],
    *,
    price_score_column: str,
    return_column: str,
    top_n: int,
    starting_equity: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_date.setdefault(str(row.get("decision_timestamp", ""))[:10], []).append(row)
    control_periods = []
    experiment_periods = []
    for date_key in sorted(by_date):
        ranked = sorted(
            by_date[date_key],
            key=lambda row: _number(row.get(price_score_column)) or float("-inf"),
            reverse=True,
        )[: max(1, top_n)]
        if not ranked:
            continue
        control_periods.append(
            period_accounting_row(
                date_key,
                ranked,
                return_column=return_column,
                overlay=False,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
        )
        experiment_periods.append(
            period_accounting_row(
                date_key,
                ranked,
                return_column=return_column,
                overlay=True,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
        )
    control_curve = equity_curve(control_periods, starting_equity=starting_equity, prefix="price_only")
    experiment_curve = equity_curve(
        experiment_periods,
        starting_equity=starting_equity,
        prefix="price_plus_news",
    )
    control_stats = portfolio_stats(control_curve, return_column="price_only_period_return_net")
    experiment_stats = portfolio_stats(experiment_curve, return_column="price_plus_news_period_return_net")
    return {
        "price_score_column": price_score_column,
        "return_column": return_column,
        "top_n": top_n,
        "starting_equity": starting_equity,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "accounting_approximation": (
            "Decision-level marked-to-market approximation using realized forward returns "
            "from candidate artifacts. Overlapping holdings are approximated by one "
            "equal-weight decision-period basket per timestamp because the artifacts do "
            "not include full open-position daily mark-to-market paths."
        ),
        "price_only": control_stats,
        "price_plus_news": experiment_stats,
        "news_overlay_lowered_drawdown": (
            experiment_stats["maximum_drawdown"] > control_stats["maximum_drawdown"]
        ),
        "incremental_total_return_decimal": (
            experiment_stats["total_return_decimal"] - control_stats["total_return_decimal"]
        ),
        "equity_curve": merge_curves(control_curve, experiment_curve),
        "drawdown_curve": drawdown_curve(control_curve, experiment_curve),
    }


def period_accounting_row(
    date_key: str,
    rows: list[Mapping[str, Any]],
    *,
    return_column: str,
    overlay: bool,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    # trade_return_net is the candidate realized forward return after applying
    # overlay exposure and subtracting one-way transaction cost plus slippage.
    # It is not a portfolio total return until basket returns are compounded
    # through _equity_curve as ending_equity / starting_equity - 1.
    selected = []
    total_cost_bps = transaction_cost_bps + slippage_bps
    for row in rows:
        multiplier = float(row.get("news_position_multiplier", 1.0) or 0.0) if overlay else 1.0
        gross_return = (_number(row.get(return_column)) or 0.0) * multiplier
        trade_cost = abs(multiplier) * total_cost_bps / 10_000.0
        selected.append(
            {
                "symbol": row.get("symbol", ""),
                "gross_return": gross_return,
                "trade_return_net": gross_return - trade_cost,
                "gross_exposure": abs(multiplier),
                "transaction_cost": trade_cost,
                "max_adverse_excursion": adverse_excursion(row),
            }
        )
    denominator = max(len(selected), 1)
    return {
        "date": date_key,
        "period_return_net": sum(row["trade_return_net"] for row in selected) / denominator,
        "gross_exposure": sum(row["gross_exposure"] for row in selected) / denominator,
        "net_exposure": sum(row["gross_exposure"] for row in selected) / denominator,
        "transaction_costs": sum(row["transaction_cost"] for row in selected) / denominator,
        "number_of_positions": sum(row["gross_exposure"] > 0 for row in selected),
        "worst_trade": min((row["trade_return_net"] for row in selected), default=0.0),
        "maximum_adverse_excursion": min(
            (row["max_adverse_excursion"] for row in selected if row["max_adverse_excursion"] is not None),
            default=0.0,
        ),
    }


def equity_curve(
    period_rows: list[Mapping[str, Any]],
    *,
    starting_equity: float,
    prefix: str,
) -> list[dict[str, Any]]:
    equity = starting_equity
    peak = starting_equity
    curve = []
    previous_exposure = 0.0
    drawdown_duration = 0
    for row in period_rows:
        period_return = float(row["period_return_net"])
        equity *= 1.0 + period_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak else 0.0
        drawdown_duration = drawdown_duration + 1 if drawdown < 0 else 0
        exposure = float(row["gross_exposure"])
        curve.append(
            {
                "date": row["date"],
                f"{prefix}_period_return_net": period_return,
                f"{prefix}_ending_equity": equity,
                f"{prefix}_drawdown": drawdown,
                f"{prefix}_drawdown_duration": drawdown_duration,
                f"{prefix}_gross_exposure": exposure,
                f"{prefix}_net_exposure": float(row["net_exposure"]),
                f"{prefix}_turnover": abs(exposure - previous_exposure),
                f"{prefix}_transaction_costs": float(row["transaction_costs"]),
                f"{prefix}_number_of_positions": int(row["number_of_positions"]),
                f"{prefix}_worst_trade": float(row["worst_trade"]),
                f"{prefix}_maximum_adverse_excursion": float(row["maximum_adverse_excursion"]),
            }
        )
        previous_exposure = exposure
    return curve


def portfolio_stats(curve: list[Mapping[str, Any]], *, return_column: str) -> dict[str, float]:
    if not curve:
        return {
            "periods": 0,
            "starting_equity": 0.0,
            "ending_equity": 0.0,
            "total_return_decimal": 0.0,
            "total_return_percent": 0.0,
            "wealth_multiple": 0.0,
            "CAGR": 0.0,
            "annualised_volatility": 0.0,
            "maximum_drawdown": 0.0,
            "average_drawdown": 0.0,
            "longest_drawdown_duration": 0.0,
            "Sharpe_ratio": 0.0,
            "Sortino_ratio": 0.0,
            "Calmar_ratio": 0.0,
            "worst_day": 0.0,
            "worst_trade": 0.0,
            "maximum_adverse_excursion": 0.0,
            "expected_shortfall_CVaR_5pct": 0.0,
            "average_gross_exposure": 0.0,
            "average_net_exposure": 0.0,
            "turnover": 0.0,
            "transaction_costs": 0.0,
            "number_of_positions": 0.0,
        }
    prefix = return_column.replace("_period_return_net", "")
    returns = [float(row[return_column]) for row in curve]
    starting_equity = float(curve[0][f"{prefix}_ending_equity"]) / (1.0 + returns[0])
    ending_equity = float(curve[-1][f"{prefix}_ending_equity"])
    wealth_multiple = ending_equity / starting_equity if starting_equity else 0.0
    total_return_decimal = wealth_multiple - 1.0
    periods_per_year = 252.0
    years = max(len(returns) / periods_per_year, 1.0 / periods_per_year)
    downside = [min(value, 0.0) for value in returns]
    volatility = pstdev(returns) * math.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    downside_volatility = pstdev(downside) * math.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    average_period_return = mean(returns)
    maximum_drawdown = min(float(row[f"{prefix}_drawdown"]) for row in curve)
    expected_shortfall = expected_shortfall_cvar(returns)
    return {
        "periods": len(returns),
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return_decimal": total_return_decimal,
        "total_return_percent": total_return_decimal * 100.0,
        "wealth_multiple": wealth_multiple,
        "CAGR": wealth_multiple ** (1.0 / years) - 1.0 if wealth_multiple > 0 else -1.0,
        "annualised_volatility": volatility,
        "maximum_drawdown": maximum_drawdown,
        "average_drawdown": mean(float(row[f"{prefix}_drawdown"]) for row in curve),
        "longest_drawdown_duration": max(float(row[f"{prefix}_drawdown_duration"]) for row in curve),
        "Sharpe_ratio": (average_period_return * periods_per_year) / volatility if volatility else 0.0,
        "Sortino_ratio": (average_period_return * periods_per_year) / downside_volatility if downside_volatility else 0.0,
        "Calmar_ratio": (wealth_multiple ** (1.0 / years) - 1.0) / abs(maximum_drawdown) if maximum_drawdown else 0.0,
        "worst_day": min(returns),
        "worst_trade": min(float(row[f"{prefix}_worst_trade"]) for row in curve),
        "maximum_adverse_excursion": min(float(row[f"{prefix}_maximum_adverse_excursion"]) for row in curve),
        "expected_shortfall_CVaR_5pct": expected_shortfall,
        "average_gross_exposure": mean(float(row[f"{prefix}_gross_exposure"]) for row in curve),
        "average_net_exposure": mean(float(row[f"{prefix}_net_exposure"]) for row in curve),
        "turnover": sum(float(row[f"{prefix}_turnover"]) for row in curve),
        "transaction_costs": sum(float(row[f"{prefix}_transaction_costs"]) for row in curve),
        "number_of_positions": sum(float(row[f"{prefix}_number_of_positions"]) for row in curve),
    }


def merge_curves(
    control_curve: list[Mapping[str, Any]],
    experiment_curve: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for control, experiment in zip(control_curve, experiment_curve):
        rows.append({**control, **{k: v for k, v in experiment.items() if k != "date"}})
    return rows


def drawdown_curve(
    control_curve: list[Mapping[str, Any]],
    experiment_curve: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for control, experiment in zip(control_curve, experiment_curve):
        rows.append(
            {
                "date": control["date"],
                "price_only_drawdown": control["price_only_drawdown"],
                "price_plus_news_drawdown": experiment["price_plus_news_drawdown"],
                "price_only_drawdown_duration": control["price_only_drawdown_duration"],
                "price_plus_news_drawdown_duration": experiment["price_plus_news_drawdown_duration"],
            }
        )
    return rows


def adverse_excursion(row: Mapping[str, Any]) -> float | None:
    for column in (
        "actual_max_adverse_excursion",
        "forward_max_adverse_excursion",
        "max_adverse_excursion",
    ):
        value = _number(row.get(column))
        if value is not None:
            return value
    drawdown = _number(row.get("actual_future_drawdown"))
    if drawdown is not None:
        return -abs(drawdown)
    return None


def expected_shortfall_cvar(returns: list[float], tail_fraction: float = 0.05) -> float:
    if not returns:
        return 0.0
    count = max(1, math.ceil(len(returns) * tail_fraction))
    return mean(sorted(returns)[:count])


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
