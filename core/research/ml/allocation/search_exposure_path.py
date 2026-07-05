from __future__ import annotations

from typing import Any

from core.research.ml.allocation.allocation_v2_variants import AllocationVariant
from core.research.ml.allocation.exposures import _forecast_values, _variant_scores
from core.research.ml.allocation.utils import _finite_float


def _selected_optimizer_exposure_path(
    rows: list[dict[str, str]],
    exposures: list[float],
    transaction_cost_bps: float,
    variant: AllocationVariant,
) -> list[dict[str, Any]]:
    if len(rows) != len(exposures):
        raise ValueError("Optimizer exposure path rows and exposures must align")
    scores = _variant_scores(rows, variant)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row, exposure, score in zip(rows, exposures, scores):
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not date:
            raise ValueError("Optimizer exposure path row is missing rebalance_date")
        grouped.setdefault(date, []).append({
            "period_return": _finite_float(
                row.get("champion_return_next_period", 0.0) or 0.0
            ),
            "exposure": _finite_float(exposure),
            "score": _finite_float(score),
            "predicted_forward_return": _mean_or_none(
                _forecast_values(row, "predicted_forward_return_10d")
                or _forecast_values(row, "predicted_forward_return_5d")
            ),
            "predicted_future_drawdown": _mean_or_none(
                _forecast_values(row, "predicted_future_drawdown")
                or _forecast_values(row, "predicted_max_adverse_excursion")
            ),
            "predicted_future_volatility": _mean_or_none(
                _forecast_values(row, "predicted_future_volatility")
            ),
            "outcome_end_date": str(
                row.get("outcome_end_date")
                or row.get("label_end_date")
                or date
            ),
            "selected_symbols": _selected_symbols(row),
        })

    equity = 1.0
    peak = 1.0
    previous_exposure = 1.0
    path_rows = []
    for date, values in sorted(grouped.items()):
        period_return = sum(value["period_return"] for value in values) / len(values)
        exposure = sum(value["exposure"] for value in values) / len(values)
        turnover = abs(exposure - previous_exposure)
        cost = turnover * transaction_cost_bps / 10_000.0
        net_return = (period_return * exposure) - cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        path_rows.append({
            "rebalance_date": date,
            "outcome_end_date": max(
                str(value["outcome_end_date"]) for value in values
            ),
            "source_row_count": len(values),
            "period_return": period_return,
            "exposure": exposure,
            "score": sum(value["score"] for value in values) / len(values),
            "predicted_forward_return": _mean_or_none(
                [value["predicted_forward_return"] for value in values]
            ),
            "predicted_future_drawdown": _mean_or_none([
                value["predicted_future_drawdown"]
                for value in values
                if value["predicted_future_drawdown"] is not None
            ]),
            "predicted_future_volatility": _mean_or_none([
                value["predicted_future_volatility"]
                for value in values
                if value["predicted_future_volatility"] is not None
            ]),
            "turnover": turnover,
            "transaction_cost_bps": transaction_cost_bps,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
            "drawdown": drawdown,
            "selected_symbols": sorted({
                symbol
                for value in values
                for symbol in value["selected_symbols"]
            }),
            "research_only": True,
            "trading_impact": "none",
            "production_validated": False,
        })
        previous_exposure = exposure
    return path_rows
def _selected_symbols(row: dict[str, Any]) -> list[str]:
    raw_value = row.get("selected_symbols", "")
    if isinstance(raw_value, list):
        return [str(symbol) for symbol in raw_value if str(symbol)]
    return [
        symbol.strip()
        for symbol in str(raw_value).split(",")
        if symbol.strip()
    ]
def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None
