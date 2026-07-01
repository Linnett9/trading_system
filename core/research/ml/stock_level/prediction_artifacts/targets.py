from __future__ import annotations

import math
from statistics import pstdev
from typing import Any

from core.research.ml.stock_level.prediction_artifacts.math import (
    _forward_return,
    _trailing_volatility,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
)


def _actual_targets(
    symbol_data: dict[str, Any],
    rebalance_date: str,
    *,
    market_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    closes_by_date = symbol_data.get("close", {})
    dates = symbol_data.get("close_dates", [])
    if rebalance_date not in closes_by_date:
        return {column: "" for column in ACTUAL_COLUMNS}
    index = dates.index(rebalance_date)
    start = closes_by_date[rebalance_date]
    if start <= 0.0:
        return {column: "" for column in ACTUAL_COLUMNS}
    forward = []
    for horizon in (5, 10):
        end_index = index + horizon
        if end_index < len(dates):
            forward.append((horizon, closes_by_date[dates[end_index]] / start - 1.0))
        else:
            forward.append((horizon, ""))
    future_prices = [
        closes_by_date[date]
        for date in dates[index + 1 : index + 11]
        if closes_by_date[date] > 0.0
    ]
    returns = [
        (future_prices[i] / future_prices[i - 1]) - 1.0
        for i in range(1, len(future_prices))
        if future_prices[i - 1] > 0.0
    ]
    drawdowns = [(price / start) - 1.0 for price in future_prices]
    raw_10d = forward[1][1]
    market_10d = _forward_return(market_data or {}, rebalance_date, 10)
    pre_vol = _trailing_volatility(dates, symbol_data.get("close_values", []), rebalance_date, lookback=20)
    adverse = min(drawdowns) if drawdowns else ""
    return {
        "actual_forward_return_5d": forward[0][1],
        "actual_forward_return_10d": forward[1][1],
        "actual_future_volatility": pstdev(returns) if len(returns) > 1 else "",
        "actual_future_drawdown": min(drawdowns) if drawdowns else "",
        "actual_max_adverse_excursion": min(drawdowns) if drawdowns else "",
        "actual_market_residual_return_10d": (
            raw_10d - market_10d if raw_10d != "" and market_10d != "" else ""
        ),
        "actual_vol_adjusted_forward_return_10d": (
            raw_10d / pre_vol
            if raw_10d != "" and pre_vol != "" and pre_vol > 0.0
            else ""
        ),
        "actual_drawdown_adjusted_forward_return_10d": (
            raw_10d - abs(min(0.0, adverse))
            if raw_10d != "" and adverse != ""
            else ""
        ),
        "actual_rank_normalized_forward_return_10d": "",
        "actual_top_decile_label_10d": "",
    }

def _add_cross_sectional_targets(rows: list[dict[str, Any]]) -> None:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("actual_forward_return_10d") != "":
            by_date.setdefault(str(row["rebalance_date"]), []).append(row)
    for date_rows in by_date.values():
        ordered = sorted(date_rows, key=lambda row: (float(row["actual_forward_return_10d"]), str(row["symbol"])))
        count = len(ordered)
        top_count = max(1, math.ceil(count * 0.1))
        top_symbols = {row["symbol"] for row in ordered[-top_count:]}
        for index, row in enumerate(ordered):
            row["actual_rank_normalized_forward_return_10d"] = (
                index / (count - 1) if count > 1 else 0.5
            )
            row["actual_top_decile_label_10d"] = int(row["symbol"] in top_symbols)
