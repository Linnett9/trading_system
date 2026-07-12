from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from core.research.ml.stock_level.prediction_artifacts.math import (
    _trailing_volatility,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
)


@dataclass(frozen=True)
class ForwardTarget:
    value: float | str
    start_timestamp: str
    label_start_timestamp: str
    end_timestamp: str
    available_timestamp: str
    horizon: int

    @property
    def horizon_label(self) -> str:
        return f"{self.horizon}_trading_observations"


def _actual_targets(
    symbol_data: dict[str, Any],
    rebalance_date: str,
    *,
    market_data: dict[str, Any] | None = None,
    decision_dates: list[str] | tuple[str, ...] | None = None,
    decision_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    closes_by_date = symbol_data.get("close", {})
    dates = symbol_data.get("close_dates", [])
    if rebalance_date not in closes_by_date:
        return _empty_targets(decision_metadata)
    index_by_date = symbol_data.get("close_index_by_date", {})
    index = index_by_date.get(rebalance_date)
    if index is None:
        return _empty_targets(decision_metadata)
    start = closes_by_date[rebalance_date]
    if start <= 0.0:
        return _empty_targets(decision_metadata)
    forward_5d = _forward_target(
        symbol_data,
        rebalance_date,
        5,
        decision_dates=decision_dates,
    )
    forward_10d = _forward_target(
        symbol_data,
        rebalance_date,
        10,
        decision_dates=decision_dates,
    )
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
    raw_10d = forward_10d.value
    benchmark_10d = _forward_target(
        market_data or {},
        rebalance_date,
        10,
        decision_dates=decision_dates,
    )
    market_10d = benchmark_10d.value
    pre_vol = _trailing_volatility(dates, symbol_data.get("close_values", []), rebalance_date, lookback=20)
    adverse = min(drawdowns) if drawdowns else ""
    targets = {
        "actual_forward_return_5d": forward_5d.value,
        "actual_forward_return_10d": forward_10d.value,
        "actual_future_volatility": pstdev(returns) if len(returns) > 1 else "",
        "actual_future_drawdown": min(drawdowns) if drawdowns else "",
        "actual_max_adverse_excursion": min(drawdowns) if drawdowns else "",
        "actual_benchmark_return_10d": market_10d,
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
    targets.update(_target_provenance(rebalance_date, forward_10d, benchmark_10d, decision_metadata or {}))
    return targets


def _forward_target(
    symbol_data: dict[str, Any],
    rebalance_date: str,
    horizon: int,
    *,
    decision_dates: list[str] | tuple[str, ...] | None = None,
) -> ForwardTarget:
    closes_by_date = symbol_data.get("close", {})
    dates = list(symbol_data.get("close_dates", []))
    if rebalance_date not in closes_by_date:
        return ForwardTarget("", "", "", "", "", horizon)
    index_by_date = symbol_data.get("close_index_by_date", {})
    index = index_by_date.get(rebalance_date)
    if index is None:
        return ForwardTarget("", "", "", "", "", horizon)
    start = closes_by_date[rebalance_date]
    end_index = index + horizon
    if start <= 0.0 or end_index >= len(dates):
        return ForwardTarget("", "", "", "", "", horizon)
    end_timestamp = dates[end_index]
    end = closes_by_date[end_timestamp]
    if end <= 0.0:
        return ForwardTarget("", "", "", "", "", horizon)
    label_start = dates[index + 1] if index + 1 < len(dates) else ""
    candidates = sorted(set(decision_dates or dates))
    available = _first_after(candidates, end_timestamp)
    if not available:
        return ForwardTarget("", "", "", "", "", horizon)
    return ForwardTarget(
        value=(end / start) - 1.0,
        start_timestamp=rebalance_date,
        label_start_timestamp=label_start,
        end_timestamp=end_timestamp,
        available_timestamp=available,
        horizon=horizon,
    )


def _target_provenance(
    rebalance_date: str,
    target: ForwardTarget,
    benchmark: ForwardTarget,
    decision_metadata: dict[str, Any],
) -> dict[str, Any]:
    if target.value == "":
        output = {column: "" for column in TARGET_PROVENANCE_COLUMNS}
        output.update(_decision_contract_fields(decision_metadata))
        output["target_status"] = "unrealized_boundary"
        return output
    return {
        **_decision_contract_fields(decision_metadata),
        "target_provenance_contract_version": TARGET_PROVENANCE_CONTRACT_VERSION,
        "target_horizon": target.horizon_label,
        "target_observation_count": target.horizon,
        "target_start_timestamp": target.start_timestamp,
        "label_start_timestamp": target.label_start_timestamp,
        "label_end_timestamp": target.end_timestamp,
        "label_available_timestamp": target.available_timestamp,
        "target_price_convention": "simple_close_to_close",
        "benchmark_target_start_timestamp": benchmark.start_timestamp,
        "benchmark_label_start_timestamp": benchmark.label_start_timestamp,
        "benchmark_label_end_timestamp": benchmark.end_timestamp,
        "benchmark_label_available_timestamp": benchmark.available_timestamp,
        "target_status": "realized",
    }


def _first_after(values: list[str], timestamp: str) -> str:
    return next((value for value in values if value > timestamp), "")


def _decision_contract_fields(decision_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_session_date": decision_metadata.get("decision_session_date", ""),
        "feature_timestamp": decision_metadata.get("decision_session_date", ""),
        "feature_data_cutoff_timestamp": decision_metadata.get("feature_data_cutoff_timestamp", ""),
        "decision_timestamp": decision_metadata.get("decision_timestamp", ""),
        "first_actionable_session": decision_metadata.get("first_actionable_session", ""),
        "decision_grid_version": decision_metadata.get("decision_grid_version", ""),
        "decision_grid_identity": decision_metadata.get("decision_grid_identity", ""),
        "exchange_calendar_identity": decision_metadata.get("exchange_calendar_identity", ""),
        "decision_frequency": decision_metadata.get("decision_frequency", "source"),
        "target_horizon_trading_days": decision_metadata.get("target_horizon_trading_days", 10),
        "overlapping_targets": decision_metadata.get("overlapping_targets", False),
        "required_purge_horizon_trading_days": decision_metadata.get("required_purge_horizon_trading_days", 10),
    }


def _empty_targets(decision_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        **{column: "" for column in ACTUAL_COLUMNS},
        **{column: "" for column in TARGET_PROVENANCE_COLUMNS},
        **_decision_contract_fields(decision_metadata or {}),
        "target_status": "missing_source_price",
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
