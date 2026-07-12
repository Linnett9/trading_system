from __future__ import annotations

import math
from statistics import mean
from typing import Any

from core.research.ml.stock_level.stock_level_alpha_features import (
    ENGINEERED_FEATURE_COLUMNS,
)
from core.research.ml.stock_level_benchmark_execution import _walk_forward_partitions
from core.research.ml.stock_level_benchmark_types import (
    AUXILIARY_TARGET_COLUMNS,
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_OUTPUT_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
    TARGET_PROVENANCE_CONTRACT_VERSION,
)


def _build_oos_prediction_rows(
    prepared_rows: list[dict[str, Any]],
    dates: list[str],
    *,
    first_test_index: int,
    test_window_dates: int,
    embargo_dates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold_id, train_rows, test_rows, train_dates, test_dates, embargoed_dates, purged_row_count in (
        _walk_forward_partitions(
            prepared_rows,
            dates,
            first_test_index=first_test_index,
            test_window_dates=test_window_dates,
            embargo_dates=embargo_dates,
        )
    ):
        predictions.extend(_base_prediction_row(row, fold_id) for row in test_rows)
        train_label_available = [
            str(row.get("label_available_timestamp", row.get("label_end_timestamp", row["rebalance_date"])))
            for row in train_rows
        ]
        test_dates_only = [str(row["rebalance_date"]) for row in test_rows]
        folds.append(
            {
                "fold_id": fold_id,
                "decision_timestamp": test_dates[0],
                "train_start_date": train_dates[0],
                "train_end_date": train_dates[-1],
                "train_date_count": len(train_dates),
                "train_row_count": len(train_rows),
                "training_feature_date_min": train_dates[0],
                "training_feature_date_max": train_dates[-1],
                "training_label_available_max": max(train_label_available)
                if train_label_available else None,
                "embargoed_dates": embargoed_dates,
                "purged_row_count": purged_row_count,
                "embargoed_row_count": len(embargoed_dates) * len({row["symbol"] for row in test_rows}),
                "test_start_date": test_dates[0],
                "test_end_date": test_dates[-1],
                "test_date_count": len(test_dates),
                "test_row_count": len(test_rows),
                "oos_prediction_date_min": min(test_dates_only) if test_dates_only else None,
                "oos_prediction_date_max": max(test_dates_only) if test_dates_only else None,
                "preprocessing_fit_row_count": len(train_rows),
                "chronological_guard_passed": train_dates[-1] < test_dates[0],
                "label_availability_guard_passed": (
                    max(train_label_available) <= test_dates[0]
                    if train_label_available else False
                ),
            }
        )
    predictions.sort(key=lambda row: (row["rebalance_date"], row["symbol"]))
    return folds, predictions


def _prepare_rows(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[list[dict[str, Any]], int]:
    required = ("rebalance_date", "symbol", TARGET_COLUMN)
    prepared = []
    for source in rows:
        date = str(source.get("rebalance_date", "")).strip()
        symbol = str(source.get("symbol", "")).strip().upper()
        benchmark_symbol = str(source.get("benchmark_symbol", "")).strip().upper()
        numbers = {column: _number(source.get(column)) for column in required[2:]}
        if not date or not symbol or any(value is None for value in numbers.values()):
            continue
        optional_columns = {
            *CONTEXT_COLUMNS,
            *AUXILIARY_TARGET_COLUMNS,
            *(
                name
                for name in source
                if name.startswith("news_") or "sentiment" in name.lower()
            ),
        }
        label_end = _clean_text(
            source.get("label_end_timestamp")
            or source.get("label_end_date")
            or source.get("outcome_end_date")
            or ""
        )
        label_available = _clean_text(source.get("label_available_timestamp"))
        if not label_end or not label_available:
            raise ValueError(
                "Stock-level forward target rows require explicit or safely derived "
                "label availability; rebalance_date cannot be used as label availability. "
                f"rebalance_date={date}; symbol={symbol}"
            )
        if label_available <= date:
            raise ValueError(
                "Stock-level forward target label availability must be after the "
                "decision/rebalance date; rebalance_date cannot masquerade as "
                f"label availability. rebalance_date={date}; "
                f"label_available_timestamp={label_available}; symbol={symbol}"
            )
        provenance = {column: _clean_text(source.get(column)) for column in TARGET_PROVENANCE_COLUMNS}
        provenance["feature_timestamp"] = provenance.get("feature_timestamp") or date
        provenance["decision_timestamp"] = provenance.get("decision_timestamp") or date
        provenance["label_end_timestamp"] = label_end
        provenance["label_available_timestamp"] = label_available
        _validate_target_provenance(provenance, date, symbol)
        prepared.append({
            "rebalance_date": date,
            **provenance,
            "symbol": symbol,
            "benchmark_symbol": benchmark_symbol,
            **{column: float(value) for column, value in numbers.items()},
            **{
                column: (
                    float(value)
                    if (value := _number(source.get(column))) is not None
                    else math.nan
                )
                for column in feature_columns
            },
            **{
                column: _number(source.get(column)) or 0.0
                for column in optional_columns
            },
            **{column: _number(source.get(column)) for column in TARGET_OUTPUT_COLUMNS},
        })
    return prepared, len(rows) - len(prepared)


def _available_feature_columns(
    rows: list[dict[str, Any]],
    *,
    include_engineered: bool,
) -> tuple[str, ...]:
    if not include_engineered:
        return FEATURE_COLUMNS
    available_engineered = tuple(
        column
        for column in ENGINEERED_FEATURE_COLUMNS
        if any(_number(row.get(column)) is not None for row in rows)
    )
    return (*FEATURE_COLUMNS, *available_engineered)


def _base_prediction_row(row: dict[str, Any], fold_id: int) -> dict[str, Any]:
    return {
        "rebalance_date": row["rebalance_date"],
        "symbol": row["symbol"],
        "benchmark_symbol": row.get("benchmark_symbol", ""),
        "fold_id": fold_id,
        **{column: row.get(column, "") for column in TARGET_PROVENANCE_COLUMNS},
        TARGET_COLUMN: row[TARGET_COLUMN],
        **{column: row.get(column) for column in AUXILIARY_TARGET_COLUMNS},
        **{column: row.get(column) for column in TARGET_OUTPUT_COLUMNS},
        "predicted_momentum_120d": row["predicted_momentum_120d"],
        "predicted_risk_adjusted_momentum": row[
            "predicted_risk_adjusted_momentum"
        ],
    }


def _validate_split_settings(
    min_train_dates: int,
    test_window_dates: int,
    embargo_dates: int,
) -> None:
    if min_train_dates < 1:
        raise ValueError("min_train_dates must be at least one")
    if test_window_dates < 1:
        raise ValueError("test_window_dates must be at least one")
    if embargo_dates < 0:
        raise ValueError("embargo_dates cannot be negative")


def _validate_unique_keys(rows: list[dict[str, Any]]) -> None:
    keys = [(str(row["rebalance_date"]), str(row["symbol"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Stock-level rows must be unique by rebalance_date and symbol")


def _average(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "none", "null", "nan", "nat"} else text


def _validate_target_provenance(
    provenance: dict[str, str],
    rebalance_date: str,
    symbol: str,
) -> None:
    if provenance.get("target_provenance_contract_version") != TARGET_PROVENANCE_CONTRACT_VERSION:
        raise ValueError(
            "Stock-level forward target rows require canonical target provenance; "
            "pre-4D artifacts missing provenance are incompatible. "
            f"rebalance_date={rebalance_date}; symbol={symbol}"
        )
    required = (
        "feature_timestamp",
        "decision_timestamp",
        "target_horizon",
        "target_observation_count",
        "target_start_timestamp",
        "label_start_timestamp",
        "label_end_timestamp",
        "label_available_timestamp",
        "target_price_convention",
    )
    missing = [column for column in required if not provenance.get(column)]
    if missing:
        raise ValueError(
            "Stock-level forward target rows require complete canonical target "
            f"provenance; missing_columns={missing}; rebalance_date={rebalance_date}; "
            f"symbol={symbol}"
        )


def _sort_value(value: Any) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else -math.inf
