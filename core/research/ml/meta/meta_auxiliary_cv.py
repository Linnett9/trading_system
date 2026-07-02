from __future__ import annotations

from datetime import date
from typing import Any

from core.research.ml.meta.meta_auxiliary_math import _finite_value
from core.research.ml.meta.meta_auxiliary_model import _fit_regressor


def _chronological_cross_fitted_predictions(
    rows: list[dict[str, str]],
    actual_name: str,
    feature_names: list[str],
    *,
    fold_count: int,
    embargo_rebalance_dates: int,
    purge_overlapping_labels: bool,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    if not rows:
        return [], []
    unique_dates = sorted({_rebalance_date(row) for row in rows})
    if len(unique_dates) < 2:
        return [None] * len(rows), []
    initial_training_date_count = max(1, len(unique_dates) // (fold_count + 1))
    validation_dates = unique_dates[initial_training_date_count:]
    date_blocks = _contiguous_blocks(validation_dates, fold_count)
    predictions: list[float | None] = [None] * len(rows)
    audits: list[dict[str, Any]] = []
    for fold_number, block in enumerate(date_blocks, start=1):
        validation_start = block[0]
        validation_date_set = set(block)
        validation_indexes = [
            index
            for index, row in enumerate(rows)
            if _rebalance_date(row) in validation_date_set
        ]
        candidate_training_rows = [
            row
            for row in rows
            if _rebalance_date(row) < validation_start
            and _finite_value(row.get(actual_name))
        ]
        training_rows, purge_audit = _purged_training_rows(
            candidate_training_rows,
            validation_start=validation_start,
            embargo_rebalance_dates=embargo_rebalance_dates,
            purge_overlapping_labels=purge_overlapping_labels,
        )
        if not training_rows:
            audits.append({
                "fold": fold_number,
                "validation_start": validation_start.isoformat(),
                "validation_end": block[-1].isoformat(),
                "validation_row_count": len(validation_indexes),
                "prediction_generated": False,
                **purge_audit,
            })
            continue
        model = _fit_regressor(training_rows, actual_name, feature_names)
        fold_predictions = model.predict([rows[index] for index in validation_indexes])
        for index, prediction in zip(validation_indexes, fold_predictions):
            predictions[index] = prediction
        audits.append({
            "fold": fold_number,
            "validation_start": validation_start.isoformat(),
            "validation_end": block[-1].isoformat(),
            "validation_row_count": len(validation_indexes),
            "prediction_generated": True,
            **purge_audit,
        })
    return predictions, audits
def _purged_training_rows(
    rows: list[dict[str, str]],
    *,
    validation_start: date | None,
    embargo_rebalance_dates: int,
    purge_overlapping_labels: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if validation_start is None:
        return [], {
            "training_row_count": 0,
            "purged_label_overlap_count": 0,
            "embargoed_rebalance_date_count": 0,
            "max_training_rebalance_date": None,
        }
    chronological = [
        row for row in rows if _rebalance_date(row) < validation_start
    ]
    before_label_purge = len(chronological)
    if purge_overlapping_labels:
        chronological = [
            row
            for row in chronological
            if _label_window_ends_before(row, validation_start)
        ]
    eligible_dates = sorted({_rebalance_date(row) for row in chronological})
    embargoed_dates = set(
        eligible_dates[-embargo_rebalance_dates:]
        if embargo_rebalance_dates
        else []
    )
    retained = [
        row for row in chronological if _rebalance_date(row) not in embargoed_dates
    ]
    retained_dates = [_rebalance_date(row) for row in retained]
    return retained, {
        "training_row_count": len(retained),
        "purged_label_overlap_count": before_label_purge - len(chronological),
        "embargoed_rebalance_date_count": len(embargoed_dates),
        "max_training_rebalance_date": (
            max(retained_dates).isoformat() if retained_dates else None
        ),
    }
def _contiguous_blocks(values: list[date], block_count: int) -> list[list[date]]:
    resolved_count = min(max(1, block_count), len(values))
    quotient, remainder = divmod(len(values), resolved_count)
    blocks = []
    start = 0
    for index in range(resolved_count):
        size = quotient + (1 if index < remainder else 0)
        blocks.append(values[start:start + size])
        start += size
    return [block for block in blocks if block]
def _minimum_rebalance_date(rows: list[dict[str, str]]) -> date | None:
    dates = [_rebalance_date(row) for row in rows]
    return min(dates) if dates else None
def _rebalance_date(row: dict[str, str]) -> date:
    raw_value = row.get("rebalance_date") or row.get("date")
    if not raw_value:
        raise ValueError("Meta auxiliary row is missing rebalance_date")
    return date.fromisoformat(str(raw_value))
def _label_window_ends_before(row: dict[str, str], validation_start: date) -> bool:
    raw_value = row.get("label_end_date") or row.get("outcome_end_date")
    if not raw_value:
        return True
    return date.fromisoformat(str(raw_value)) < validation_start
