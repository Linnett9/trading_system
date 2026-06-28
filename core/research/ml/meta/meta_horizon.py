from __future__ import annotations

from typing import Any

from core.research.ml.meta.meta_evaluation import _chronological_meta_probabilities


def _extended_horizon_rows(
    *,
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    holdout_probabilities: list[float],
    model_type: str,
    config: dict[str, Any],
    random_seed: int,
    sklearn_n_jobs: int,
) -> dict[str, Any]:
    horizon_config = config.get("meta_canonical_horizon", {}) or {}
    enabled = bool(horizon_config.get("expand_from_source_predictions", False))
    if not enabled:
        return {"enabled": False, "available": False}
    min_selection_dates = int(
        horizon_config.get("minimum_selection_rebalance_dates", 26)
    )
    fold_count = int(
        horizon_config.get(
            "walk_forward_folds",
            config.get("meta_auxiliary_walk_forward_folds", 3),
        )
    )
    embargo = int(
        horizon_config.get(
            "embargo_rebalance_dates",
            config.get("meta_auxiliary_embargo_rebalance_dates", 1),
        )
    )
    purge_overlaps = bool(
        horizon_config.get(
            "purge_overlapping_labels",
            config.get("meta_auxiliary_purge_overlapping_labels", True),
        )
    )
    train_probabilities, audits = _chronological_meta_probabilities(
        train_rows,
        model_type=model_type,
        fold_count=fold_count,
        embargo_rebalance_dates=embargo,
        purge_overlapping_labels=purge_overlaps,
        random_seed=random_seed,
        sklearn_n_jobs=sklearn_n_jobs,
    )
    eligible_indexes = [
        index
        for index, probability in enumerate(train_probabilities)
        if probability is not None and _has_allocation_forecasts(train_rows[index])
    ]
    eligible_dates = sorted(
        {
            train_rows[index].get("rebalance_date", "")
            for index in eligible_indexes
            if train_rows[index].get("rebalance_date")
        }
    )
    if len(eligible_dates) <= min_selection_dates:
        return {
            "enabled": True,
            "available": False,
            "audit": {
                "available": False,
                "reason": "insufficient_cross_fitted_source_prediction_dates",
                "eligible_rebalance_date_count": len(eligible_dates),
                "minimum_selection_rebalance_dates": min_selection_dates,
                "source": "existing_prediction_artifacts_only",
            },
        }
    evaluation_start = eligible_dates[min_selection_dates]
    selection_rows = [
        train_rows[index]
        for index in eligible_indexes
        if train_rows[index].get("rebalance_date", "") < evaluation_start
    ]
    selection_probabilities = [
        float(train_probabilities[index])
        for index in eligible_indexes
        if train_rows[index].get("rebalance_date", "") < evaluation_start
    ]
    out_of_fold_evaluation_rows = [
        {
            **train_rows[index],
            "split": "extended_out_of_fold_evaluation",
            "meta_prediction_source": "chronological_cross_fitted_meta_model",
        }
        for index in eligible_indexes
        if train_rows[index].get("rebalance_date", "") >= evaluation_start
    ]
    out_of_fold_evaluation_probabilities = [
        float(train_probabilities[index])
        for index in eligible_indexes
        if train_rows[index].get("rebalance_date", "") >= evaluation_start
    ]
    frozen_holdout_rows = [
        {
            **row,
            "split": "holdout",
            "meta_prediction_source": "frozen_meta_model_holdout",
        }
        for row in holdout_rows
    ]
    evaluation_rows = out_of_fold_evaluation_rows + frozen_holdout_rows
    evaluation_probabilities = (
        out_of_fold_evaluation_probabilities + list(holdout_probabilities)
    )
    return {
        "enabled": True,
        "available": bool(evaluation_rows and selection_rows),
        "selection_rows": selection_rows,
        "selection_probabilities": selection_probabilities,
        "evaluation_rows": evaluation_rows,
        "evaluation_probabilities": evaluation_probabilities,
        "audit": {
            "available": bool(evaluation_rows and selection_rows),
            "source": "existing_prediction_artifacts_only",
            "mode": "chronological_cross_fitted_out_of_fold_plus_frozen_holdout",
            "old_holdout_start_date": _minimum_row_date(holdout_rows),
            "old_holdout_end_date": _maximum_row_date(holdout_rows),
            "new_evaluation_start_date": _minimum_row_date(evaluation_rows),
            "new_evaluation_end_date": _maximum_row_date(evaluation_rows),
            "selection_rebalance_date_count": len({
                row.get("rebalance_date")
                for row in selection_rows
                if row.get("rebalance_date")
            }),
            "extended_out_of_fold_rebalance_date_count": len({
                row.get("rebalance_date")
                for row in out_of_fold_evaluation_rows
                if row.get("rebalance_date")
            }),
            "holdout_rebalance_date_count": len({
                row.get("rebalance_date")
                for row in holdout_rows
                if row.get("rebalance_date")
            }),
            "evaluation_rebalance_date_count": len({
                row.get("rebalance_date")
                for row in evaluation_rows
                if row.get("rebalance_date")
            }),
            "minimum_selection_rebalance_dates": min_selection_dates,
            "embargo_rebalance_dates": embargo,
            "purge_overlapping_labels": purge_overlaps,
            "walk_forward_folds": fold_count,
            "fold_audits": audits,
            "in_sample_meta_predictions": False,
            "base_models_rerun": False,
        },
    }


def _has_allocation_forecasts(row: dict[str, str]) -> bool:
    return all(
        _row_has_forecast(row, requirement)
        for requirement in (
            "predicted_forward_return_10d",
            "predicted_future_drawdown",
            "predicted_future_volatility",
        )
    )


def _row_has_forecast(row: dict[str, str], suffix: str) -> bool:
    meta_name = f"meta_{suffix}"
    if row.get(meta_name) not in (None, ""):
        return True
    return any(
        value not in (None, "")
        and (name == suffix or name.endswith(f"_{suffix}"))
        for name, value in row.items()
    )


def _minimum_row_date(rows: list[dict[str, str]]) -> str | None:
    dates = sorted(row.get("rebalance_date", "") for row in rows if row.get("rebalance_date"))
    return dates[0] if dates else None


def _maximum_row_date(rows: list[dict[str, str]]) -> str | None:
    dates = sorted(row.get("rebalance_date", "") for row in rows if row.get("rebalance_date"))
    return dates[-1] if dates else None
