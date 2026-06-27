from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any


AUXILIARY_TARGETS = {
    "actual_forward_return_5d": "meta_predicted_forward_return_5d",
    "actual_forward_return_10d": "meta_predicted_forward_return_10d",
    "actual_future_volatility": "meta_predicted_future_volatility",
    "actual_future_drawdown": "meta_predicted_future_drawdown",
    "actual_max_adverse_excursion": "meta_predicted_max_adverse_excursion",
    "actual_max_favourable_excursion": "meta_predicted_max_favourable_excursion",
}

AUXILIARY_PREDICTION_COLUMNS = tuple(
    prediction_name.removeprefix("meta_")
    for prediction_name in AUXILIARY_TARGETS.values()
)


@dataclass(frozen=True)
class MetaAuxiliaryResult:
    train_rows: list[dict[str, str]]
    holdout_rows: list[dict[str, str]]
    selection_train_indexes: tuple[int, ...]
    predictions_path: Path
    metrics_json_path: Path
    metrics_markdown_path: Path
    metrics: dict[str, Any]


@dataclass
class _AuxiliaryRegressor:
    feature_names: list[str]
    estimator: Any = None
    constant: float | None = None

    def predict(self, rows: list[dict[str, str]]) -> list[float]:
        if self.constant is not None:
            return [self.constant for _ in rows]
        return [
            float(value)
            for value in self.estimator.predict(_feature_matrix(rows, self.feature_names))
        ]


def run_meta_auxiliary_ensemble(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    output_dir: Path,
    *,
    walk_forward_folds: int = 3,
    embargo_rebalance_dates: int = 1,
    purge_overlapping_labels: bool = True,
) -> MetaAuxiliaryResult:
    if walk_forward_folds < 1:
        raise ValueError("walk_forward_folds must be at least one")
    if embargo_rebalance_dates < 0:
        raise ValueError("embargo_rebalance_dates must be non-negative")
    augmented_train = [dict(row) for row in train_rows]
    augmented_holdout = [dict(row) for row in holdout_rows]
    feature_names = _auxiliary_feature_names(train_rows)
    target_metrics: dict[str, dict[str, Any]] = {}
    fold_audits: dict[str, list[dict[str, Any]]] = {}
    predicted_indexes_by_target: list[set[int]] = []
    holdout_start = _minimum_rebalance_date(holdout_rows)

    for actual_name, prediction_name in AUXILIARY_TARGETS.items():
        usable_train = [row for row in train_rows if _finite_value(row.get(actual_name))]
        if not usable_train or not feature_names:
            target_metrics[actual_name] = {
                "available": False,
                "reason": "missing training targets or auxiliary source features",
                "prediction_column": prediction_name,
                "sample_count": 0,
            }
            continue
        holdout_training_rows, holdout_training_audit = _purged_training_rows(
            usable_train,
            validation_start=holdout_start,
            embargo_rebalance_dates=embargo_rebalance_dates,
            purge_overlapping_labels=purge_overlapping_labels,
        )
        if not holdout_training_rows:
            target_metrics[actual_name] = {
                "available": False,
                "reason": "no eligible training rows before purged holdout",
                "prediction_column": prediction_name,
                "sample_count": 0,
            }
            continue
        model = _fit_regressor(holdout_training_rows, actual_name, feature_names)
        train_predictions, target_fold_audits = _chronological_cross_fitted_predictions(
            augmented_train,
            actual_name,
            feature_names,
            fold_count=walk_forward_folds,
            embargo_rebalance_dates=embargo_rebalance_dates,
            purge_overlapping_labels=purge_overlapping_labels,
        )
        holdout_predictions = model.predict(augmented_holdout)
        for row, prediction in zip(augmented_train, train_predictions):
            if prediction is not None:
                row[prediction_name] = str(prediction)
        for row, prediction in zip(augmented_holdout, holdout_predictions):
            row[prediction_name] = str(prediction)
        predicted_indexes_by_target.append({
            index
            for index, prediction in enumerate(train_predictions)
            if prediction is not None
        })
        fold_audits[actual_name] = target_fold_audits
        target_metrics[actual_name] = _regression_metrics(
            augmented_holdout,
            actual_name,
            prediction_name,
        )
        target_metrics[actual_name]["holdout_training_audit"] = holdout_training_audit

    selection_train_indexes = tuple(sorted(
        set.intersection(*predicted_indexes_by_target)
        if predicted_indexes_by_target
        else set()
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "meta_auxiliary_predictions.csv"
    metrics_json_path = output_dir / "meta_auxiliary_metrics.json"
    metrics_markdown_path = output_dir / "meta_auxiliary_metrics.md"
    _write_predictions(predictions_path, augmented_holdout)
    metrics = {
        "mode": "meta_auxiliary_ensemble_research_only",
        "feature_columns": feature_names,
        "targets": target_metrics,
        "available_targets": [
            name for name, payload in target_metrics.items() if payload.get("available")
        ],
        "train_prediction_method": "purged_chronological_walk_forward",
        "holdout_prediction_method": (
            "refit_purged_out_of_fold_rows_then_predict_frozen_holdout"
        ),
        "fold_design": {
            "walk_forward_folds": walk_forward_folds,
            "embargo_rebalance_dates": embargo_rebalance_dates,
            "purge_overlapping_labels": purge_overlapping_labels,
            "date_grouping": "rebalance_date",
            "training_window": "expanding",
            "validation_window": "contiguous_future_date_blocks",
            "warmup_rows_are_forecasted": False,
            "selection_train_row_count": len(selection_train_indexes),
        },
        "fold_audits": fold_audits,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_metrics_markdown(metrics_markdown_path, metrics)
    return MetaAuxiliaryResult(
        train_rows=augmented_train,
        holdout_rows=augmented_holdout,
        selection_train_indexes=selection_train_indexes,
        predictions_path=predictions_path,
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        metrics=metrics,
    )


def namespaced_auxiliary_features(
    model: str,
    prediction: dict[str, str],
) -> dict[str, str]:
    return {
        f"{model}__{name}": str(prediction[name])
        for name in AUXILIARY_PREDICTION_COLUMNS
        if prediction.get(name) not in (None, "")
    }


def actual_auxiliary_values(
    expanded_row: dict[str, str],
    source_rows: list[dict[str, str]],
) -> dict[str, str]:
    values = {}
    for actual_name in AUXILIARY_TARGETS:
        candidates = []
        expanded_value = expanded_row.get(actual_name)
        if expanded_value not in (None, ""):
            candidates.append(str(expanded_value))
        candidates.extend(
            str(row[actual_name])
            for row in source_rows
            if row.get(actual_name) not in (None, "")
        )
        finite = [value for value in candidates if _finite_value(value)]
        if finite:
            values[actual_name] = finite[0]
    return values


def _fit_regressor(
    rows: list[dict[str, str]],
    actual_name: str,
    feature_names: list[str],
) -> _AuxiliaryRegressor:
    targets = [float(row[actual_name]) for row in rows]
    if len(rows) < 2 or max(targets) == min(targets):
        return _AuxiliaryRegressor(feature_names, constant=mean(targets))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(_feature_matrix(rows, feature_names), targets)
    return _AuxiliaryRegressor(feature_names, estimator=estimator)


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


def _auxiliary_feature_names(rows: list[dict[str, str]]) -> list[str]:
    return sorted({
        name
        for row in rows
        for name in row
        if "__predicted_" in name
        or name.endswith("_raw_probability")
        or name.endswith("_calibrated_probability")
    })


def _feature_matrix(
    rows: list[dict[str, str]],
    feature_names: list[str],
) -> list[list[float]]:
    return [
        [
            float(row.get(name, 0.0) or 0.0)
            if _finite_value(row.get(name, 0.0))
            else 0.0
            for name in feature_names
        ]
        for row in rows
    ]


def _regression_metrics(
    rows: list[dict[str, str]],
    actual_name: str,
    prediction_name: str,
) -> dict[str, Any]:
    pairs = [
        (float(row[actual_name]), float(row[prediction_name]))
        for row in rows
        if _finite_value(row.get(actual_name))
        and _finite_value(row.get(prediction_name))
    ]
    if not pairs:
        return {
            "available": True,
            "prediction_column": prediction_name,
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "directional_accuracy": None,
            "residual_quantiles": {},
        }
    actual = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    errors = [estimate - observed for observed, estimate in pairs]
    is_return = actual_name in {
        "actual_forward_return_5d",
        "actual_forward_return_10d",
    }
    return {
        "available": True,
        "prediction_column": prediction_name,
        "sample_count": len(pairs),
        "mae": mean(abs(value) for value in errors),
        "rmse": math.sqrt(mean(value * value for value in errors)),
        "pearson_correlation": _pearson(actual, predicted),
        "spearman_correlation": _pearson(_ranks(actual), _ranks(predicted)),
        "directional_accuracy": (
            mean(
                float((observed >= 0.0) == (estimate >= 0.0))
                for observed, estimate in pairs
            )
            if is_return
            else None
        ),
        "residual_quantiles": {
            "p10": _quantile(errors, 0.10),
            "p50": _quantile(errors, 0.50),
            "p90": _quantile(errors, 0.90),
        },
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[ordered[position][0]] = average_rank
        index = end
    return output


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    prediction_columns = [
        name for name in AUXILIARY_TARGETS.values() if any(name in row for row in rows)
    ]
    fieldnames = [
        "feature_id",
        "rebalance_date",
        "variant_id",
        *AUXILIARY_TARGETS.keys(),
        *prediction_columns,
        "research_only",
        "trading_impact",
        "production_validated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{name: row.get(name, "") for name in fieldnames},
                "research_only": True,
                "trading_impact": "none",
                "production_validated": False,
            })


def _write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Meta Auxiliary Metrics",
        "",
        "|target|samples|mae|rmse|pearson|spearman|directional_accuracy|",
        "|---|---|---|---|---|---|---|",
    ]
    for target, payload in metrics["targets"].items():
        lines.append(
            "|{target}|{samples}|{mae}|{rmse}|{pearson}|{spearman}|{directional}|".format(
                target=target,
                samples=payload.get("sample_count", 0),
                mae=_format_metric(payload.get("mae")),
                rmse=_format_metric(payload.get("rmse")),
                pearson=_format_metric(payload.get("pearson_correlation")),
                spearman=_format_metric(payload.get("spearman_correlation")),
                directional=_format_metric(payload.get("directional_accuracy")),
            )
        )
    lines.extend([
        "",
        "Research only. Trading impact: none. Production validated: false.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
